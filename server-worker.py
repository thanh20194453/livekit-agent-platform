"""
LiveKit Voice Bot Server Worker
Main entry point for the voice bot service with support for multiple configurations.
"""
import asyncio
import logging
import json
import os
from typing import Dict, Any, Optional
import signal
import time

from livekit.api import LiveKitAPI, ListParticipantsRequest, RoomParticipantIdentity, UpdateRoomMetadataRequest
from livekit.api.twirp_client import TwirpError
from livekit.plugins import noise_cancellation
from livekit.agents.llm import function_tool
from livekit.agents import Agent, RunContext, JobContext, RoomInputOptions, RoomOutputOptions, cli, WorkerOptions
from livekit import api as livekit_api
from dotenv import load_dotenv

# Utils imports (modular components)
from utils.bot_config import BotConfiguration
from utils.session_factory import SessionFactory
from utils.instructions_builder import build_agent_instructions
from utils.recording import RecordingManager

# Tools imports
from tools.search_information_from_knowledge_base import search_information_from_knowledge_base as search_knowledge_base
from tools.ddgs import ddgs
from tools.scrawl_website import trafilatura_tools
from tools.excel_calculator import ExcelRef, excel_calculator
from tools.routing_agent import routingagent

#Externals import
from external.nodo.nodo import NodoManager
from external.nodo.nodo_tool.get_apartment_info import gaitool
from external.nodo.nodo_tool.get_project_info import gpji
from external.crm.crm import CRMManager

load_dotenv()
logger = logging.getLogger("voice-bot-worker")

class DynamicVoiceAgent(Agent):
    """Voice agent that dynamically configures based on voice bot data"""

    def __init__(
            self,
            instructions: str,
            knowledge_base_tables: list = [],
            tool_name: list[str] = [],
            call_info: Optional[Dict[str, Any]] = None,
            ctx: Optional[JobContext] = None,
            is_outbound_call: bool = False,
        ) -> None:
        super().__init__(instructions=instructions)
        self.knowledge_base_tables = knowledge_base_tables or []
        self.tool_name = tool_name or []
        self.call_info = call_info or {}
        self.ctx = ctx
        self.is_outbound_call = is_outbound_call
        self._greeting_sent = False
    async def on_enter(self):
        """Called when agent enters the session"""
        try:
            # For outbound calls, wait for SIP participant before greeting
            if self.is_outbound_call and not self._greeting_sent:
                logger.info("[Agent] Outbound call - waiting for SIP participant before greeting")
                # Don't generate greeting yet - will be triggered when SIP joins
                return
            
            # For inbound calls or already greeted, generate greeting immediately
            if not self._greeting_sent:
                self._greeting_sent = True
                await self.session.generate_reply()
        except Exception as e:
            logger.error(f"Error in on_enter: {e}")
    
    async def send_delayed_greeting(self):
        """Send greeting after SIP participant has joined"""
        try:
            if not self._greeting_sent:
                logger.info("[Agent] SIP participant joined - sending greeting now")
                self._greeting_sent = True
                await self.session.generate_reply()
        except Exception as e:
            logger.error(f"Error sending delayed greeting: {e}")

    @function_tool(description="Search a specified information in the knowledge base with user query")
    async def search_information_from_knowledge_base(self, run_ctx: RunContext, query: str) -> str:
        """Search the knowledge base for specific information"""
        if len(self.knowledge_base_tables) == 0:
            return json.dumps({"error": "No knowledge base found!"})
        try:
            knowledge_base_table_name = self.knowledge_base_tables[0]  # Default to the first table
            # Validate table name against available tables
            logger.info(f"Searching knowledge base for query: {query}")
            if knowledge_base_table_name not in self.knowledge_base_tables:
                error_msg = f"Table {knowledge_base_table_name} not available. Available: {self.knowledge_base_tables}"
                return json.dumps({"error": error_msg})

            result = await search_knowledge_base(query, table_name=knowledge_base_table_name)
            logger.info(f"Knowledge base search result: {result}")
            return result
            
        except Exception as e:
            return json.dumps({"error": f"Search failed: {str(e)}"})

    @function_tool(description="A tool for performing DuckDuckGo searches to find information on the web. Real-time, current external information.")
    async def duckduckgo_search(self, run_ctx: RunContext, query: str) -> str:
        """Perform a DuckDuckGo search"""
        if not "duck_duck_go_search" in self.tool_name:
            return "DuckDuckGo search tool is not enabled."
        try:
            results = await asyncio.to_thread(ddgs.duckduckgo_search, query)
            return results
        except Exception as e:
            logger.error(f"Error in duckduckgo_search: {e}")
            return f"Error performing search: {str(e)}"
        
    @function_tool(description="A tool for performing DuckDuckGo news searches to find current news articles.")
    async def duckduckgo_news(self, run_ctx: RunContext, query: str) -> str:
        """Perform a DuckDuckGo search for news"""
        if not "duck_duck_go_search" in self.tool_name:
            return "DuckDuckGo news tool is not enabled."
        try:
            results = await asyncio.to_thread(ddgs.duckduckgo_news, query)
            return results
        except Exception as e:
            logger.error(f"Error in duckduckgo_news: {e}")
            return f"Error performing search: {str(e)}"

    @function_tool(description="A tool for exploring websites and extracting text content for entire website crawling.")
    async def crawl_website(self, run_ctx: RunContext, website_url: str) -> str:
        """Crawl a website and extract text content"""
        if not "trafilatura" in self.tool_name:
            return "Website crawling tool is not enabled."
        try:
            # Run in thread to avoid blocking event loop
            result = await asyncio.to_thread(trafilatura_tools.crawl_website, homepage_url=website_url, extract_content=True)
            return result
        except Exception as e:
            logger.error(f"Error in crawl_website: {e}")
            return f"Error crawling website: {str(e)}"
        
    @function_tool(description="A tool for crawling websites and extracting text content from a given URL.")
    async def extract_text(self, run_ctx: RunContext, website_url: str) -> str:
        """Crawl a website and extract text content"""
        if not "trafilatura" in self.tool_name:
            return "Website crawling tool is not enabled."
        try:
            result = await asyncio.to_thread(trafilatura_tools.extract_text, url=website_url, output_format="markdown")
            return result
        except Exception as e:
            logger.error(f"Error in extract_text: {e}")
            return f"Error extracting text: {str(e)}"

    @function_tool(description="A tool for batch extracting text content from a list of URLs.")
    async def batch_extract(self, run_ctx: RunContext, urls: list[str]) -> str:
        """Batch extract text content from a list of URLs"""
        if not "trafilatura" in self.tool_name:
                return "Website crawling tool is not enabled."
        try:
            if not isinstance(urls, list):
                raise ValueError("Input must be a list of URLs")

            result = await asyncio.to_thread(trafilatura_tools.extract_batch, urls=urls)
            return result
        except Exception as e:
            logger.error(f"Error in batch_extract: {e}")
            return f"Error in batch extraction: {str(e)}"
        
    @function_tool(description="A tool for performing calculations in an Excel file by updating specified cells and returning the updated file content in markdown format.")
    async def excel_calculator(self, run_ctx: RunContext, file_url: str, dataRef: list[ExcelRef]) -> str:
        """Perform calculations in an Excel file by updating specified cells and returning the updated file content in markdown format."""
        if not "excel_calculator" in self.tool_name:
            return "Excel calculator tool is not enabled."
        try:
            result = await asyncio.to_thread(lambda: asyncio.run(excel_calculator.calculate(file_url, dataRef)))
            return result
        except Exception as e:
            logger.error(f"Error in excel_calculator: {e}")
            return f"Error in Excel calculation: {str(e)}"
        
    @function_tool(description="A tool for searching suitable apratment. Do not need use all parameters")
    async def get_apartment_info(self, run_ctx: RunContext, location: str = "",property_type:str = "", area:str= "",direction:str= "",purpose:str= "",num_rooms:str= "",price:str= "",num_floors:str= "") -> str:
        """Searching suitable apratments. Do not need use all parameters"""
        if not "get_apartment_info" in self.tool_name:
            return "Get apartment info tool is not enabled."
        try:
            project_id=self.call_info.get("call_config", {}).get("project_id", "")
            result = await asyncio.to_thread(lambda: asyncio.run(gaitool.get_apartment_info(location,property_type, area,direction,purpose,num_rooms,price,num_floors,project_id)))
            return result
        except Exception as e:
            logger.error(f"Error in get_apartment_info: {e}")
            return f"Error in getting apartment infomation: {str(e)}"
    
    @function_tool(description="A tool for Hanging up call after saying goodbye to the customer")
    async def hang_up_call(self, run_ctx: RunContext) -> str:
        """Hanging up call after saying goodbye to the customer"""
        if not "hang_up_call" in self.tool_name:
            return "hang_up_call tool is not enabled."
        try:
            await asyncio.sleep(8)
            if self.ctx is None:
                return "Context is not available to hang up the call."
            await self.ctx.api.room.delete_room(
            livekit_api.DeleteRoomRequest(
                room=self.ctx.room.name,
            )
            )
            return "Hang up call successfully"
        except Exception as e:
            logger.error(f"Error in hang_up_call: {e}")
            return f"Error in hanging up call: {str(e)}"
    @function_tool(description="A tool for transfering call from current number to other number")
    async def transfer_call(self, run_ctx: RunContext,transfer_number:str=""):
        """Transfer call from current number to other number"""
        if not "transfer_call" in self.tool_name:
            return "transfer_call tool is not enabled."
        try:
            phone_number=self.call_info.get("call_config", {}).get("phone_number", "")
            result = await asyncio.to_thread(lambda: asyncio.run(routingagent.routing_agent(self.ctx,phone_number,transfer_number)))
            return result
        except Exception as e:
            logger.error(f"Error in transfer_call: {e}")
            return f"Error in Transfer call tool: {str(e)}"

class VoiceBotSession:
    """Manages a voice bot session with LiveKit."""
    
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.session = None
        self.recording_manager = None
        self._shutdown_event = asyncio.Event()
        self.agent = None  # Store agent reference
        self.nodo_manager = None

    async def _create_agent(self, bot_config: BotConfiguration, ctx: JobContext, is_outbound: bool = False) -> Agent:
        """Create the voice agent with configuration"""
        if bot_config.call_config and bot_config.call_config.get("company", "")=="nodo":
            bot_config.tools=bot_config.tools + ["hang_up_call", "get_apartment_info"]
            instructions = build_agent_instructions(bot_config)
        else:
            instructions = build_agent_instructions(bot_config)
        agent = DynamicVoiceAgent(
                    instructions=instructions,
                    knowledge_base_tables=bot_config.knowledge_base_tables,
                    tool_name=bot_config.tools,
                    call_info=bot_config.metadata,
                    ctx=ctx,
                    is_outbound_call=is_outbound,
                )
        return agent

    async def _setup_shutdown_handlers(self):
        """Setup graceful shutdown handlers (only works in main thread)"""
        try:
            def signal_handler(signum, frame):
                self._shutdown_event.set()
                
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except ValueError as e:
            # Signal handlers only work in main thread - LiveKit handles this at framework level
            logger.debug(f"Cannot setup signal handlers in worker thread: {e}")
            
    async def _remove_non_agent_or_sip(self, room_name):
        async with LiveKitAPI() as lkapi:
            res = await lkapi.room.list_participants(
                ListParticipantsRequest(room=room_name)
            )
            for p in res.participants:
                identity = p.identity

                if identity.startswith("agent") or identity.startswith("sip"):
                    continue
                else:
                    await lkapi.room.remove_participant(
                        RoomParticipantIdentity(
                            room=room_name,
                            identity=identity
                        )
                    )

                return None
    
    async def _update_room_status(self, room_name: str, status: str, error_details: Optional[Dict[str, Any]] = None):
        """
        Update room metadata with call status for FastAPI to poll
        """
        try:
            async with LiveKitAPI() as lkapi:
                # Get current room info
                room_list = await lkapi.room.list_rooms(
                    livekit_api.ListRoomsRequest(names=[room_name])
                )
                
                if not room_list.rooms:
                    logger.warning(f"Room {room_name} not found for metadata update")
                    return
                
                # Parse existing metadata
                current_metadata = json.loads(room_list.rooms[0].metadata or "{}")
                
                # Update call_status in metadata
                current_metadata["call_status"] = {
                    "status": status,  # "success", "failed", "completed"
                    "timestamp": time.time(),
                    "error_details": error_details
                }
                
                # Update room metadata
                await lkapi.room.update_room_metadata(
                    UpdateRoomMetadataRequest(
                        room=room_name,
                        metadata=json.dumps(current_metadata)
                    )
                )
                
                logger.info(f"Room metadata updated: {room_name} - status: {status}")
                
        except Exception as e:
            logger.error(f"Error updating room metadata for {room_name}: {e}")

    async def entrypoint(self, ctx: JobContext):
        """
        Main entry point for the voice bot session.
        
        Handles all 4 cases:
        - Case 1: Agent Mode + TTS Model (non-realtime, NOT call outbound)
        - Case 2: Agent Mode + Realtime Model (realtime, NOT call outbound)  
        - Case 3: Call Outbound + TTS Model (non-realtime, call outbound)
        - Case 4: Call Outbound + Realtime Model (realtime, call outbound)
        """
        try:
            # Setup shutdown handling (will be skipped if not in main thread)
            await self._setup_shutdown_handlers()
            # Connect to room FIRST to get full metadata
            await ctx.connect()
            # Parse room metadata to get bot configuration AFTER connecting
            json_string= ctx.room.metadata or ctx.job.metadata
            if not json_string or json_string.strip() == "":
                logger.error("Room metadata is empty")
                return
            
            try:
                metadata_dict: Dict[str, Any] = json.loads(json_string)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in room metadata: {e}")
                logger.error(f"Metadata content: {json_string[:200]}")
                return
            
            # Get and validate bot configuration
            raw_bot_config = metadata_dict.get("bot_config", {})
            if not raw_bot_config:
                logger.error("Cannot start session without bot configuration")
                return
            # Parse into BotConfiguration dataclass for type safety
            bot_config = BotConfiguration.from_dict(raw_bot_config)
            self.bot_config = bot_config
            
            # Start background cleanup task for Excel cache
            await excel_calculator.start_cleanup_task()
            # Check if this is an outbound call
            sip_trunk_id = os.getenv("SIP_TRUNK_ID", "")
            is_outbound = bot_config.has_call_outbound() and bool(sip_trunk_id)
            
            # Create session and agent based on configuration
            self.session = await SessionFactory.create_session(bot_config)
            self.agent = await self._create_agent(bot_config, ctx, is_outbound=is_outbound)
            # Initialize recording manager (for both inbound and outbound calls)
            # But DON'T start recording yet - will start after session starts or when SIP joins
            recording_enabled = bot_config.has_recording_enabled()
            logger.info(f"[VoiceBot] Recording enabled check result: {recording_enabled}")
            if recording_enabled:
                call_type = "outbound" if bot_config.has_call_outbound() else "inbound"
                logger.info(f"[VoiceBot] Initializing recording manager for {call_type} call")
                self.recording_manager = RecordingManager(bot_config)
                logger.info(f"[VoiceBot] Recording manager initialized (will start when appropriate)")
                if self.bot_config.call_config and self.bot_config.call_config.get("company", "")=="nodo":
                    self.nodo_manager= NodoManager(bot_config)
                    self.nodo_manager.time_start_call=str(time.time())
                if self.bot_config.call_config and self.bot_config.call_config.get("company", "")=="CRM":
                    self.crm_manager= CRMManager(bot_config)
            else:
                logger.info(f"[VoiceBot] Recording NOT enabled - no valid S3 config with bucket")
            # Start the session IMMEDIATELY to avoid 10s timeout warning
            # This establishes the RoomIO connection
            await self._start_session(ctx, self.agent, bot_config)
            # Handle Call Outbound AFTER session is started (Cases 3 & 4)
            if is_outbound:
                try:
                    if self.bot_config.call_config and self.bot_config.call_config.get("call_type", "")!="inbound":
                        await self._initiate_outbound_call(ctx, bot_config, sip_trunk_id)
                    # Update room metadata that call was initiated successfully
                    await self._update_room_status(ctx.room.name, "success")
                except (ValueError, TwirpError) as e:
                    # These errors are already handled in _initiate_outbound_call
                    # Just log and continue with graceful shutdown
                    logger.info(f"Outbound call setup completed with issues: {type(e).__name__}")
                except Exception as e:
                    logger.error(f"Unexpected error during outbound call setup: {e}")
                    raise
            # Setup shutdown callback to ensure recording stops properly
            ctx.add_shutdown_callback(self._on_shutdown)
            await self._shutdown_event.wait()
            
        except TwirpError as e:
            # SIP-related errors - already handled, log summary only
            sip_status = e.metadata.get('sip_status', 'unknown')
            sip_code = e.metadata.get('sip_status_code', 'unknown')
            logger.info(f"Voice bot session ended due to SIP error: {sip_code} - {sip_status}")
        except ValueError as e:
            # Configuration errors - log and exit gracefully
            logger.error(f"Configuration error in voice bot session: {e}")
        except Exception as e:
            # Unexpected errors - log with full traceback
            logger.error(f"Unexpected error in voice bot session: {e}", exc_info=True)
            raise
        finally:
            # Update room metadata that the session has completed
            await self._update_room_status(ctx.room.name, "completed")
            await self._cleanup()
    
    async def _initiate_outbound_call(
        self, 
        ctx: JobContext, 
        bot_config: BotConfiguration, 
        sip_trunk_id: str
    ):
        """
        Initiate outbound SIP call for Cases 3 & 4 (Call Outbound modes)
        Simple error handling without retry logic
        """
        if not bot_config.call_config:
            logger.error("Call config is missing for outbound call")
            raise ValueError("Call config is missing for outbound call")
            
        room_name = ctx.room.name
        await self._remove_non_agent_or_sip(room_name)
        
        phone_number = bot_config.call_config.get("phone_number", "")
        user_name = bot_config.call_config.get("user_name", "")
        
        if not phone_number:
            logger.error("Phone number is missing for outbound call")
            raise ValueError("Phone number is missing for outbound call")
        
        logger.info(f"Initiating outbound call to {phone_number} for user {user_name}")
        
        try:
            await ctx.api.sip.create_sip_participant(
                livekit_api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=sip_trunk_id,
                    sip_call_to=phone_number,
                    participant_identity=f"sip_{phone_number}",
                    wait_until_answered=True,
                )
            )
            
            logger.info(f"Outbound call initiated successfully to {phone_number}")
            
            # Wait for SIP participant to join before triggering greeting
            await self._wait_for_sip_participant_and_greet(ctx.room.name, phone_number)
            
        except TwirpError as e:
            # Handle SIP errors gracefully without retry
            await self._handle_call_failure(phone_number, user_name, e, ctx)
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error during SIP call: {e}")
            await self._handle_call_failure(phone_number, user_name, e, ctx)
    
    async def _wait_for_sip_participant_and_greet(self, room_name: str, phone_number: str, timeout: int = 30):
        """
        Wait for SIP participant to join the room, then trigger agent greeting
        """
        try:
            sip_identity = f"sip_{phone_number}"
            start_time = asyncio.get_event_loop().time()
            
            logger.info(f"Waiting for SIP participant {sip_identity} to join...")
            
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                async with LiveKitAPI() as lkapi:
                    participants = await lkapi.room.list_participants(
                        ListParticipantsRequest(room=room_name)
                    )
                    
                    # Check if SIP participant has joined
                    for p in participants.participants:
                        if p.identity == sip_identity:
                            logger.info(f"[VoiceBot] SIP participant {sip_identity} joined - user answered the call")
                            
                            # Start recording NOW that user has answered
                            if self.recording_manager:
                                try:
                                    logger.info(f"[VoiceBot] Starting recording now that SIP joined...")
                                    success = await self.recording_manager.start_recording(room_name)
                                    if success:
                                        logger.info(f"[VoiceBot] Recording started successfully")
                                    else:
                                        logger.error(f"[VoiceBot] Failed to start recording")
                                except Exception as e:
                                    logger.error(f"[VoiceBot] Error starting recording: {e}")
                            
                            # Trigger delayed greeting
                            if self.agent and isinstance(self.agent, DynamicVoiceAgent):
                                await self.agent.send_delayed_greeting()
                            
                            return
                
                # Wait a bit before checking again
                await asyncio.sleep(0.5)
            
            logger.warning(f"Timeout waiting for SIP participant {sip_identity} to join")
            
        except Exception as e:
            logger.error(f"Error waiting for SIP participant: {e}")
    

    async def _handle_call_failure(
        self, 
        phone_number: str, 
        user_name: str, 
        error: Exception, 
        ctx: JobContext
    ):
        """
        Handle call failure with appropriate logging, webhook notification, and cleanup
        """
        error_details = {
            "phone_number": phone_number,
            "user_name": user_name,
            "error_type": type(error).__name__
        }
        
        if isinstance(error, TwirpError):
            sip_status_code = error.metadata.get('sip_status_code', 'unknown')
            sip_status = error.metadata.get('sip_status', 'unknown')
            
            error_details.update({
                "sip_status_code": sip_status_code,
                "sip_status": sip_status
            })
            
            if sip_status_code == '486':  # Busy Here
                logger.warning(
                    f"Call to {phone_number} ({user_name}) was not answered - "
                    f"line busy or no answer after timeout"
                )
                if self.bot_config.call_config and self.bot_config.call_config.get("company", "")=="nodo" and self.nodo_manager:
                    self.nodo_manager.call_status_num="101"
                    self.nodo_manager.call_status_text="CALL_RING_TIMEOUT"
            elif sip_status_code == '480':  # Temporarily Unavailable
                logger.warning(
                    f"Call to {phone_number} ({user_name}) temporarily unavailable - "
                    f"user may be offline or unreachable"
                )
                if self.bot_config.call_config and self.bot_config.call_config.get("company", "")=="nodo" and self.nodo_manager:
                    self.nodo_manager.call_status_num="102"
                    self.nodo_manager.call_status_text="TEMPORARILY_UNAVAILABLE"
            else:
                logger.warning(
                    f"Call to {phone_number} ({user_name}) failed - "
                    f"SIP {sip_status_code}: {sip_status}"
                )
        else:
            logger.error(
                f"Unexpected error calling {phone_number} ({user_name}): {error}"
            )
            error_details["error_message"] = str(error)
        
        # Update room metadata about the failure
        await self._update_room_status(ctx.room.name, "failed", error_details)
        
        # Gracefully shutdown the session instead of crashing
        logger.info(f"Initiating graceful shutdown due to call failure")
        self._shutdown_event.set()
    
    async def _start_session(
        self, 
        ctx: JobContext, 
        agent: Agent, 
        bot_config: BotConfiguration
    ):
        """
        Start the agent session with appropriate room options
        """
        if not self.session:
            logger.error("Session not initialized")
            return
        
        # Start the actual agent session
        await self._start_agent_session(ctx, agent, bot_config)
    
    async def _start_agent_session(
        self, 
        ctx: JobContext, 
        agent: Agent, 
        bot_config: BotConfiguration
    ):
        """
        Start the actual agent session
        """
        if not self.session:
            logger.error("Session not initialized in _start_agent_session")
            return
            
        await self.session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                close_on_disconnect=True,
                audio_enabled=True,
                video_enabled=False,
                # noise_cancellation=noise_cancellation.BVC(), Hide for self-hosted LiveKit
            ),
            room_output_options=RoomOutputOptions(
                transcription_enabled=True,
            ),
        )
        
        logger.info(f"Session started: {bot_config.get_case_description()}")
    
    async def _on_shutdown(self):
        """Shutdown callback for cleanup"""
        # Clean up recording manager IMMEDIATELY
        if self.recording_manager:
            try:
                logger.info("[VoiceBot] Stopping recording immediately on shutdown...")
                await self.recording_manager.stop_recording()
                logger.info("[VoiceBot] Recording stopped successfully on shutdown")
            except Exception as e:
                logger.error(f"[VoiceBot] Error stopping recording on shutdown: {e}")
        if self.bot_config.call_config and self.bot_config.call_config.get("company", "")=="nodo" and self.nodo_manager:
            await self.nodo_manager.wait_and_process()
        if self.bot_config.call_config and self.bot_config.call_config.get("company", "")=="CRM":
            await self.crm_manager.wait_and_process()

    async def _cleanup(self):
        """Clean up resources"""
        try:
            # Clean up session
            if self.session:
                await asyncio.wait_for(self.session.aclose(), timeout=5.0)
            
            # Close TTS service
            from llm.speech import MySpeechClass
            await MySpeechClass.close_service()
            
            # Close excel calculator session
            await excel_calculator.close()
            
        except asyncio.TimeoutError:
            logger.warning("Session cleanup timed out")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def entrypoint(ctx: JobContext):
    """Entry point for LiveKit agents"""
    voice_bot_session = VoiceBotSession(ctx)
    await voice_bot_session.entrypoint(ctx)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        shutdown_process_timeout=600,
    ))