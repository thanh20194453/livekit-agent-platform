import asyncio
import logging
import json
import os
from typing import Dict, Any, Optional
import signal
import re, httpx
from livekit.plugins import openai, noise_cancellation
from livekit.agents.llm import function_tool
from livekit.agents import Agent, AgentSession, RunContext, JobContext, RoomInputOptions, RoomOutputOptions, cli, WorkerOptions
from livekit.plugins.openai.llm import LLM
from openai.types.beta.realtime.session import TurnDetection
from livekit.agents.types import NOT_GIVEN
from livekit import api as livekit_api
from dotenv import load_dotenv

from tools.search_information_from_knowledge_base import search_information_from_knowledge_base as search_knowledge_base
from tools.ddgs import ddgs
from tools.scrawl_website import trafilatura_tools
from tools.excel_calculator import ExcelRef, excel_calculator
from tools import nodo_tool
from livekit.api import LiveKitAPI, ListParticipantsRequest, RoomParticipantIdentity

load_dotenv()
logger = logging.getLogger("voice-bot-worker")
logger.setLevel(logging.INFO)




class DynamicVoiceAgent(Agent):
    """Voice agent that dynamically configures based on voice bot data"""

    def __init__(self, instructions: str, knowledge_base_tables: list = [], tool_name: list[str] = [],bot_config: Dict[str, Any] = {} ) -> None:
        super().__init__(instructions=instructions)
        self.knowledge_base_tables = knowledge_base_tables or []
        self.tool_name = tool_name or []
        self.bot_config = bot_config or {}

    async def on_enter(self):
        """Called when agent enters the session"""
        try:
            # Generate initial greeting
            await self.session.generate_reply()
        except Exception as e:
            logger.error(f"Error in on_enter: {e}")

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

    @function_tool(description="Lấy thông tin chi tiết về dự án đang tư vấn từ cơ sở dữ liệu (RAG).")
    async def get_info(self, run_ctx: RunContext, question: str) -> str:
        """Sử dụng tool này khi khách hàng hỏi về thông tin dự án."""
        if "get_info" not in self.tool_name:
             return "Get info tool is not enabled."
        
        # Lấy duan_id từ config
        duan_id = self.bot_config.get("duan_id")
        return await nodo_tool.get_info(question, duan_id)

    @function_tool(description="Tìm kiếm căn hộ theo tiêu chí: location, property_type, area, direction, purpose, num_rooms, price, num_floors.")
    async def get_apartment_info(self, run_ctx: RunContext, location: str = "", property_type:str = "", area:str= "", direction:str= "", purpose:str= "", num_rooms:str= "", price:str= "", num_floors:str= "") -> str:
        """Sử dụng tool này khi khách hàng muốn tìm căn hộ cụ thể."""
        if "get_apartment_info" not in self.tool_name:
             return "Get apartment info tool is not enabled."

        duan_id = self.bot_config.get("duan_id")
        # Gọi hàm từ file nodo_tools
        return await nodo_tool.get_apartment_info(
            duan_id=duan_id,
            location=location,
            property_type=property_type,
            area=area,
            direction=direction,
            purpose=purpose,
            num_rooms=num_rooms,
            price=price,
            num_floors=num_floors
        )

    @function_tool(description="Hỏi thông tin về các dự án khác ngoài dự án hiện tại.")
    async def ask_other_project(self, run_ctx: RunContext, question: str) -> str:
        """Sử dụng tool này khi khách hàng hỏi về dự án khác."""
        if "ask_other_project" not in self.tool_name:
             return "Ask other project tool is not enabled."
        
        return await nodo_tool.ask_other_project(question)


class VoiceBotSession:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.session = None
        self._shutdown_event = asyncio.Event()

    def _build_instructions(self, bot_config: Dict[str, Any]) -> str:
        # ... (Code giữ nguyên như cũ) ...
        base_instructions = bot_config.get("voice_bot_instructions", "")
        bot_name = bot_config.get("voice_bot_name", "AI Assistant")
        # ... 
        return f"""...""" # (Copy lại phần build instructions của bạn nếu cần)

    # ... (Các hàm _get_bot_configuration, _verify_realtime_model_access giữ nguyên) ...

    async def _create_agent(self, bot_config: Dict[str, Any]) -> Agent:
        """Create the voice agent with configuration"""
        # Cần đảm bảo hàm _build_instructions có sẵn ở trên
        instructions = self._build_instructions(bot_config) 
        kb_tables = bot_config.get("knowledge_base_table_names", [])
        tool_name = bot_config.get("voice_bot_tools", [])
        
        agent = DynamicVoiceAgent(
            instructions=instructions,
            knowledge_base_tables=kb_tables,
            tool_name=tool_name,
            bot_config=bot_config  # <--- Quan trọng: Truyền bot_config vào đây
        )
        return agent

class VoiceBotSession:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.session = None
        self._shutdown_event = asyncio.Event()

    def _build_instructions(self, bot_config: Dict[str, Any]) -> str:
        """Build comprehensive instructions for the dynamic agent"""
        base_instructions = bot_config.get("voice_bot_instructions", "")
        bot_name = bot_config.get("voice_bot_name", "AI Assistant")
        kb_tables = bot_config.get("knowledge_base_table_names", [])
        call_info = bot_config.get("call_info", {})
        user_name = call_info.get("user_name", "User")
        
        instructions = f"""# CORE IDENTITY
You are {bot_name}, a helpful AI voice assistant.

# LANGUAGE & COMMUNICATION
- Always respond in Vietnamese unless the user using English first
- Use natural, conversational tone appropriate for voice interaction
- Keep responses concise (1-3 sentences) unless detailed explanation is requested
- Speak clearly and at moderate pace
- Show empathy and understanding

# BEHAVIOR GUIDELINES
- Be professional yet warm and approachable
- Listen actively and ask clarifying questions when needed
- Respond immediately if you understand the user's request and need to avoid repeating information
- If uncertain about something, acknowledge it honestly
- Stay focused on helping the user achieve their goals
- Maintain conversation context throughout the session, but do not repeat.

# INITIAL GREETING
When you first join the conversation, greet warmly:
"Xin chào! Tôi là {bot_name}, trợ lý AI của bạn. Tôi có thể giúp gì cho bạn hôm nay?"

# CONVERSATION FLOW
- Acknowledge user input clearly
- Provide helpful, accurate responses
- Offer additional assistance when appropriate
- If you need to search for information, tell the user: "Để tôi tìm kiếm thông tin này..." and make a sound like you are typing on a computer
- If a user needs information on the internet or asks you to search for information from a website, use web search engines instead of searching your knowledge base.
"""
        # Add custom instructions
        if base_instructions.strip():
            instructions += f"\n# SPECIFIC ROLE INSTRUCTIONS\n{base_instructions}\n"
            
        # Add knowledge base info
        if kb_tables:
            instructions += f"""
# KNOWLEDGE BASE ACCESS
You have access to search the following knowledge bases: {', '.join(kb_tables)}
- Use when users ask questions requiring specific information
- Always inform users when searching: "Để tôi tìm kiếm thông tin này trong cơ sở dữ liệu..."
- Present search results clearly and helpfully
- If search fails, apologize and offer alternative help
"""

        # Add error handling guidance
        instructions += """
# ERROR HANDLING
- If you don't understand: "Xin lỗi, tôi không hiểu rõ. Bạn có thể nói rõ hơn được không?"
- If search fails: "Xin lỗi, tôi gặp khó khăn khi tìm kiếm. Bạn có thể thử hỏi cách khác?"
- If outside expertise: "Đây không phải chuyên môn của tôi, nhưng tôi sẽ cố gắng giúp bạn."

# CONVERSATION EXAMPLES
User: "Xin chào!"
You: "Em chào anh/chị {user_name}! Em là {bot_name}. Em có thể giúp gì cho bạn hôm nay?"

User: "Can you help me?"
You: "Of course! I'm here to help. What do you need assistance with?"
""".format(bot_name=bot_name, user_name=user_name)
        
        return instructions

    def _get_fallback_instructions(self) -> str:
        """Fallback instructions when no config is available"""
        return """# DEFAULT AI ASSISTANT

You are a helpful AI voice assistant.

# BEHAVIOR
- Respond primarily in Vietnamese, switch to English if user prefers
- Use natural, conversational tone for voice interaction
- Keep responses brief and clear
- Be helpful, friendly, and professional

# GREETING
Start conversations with: "Xin chào! Tôi là trợ lý AI. Tôi có thể giúp gì cho bạn?"
"""

    async def _get_bot_configuration(self) -> Optional[Dict[str, Any]]:
        """Get bot configuration with improved error handling"""
        try:
            # Wait for participant with timeout
            participant = await asyncio.wait_for(
                self.ctx.wait_for_participant(), 
                timeout=300.0
            )
            
            # Check if participant has metadata
            if not hasattr(participant, 'metadata') or not participant.metadata:
                return None
                
            try:
                metadata = json.loads(participant.metadata)
                
                if 'bot_config' in metadata:
                    bot_config = metadata['bot_config']
                    return bot_config
                else:
                    return None
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse participant metadata as JSON: {e}")
                return None
                
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for participant")
            return None
        except Exception as e:
            logger.error(f"Error getting bot configuration: {e}")
            return None
        
    def _verify_realtime_model_access(self, model: str) -> bool:
        """Verify access to the specified OpenAI Realtime model"""
        try:
            REALTIME_MODELS = [
                "gpt-realtime",
                "gpt-realtime-2025-08-28",
                "gpt-4o-realtime-preview",
                "gpt-4o-mini-realtime-preview-2024-12-17"
            ]
            
            if model not in REALTIME_MODELS:
                logger.error(f"Model {model} is not in the list of supported realtime models")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to verify model access for {model}: {e}")
            return False

    async def _create_agent_session(self, bot_config: Dict[str, Any]) -> AgentSession:
        """Create and configure the agent session"""
        model = bot_config.get("voice_bot_model_id", "gpt-4o-mini-realtime-preview-2024-12-17")
        voice = bot_config.get("voice_bot_voice", "shimmer")
        
        is_realtime_model = self._verify_realtime_model_access(model)
        api_key = os.getenv("OPENAI_API_KEY_VOICE")
        
        if not is_realtime_model:
            # Create standard LLM session with STT and TTS
            return AgentSession(
                llm=LLM(
                    model=model,
                    api_key=api_key if api_key else NOT_GIVEN,
                    parallel_tool_calls=True,
                ),
                preemptive_generation=True,
                stt=openai.STT(
                    model="gpt-4o-mini-transcribe",
                    api_key=api_key if api_key else NOT_GIVEN,
                    use_realtime=True,
                    language="vi"  # Force Vietnamese for better transcription
                ),
                tts=openai.TTS(
                    model="gpt-4o-mini-tts",
                    voice=voice,
                    api_key=api_key if api_key else NOT_GIVEN,
                ),
            )

        # Create realtime model session
        llm = openai.realtime.RealtimeModel(
            model=model,
            voice=voice,
            api_key=api_key,
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.3,
                interrupt_response=True,
                silence_duration_ms=800,
                create_response=True,
                prefix_padding_ms=300,
            ),
        )

        return AgentSession(llm=llm)

    async def _create_agent(self, bot_config: Dict[str, Any]) -> Agent:
        """Create the voice agent with configuration"""
        instructions = self._build_instructions(bot_config)
        kb_tables = bot_config.get("knowledge_base_table_names", [])
        tool_name = bot_config.get("voice_bot_tools", [])
        logger.info(f"Creating agent with tools: {tool_name} and knowledge bases: {kb_tables}")
        
        agent = DynamicVoiceAgent(
            instructions=instructions,
            knowledge_base_tables=kb_tables,
            tool_name=tool_name,
            bot_config=bot_config
        )
        
        return agent

    async def _setup_shutdown_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            self._shutdown_event.set()
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def _remove_non_agent_or_sip(self,room_name):
        logger.info(f"ok")
        print("ok")
        async with LiveKitAPI() as lkapi:
            logger.info(f"ok2")
            print("ok2")
            # 1. Lấy danh sách participant
            res = await lkapi.room.list_participants(
                ListParticipantsRequest(room=room_name)
            )
            print(res)
            for p in res.participants:
                print(p)
                identity = p.identity

                # 2. Bỏ qua agent và sip
                if identity.startswith("agent") or identity.startswith("sip"):
                    continue
                # 3. Remove participant
                else:
                    await lkapi.room.remove_participant(
                        RoomParticipantIdentity(
                            room=room_name,
                            identity=identity
                        )
                    )

                    print(f"Đã xóa participant: {identity}")
                return None

    async def entrypoint(self, ctx: JobContext):
        """Main entry point for the voice bot session"""
        try:
            # Setup shutdown handling
            await self._setup_shutdown_handlers()
            
            # Connect to room
            await ctx.connect()
            
            # Start background cleanup task for Excel cache
            await excel_calculator.start_cleanup_task()
            # logger.info(ctx.room.metadata)
            json_string = ctx.room.metadata
            metadata_dict: Dict[str, Any] = json.loads(json_string)
            # Get bot configuration
            # bot_config = await self._get_bot_configuration()
            bot_config=metadata_dict.get('bot_config')
            logger.info(bot_config)
            if not bot_config:
                logger.error("Cannot start session without bot configuration")
                return
            
            logger.info(f"Bot configuration: {bot_config}")
            
            # Create session and agent
            self.session = await self._create_agent_session(bot_config)
            agent = await self._create_agent(bot_config)
            
            # Get Call Info
            call_info = bot_config.get("call_info", {})
            sip_trunk_id = os.getenv("SIP_TRUNK_ID", "")
            logger.info(f"SIP Call Info: {call_info}")
            logger.info(f"Call Info: {call_info}")
            # nameroom=ctx.room.name
            # a= await self._remove_non_agent_or_sip(nameroom)
            if call_info and sip_trunk_id:
                nameroom=ctx.room.name
                a= await self._remove_non_agent_or_sip(nameroom)
                logger.info(f"Voice bot session started for call: {call_info}")
                phone_number = call_info.get("phone_number", "")
                if not phone_number:
                    logger.info("Phone number is missing in call info")
                    return
                await ctx.api.sip.create_sip_participant(
                    livekit_api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=sip_trunk_id,
                    sip_call_to=phone_number,
                    participant_identity="sip_"+phone_number,
                    wait_until_answered=True,
                    )
                )
            
            #await ctx.api.sip.create_sip_participant(
            #    livekit_api.CreateSIPParticipantRequest(
            #    room_name=ctx.room.name,
            #    sip_trunk_id=sip_trunk_id="ST_yu3yDWpW98QN",
            #    sip_call_to="0968054796",
            #    participant_identity="sip_"+"0968054796",
            #    wait_until_answered=True,
            #    )
            #)
                
            # Start the session
            await self.session.start(
                agent=agent,
                room=ctx.room,
                room_input_options=RoomInputOptions(
                    close_on_disconnect=True,
                    # Optimize for smoother voice processing
                    audio_enabled=True,
                    video_enabled=False,
                    
                    # Enable noise cancellation
                    noise_cancellation=noise_cancellation.BVC()
                ),
                room_output_options=RoomOutputOptions(
                    transcription_enabled=True,
                ),
            )
            
            # Wait for shutdown signal or room disconnect
            await self._shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Error in voice bot session: {e}", exc_info=True)
            raise
        finally:
            await self._cleanup()

    async def _cleanup(self):
        """Clean up resources"""
        try:
            if self.session:
                await asyncio.wait_for(self.session.aclose(), timeout=5.0)
            
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
    ))