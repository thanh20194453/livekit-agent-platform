"""
Session Factory Module
Provides factory methods for creating agent sessions based on bot configuration.
"""
import os
import logging
from typing import Optional

from livekit.plugins import openai, silero
from livekit.agents import AgentSession, stt
from openai.types.beta.realtime.session import TurnDetection

from utils.bot_config import BotConfiguration, ModelType
# from llm.transcrbile import MyTranscribeClass
from llm.speech import MySpeechClass
from llm.aws_transcrible import AWSTranscribeSTT
from config.settings import settings

logger = logging.getLogger("voice-bot-worker")
logger.setLevel(logging.ERROR)


class SessionFactory:
    """
    Factory for creating agent sessions based on bot configuration.
    
    Handles all 4 cases:
    - Case 1: Agent Mode + Pipeline configuration (non-realtime, NOT call outbound)
    - Case 2: Agent Mode + Realtime Model (realtime, NOT call outbound)
    - Case 3: Call Outbound + Pipeline configuration (non-realtime, call outbound)
    - Case 4: Call Outbound + Realtime Model (realtime, call outbound)
    """
    
    @classmethod
    async def create_session(cls, bot_config: BotConfiguration) -> AgentSession:
        """
        Create and configure the agent session based on bot configuration.
        
        Args:
            bot_config: Parsed bot configuration
            
        Returns:
            AgentSession: Configured agent session
        """
        logger.info(f"Creating agent session: {bot_config.get_case_description()}")
        logger.info(f"Model: {bot_config.model_id}, Voice: {bot_config.voice}")
        
        if bot_config.model_type == ModelType.REALTIME:
            # Cases 2 & 4: Realtime model (both Agent Mode and Call Outbound)
            return cls._create_realtime_session(bot_config)
        else:
            # Cases 1 & 3: Pipeline configuration (both Agent Mode and Call Outbound)
            return await cls._create_pipeline_session(bot_config)
    
    @classmethod
    def _create_realtime_session(cls, bot_config: BotConfiguration) -> AgentSession:
        """
        Create session for Realtime model (Cases 2 & 4)
        - Uses OpenAI Realtime API for integrated STT/LLM/TTS
        - Supports both Agent Mode and Call Outbound
        """
        api_key = os.getenv("OPENAI_API_KEY_VOICE")
        
        llm = openai.realtime.RealtimeModel(
            model=bot_config.model_id,
            voice=bot_config.voice,
            api_key=api_key,
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.5,
                interrupt_response=True,
                silence_duration_ms=800,
                create_response=True,
                prefix_padding_ms=300,
            ),
        )
        
        logger.info(f"Created Realtime session with model: {bot_config.model_id}")
        return AgentSession(llm=llm)
    
    @classmethod
    async def _create_pipeline_session(cls, bot_config: BotConfiguration) -> AgentSession:
        """
        Create session for pipeline configuration (Cases 1 & 3)
        - Uses separate STT + LLM + TTS pipeline
        - Supports both Agent Mode and Call Outbound
        - Reference audio/text used for voice cloning (if available)
        """
        # Create STT based on provider
        try:
            stt_instance = cls._create_stt_instance(bot_config)
            logger.info(f"✓ STT instance created: {type(stt_instance).__name__}")
        except Exception as e:
            logger.error(f"✗ Failed to create STT instance: {e}", exc_info=True)
            raise
        
        # Create TTS with reference audio from bot config
        tts_instance = cls._create_tts_instance(bot_config)
        
        # Create LLM with pipeline configuration
        llm_instance = cls._create_llm_instance(bot_config)
        
        # Configure VAD for phone calls - balanced sensitivity
        # Higher values = less sensitive, fewer interruptions
        vad = silero.VAD.load()
        
        return AgentSession(
            llm=llm_instance,
            stt=stt_instance,
            tts=tts_instance,
            vad=vad,
            min_endpointing_delay=0.8,  # Wait longer before ending turn
            preemptive_generation=True,
        )
        
    @classmethod
    def _create_llm_instance(cls, bot_config: BotConfiguration) -> openai.LLM:
        """
        Create LLM instance based on pipeline configuration.
        
        Args:
            model_id: LLM model identifier
            
        Returns:
            openai.LLM: Configured LLM instance
        """
        try:
            is_local_llm_model = bot_config.check_local_llm_model()
            if bot_config.pipeline_config and len(bot_config.pipeline_config["llm"]) > 0:
                llm_model = bot_config.pipeline_config["llm"]
                if is_local_llm_model:
                    return openai.LLM(
                        model=settings.llm.model,
                        base_url=settings.llm.base_url,
                    )
                else:
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key is None:
                        raise ValueError("OPENAI_API_KEY environment variable is not set")
                    return openai.LLM(
                        model=llm_model,
                        api_key=api_key,
                    )
            else:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key is None:
                    raise ValueError("OPENAI_API_KEY environment variable is not set")
                return openai.LLM(
                    model="gpt-4o",
                    api_key=api_key,
                )
        except Exception as e:
            logger.error(f"✗ Failed to create LLM instance: {e}", exc_info=True)
            raise
    
    @classmethod
    def _create_stt_instance(cls, bot_config: BotConfiguration):
        """
        Create STT instance based on provider configuration.
        
        Args:
            provider: STT provider name ("aws" or "local")
            
        Returns:
            STT instance
        """
        # AWS Transcribe - streaming STT (16kHz for speech recognition)
        return AWSTranscribeSTT(language="vi-VN", sample_rate=16000)
        
        # if provider == "local":
        #     # Local Whisper STT needs StreamAdapter because it's non-streaming
        #     # Create VAD specifically for StreamAdapter
        #     vad_for_adapter = silero.VAD.load(
        #         min_speech_duration=0.1,
        #         min_silence_duration=0.3,
        #         padding_duration=0.1,
        #         sample_rate=16000,
        #     )
        #     logger.info("Creating Local Whisper STT with StreamAdapter + VAD")
        #     return stt.StreamAdapter(
        #         stt=MyTranscribeClass(),
        #         vad=vad_for_adapter,
        #     )
        # else:
        #     # AWS Transcribe - streaming STT (16kHz for speech recognition)
        #     return AWSTranscribeSTT(language="vi-VN", sample_rate=16000)
    
    @classmethod
    def _create_tts_instance(cls, bot_config: BotConfiguration) -> MySpeechClass:
        """
        Create TTS instance with reference audio from bot configuration.
        
        Reference audio is extracted based on the session mode:
        - Agent Mode: from metadata root
        - Call Outbound: from call_config
        
        Args:
            bot_config: Bot configuration with reference audio/text
            
        Returns:
            MySpeechClass: Configured TTS instance
        """
        reference_audio = bot_config.reference_audio or None
        reference_text = bot_config.reference_text or None
        
        if reference_audio and reference_text:
            logger.info(f"Using reference audio: {reference_audio}")
        else:
            logger.info("Using default TTS voice (no reference audio)")
        
        return MySpeechClass(
            reference_audio=reference_audio,
            reference_text=reference_text
        )
