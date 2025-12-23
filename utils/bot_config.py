"""
Bot Configuration Module
Provides enums and dataclass for parsing and validating voice bot configurations.
"""
import os
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional


class ModelType(Enum):
    """Type of voice model being used"""
    REALTIME = "realtime"  # OpenAI Realtime API with integrated STT/LLM/TTS
    PIPELINE = "pipeline"  # Custom realtime model (not used in current implementation)
    TTS = "tts"  # Non-realtime, uses separate STT + LLM + TTS pipeline
    STT = "stt"  # STT-only mode (not used in current implementation)
    LLM = "llm"  # LLM-only mode (not used in current implementation)


class SessionMode(Enum):
    """Session mode based on agent mode and call outbound"""
    AGENT_MODE = "agent"           # is_agent_mode=True, no call outbound
    CALL_OUTBOUND = "call_outbound"  # is_agent_mode=False, with call outbound


# Constants for realtime models
REALTIME_MODELS = [
    "gpt-realtime",
    "gpt-realtime-2025-08-28",
    "gpt-4o-realtime-preview",
    "gpt-4o-mini-realtime-preview-2024-12-17"
]


@dataclass
class BotConfiguration:
    """
    Parsed and validated bot configuration.
    
    Automatically determines model type and session mode based on the configuration.
    Handles 4 cases:
    - Case 1: Agent Mode + TTS Model (non-realtime, NOT call outbound)
    - Case 2: Agent Mode + Realtime Model (realtime, NOT call outbound)
    - Case 3: Call Outbound + TTS Model (non-realtime, call outbound)
    - Case 4: Call Outbound + Realtime Model (realtime, call outbound)
    """
    bot_id: str
    bot_name: str
    instructions: str
    model_id: str
    voice: str
    tools: list[str]
    knowledge_base_tables: list[str]
    is_agent_mode: bool
    metadata: Dict[str, Any]
    user_id: str
    workspace_id: str
    
    # Derived properties (initialized in __post_init__)
    model_type: Optional[ModelType] = None
    session_mode: Optional[SessionMode] = None
    call_config: Optional[Dict[str, Any]] = None
    reference_audio: str = ""
    reference_text: str = ""
    pipeline_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Determine model type and session mode after initialization"""
        # Determine model type
        self.model_type = (
            ModelType.REALTIME if self.model_id in REALTIME_MODELS 
            else ModelType.PIPELINE
        )
        
        # Determine session mode
        self.session_mode = (
            SessionMode.AGENT_MODE if self.is_agent_mode 
            else SessionMode.CALL_OUTBOUND
        )
        
        # Extract call config for outbound calls
        if self.session_mode == SessionMode.CALL_OUTBOUND:
            self.call_config = self.metadata.get("call_config", {})
        
        # Extract reference audio/text based on model type and mode
        self._extract_reference_config()
    
    def _extract_reference_config(self):
        """
        Extract reference audio and text from metadata based on configuration.
        
        Reference audio/text location varies by case:
        - Agent Mode (Case 1): metadata.reference_audio, metadata.reference_text
        - Call Outbound (Case 3): metadata.call_config.reference_audio, metadata.call_config.reference_text
        - Realtime models (Cases 2 & 4): Not used (voice is selected via API)
        """
        if self.model_type == ModelType.PIPELINE:
            if self.session_mode == SessionMode.CALL_OUTBOUND and self.call_config:
                # Non-realtime + Call Outbound: reference in call_config
                self.reference_audio = self.call_config.get("reference_audio", "")
                self.reference_text = self.call_config.get("reference_text", "")
                self.pipeline_config = self.call_config.get("pipeline", {})
            else:
                # Non-realtime + Agent Mode: reference in metadata root
                self.reference_audio = self.metadata.get("reference_audio", "")
                self.reference_text = self.metadata.get("reference_text", "")
                self.pipeline_config = self.metadata.get("pipeline", {})
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BotConfiguration":
        """Create BotConfiguration from a dictionary"""
        return cls(
            bot_id=config.get("voice_bot_id", ""),
            bot_name=config.get("voice_bot_name", "AI Assistant"),
            instructions=config.get("voice_bot_instructions", ""),
            model_id=config.get("voice_bot_model_id", "gpt-4o-mini-realtime-preview-2024-12-17"),
            voice=config.get("voice_bot_voice", "shimmer"),
            tools=config.get("voice_bot_tools", []),
            knowledge_base_tables=config.get("knowledge_base_table_names", []),
            is_agent_mode=config.get("is_agent_mode", True),
            metadata=config.get("metadata", {}),
            user_id=config.get("user_id", ""),
            workspace_id=config.get("workspace_id", ""),
        )
    
    def get_case_description(self) -> str:
        """Return human-readable description of the current case"""
        model_desc = "Realtime" if self.model_type == ModelType.REALTIME else "TTS (Non-realtime)"
        mode_desc = "Agent Mode" if self.session_mode == SessionMode.AGENT_MODE else "Call Outbound"
        return f"{mode_desc} with {model_desc} model"
    
    def has_call_outbound(self) -> bool:
        """Check if this configuration has call outbound enabled"""
        if self.session_mode != SessionMode.CALL_OUTBOUND:
            return False
        if not self.call_config:
            return False
        phone_number = self.call_config.get("phone_number", "")
        return len(phone_number) > 0
    
    def get_phone_number(self) -> str:
        """Get phone number for outbound call"""
        if self.call_config:
            return self.call_config.get("phone_number", "")
        return ""
    
    def get_user_name(self) -> str:
        """Get user name for outbound call"""
        if self.call_config:
            return self.call_config.get("user_name", "")
        return ""
    
    def get_s3_config(self) -> Optional[Dict[str, Any]]:
        """Get S3 configuration for call recording"""
        if self.call_config:
            return self.call_config.get("s3", {})
        return self.metadata.get("s3", {})
    
    def has_recording_enabled(self) -> bool:
        """Check if recording is enabled for this bot configuration"""
        import logging
        logger = logging.getLogger("voice-bot-worker")
        
        s3_config = self.get_s3_config()
        if not s3_config:
            logger.info("[BotConfig] Recording disabled: No S3 config found")
            return False
        
        # Check if required S3 fields are present
        required_fields = ["uri", "recording_url", "transcript_url"]
        has_bucket = all(s3_config.get(field) for field in required_fields)
        
        if not has_bucket:
            logger.info(f"[BotConfig] Recording disabled: Missing bucket in S3 config: {s3_config}")
        else:
            logger.info(f"[BotConfig] Recording enabled with S3 config: {s3_config}")
        
        return has_bucket
    
    def check_local_llm_model(self) -> bool:
        """Check if the LLM model is a local model"""
        local_models = ["openai/gpt-oss-20b"]
        if self.pipeline_config and len(self.pipeline_config["llm"]) > 0:
            llm_model = self.pipeline_config["llm"]
            if llm_model in local_models:
                return True

        return False
    