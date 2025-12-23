"""
Recording Configuration Module
Provides configuration constants and utilities for call recording.
"""
import os
from typing import Dict, Any


class RecordingConfig:
    """Configuration constants for recording functionality"""
    
    # S3 Configuration
    DEFAULT_S3_BUCKET = os.getenv("S3_BUCKET_NAME", "livekit-recordings")
    DEFAULT_S3_REGION = os.getenv("AWS_REGION", "us-east-1")
    DEFAULT_S3_PREFIX = "recordings"
    
    # Recording Settings
    DEFAULT_AUDIO_FORMAT = "mp4"
    DEFAULT_AUDIO_PRESET = "H264_720P_30"  # LiveKit preset for audio recording
    
    # Transcription Settings
    DEFAULT_TRANSCRIBE_API_ENDPOINT = "http://54.255.208.182:8129/process_and_wait"
    DEFAULT_POLL_INTERVAL = 5
    DEFAULT_MAX_WAIT_TIME = 1800  # 30 minutes
    
    # File Waiting Settings
    DEFAULT_MAX_WAIT_SECONDS = 300  # 5 minutes to wait for S3 file
    DEFAULT_CHECK_INTERVAL = 3  # Check every 3 seconds
    
    @classmethod
    def get_default_s3_config(cls) -> Dict[str, Any]:
        """Get default S3 configuration from environment variables"""
        return {
            "bucket": cls.DEFAULT_S3_BUCKET,
            "region": cls.DEFAULT_S3_REGION,
            "access_key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "prefix": cls.DEFAULT_S3_PREFIX
        }
    
    @classmethod
    def get_default_transcription_config(cls) -> Dict[str, Any]:
        """Get default transcription configuration"""
        return {
            "enabled": True,
            "api_endpoint": os.getenv("TRANSCRIBE_API_ENDPOINT", cls.DEFAULT_TRANSCRIBE_API_ENDPOINT),
            "poll_interval": cls.DEFAULT_POLL_INTERVAL,
            "max_wait_time": cls.DEFAULT_MAX_WAIT_TIME
        }
    
    @classmethod
    def validate_s3_config(cls, config: Dict[str, Any]) -> bool:
        """Validate S3 configuration"""
        required_fields = ["bucket", "region", "access_key", "secret_key"]
        return all(config.get(field) for field in required_fields)


# Environment variable validation
def check_recording_environment() -> Dict[str, str]:
    """
    Check if all required environment variables for recording are set.
    
    Returns:
        Dict with missing environment variables and their descriptions
    """
    required_env_vars = {
        "AWS_ACCESS_KEY_ID": "AWS access key for S3 uploads",
        "AWS_SECRET_ACCESS_KEY": "AWS secret key for S3 uploads", 
        "AWS_REGION": "AWS region for S3 bucket",
        "S3_BUCKET_NAME": "S3 bucket name for storing recordings",
        "LIVEKIT_API_KEY": "LiveKit API key for egress operations",
        "LIVEKIT_API_SECRET": "LiveKit API secret for egress operations"
    }
    
    missing_vars = {}
    for var_name, description in required_env_vars.items():
        if not os.getenv(var_name):
            missing_vars[var_name] = description
    
    return missing_vars


# Optional environment variables
OPTIONAL_ENV_VARS = {
    "TRANSCRIBE_API_ENDPOINT": f"API endpoint for transcription (default: {RecordingConfig.DEFAULT_TRANSCRIBE_API_ENDPOINT})",
    "SIP_TRUNK_ID": "SIP trunk ID for outbound calls (required for call outbound mode)"
}