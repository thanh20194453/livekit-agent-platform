# ============================================================================
# AWS Transcribe STT Implementation for LiveKit
# ============================================================================

import os
import logging
from livekit.plugins import aws
from dotenv import load_dotenv
from config.settings import settings

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class AWSTranscribeSTT(aws.STT):
    """
    AWS Transcribe STT wrapper for LiveKit.
    
    Supports streaming transcription with:
    - Sample rates: 8000, 16000, 32000, 48000 Hz
    - Vietnamese (vi-VN) and other AWS supported languages
    
    Note: Uses environment variables for credentials due to livekit-plugins-aws compatibility.
    """
    
    def __init__(
        self,
        language: str = "vi-VN",
        sample_rate: int = 16000,
    ) -> None:
        # Validate sample rate
        valid_rates = [8000, 16000, 32000, 48000]
        if sample_rate not in valid_rates:
            logger.warning(f"Invalid sample rate {sample_rate}Hz, using 16000Hz")
            sample_rate = 16000
        
        # Set AWS credentials via environment variables
        os.environ["AWS_ACCESS_KEY_ID"] = settings.aws.access_key.get_secret_value()
        os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws.secret_key.get_secret_value()
        os.environ["AWS_REGION"] = settings.aws.region
        
        super().__init__(
            language=language,
            region=settings.aws.region,
            sample_rate=sample_rate,
        )
        
        logger.info(f"AWS Transcribe STT initialized: {language}, {sample_rate}Hz")