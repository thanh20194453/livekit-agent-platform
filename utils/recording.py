from datetime import datetime, timezone, timedelta
import os
import uuid
import logging
from typing import Dict, Any, Optional, Callable
from contextlib import asynccontextmanager

from livekit import api

from config.settings import settings

logger = logging.getLogger("recording-manager")
logger.setLevel(logging.ERROR)


def get_current_time() -> str:
    """Get current time in Vietnam timezone"""
    utc_now = datetime.now(timezone.utc)
    tz_plus7 = timezone(timedelta(hours=7))
    local_time = utc_now.astimezone(tz_plus7)
    return local_time.strftime("%d-%m-%Y %H:%M:%S")


class RecordingManager:
    """
    Manages call recordings with LiveKit egress and S3 storage.
    
    Features:
    - Automatic S3 path generation based on call metadata
    - Configurable S3 settings from bot configuration
    - Transcription processing after recording
    - Proper error handling and cleanup
    - Context manager support for resource management
    """
    
    def __init__(self, bot_config, call_id: Optional[str] = None):
        self.bot_config = bot_config
        self.call_id = call_id or str(uuid.uuid4())
        self.egress_info = None
        self.lkapi = None
        self.call_start_time = get_current_time()
        
        # Get S3 configuration from bot config
        s3_config = bot_config.get_s3_config() or {}
        
        # Extract predefined S3 URLs from bot config
        self.s3_recording_url = s3_config.get("recording_url", "")
        self.s3_transcript_url = s3_config.get("transcript_url", "")
        self.s3_uri = s3_config.get("uri", "")
        
        # Parse S3 bucket and region from recording_url or use defaults
        self.s3_bucket, self.s3_region, self.s3_file_path = self._parse_recording_url()
        
        # S3 credentials for validation and operations
        self.s3_access_key = settings.aws.access_key.get_secret_value()
        self.s3_secret_key = settings.aws.secret_key.get_secret_value()
        
        logger.info(f"[Recording] Initialized RecordingManager for call {self.call_id}")
        logger.info(f"[Recording] S3 URI: {self.s3_uri}")
        logger.info(f"[Recording] Recording URL: {self.s3_recording_url}")
        logger.info(f"[Recording] Parsed - Bucket: {self.s3_bucket}, Region: {self.s3_region}, Path: {self.s3_file_path}")
    
    def _parse_recording_url(self) -> tuple[str, str, str]:
        """Parse recording URL to extract bucket, region and file path
        
        Returns:
            tuple: (bucket_name, region, file_path)
        """
        if not self.s3_recording_url:
            # Fallback to defaults if no recording_url
            logger.warning("[Recording] No recording_url found, using defaults")
            return (
                settings.aws.s3_bucket_name,
                settings.aws.region,
                f"recordings/{self.call_id}/{self.call_start_time.replace(' ', '_').replace(':', '-')}.wav"
            )
        
        try:
            # Parse URL like: https://svisor-dev.s3.ap-southeast-1.amazonaws.com/workspaces/.../recording.ogg
            from urllib.parse import urlparse
            parsed = urlparse(self.s3_recording_url)
            
            # Extract bucket from hostname (e.g., "svisor-dev.s3.ap-southeast-1.amazonaws.com")
            hostname_parts = parsed.hostname.split('.')
            bucket = hostname_parts[0] if hostname_parts else settings.aws.s3_bucket_name
            
            # Extract region from hostname (e.g., "ap-southeast-1")
            region = hostname_parts[2] if len(hostname_parts) > 2 else settings.aws.region
            
            # Extract file path (remove leading slash)
            file_path = parsed.path.lstrip('/')
            
            logger.info(f"[Recording] Parsed URL - Bucket: {bucket}, Region: {region}, Path: {file_path}")
            return (bucket, region, file_path)
            
        except Exception as e:
            logger.error(f"[Recording] Failed to parse recording_url: {e}, using defaults")
            return (
                settings.aws.s3_bucket_name,
                settings.aws.region,
                f"recordings/{self.call_id}/{self.call_start_time.replace(' ', '_').replace(':', '-')}.wav"
            )
    
    def _get_recording_file_path(self) -> str:
        """Get the recording file path from parsed URL or generate default"""
        return self.s3_file_path
    
    def _generate_s3_path(self) -> str:
        """Use predefined S3 path from bot config"""
        return self._get_recording_file_path()
    
    def _get_s3_uri(self) -> str:
        """Get S3 URI from bot config"""
        return self.s3_uri
    
    async def _validate_s3_config(self) -> bool:
        """Validate S3 configuration"""
        required_fields = {
            "bucket": self.s3_bucket,
            "region": self.s3_region,
            "access_key": self.s3_access_key,
            "secret_key": self.s3_secret_key
        }
        
        missing_fields = [field for field, value in required_fields.items() if not value]
        if missing_fields:
            logger.error(f"[Recording] Missing S3 configuration: {missing_fields}")
            return False
        
        return True
    
    async def start_recording(self, room_name: str) -> bool:
        """
        Start recording for the given room.
        
        Args:
            room_name: LiveKit room name
            
        Returns:
            bool: True if recording started successfully
        """
        try:
            # Validate S3 configuration
            if not await self._validate_s3_config():
                return False
            
            # Initialize LiveKit API
            self.lkapi = api.LiveKitAPI()
            
            # Use predefined S3 path from config
            s3_path = self._get_recording_file_path()
            
            if not self.s3_uri:
                logger.error("[Recording] No S3 URI configured in bot config")
                return False
            
            logger.info(f"[Recording] Starting recording for room: {room_name}")
            logger.info(f"[Recording] S3 URI: {self.s3_uri}")
            logger.info(f"[Recording] S3 File Path: {s3_path}")
            
            # Configure S3 output
            s3_output = api.S3Upload(
                bucket=self.s3_bucket,
                region=self.s3_region,
                access_key=self.s3_access_key,
                secret=self.s3_secret_key,
            )
            
            # Configure file output
            file_output = api.EncodedFileOutput(
                filepath=s3_path,
                s3=s3_output,
            )
            
            # Configure egress request for audio-only recording
            egress_request = api.RoomCompositeEgressRequest(
                room_name=room_name,
                audio_only=True,
                file_outputs=[file_output],
                preset=api.EncodingOptionsPreset.H264_720P_30,
            )
            
            # Start recording
            self.egress_info = await self.lkapi.egress.start_room_composite_egress(egress_request)
            logger.info(f"[Recording] Recording started successfully. Egress ID: {self.egress_info.egress_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"[Recording] Failed to start recording: {e}")
            await self._cleanup_api()
            return False
    
    async def stop_recording(self) -> bool:
        """
        Stop the current recording.
        
        Returns:
            bool: True if recording stopped successfully
        """
        if not self.egress_info:
            logger.warning("[Recording] No active recording to stop")
            return False
        
        try:
            logger.info(f"[Recording] Stopping recording - Egress ID: {self.egress_info.egress_id}")
            
            if self.lkapi:
                try:
                    # Stop the egress
                    stop_response = await self.lkapi.egress.stop_egress(
                        api.StopEgressRequest(egress_id=self.egress_info.egress_id)
                    )
                    logger.info(f"[Recording] Recording stopped successfully. Egress ID: {self.egress_info.egress_id}")
                    
                    # Log file results if available
                    if hasattr(stop_response, 'file_results') and stop_response.file_results:
                        for idx, file_result in enumerate(stop_response.file_results):
                            if hasattr(file_result, 'size'):
                                logger.info(f"[Recording] File {idx} size: {file_result.size} bytes")
                                
                except Exception as stop_error:
                    # Handle case where egress is already complete
                    error_msg = str(stop_error)
                    if "EGRESS_COMPLETE cannot be stopped" in error_msg or "failed_precondition" in error_msg:
                        logger.info(f"[Recording] Egress already completed, no need to stop. Egress ID: {self.egress_info.egress_id}")
                    else:
                        # Re-raise if it's a different error
                        raise
                
                logger.info(f"[Recording] File location: {self.s3_recording_url}")
            
            return True
            
        except Exception as e:
            logger.error(f"[Recording] Error stopping recording: {e}")
            return False
        finally:
            await self._cleanup_api()
    
    def get_recording_info(self) -> Dict[str, str]:
        """
        Get recording information including URLs.
        
        Returns:
            Dict with recording and transcript URLs
        """
        return {
            "s3_uri": self.s3_uri,
            "recording_url": self.s3_recording_url,
            "transcript_url": self.s3_transcript_url,
            "call_id": self.call_id
        }
    
    async def _cleanup_api(self) -> None:
        """Clean up LiveKit API resources"""
        if self.lkapi:
            try:
                await self.lkapi.aclose()
            except Exception as e:
                logger.warning(f"[Recording] Error closing LiveKit API: {e}")
            finally:
                self.lkapi = None
    
    @asynccontextmanager
    async def recording_session(self, room_name: str):
        """
        Context manager for automatic recording lifecycle management.
        
        Usage:
            async with recording_manager.recording_session(room_name):
                # Recording is active during this block
                await do_call_stuff()
            # Recording is automatically stopped and transcription started
        """
        try:
            success = await self.start_recording(room_name)
            if not success:
                logger.error(f"[Recording] Failed to start recording session")
                yield False
                return
            
            logger.info(f"[Recording] Recording session started for room: {room_name}")
            yield True
            
        except Exception as e:
            logger.error(f"[Recording] Error in recording session: {e}")
            yield False
        finally:
            # Always try to stop recording
            await self.stop_recording()
            logger.info(f"[Recording] Recording session ended for room: {room_name}")
            
            # Log recording info for reference
            recording_info = self.get_recording_info()
            logger.info(f"[Recording] Recording info: {recording_info}")
