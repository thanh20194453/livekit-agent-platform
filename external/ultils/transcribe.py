import logging
import boto3
from botocore.exceptions import ClientError
from typing import Any, Optional

from config.settings import settings
import requests
from urllib.parse import urlparse


logger = logging.getLogger("nodo-manager")
logger.setLevel(logging.ERROR)


class TranscribeUtils:
    """Enhanced Service for AWS S3 operations."""
    def __init__(self, bot_config: Any) -> None:
        self.bot_config = bot_config
        
        # Get S3 configuration from bot config
        s3_config: dict[str, Any] = bot_config.get_s3_config() or {}

        # Extract predefined S3 URLs from bot config
        self.s3_recording_url = self.http_s3_to_uri(s3_config.get("recording_url", ""))
        self.s3_transcript_url = self.http_s3_to_uri(s3_config.get("transcript_url", ""))
        self.s3_uri = s3_config.get("uri", "")
        
        # S3 credentials for validation and operations
        self.s3_bucket: str = settings.aws.s3_bucket_name
        self.s3_region: str = settings.aws.region
        self.s3_access_key: str = settings.aws.access_key.get_secret_value()
        self.s3_secret_key: str = settings.aws.secret_key.get_secret_value()


        # call status
        self.call_status_num: str = ""
        self.call_status_text: str = ""
        self.transcribe: Any = boto3.client(
                "transcribe",
                aws_access_key_id=self.s3_access_key,
                aws_secret_access_key=self.s3_secret_key,
                region_name=settings.aws.region,
            )
        # call_info:
        self.phone_number: str = bot_config.call_config.get("phone_number", "")
        self.lead_id: str = bot_config.call_config.get("lead_id", "")
        self.time_start_call: str = ""
    
    def http_s3_to_uri(self, url: str) -> str:
        """
        Convert S3 HTTP URL to s3://bucket/key format.
        """
        parsed = urlparse(url)
        host = parsed.netloc  
        path = parsed.path.lstrip("/") 

        bucket = host.split(".s3.")[0]

        return f"s3://{bucket}/{path}"
    
    def start_transcribe_job(self, job_name: str, language_code: str = "vi-VN") -> None:
        """Start a transcribe job with 2 speaker"""
        try:
            self.transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={"MediaFileUri": self.s3_recording_url},
                MediaFormat="wav", 
                LanguageCode=language_code,
                Settings={
                    "ShowSpeakerLabels": True,
                    "MaxSpeakerLabels": 2
                }
            )
        except self.transcribe.exceptions.ConflictException:
            logger.error(f"Job {job_name} is exist, continuous check status.")
        except Exception as e:
            logger.error(f"Error in creating Transcribe Job: {e}")
            raise
    
    async def s3_exists(self) -> bool:
        """Checking file is existed in s3 or not"""
        parts = self.s3_recording_url.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1]
        try:

            s3 = boto3.client(
                "s3",
                aws_access_key_id=self.s3_access_key,
                aws_secret_access_key=self.s3_secret_key,
                region_name=settings.aws.region,
            )
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False

    def format_transcribe_result(self, job_name: str) -> str:
        """Format transcribe result"""
        try:
            status = self.transcribe.get_transcription_job(TranscriptionJobName=job_name)
            transcript_uri = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            response = requests.get(transcript_uri)
            results = response.json()["results"]
            
            items = results["items"]
            speaker_transcripts = []
            current_speaker = None
            current_text = ""
            
            for item in items:
                if "speaker_label" in item:
                    speaker_label = item["speaker_label"]
                    if speaker_label != current_speaker:
                        if current_speaker is not None:
                            speaker_transcripts.append(f"{current_speaker.upper()}: {current_text.strip()}")
                        current_speaker = speaker_label
                        current_text = item["alternatives"][0]["content"]
                    else:
                        current_text += " " + item["alternatives"][0]["content"]
                else:
                    current_text += item["alternatives"][0]["content"]

            if current_speaker is not None:
                speaker_transcripts.append(f"{current_speaker.upper()}: {current_text.strip()}")
            
            return "\n".join(speaker_transcripts)
        except Exception as e:
            logger.error(f"error when format transcribe text {e}")
            return ""
    
    def upload_string_to_s3(
        self,
        content_string: str,
    ) -> str:
        """Upload transcribe to s3"""
        try:
            if not hasattr(self, "s3_transcript_url"):
                raise ValueError("self.s3_transcript_url not declare.")

            parsed_uri = urlparse(self.s3_transcript_url)

            if parsed_uri.scheme != "s3":
                raise ValueError("self.s3_transcript_url invalid. Must be  's3://bucket/key'.")

            bucket_name = parsed_uri.netloc
            object_key = parsed_uri.path.lstrip("/")

            s3 = boto3.client(
                "s3",
                aws_access_key_id=self.s3_access_key,
                aws_secret_access_key=self.s3_secret_key,
                region_name=settings.aws.region,
            )

            s3.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=content_string.encode("utf-8"),
                ContentType="text/plain; charset=utf-8",
            )

            return f"s3://{bucket_name}/{object_key}"

        except Exception as e:
            logger.error(f"Error when upload to S3: {e}", exc_info=True)
            raise