"""
Enhanced AWS S3 Service for file upload/download operations with presigned URLs.
"""
from functools import partial
import logging
import boto3
import asyncio
import base64
from typing import Dict, Any, Optional, List
from uuid import UUID
from pathlib import Path
from botocore.config import Config
from botocore.exceptions import NoCredentialsError, BotoCoreError, ClientError
from pydantic import BaseModel
from logging import getLogger

from config.settings import settings
from utils.dttm import current_utc_str

logger = getLogger(__name__)
logger.setLevel(logging.ERROR)

# Data Models

class S3FileMetadata(BaseModel):
    s3_key: str
    content_length: int
    content_type: str
    metadata: dict[str, Any]
    
    class Config:
        from_attributes = True
        

class S3PresignedURLResponse(BaseModel):
    presigned_url: str
    expiry: Optional[int]  # in seconds
    

# Exceptions

class S3UploadError(Exception):
    """Custom exception for S3 upload errors."""
    pass

class S3Error(Exception):
    """Custom exception for S3 related errors."""
    pass


class S3Service:
    """Enhanced Service for AWS S3 operations."""
    
    def __init__(self):
        """Initialize S3 service with configuration."""
        self.s3_client = self._init_s3_client()
        self.bucket_name = settings.aws.s3_bucket_name
        self._executor = None
        self._private_folder_prefix = "workspaces"
        self._public_folder_prefix = "public"
    
    def _init_s3_client(self):
        """Initialize S3 client with configuration."""
        try:
            if not all([
                settings.aws.access_key.get_secret_value(), 
                settings.aws.secret_key.get_secret_value(), 
                settings.aws.region
            ]):
                # Use default credentials
                return boto3.client('s3', region_name=settings.aws.region)
                
            return boto3.client(
                's3',
                aws_access_key_id=settings.aws.access_key.get_secret_value(),
                aws_secret_access_key=settings.aws.secret_key.get_secret_value(),
                region_name=settings.aws.region,
                config=Config(
                    signature_version='s3v4',
                    retries={'max_attempts': 3, 'mode': 'adaptive'},
                    max_pool_connections=50
                )
            )
        
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise
    
    
    def _encode_filename_for_metadata(self, filename: str) -> str:
        """
        Encode filename to ASCII-safe format for S3 metadata.
        
        Args:
            filename: Original filename that may contain non-ASCII characters
            
        Returns:
            str: Base64-encoded filename safe for S3 metadata
        """
        try:
            # Encode to UTF-8 bytes, then base64 encode to ASCII string
            filename_bytes = filename.encode('utf-8')
            encoded_filename = base64.b64encode(filename_bytes).decode('ascii')
            return encoded_filename
        except Exception as e:
            logger.warning(f"Failed to encode filename '{filename}': {e}")
            # Fallback: remove non-ASCII characters
            return ''.join(char for char in filename if ord(char) < 128)
    
    
    def _decode_filename_from_metadata(self, encoded_filename: str) -> str:
        """
        Decode filename from S3 metadata back to original format.
        
        Args:
            encoded_filename: Base64-encoded filename from S3 metadata
            
        Returns:
            str: Original filename with Unicode characters
        """
        try:
            # Decode from base64 to bytes, then decode UTF-8 to string
            filename_bytes = base64.b64decode(encoded_filename.encode('ascii'))
            original_filename = filename_bytes.decode('utf-8')
            return original_filename
        except Exception as e:
            logger.warning(f"Failed to decode filename '{encoded_filename}': {e}")
            # Fallback: return as-is
            return encoded_filename
    
    
    def extract_s3_key_from_url(self, s3_url: str) -> str:
        """
        Extract S3 key from S3 URL.
        
        Args:
            s3_url: S3 URL (e.g., https://bucket.s3.region.amazonaws.com/key)
            
        Returns:
            str: S3 key (path in bucket)
        """
        try:
            # Handle both formats:
            # https://bucket.s3.region.amazonaws.com/key
            # https://s3.region.amazonaws.com/bucket/key
            
            if '.s3.' in s3_url:
                # Extract key from URL
                parts = s3_url.split('/')
                if len(parts) >= 4:
                    # Join everything after the domain
                    return '/'.join(parts[3:])
            
            raise ValueError(f"Invalid S3 URL format: {s3_url}")
            
        except Exception as e:
            logger.error(f"Failed to extract S3 key from URL {s3_url}: {e}")
            raise S3Error(f"Invalid S3 URL format: {s3_url}")
    
    
    async def upload_file(
        self,
        user_id: UUID,
        workspace_id: UUID,
        folder: Optional[str],
        file_id: UUID,
        file_content: bytes,
        filename: str,
        mime_type: str,
        file_size: int,
        tags: Optional[Dict[str, str]] = None,
        is_public: bool = False,
        server_side_encryption: Optional[str] = None,
        storage_class: Optional[str] = None
    ) -> str:
        """
        Upload file to S3 with enhanced options.
        
        Args:
            user_id: User UUID
            workspace_id: Workspace UUID
            folder: Optional folder name
            file_id: File UUID
            file: FastAPI UploadFile object
            tags: Optional tags for the S3 object
            server_side_encryption: Optional encryption type ('AES256' or 'aws:kms')
            storage_class: Optional storage class ('STANDARD', 'INTELLIGENT_TIERING', etc.)
            
        Returns:
            str: S3 file URL
        """
        try:
            file_extension = Path(filename).suffix
            unique_filename = f"{file_id}{file_extension}"
            folder_name = folder or settings.aws.voice_call_folder
            s3_key = f"{self._private_folder_prefix}/{workspace_id}/{user_id}/{folder_name}/{unique_filename}"
            if is_public:
                s3_key = f"{self._public_folder_prefix}/{workspace_id}/{folder_name}/{unique_filename}"
            
            # Prepare upload parameters
            upload_params = {
                'Bucket': self.bucket_name,
                'Key': s3_key,
                'Body': file_content,
                'ContentType': mime_type,
                'Metadata': {
                    'original_filename': self._encode_filename_for_metadata(filename),
                    'uploaded_by': str(user_id),
                    'workspace_id': str(workspace_id),
                    'upload_timestamp': current_utc_str(),
                    'file_size': str(file_size)
                }
            }
            
            # Add optional parameters
            if server_side_encryption:
                upload_params['ServerSideEncryption'] = server_side_encryption
            
            if storage_class:
                upload_params['StorageClass'] = storage_class
            
            if tags:
                # Convert tags to S3 format
                tag_set = [{'Key': k, 'Value': v} for k, v in tags.items()]
                upload_params['Tagging'] = '&'.join([f"{t['Key']}={t['Value']}" for t in tag_set])
            
            # Upload to S3 in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            upload_func = partial(self.s3_client.put_object, **upload_params)
            await loop.run_in_executor(None, upload_func)
            
            # Generate S3 URL
            s3_url = f"https://{self.bucket_name}.s3.{settings.aws.region}.amazonaws.com/{s3_key}"
            
            logger.info(f"File uploaded successfully to S3: {s3_url}")
            
            return s3_url
            
        except (NoCredentialsError, BotoCoreError, ClientError) as e:
            logger.error(f"S3 error uploading file: {e}")
            raise S3UploadError(f"S3 upload failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error uploading to S3: {e}")
            raise S3UploadError(f"Upload failed: {str(e)}")
    
    async def update_file(
        self,
        s3_key: str,
        file_content: bytes,
        mime_type: str,
        file_size: int
    ) -> str:
        """
        Update existing file in S3 with new content while keeping the same S3 key.
        This is useful for updating widget files without changing their URLs.
        
        Args:
            s3_key: S3 object key (must be an existing file)
            file_content: New file content in bytes
            mime_type: MIME type of the file
            file_size: Size of the file in bytes
            
        Returns:
            str: S3 URL of the updated file
            
        Raises:
            S3UploadError: If update fails
        """
        try:
            loop = asyncio.get_event_loop()
            
            def _update():
                return self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=file_content,
                    ContentType=mime_type,
                    Metadata={
                        'updated_timestamp': current_utc_str(),
                        'file_size': str(file_size)
                    }
                )
            
            await loop.run_in_executor(None, _update)
            
            s3_url = f"https://{self.bucket_name}.s3.{settings.aws.region}.amazonaws.com/{s3_key}"
            logger.info(f"Successfully updated file in S3: {s3_url}")
            return s3_url
            
        except (NoCredentialsError, BotoCoreError, ClientError) as e:
            logger.error(f"S3 error updating file: {e}")
            raise S3UploadError(f"S3 update failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error updating file in S3: {e}")
            raise S3UploadError(f"Update failed: {str(e)}")

    async def delete_file(self, s3_key: str) -> bool:
        """
        Delete file from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            bool: True if deletion successful
        """
        try:
            loop = asyncio.get_event_loop()
            
            def _delete():
                return self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
            
            await loop.run_in_executor(None, _delete)
            
            logger.info(f"Successfully deleted file from S3: {s3_key}")
            return True

        except (NoCredentialsError, BotoCoreError, ClientError) as e:
            logger.error(f"S3 error deleting file: {e}")
            raise S3UploadError(f"S3 delete failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error deleting from S3: {e}")
            raise S3UploadError(f"Delete failed: {str(e)}")

    
    async def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            bool: True if file exists
        """
        try:
            loop = asyncio.get_event_loop()
            
            def _head_object():
                return self.s3_client.head_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
            
            await loop.run_in_executor(None, _head_object)
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error(f"Error checking file existence: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking file existence: {e}")
            return False

# Singleton instance
s3_service = S3Service()