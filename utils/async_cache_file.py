import aiohttp
import aiofiles
import asyncio
import hashlib
import os
import shutil
import time
import logging

from typing import Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AsyncStaticExcelCache:
    def __init__(
        self,
        cache_dir: str ="./temps/excel_cache",
        ttl: int =86400,  # 1 day
    ):
        self.cache_dir = cache_dir
        self.ttl = ttl
        os.makedirs(cache_dir, exist_ok=True)
        self._download_locks = {}  # Prevent concurrent downloads of same file
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def get_template(self, template_url: str, template_name: Optional[str] = None, session: Optional[aiohttp.ClientSession] = None):
        """
        Async cache static template
        """
        # Generate cache file path
        if template_name:
            cache_file = os.path.join(self.cache_dir, f"{template_name}.xlsx")
        else:
            url_hash = hashlib.md5(template_url.encode()).hexdigest()
            cache_file = os.path.join(self.cache_dir, f"template_{url_hash}.xlsx")
            
        # Check cache validity
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < self.ttl:
                print(f"Cache HIT: {cache_file}")
                return cache_file
        
        # Prevent concurrent downloads of same file
        if cache_file in self._download_locks:
            await self._download_locks[cache_file].wait()
            return cache_file
        
        # Create download lock
        download_event = asyncio.Event()
        self._download_locks[cache_file] = download_event
        
        try:
            await self._download_template(template_url, cache_file, session)
            download_event.set()  # Signal download complete
            return cache_file
        finally:
            # Cleanup lock
            if cache_file in self._download_locks:
                del self._download_locks[cache_file]

    async def _download_template(self, template_url: str, cache_file: str, session: Optional[aiohttp.ClientSession] = None):
        """Download template file async"""
        logger.info(f"Downloading template (first time): {template_url}")
        
        # Use provided session or create new one
        if session is None:
            async with aiohttp.ClientSession() as session:
                await self._do_download(session, template_url, cache_file)
        else:
            await self._do_download(session, template_url, cache_file)

    async def _do_download(self, session: aiohttp.ClientSession, template_url: str, cache_file: str):
        """Actual download logic"""
        async with session.get(template_url) as response:
            response.raise_for_status()
            
            # Write to temp file first, then move (atomic operation)
            temp_file = f"{cache_file}.tmp"
            async with aiofiles.open(temp_file, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    await f.write(chunk)
            
            # Atomic move
            shutil.move(temp_file, cache_file)
            logger.info(f"Template cached at: {cache_file}")

    async def _do_upload(self, file_content: bytes, filename: str, content_type: str, session: Optional[aiohttp.ClientSession] = None) -> Any:
        """Upload file async"""
        if session is None:
            async with aiohttp.ClientSession() as session:
                return await self._upload_to_cdn(session, file_content, filename, content_type)
        else:
            return await self._upload_to_cdn(session, file_content, filename, content_type)

    async def _upload_to_cdn(self, session: aiohttp.ClientSession, file_content: bytes, filename: str, content_type: str) -> tuple[str, str, str]:
        """Upload file to CDN"""
        try:
            cdn_base_url = os.getenv("CDN_BASE_URL")
            cdn_folder = os.getenv("CDN_ATTACHMENT_FOLDER")
            upload_url = f"{cdn_base_url}{os.getenv("CDN_UPLOAD_ENDPOINT")}?folderName={cdn_folder}"
            cdn_api_key = os.getenv("CDN_API_KEY")
            
            headers = {
                    "accept": "*/*",
                    "cdn-key": cdn_api_key
                }
            data = aiohttp.FormData()
            data.add_field(
                'inputFile',
                file_content,
                filename=filename,
                content_type=content_type)
            data.add_field('cdn-key', cdn_api_key)

            async with session.post(upload_url, data=data, headers=headers) as response:
                response.raise_for_status()
                upload_result = await response.json()
                
                # Extract URL from nested response structure
                response_data = upload_result.get('responseData', {})
                cdn_url = response_data.get('dataUrlFull')
                
                return cdn_url, filename, content_type
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise
            
    async def start_cleanup_task(self, cleanup_interval: int = 3600):  # 1 hour
        """Start background cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            return
            
        self._running = True
        self._cleanup_task = asyncio.create_task(self._background_cleanup(cleanup_interval))
        logger.info("Background cleanup task started")
    
    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Background cleanup task stopped")
    
    async def _background_cleanup(self, interval: int):
        """Background cleanup coroutine"""
        while self._running:
            try:
                await asyncio.sleep(interval)
                if self._running:  # Check again after sleep
                    self._clear_expired_cache()
                    logger.info("Background cache cleanup completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")
    
    def _clear_expired_cache(self, max_age: Optional[int] = None):
        """Clean up old cache files"""
        max_age = max_age or self.ttl
        current_time = time.time()
        cleaned_count = 0
        
        try:
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > max_age:
                        os.remove(filepath)
                        cleaned_count += 1
                        logger.debug(f"Removed expired cache: {filename}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired cache files")
                
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
