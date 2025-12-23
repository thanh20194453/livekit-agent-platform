"""
Optimized Speech (TTS) Service Module
- Async HTTP with aiohttp and connection pooling
- Streaming audio chunks for low latency
- Environment variable configuration
- Error handling with retry logic
- Proper resource cleanup
"""
import asyncio
import base64
import io
import json
import os
import time
from typing import Optional, AsyncIterator
from logging import getLogger, ERROR
from uuid import uuid4

import aiohttp
import numpy as np
import requests
import soundfile as sf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from livekit import rtc
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from dotenv import load_dotenv

from llm.metrics import TTSMetrics, log_tts_metrics

load_dotenv()
logger = getLogger(__name__)
logger.setLevel(ERROR)

def wav_to_audioframe(sr: int, wav: np.ndarray) -> rtc.AudioFrame:
    """Convert numpy audio array to LiveKit AudioFrame."""
    # wav float32 → int16
    pcm16 = np.clip(wav, -1.0, 1.0)
    pcm16 = (pcm16 * 32767).astype(np.int16)

    if pcm16.ndim == 1:  # mono
        num_channels = 1
        samples_per_channel = len(pcm16)
        raw_bytes = pcm16.tobytes()
    else:  # stereo → interleaved
        num_channels = pcm16.shape[1]
        samples_per_channel = pcm16.shape[0]
        raw_bytes = pcm16.flatten().tobytes()

    return rtc.AudioFrame(
        data=raw_bytes,
        sample_rate=sr,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel
    )


class SpeechService:
    """
    Optimized TTS Service with async HTTP and connection pooling.
    """
    _instance: Optional["SpeechService"] = None
    _lock = asyncio.Lock()
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "F5TTS",
        sample_rate: int = 24000,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
        }
        
        logger.info(f"SpeechService initialized with base_url: {base_url}, reference_audio: {self.reference_audio}")

    @classmethod
    async def get_instance(
        cls,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
    ) -> "SpeechService":
        """Get or create singleton instance with optional reference audio config."""
        async with cls._lock:
            # If reference audio/text provided, check if we need to create new instance
            if reference_audio or reference_text:
                if cls._instance is not None:
                    # Check if reference config changed
                    ref_audio_changed = reference_audio and cls._instance.reference_audio != reference_audio
                    ref_text_changed = reference_text and cls._instance.reference_text != reference_text
                    
                    if ref_audio_changed or ref_text_changed:
                        # Close old instance and create new one with updated config
                        await cls._instance.close()
                        cls._instance = None
            
            if cls._instance is None:
                base_url = os.getenv("TTS_BASE_URL", "http://localhost:8000")
                api_key = os.getenv("TTS_API_KEY", "secret_api_key")
                sample_rate = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
                
                cls._instance = cls(
                    base_url, 
                    api_key, 
                    sample_rate=sample_rate,
                    reference_audio=reference_audio,
                    reference_text=reference_text,
                )
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance."""
        cls._instance = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling."""
        # Get current event loop
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        
        async with self._session_lock:
            # Check if session exists and is valid for current loop
            needs_new_session = (
                self._session is None or 
                self._session.closed or
                (current_loop and hasattr(self, '_session_loop') and self._session_loop != current_loop)
            )
            
            if needs_new_session:
                # Close old session if exists
                if self._session and not self._session.closed:
                    try:
                        await self._session.close()
                    except Exception:
                        pass
                
                connector = aiohttp.TCPConnector(
                    limit=10,  # Max connections
                    limit_per_host=5,  # Max connections per host
                    keepalive_timeout=30,  # Keep connections alive
                    enable_cleanup_closed=True,
                )
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "Accept": "*/*",
                    }
                )
                self._session_loop = current_loop
            return self._session  # type: ignore[return-value]

    async def check_health(self) -> bool:
        """
        Check if TTS service is reachable and responding.
        
        Returns:
            bool: True if service is healthy, False otherwise
        """
        try:
            session = await self._get_session()
            health_url = f"{self.base_url}/health"
            
            timeout = aiohttp.ClientTimeout(total=5)
            async with session.get(health_url, timeout=timeout) as resp:
                if resp.status == 200:
                    logger.info(f"TTS service health check passed: {health_url}")
                    return True
                else:
                    logger.warning(f"TTS service health check failed: {resp.status}")
                    return False
        except Exception as e:
            logger.warning(f"TTS service health check error: {e}")
            return False

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
          
  
    def inference(self, text: str, speed: float = 1.2, first_llm_chunk: bool = False):
        """
        Stream audio từ API (đồng bộ), yield từng chunk (sr, np.ndarray)
        """

        url = f"{self.base_url}/run_inference"
        payload = {
            "server_addr": "localhost",
            "server_port": 8001,
            "model_name": "f5_tts",
            "log_dir": "logs",
            "num_tasks": 2,
            "log_interval": 1,
            "reference_audio": self.reference_audio,
            "reference_text": self.reference_text,
            "target_text": text,
            "huggingface_dataset": "string",
            "split_name": "string",
            "manifest_path": "string"
        }

        resp = requests.post(url, headers=self.headers, json=payload, timeout=300)
        
        if resp.status_code != 200:
            raise RuntimeError(f"TTS API error: {resp.status_code} - {resp.text}")
        
        try:
            res = resp.json()
        except requests.exceptions.JSONDecodeError:
            raise RuntimeError(f"TTS API returned invalid JSON response: {resp.text[:500]}")
        
        if res is None:
            raise RuntimeError("TTS API returned null response")
            
        audio_samples = res.get("audio", [])
        float32_samples = np.frombuffer(base64.b64decode(audio_samples[0]), dtype=np.float32)
        pcm_int16 = (float32_samples * 32767).clip(-32768, 32767).astype(np.int16)
        
        return pcm_int16.tobytes()

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True
    )
    async def inference_async(self, text: str, speed: float = 1.0) -> bytes:
        """
        Async TTS inference with retry logic.
        
        Args:
            text: Text to synthesize
            speed: Speech speed multiplier
            
        Returns:
            bytes: Raw PCM audio data (int16)
        """
        # Validate input text
        if not text or not text.strip():
            logger.warning("TTS called with empty text, returning empty audio")
            return b""
        
        # Validate reference audio if provided
        if self.reference_audio:
            # Check if file path exists (basic check - file may not exist on TTS server)
            logger.debug(f"Reference audio path provided: {self.reference_audio}")
            # Note: File check would need to be done on TTS server side
            # We'll rely on TTS backend error handling
        else:
            logger.debug("No reference audio provided - using TTS default voice")
        
        start_time = time.time()
        session = await self._get_session()
        
        url = f"{self.base_url}/run_inference"
        payload = {
            "server_addr": "localhost",
            "server_port": 8001,
            "model_name": "f5_tts",
            "log_dir": "logs",
            "num_tasks": 2,
            "log_interval": 1,
            "reference_audio": self.reference_audio,
            "reference_text": self.reference_text,
            "target_text": text,
            "huggingface_dataset": "string",
            "split_name": "string",
            "manifest_path": "string"
        }
        
        # Log request details for debugging
        logger.debug(f"TTS API request to {url} with text: '{text[:50]}...'")
        logger.debug(f"Reference audio: {self.reference_audio}, Reference text: {self.reference_text}")
        
        # Create timeout per-request to avoid event loop issues
        timeout = aiohttp.ClientTimeout(total=300, connect=10)

        try:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"TTS API error: {resp.status} - {error_text[:200]}")
                    raise RuntimeError(f"TTS API error: {resp.status}")
                
                # Get response text first for better debugging
                response_text = await resp.text()
                
                # Log response for debugging
                logger.debug(f"TTS API response (first 500 chars): {response_text[:500]}")
                
                # Try to parse JSON
                try:
                    if not response_text or response_text.strip() in ('', 'null', 'None'):
                        logger.error(f"TTS API returned empty or null response body: '{response_text}'")
                        logger.error(f"⚠️ TTS backend at {self.base_url} is returning null - check service health!")
                        if self.reference_audio:
                            logger.error(f"Reference audio file may not exist on TTS server: {self.reference_audio}")
                        logger.warning("Returning empty audio to prevent agent crash")
                        return b""
                    
                    data = json.loads(response_text)
                except json.JSONDecodeError as je:
                    logger.error(f"TTS API returned invalid JSON: {response_text[:500]}")
                    logger.warning(f"Returning empty audio due to invalid JSON: {str(je)}")
                    return b""
                
                if data is None or not isinstance(data, dict):
                    logger.error(f"TTS API returned null or non-dict response: {type(data)} - {data}")
                    logger.warning("Returning empty audio to prevent agent crash")
                    return b""
                
                audio_samples = data.get("audio", [])
                
                if not audio_samples:
                    logger.error(f"TTS API returned empty audio array. Response keys: {list(data.keys())}, Text: '{text[:100]}'")
                    logger.error(f"Full response data: {data}")
                    return b""
                
                # Decode and concatenate audio chunks
                pcm_list = []
                detected_sr = None
                
                for b64_audio in audio_samples:
                    if not b64_audio:
                        continue
                    
                    audio_bytes = base64.b64decode(b64_audio)
                    
                    # Try to read as WAV first (read as float32 for proper processing)
                    try:
                        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                        detected_sr = sr
                        
                        # Resample if needed
                        if sr != self.sample_rate:
                            # Simple resampling using numpy interpolation
                            duration = len(wav) / sr
                            new_length = int(duration * self.sample_rate)
                            wav = np.interp(
                                np.linspace(0, len(wav), new_length),
                                np.arange(len(wav)),
                                wav
                            )
                            logger.debug(f"Resampled audio from {sr}Hz to {self.sample_rate}Hz")
                        
                        # Convert float32 to int16 (multiply first, then clip)
                        pcm_int16 = (wav * 32767).clip(-32768, 32767).astype(np.int16)
                        pcm_list.append(pcm_int16)
                        
                    except Exception as wav_err:
                        # Not WAV format - TTS API returns raw float32 PCM at 24kHz
                        logger.debug(f"Not WAV format, treating as raw float32 PCM: {wav_err}")
                        
                        # Parse as float32 (TTS API returns normalized float32 audio)
                        float32_samples = np.frombuffer(audio_bytes, dtype=np.float32)
                        
                        # Convert float32 to int16 (multiply first, then clip)
                        pcm_int16 = (float32_samples * 32767).clip(-32768, 32767).astype(np.int16)
                        pcm_list.append(pcm_int16)
                        
                        logger.debug(f"Decoded {len(float32_samples)} float32 samples -> {len(pcm_int16)} int16 samples")
                
                if not pcm_list:
                    logger.warning("TTS: no valid audio chunks decoded")
                    return b""
                
                if detected_sr:
                    logger.debug(f"TTS audio sample rate: {detected_sr}Hz -> target: {self.sample_rate}Hz")
                
                # Merge all chunks
                merged_pcm = np.concatenate(pcm_list, axis=0)
                merged_audio_bytes = merged_pcm.astype(np.int16).tobytes()
                
                inference_time = time.time() - start_time
                audio_duration = len(merged_pcm) / self.sample_rate
                rtf = inference_time / audio_duration if audio_duration > 0 else 0
                
                # Log metrics
                metrics = TTSMetrics(
                    request_id=str(uuid4())[:8],
                    text_length=len(text),
                    audio_duration_s=audio_duration,
                    processing_time_ms=inference_time * 1000,
                    sample_rate=self.sample_rate,
                    audio_bytes=len(merged_audio_bytes),
                    rtf=rtf,
                    success=True
                )
                log_tts_metrics(metrics)
                
                return merged_audio_bytes
                
        except aiohttp.ClientError as e:
            inference_time = time.time() - start_time
            metrics = TTSMetrics(
                request_id=str(uuid4())[:8],
                text_length=len(text),
                audio_duration_s=0,
                processing_time_ms=inference_time * 1000,
                sample_rate=self.sample_rate,
                audio_bytes=0,
                rtf=0,
                success=False,
                error=str(e)
            )
            log_tts_metrics(metrics)
            logger.error(f"TTS connection error: {e}")
            logger.warning("Returning empty audio to keep agent alive")
            return b""
        except Exception as e:
            inference_time = time.time() - start_time
            metrics = TTSMetrics(
                request_id=str(uuid4())[:8],
                text_length=len(text),
                audio_duration_s=0,
                processing_time_ms=inference_time * 1000,
                sample_rate=self.sample_rate,
                audio_bytes=0,
                rtf=0,
                success=False,
                error=str(e)
            )
            log_tts_metrics(metrics)
            logger.error(f"TTS inference error: {e}")
            logger.warning("Returning empty audio to keep agent alive")
            return b""

    async def inference_streaming(self, text: str, speed: float = 1.0) -> AsyncIterator[bytes]:
        """
        Stream TTS audio chunks as they become available.
        
        Args:
            text: Text to synthesize
            speed: Speech speed multiplier
            
        Yields:
            bytes: Audio chunks as they arrive
        """
        start_time = time.time()
        session = await self._get_session()
        
        url = f"{self.base_url}/run_inference"
        payload = {
            "server_addr": "localhost",
            "server_port": 8001,
            "model_name": "f5_tts",
            "log_dir": "logs",
            "num_tasks": 2,
            "log_interval": 1,
            "reference_audio": self.reference_audio,
            "reference_text": self.reference_text,
            "target_text": text,
            "huggingface_dataset": "string",
            "split_name": "string",
            "manifest_path": "string"
        }
        
        # Create timeout per-request to avoid event loop issues
        timeout = aiohttp.ClientTimeout(total=300, connect=10)

        try:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"TTS API error: {resp.status} - {error_text[:200]}")
                
                data = await resp.json()
                audio_samples = data.get("audio", [])
                
                first_chunk = True
                for b64_audio in audio_samples:
                    if not b64_audio:
                        continue
                    
                    audio_bytes = base64.b64decode(b64_audio)
                    
                    # Try to read as WAV first (read as float32 for proper processing)
                    try:
                        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                        
                        # Resample if needed
                        if sr != self.sample_rate:
                            duration = len(wav) / sr
                            new_length = int(duration * self.sample_rate)
                            wav = np.interp(
                                np.linspace(0, len(wav), new_length),
                                np.arange(len(wav)),
                                wav
                            )
                        
                        # Convert float32 to int16 (multiply first, then clip)
                        pcm_int16 = (wav * 32767).clip(-32768, 32767).astype(np.int16)
                        pcm_bytes = pcm_int16.tobytes()
                    except Exception:
                        # Not WAV format - TTS API returns raw float32 PCM at 24kHz
                        float32_samples = np.frombuffer(audio_bytes, dtype=np.float32)
                        pcm_int16 = (float32_samples * 32767).clip(-32768, 32767).astype(np.int16)
                        pcm_bytes = pcm_int16.tobytes()
                    
                    if first_chunk:
                        first_chunk_time = time.time() - start_time
                        logger.debug(f"TTS first chunk latency: {first_chunk_time:.3f}s")
                        first_chunk = False
                    
                    yield pcm_bytes
                    
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
            raise


class MySpeechClass(tts.TTS):
    """LiveKit-compatible TTS wrapper with optimized async performance."""
    
    _shared_service: Optional[SpeechService] = None
    _service_lock = asyncio.Lock()
    
    def __init__(
        self,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
    ):
        """
        Initialize MySpeechClass with optional reference audio for voice cloning.
        
        Args:
            reference_audio: Path to reference audio file for voice cloning
            reference_text: Transcript of the reference audio
        """
        sample_rate = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._reference_audio = reference_audio
        self._reference_text = reference_text
    
    @classmethod
    async def _get_service(
        cls,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
    ) -> SpeechService:
        """Get shared SpeechService instance with optional reference audio config."""
        async with cls._service_lock:
            # Always get instance with the provided reference config
            cls._shared_service = await SpeechService.get_instance(
                reference_audio=reference_audio,
                reference_text=reference_text,
            )
            return cls._shared_service
    
    @classmethod
    async def close_service(cls):
        """Close shared service instance."""
        async with cls._service_lock:
            if cls._shared_service:
                await cls._shared_service.close()
                cls._shared_service = None
                logger.info("MySpeechClass shared service closed")

    def synthesize(self, text: str, *, conn_options=DEFAULT_API_CONNECT_OPTIONS) -> tts.ChunkedStream:
        """
        Synthesize text to speech.
        
        Args:
            text: The text to synthesize
            conn_options: Connection options
            
        Returns:
            ChunkedStream: Stream of synthesized audio chunks
        """
        logger.debug(f"TTS synthesize request: {text[:50]}...")
        return MySpeechChunkStream(
            tts=self, 
            input_text=text, 
            conn_options=conn_options,
            reference_audio=self._reference_audio,
            reference_text=self._reference_text,
        )


class MySpeechChunkStream(tts.ChunkedStream):
    """Optimized TTS chunked stream with async inference."""
    
    def __init__(
        self, 
        *, 
        tts: MySpeechClass, 
        input_text: str, 
        conn_options,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: MySpeechClass = tts
        self._audio_generated = False
        self._reference_audio = reference_audio
        self._reference_text = reference_text

    async def _run(self, output_emitter: tts.AudioEmitter):
        """
        Generate audio using async TTS and emit through output emitter.
        """
        if self._audio_generated:
            return

        initialized = False
        try:
            start_time = time.time()
            padded_text = self.input_text.strip()
            
            if not padded_text:
                logger.warning("TTS received empty text")
                return
            
            logger.debug(f"TTS generating audio for: {padded_text[:50]}...")
            
            # Get shared service with reference audio config
            service = await MySpeechClass._get_service(
                reference_audio=self._reference_audio,
                reference_text=self._reference_text,
            )
            
            # Always initialize emitter before any operations
            output_emitter.initialize(
                request_id=str(id(self)),
                sample_rate=self._tts.sample_rate,
                num_channels=1,
                mime_type="audio/pcm",
            )
            initialized = True
            
            # CRITICAL: Use async method to avoid blocking event loop
            # This prevents STT from being blocked during TTS generation
            audio_bytes = await service.inference_async(padded_text)
            
            if audio_bytes:
                # Push audio data
                output_emitter.push(audio_bytes)
                output_emitter.flush()
                self._audio_generated = True
            else:
                logger.warning(f"TTS returned empty audio for text: '{padded_text[:100]}...'")
                logger.warning("Agent will continue without audio for this response")
            
            total_time = time.time() - start_time
            logger.info(f"TTS total time: {total_time:.2f}s for text: {padded_text[:30]}...")
            
        except asyncio.CancelledError:
            logger.debug("TTS stream cancelled")
            raise
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}", exc_info=True)
            if initialized:
                try:
                    output_emitter.end_input()
                except Exception:
                    pass
            raise RuntimeError(f"TTS synthesis failed: {str(e)}")
        finally:
            try:
                if initialized:
                    output_emitter.end_input()
            except Exception:
                pass
