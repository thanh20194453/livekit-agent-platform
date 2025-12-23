# """
# Optimized Speech-to-Text (STT) Transcription Module
# - Thread pool executor for non-blocking transcription
# - GPU optimization with configurable settings
# - Direct PCM processing (no WAV conversion overhead)
# - Environment variable configuration
# - Batch processing support
# - Comprehensive error handling
# """
# import asyncio
# import os
# import time
# from concurrent.futures import ThreadPoolExecutor
# from io import BytesIO
# from typing import Optional
# from logging import getLogger, ERROR
# from uuid import uuid4

# import numpy as np
# import torch
# from faster_whisper import WhisperModel, BatchedInferencePipeline
# from pydub import AudioSegment

# from livekit.agents import APIConnectOptions, stt, utils
# from livekit.agents.utils import AudioBuffer
# from livekit.agents.types import NOT_GIVEN, NotGivenOr
# from dotenv import load_dotenv

# from llm.metrics import STTMetrics, log_stt_metrics

# load_dotenv()
# logger = getLogger(__name__)
# logger.setLevel(ERROR)

# # ============================================================================
# # GPU/Device Configuration
# # ============================================================================

# def setup_device():
#     """Configure device based on environment and hardware availability."""
#     forced_device = os.getenv("WHISPER_DEVICE")
#     has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
#     default_device = "cuda" if has_cuda else "cpu"
#     device = forced_device or default_device

#     if device == "cuda" and not has_cuda:
#         raise RuntimeError(
#             "WHISPER_DEVICE is set to 'cuda' but no CUDA device is available. "
#             "Check your PyTorch installation and GPU drivers."
#         )

#     if device == "cuda":
#         gpu_name = torch.cuda.get_device_name(0)
#         gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
#         logger.info(f"STT Device: {device} ({gpu_name}, {gpu_memory:.1f}GB)")
#     else:
#         logger.warning(
#             "STT Device: cpu (CUDA not available). "
#             "Set WHISPER_DEVICE=cuda if GPU is present."
#         )

#     return device


# def get_compute_type(device: str) -> str:
#     """Get optimal compute type for device."""
#     if device == "cuda":
#         # Check GPU capability for optimal compute type
#         if torch.cuda.is_available():
#             capability = torch.cuda.get_device_capability(0)
#             # Use float16 for newer GPUs (compute capability >= 7.0)
#             if capability[0] >= 7:
#                 return "float16"
#             # Use int8_float16 for older GPUs
#             return "int8_float16"
#     return "int8"


# # Initialize device
# DEVICE = setup_device()
# COMPUTE_TYPE = get_compute_type(DEVICE)

# # ============================================================================
# # Whisper Model Configuration
# # ============================================================================

# MODEL_NAME = os.getenv("WHISPER_MODEL", "erax-ai/EraX-WoW-Turbo-V1.1-CT2")
# MODEL_CACHE_DIR = os.getenv("WHISPER_CACHE_DIR", "./models")
# NUM_WORKERS = int(os.getenv("WHISPER_NUM_WORKERS", "4"))
# BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))  # 1 for real-time, 5 for quality
# BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "16"))


# def load_whisper_model() -> WhisperModel:
#     """Load Whisper model with optimized settings."""
#     logger.info(f"Loading Whisper model: {MODEL_NAME}")
#     logger.info(f"  Device: {DEVICE}, Compute: {COMPUTE_TYPE}")
#     logger.info(f"  Workers: {NUM_WORKERS}, Beam: {BEAM_SIZE}")
    
#     start_time = time.time()
    
#     model = WhisperModel(
#         MODEL_NAME,
#         device=DEVICE,
#         compute_type=COMPUTE_TYPE,
#         device_index=0 if DEVICE == "cuda" else 0,  # Default to 0
#         num_workers=NUM_WORKERS,
#         download_root=MODEL_CACHE_DIR,
#     )
    
#     load_time = time.time() - start_time
#     logger.info(f"Whisper model loaded in {load_time:.2f}s")
    
#     return model


# # Global model instance (loaded once at module import)
# whisper_model: Optional[WhisperModel] = None
# batched_model: Optional[BatchedInferencePipeline] = None


# def get_whisper_model() -> WhisperModel:
#     """Get or create whisper model instance."""
#     global whisper_model
#     if whisper_model is None:
#         whisper_model = load_whisper_model()
#     return whisper_model


# def get_batched_model() -> BatchedInferencePipeline:
#     """Get or create batched inference pipeline."""
#     global batched_model
#     if batched_model is None:
#         batched_model = BatchedInferencePipeline(model=get_whisper_model())
#     return batched_model


# # ============================================================================
# # Transcriber Service
# # ============================================================================

# class RealtimeTranscriber:
#     """
#     High-performance real-time transcriber using faster-whisper.
#     """
    
#     def __init__(self, batch_size: int = BATCH_SIZE):
#         self.model = get_whisper_model()
#         self.batched_model = get_batched_model()
#         self.batch_size = batch_size
#         self._executor = ThreadPoolExecutor(
#             max_workers=2,
#             thread_name_prefix="whisper"
#         )

#     def transcribe_sync(
#         self,
#         audio: np.ndarray | BytesIO,
#         language: Optional[str] = None,
#         sr: int = 16000
#     ) -> tuple[str, float, float]:
#         """
#         Synchronous transcription (runs in thread pool).
        
#         Args:
#             audio: Audio data as numpy array or BytesIO
#             language: Language code (auto-detect if None)
#             sr: Sample rate
            
#         Returns:
#             tuple: (text, processing_time, audio_duration)
#         """
#         start_time = time.time()

#         segments, info = self.model.transcribe(
#             audio,
#             beam_size=BEAM_SIZE,
#             vad_filter=True,
#             vad_parameters=dict(
#                 min_silence_duration_ms=500,
#                 speech_pad_ms=400,
#             ),
#             language=language,
#         )

#         # Collect text efficiently
#         text_parts = []
#         for segment in segments:
#             text_parts.append(segment.text.strip())
        
#         text = " ".join(text_parts)
#         processing_time = time.time() - start_time
        
#         rtf = processing_time / info.duration if info.duration > 0 else 0
        
#         # Log metrics
#         metrics = STTMetrics(
#             request_id=str(uuid4())[:8],
#             audio_duration_s=info.duration,
#             processing_time_ms=processing_time * 1000,
#             text_length=len(text),
#             language=info.language or "unknown",
#             device=DEVICE,
#             rtf=rtf,
#             success=True
#         )
#         log_stt_metrics(metrics)

#         return text, processing_time, info.duration

#     async def transcribe_async(
#         self,
#         audio: np.ndarray | BytesIO,
#         language: Optional[str] = None,
#         sr: int = 16000
#     ) -> tuple[str, float, float]:
#         """
#         Async transcription using thread pool executor.
        
#         Args:
#             audio: Audio data as numpy array or BytesIO
#             language: Language code (auto-detect if None)
#             sr: Sample rate
            
#         Returns:
#             tuple: (text, processing_time, audio_duration)
#         """
#         loop = asyncio.get_event_loop()
#         return await loop.run_in_executor(
#             self._executor,
#             self.transcribe_sync,
#             audio,
#             language,
#             sr
#         )

#     def close(self):
#         """Shutdown thread pool executor."""
#         self._executor.shutdown(wait=True)


# # Global transcriber service
# transcribe_service: Optional[RealtimeTranscriber] = None


# def get_transcribe_service() -> RealtimeTranscriber:
#     """Get or create transcriber service."""
#     global transcribe_service
#     if transcribe_service is None:
#         transcribe_service = RealtimeTranscriber(batch_size=BATCH_SIZE)
#     return transcribe_service


# # ============================================================================
# # LiveKit STT Implementation
# # ============================================================================

# class MyTranscribeClass(stt.STT):
#     """
#     LiveKit-compatible STT with thread pool for non-blocking transcription.
#     """
    
#     def __init__(self):
#         super().__init__(
#             capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
#         )
#         self._transcriber = get_transcribe_service()

#     async def _recognize_impl(
#         self,
#         buffer: AudioBuffer,
#         *,
#         language: NotGivenOr[str] = NOT_GIVEN,
#         conn_options: APIConnectOptions,
#     ) -> stt.SpeechEvent:
#         """
#         Recognize speech from audio buffer.
        
#         Args:
#             buffer: Audio buffer from LiveKit
#             language: Language hint (auto-detect if NOT_GIVEN)
#             conn_options: Connection options
            
#         Returns:
#             SpeechEvent with transcription result
#         """
#         start_time = time.time()
        
#         # Convert language to optional string
#         lang_str: Optional[str] = None
#         if language is not NOT_GIVEN and isinstance(language, str):
#             lang_str = language
        
#         try:
#             # Merge audio frames
#             merged_buffer = utils.merge_frames(buffer)
#             audio_data = bytes(merged_buffer.data)
            
#             # Convert to format expected by Whisper
#             # Optimize: Direct PCM to numpy conversion when possible
#             audio = AudioSegment.from_raw(
#                 BytesIO(audio_data),
#                 sample_width=2,
#                 frame_rate=merged_buffer.sample_rate,
#                 channels=merged_buffer.num_channels
#             )
            
#             # Convert to mono 16kHz
#             if audio.channels > 1:
#                 audio = audio.set_channels(1)
#             if audio.frame_rate != 16000:
#                 audio = audio.set_frame_rate(16000)
            
#             # Export to WAV in memory
#             wav_buffer = BytesIO()
#             audio.export(wav_buffer, format="wav")
#             wav_buffer.seek(0)
            
#             # Run transcription in thread pool (non-blocking)
#             text, proc_time, audio_duration = await self._transcriber.transcribe_async(
#                 wav_buffer,
#                 language=lang_str
#             )
            
#             # Get detected language from last transcription
#             # Note: faster-whisper returns this in the info object
#             detected_lang = lang_str or "vi"  # Default to Vietnamese
            
#             total_time = time.time() - start_time
#             logger.info(
#                 f"STT complete: '{text[:50]}...' in {total_time:.3f}s "
#                 f"(audio: {audio_duration:.2f}s)"
#             )
            
#             return stt.SpeechEvent(
#                 type=stt.SpeechEventType.FINAL_TRANSCRIPT,
#                 alternatives=[
#                     stt.SpeechData(
#                         text=text or "",
#                         language=lang_str or ""
#                     )
#                 ],
#             )
            
#         except asyncio.CancelledError:
#             logger.debug("STT cancelled")
#             raise
#         except Exception as e:
#             total_time = time.time() - start_time
#             # Log error metrics
#             metrics = STTMetrics(
#                 request_id=str(uuid4())[:8],
#                 audio_duration_s=0,
#                 processing_time_ms=total_time * 1000,
#                 text_length=0,
#                 language=lang_str or "unknown",
#                 device=DEVICE,
#                 rtf=0,
#                 success=False,
#                 error=str(e)
#             )
#             log_stt_metrics(metrics)
#             logger.error(f"STT error: {e}", exc_info=True)
#             # Return empty result on error instead of crashing
#             return stt.SpeechEvent(
#                 type=stt.SpeechEventType.FINAL_TRANSCRIPT,
#                 alternatives=[
#                     stt.SpeechData(text="", language=language or "")
#                 ],
#             )


# # ============================================================================
# # Cleanup
# # ============================================================================

# def cleanup():
#     """Cleanup resources on shutdown."""
#     global transcribe_service
#     if transcribe_service:
#         transcribe_service.close()
#         transcribe_service = None
#     logger.info("STT resources cleaned up")

