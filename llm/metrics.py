"""
Performance Metrics Logger for Voice Service
- Centralized metrics collection
- Structured JSON logging for monitoring
- Statistics aggregation
"""
import time
import json
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import deque
from logging import getLogger, ERROR
from datetime import datetime

logger = getLogger("voice-metrics")
logger.setLevel(ERROR)


@dataclass
class LLMMetrics:
    """Metrics for LLM inference"""
    request_id: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    first_token_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    success: bool = True
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class STTMetrics:
    """Metrics for Speech-to-Text"""
    request_id: str
    audio_duration_s: float = 0.0
    processing_time_ms: float = 0.0
    rtf: float = 0.0  # Real-time factor (processing_time / audio_duration)
    text_length: int = 0
    language: str = ""
    device: str = ""
    success: bool = True
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class TTSMetrics:
    """Metrics for Text-to-Speech"""
    request_id: str
    text_length: int = 0
    audio_duration_s: float = 0.0
    processing_time_ms: float = 0.0
    rtf: float = 0.0  # Real-time factor
    sample_rate: int = 0
    audio_bytes: int = 0
    success: bool = True
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class MetricsAggregator:
    """Aggregates metrics and computes statistics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._llm_metrics: deque = deque(maxlen=window_size)
        self._stt_metrics: deque = deque(maxlen=window_size)
        self._tts_metrics: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
    
    def add_llm_metrics(self, metrics: LLMMetrics):
        with self._lock:
            self._llm_metrics.append(metrics)
    
    def add_stt_metrics(self, metrics: STTMetrics):
        with self._lock:
            self._stt_metrics.append(metrics)
    
    def add_tts_metrics(self, metrics: TTSMetrics):
        with self._lock:
            self._tts_metrics.append(metrics)
    
    def get_llm_stats(self) -> Dict[str, Any]:
        with self._lock:
            if not self._llm_metrics:
                return {}
            
            successful = [m for m in self._llm_metrics if m.success]
            if not successful:
                return {"error_rate": 1.0}
            
            latencies = [m.total_latency_ms for m in successful]
            first_tokens = [m.first_token_latency_ms for m in successful if m.first_token_latency_ms > 0]
            tps = [m.tokens_per_second for m in successful if m.tokens_per_second > 0]
            
            return {
                "count": len(self._llm_metrics),
                "success_rate": len(successful) / len(self._llm_metrics),
                "avg_latency_ms": sum(latencies) / len(latencies),
                "avg_first_token_ms": sum(first_tokens) / len(first_tokens) if first_tokens else 0,
                "avg_tokens_per_second": sum(tps) / len(tps) if tps else 0,
                "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
            }
    
    def get_stt_stats(self) -> Dict[str, Any]:
        with self._lock:
            if not self._stt_metrics:
                return {}
            
            successful = [m for m in self._stt_metrics if m.success]
            if not successful:
                return {"error_rate": 1.0}
            
            latencies = [m.processing_time_ms for m in successful]
            rtfs = [m.rtf for m in successful if m.rtf > 0]
            
            return {
                "count": len(self._stt_metrics),
                "success_rate": len(successful) / len(self._stt_metrics),
                "avg_latency_ms": sum(latencies) / len(latencies),
                "avg_rtf": sum(rtfs) / len(rtfs) if rtfs else 0,
                "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
            }
    
    def get_tts_stats(self) -> Dict[str, Any]:
        with self._lock:
            if not self._tts_metrics:
                return {}
            
            successful = [m for m in self._tts_metrics if m.success]
            if not successful:
                return {"error_rate": 1.0}
            
            latencies = [m.processing_time_ms for m in successful]
            rtfs = [m.rtf for m in successful if m.rtf > 0]
            
            return {
                "count": len(self._tts_metrics),
                "success_rate": len(successful) / len(self._tts_metrics),
                "avg_latency_ms": sum(latencies) / len(latencies),
                "avg_rtf": sum(rtfs) / len(rtfs) if rtfs else 0,
                "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        return {
            "llm": self.get_llm_stats(),
            "stt": self.get_stt_stats(),
            "tts": self.get_tts_stats(),
        }


# Global metrics aggregator
metrics_aggregator = MetricsAggregator()


def log_llm_metrics(metrics: LLMMetrics):
    """Log LLM metrics in structured format"""
    metrics_aggregator.add_llm_metrics(metrics)
    logger.info(
        f"[LLM] request_id={metrics.request_id} "
        f"model={metrics.model} "
        f"tokens={metrics.total_tokens} "
        f"first_token={metrics.first_token_latency_ms:.0f}ms "
        f"total={metrics.total_latency_ms:.0f}ms "
        f"tps={metrics.tokens_per_second:.1f} "
        f"success={metrics.success}"
    )
    # Also log as JSON for parsing
    logger.debug(f"LLM_METRICS_JSON: {json.dumps(asdict(metrics))}")


def log_stt_metrics(metrics: STTMetrics):
    """Log STT metrics in structured format"""
    metrics_aggregator.add_stt_metrics(metrics)
    logger.info(
        f"[STT] request_id={metrics.request_id} "
        f"audio={metrics.audio_duration_s:.2f}s "
        f"process={metrics.processing_time_ms:.0f}ms "
        f"rtf={metrics.rtf:.2f}x "
        f"text_len={metrics.text_length} "
        f"lang={metrics.language} "
        f"device={metrics.device} "
        f"success={metrics.success}"
    )
    logger.debug(f"STT_METRICS_JSON: {json.dumps(asdict(metrics))}")


def log_tts_metrics(metrics: TTSMetrics):
    """Log TTS metrics in structured format"""
    metrics_aggregator.add_tts_metrics(metrics)
    logger.info(
        f"[TTS] request_id={metrics.request_id} "
        f"text_len={metrics.text_length} "
        f"audio={metrics.audio_duration_s:.2f}s "
        f"process={metrics.processing_time_ms:.0f}ms "
        f"rtf={metrics.rtf:.2f}x "
        f"bytes={metrics.audio_bytes} "
        f"success={metrics.success}"
    )
    logger.debug(f"TTS_METRICS_JSON: {json.dumps(asdict(metrics))}")


def log_stats_summary():
    """Log aggregated statistics summary"""
    stats = metrics_aggregator.get_all_stats()
    logger.info(f"[STATS] Performance Summary: {json.dumps(stats, indent=2)}")
