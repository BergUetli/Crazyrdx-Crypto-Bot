"""
compute_meter.py
Tracks CPU/GPU time per layer and per model call.
"""

import time
import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LayerUsage:
    cpu_seconds: float = 0.0
    gpu_seconds: float = 0.0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class ComputeMeter:
    """Singleton-style meter. Import and use `meter` instance."""
    _usage: Dict[str, LayerUsage] = field(default_factory=dict)

    def record_cpu(self, layer: str, seconds: float):
        u = self._usage.setdefault(layer, LayerUsage())
        u.cpu_seconds += seconds
        u.calls += 1

    def record_gpu(self, layer: str, seconds: float):
        u = self._usage.setdefault(layer, LayerUsage())
        u.gpu_seconds += seconds
        u.calls += 1

    def record_tokens(self, layer: str, tokens_in: int, tokens_out: int):
        u = self._usage.setdefault(layer, LayerUsage())
        u.tokens_in += tokens_in
        u.tokens_out += tokens_out

    def get_summary(self) -> Dict[str, dict]:
        return {
            layer: {
                "cpu_seconds": round(u.cpu_seconds, 3),
                "gpu_seconds": round(u.gpu_seconds, 3),
                "calls": u.calls,
                "tokens_in": u.tokens_in,
                "tokens_out": u.tokens_out,
            }
            for layer, u in self._usage.items()
        }

    def reset(self):
        self._usage.clear()


meter = ComputeMeter()


@contextmanager
def track_cpu(layer: str):
    """Context manager to time a CPU-bound block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        meter.record_cpu(layer, elapsed)


@contextmanager
def track_gpu(layer: str):
    """Context manager to time a GPU-bound block (MPS on Mac)."""
    start = time.process_time()
    try:
        yield
    finally:
        elapsed = time.process_time() - start
        meter.record_gpu(layer, elapsed)


def timed(layer: str, gpu: bool = False):
    """Decorator for functions."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if gpu:
                with track_gpu(layer):
                    return fn(*args, **kwargs)
            else:
                with track_cpu(layer):
                    return fn(*args, **kwargs)
        return wrapper
    return decorator
