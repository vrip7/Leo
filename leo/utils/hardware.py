# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""Timing, memory tracking, and hardware profiling utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

import torch

from leo.utils.logging import get_logger

logger = get_logger("hardware")


@dataclass
class ProfileMetrics:
    """Collected metrics from a profiled run."""

    wall_time_seconds: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    peak_cpu_memory_mb: float = 0.0
    tokens_processed: int = 0
    tokens_per_second: float = 0.0
    samples_processed: int = 0
    samples_per_second: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_time_seconds": round(self.wall_time_seconds, 3),
            "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 2),
            "peak_cpu_memory_mb": round(self.peak_cpu_memory_mb, 2),
            "tokens_processed": self.tokens_processed,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "samples_processed": self.samples_processed,
            "samples_per_second": round(self.samples_per_second, 2),
            **self.extra,
        }


def get_gpu_memory_mb() -> float:
    """Current allocated GPU memory in MB across all devices."""
    if not torch.cuda.is_available():
        return 0.0
    total = 0.0
    for i in range(torch.cuda.device_count()):
        total += torch.cuda.memory_allocated(i) / (1024 * 1024)
    return total


def get_peak_gpu_memory_mb() -> float:
    """Peak GPU memory allocated since last reset, in MB."""
    if not torch.cuda.is_available():
        return 0.0
    total = 0.0
    for i in range(torch.cuda.device_count()):
        total += torch.cuda.max_memory_allocated(i) / (1024 * 1024)
    return total


def reset_gpu_memory_stats() -> None:
    """Reset peak memory tracking on all CUDA devices."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def get_cpu_memory_mb() -> float:
    """Current process RSS in MB."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


@contextmanager
def profile_section(name: str = "section") -> Generator[ProfileMetrics, None, None]:
    """Context manager that profiles wall time and GPU memory for a section of code."""
    metrics = ProfileMetrics()

    reset_gpu_memory_stats()
    cpu_before = get_cpu_memory_mb()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    yield metrics

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start

    metrics.wall_time_seconds = elapsed
    metrics.peak_gpu_memory_mb = get_peak_gpu_memory_mb()
    metrics.peak_cpu_memory_mb = get_cpu_memory_mb() - cpu_before

    if metrics.tokens_processed > 0 and elapsed > 0:
        metrics.tokens_per_second = metrics.tokens_processed / elapsed
    if metrics.samples_processed > 0 and elapsed > 0:
        metrics.samples_per_second = metrics.samples_processed / elapsed

    logger.debug(
        "[%s] %.2fs | GPU peak: %.0f MB | %.1f tok/s",
        name,
        elapsed,
        metrics.peak_gpu_memory_mb,
        metrics.tokens_per_second,
    )


class LatencyTracker:
    """Accumulates per-sample latencies for percentile reporting."""

    def __init__(self) -> None:
        self._latencies: list[float] = []

    def record(self, seconds: float) -> None:
        self._latencies.append(seconds)

    @property
    def count(self) -> int:
        return len(self._latencies)

    @property
    def mean(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    def percentile(self, p: float) -> float:
        """Return the p-th percentile latency."""
        if not self._latencies:
            return 0.0
        import numpy as np

        return float(np.percentile(self._latencies, p))

    def summary(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean * 1000, 2),
            "p50_ms": round(self.percentile(50) * 1000, 2),
            "p90_ms": round(self.percentile(90) * 1000, 2),
            "p95_ms": round(self.percentile(95) * 1000, 2),
            "p99_ms": round(self.percentile(99) * 1000, 2),
        }
