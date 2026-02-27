# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""Core type definitions for Leo."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ModelBackend(str, Enum):
    """Supported model loading backends."""

    HUGGINGFACE = "huggingface"
    UNSLOTH = "unsloth"
    VLLM = "vllm"
    GGUF = "gguf"
    AUTO = "auto"

    @classmethod
    def from_string(cls, value: str) -> ModelBackend:
        """Resolve backend from a string, case-insensitive."""
        _map = {
            "hf": cls.HUGGINGFACE,
            "huggingface": cls.HUGGINGFACE,
            "transformers": cls.HUGGINGFACE,
            "unsloth": cls.UNSLOTH,
            "vllm": cls.VLLM,
            "gguf": cls.GGUF,
            "ggml": cls.GGUF,
            "llama.cpp": cls.GGUF,
            "llama-cpp": cls.GGUF,
            "auto": cls.AUTO,
        }
        normalized = value.strip().lower()
        if normalized in _map:
            return _map[normalized]
        raise ValueError(
            f"Unknown model backend '{value}'. "
            f"Supported: {', '.join(sorted(_map.keys()))}"
        )


class BenchmarkCategory(str, Enum):
    """Categories of benchmark suites."""

    KNOWLEDGE = "knowledge"
    REASONING = "reasoning"
    MATHEMATICS = "mathematics"
    CODE = "code"
    LANGUAGE = "language"
    LONG_CONTEXT = "long_context"
    MULTILINGUAL = "multilingual"
    SAFETY = "safety"
    PERFORMANCE = "performance"
    LEADERBOARD = "leaderboard"
    CUSTOM = "custom"


class DeviceType(str, Enum):
    """Device placement targets."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

    @classmethod
    def resolve(cls, value: str) -> DeviceType:
        """Normalise a device specification."""
        v = value.strip().lower()
        if v.startswith("cuda"):
            return cls.CUDA
        if v == "mps":
            return cls.MPS
        if v == "cpu":
            return cls.CPU
        return cls.AUTO


class QuantizationType(str, Enum):
    """Supported quantization methods."""

    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF_Q4 = "gguf_q4"
    GGUF_Q5 = "gguf_q5"
    GGUF_Q8 = "gguf_q8"


class TaskOutputType(str, Enum):
    """Task output types matching lm-evaluation-harness."""

    GENERATE_UNTIL = "generate_until"
    LOGLIKELIHOOD = "loglikelihood"
    LOGLIKELIHOOD_ROLLING = "loglikelihood_rolling"
    MULTIPLE_CHOICE = "multiple_choice"


class BenchmarkResult:
    """Container for a single benchmark run result."""

    def __init__(
        self,
        benchmark_name: str,
        metrics: dict[str, float],
        samples: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.benchmark_name = benchmark_name
        self.metrics = metrics
        self.samples = samples
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        top_metric = next(iter(self.metrics.items())) if self.metrics else ("N/A", 0.0)
        return (
            f"BenchmarkResult(name={self.benchmark_name!r}, "
            f"{top_metric[0]}={top_metric[1]:.4f}, samples={self.samples})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "metrics": self.metrics,
            "samples": self.samples,
            "metadata": self.metadata,
        }
