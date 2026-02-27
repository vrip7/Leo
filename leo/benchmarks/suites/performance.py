# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""
Performance benchmarking suite — measures latency, throughput, memory usage,
and tokens-per-second for model inference.

This suite does NOT run lm-evaluation-harness tasks but instead directly
profiles the loaded model.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from leo.core.types import BenchmarkResult
from leo.models.base import BaseModel
from leo.utils.hardware import (
    LatencyTracker,
    ProfileMetrics,
    get_gpu_memory_mb,
    get_peak_gpu_memory_mb,
    profile_section,
    reset_gpu_memory_stats,
)
from leo.utils.logging import get_logger

logger = get_logger("benchmarks.performance")

# Standardized prompts for performance testing (varying lengths)
_PERF_PROMPTS = {
    "short": "What is the capital of France?",
    "medium": (
        "Explain the theory of general relativity in simple terms. "
        "Include the key concepts of spacetime curvature, the equivalence principle, "
        "and gravitational time dilation."
    ),
    "long": (
        "Write a detailed technical analysis of the transformer architecture used in modern large language models. "
        "Cover the self-attention mechanism, multi-head attention, positional encodings, "
        "feed-forward networks, layer normalization, and residual connections. "
        "Discuss the computational complexity of each component and how techniques like "
        "FlashAttention, grouped query attention, and sliding window attention "
        "have been developed to improve efficiency. Also discuss the KV cache "
        "and its impact on memory usage during autoregressive generation. "
        "Compare the original Vaswani et al. architecture with modern variants "
        "like LLaMA, Mistral, and GPT-4 architecture decisions."
    ),
}

# Generation lengths for throughput testing
_GEN_LENGTHS = [32, 128, 512]


class PerformanceBenchmark:
    """
    Measures model performance characteristics:
    - Prefill latency (time to process prompt)
    - Generation latency (time per token during generation)
    - Throughput (tokens/second)
    - Memory footprint (GPU/CPU)
    - Batch scaling efficiency
    """

    def __init__(
        self,
        model: BaseModel,
        warmup_iterations: int = 3,
        test_iterations: int = 10,
    ) -> None:
        self.model = model
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations

    def run(self) -> list[BenchmarkResult]:
        """Execute all performance benchmarks."""
        assert self.model.is_loaded, "Model must be loaded for performance benchmarking"

        results: list[BenchmarkResult] = []

        logger.info("Running performance benchmarks...")

        # 1. Memory footprint
        results.append(self._benchmark_memory())

        # 2. Prefill latency (prompt processing speed)
        results.append(self._benchmark_prefill())

        # 3. Generation throughput
        results.append(self._benchmark_generation())

        # 4. Batch scaling
        results.append(self._benchmark_batch_scaling())

        logger.info("Performance benchmarks complete")
        return results

    def _warmup(self, prompt: str, max_tokens: int = 16) -> None:
        """Run warmup iterations to stabilize GPU clocks and caches."""
        for _ in range(self.warmup_iterations):
            self.model.generate([prompt], max_new_tokens=max_tokens, temperature=0)

    def _benchmark_memory(self) -> BenchmarkResult:
        """Measure static memory footprint of the loaded model."""
        logger.info("Benchmarking memory footprint...")

        metrics: dict[str, float] = {}

        gpu_mem = get_gpu_memory_mb()
        metrics["model_gpu_memory_mb"] = round(gpu_mem, 2)

        from leo.utils.hardware import get_cpu_memory_mb

        cpu_mem = get_cpu_memory_mb()
        metrics["process_rss_mb"] = round(cpu_mem, 2)

        num_params = self.model.get_num_parameters()
        metrics["num_parameters"] = num_params
        metrics["parameters_billions"] = round(num_params / 1e9, 3)

        return BenchmarkResult(
            benchmark_name="performance:memory",
            metrics=metrics,
            samples=1,
            metadata={"type": "performance", "subtype": "memory"},
        )

    def _benchmark_prefill(self) -> BenchmarkResult:
        """Measure prompt processing (prefill) latency."""
        logger.info("Benchmarking prefill latency...")

        self._warmup(_PERF_PROMPTS["short"], max_tokens=1)

        tracker_by_length: dict[str, LatencyTracker] = {}

        for prompt_name, prompt_text in _PERF_PROMPTS.items():
            tracker = LatencyTracker()
            for _ in range(self.test_iterations):
                start = time.perf_counter()
                self.model.generate([prompt_text], max_new_tokens=1, temperature=0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                tracker.record(elapsed)
            tracker_by_length[prompt_name] = tracker

        metrics: dict[str, float] = {}
        for name, tracker in tracker_by_length.items():
            summary = tracker.summary()
            for k, v in summary.items():
                metrics[f"prefill_{name}_{k}"] = v

        return BenchmarkResult(
            benchmark_name="performance:prefill",
            metrics=metrics,
            samples=self.test_iterations * len(_PERF_PROMPTS),
            metadata={"type": "performance", "subtype": "prefill"},
        )

    def _benchmark_generation(self) -> BenchmarkResult:
        """Measure token generation throughput."""
        logger.info("Benchmarking generation throughput...")

        self._warmup(_PERF_PROMPTS["medium"], max_tokens=16)

        metrics: dict[str, float] = {}

        for gen_length in _GEN_LENGTHS:
            tracker = LatencyTracker()
            tokens_per_sec_samples: list[float] = []

            for _ in range(self.test_iterations):
                reset_gpu_memory_stats()
                prompt = _PERF_PROMPTS["medium"]

                start = time.perf_counter()
                outputs = self.model.generate(
                    [prompt], max_new_tokens=gen_length, temperature=0
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                tracker.record(elapsed)
                # Approximate generated tokens from output length
                if outputs and len(outputs[0]) > 0:
                    approx_tokens = gen_length  # Upper bound
                    tps = approx_tokens / elapsed if elapsed > 0 else 0
                    tokens_per_sec_samples.append(tps)

            summary = tracker.summary()
            for k, v in summary.items():
                metrics[f"gen_{gen_length}tok_{k}"] = v

            if tokens_per_sec_samples:
                avg_tps = sum(tokens_per_sec_samples) / len(tokens_per_sec_samples)
                metrics[f"gen_{gen_length}tok_tokens_per_sec"] = round(avg_tps, 2)

        return BenchmarkResult(
            benchmark_name="performance:generation",
            metrics=metrics,
            samples=self.test_iterations * len(_GEN_LENGTHS),
            metadata={"type": "performance", "subtype": "generation"},
        )

    def _benchmark_batch_scaling(self) -> BenchmarkResult:
        """Measure how throughput scales with batch size."""
        logger.info("Benchmarking batch scaling...")

        batch_sizes = [1, 2, 4, 8]
        prompt = _PERF_PROMPTS["short"]
        gen_length = 32
        metrics: dict[str, float] = {}

        self._warmup(prompt, max_tokens=gen_length)

        for bs in batch_sizes:
            prompts = [prompt] * bs
            times: list[float] = []

            for _ in range(max(3, self.test_iterations // 2)):
                try:
                    start = time.perf_counter()
                    self.model.generate(prompts, max_new_tokens=gen_length, temperature=0)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    logger.warning("OOM at batch size %d, stopping batch scaling test", bs)
                    break

            if times:
                avg_time = sum(times) / len(times)
                total_tokens = bs * gen_length
                tps = total_tokens / avg_time if avg_time > 0 else 0
                metrics[f"batch_{bs}_avg_time_ms"] = round(avg_time * 1000, 2)
                metrics[f"batch_{bs}_tokens_per_sec"] = round(tps, 2)
                metrics[f"batch_{bs}_samples_per_sec"] = round(bs / avg_time if avg_time > 0 else 0, 2)

        return BenchmarkResult(
            benchmark_name="performance:batch_scaling",
            metrics=metrics,
            samples=sum(len(v) for v in [batch_sizes]),
            metadata={"type": "performance", "subtype": "batch_scaling"},
        )
