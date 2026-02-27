# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""
Benchmark runner — orchestrates lm-eval harness and performance benchmarks.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from leo.benchmarks.harness import HarnessRunner
from leo.benchmarks.registry import resolve_suite, SUITE_REGISTRY
from leo.benchmarks.suites.performance import PerformanceBenchmark
from leo.core.config import BenchmarkConfig, ModelConfig, ReportingConfig
from leo.core.types import BenchmarkResult
from leo.models.base import BaseModel
from leo.utils.hardware import profile_section
from leo.utils.logging import get_logger

logger = get_logger("benchmarks.runner")


class BenchmarkRunner:
    """
    High-level orchestrator that:
    1. Resolves which benchmarks to run (from config, suites, individual tasks)
    2. Delegates to HarnessRunner for standard benchmarks
    3. Runs PerformanceBenchmark for throughput/latency profiling
    4. Collects and returns all results
    """

    def __init__(
        self,
        model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
        reporting_config: ReportingConfig,
        seed: int = 42,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_config = model_config
        self.benchmark_config = benchmark_config
        self.reporting_config = reporting_config
        self.seed = seed
        self.cache_dir = cache_dir

    def run(self, model: Optional[BaseModel] = None) -> list[BenchmarkResult]:
        """
        Execute all configured benchmarks.

        If a loaded model is provided, it will be used for performance benchmarks.
        Standard benchmarks always go through lm-evaluation-harness which loads
        its own model instance.
        """
        all_results: list[BenchmarkResult] = []
        run_performance = False

        # Check if performance suite is requested
        suite = self.benchmark_config.suite
        benchmarks = self.benchmark_config.benchmarks

        if suite == "performance" or "performance" in benchmarks:
            run_performance = True
            # Remove 'performance' from benchmarks for harness
            benchmarks = [b for b in benchmarks if b != "performance"]

        # Determine if we have standard benchmarks to run
        has_standard = bool(benchmarks) or (suite and suite != "performance")

        # Run standard benchmarks via lm-evaluation-harness
        if has_standard:
            # Create a copy of benchmark config without performance
            std_config = BenchmarkConfig(
                benchmarks=benchmarks,
                suite=suite if suite != "performance" else None,
                category=self.benchmark_config.category,
                num_fewshot=self.benchmark_config.num_fewshot,
                limit=self.benchmark_config.limit,
                tasks=self.benchmark_config.tasks,
                custom_task_yaml=self.benchmark_config.custom_task_yaml,
                generation_kwargs=self.benchmark_config.generation_kwargs,
                log_samples=self.benchmark_config.log_samples,
                check_integrity=self.benchmark_config.check_integrity,
            )

            harness = HarnessRunner(
                model_config=self.model_config,
                benchmark_config=std_config,
                reporting_config=self.reporting_config,
                seed=self.seed,
                cache_dir=self.cache_dir,
            )

            logger.info("=== Running standard benchmarks ===")
            with profile_section("standard_benchmarks") as prof:
                harness_results = harness.run()
                prof.samples_processed = len(harness_results)

            all_results.extend(harness_results)

            logger.info(
                "Standard benchmarks complete: %d results in %.1fs",
                len(harness_results),
                prof.wall_time_seconds,
            )

        # Run performance benchmarks
        if run_performance and model is not None:
            logger.info("=== Running performance benchmarks ===")
            perf_bench = PerformanceBenchmark(
                model=model,
                warmup_iterations=3,
                test_iterations=10,
            )
            perf_results = perf_bench.run()
            all_results.extend(perf_results)

        return all_results
