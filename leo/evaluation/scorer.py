# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""
Scorer: aggregates raw benchmark results into structured evaluation reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from leo.benchmarks.registry import BENCHMARK_REGISTRY, get_benchmark_info
from leo.core.types import BenchmarkResult
from leo.evaluation.metrics import (
    compute_composite_score,
    compute_normalized_score,
    standard_error,
)
from leo.utils.logging import get_logger

logger = get_logger("evaluation.scorer")


@dataclass
class ScoredBenchmark:
    """A benchmark result with computed normalized scores and rankings."""

    name: str
    display_name: str
    category: str
    primary_metric: str
    primary_value: float
    normalized_score: float
    all_metrics: dict[str, float]
    stderr: float | None = None
    samples: int = 0


@dataclass
class EvaluationReport:
    """Complete evaluation report for a model."""

    model_name: str
    benchmarks: list[ScoredBenchmark]
    composite_score: float
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    system_info: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "composite_score": round(self.composite_score, 4),
            "timestamp": self.timestamp,
            "benchmarks": [
                {
                    "name": b.name,
                    "display_name": b.display_name,
                    "category": b.category,
                    "primary_metric": b.primary_metric,
                    "primary_value": round(b.primary_value, 4),
                    "normalized_score": round(b.normalized_score, 4),
                    "metrics": {k: round(v, 4) for k, v in b.all_metrics.items()},
                    "stderr": round(b.stderr, 4) if b.stderr else None,
                    "samples": b.samples,
                }
                for b in self.benchmarks
            ],
            "performance": self.performance_metrics,
            "system_info": self.system_info,
            "config": self.config,
        }


class Scorer:
    """Transforms raw BenchmarkResult objects into a scored EvaluationReport."""

    # Random baseline for common benchmark types
    BASELINES = {
        "multiple_choice_4": 0.25,
        "multiple_choice_2": 0.50,
        "multiple_choice_10": 0.10,
        "generation": 0.0,
        "perplexity": 0.0,
    }

    def score(
        self,
        model_name: str,
        results: list[BenchmarkResult],
        system_info: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Produce a full evaluation report from raw benchmark results."""
        import datetime

        scored_benchmarks: list[ScoredBenchmark] = []
        perf_metrics: dict[str, Any] = {}

        for result in results:
            # Separate performance benchmarks
            if result.benchmark_name.startswith("performance:"):
                perf_metrics[result.benchmark_name] = result.metrics
                continue

            scored = self._score_benchmark(result)
            if scored:
                scored_benchmarks.append(scored)

        # Composite score from all non-performance benchmarks
        results_for_composite = {
            b.name: {b.primary_metric: b.primary_value}
            for b in scored_benchmarks
        }
        composite = compute_composite_score(results_for_composite)

        return EvaluationReport(
            model_name=model_name,
            benchmarks=scored_benchmarks,
            composite_score=composite,
            performance_metrics=perf_metrics,
            system_info=system_info or {},
            config=config or {},
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    def _score_benchmark(self, result: BenchmarkResult) -> ScoredBenchmark | None:
        """Score a single benchmark result."""
        if not result.metrics:
            logger.warning("No metrics for benchmark: %s", result.benchmark_name)
            return None

        info = get_benchmark_info(result.benchmark_name)

        # Determine primary metric
        if info and info.metrics:
            primary_metric = info.metrics[0]
        else:
            # Prefer acc > acc_norm > exact_match > f1
            priority = ["acc", "acc_norm", "exact_match", "f1", "em"]
            primary_metric = next(
                (m for m in priority if m in result.metrics),
                next(iter(result.metrics)),
            )

        primary_value = result.metrics.get(primary_metric, 0.0)

        # Get stderr if available
        stderr_key = f"{primary_metric}_stderr"
        stderr = result.metrics.get(stderr_key)

        # Determine baseline for normalized scoring
        baseline = 0.25  # Default: 4-choice MC
        if info:
            if info.output_type == "generate_until":
                baseline = 0.0
            elif "mc2" in info.name or "truthfulqa" in info.name:
                baseline = 0.0  # TruthfulQA MC2 doesn't have a simple random baseline
            elif info.output_type == "loglikelihood_rolling":
                baseline = 0.0

        normalized = compute_normalized_score(primary_value, baseline=baseline)

        category = info.category.value if info else "unknown"
        display_name = info.display_name if info else result.benchmark_name

        return ScoredBenchmark(
            name=result.benchmark_name,
            display_name=display_name,
            category=category,
            primary_metric=primary_metric,
            primary_value=primary_value,
            normalized_score=normalized,
            all_metrics=result.metrics,
            stderr=stderr,
            samples=result.samples,
        )
