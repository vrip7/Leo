# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""
Cross-model result comparison and statistical analysis.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from leo.evaluation.scorer import EvaluationReport
from leo.reporting.formatter import ResultsFormatter
from leo.utils.logging import get_logger

logger = get_logger("reporting.comparator")
console = Console()


@dataclass
class ComparisonEntry:
    """One benchmark row across multiple models."""

    benchmark: str
    category: str
    primary_metric: str
    scores: dict[str, float] = field(default_factory=dict)
    stderrs: dict[str, float | None] = field(default_factory=dict)
    best_model: str = ""
    delta_max: float = 0.0


@dataclass
class ComparisonResult:
    """Full comparison across models."""

    model_names: list[str]
    entries: list[ComparisonEntry]
    composite_scores: dict[str, float] = field(default_factory=dict)
    winner: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_names": self.model_names,
            "composite_scores": self.composite_scores,
            "winner": self.winner,
            "entries": [
                {
                    "benchmark": e.benchmark,
                    "category": e.category,
                    "primary_metric": e.primary_metric,
                    "scores": e.scores,
                    "stderrs": e.stderrs,
                    "best_model": e.best_model,
                    "delta_max": e.delta_max,
                }
                for e in self.entries
            ],
        }


class ResultsComparator:
    """Compare evaluation results across multiple models."""

    def __init__(self) -> None:
        self._reports: dict[str, EvaluationReport] = {}
        self._formatter = ResultsFormatter()

    # ── Loading ──────────────────────────────────────────────────

    def add_report(self, report: EvaluationReport) -> None:
        """Register an evaluation report for comparison."""
        self._reports[report.model_name] = report
        logger.info("Added report for model: %s", report.model_name)

    def load_report(self, path: str | Path) -> EvaluationReport:
        """Load a report from a JSON file and register it."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        report = self._dict_to_report(data)
        self.add_report(report)
        return report

    @staticmethod
    def _dict_to_report(data: dict[str, Any]) -> EvaluationReport:
        """Convert a raw dict (from JSON) to an EvaluationReport."""
        from leo.evaluation.scorer import ScoredBenchmark

        benchmarks = []
        for bdict in data.get("benchmarks", []):
            benchmarks.append(
                ScoredBenchmark(
                    name=bdict["name"],
                    display_name=bdict.get("display_name", bdict["name"]),
                    category=bdict.get("category", "unknown"),
                    primary_metric=bdict.get("primary_metric", ""),
                    primary_value=bdict.get("primary_value", 0.0),
                    all_metrics=bdict.get("all_metrics", {}),
                    samples=bdict.get("samples", 0),
                    stderr=bdict.get("stderr"),
                    normalized_score=bdict.get("normalized_score", 0.0),
                    random_baseline=bdict.get("random_baseline", 0.0),
                )
            )

        return EvaluationReport(
            model_name=data.get("model_name", "unknown"),
            benchmarks=benchmarks,
            composite_score=data.get("composite_score", 0.0),
            timestamp=data.get("timestamp", ""),
            performance_metrics=data.get("performance_metrics", {}),
            system_info=data.get("system_info", {}),
            config=data.get("config", {}),
        )

    # ── Core comparison ──────────────────────────────────────────

    def compare(self) -> ComparisonResult:
        """Run comparison across all registered reports."""
        if len(self._reports) < 2:
            logger.warning("Comparison requires at least 2 reports")
            if len(self._reports) == 1:
                name = list(self._reports.keys())[0]
                return ComparisonResult(
                    model_names=[name],
                    entries=[],
                    composite_scores={name: self._reports[name].composite_score},
                    winner=name,
                )
            return ComparisonResult(model_names=[], entries=[], composite_scores={}, winner="")

        model_names = list(self._reports.keys())
        composite_scores = {name: report.composite_score for name, report in self._reports.items()}

        # Collect all benchmarks
        all_benchmarks: dict[str, dict[str, Any]] = {}
        for model_name, report in self._reports.items():
            for bench in report.benchmarks:
                key = bench.name
                if key not in all_benchmarks:
                    all_benchmarks[key] = {
                        "category": bench.category,
                        "primary_metric": bench.primary_metric,
                        "scores": {},
                        "stderrs": {},
                    }
                all_benchmarks[key]["scores"][model_name] = bench.primary_value
                all_benchmarks[key]["stderrs"][model_name] = bench.stderr

        entries = []
        for bname, bdata in sorted(all_benchmarks.items()):
            scores = bdata["scores"]
            if scores:
                best_model = max(scores, key=scores.get)  # type: ignore[arg-type]
                values = list(scores.values())
                delta_max = max(values) - min(values) if len(values) > 1 else 0.0
            else:
                best_model = ""
                delta_max = 0.0

            entries.append(
                ComparisonEntry(
                    benchmark=bname,
                    category=bdata["category"],
                    primary_metric=bdata["primary_metric"],
                    scores=scores,
                    stderrs=bdata["stderrs"],
                    best_model=best_model,
                    delta_max=delta_max,
                )
            )

        winner = max(composite_scores, key=composite_scores.get) if composite_scores else ""  # type: ignore[arg-type]

        return ComparisonResult(
            model_names=model_names,
            entries=entries,
            composite_scores=composite_scores,
            winner=winner,
        )

    # ── Statistical helpers ──────────────────────────────────────

    @staticmethod
    def is_significant(
        score_a: float,
        score_b: float,
        stderr_a: float | None,
        stderr_b: float | None,
        threshold: float = 1.96,
    ) -> bool:
        """Test whether the difference between two scores is statistically significant
        at the 95% confidence level (z > threshold)."""
        if stderr_a is None or stderr_b is None:
            return False
        if stderr_a == 0.0 and stderr_b == 0.0:
            return score_a != score_b

        combined_se = math.sqrt(stderr_a**2 + stderr_b**2)
        if combined_se == 0.0:
            return False

        z = abs(score_a - score_b) / combined_se
        return z > threshold

    # ── Display ──────────────────────────────────────────────────

    def print_comparison(self, result: ComparisonResult | None = None) -> None:
        """Print a rich comparison table to the console."""
        if result is None:
            result = self.compare()

        # Composite scores table
        composite_table = Table(title="Composite Scores", show_header=True, header_style="bold cyan")
        composite_table.add_column("Model", style="bold")
        composite_table.add_column("Composite", justify="right")
        composite_table.add_column("Winner", justify="center")

        for name in result.model_names:
            score = result.composite_scores.get(name, 0.0)
            is_winner = name == result.winner
            composite_table.add_row(
                name,
                f"{score:.4f}",
                "★" if is_winner else "",
            )

        console.print(composite_table)
        console.print()

        if not result.entries:
            return

        # Detailed comparison table
        detail_table = Table(
            title="Benchmark Comparison",
            show_header=True,
            header_style="bold cyan",
        )
        detail_table.add_column("Benchmark", style="bold")
        detail_table.add_column("Category")
        detail_table.add_column("Metric")
        for name in result.model_names:
            detail_table.add_column(name, justify="right")
        detail_table.add_column("Best", style="green")
        detail_table.add_column("Δ", justify="right")

        for entry in result.entries:
            row = [entry.benchmark, entry.category, entry.primary_metric]
            for name in result.model_names:
                score = entry.scores.get(name)
                stderr = entry.stderrs.get(name)
                if score is not None:
                    cell = f"{score:.4f}"
                    if stderr is not None:
                        cell += f" ±{stderr:.4f}"
                    if name == entry.best_model:
                        cell = f"[green]{cell}[/green]"
                    row.append(cell)
                else:
                    row.append("—")
            row.append(entry.best_model)
            row.append(f"{entry.delta_max:.4f}")
            detail_table.add_row(*row)

        console.print(detail_table)

    def save_comparison(self, path: str | Path, result: ComparisonResult | None = None) -> None:
        """Save comparison result to a JSON file."""
        if result is None:
            result = self.compare()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info("Comparison saved to: %s", path)
