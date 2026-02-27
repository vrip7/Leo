# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""
Results formatting — JSON, table, and summary output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from leo.evaluation.scorer import EvaluationReport, ScoredBenchmark
from leo.utils.logging import get_console, get_logger

logger = get_logger("reporting.formatter")


class ResultsFormatter:
    """Formats evaluation reports for console display and file output."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or get_console()

    # ── Console output ───────────────────────────────────────────

    def print_summary(self, report: EvaluationReport) -> None:
        """Print a rich summary table to the console."""
        self.console.print()
        self.console.print(
            Panel.fit(
                f"[bold blue]{report.model_name}[/bold blue]\n"
                f"Composite Score: [bold green]{report.composite_score:.4f}[/bold green]\n"
                f"Benchmarks: {len(report.benchmarks)} | "
                f"Timestamp: {report.timestamp}",
                title="[bold]Leo Evaluation Results[/bold]",
                border_style="cyan",
            )
        )

        # Results table
        table = Table(
            title="Benchmark Results",
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
        )
        table.add_column("Benchmark", style="cyan", min_width=20)
        table.add_column("Category", style="dim")
        table.add_column("Metric", style="white")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Normalized", justify="right", style="yellow")
        table.add_column("Stderr", justify="right", style="dim")
        table.add_column("Samples", justify="right", style="dim")

        # Sort by category then name
        sorted_benchmarks = sorted(report.benchmarks, key=lambda b: (b.category, b.name))

        for b in sorted_benchmarks:
            stderr_str = f"±{b.stderr:.4f}" if b.stderr else "—"
            table.add_row(
                b.display_name,
                b.category,
                b.primary_metric,
                f"{b.primary_value:.4f}",
                f"{b.normalized_score:.4f}",
                stderr_str,
                str(b.samples),
            )

        self.console.print(table)

        # Performance metrics
        if report.performance_metrics:
            self.console.print()
            perf_table = Table(
                title="Performance Metrics",
                show_header=True,
                header_style="bold magenta",
                show_lines=True,
            )
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", justify="right", style="green")

            for section, metrics in report.performance_metrics.items():
                section_name = section.replace("performance:", "").replace("_", " ").title()
                perf_table.add_row(f"[bold]{section_name}[/bold]", "")
                for k, v in metrics.items():
                    display_key = k.replace("_", " ")
                    if isinstance(v, float):
                        perf_table.add_row(f"  {display_key}", f"{v:.2f}")
                    else:
                        perf_table.add_row(f"  {display_key}", str(v))

            self.console.print(perf_table)

        self.console.print()

    # ── File output ──────────────────────────────────────────────

    def save_json(self, report: EvaluationReport, path: str | Path) -> None:
        """Save the evaluation report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        logger.info("Results saved to: %s", path)

    def save_csv(self, report: EvaluationReport, path: str | Path) -> None:
        """Save benchmark results as CSV."""
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "benchmark", "display_name", "category", "primary_metric",
                "primary_value", "normalized_score", "stderr", "samples",
            ])

            for b in report.benchmarks:
                writer.writerow([
                    b.name, b.display_name, b.category, b.primary_metric,
                    f"{b.primary_value:.4f}", f"{b.normalized_score:.4f}",
                    f"{b.stderr:.4f}" if b.stderr else "",
                    b.samples,
                ])

        logger.info("CSV results saved to: %s", path)

    @staticmethod
    def load_json(path: str | Path) -> EvaluationReport:
        """Load an evaluation report from JSON."""
        from leo.evaluation.scorer import ScoredBenchmark

        with open(path) as f:
            data = json.load(f)

        benchmarks = [
            ScoredBenchmark(
                name=b["name"],
                display_name=b["display_name"],
                category=b["category"],
                primary_metric=b["primary_metric"],
                primary_value=b["primary_value"],
                normalized_score=b["normalized_score"],
                all_metrics=b["metrics"],
                stderr=b.get("stderr"),
                samples=b.get("samples", 0),
            )
            for b in data.get("benchmarks", [])
        ]

        return EvaluationReport(
            model_name=data.get("model", "unknown"),
            benchmarks=benchmarks,
            composite_score=data.get("composite_score", 0.0),
            performance_metrics=data.get("performance", {}),
            system_info=data.get("system_info", {}),
            config=data.get("config", {}),
            timestamp=data.get("timestamp", ""),
        )
