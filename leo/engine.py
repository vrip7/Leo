# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""
Main orchestration engine.

Provides a unified API for configuring, running, and reporting benchmarks.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Optional

from leo.benchmarks.runner import BenchmarkRunner
from leo.core.config import (
    BenchmarkConfig,
    LeoConfig,
    ModelConfig,
    ReportingConfig,
)
from leo.core.device import get_system_info
from leo.core.types import BenchmarkResult
from leo.evaluation.scorer import EvaluationReport, Scorer
from leo.models.base import BaseModel
from leo.models.loader import ModelLoader
from leo.reporting.comparator import ResultsComparator
from leo.reporting.formatter import ResultsFormatter
from leo.reporting.html_report import HTMLReportGenerator
from leo.utils.logging import get_logger, setup_logging

logger = get_logger("engine")


class LeoResults:
    """
    Wrapper around an EvaluationReport providing convenient save/export helpers.

    Returned by ``Leo.run()`` so the user can chain:
        results = engine.run()
        results.save("results/model.json")
        results.to_html("reports/model.html")
    """

    def __init__(self, report: EvaluationReport, raw_results: list[BenchmarkResult]) -> None:
        self.report = report
        self.raw_results = raw_results
        self._formatter = ResultsFormatter()
        self._html_gen = HTMLReportGenerator()

    # ── Properties ───────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self.report.model_name

    @property
    def composite_score(self) -> float:
        return self.report.composite_score

    @property
    def benchmarks(self) -> list:
        return self.report.benchmarks

    @property
    def performance_metrics(self) -> dict[str, Any]:
        return self.report.performance_metrics

    # ── Display ──────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a rich console summary."""
        self._formatter.print_summary(self.report)

    # ── Persistence ──────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save results to a JSON file."""
        path = Path(path)
        if path.suffix == ".csv":
            self._formatter.save_csv(self.report, path)
        else:
            self._formatter.save_json(self.report, path)
        logger.info("Results saved to: %s", path)

    def to_html(self, path: str | Path) -> None:
        """Generate and save an HTML report."""
        self._html_gen.save(self.report, path)

    def to_dict(self) -> dict[str, Any]:
        """Return the full report as a dictionary."""
        return self.report.to_dict()


class Leo:
    """
    Primary interface for Leo benchmarking.

    Usage::

        engine = Leo(
            model="meta-llama/Llama-3.1-8B",
            benchmarks=["mmlu", "hellaswag", "arc_challenge"],
            device="auto",
        )
        results = engine.run()
        results.save("results/llama3.json")
        results.to_html("reports/llama3.html")
    """

    def __init__(
        self,
        model: str | ModelConfig | None = None,
        benchmarks: list[str] | None = None,
        suite: str | None = None,
        device: str = "auto",
        *,
        config: LeoConfig | None = None,
        backend: str = "auto",
        dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
        use_flash_attention: bool = True,
        batch_size: int | str = "auto",
        num_fewshot: int | None = None,
        limit: int | None = None,
        output_dir: str = "results",
        generate_html: bool = True,
        seed: int = 42,
        verbosity: str = "INFO",
        peft_model: str | None = None,
        max_length: int | None = None,
        cache_dir: str | None = None,
        extra_model_args: dict[str, Any] | None = None,
    ) -> None:
        # If a full LeoConfig was provided, use it directly
        if config is not None:
            self.config = config
        else:
            # Build model config
            if isinstance(model, ModelConfig):
                model_cfg = model
            else:
                from leo.core.types import ModelBackend, QuantizationType

                model_cfg = ModelConfig(
                    model_name_or_path=model or "",
                    backend=ModelBackend.from_string(backend),
                    dtype=dtype,
                    device=device,
                    device_map="auto" if device == "auto" else device,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    trust_remote_code=trust_remote_code,
                    use_flash_attention=use_flash_attention,
                    batch_size=batch_size,
                    peft_model=peft_model,
                    max_length=max_length,
                    extra_model_args=extra_model_args or {},
                )

            benchmark_cfg = BenchmarkConfig(
                benchmarks=benchmarks or [],
                suite=suite,
                num_fewshot=num_fewshot,
                limit=limit,
            )

            reporting_cfg = ReportingConfig(
                output_dir=output_dir,
                generate_html=generate_html,
            )

            self.config = LeoConfig(
                model=model_cfg,
                benchmark=benchmark_cfg,
                reporting=reporting_cfg,
                seed=seed,
                verbosity=verbosity,
                cache_dir=cache_dir,
            )

        setup_logging(self.config.verbosity)

        self._model: BaseModel | None = None
        self._scorer = Scorer()
        self._formatter = ResultsFormatter()

    # ── Class methods ────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: LeoConfig) -> Leo:
        """Create a Leo engine from a full configuration object."""
        return cls(config=config)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Leo:
        """Create a Leo engine from a YAML configuration file."""
        config = LeoConfig.from_yaml(path)
        return cls(config=config)

    # ── Model management ─────────────────────────────────────────

    def load_model(self) -> BaseModel:
        """Explicitly load the model. Called automatically by run() if needed."""
        if self._model is None:
            logger.info("Loading model: %s", self.config.model.model_name_or_path)
            self._model = ModelLoader.load(self.config.model)
            logger.info(
                "Model loaded: %s (%.1fM parameters)",
                self.config.model.model_name_or_path,
                self._model.get_num_parameters() / 1e6,
            )
        return self._model

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            self._model.close()
            self._model = None
            logger.info("Model unloaded")

            # Try to reclaim GPU memory
            try:
                import torch
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    # ── Run ──────────────────────────────────────────────────────

    def run(self) -> LeoResults:
        """
        Execute all configured benchmarks and return the results.

        This is the main entry point. It:
        1. Collects system information
        2. Loads the model if needed for performance benchmarks
        3. Runs all standard + performance benchmarks
        4. Scores and aggregates results
        5. Returns a ``LeoResults`` object
        """
        logger.info(
            "Starting Leo benchmark run for: %s",
            self.config.model.model_name_or_path,
        )
        start_time = datetime.datetime.now(datetime.timezone.utc)

        # Collect system info
        try:
            sys_info = get_system_info()
            sys_info_dict = {
                "platform": sys_info.platform,
                "python_version": sys_info.python_version,
                "torch_version": sys_info.torch_version,
                "total_ram_gb": round(sys_info.total_ram_gb, 1),
                "cpu_count": sys_info.cpu_count,
                "gpus": [
                    {
                        "name": gpu.name,
                        "memory_total_mb": gpu.memory_total_mb,
                        "compute_capability": gpu.compute_capability,
                    }
                    for gpu in sys_info.gpus
                ],
            }
        except Exception:
            sys_info_dict = {}

        # Determine if we need a loaded model (for performance benchmarks)
        needs_model = (
            self.config.benchmark.suite == "performance"
            or "performance" in self.config.benchmark.benchmarks
        )

        model = None
        if needs_model:
            model = self.load_model()

        # Run benchmarks
        runner = BenchmarkRunner(
            model_config=self.config.model,
            benchmark_config=self.config.benchmark,
            reporting_config=self.config.reporting,
            seed=self.config.seed,
            cache_dir=self.config.cache_dir,
        )

        raw_results = runner.run(model=model)

        # Score results
        report = self._scorer.score(
            model_name=self.config.model.model_name_or_path,
            results=raw_results,
            system_info=sys_info_dict,
            config=self.config.to_dict(),
        )

        end_time = datetime.datetime.now(datetime.timezone.utc)
        elapsed = (end_time - start_time).total_seconds()

        logger.info(
            "Benchmark run completed in %.1f seconds — composite score: %.4f",
            elapsed,
            report.composite_score,
        )

        # Print summary
        results = LeoResults(report=report, raw_results=raw_results)
        results.print_summary()

        # Auto-save if output_dir is configured
        output_dir = Path(self.config.reporting.output_dir)
        model_slug = (
            self.config.model.model_name_or_path.replace("/", "_").replace("\\", "_")
        )
        ts = start_time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{model_slug}_{ts}"

        results.save(output_dir / f"{base_name}.json")

        if self.config.reporting.generate_html:
            results.to_html(output_dir / f"{base_name}.html")

        return results

    # ── Comparison ───────────────────────────────────────────────

    @staticmethod
    def compare(*paths: str | Path) -> ResultsComparator:
        """
        Compare results across multiple saved JSON files.

        Usage::

            comparator = Leo.compare("results/a.json", "results/b.json")
            comparator.print_comparison()
        """
        comp = ResultsComparator()
        for p in paths:
            comp.load_report(p)
        return comp

    # ── Context manager ──────────────────────────────────────────

    def __enter__(self) -> Leo:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.unload_model()

    def __repr__(self) -> str:
        return (
            f"Leo(model={self.config.model.model_name_or_path!r}, "
            f"benchmarks={self.config.benchmark.benchmarks!r}, "
            f"suite={self.config.benchmark.suite!r})"
        )
