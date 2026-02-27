# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""
Integration with EleutherAI lm-evaluation-harness for running standard benchmarks.

This module provides the bridge between Leo's configuration system and
the lm-evaluation-harness `simple_evaluate` API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from leo.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    resolve_benchmark_tasks,
)
from leo.core.config import BenchmarkConfig, ModelConfig, ReportingConfig
from leo.core.types import BenchmarkResult
from leo.utils.logging import get_logger

logger = get_logger("benchmarks.harness")


class HarnessRunner:
    """
    Wraps lm-evaluation-harness to execute benchmarks via its `simple_evaluate` API.

    This is the primary evaluation engine for all standard benchmarks (MMLU,
    ARC, HellaSwag, GSM8K, etc.).
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

    def _build_model_args_string(self) -> str:
        """Construct the comma-separated model_args string for lm-eval."""
        args_parts: list[str] = [
            f"pretrained={self.model_config.model_name_or_path}",
        ]

        if self.model_config.revision:
            args_parts.append(f"revision={self.model_config.revision}")

        if self.model_config.dtype != "auto":
            args_parts.append(f"dtype={self.model_config.dtype}")

        if self.model_config.trust_remote_code:
            args_parts.append("trust_remote_code=True")

        if self.model_config.load_in_4bit:
            args_parts.append("load_in_4bit=True")
        elif self.model_config.load_in_8bit:
            args_parts.append("load_in_8bit=True")

        if self.model_config.peft_model:
            args_parts.append(f"peft={self.model_config.peft_model}")

        if self.model_config.use_flash_attention:
            args_parts.append("attn_implementation=flash_attention_2")

        if self.model_config.device_map != "auto":
            args_parts.append(f"device_map={self.model_config.device_map}")
        else:
            args_parts.append("parallelize=True")

        if self.model_config.tokenizer_name:
            args_parts.append(f"tokenizer={self.model_config.tokenizer_name}")

        for k, v in self.model_config.extra_model_args.items():
            args_parts.append(f"{k}={v}")

        return ",".join(args_parts)

    def _resolve_model_type(self) -> str:
        """Map Leo's backend to lm-eval's model type string."""
        from leo.core.types import ModelBackend

        backend = self.model_config.backend
        if isinstance(backend, str):
            backend = ModelBackend.from_string(backend)

        model_type_map = {
            ModelBackend.HUGGINGFACE: "hf",
            ModelBackend.UNSLOTH: "hf",  # Unsloth patches produce HF-compatible models
            ModelBackend.VLLM: "vllm",
            ModelBackend.GGUF: "gguf",
            ModelBackend.AUTO: "hf",
        }
        return model_type_map.get(backend, "hf")

    def _resolve_tasks(self) -> list[str]:
        """Get final list of lm-eval task names from the benchmark config."""
        benchmark_names = self.benchmark_config.resolve_tasks()
        return resolve_benchmark_tasks(benchmark_names)

    def run(self) -> list[BenchmarkResult]:
        """
        Execute all configured benchmarks via lm-evaluation-harness.

        Returns a list of BenchmarkResult objects.
        """
        import lm_eval

        tasks = self._resolve_tasks()
        if not tasks:
            logger.warning("No tasks resolved to run. Check benchmark/suite configuration.")
            return []

        task_str = ",".join(tasks)
        logger.info("Running benchmarks: %s", task_str)

        model_type = self._resolve_model_type()
        model_args = self._build_model_args_string()

        logger.info("Model type: %s", model_type)
        logger.info("Model args: %s", model_args)

        # Build lm-eval arguments
        eval_kwargs: dict[str, Any] = {
            "model": model_type,
            "model_args": model_args,
            "tasks": tasks,
            "batch_size": self.model_config.batch_size,
            "seed": [self.seed, self.seed, self.seed],
            "log_samples": self.benchmark_config.log_samples,
        }

        if self.model_config.device != "auto":
            eval_kwargs["device"] = self.model_config.device

        if self.benchmark_config.num_fewshot is not None:
            eval_kwargs["num_fewshot"] = self.benchmark_config.num_fewshot

        if self.benchmark_config.limit is not None:
            eval_kwargs["limit"] = self.benchmark_config.limit

        if self.benchmark_config.generation_kwargs:
            eval_kwargs["gen_kwargs"] = ",".join(
                f"{k}={v}" for k, v in self.benchmark_config.generation_kwargs.items()
            )

        if self.cache_dir:
            eval_kwargs["cache_requests"] = "true"

        if self.benchmark_config.check_integrity:
            eval_kwargs["check_integrity"] = True

        # Execute evaluation
        logger.info("Starting lm-evaluation-harness evaluation...")
        results = lm_eval.simple_evaluate(**eval_kwargs)

        # Parse results into BenchmarkResult objects
        return self._parse_results(results, tasks)

    def _parse_results(
        self,
        raw_results: dict[str, Any],
        task_names: list[str],
    ) -> list[BenchmarkResult]:
        """Convert lm-eval's results dict into Leo BenchmarkResult objects."""
        parsed: list[BenchmarkResult] = []

        if "results" not in raw_results:
            logger.error("No 'results' key in lm-eval output")
            return parsed

        eval_results = raw_results["results"]

        for task_name in task_names:
            if task_name not in eval_results:
                # Try to find it as a subtask in group results
                found = False
                for key in eval_results:
                    if task_name in key or key in task_name:
                        task_data = eval_results[key]
                        found = True
                        metrics = self._extract_metrics(task_data)
                        samples = task_data.get("samples", 0)
                        parsed.append(BenchmarkResult(
                            benchmark_name=key,
                            metrics=metrics,
                            samples=samples,
                            metadata={
                                "task_name": key,
                                "alias": task_data.get("alias", key),
                            },
                        ))
                if not found:
                    logger.warning("Task '%s' not found in results", task_name)
                continue

            task_data = eval_results[task_name]
            metrics = self._extract_metrics(task_data)
            samples = task_data.get("samples", 0)

            # Map back to Leo benchmark name
            leo_name = task_name
            for bname, binfo in BENCHMARK_REGISTRY.items():
                if task_name in binfo.task_names:
                    leo_name = bname
                    break

            parsed.append(BenchmarkResult(
                benchmark_name=leo_name,
                metrics=metrics,
                samples=samples,
                metadata={
                    "task_name": task_name,
                    "alias": task_data.get("alias", task_name),
                },
            ))

        # Also capture any group-level results
        if "groups" in raw_results:
            for group_name, group_data in raw_results["groups"].items():
                if isinstance(group_data, dict):
                    metrics = self._extract_metrics(group_data)
                    if metrics:
                        parsed.append(BenchmarkResult(
                            benchmark_name=f"group:{group_name}",
                            metrics=metrics,
                            samples=0,
                            metadata={"type": "group"},
                        ))

        return parsed

    def _extract_metrics(self, task_data: dict[str, Any]) -> dict[str, float]:
        """Extract numeric metrics from a task result dict."""
        metrics: dict[str, float] = {}
        skip_keys = {"alias", "samples", "group", "task", "version"}

        for key, value in task_data.items():
            if key in skip_keys:
                continue
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
            elif isinstance(value, dict):
                # Some metrics are nested: {"acc,none": 0.75, "acc_stderr,none": 0.01}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        clean_key = sub_key.replace(",none", "").replace(",", "_")
                        metrics[clean_key] = float(sub_value)

        return metrics

    def save_raw_results(self, results: dict[str, Any], output_path: str) -> None:
        """Save raw lm-eval results to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Raw results saved to: %s", path)
