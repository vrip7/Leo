# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""Configuration dataclasses for Leo."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from leo.core.types import BenchmarkCategory, DeviceType, ModelBackend, QuantizationType


@dataclass
class ModelConfig:
    """Configuration for model loading."""

    model_name_or_path: str
    backend: ModelBackend = ModelBackend.AUTO
    revision: str | None = None
    dtype: str = "auto"
    device: str = "auto"
    device_map: str = "auto"
    quantization: QuantizationType = QuantizationType.NONE
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    max_memory_per_gpu: str | None = None
    offload_folder: str | None = None
    peft_model: str | None = None
    tokenizer_name: str | None = None
    max_length: int | None = None
    batch_size: int | str = "auto"
    extra_model_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.backend, str):
            self.backend = ModelBackend.from_string(self.backend)
        if isinstance(self.quantization, str):
            self.quantization = QuantizationType(self.quantization)
        if self.load_in_4bit:
            self.quantization = QuantizationType.INT4
        if self.load_in_8bit:
            self.quantization = QuantizationType.INT8


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    benchmarks: list[str] = field(default_factory=list)
    suite: str | None = None
    category: BenchmarkCategory | None = None
    num_fewshot: int | None = None
    limit: int | None = None
    tasks: list[str] = field(default_factory=list)
    custom_task_yaml: str | None = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    log_samples: bool = False
    check_integrity: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.category, str):
            self.category = BenchmarkCategory(self.category)

    def resolve_tasks(self) -> list[str]:
        """Return the final list of task names to run."""
        from leo.benchmarks.registry import SUITE_REGISTRY

        tasks: list[str] = list(self.tasks)

        if self.suite and self.suite in SUITE_REGISTRY:
            suite_tasks = SUITE_REGISTRY[self.suite]
            tasks.extend(suite_tasks)

        tasks.extend(self.benchmarks)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for t in tasks:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique


@dataclass
class ReportingConfig:
    """Configuration for results output."""

    output_dir: str = "results"
    output_format: str = "json"
    generate_html: bool = True
    save_samples: bool = False
    comparison_baseline: str | None = None

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class LeoConfig:
    """Top-level configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=lambda: ModelConfig(model_name_or_path=""))
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    seed: int = 42
    verbosity: str = "INFO"
    cache_dir: str | None = None
    num_workers: int = 4

    @classmethod
    def from_yaml(cls, path: str | Path) -> LeoConfig:
        """Load a configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LeoConfig:
        """Construct from a dictionary (e.g. parsed YAML or JSON)."""
        model_data = data.get("model", {})
        benchmark_data = data.get("benchmark", {})
        reporting_data = data.get("reporting", {})

        model_cfg = ModelConfig(**model_data) if model_data else ModelConfig(model_name_or_path="")
        benchmark_cfg = BenchmarkConfig(**benchmark_data) if benchmark_data else BenchmarkConfig()
        reporting_cfg = ReportingConfig(**reporting_data) if reporting_data else ReportingConfig()

        return cls(
            model=model_cfg,
            benchmark=benchmark_cfg,
            reporting=reporting_cfg,
            seed=data.get("seed", 42),
            verbosity=data.get("verbosity", "INFO"),
            cache_dir=data.get("cache_dir"),
            num_workers=data.get("num_workers", 4),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dict."""
        import dataclasses

        def _asdict_recursive(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _asdict_recursive(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, Enum):
                return obj.value
            return obj

        from enum import Enum

        return _asdict_recursive(self)

    def save_yaml(self, path: str | Path) -> None:
        """Write config to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
