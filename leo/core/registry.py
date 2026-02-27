# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""Global registry for model backends and benchmark providers."""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("leo.registry")

# ── Model backend registry ──────────────────────────────────────────

_MODEL_BACKENDS: dict[str, type] = {}


def register_model_backend(name: str) -> Callable:
    """Decorator to register a model backend class."""

    def decorator(cls: type) -> type:
        if name in _MODEL_BACKENDS:
            logger.warning("Overwriting model backend '%s'", name)
        _MODEL_BACKENDS[name] = cls
        return cls

    return decorator


def get_model_backend(name: str) -> type:
    """Retrieve a registered model backend by name."""
    if name not in _MODEL_BACKENDS:
        available = ", ".join(sorted(_MODEL_BACKENDS.keys()))
        raise KeyError(f"Model backend '{name}' not found. Available: {available}")
    return _MODEL_BACKENDS[name]


def list_model_backends() -> list[str]:
    return sorted(_MODEL_BACKENDS.keys())


# ── Benchmark registry ──────────────────────────────────────────────

_BENCHMARK_PROVIDERS: dict[str, type] = {}


def register_benchmark(name: str) -> Callable:
    """Decorator to register a benchmark provider class."""

    def decorator(cls: type) -> type:
        if name in _BENCHMARK_PROVIDERS:
            logger.warning("Overwriting benchmark provider '%s'", name)
        _BENCHMARK_PROVIDERS[name] = cls
        return cls

    return decorator


def get_benchmark_provider(name: str) -> type:
    """Retrieve a registered benchmark provider by name."""
    if name not in _BENCHMARK_PROVIDERS:
        available = ", ".join(sorted(_BENCHMARK_PROVIDERS.keys()))
        raise KeyError(f"Benchmark provider '{name}' not found. Available: {available}")
    return _BENCHMARK_PROVIDERS[name]


def list_benchmark_providers() -> list[str]:
    return sorted(_BENCHMARK_PROVIDERS.keys())


# ── Metric registry ─────────────────────────────────────────────────

_METRICS: dict[str, Callable] = {}


def register_metric(name: str) -> Callable:
    """Decorator to register a metric function."""

    def decorator(fn: Callable) -> Callable:
        _METRICS[name] = fn
        return fn

    return decorator


def get_metric(name: str) -> Callable:
    if name not in _METRICS:
        raise KeyError(f"Metric '{name}' not registered. Available: {', '.join(sorted(_METRICS.keys()))}")
    return _METRICS[name]


def list_metrics() -> list[str]:
    return sorted(_METRICS.keys())
