# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7
# Licensed under the Apache License, Version 2.0
# https://vrip7.com | https://pradyumntandon.com

"""
Leo: A comprehensive AI model benchmarking platform.

Evaluate any language model across 60+ benchmarks with a single command.
Supports HuggingFace Transformers, Unsloth, vLLM, GGUF, and more.
"""

__version__ = "1.0.0"
__author__ = "Pradyumn Tandon"
__author_url__ = "https://pradyumntandon.com"
__org__ = "VRIP7"
__org_url__ = "https://vrip7.com"
__license__ = "Apache-2.0"

from leo.core.config import LeoConfig, ModelConfig, BenchmarkConfig
from leo.core.types import ModelBackend, BenchmarkCategory, DeviceType
from leo.engine import Leo

__all__ = [
    "Leo",
    "LeoConfig",
    "ModelConfig",
    "BenchmarkConfig",
    "ModelBackend",
    "BenchmarkCategory",
    "DeviceType",
    "__version__",
]
