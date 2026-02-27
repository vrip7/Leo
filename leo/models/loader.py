# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""Unified model loader with auto-detection of the best backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from leo.core.config import ModelConfig
from leo.core.registry import get_model_backend, list_model_backends
from leo.core.types import ModelBackend
from leo.models.base import BaseModel
from leo.utils.logging import get_logger

logger = get_logger("models.loader")

# Force registration of all backends by importing them
import leo.models.hf_model  # noqa: F401
import leo.models.unsloth_model  # noqa: F401
import leo.models.vllm_model  # noqa: F401
import leo.models.gguf_model  # noqa: F401


class ModelLoader:
    """
    Factory that selects and instantiates the right model backend
    based on configuration and what is available on the system.
    """

    @staticmethod
    def detect_backend(config: ModelConfig) -> ModelBackend:
        """
        Auto-detect the best backend for a given model configuration.

        Priority:
        1. Explicit backend in config
        2. GGUF file → gguf backend
        3. Unsloth available + 4bit → unsloth
        4. vLLM available + multi-GPU → vllm
        5. Default to huggingface
        """
        if config.backend != ModelBackend.AUTO:
            return config.backend

        model_path = config.model_name_or_path

        # Check for GGUF files
        path = Path(model_path)
        if path.suffix == ".gguf" or (path.is_dir() and any(path.glob("*.gguf"))):
            logger.info("Auto-detected GGUF model, using gguf backend")
            return ModelBackend.GGUF

        # If 4-bit requested, prefer Unsloth if available
        if config.load_in_4bit:
            try:
                import unsloth  # noqa: F401

                logger.info("Auto-detected: 4-bit requested + Unsloth available → using unsloth backend")
                return ModelBackend.UNSLOTH
            except ImportError:
                pass

        # If multiple GPUs available and model is large, prefer vLLM
        try:
            import torch

            if torch.cuda.device_count() > 1:
                try:
                    import vllm  # noqa: F401

                    logger.info("Auto-detected: multi-GPU + vLLM available → using vllm backend")
                    return ModelBackend.VLLM
                except ImportError:
                    pass
        except ImportError:
            pass

        # Default to HuggingFace transformers
        return ModelBackend.HUGGINGFACE

    @classmethod
    def load(cls, config: ModelConfig) -> BaseModel:
        """
        Detect the backend and instantiate the model.

        Returns a loaded BaseModel ready for inference.
        """
        backend = cls.detect_backend(config)
        backend_name = backend.value

        logger.info("Selected backend: %s", backend_name)

        backend_cls = get_model_backend(backend_name)
        model = backend_cls(config)
        model.load()
        return model

    @classmethod
    def create(cls, config: ModelConfig) -> BaseModel:
        """
        Create a model instance without loading it.

        Useful when you need to configure before loading.
        """
        backend = cls.detect_backend(config)
        backend_cls = get_model_backend(backend.value)
        return backend_cls(config)

    @staticmethod
    def available_backends() -> list[str]:
        """List all registered and importable backends."""
        return list_model_backends()
