# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""Abstract base class for all model backends."""

from __future__ import annotations

import abc
from typing import Any, Optional

import torch

from leo.core.config import ModelConfig
from leo.utils.logging import get_logger

logger = get_logger("models.base")


class BaseModel(abc.ABC):
    """
    Abstract base for model wrappers.

    Every backend (HuggingFace, Unsloth, vLLM, GGUF) subclasses this to provide
    a uniform interface for loading, generating, and computing log-likelihoods.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Optional[torch.device] = None
        self._loaded = False

    @property
    def model_name(self) -> str:
        return self.config.model_name_or_path

    @property
    def device(self) -> torch.device:
        if self._device is None:
            from leo.core.device import resolve_device

            self._device = resolve_device(self.config.device)
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Lifecycle ────────────────────────────────────────────────

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights and tokenizer into memory."""
        ...

    def unload(self) -> None:
        """Release model from memory and free GPU resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded: %s", self.model_name)

    # ── Inference ────────────────────────────────────────────────

    @abc.abstractmethod
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate completions for a batch of prompts."""
        ...

    @abc.abstractmethod
    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[tuple[float, bool]]:
        """
        Compute log-likelihoods for (context, continuation) pairs.

        Returns list of (log_prob, is_greedy) tuples.
        """
        ...

    @abc.abstractmethod
    def loglikelihood_rolling(self, requests: list[str]) -> list[float]:
        """Compute rolling log-likelihood (perplexity) for each text."""
        ...

    # ── Info ─────────────────────────────────────────────────────

    @abc.abstractmethod
    def get_num_parameters(self) -> int:
        """Return total number of model parameters."""
        ...

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the loaded model."""
        return {
            "model_name": self.model_name,
            "backend": self.config.backend.value,
            "parameters": self.get_num_parameters() if self._loaded else None,
            "dtype": self.config.dtype,
            "quantization": self.config.quantization.value,
            "device": str(self.device),
            "loaded": self._loaded,
        }

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<{self.__class__.__name__}({self.model_name}, {status})>"

    def __enter__(self) -> BaseModel:
        self.load()
        return self

    def __exit__(self, *args: Any) -> None:
        self.unload()
