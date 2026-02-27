# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""GGUF/llama.cpp backend for running quantized models locally."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from leo.core.config import ModelConfig
from leo.core.registry import register_model_backend
from leo.models.base import BaseModel
from leo.utils.logging import get_logger

logger = get_logger("models.gguf")


@register_model_backend("gguf")
class GGUFModel(BaseModel):
    """
    GGUF backend via llama-cpp-python for running quantized models on CPU/GPU.

    Supports GGUF Q4, Q5, Q8 quantized models via llama.cpp.

    Requires: pip install leo-benchmarks[gguf]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

    def _resolve_model_path(self) -> str:
        """Find the .gguf file, either direct path or discover in directory."""
        path = Path(self.config.model_name_or_path)
        if path.is_file() and path.suffix == ".gguf":
            return str(path)
        if path.is_dir():
            gguf_files = list(path.glob("*.gguf"))
            if gguf_files:
                # Prefer the largest file (usually the main model)
                chosen = max(gguf_files, key=lambda f: f.stat().st_size)
                logger.info("Auto-selected GGUF file: %s", chosen.name)
                return str(chosen)
            raise FileNotFoundError(f"No .gguf files found in {path}")
        # Assume it's a HuggingFace repo — try downloading
        from leo.utils.download import download_model

        local = download_model(self.config.model_name_or_path, cache_dir=None)
        return self._resolve_model_path_from_dir(local)

    def _resolve_model_path_from_dir(self, directory: str) -> str:
        d = Path(directory)
        gguf_files = list(d.glob("*.gguf"))
        if gguf_files:
            return str(max(gguf_files, key=lambda f: f.stat().st_size))
        raise FileNotFoundError(f"No .gguf files found in {directory}")

    def load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is not installed. Install with: pip install leo-benchmarks[gguf]"
            ) from e

        model_path = self._resolve_model_path()
        logger.info("Loading GGUF model: %s", model_path)

        n_ctx = self.config.max_length or 4096
        n_gpu_layers = self.config.extra_model_args.get("n_gpu_layers", -1)  # -1 = all layers on GPU
        n_threads = self.config.extra_model_args.get("n_threads", None)

        llama_kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
        }
        if n_threads:
            llama_kwargs["n_threads"] = n_threads

        self._model = Llama(**llama_kwargs)
        self._loaded = True
        logger.info("GGUF model loaded: %s (context=%d)", model_path, n_ctx)

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        assert self._loaded, "Model must be loaded before generation"

        results: list[str] = []
        for prompt in prompts:
            output = self._model(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.0,
                top_p=top_p,
                stop=stop_sequences or [],
                echo=False,
            )
            text = output["choices"][0]["text"]
            results.append(text)

        return results

    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[tuple[float, bool]]:
        assert self._loaded, "Model must be loaded"

        results: list[tuple[float, bool]] = []
        for context, continuation in requests:
            full_text = context + continuation

            # Tokenize
            ctx_tokens = self._model.tokenize(context.encode("utf-8"), add_bos=False)
            full_tokens = self._model.tokenize(full_text.encode("utf-8"), add_bos=True)

            # Evaluate full sequence
            self._model.reset()
            self._model.eval(full_tokens)

            # Get logits and compute log-probabilities for continuation tokens
            import numpy as np

            ctx_len = len(ctx_tokens)
            # Account for BOS
            offset = len(full_tokens) - len(ctx_tokens) - len(
                self._model.tokenize(continuation.encode("utf-8"), add_bos=False)
            )
            offset = max(offset, 0)
            cont_start = offset + ctx_len

            total_lp = 0.0
            is_greedy = True

            scores = self._model.scores
            if scores is not None:
                for i in range(cont_start, len(full_tokens)):
                    if i - 1 < len(scores):
                        logits = np.array(scores[i - 1])
                        log_probs = logits - np.logaddexp.reduce(logits)
                        token_id = full_tokens[i]
                        total_lp += log_probs[token_id]
                        if np.argmax(logits) != token_id:
                            is_greedy = False
            results.append((total_lp, is_greedy))

        return results

    def loglikelihood_rolling(self, requests: list[str]) -> list[float]:
        assert self._loaded, "Model must be loaded"
        import numpy as np

        results: list[float] = []
        for text in requests:
            tokens = self._model.tokenize(text.encode("utf-8"), add_bos=True)
            self._model.reset()
            self._model.eval(tokens)

            total_lp = 0.0
            scores = self._model.scores
            if scores is not None:
                for i in range(1, len(tokens)):
                    if i - 1 < len(scores):
                        logits = np.array(scores[i - 1])
                        log_probs = logits - np.logaddexp.reduce(logits)
                        total_lp += log_probs[tokens[i]]

            results.append(total_lp)

        return results

    def get_num_parameters(self) -> int:
        if self._model is None:
            return 0
        try:
            metadata = self._model.metadata
            if metadata and "general.parameter_count" in metadata:
                return int(metadata["general.parameter_count"])
        except Exception:
            pass
        return 0
