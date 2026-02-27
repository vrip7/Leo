# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""vLLM backend for high-throughput batched inference."""

from __future__ import annotations

from typing import Any

import torch

from leo.core.config import ModelConfig
from leo.core.registry import register_model_backend
from leo.models.base import BaseModel
from leo.utils.logging import get_logger

logger = get_logger("models.vllm")


@register_model_backend("vllm")
class VLLMModel(BaseModel):
    """
    vLLM backend — optimized for high throughput with PagedAttention,
    continuous batching, and tensor parallelism.

    Requires: pip install leo-benchmarks[vllm]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._sampling_params_cls: Any = None

    def load(self) -> None:
        try:
            from vllm import LLM, SamplingParams

            self._sampling_params_cls = SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed. Install with: pip install leo-benchmarks[vllm]"
            ) from e

        logger.info("Loading model via vLLM: %s", self.model_name)

        dtype_map = {
            "float16": "float16",
            "bfloat16": "bfloat16",
            "float32": "float32",
            "auto": "auto",
        }
        dtype = dtype_map.get(self.config.dtype.lower(), "auto")

        # Determine tensor parallel size from available GPUs
        tp_size = 1
        if torch.cuda.is_available():
            tp_size = self.config.extra_model_args.get(
                "tensor_parallel_size", torch.cuda.device_count()
            )

        vllm_kwargs: dict[str, Any] = {
            "model": self.config.model_name_or_path,
            "dtype": dtype,
            "trust_remote_code": self.config.trust_remote_code,
            "tensor_parallel_size": tp_size,
        }

        if self.config.max_length:
            vllm_kwargs["max_model_len"] = self.config.max_length

        if self.config.revision:
            vllm_kwargs["revision"] = self.config.revision

        gpu_mem_util = self.config.extra_model_args.get("gpu_memory_utilization", 0.9)
        vllm_kwargs["gpu_memory_utilization"] = gpu_mem_util

        if self.config.quantization.value not in ("none",):
            vllm_kwargs["quantization"] = self.config.quantization.value

        # Merge any extra vllm-specific args
        for k in ("seed", "swap_space", "enforce_eager", "max_num_seqs"):
            if k in self.config.extra_model_args:
                vllm_kwargs[k] = self.config.extra_model_args[k]

        self._model = LLM(**vllm_kwargs)
        self._loaded = True
        logger.info("vLLM model loaded: %s (tp=%d)", self.model_name, tp_size)

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

        sampling_params = self._sampling_params_cls(
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0,
            top_p=top_p,
            stop=stop_sequences or [],
        )

        outputs = self._model.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[tuple[float, bool]]:
        """
        Compute log-likelihoods using vLLM's prompt_logprobs feature.
        """
        assert self._loaded, "Model must be loaded"

        results: list[tuple[float, bool]] = []

        for context, continuation in requests:
            full_text = context + continuation

            sampling_params = self._sampling_params_cls(
                max_tokens=1,
                temperature=0,
                prompt_logprobs=0,
            )

            outputs = self._model.generate([full_text], sampling_params)
            output = outputs[0]

            if output.prompt_logprobs is None:
                results.append((0.0, False))
                continue

            # Sum the log-probs corresponding to the continuation
            tokenizer = self._model.get_tokenizer()
            ctx_tokens = tokenizer.encode(context, add_special_tokens=False)
            full_tokens = tokenizer.encode(full_text, add_special_tokens=True)

            ctx_len = len(ctx_tokens)
            # Account for BOS if present
            if len(full_tokens) > len(ctx_tokens) + len(tokenizer.encode(continuation, add_special_tokens=False)):
                ctx_len += 1

            total_lp = 0.0
            is_greedy = True
            for i in range(ctx_len, len(full_tokens)):
                if i < len(output.prompt_logprobs) and output.prompt_logprobs[i] is not None:
                    token_id = full_tokens[i]
                    logprob_info = output.prompt_logprobs[i]
                    if token_id in logprob_info:
                        lp = logprob_info[token_id].logprob
                        total_lp += lp
                    else:
                        is_greedy = False

            results.append((total_lp, is_greedy))

        return results

    def loglikelihood_rolling(self, requests: list[str]) -> list[float]:
        assert self._loaded, "Model must be loaded"

        results: list[float] = []
        for text in requests:
            sampling_params = self._sampling_params_cls(
                max_tokens=1,
                temperature=0,
                prompt_logprobs=0,
            )
            outputs = self._model.generate([text], sampling_params)
            output = outputs[0]

            total_lp = 0.0
            if output.prompt_logprobs:
                for i, lp_dict in enumerate(output.prompt_logprobs):
                    if lp_dict is not None and i > 0:
                        tokenizer = self._model.get_tokenizer()
                        tokens = tokenizer.encode(text, add_special_tokens=True)
                        if i < len(tokens):
                            token_id = tokens[i]
                            if token_id in lp_dict:
                                total_lp += lp_dict[token_id].logprob

            results.append(total_lp)

        return results

    def get_num_parameters(self) -> int:
        # vLLM doesn't expose parameter count directly in a standard way
        if self._model is None:
            return 0
        try:
            config = self._model.llm_engine.model_config.hf_config
            if hasattr(config, "num_parameters"):
                return config.num_parameters
            # Estimate from hidden_size and num_layers
            h = getattr(config, "hidden_size", 4096)
            l = getattr(config, "num_hidden_layers", 32)
            v = getattr(config, "vocab_size", 32000)
            return h * h * 4 * l + v * h * 2  # Rough estimate
        except Exception:
            return 0
