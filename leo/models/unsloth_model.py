# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""Unsloth-optimized model backend for faster inference with lower memory."""

from __future__ import annotations

from typing import Any

import torch

from leo.core.config import ModelConfig
from leo.core.registry import register_model_backend
from leo.models.base import BaseModel
from leo.utils.logging import get_logger

logger = get_logger("models.unsloth")


@register_model_backend("unsloth")
class UnslothModel(BaseModel):
    """
    Unsloth backend — provides 2-5× faster inference and 60-80% less memory
    by patching HuggingFace models with optimized kernels.

    Requires: pip install leo-benchmarks[unsloth]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

    def load(self) -> None:
        try:
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise ImportError(
                "Unsloth is not installed. Install with: pip install leo-benchmarks[unsloth]"
            ) from e

        logger.info("Loading model via Unsloth: %s", self.model_name)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": None,
        }
        dtype = dtype_map.get(self.config.dtype.lower(), None)

        load_in_4bit = self.config.load_in_4bit
        max_seq_length = self.config.max_length or 4096

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name_or_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Enable faster inference mode
        FastLanguageModel.for_inference(self._model)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._loaded = True
        num_params = self.get_num_parameters()
        logger.info(
            "Unsloth model loaded: %s (%.2fB params, 4bit=%s)",
            self.model_name,
            num_params / 1e9,
            load_in_4bit,
        )

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

        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self._model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "use_cache": True,
        }

        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        gen_kwargs.update(kwargs)

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, input_length:]
        return self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[tuple[float, bool]]:
        assert self._loaded, "Model must be loaded"

        results: list[tuple[float, bool]] = []
        for context, continuation in requests:
            full_text = context + continuation
            ctx_enc = self._tokenizer.encode(context, add_special_tokens=False)
            full_enc = self._tokenizer.encode(full_text, add_special_tokens=True)

            input_ids = torch.tensor([full_enc], device=self._model.device)

            with torch.inference_mode():
                outputs = self._model(input_ids)
                logits = outputs.logits

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

            ctx_len = len(ctx_enc)
            cont_start = max(ctx_len - 1, 0)
            greedy_tokens = torch.argmax(shift_logits[:, cont_start:, :], dim=-1)
            actual_tokens = shift_labels[:, cont_start:]

            total_lp = 0.0
            for i in range(actual_tokens.shape[1]):
                token_id = actual_tokens[0, i].item()
                total_lp += log_probs[0, cont_start + i, token_id].item()

            is_greedy = bool(torch.all(greedy_tokens == actual_tokens).item())
            results.append((total_lp, is_greedy))

        return results

    def loglikelihood_rolling(self, requests: list[str]) -> list[float]:
        assert self._loaded, "Model must be loaded"

        results: list[float] = []
        for text in requests:
            enc = self._tokenizer.encode(text, add_special_tokens=True)
            input_ids = torch.tensor([enc], device=self._model.device)

            with torch.inference_mode():
                outputs = self._model(input_ids)
                logits = outputs.logits

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

            total = 0.0
            for i in range(shift_labels.shape[1]):
                total += log_probs[0, i, shift_labels[0, i].item()].item()
            results.append(total)

        return results

    def get_num_parameters(self) -> int:
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters())
