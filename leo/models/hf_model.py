# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""HuggingFace Transformers model backend — supports CausalLM, Seq2SeqLM,
quantization (GPTQ, AWQ, bitsandbytes), PEFT/LoRA adapters, and multi-GPU."""

from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from leo.core.config import ModelConfig
from leo.core.registry import register_model_backend
from leo.core.types import QuantizationType
from leo.models.base import BaseModel
from leo.utils.logging import get_logger

logger = get_logger("models.hf")


class StopOnSequences(StoppingCriteria):
    """Custom stopping criteria for stop sequences."""

    def __init__(self, stop_ids: list[list[int]], batch_size: int = 1) -> None:
        self.stop_ids = stop_ids
        self.batch_size = batch_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        for stop_seq in self.stop_ids:
            seq_len = len(stop_seq)
            if input_ids.shape[1] >= seq_len:
                for b in range(input_ids.shape[0]):
                    if input_ids[b, -seq_len:].tolist() == stop_seq:
                        return True
        return False


@register_model_backend("huggingface")
class HuggingFaceModel(BaseModel):
    """Full-featured HuggingFace Transformers backend."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._is_seq2seq = False

    def _resolve_dtype(self) -> torch.dtype:
        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        val = self.config.dtype.strip().lower()
        return dtype_map.get(val, "auto")

    def _build_quantization_config(self) -> BitsAndBytesConfig | None:
        q = self.config.quantization
        if q == QuantizationType.INT4 or self.config.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        if q == QuantizationType.INT8 or self.config.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def _load_tokenizer(self) -> None:
        tok_name = self.config.tokenizer_name or self.config.model_name_or_path
        self._tokenizer = AutoTokenizer.from_pretrained(
            tok_name,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.revision,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def _load_model(self) -> None:
        model_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.config.model_name_or_path,
            "trust_remote_code": self.config.trust_remote_code,
            "revision": self.config.revision,
        }

        # Dtype
        dtype = self._resolve_dtype()
        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = "auto"

        # Quantization
        bnb_config = self._build_quantization_config()
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        # Flash attention
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Device map
        model_kwargs["device_map"] = self.config.device_map

        if self.config.max_memory_per_gpu:
            import torch

            num_gpus = torch.cuda.device_count()
            max_mem = {i: self.config.max_memory_per_gpu for i in range(num_gpus)}
            model_kwargs["max_memory"] = max_mem

        if self.config.offload_folder:
            model_kwargs["offload_folder"] = self.config.offload_folder

        # Extra model args
        model_kwargs.update(self.config.extra_model_args)

        # Try CausalLM first, fall back to Seq2Seq
        try:
            self._model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            self._is_seq2seq = False
        except (ValueError, OSError):
            logger.info("CausalLM loading failed, trying Seq2SeqLM...")
            self._model = AutoModelForSeq2SeqLM.from_pretrained(**model_kwargs)
            self._is_seq2seq = True

        # Apply PEFT adapter if specified
        if self.config.peft_model:
            from peft import PeftModel

            logger.info("Loading PEFT adapter: %s", self.config.peft_model)
            self._model = PeftModel.from_pretrained(self._model, self.config.peft_model)

        self._model.eval()

    def load(self) -> None:
        logger.info("Loading HuggingFace model: %s", self.model_name)
        self._load_tokenizer()
        self._load_model()
        self._loaded = True
        num_params = self.get_num_parameters()
        logger.info(
            "Model loaded: %s (%.2fB params, dtype=%s, device=%s)",
            self.model_name,
            num_params / 1e9,
            self.config.dtype,
            self._model.device if hasattr(self._model, "device") else "multiple",
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
        }

        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        # Stop sequences
        if stop_sequences:
            stop_ids = [self._tokenizer.encode(s, add_special_tokens=False) for s in stop_sequences]
            stopping = StopOnSequences(stop_ids, batch_size=len(prompts))
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([stopping])

        gen_kwargs.update(kwargs)

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        # Decode only the generated portion
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, input_length:]
        decoded = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded

    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[tuple[float, bool]]:
        assert self._loaded, "Model must be loaded before computing log-likelihoods"

        results: list[tuple[float, bool]] = []

        for context, continuation in requests:
            full_text = context + continuation

            ctx_enc = self._tokenizer.encode(context, add_special_tokens=False)
            full_enc = self._tokenizer.encode(full_text, add_special_tokens=True)

            input_ids = torch.tensor([full_enc], device=self._model.device)

            with torch.inference_mode():
                outputs = self._model(input_ids)
                logits = outputs.logits

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

            # Only count continuation tokens
            ctx_len = len(ctx_enc)
            # Adjust for possible BOS token
            if full_enc[0] != ctx_enc[0] and len(full_enc) > len(ctx_enc):
                offset = len(full_enc) - len(ctx_enc) - len(self._tokenizer.encode(continuation, add_special_tokens=False))
            else:
                offset = ctx_len

            cont_start = max(offset - 1, 0)  # -1 because of shift
            cont_log_probs = []
            greedy_tokens = torch.argmax(shift_logits[:, cont_start:, :], dim=-1)
            actual_tokens = shift_labels[:, cont_start:]

            for i in range(actual_tokens.shape[1]):
                token_id = actual_tokens[0, i].item()
                lp = log_probs[0, cont_start + i, token_id].item()
                cont_log_probs.append(lp)

            total_log_prob = sum(cont_log_probs)
            is_greedy = bool(torch.all(greedy_tokens == actual_tokens).item())
            results.append((total_log_prob, is_greedy))

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
                token_id = shift_labels[0, i].item()
                total += log_probs[0, i, token_id].item()

            results.append(total)

        return results

    def get_num_parameters(self) -> int:
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters())
