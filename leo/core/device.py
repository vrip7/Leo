# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""Device detection, selection, and hardware information utilities."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import psutil
import torch

logger = logging.getLogger("leo.device")


@dataclass
class GPUInfo:
    """Information about a single GPU."""

    index: int
    name: str
    total_memory_mb: int
    free_memory_mb: int
    used_memory_mb: int
    compute_capability: tuple[int, int] | None = None
    temperature_c: int | None = None
    utilization_pct: int | None = None


@dataclass
class SystemInfo:
    """Complete system hardware information."""

    cpu_count: int
    cpu_freq_mhz: float | None
    ram_total_mb: int
    ram_available_mb: int
    gpus: list[GPUInfo]
    cuda_available: bool
    cuda_version: str | None
    torch_version: str
    platform: str

    def to_dict(self) -> dict[str, Any]:
        import dataclasses

        result = dataclasses.asdict(self)
        return result

    @property
    def total_gpu_memory_mb(self) -> int:
        return sum(g.total_memory_mb for g in self.gpus)

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    def summary(self) -> str:
        lines = [
            f"Platform: {self.platform}",
            f"CPU: {self.cpu_count} cores",
            f"RAM: {self.ram_total_mb / 1024:.1f} GB total, {self.ram_available_mb / 1024:.1f} GB available",
            f"PyTorch: {self.torch_version}",
            f"CUDA: {self.cuda_version or 'Not available'}",
            f"GPUs: {self.gpu_count}",
        ]
        for gpu in self.gpus:
            lines.append(
                f"  [{gpu.index}] {gpu.name} — "
                f"{gpu.total_memory_mb / 1024:.1f} GB total, "
                f"{gpu.free_memory_mb / 1024:.1f} GB free"
            )
        return "\n".join(lines)


def detect_gpus() -> list[GPUInfo]:
    """Enumerate available NVIDIA GPUs using pynvml when available, falling back to torch."""
    gpus: list[GPUInfo] = []

    if not torch.cuda.is_available():
        return gpus

    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                temp = None
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except pynvml.NVMLError:
                gpu_util = None

            cc = torch.cuda.get_device_capability(i)
            gpus.append(
                GPUInfo(
                    index=i,
                    name=name,
                    total_memory_mb=mem_info.total // (1024 * 1024),
                    free_memory_mb=mem_info.free // (1024 * 1024),
                    used_memory_mb=mem_info.used // (1024 * 1024),
                    compute_capability=cc,
                    temperature_c=temp,
                    utilization_pct=gpu_util,
                )
            )
        pynvml.nvmlShutdown()
    except Exception:
        logger.debug("pynvml unavailable, falling back to torch.cuda for GPU detection")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory // (1024 * 1024)
            free = total  # Approximation when pynvml is not available
            gpus.append(
                GPUInfo(
                    index=i,
                    name=props.name,
                    total_memory_mb=total,
                    free_memory_mb=free,
                    used_memory_mb=0,
                    compute_capability=(props.major, props.minor),
                )
            )

    return gpus


def get_system_info() -> SystemInfo:
    """Gather comprehensive system information."""
    import platform as _platform

    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    gpus = detect_gpus()

    cuda_version: str | None = None
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda

    return SystemInfo(
        cpu_count=psutil.cpu_count(logical=True) or os.cpu_count() or 1,
        cpu_freq_mhz=cpu_freq.current if cpu_freq else None,
        ram_total_mb=mem.total // (1024 * 1024),
        ram_available_mb=mem.available // (1024 * 1024),
        gpus=gpus,
        cuda_available=torch.cuda.is_available(),
        cuda_version=cuda_version,
        torch_version=torch.__version__,
        platform=_platform.platform(),
    )


def resolve_device(requested: str = "auto") -> torch.device:
    """Determine the best available device, respecting the user's preference."""
    req = requested.strip().lower()

    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(req)

    if req == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")

    return torch.device("cpu")


def estimate_model_memory_mb(
    num_params: int,
    dtype: str = "float16",
    quantization: str = "none",
) -> float:
    """Estimate peak memory requirement for a model in MB."""
    bytes_per_param = {
        "float32": 4.0,
        "float16": 2.0,
        "bfloat16": 2.0,
        "int8": 1.0,
        "int4": 0.5,
    }

    if quantization in ("int4", "gptq", "awq", "gguf_q4"):
        bpp = 0.5
    elif quantization == "int8":
        bpp = 1.0
    else:
        bpp = bytes_per_param.get(dtype, 2.0)

    # Model weights + ~20% overhead for activations/KV cache
    weight_mb = (num_params * bpp) / (1024 * 1024)
    return weight_mb * 1.2
