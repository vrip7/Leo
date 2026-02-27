# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""Model and dataset download utilities with caching."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, hf_hub_download
from filelock import FileLock

from leo.utils.logging import get_logger

logger = get_logger("download")

_DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "leo")


def get_cache_dir(override: Optional[str] = None) -> Path:
    """Resolve the Leo cache directory."""
    cache = Path(override) if override else Path(os.environ.get("LEO_CACHE_DIR", _DEFAULT_CACHE))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def download_model(
    model_name_or_path: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
) -> str:
    """
    Ensure a model is downloaded and cached. Returns the local path.

    If model_name_or_path is already a local directory, returns it as-is.
    """
    if os.path.isdir(model_name_or_path):
        logger.info("Using local model directory: %s", model_name_or_path)
        return model_name_or_path

    cache = get_cache_dir(cache_dir)
    model_cache = cache / "models"
    model_cache.mkdir(parents=True, exist_ok=True)

    # Use a file lock to prevent concurrent downloads of the same model
    lock_name = hashlib.sha256(model_name_or_path.encode()).hexdigest()[:16]
    lock_path = model_cache / f".{lock_name}.lock"

    with FileLock(str(lock_path), timeout=3600):
        logger.info("Downloading/verifying model: %s", model_name_or_path)
        local_dir = snapshot_download(
            repo_id=model_name_or_path,
            revision=revision,
            cache_dir=str(model_cache),
            local_dir_use_symlinks=True,
        )
        logger.info("Model ready at: %s", local_dir)
        return local_dir


def download_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Ensure a dataset is available, return cache path."""
    cache = get_cache_dir(cache_dir)
    dataset_cache = cache / "datasets"
    dataset_cache.mkdir(parents=True, exist_ok=True)

    # Datasets library handles caching internally
    os.environ.setdefault("HF_DATASETS_CACHE", str(dataset_cache))
    logger.info("Dataset cache directory: %s", dataset_cache)
    return str(dataset_cache)
