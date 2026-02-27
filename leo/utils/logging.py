# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""Structured logging for Leo with rich formatting."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

LEO_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "benchmark": "bold magenta",
        "model": "bold blue",
        "metric": "green",
    }
)

_console: Optional[Console] = None
_initialized = False


def get_console() -> Console:
    """Return the shared rich Console instance."""
    global _console
    if _console is None:
        _console = Console(theme=LEO_THEME, stderr=True)
    return _console


def setup_logging(verbosity: str = "INFO") -> None:
    """Configure root and Leo loggers with rich handler."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    level = getattr(logging, verbosity.upper(), logging.INFO)
    console = get_console()

    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    rich_handler.setLevel(level)

    fmt = logging.Formatter("%(message)s", datefmt="[%X]")
    rich_handler.setFormatter(fmt)

    # Configure leo-specific logger
    leo_logger = logging.getLogger("leo")
    leo_logger.setLevel(level)
    leo_logger.addHandler(rich_handler)
    leo_logger.propagate = False

    # Suppress noisy third-party loggers
    for noisy in ("transformers", "datasets", "tokenizers", "accelerate", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Also configure lm_eval logger if present
    lm_eval_logger = logging.getLogger("lm-eval")
    lm_eval_logger.setLevel(level)
    lm_eval_logger.addHandler(rich_handler)
    lm_eval_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'leo' namespace."""
    return logging.getLogger(f"leo.{name}")
