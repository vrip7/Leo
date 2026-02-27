# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""
Metric computation utilities.

Most metrics are computed by lm-evaluation-harness internally.
This module provides additional metric aggregation, normalization,
and composite scoring functions used by Leo's reporting layer.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from leo.core.registry import register_metric
from leo.utils.logging import get_logger

logger = get_logger("evaluation.metrics")


# ── Standard aggregation metrics ─────────────────────────────────────

@register_metric("accuracy")
def accuracy(predictions: Sequence[Any], references: Sequence[Any]) -> float:
    """Simple accuracy: fraction of exact matches."""
    if len(predictions) == 0:
        return 0.0
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(predictions)


@register_metric("exact_match")
def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Exact string match after normalization."""
    if len(predictions) == 0:
        return 0.0

    def _normalize(s: str) -> str:
        return s.strip().lower()

    correct = sum(
        1 for p, r in zip(predictions, references) if _normalize(p) == _normalize(r)
    )
    return correct / len(predictions)


@register_metric("f1")
def f1_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Token-level F1 score (used in QA tasks like SQuAD, DROP)."""
    if len(predictions) == 0:
        return 0.0

    scores: list[float] = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        common = set(pred_tokens) & set(ref_tokens)
        if len(common) == 0:
            scores.append(0.0)
            continue

        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0

        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))

    return sum(scores) / len(scores)


@register_metric("perplexity")
def perplexity(log_likelihoods: Sequence[float], token_counts: Sequence[int]) -> float:
    """Compute perplexity from log-likelihoods and token counts."""
    total_ll = sum(log_likelihoods)
    total_tokens = sum(token_counts)
    if total_tokens == 0:
        return float("inf")
    avg_nll = -total_ll / total_tokens
    return math.exp(avg_nll)


@register_metric("bits_per_byte")
def bits_per_byte(log_likelihoods: Sequence[float], byte_counts: Sequence[int]) -> float:
    """Compute bits per byte."""
    total_ll = sum(log_likelihoods)
    total_bytes = sum(byte_counts)
    if total_bytes == 0:
        return float("inf")
    return -total_ll / (total_bytes * math.log(2))


@register_metric("pass_at_k")
def pass_at_k(
    num_samples: int,
    num_correct: int,
    k: int = 1,
) -> float:
    """
    Estimator for pass@k metric used in code generation benchmarks.
    Uses the unbiased estimator from the Codex paper.
    """
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


# ── Composite scoring ────────────────────────────────────────────────

def compute_composite_score(
    results: dict[str, dict[str, float]],
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute a weighted composite score across multiple benchmarks.

    Args:
        results: Dict mapping benchmark_name → {metric_name: value}
        weights: Optional dict mapping benchmark_name → weight (default: equal)

    Returns:
        A single composite score between 0 and 1.
    """
    if not results:
        return 0.0

    if weights is None:
        weights = {name: 1.0 for name in results}

    total_weight = sum(weights.get(name, 1.0) for name in results)
    if total_weight == 0:
        return 0.0

    weighted_sum = 0.0
    for name, metrics in results.items():
        w = weights.get(name, 1.0)
        # Use the primary metric (first one, usually 'acc' or 'exact_match')
        primary_value = next(iter(metrics.values()), 0.0) if metrics else 0.0
        weighted_sum += primary_value * w

    return weighted_sum / total_weight


def compute_normalized_score(
    value: float,
    baseline: float = 0.25,  # Random chance for 4-choice MC
    ceiling: float = 1.0,
) -> float:
    """
    Normalize a metric value to [0, 1] accounting for random baseline.

    Useful for comparing across benchmarks with different chance levels.
    """
    if ceiling <= baseline:
        return 0.0
    normalized = (value - baseline) / (ceiling - baseline)
    return max(0.0, min(1.0, normalized))


# ── Statistical utilities ────────────────────────────────────────────

def confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a confidence interval using bootstrap."""
    arr = np.array(values)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)

    mean = float(np.mean(arr))
    if n == 1:
        return (mean, mean)

    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    from scipy import stats

    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)
    return (mean - z * se, mean + z * se)


def standard_error(values: Sequence[float]) -> float:
    """Compute standard error of the mean."""
    arr = np.array(values)
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
