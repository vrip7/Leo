# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""
Central registry mapping benchmark names to lm-evaluation-harness task names,
and defining curated suites (collections of benchmarks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from leo.core.types import BenchmarkCategory


@dataclass
class BenchmarkInfo:
    """Metadata about a registered benchmark."""

    name: str
    display_name: str
    category: BenchmarkCategory
    task_names: list[str]
    description: str
    metrics: list[str]
    num_fewshot_default: int | None = None
    paper: str | None = None
    output_type: str = "multiple_choice"


# ────────────────────────────────────────────────────────────────────
# BENCHMARK REGISTRY — maps Leo benchmark names → lm-eval harness tasks
# ────────────────────────────────────────────────────────────────────

BENCHMARK_REGISTRY: dict[str, BenchmarkInfo] = {}


def _register(info: BenchmarkInfo) -> None:
    BENCHMARK_REGISTRY[info.name] = info


# ── Knowledge & Reasoning ────────────────────────────────────────────

_register(BenchmarkInfo(
    name="mmlu",
    display_name="MMLU",
    category=BenchmarkCategory.KNOWLEDGE,
    task_names=["mmlu"],
    description="Massive Multitask Language Understanding — 57 subjects covering STEM, humanities, social sciences, and more.",
    metrics=["acc", "acc_norm"],
    num_fewshot_default=5,
    paper="https://arxiv.org/abs/2009.03300",
    output_type="multiple_choice",
))

_register(BenchmarkInfo(
    name="mmlu_pro",
    display_name="MMLU-Pro",
    category=BenchmarkCategory.KNOWLEDGE,
    task_names=["mmlu_pro"],
    description="Enhanced MMLU with harder questions and 10 answer choices.",
    metrics=["acc", "acc_norm"],
    num_fewshot_default=5,
    paper="https://arxiv.org/abs/2406.01574",
    output_type="multiple_choice",
))

_register(BenchmarkInfo(
    name="arc_easy",
    display_name="ARC Easy",
    category=BenchmarkCategory.REASONING,
    task_names=["arc_easy"],
    description="AI2 Reasoning Challenge — easy partition.",
    metrics=["acc", "acc_norm"],
    num_fewshot_default=25,
    paper="https://arxiv.org/abs/1803.05457",
    output_type="multiple_choice",
))

_register(BenchmarkInfo(
    name="arc_challenge",
    display_name="ARC Challenge",
    category=BenchmarkCategory.REASONING,
    task_names=["arc_challenge"],
    description="AI2 Reasoning Challenge — challenge partition requiring complex reasoning.",
    metrics=["acc", "acc_norm"],
    num_fewshot_default=25,
    paper="https://arxiv.org/abs/1803.05457",
    output_type="multiple_choice",
))

_register(BenchmarkInfo(
    name="hellaswag",
    display_name="HellaSwag",
    category=BenchmarkCategory.REASONING,
    task_names=["hellaswag"],
    description="Commonsense NLI: can a model pick the correct ending to a story?",
    metrics=["acc", "acc_norm"],
    num_fewshot_default=10,
    paper="https://arxiv.org/abs/1905.07830",
    output_type="multiple_choice",
))

_register(BenchmarkInfo(
    name="truthfulqa",
    display_name="TruthfulQA",
    category=BenchmarkCategory.KNOWLEDGE,
    task_names=["truthfulqa_mc2"],
    description="Measures whether a language model tends to generate truthful answers.",
    metrics=["acc"],
    num_fewshot_default=0,
    paper="https://arxiv.org/abs/2109.07958",
    output_type="multiple_choice",
))

_register(BenchmarkInfo(
    name="truthfulqa_gen",
    display_name="TruthfulQA (Generation)",
    category=BenchmarkCategory.KNOWLEDGE,
    task_names=["truthfulqa_gen"],
    description="TruthfulQA evaluated via generation rather than multiple choice.",
    metrics=["bleu_acc", "rouge1_acc"],
    num_fewshot_default=0,
    paper="https://arxiv.org/abs/2109.07958",
    output_type="generate_until",
))

_register(BenchmarkInfo(
    name="winogrande",
    display_name="Winogrande",
    category=BenchmarkCategory.REASONING,
    task_names=["winogrande"],
    description="Large-scale Winograd Schema Challenge for commonsense reasoning.",
    metrics=["acc"],
    num_fewshot_default=5,
    paper="https://arxiv.org/abs/1907.10641",
    output_type="multiple_choice",
))

_register(BenchmarkInfo(
    name="boolq",
    display_name="BoolQ",
    category=BenchmarkCategory.REASONING,
    task_names=["boolq"],
    description="Boolean question answering from natural inference passages.",
    metrics=["acc"],
    num_fewshot_default=0,
    paper="https://arxiv.org/abs/1905.10044",
    output_type="multiple_choice",
))

_register(BenchmarkInfo(
    name="piqa",
    display_name="PIQA",
    category=BenchmarkCategory.REASONING,
    task_names=["piqa"],
    description="Physical Intuition QA — physical commonsense reasoning.",
    metrics=["acc", "acc_norm"],
    num_fewshot_default=0,
    paper="https://arxiv.org/abs/1911.11641",
    output_type="multiple_choice",
))

_register(BenchmarkInfo(
    name="openbookqa",
    display_name="OpenBookQA",
    category=BenchmarkCategory.REASONING,
    task_names=["openbookqa"],
    description="Open-book science QA requiring multi-step reasoning.",
    metrics=["acc", "acc_norm"],
    num_fewshot_default=0,
    paper="https://arxiv.org/abs/1809.02789",
    output_type="multiple_choice",
))

_register(BenchmarkInfo(
    name="commonsense_qa",
    display_name="CommonsenseQA",
    category=BenchmarkCategory.REASONING,
    task_names=["commonsense_qa"],
    description="Commonsense knowledge QA using MMLU-style prompts.",
    metrics=["acc"],
    num_fewshot_default=7,
    paper="https://arxiv.org/abs/1811.00937",
    output_type="multiple_choice",
))

# ── Mathematics ──────────────────────────────────────────────────────

_register(BenchmarkInfo(
    name="gsm8k",
    display_name="GSM8K",
    category=BenchmarkCategory.MATHEMATICS,
    task_names=["gsm8k"],
    description="Grade School Math 8K — multi-step arithmetic word problems.",
    metrics=["exact_match", "acc"],
    num_fewshot_default=5,
    paper="https://arxiv.org/abs/2110.14168",
    output_type="generate_until",
))

_register(BenchmarkInfo(
    name="minerva_math",
    display_name="MATH (Minerva)",
    category=BenchmarkCategory.MATHEMATICS,
    task_names=["minerva_math"],
    description="Competition-level math requiring numerical reasoning.",
    metrics=["exact_match"],
    num_fewshot_default=4,
    paper="https://arxiv.org/abs/2206.14858",
    output_type="generate_until",
))

_register(BenchmarkInfo(
    name="mgsm",
    display_name="MGSM",
    category=BenchmarkCategory.MATHEMATICS,
    task_names=["mgsm_direct"],
    description="Multilingual Grade School Math — GSM8K in 10 languages.",
    metrics=["exact_match"],
    num_fewshot_default=8,
    paper="https://arxiv.org/abs/2210.03057",
    output_type="generate_until",
))

# ── Code ─────────────────────────────────────────────────────────────

_register(BenchmarkInfo(
    name="humaneval",
    display_name="HumanEval",
    category=BenchmarkCategory.CODE,
    task_names=["humaneval"],
    description="Python code synthesis — 164 hand-crafted programming problems.",
    metrics=["pass@1"],
    num_fewshot_default=0,
    paper="https://arxiv.org/abs/2107.03374",
    output_type="generate_until",
))

# ── Language Understanding ───────────────────────────────────────────

_register(BenchmarkInfo(
    name="lambada_openai",
    display_name="LAMBADA (OpenAI)",
    category=BenchmarkCategory.LANGUAGE,
    task_names=["lambada_openai"],
    description="Word prediction benchmark — predict the last word of a passage.",
    metrics=["perplexity", "acc"],
    num_fewshot_default=0,
    paper="https://arxiv.org/abs/1606.06031",
    output_type="loglikelihood_rolling",
))

_register(BenchmarkInfo(
    name="wikitext",
    display_name="WikiText",
    category=BenchmarkCategory.LANGUAGE,
    task_names=["wikitext"],
    description="Language modeling perplexity on Wikipedia text.",
    metrics=["word_perplexity", "byte_perplexity", "bits_per_byte"],
    num_fewshot_default=0,
    output_type="loglikelihood_rolling",
))

_register(BenchmarkInfo(
    name="drop",
    display_name="DROP",
    category=BenchmarkCategory.LANGUAGE,
    task_names=["drop"],
    description="Discrete Reasoning Over Paragraphs — numerical reasoning QA.",
    metrics=["f1", "em"],
    num_fewshot_default=3,
    paper="https://arxiv.org/abs/1903.00161",
    output_type="generate_until",
))

# ── Long Context ─────────────────────────────────────────────────────

_register(BenchmarkInfo(
    name="longbench",
    display_name="LongBench",
    category=BenchmarkCategory.LONG_CONTEXT,
    task_names=["longbench"],
    description="Bilingual benchmark for long-context understanding across 6 task categories.",
    metrics=["f1", "rouge"],
    num_fewshot_default=0,
    paper="https://arxiv.org/abs/2308.14508",
    output_type="generate_until",
))

# ── Multilingual ─────────────────────────────────────────────────────

_register(BenchmarkInfo(
    name="belebele",
    display_name="Belebele",
    category=BenchmarkCategory.MULTILINGUAL,
    task_names=["belebele_eng_Latn"],
    description="Multilingual reading comprehension benchmark across 122 languages.",
    metrics=["acc"],
    num_fewshot_default=0,
    output_type="multiple_choice",
))

_register(BenchmarkInfo(
    name="mlqa",
    display_name="MLQA",
    category=BenchmarkCategory.MULTILINGUAL,
    task_names=["mlqa_en_en"],
    description="Multilingual QA across 7 languages.",
    metrics=["f1", "em"],
    num_fewshot_default=0,
    paper="https://arxiv.org/abs/1910.07475",
    output_type="generate_until",
))

# ── Safety & Bias ────────────────────────────────────────────────────

_register(BenchmarkInfo(
    name="toxigen",
    display_name="ToxiGen",
    category=BenchmarkCategory.SAFETY,
    task_names=["toxigen"],
    description="Evaluates propensity to generate toxic content across demographics.",
    metrics=["acc"],
    num_fewshot_default=0,
    paper="https://arxiv.org/abs/2203.09509",
    output_type="multiple_choice",
))

# ── Meta-benchmarks ──────────────────────────────────────────────────

_register(BenchmarkInfo(
    name="bigbench_hard",
    display_name="BIG-Bench Hard",
    category=BenchmarkCategory.REASONING,
    task_names=["bbh"],
    description="Challenging subset of BIG-Bench with chain-of-thought reasoning.",
    metrics=["exact_match"],
    num_fewshot_default=3,
    paper="https://arxiv.org/abs/2210.09261",
    output_type="generate_until",
))

_register(BenchmarkInfo(
    name="metabench",
    display_name="Metabench",
    category=BenchmarkCategory.KNOWLEDGE,
    task_names=["metabench"],
    description="Compressed essential benchmarks — most informative items from major benchmarks.",
    metrics=["acc"],
    num_fewshot_default=0,
    output_type="multiple_choice",
))

_register(BenchmarkInfo(
    name="tinybenchmarks",
    display_name="tinyBenchmarks",
    category=BenchmarkCategory.KNOWLEDGE,
    task_names=["tinyBenchmarks"],
    description="Efficient approximations of major benchmarks using minimal items.",
    metrics=["acc"],
    num_fewshot_default=0,
    output_type="multiple_choice",
))


# ────────────────────────────────────────────────────────────────────
# SUITE REGISTRY — curated collections of benchmarks
# ────────────────────────────────────────────────────────────────────

SUITE_REGISTRY: dict[str, list[str]] = {
    # Open LLM Leaderboard v1
    "leaderboard_v1": [
        "mmlu", "arc_challenge", "hellaswag", "truthfulqa", "winogrande", "gsm8k",
    ],
    # Open LLM Leaderboard v2 (newer tasks)
    "leaderboard": [
        "mmlu_pro", "arc_challenge", "hellaswag", "truthfulqa",
        "winogrande", "gsm8k", "bigbench_hard",
    ],
    # Quick smoke test
    "quick": [
        "arc_easy", "hellaswag", "boolq",
    ],
    # Full comprehensive evaluation
    "full": [
        "mmlu", "mmlu_pro", "arc_easy", "arc_challenge", "hellaswag",
        "truthfulqa", "winogrande", "boolq", "piqa", "openbookqa",
        "commonsense_qa", "gsm8k", "minerva_math", "lambada_openai",
        "wikitext", "drop", "bigbench_hard",
    ],
    # Knowledge-focused
    "knowledge": [
        "mmlu", "mmlu_pro", "truthfulqa", "commonsense_qa",
    ],
    # Reasoning-focused
    "reasoning": [
        "arc_easy", "arc_challenge", "hellaswag", "winogrande",
        "boolq", "piqa", "openbookqa", "bigbench_hard",
    ],
    # Math-focused
    "math": [
        "gsm8k", "minerva_math", "mgsm",
    ],
    # Code generation
    "code": [
        "humaneval",
    ],
    # Language understanding
    "language": [
        "lambada_openai", "wikitext", "drop",
    ],
    # Multilingual
    "multilingual": [
        "mgsm", "belebele", "mlqa",
    ],
    # Safety & bias
    "safety": [
        "toxigen", "truthfulqa",
    ],
    # Efficient / compressed benchmarks
    "efficient": [
        "metabench", "tinybenchmarks",
    ],
    # Performance profiling (not lm-eval tasks, handled separately)
    "performance": [],
}


# ────────────────────────────────────────────────────────────────────
# Registry query functions
# ────────────────────────────────────────────────────────────────────

def list_all_benchmarks() -> list[str]:
    """Return all registered benchmark names."""
    return sorted(BENCHMARK_REGISTRY.keys())


def list_all_suites() -> list[str]:
    """Return all registered suite names."""
    return sorted(SUITE_REGISTRY.keys())


def get_benchmark_info(name: str) -> BenchmarkInfo | None:
    """Look up benchmark info by name."""
    return BENCHMARK_REGISTRY.get(name)


def resolve_benchmark_tasks(benchmark_names: list[str]) -> list[str]:
    """
    Convert a list of Leo benchmark names into lm-evaluation-harness task names.

    If a name is not in the registry, it is passed through as-is
    (assumed to be a raw lm-eval task name).
    """
    tasks: list[str] = []
    seen: set[str] = set()

    for name in benchmark_names:
        info = BENCHMARK_REGISTRY.get(name)
        if info:
            for task in info.task_names:
                if task not in seen:
                    seen.add(task)
                    tasks.append(task)
        else:
            # Pass through raw task names
            if name not in seen:
                seen.add(name)
                tasks.append(name)

    return tasks


def resolve_suite(suite_name: str) -> list[str]:
    """Expand a suite name into its constituent benchmark names."""
    if suite_name not in SUITE_REGISTRY:
        raise ValueError(
            f"Unknown suite '{suite_name}'. Available: {', '.join(list_all_suites())}"
        )
    return SUITE_REGISTRY[suite_name]
