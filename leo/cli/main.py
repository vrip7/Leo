# Leo — AI Model Benchmarking Platform
# Copyright 2024-2026 Pradyumn Tandon / VRIP7

"""
Click-based CLI for Leo.

Commands:
    leo run         — run benchmarks on a model
    leo list        — list available benchmarks, suites, or backends
    leo compare     — compare results across multiple models
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(package_name="leo-bench")
def cli() -> None:
    """Leo — AI Model Benchmarking Platform.

    Evaluate any language model across 60+ benchmarks with a single command.
    Built by Pradyumn Tandon (https://pradyumntandon.com) at VRIP7 (https://vrip7.com).
    """


# ═══════════════════════════════════════════════════════════════════
# leo run
# ═══════════════════════════════════════════════════════════════════


@cli.command()
@click.option(
    "--model", "-m",
    required=True,
    help="Model name or path (e.g. 'meta-llama/Llama-3.1-8B').",
)
@click.option(
    "--benchmarks", "-b",
    default=None,
    help="Comma-separated list of benchmarks (e.g. 'mmlu,hellaswag').",
)
@click.option(
    "--suite", "-s",
    default=None,
    help="Predefined benchmark suite (e.g. 'leaderboard', 'quick', 'performance').",
)
@click.option(
    "--backend",
    default="auto",
    type=click.Choice(["auto", "huggingface", "unsloth", "vllm", "gguf"], case_sensitive=False),
    help="Model backend (default: auto-detect).",
)
@click.option("--device", default="auto", help="Device to use (auto, cuda, cpu, mps).")
@click.option("--dtype", default="auto", help="Data type (auto, float16, bfloat16, float32).")
@click.option("--load-in-4bit", is_flag=True, help="Load model in 4-bit quantization.")
@click.option("--load-in-8bit", is_flag=True, help="Load model in 8-bit quantization.")
@click.option("--batch-size", default="auto", help="Batch size for evaluation.")
@click.option("--num-fewshot", default=None, type=int, help="Number of few-shot examples.")
@click.option("--limit", default=None, type=int, help="Limit number of examples per benchmark.")
@click.option("--output-dir", "-o", default="results", help="Output directory for results.")
@click.option("--no-html", is_flag=True, help="Disable HTML report generation.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--verbosity", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
@click.option("--peft-model", default=None, help="Path to PEFT/LoRA adapter.")
@click.option("--max-length", default=None, type=int, help="Maximum sequence length.")
@click.option("--trust-remote-code/--no-trust-remote-code", default=True, help="Trust remote code from HF Hub.")
@click.option("--flash-attention/--no-flash-attention", default=True, help="Use Flash Attention 2.")
@click.option("--cache-dir", default=None, help="Cache directory for downloads.")
@click.option("--config", "config_file", default=None, type=click.Path(exists=True), help="YAML config file (overrides CLI options).")
def run(
    model: str,
    benchmarks: Optional[str],
    suite: Optional[str],
    backend: str,
    device: str,
    dtype: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
    batch_size: str,
    num_fewshot: Optional[int],
    limit: Optional[int],
    output_dir: str,
    no_html: bool,
    seed: int,
    verbosity: str,
    peft_model: Optional[str],
    max_length: Optional[int],
    trust_remote_code: bool,
    flash_attention: bool,
    cache_dir: Optional[str],
    config_file: Optional[str],
) -> None:
    """Run benchmarks on a model."""
    from leo.engine import Leo

    if config_file:
        engine = Leo.from_yaml(config_file)
    else:
        benchmark_list = [b.strip() for b in benchmarks.split(",")] if benchmarks else []

        if not benchmark_list and not suite:
            console.print(
                "[yellow]Warning:[/yellow] No benchmarks or suite specified. "
                "Use --benchmarks or --suite. Defaulting to 'quick' suite."
            )
            suite = "quick"

        engine = Leo(
            model=model,
            benchmarks=benchmark_list,
            suite=suite,
            device=device,
            backend=backend,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            trust_remote_code=trust_remote_code,
            use_flash_attention=flash_attention,
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=limit,
            output_dir=output_dir,
            generate_html=not no_html,
            seed=seed,
            verbosity=verbosity,
            peft_model=peft_model,
            max_length=max_length,
            cache_dir=cache_dir,
        )

    try:
        results = engine.run()
        console.print(
            f"\n[bold green]Done![/bold green] Composite score: "
            f"[bold]{results.composite_score:.4f}[/bold]"
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc
    finally:
        engine.unload_model()


# ═══════════════════════════════════════════════════════════════════
# leo list
# ═══════════════════════════════════════════════════════════════════


@cli.group("list")
def list_cmd() -> None:
    """List available benchmarks, suites, or backends."""


@list_cmd.command("benchmarks")
@click.option(
    "--category", "-c", default=None,
    help="Filter by category (knowledge, reasoning, math, code, language, multilingual, safety, efficiency).",
)
def list_benchmarks(category: Optional[str]) -> None:
    """List all available benchmarks."""
    from leo.benchmarks.registry import BENCHMARK_REGISTRY
    from leo.core.types import BenchmarkCategory

    table = Table(title="Available Benchmarks", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Display Name")
    table.add_column("Category")
    table.add_column("Tasks")
    table.add_column("Metrics")

    for name, info in sorted(BENCHMARK_REGISTRY.items()):
        if category:
            try:
                cat = BenchmarkCategory(category)
                if info.category != cat:
                    continue
            except ValueError:
                # Partial match on category name
                if category.lower() not in info.category.value.lower():
                    continue

        task_str = ", ".join(info.task_names[:3])
        if len(info.task_names) > 3:
            task_str += f" (+{len(info.task_names) - 3})"

        metric_str = ", ".join(info.metrics[:3]) if info.metrics else "auto"

        table.add_row(name, info.display_name, info.category.value, task_str, metric_str)

    console.print(table)
    console.print(f"\n[dim]Total: {len(BENCHMARK_REGISTRY)} benchmarks[/dim]")


@list_cmd.command("suites")
def list_suites() -> None:
    """List all available benchmark suites."""
    from leo.benchmarks.registry import SUITE_REGISTRY

    table = Table(title="Available Suites", show_header=True, header_style="bold cyan")
    table.add_column("Suite", style="bold")
    table.add_column("Benchmarks", max_width=80)

    for name, tasks in sorted(SUITE_REGISTRY.items()):
        task_str = ", ".join(tasks[:6])
        if len(tasks) > 6:
            task_str += f" (+{len(tasks) - 6} more)"
        table.add_row(name, task_str)

    console.print(table)


@list_cmd.command("backends")
def list_backends() -> None:
    """List all available model backends."""
    from leo.models.loader import ModelLoader

    table = Table(title="Available Backends", show_header=True, header_style="bold cyan")
    table.add_column("Backend", style="bold")
    table.add_column("Status")

    backends = {
        "huggingface": "transformers",
        "unsloth": "unsloth",
        "vllm": "vllm",
        "gguf": "llama_cpp",
    }

    for backend_name, pkg_name in backends.items():
        try:
            __import__(pkg_name)
            status = "[green]available[/green]"
        except ImportError:
            status = "[dim]not installed[/dim]"
        table.add_row(backend_name, status)

    console.print(table)


# ═══════════════════════════════════════════════════════════════════
# leo compare
# ═══════════════════════════════════════════════════════════════════


@cli.command()
@click.argument("result_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Save comparison to JSON file.")
def compare(result_files: tuple[str, ...], output: Optional[str]) -> None:
    """Compare results across multiple models.

    Provide two or more JSON result files produced by 'leo run'.
    """
    from leo.reporting.comparator import ResultsComparator

    if len(result_files) < 2:
        console.print("[red]Error:[/red] At least 2 result files required for comparison.")
        raise SystemExit(1)

    comp = ResultsComparator()
    for path in result_files:
        try:
            comp.load_report(path)
        except Exception as exc:
            console.print(f"[red]Error loading {path}:[/red] {exc}")
            raise SystemExit(1) from exc

    result = comp.compare()
    comp.print_comparison(result)

    if output:
        comp.save_comparison(output, result)
        console.print(f"\n[dim]Comparison saved to {output}[/dim]")


# ═══════════════════════════════════════════════════════════════════
# leo info
# ═══════════════════════════════════════════════════════════════════


@cli.command()
def info() -> None:
    """Show system information and Leo configuration."""
    from leo import __version__
    from leo.core.device import get_system_info

    sys_info = get_system_info()

    console.print(f"[bold cyan]Leo[/bold cyan] v{__version__}")
    console.print(f"  Platform: {sys_info.platform}")
    console.print(f"  Python:   {sys_info.python_version}")
    console.print(f"  PyTorch:  {sys_info.torch_version}")
    console.print(f"  RAM:      {sys_info.total_ram_gb:.1f} GB")
    console.print(f"  CPUs:     {sys_info.cpu_count}")

    if sys_info.gpus:
        console.print(f"  GPUs:     {len(sys_info.gpus)}")
        for gpu in sys_info.gpus:
            console.print(
                f"    [green]{gpu.name}[/green] — "
                f"{gpu.memory_total_mb / 1024:.1f} GB, "
                f"CC {gpu.compute_capability}"
            )
    else:
        console.print("  GPUs:     [dim]none detected[/dim]")

    # Backend availability
    console.print("\n[bold]Backends:[/bold]")
    for pkg, label in [("transformers", "HuggingFace"), ("unsloth", "Unsloth"),
                       ("vllm", "vLLM"), ("llama_cpp", "GGUF")]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            console.print(f"  {label}: [green]{ver}[/green]")
        except ImportError:
            console.print(f"  {label}: [dim]not installed[/dim]")


if __name__ == "__main__":
    cli()
