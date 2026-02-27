# Leo — AI Model Benchmarking Platform

<p align="center">
  <strong>Evaluate any AI model across 60+ benchmarks with a single command.</strong>
</p>

<p align="center">
  <a href="https://github.com/vrip7/leo/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://pypi.org/project/leo-benchmarks/">
    <img src="https://img.shields.io/pypi/v/leo-benchmarks.svg" alt="PyPI">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python">
  </a>
</p>

---

**Leo** is a production-grade, open-source AI model evaluation platform built by [Pradyumn Tandon](https://pradyumntandon.com) from [VRIP7](https://vrip7.com). It provides a unified interface to benchmark any language model — from tiny models to frontier-scale — across every major evaluation suite.

## Features

- **60+ Benchmark Suites**: MMLU, ARC, HellaSwag, TruthfulQA, GSM8K, HumanEval, BigBench, GLUE, SuperGLUE, LongBench, and many more
- **Universal Model Support**: HuggingFace Transformers, Unsloth, vLLM, GGUF/llama.cpp, PEFT/LoRA adapters, quantized models (GPTQ, AWQ, bitsandbytes)
- **Hardware Aware**: Automatic device detection, multi-GPU support via Accelerate, memory optimization
- **Performance Profiling**: Latency, throughput, memory footprint, tokens/second tracking
- **Rich Reporting**: JSON, HTML, and comparison reports with full metric breakdowns
- **Production Ready**: Enterprise-grade logging, caching, checkpointing, and error recovery

## Quick Start

### Installation

```bash
pip install leo-benchmarks

# With optional backends
pip install leo-benchmarks[unsloth]   # Unsloth optimization
pip install leo-benchmarks[vllm]      # vLLM fast inference
pip install leo-benchmarks[gguf]      # GGUF/llama.cpp support
pip install leo-benchmarks[all]       # Everything
```

### Usage

```bash
# Run MMLU benchmark on a model
leo run --model meta-llama/Llama-3.1-8B --benchmarks mmlu

# Run multiple benchmarks
leo run --model mistralai/Mistral-7B-v0.3 --benchmarks mmlu,hellaswag,arc,truthfulqa

# Run the full Open LLM Leaderboard suite
leo run --model microsoft/phi-3-mini-4k-instruct --suite leaderboard

# Run with quantization
leo run --model meta-llama/Llama-3.1-70B --load-in-4bit --benchmarks mmlu

# Run performance benchmarks
leo run --model meta-llama/Llama-3.1-8B --suite performance

# List all available benchmarks
leo list benchmarks

# List all available suites
leo list suites

# Compare results
leo compare results/model_a.json results/model_b.json
```

### Python API

```python
from leo import Leo

engine = Leo(
    model="meta-llama/Llama-3.1-8B",
    benchmarks=["mmlu", "hellaswag", "arc_challenge"],
    device="auto",
)
results = engine.run()
results.save("results/llama3.json")
results.to_html("reports/llama3.html")
```

## Supported Benchmarks

### Knowledge & Reasoning
| Benchmark | Type | Description |
|-----------|------|-------------|
| MMLU | Multiple Choice | 57-subject massive multitask language understanding |
| MMLU-Pro | Multiple Choice | Enhanced MMLU with harder questions |
| ARC (Easy/Challenge) | Multiple Choice | AI2 Reasoning Challenge |
| HellaSwag | Multiple Choice | Commonsense NLI |
| TruthfulQA | MC / Generation | Truthfulness evaluation |
| Winogrande | Multiple Choice | Commonsense reasoning |
| BoolQ | Multiple Choice | Boolean question answering |
| PIQA | Multiple Choice | Physical intuition QA |
| OpenBookQA | Multiple Choice | Open-book science QA |
| CommonsenseQA | Multiple Choice | Commonsense knowledge QA |

### Mathematics
| Benchmark | Type | Description |
|-----------|------|-------------|
| GSM8K | Generation | Grade school math problems |
| MATH | Generation | Competition-level mathematics |
| MGSM | Generation | Multilingual grade school math |

### Code
| Benchmark | Type | Description |
|-----------|------|-------------|
| HumanEval | Generation | Python code synthesis |

### Language Understanding
| Benchmark | Type | Description |
|-----------|------|-------------|
| LAMBADA | Perplexity | Word prediction |
| GLUE | Mixed | General Language Understanding |
| SuperGLUE | Mixed | Advanced Language Understanding |

### Long Context
| Benchmark | Type | Description |
|-----------|------|-------------|
| LongBench | Mixed | Long context understanding |

### Safety & Bias
| Benchmark | Type | Description |
|-----------|------|-------------|
| ToxiGen | Classification | Toxic content detection |
| BBQ | Multiple Choice | Bias evaluation |

### Meta-benchmarks
| Benchmark | Type | Description |
|-----------|------|-------------|
| BigBench Hard | Mixed | 200+ diverse tasks |
| Metabench | Mixed | Compressed essential benchmarks |
| tinyBenchmarks | Mixed | Efficient approximations |

## Supported Model Backends

| Backend | Supported | Description |
|---------|-----------|-------------|
| HuggingFace Transformers | ✅ | AutoModelForCausalLM, Seq2Seq, quantized |
| Unsloth | ✅ | Optimized fine-tuning and inference |
| vLLM | ✅ | High-throughput serving engine |
| GGUF (llama.cpp) | ✅ | Quantized local models |
| PEFT/LoRA | ✅ | Adapter models |
| GPTQ | ✅ | GPTQ-quantized models |
| AWQ | ✅ | AWQ-quantized models |
| bitsandbytes | ✅ | 4-bit/8-bit quantization |

## Architecture

```
leo/
├── cli/           # Rich CLI interface
├── core/          # Configuration, device management, registry
├── models/        # Model backend implementations
├── benchmarks/    # Benchmark suite definitions & runner
│   └── suites/    # Individual benchmark categories
├── evaluation/    # Metrics computation & scoring
├── reporting/     # Result formatting, HTML reports, comparison
└── utils/         # Hardware info, logging, downloads
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

## License

Leo is released under the [Apache License 2.0](LICENSE).

## Author

Built with ❤️ by [Pradyumn Tandon](https://pradyumntandon.com) at [VRIP7](https://vrip7.com)

- Website: [https://pradyumntandon.com](https://pradyumntandon.com)
- Organization: [https://vrip7.com](https://vrip7.com)
- GitHub: [https://github.com/vrip7](https://github.com/vrip7)
