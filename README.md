# FSI Stock Analysis

AI-powered stock analysis combining real-time market data, technical indicators, and LLM-generated insights. Supports multiple inference backends with zero Python changes — switch models and runtimes by editing one config file.

## Features

- **Real-time stock data** via Yahoo Finance, with disk cache so repeat runs skip re-fetching
- **Technical indicators**: SMA, RSI, momentum, price/SMA comparison
- **LLM analysis**: buy/sell/hold recommendations tailored to investor type (Conservative, Moderate, Aggressive, Day Trader)
- **Multi-stock**: comma-separate symbols in the UI to analyze a portfolio in one pass
- **Benchmark infrastructure**: programmatic testing across concurrency levels, saves JSON results
- **Flexible inference**: Ollama (CPU/GPU), vLLM, or TensorRT-LLM — configured via `config.toml`

---

## Quick Start

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and sync dependencies

```bash
git clone https://github.com/russfellows/fsi-stock-analysis.git
cd fsi-stock-analysis
uv sync
```

### 3. Install Ollama and pull a model

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &   # start the daemon if not already running

# Pull your preferred model (pick one)
ollama pull qwen3.5:4b        # fast, ~3.4 GB — recommended for CPU
ollama pull llama3.1:8b       # general-purpose, ~4.7 GB
ollama pull llama3.1:70b      # high quality, requires GPU or large RAM
```

### 4. Configure the runtime

Edit `config.toml` to match the model you pulled:

```toml
[llm]
runtime      = "ollama"
ollama_model = "qwen3.5:4b"   # change to llama3.1:8b, etc.
```

### 5. Launch the web UI

```bash
uv run FSI_StockAnalysis.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## Configuration (`config.toml`)

All runtime options live in `config.toml`. No Python edits are needed to switch models or backends.

```toml
[llm]
# Which inference runtime to use: "ollama" | "vllm" | "tensorrt"
runtime = "ollama"

# Ollama model tag — used when runtime = "ollama"
# Pull with: ollama pull <tag>
ollama_model = "qwen3.5:4b"

# HuggingFace model ID — used when runtime = "vllm" or "tensorrt"
# Models are cached in ~/.cache/huggingface/hub/ on first run.
hf_model = "Qwen/Qwen3.5-4B"

# API base for vLLM/TensorRT (OpenAI-compatible endpoint)
api_base_url = "http://localhost:8000/v1"

temperature = 0.3
max_tokens  = 2048

[app]
host = "0.0.0.0"   # "127.0.0.1" for localhost only
port = 7860

default_start_date = "2024-08-13"
default_end_date   = "2025-08-13"

[cache]
stock_data_dir = "~/.cache/fsi-stock-analysis/stock_data"
```

---

## Switching Models and Inference Stacks

### Ollama (CPU or GPU)

Ollama is the easiest path — no GPU required, works on any machine.

```toml
[llm]
runtime      = "ollama"
ollama_model = "qwen3.5:4b"    # fast, good quality
```

Other Ollama models to try:

| Model tag | Size | Notes |
|---|---|---|
| `qwen3.5:4b` | 3.4 GB | Fast on CPU; recommended default |
| `qwen3.5:8b` | 6.2 GB | Better quality, still CPU-feasible |
| `llama3.1:8b` | 4.7 GB | Meta's general-purpose model |
| `llama3.1:70b` | 40 GB | High quality; needs GPU or large RAM |
| `mistral:7b` | 4.1 GB | Good instruction-following |
| `phi4:14b` | 8.5 GB | Microsoft; strong reasoning |

Pull any model before use:

```bash
ollama pull llama3.1:8b
```

Update `config.toml` to switch — no restart of the app needed if you restart the Python process:

```toml
ollama_model = "llama3.1:8b"
```

### vLLM (NVIDIA or AMD GPU)

vLLM exposes an OpenAI-compatible API. Start the server, then point `config.toml` at it.

```bash
# NVIDIA
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-4B \
    --port 8000

# AMD ROCm
pip install vllm-rocm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-4B \
    --port 8000
```

`config.toml`:

```toml
[llm]
runtime      = "vllm"
hf_model     = "Qwen/Qwen3.5-4B"
api_base_url = "http://localhost:8000/v1"
```

Other models to try with vLLM:

```toml
hf_model = "meta-llama/Llama-3.1-8B-Instruct"
hf_model = "meta-llama/Llama-3.1-70B-Instruct"
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
hf_model = "microsoft/Phi-4"
```

Model weights are downloaded once to `~/.cache/huggingface/hub/` on first run.

### TensorRT-LLM / Triton Inference Server

TensorRT-LLM also exposes an OpenAI-compatible endpoint once the server is running.

```bash
# Assuming TRT-LLM server is already configured and started
# It typically listens on port 8000 with the same API shape as vLLM
```

`config.toml`:

```toml
[llm]
runtime      = "tensorrt"
hf_model     = "meta-llama/Llama-3.1-8B-Instruct"
api_base_url = "http://localhost:8000/v1"
```

The app code treats `vllm` and `tensorrt` identically — both use the OpenAI-compatible `/chat/completions` endpoint. Use whichever name is clearer to you.

---

## Benchmark Infrastructure

The benchmark tools run the analysis programmatically — no browser needed — and produce JSON results you can compare across runs.

### Config files

| File | Purpose |
|---|---|
| `config.toml` | LLM runtime, model, temperature — **inference config** |
| `benchmark/suite.toml` | Warmup runs, timed runs, concurrency levels — **execution params** |
| `benchmark/portfolios/default.toml` | Which stocks to analyze — **portfolio** |

Create additional portfolio files for different test sets:

```bash
cp benchmark/portfolios/default.toml benchmark/portfolios/tech_heavy.toml
# edit tech_heavy.toml to add GOOGL, META, AMZN, etc.
```

### Running a benchmark

```bash
# Default: all stocks in portfolios/default.toml, params from suite.toml
uv run benchmark/run_benchmark.py

# Specify a different portfolio
uv run benchmark/run_benchmark.py --portfolio benchmark/portfolios/tech_heavy.toml

# Custom suite params
uv run benchmark/run_benchmark.py --suite benchmark/suite.toml --concurrency 1 2 4 --runs 5

# Run only specific scenarios from the portfolio
uv run benchmark/run_benchmark.py --scenarios aapl_moderate tsla_aggressive

# Save results to a custom directory
uv run benchmark/run_benchmark.py --output-dir /tmp/bench_results
```

**Example output:**

```
============================================================
FSI LLM Benchmark — standard
Run ID:      20250622_143012
Runtime:     ollama
Model:       qwen3.5:4b
Portfolio:   benchmark/portfolios/default.toml
Scenarios:   ['aapl_moderate', 'tsla_aggressive', 'msft_conservative', 'nvda_day_trader']
Concurrency: [1, 4, 8]
Timed runs:  3  Warmup runs: 1
============================================================

[Phase 1] Prefetching prompts for 4 scenario(s)...
  Fetching data for aapl_moderate (AAPL, Moderate)... OK (4821 chars)
  Fetching data for tsla_aggressive (TSLA, Aggressive)... OK (5103 chars)
  Fetching data for msft_conservative (MSFT, Conservative)... OK (4934 chars)
  Fetching data for nvda_day_trader (NVDA, Day Trader)... OK (5218 chars)

[Phase 2] Warmup (1 run(s) per scenario)...
  Warming up aapl_moderate run 1/1... OK (6.3s, 412 tok)
  Warming up tsla_aggressive run 1/1... OK (5.8s, 389 tok)
  ...

[Phase 3] Timed benchmark (3 batch(es) per concurrency level)...

  Concurrency=1:
    Batch 1/3 (concurrency=1)... OK (6.1s, 408 tokens)
    ...
    --> mean_latency=6.2s  p95=6.8s  agg_tps=65.3

[Done] Results saved to: benchmark/results/20250622_143012_ollama_qwen3.5_4b.json
```

### Comparing results with `analyze.py`

After collecting results for different models or runtimes, compare them side-by-side:

```bash
# Compare two runs
uv run benchmark/analyze.py \
    benchmark/results/20250622_143012_ollama_qwen3.5_4b.json \
    benchmark/results/20250622_151847_ollama_llama3.1_8b.json

# Compare all results in the directory
uv run benchmark/analyze.py benchmark/results/*.json

# Filter to specific concurrency levels
uv run benchmark/analyze.py benchmark/results/*.json --concurrency 1 4

# Export to CSV for Excel / polars analysis
uv run benchmark/analyze.py benchmark/results/*.json --csv > comparison.csv
```

**Example terminal output** (single 11-column table, concurrency = 1):

```
──────────────────────────── Concurrency = 1 ─────────────────────────────────────────────────────────────────────────────────────────────────
 Run ID               Timestamp            Runtime  Model              Concurrency  Mean Lat (s)  P95 Lat (s)  Mean TPS    Agg TPS   Tokens  Wall (s)
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 20250622_143012      2025-06-22T14:30:12  ollama   qwen3.5:4b                  1  6.21          6.84         65.80       63.50     1224    19.29
 20250622_151847      2025-06-22T15:18:47  ollama   llama3.1:8b                 1  9.47          10.31        44.10       43.20     1267    29.32

  Green = best. Agg TPS = total_tokens / wall_clock across all concurrent requests.

──────────────────────────── Concurrency = 4 ─────────────────────────────────────────────────────────────────────────────────────────────────
 Run ID               Timestamp            Runtime  Model              Concurrency  Mean Lat (s)  P95 Lat (s)  Mean TPS    Agg TPS   Tokens  Wall (s)
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 20250622_143012      2025-06-22T14:30:12  ollama   qwen3.5:4b                  4  22.10         24.60        12.10       55.80     4891    87.65
 20250622_151847      2025-06-22T15:18:47  ollama   llama3.1:8b                 4  33.81         36.10         9.40       41.30     5068   122.73
```

Best values are highlighted in **green** in the actual terminal output.

### Columns reference

| Column | Description |
|---|---|
| Run ID | Timestamp-based run identifier; ties to the JSON filename |
| Timestamp | ISO-8601 wall-clock time the run started |
| Runtime | `ollama`, `vllm`, or `tensorrt` |
| Model | Model tag or HuggingFace ID |
| Concurrency | Number of simultaneous requests in this batch |
| Mean Lat (s) | Average end-to-end latency per request |
| P95 Lat (s) | 95th-percentile latency — captures tail behavior |
| Mean TPS | Average tokens/second per request (from model timing) |
| Agg TPS | `total_tokens / wall_clock` — whole-run throughput |
| Tokens | Total output tokens across all requests in this level |
| Wall (s) | Wall-clock time for all batches at this concurrency |

### Analyzing results with polars

The `--csv` export is designed for further analysis. Example using polars:

```python
import polars as pl

df = pl.read_csv("comparison.csv")

# Best mean latency per model
print(
    df.group_by(["runtime", "model"])
      .agg(pl.col("mean_latency_s").min().alias("best_mean_lat_s"))
      .sort("best_mean_lat_s")
)

# Throughput scaling: how agg_tps changes with concurrency
print(
    df.select(["model", "concurrency", "aggregate_tps"])
      .sort(["model", "concurrency"])
)
```

---

## Project Structure

```
fsi-stock-analysis/
├── config.toml                   # Runtime config — edit to switch models/backends
├── FSI_StockAnalysis.py          # Gradio web UI (entry point)
├── fsi_core.py                   # Business logic: data fetch, prompt build, LLM call
├── pyproject.toml                # Python dependencies (managed by uv)
│
├── benchmark/
│   ├── run_benchmark.py          # Programmatic benchmark runner
│   ├── analyze.py                # Result comparison and CSV export
│   ├── suite.toml                # Execution params (runs, concurrency levels)
│   ├── portfolios/
│   │   └── default.toml          # Default 4-stock benchmark portfolio
│   └── results/                  # JSON result files (gitignored)
│
├── fsi_vllm/                     # Original vLLM reference implementation
└── fsi_old.py                    # Original monolithic implementation (reference)
```

---

## Comparing Ollama vs. vLLM Performance

A typical workflow when you have access to a GPU server:

**Step 1 — Baseline on CPU with Ollama:**

```bash
# config.toml: runtime = "ollama", ollama_model = "qwen3.5:4b"
uv run benchmark/run_benchmark.py --output-dir benchmark/results/cpu_baseline
```

**Step 2 — Same model via vLLM on GPU:**

```bash
# Start vLLM server on the GPU machine
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-4B --port 8000

# Edit config.toml: runtime = "vllm", hf_model = "Qwen/Qwen3.5-4B"
uv run benchmark/run_benchmark.py --output-dir benchmark/results/gpu_vllm
```

**Step 3 — Compare:**

```bash
uv run benchmark/analyze.py \
    benchmark/results/cpu_baseline/*.json \
    benchmark/results/gpu_vllm/*.json
```

**Step 4 — Try a larger model on GPU:**

```bash
# config.toml: hf_model = "meta-llama/Llama-3.1-70B-Instruct"
uv run benchmark/run_benchmark.py \
    --portfolio benchmark/portfolios/default.toml \
    --output-dir benchmark/results/gpu_llama70b
```

**Step 5 — Three-way comparison:**

```bash
uv run benchmark/analyze.py \
    benchmark/results/cpu_baseline/*.json \
    benchmark/results/gpu_vllm/*.json \
    benchmark/results/gpu_llama70b/*.json \
    --csv > three_way.csv
```

---

## Tips

- **Stock data is cached** in `~/.cache/fsi-stock-analysis/stock_data/`. Historical ranges are cached indefinitely; only today's data bypasses the cache. Run `benchmark/run_benchmark.py` twice and the second run skips all data fetching.
- **Warmup runs** (`warmup_runs` in `suite.toml`) fire real LLM requests before timing starts. This matters on Ollama where the first request loads the model into RAM. Set to `0` only if the model is already loaded.
- **Result files are JSON** — open them directly to inspect raw per-request timings, or use `analyze.py` for formatted comparison.
- **`--csv` output** goes to stdout so you can pipe it: `analyze.py results/*.json --csv | tee comparison.csv`.

---

## Disclaimer

This tool is for educational and research purposes only. It does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions.
