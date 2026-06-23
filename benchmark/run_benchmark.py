#!/usr/bin/env python3
"""
run_benchmark.py — FSI LLM Benchmark Runner

Measures latency and throughput for stock analysis inference across
different concurrency levels, saving results as JSON for later comparison.

Usage:
    uv run benchmark/run_benchmark.py
    uv run benchmark/run_benchmark.py --portfolio benchmark/portfolios/tech_stocks.toml
    uv run benchmark/run_benchmark.py --suite benchmark/suite.toml --concurrency 1 4 --runs 5
    uv run benchmark/run_benchmark.py --scenarios aapl_moderate tsla_aggressive
    uv run benchmark/run_benchmark.py --output-dir /tmp/bench_results
"""

import argparse
import asyncio
import json
import os
import platform
import socket
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path
from statistics import mean, median, quantiles

import aiohttp

# ── Resolve project root (parent of benchmark/) ───────────────────────────────
_BENCH_DIR = Path(__file__).parent.resolve()
_PROJECT_DIR = _BENCH_DIR.parent

# Add project root to sys.path so fsi_core can be imported
sys.path.insert(0, str(_PROJECT_DIR))

import fsi_core  # noqa: E402


def load_config():
    config_path = _PROJECT_DIR / "config.toml"
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def load_portfolio(path: Path) -> list[dict]:
    """Load stock scenarios from a portfolio TOML file (key: [[stock]])."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data.get("stock", [])


def load_suite(path: Path) -> dict:
    """Load suite execution params from a suite TOML file (key: [suite])."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data.get("suite", {})


def slugify(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace(" ", "_").lower()


def compute_stats(values: list[float]) -> dict:
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    # quantiles requires at least 2 values for p95/p99; fall back gracefully
    if n >= 2:
        qs = quantiles(sorted_vals, n=100)
        p95 = qs[94]  # index 94 = 95th percentile
        p99 = qs[98]  # index 98 = 99th percentile
    else:
        p95 = sorted_vals[-1]
        p99 = sorted_vals[-1]
    return {
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "p95": round(p95, 4),
        "p99": round(p99, 4),
        "min": round(sorted_vals[0], 4),
        "max": round(sorted_vals[-1], 4),
    }


# ── Direct API call helpers ────────────────────────────────────────────────────

async def call_ollama(session: aiohttp.ClientSession, model: str, prompt: str,
                      temperature: float, max_tokens: int) -> dict:
    """POST to Ollama /api/chat. Returns latency_s, tokens_out, tps."""
    url = "http://localhost:11434/api/chat"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    t0 = time.perf_counter()
    async with session.post(url, json=body) as resp:
        resp.raise_for_status()
        data = await resp.json()
    latency_s = time.perf_counter() - t0

    tokens_out = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 0)
    # eval_duration is in nanoseconds; derive TPS from it when available
    if eval_duration_ns and eval_duration_ns > 0:
        tps = tokens_out / (eval_duration_ns / 1e9)
    elif latency_s > 0 and tokens_out > 0:
        tps = tokens_out / latency_s
    else:
        tps = 0.0

    return {"latency_s": round(latency_s, 4), "tokens_out": tokens_out, "tps": round(tps, 2)}


async def call_openai_compat(session: aiohttp.ClientSession, api_base_url: str,
                              model: str, prompt: str,
                              temperature: float, max_tokens: int) -> dict:
    """POST to an OpenAI-compatible /chat/completions endpoint (vLLM / TRT-LLM)."""
    url = f"{api_base_url.rstrip('/')}/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": "Bearer none", "Content-Type": "application/json"}
    t0 = time.perf_counter()
    async with session.post(url, json=body, headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()
    latency_s = time.perf_counter() - t0

    tokens_out = data.get("usage", {}).get("completion_tokens", 0)
    tps = round(tokens_out / latency_s, 2) if latency_s > 0 and tokens_out > 0 else 0.0

    return {"latency_s": round(latency_s, 4), "tokens_out": tokens_out, "tps": tps}


async def call_llm(session: aiohttp.ClientSession, cfg: dict, prompt: str) -> dict:
    """Dispatch to the correct backend based on config."""
    llm_cfg = cfg["llm"]
    runtime = llm_cfg["runtime"]
    temperature = llm_cfg["temperature"]
    max_tokens = llm_cfg["max_tokens"]

    if runtime == "ollama":
        return await call_ollama(session, llm_cfg["ollama_model"], prompt, temperature, max_tokens)
    elif runtime in ("vllm", "tensorrt"):
        return await call_openai_compat(
            session, llm_cfg["api_base_url"], llm_cfg["hf_model"],
            prompt, temperature, max_tokens
        )
    else:
        raise ValueError(f"Unknown runtime: {runtime!r}")


# ── Benchmark phases ───────────────────────────────────────────────────────────

def prefetch_prompts(scenarios: list[dict]) -> dict[str, str]:
    """Phase 1: Build all prompts (also warms the stock-data disk cache)."""
    prompts = {}
    print(f"\n[Phase 1] Prefetching prompts for {len(scenarios)} scenario(s)...")
    for sc in scenarios:
        name = sc["name"]
        print(f"  Fetching data for {name} ({sc['symbol']}, {sc['investor_type']})...", end=" ", flush=True)
        try:
            prompt = fsi_core.build_prompt_for_scenario(
                sc["symbol"], sc["start_date"], sc["end_date"], sc["investor_type"]
            )
            prompts[name] = prompt
            print(f"OK ({len(prompt)} chars)")
        except Exception as e:
            print(f"FAILED: {e}")
    return prompts


async def run_warmup(cfg: dict, scenarios: list[dict], prompts: dict[str, str],
                     warmup_runs: int):
    """Phase 2: Fire warmup_runs requests per scenario (results discarded)."""
    print(f"\n[Phase 2] Warmup ({warmup_runs} run(s) per scenario)...")
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for sc in scenarios:
            name = sc["name"]
            if name not in prompts:
                continue
            for i in range(warmup_runs):
                print(f"  Warming up {name} run {i+1}/{warmup_runs}...", end=" ", flush=True)
                try:
                    result = await call_llm(session, cfg, prompts[name])
                    print(f"OK ({result['latency_s']:.1f}s, {result['tokens_out']} tok)")
                except Exception as e:
                    print(f"FAILED: {e}")


async def run_concurrent_batch(cfg: dict, prompts: dict[str, str],
                                scenario_names: list[str],
                                concurrency: int) -> list[tuple[str, dict]]:
    """
    Run `concurrency` requests in parallel.  Cycles through scenarios to fill
    the batch.  Returns list of (scenario_name, result_dict).
    """
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        names = []
        for i in range(concurrency):
            sc_name = scenario_names[i % len(scenario_names)]
            names.append(sc_name)
            tasks.append(call_llm(session, cfg, prompts[sc_name]))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    out = []
    for sc_name, res in zip(names, results):
        if isinstance(res, Exception):
            print(f"    Request for {sc_name} failed: {res}")
        else:
            out.append((sc_name, res))
    return out


async def run_timed_phase(cfg: dict, scenarios: list[dict], prompts: dict[str, str],
                           concurrency_levels: list[int], timed_runs: int) -> dict:
    """Phase 3: Timed benchmark across concurrency levels."""
    print(f"\n[Phase 3] Timed benchmark ({timed_runs} batch(es) per concurrency level)...")

    sc_names = [sc["name"] for sc in scenarios if sc["name"] in prompts]
    concurrency_results = {}

    for concurrency in concurrency_levels:
        print(f"\n  Concurrency={concurrency}:")
        sc_run_data: dict[str, list[dict]] = {name: [] for name in sc_names}
        wall_start = time.perf_counter()

        for run_idx in range(timed_runs):
            print(f"    Batch {run_idx + 1}/{timed_runs} (concurrency={concurrency})...", end=" ", flush=True)
            batch_t0 = time.perf_counter()
            batch_results = await run_concurrent_batch(cfg, prompts, sc_names, concurrency)
            batch_elapsed = time.perf_counter() - batch_t0

            for sc_name, res in batch_results:
                sc_run_data[sc_name].append(res)

            total_tok = sum(r["tokens_out"] for _, r in batch_results)
            print(f"OK ({batch_elapsed:.1f}s, {total_tok} tokens)")

        wall_elapsed = time.perf_counter() - wall_start

        # Per-scenario stats
        scenarios_stats = {}
        all_latencies = []
        all_tokens = []

        for sc_name in sc_names:
            runs = sc_run_data[sc_name]
            if not runs:
                continue
            latencies = [r["latency_s"] for r in runs]
            tps_vals = [r["tps"] for r in runs]
            tokens = [r["tokens_out"] for r in runs]
            all_latencies.extend(latencies)
            all_tokens.extend(tokens)

            lat_stats = compute_stats(latencies)
            scenarios_stats[sc_name] = {
                "runs": runs,
                "stats": {
                    "mean_latency": lat_stats.get("mean"),
                    "median_latency": lat_stats.get("median"),
                    "p95_latency": lat_stats.get("p95"),
                    "p99_latency": lat_stats.get("p99"),
                    "mean_tps": round(mean(tps_vals), 2) if tps_vals else 0,
                },
            }

        # Aggregate stats
        total_requests = len(all_latencies)
        total_tokens = sum(all_tokens)
        agg_lat_stats = compute_stats(all_latencies)
        aggregate_tps = round(total_tokens / wall_elapsed, 2) if wall_elapsed > 0 else 0

        concurrency_results[str(concurrency)] = {
            "scenarios": scenarios_stats,
            "aggregate": {
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "wall_clock_s": round(wall_elapsed, 2),
                "aggregate_tps": aggregate_tps,
                "mean_latency": agg_lat_stats.get("mean"),
                "median_latency": agg_lat_stats.get("median"),
                "p95_latency": agg_lat_stats.get("p95"),
                "p99_latency": agg_lat_stats.get("p99"),
            },
        }

        agg = concurrency_results[str(concurrency)]["aggregate"]
        print(f"    --> mean_latency={agg['mean_latency']}s  p95={agg['p95_latency']}s  "
              f"agg_tps={agg['aggregate_tps']}")

    return concurrency_results


# ── Result serialization ───────────────────────────────────────────────────────

def build_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results(results: dict, output_dir: Path, runtime: str, model: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = results["run_id"]
    model_slug = slugify(model)
    filename = f"{run_id}_{runtime}_{model_slug}.json"
    out_path = output_dir / filename
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="FSI LLM benchmark runner — measures latency and throughput."
    )
    parser.add_argument(
        "--portfolio",
        default=str(_BENCH_DIR / "portfolios" / "default.toml"),
        metavar="FILE",
        help="Portfolio TOML file listing stocks to analyze (default: portfolios/default.toml)",
    )
    parser.add_argument(
        "--suite",
        default=str(_BENCH_DIR / "suite.toml"),
        metavar="FILE",
        help="Suite TOML file with execution params (default: suite.toml)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        metavar="NAME",
        help="Subset of scenario names from the portfolio to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_BENCH_DIR / "results"),
        help="Directory to save result JSON files",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        metavar="N",
        help="Override concurrency levels from suite.toml",
    )
    parser.add_argument(
        "--runs",
        type=int,
        metavar="N",
        help="Override timed_runs from suite.toml",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    cfg = load_config()
    suite = load_suite(Path(args.suite))
    all_scenarios = load_portfolio(Path(args.portfolio))

    if not all_scenarios:
        print(f"ERROR: No stocks found in portfolio: {args.portfolio}", file=sys.stderr)
        sys.exit(1)

    # Apply CLI overrides
    warmup_runs = suite.get("warmup_runs", 1)
    timed_runs = args.runs if args.runs is not None else suite.get("timed_runs", 3)
    concurrency_levels = args.concurrency if args.concurrency else suite.get("concurrency_levels", [1])

    # Filter by name if --scenarios supplied
    if args.scenarios:
        all_scenarios = [sc for sc in all_scenarios if sc["name"] in args.scenarios]
        if not all_scenarios:
            print(f"ERROR: No scenarios matched: {args.scenarios}", file=sys.stderr)
            sys.exit(1)

    output_dir = Path(args.output_dir)
    llm_cfg = cfg["llm"]
    runtime = llm_cfg["runtime"]
    model = llm_cfg["ollama_model"] if runtime == "ollama" else llm_cfg["hf_model"]
    suite_name = suite.get("name", "custom")

    run_id = build_run_id()
    timestamp = datetime.now().isoformat(timespec="seconds")

    print("=" * 60)
    print(f"FSI LLM Benchmark — {suite_name}")
    print(f"Run ID:      {run_id}")
    print(f"Runtime:     {runtime}")
    print(f"Model:       {model}")
    print(f"Portfolio:   {args.portfolio}")
    print(f"Scenarios:   {[sc['name'] for sc in all_scenarios]}")
    print(f"Concurrency: {concurrency_levels}")
    print(f"Timed runs:  {timed_runs}  Warmup runs: {warmup_runs}")
    print("=" * 60)

    # Phase 1 — Prefetch
    prompts = prefetch_prompts(all_scenarios)

    # Phase 2 — Warmup
    await run_warmup(cfg, all_scenarios, prompts, warmup_runs)

    # Phase 3 — Timed benchmark
    concurrency_results = await run_timed_phase(
        cfg, all_scenarios, prompts, concurrency_levels, timed_runs
    )

    # Build result document
    result_doc = {
        "run_id": run_id,
        "timestamp": timestamp,
        "metadata": {
            "runtime": runtime,
            "model": model,
            "temperature": llm_cfg["temperature"],
            "max_tokens": llm_cfg["max_tokens"],
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "config_snapshot": cfg,
        },
        "suite": suite_name,
        "portfolio": args.portfolio,
        "scenarios_run": [sc["name"] for sc in all_scenarios],
        "concurrency_results": concurrency_results,
    }

    out_path = save_results(result_doc, output_dir, runtime, model)

    print(f"\n[Done] Results saved to: {out_path}")
    print(f"       Run ID: {run_id}")


if __name__ == "__main__":
    asyncio.run(main())
