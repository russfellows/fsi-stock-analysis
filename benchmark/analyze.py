#!/usr/bin/env python3
"""
analyze.py — Compare FSI benchmark result files.

Usage:
    uv run benchmark/analyze.py results/run1.json results/run2.json
    uv run benchmark/analyze.py results/*.json --csv
    uv run benchmark/analyze.py results/run1.json --csv > comparison.csv
"""

import argparse
import csv
import json
import sys


def load_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def all_concurrency_levels(results: list[dict]) -> list[str]:
    levels = set()
    for r in results:
        levels.update(r.get("concurrency_results", {}).keys())
    # Sort numerically
    return sorted(levels, key=lambda x: int(x))


def format_float(val, decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def build_rows(results: list[dict], level: str) -> list[dict]:
    """Build comparison rows for a single concurrency level."""
    rows = []
    for r in results:
        agg = r.get("concurrency_results", {}).get(level, {}).get("aggregate", {})
        rows.append({
            "run_id": r.get("run_id", "?"),
            "timestamp": r.get("timestamp", "?"),
            "runtime": r.get("metadata", {}).get("runtime", "?"),
            "model": r.get("metadata", {}).get("model", "?"),
            "mean_latency_s": agg.get("mean_latency"),
            "p95_latency_s": agg.get("p95_latency"),
            "mean_tps": _compute_mean_tps(r, level),
            "aggregate_tps": agg.get("aggregate_tps"),
            "total_requests": agg.get("total_requests"),
            "total_tokens": agg.get("total_tokens"),
            "wall_clock_s": agg.get("wall_clock_s"),
        })
    return rows


def _compute_mean_tps(result: dict, level: str) -> float | None:
    """Average mean_tps across all scenarios at this concurrency level."""
    scenarios = result.get("concurrency_results", {}).get(level, {}).get("scenarios", {})
    tps_vals = [
        sc.get("stats", {}).get("mean_tps")
        for sc in scenarios.values()
        if sc.get("stats", {}).get("mean_tps") is not None
    ]
    if not tps_vals:
        return None
    return round(sum(tps_vals) / len(tps_vals), 2)


def find_best(rows: list[dict], key: str, prefer: str = "min") -> int | None:
    """Return index of row with best value. prefer='min' for latency, 'max' for tps."""
    vals = [r.get(key) for r in rows]
    valid = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not valid:
        return None
    if prefer == "min":
        return min(valid, key=lambda x: x[1])[0]
    else:
        return max(valid, key=lambda x: x[1])[0]


def print_rich_tables(results: list[dict], levels: list[str]):
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        print("ERROR: 'rich' is not installed. Run: uv add rich", file=sys.stderr)
        sys.exit(1)

    console = Console(highlight=False, width=200)

    def hl(val_str: str, is_best: bool) -> str:
        return f"[bold green]{val_str}[/]" if is_best else val_str

    for level in levels:
        rows = build_rows(results, level)

        best_lat      = find_best(rows, "mean_latency_s", "min")
        best_p95      = find_best(rows, "p95_latency_s",  "min")
        best_mean_tps = find_best(rows, "mean_tps",       "max")
        best_agg_tps  = find_best(rows, "aggregate_tps",  "max")

        console.rule(f"[bold cyan]Concurrency = {level}[/]")

        table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan", pad_edge=False)
        table.add_column("Run ID",        no_wrap=True,    style="dim")
        table.add_column("Timestamp",     no_wrap=True)
        table.add_column("Runtime",       style="bold",    no_wrap=True)
        table.add_column("Model",         no_wrap=True)
        table.add_column("Concurrency",   justify="right", no_wrap=True)
        table.add_column("Mean Lat (s)",  justify="right", no_wrap=True)
        table.add_column("P95 Lat (s)",   justify="right", no_wrap=True)
        table.add_column("Mean TPS",      justify="right", no_wrap=True)
        table.add_column("Agg TPS",       justify="right", no_wrap=True)
        table.add_column("Tokens",        justify="right", no_wrap=True)
        table.add_column("Wall (s)",      justify="right", no_wrap=True)
        for idx, row in enumerate(rows):
            table.add_row(
                row["run_id"],
                row["timestamp"],
                row["runtime"],
                row["model"],
                level,
                hl(format_float(row["mean_latency_s"]), idx == best_lat),
                hl(format_float(row["p95_latency_s"]),  idx == best_p95),
                hl(format_float(row["mean_tps"]),       idx == best_mean_tps),
                hl(format_float(row["aggregate_tps"]),  idx == best_agg_tps),
                str(row["total_tokens"] or "N/A"),
                format_float(row["wall_clock_s"]),
            )
        console.print(table)
        console.print(
            "  [dim]Green = best. "
            "Agg TPS = total_tokens / wall_clock across all concurrent requests.[/]\n"
        )


def print_csv(results: list[dict], levels: list[str]):
    fieldnames = [
        "concurrency", "run_id", "timestamp", "runtime", "model",
        "mean_latency_s", "p95_latency_s", "mean_tps", "aggregate_tps",
        "total_requests", "total_tokens", "wall_clock_s",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()

    for level in levels:
        rows = build_rows(results, level)
        for row in rows:
            writer.writerow({
                "concurrency": level,
                "run_id": row["run_id"],
                "timestamp": row["timestamp"],
                "runtime": row["runtime"],
                "model": row["model"],
                "mean_latency_s": format_float(row["mean_latency_s"]),
                "p95_latency_s": format_float(row["p95_latency_s"]),
                "mean_tps": format_float(row["mean_tps"]),
                "aggregate_tps": format_float(row["aggregate_tps"]),
                "total_requests": row["total_requests"] or "",
                "total_tokens": row["total_tokens"] or "",
                "wall_clock_s": format_float(row["wall_clock_s"]),
            })


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare FSI benchmark result JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="One or more benchmark result JSON files",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output as CSV instead of rich tables",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=str,
        metavar="N",
        help="Filter to specific concurrency levels (default: all found)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    results = []
    for path in args.files:
        try:
            results.append(load_result(path))
        except Exception as e:
            print(f"ERROR loading {path}: {e}", file=sys.stderr)
            sys.exit(1)

    if not results:
        print("No results to display.", file=sys.stderr)
        sys.exit(1)

    levels = all_concurrency_levels(results)
    if args.concurrency:
        levels = [lvl for lvl in levels if lvl in args.concurrency]

    if not levels:
        print("No matching concurrency levels found.", file=sys.stderr)
        sys.exit(1)

    if args.csv:
        print_csv(results, levels)
    else:
        print_rich_tables(results, levels)


if __name__ == "__main__":
    main()
