#!/usr/bin/env python3
"""
Compare Runs - Historical Benchmark Comparison Utility

Compare benchmark results between different runs to analyze performance
changes over time or across configurations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from tabulate import tabulate


def load_raw_result(filepath: Path) -> dict:
    """Load a raw JSON result file."""
    with open(filepath) as f:
        return json.load(f)


def find_raw_files(results_dir: Path) -> list[Path]:
    """Find all raw result JSON files."""
    raw_dir = results_dir / "raw"
    if not raw_dir.exists():
        return []
    return sorted(raw_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)


def list_runs(results_dir: Path):
    """List all available benchmark runs."""
    raw_files = find_raw_files(results_dir)

    if not raw_files:
        print("No benchmark runs found.")
        return

    print(f"\nFound {len(raw_files)} benchmark run(s):\n")

    rows = []
    for i, filepath in enumerate(raw_files, 1):
        try:
            data = load_raw_result(filepath)
            suite_name = list(data.keys())[0] if data else "unknown"
            suite_data = data.get(suite_name, {})

            # Extract key info
            dataset = suite_data.get("dataset", "unknown")
            timestamp = filepath.stem.split("_")[-1] if "_" in filepath.stem else "unknown"
            num_benchmarks = len(suite_data.get("benchmarks", {}))

            rows.append([i, filepath.name, suite_name, dataset, num_benchmarks, timestamp])
        except Exception as e:
            rows.append([i, filepath.name, "error", str(e)[:30], "-", "-"])

    headers = ["#", "Filename", "Suite", "Dataset", "Benchmarks", "Timestamp"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print()


def get_run_by_index(results_dir: Path, index: int) -> Optional[Path]:
    """Get a raw result file by its index (1-based)."""
    raw_files = find_raw_files(results_dir)
    if 1 <= index <= len(raw_files):
        return raw_files[index - 1]
    return None


def extract_suite_summary(data: dict) -> dict:
    """Extract summary metrics from a run for cross-suite comparison."""
    summary = {}

    for suite_name, suite_data in data.items():
        results = suite_data.get("results", suite_data)  # Handle both formats
        benchmarks = results.get("benchmarks", suite_data.get("benchmarks", {}))

        # Find best QPS at recall >= 95%
        best_qps_95 = None
        best_recall_95 = None
        best_p50 = None
        best_p99 = None

        for bench_name, bench_data in benchmarks.items():
            recall = bench_data.get("recall", 0)
            qps = bench_data.get("qps", 0)
            p50 = bench_data.get("p50_ms") or bench_data.get("p50_latency", 0)
            p99 = bench_data.get("p99_ms") or bench_data.get("p99_latency", 0)

            # Best QPS at recall >= 95%
            if recall >= 0.95:
                if best_qps_95 is None or qps > best_qps_95:
                    best_qps_95 = qps
                    best_recall_95 = recall

            # Track best (lowest) latencies
            if p50 and (best_p50 is None or p50 < best_p50):
                best_p50 = p50
            if p99 and (best_p99 is None or p99 < best_p99):
                best_p99 = p99

        summary[suite_name] = {
            "suite_type": "pgvector" if "pgvector" in suite_name.lower() else
                         "pgpu" if "pgpu" in suite_name.lower() else
                         "vectorchord",
            "index_build_time": results.get("index_build_time"),
            "load_time": results.get("load_time"),
            "index_size": results.get("index_size"),
            "best_qps_95": best_qps_95,
            "best_recall_95": best_recall_95,
            "best_p50": best_p50,
            "best_p99": best_p99,
            "num_benchmarks": len(benchmarks),
        }

    return summary


def compare_runs_summary(results_dir: Path, run_indices: list[int]):
    """Compare multiple runs using summary metrics (cross-suite compatible)."""
    # Load all runs
    runs = []
    for idx in run_indices:
        path = get_run_by_index(results_dir, idx)
        if not path:
            print(f"Error: Run #{idx} not found.")
            return
        data = load_raw_result(path)
        summary = extract_suite_summary(data)
        suite_name = list(summary.keys())[0]
        runs.append({
            "idx": idx,
            "path": path,
            "suite_name": suite_name,
            "summary": summary[suite_name],
        })

    show_delta = len(runs) == 2

    # Helper functions
    def fmt_time(val):
        if val is None:
            return "-"
        return f"{val}s" if isinstance(val, (int, float)) else str(val)

    def fmt_qps(val):
        return f"{val:.0f}" if val else "-"

    def fmt_recall(val):
        return f"{val:.4f}" if val else "-"

    def fmt_latency(val):
        return f"{val:.2f}" if val else "-"

    def calc_delta(val_a, val_b, lower_is_better=True):
        if val_a is None or val_b is None:
            return "-"
        try:
            a = float(str(val_a).replace('s', ''))
            b = float(str(val_b).replace('s', ''))
            if a == 0:
                return "-"
            pct = ((b - a) / a) * 100
            # + means B is better, - means B is worse
            if lower_is_better:
                indicator = "+" if pct < 0 else ("-" if pct > 0 else "=")
            else:
                indicator = "+" if pct > 0 else ("-" if pct < 0 else "=")
            return f"{indicator}{abs(pct):.1f}%"
        except (ValueError, TypeError):
            return "-"

    # Print header
    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPARISON")
    print(f"{'=' * 80}\n")

    # List runs
    for i, run in enumerate(runs):
        label = chr(65 + i)  # A, B, C, ...
        print(f"  Run {label} (#{run['idx']}): {run['suite_name']}")
    print()

    # Build column headers
    headers = ["Metric"] + [chr(65 + i) for i in range(len(runs))]
    if show_delta:
        headers.append("Δ (A→B)")

    # Build metrics table
    print(f"{'─' * 80}")
    print("BUILD METRICS")
    print(f"{'─' * 80}")

    build_rows = []
    row = ["Index Build Time"] + [fmt_time(r["summary"].get("index_build_time")) for r in runs]
    if show_delta:
        row.append(calc_delta(runs[0]["summary"].get("index_build_time"),
                              runs[1]["summary"].get("index_build_time"), lower_is_better=True))
    build_rows.append(row)

    row = ["Load Time"] + [fmt_time(r["summary"].get("load_time")) for r in runs]
    if show_delta:
        row.append(calc_delta(runs[0]["summary"].get("load_time"),
                              runs[1]["summary"].get("load_time"), lower_is_better=True))
    build_rows.append(row)

    row = ["Index Size"] + [r["summary"].get("index_size") or "-" for r in runs]
    if show_delta:
        row.append("-")
    build_rows.append(row)

    print(tabulate(build_rows, headers=headers, tablefmt="simple"))
    print()

    # Query performance table
    print(f"{'─' * 80}")
    print("QUERY PERFORMANCE (best results)")
    print(f"{'─' * 80}")

    perf_rows = []

    row = ["Best QPS (recall≥95%)"] + [fmt_qps(r["summary"].get("best_qps_95")) for r in runs]
    if show_delta:
        row.append(calc_delta(runs[0]["summary"].get("best_qps_95"),
                              runs[1]["summary"].get("best_qps_95"), lower_is_better=False))
    perf_rows.append(row)

    row = ["Recall at best QPS"] + [fmt_recall(r["summary"].get("best_recall_95")) for r in runs]
    if show_delta:
        row.append("-")
    perf_rows.append(row)

    row = ["Best P50 (ms)"] + [fmt_latency(r["summary"].get("best_p50")) for r in runs]
    if show_delta:
        row.append(calc_delta(runs[0]["summary"].get("best_p50"),
                              runs[1]["summary"].get("best_p50"), lower_is_better=True))
    perf_rows.append(row)

    row = ["Best P99 (ms)"] + [fmt_latency(r["summary"].get("best_p99")) for r in runs]
    if show_delta:
        row.append(calc_delta(runs[0]["summary"].get("best_p99"),
                              runs[1]["summary"].get("best_p99"), lower_is_better=True))
    perf_rows.append(row)

    print(tabulate(perf_rows, headers=headers, tablefmt="simple"))
    print()

    if show_delta:
        print(f"{'─' * 80}")
        print("Δ: + = B is better, - = B is worse")
    print(f"{'=' * 80}\n")


def show_run_details(results_dir: Path, run_idx: int):
    """Show detailed information about a specific run."""
    run_path = get_run_by_index(results_dir, run_idx)
    if not run_path:
        print(f"Error: Run #{run_idx} not found.")
        return

    print(f"\nDetails for run #{run_idx}: {run_path.name}\n")

    data = load_raw_result(run_path)

    for suite_name, suite_data in data.items():
        print(f"Suite: {suite_name}")
        print("-" * 40)

        # Configuration info
        config_keys = ["dataset", "lists", "m", "efConstruction", "index_build_time", "load_time"]
        for key in config_keys:
            if key in suite_data:
                print(f"  {key}: {suite_data[key]}")

        print()

        # Benchmark results
        benchmarks = suite_data.get("benchmarks", {})
        if benchmarks:
            rows = []
            for bench_name, bench_data in benchmarks.items():
                rows.append({
                    "Benchmark": bench_name,
                    "Recall": f"{bench_data.get('recall', 0):.4f}",
                    "QPS": f"{bench_data.get('qps', 0):.0f}",
                    "P50 (ms)": f"{bench_data.get('p50_ms', 0):.2f}",
                    "P99 (ms)": f"{bench_data.get('p99_ms', 0):.2f}",
                })

            print("Benchmark Results:")
            print(tabulate(rows, headers="keys", tablefmt="simple", showindex=False))
            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare benchmark results between different runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available runs
  python compare_runs.py --list

  # Show details for run #5
  python compare_runs.py --show 5

  # Compare 2 runs (shows deltas)
  python compare_runs.py --compare 3 7

  # Compare multiple runs (any suite type)
  python compare_runs.py --compare 3 7 12
        """
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Results directory (default: ./results)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available benchmark runs"
    )
    parser.add_argument(
        "--show", "-s",
        type=int,
        metavar="N",
        help="Show details for run #N"
    )
    parser.add_argument(
        "--compare", "-c",
        type=int,
        nargs="+",
        metavar="N",
        help="Compare multiple runs (works across different suite types)"
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    if args.list:
        list_runs(args.results_dir)
    elif args.show:
        show_run_details(args.results_dir, args.show)
    elif args.compare:
        if len(args.compare) < 2:
            print("Error: Need at least 2 runs to compare.")
            sys.exit(1)
        compare_runs_summary(args.results_dir, args.compare)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
