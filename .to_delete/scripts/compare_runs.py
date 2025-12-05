#!/usr/bin/env python3
"""
COMPARE RUNS - Diff Two Benchmark Reports
==========================================
Compare metrics between two benchmark runs to understand
what changed and in which direction.

Usage:
    python scripts/compare_runs.py reports/run_a.json reports/run_b.json
    python scripts/compare_runs.py --latest  # Compare two most recent runs
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def load_report(path: str) -> Dict[str, Any]:
    """Load a benchmark report from JSON."""
    with open(path) as f:
        return json.load(f)


def get_latest_reports(n: int = 2) -> List[Path]:
    """Get the N most recent benchmark reports."""
    report_dir = Path("reports")
    if not report_dir.exists():
        return []

    reports = sorted(
        report_dir.glob("benchmark_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return reports[:n]


def format_delta(
    val_a: Optional[float],
    val_b: Optional[float],
    higher_is_better: bool = False
) -> Tuple[str, str]:
    """Format a value change with delta and indicator."""
    if val_a is None or val_b is None:
        return "N/A", ""

    delta = val_b - val_a

    # Determine if change is good/bad
    if higher_is_better:
        indicator = "+" if delta > 0 else "-" if delta < 0 else "="
    else:
        indicator = "-" if delta > 0 else "+" if delta < 0 else "="

    return f"{delta:+.4f}", indicator


def compare_reports(run_a: Dict, run_b: Dict, verbose: bool = False):
    """Compare two benchmark run reports."""
    print("\n" + "=" * 70)
    print("BENCHMARK RUN COMPARISON")
    print("=" * 70)
    print(f"Run A: {run_a.get('run_id', '?')} (commit: {run_a.get('commit', '?')})")
    print(f"Run B: {run_b.get('run_id', '?')} (commit: {run_b.get('commit', '?')})")
    print()

    sum_a = run_a.get("summary", {})
    sum_b = run_b.get("summary", {})

    # Define metrics to compare with their properties
    # (key, label, format, higher_is_better)
    metrics = [
        ("avg_time_seconds", "Avg Time (s)", "{:.1f}", False),
        ("total_cost", "Total Cost ($)", "{:.4f}", False),
        ("avg_tovs", "TOVS", "{:.3f}", False),  # Lower is better
        ("avg_icr", "ICR", "{:.3f}", True),  # Higher is better
        ("avg_gcr", "GCR", "{:.2f}", False),  # Lower is better (less imbalance)
        ("avg_dsr", "DSR", "{:.3f}", True),  # Higher is better
        ("avg_oscillations", "Oscillations", "{:.1f}", False),  # Lower is better
        ("avg_token_efficiency", "Token Efficiency", "{:.3f}", True),  # Higher is better
    ]

    print(f"{'Metric':<20} {'Run A':>12} {'Run B':>12} {'Delta':>12} {'Trend':>6}")
    print("-" * 66)

    for key, label, fmt, higher_is_better in metrics:
        val_a = sum_a.get(key)
        val_b = sum_b.get(key)

        if val_a is None and val_b is None:
            continue

        str_a = fmt.format(val_a) if val_a is not None else "N/A"
        str_b = fmt.format(val_b) if val_b is not None else "N/A"

        if val_a is not None and val_b is not None:
            delta = val_b - val_a
            delta_str = f"{delta:+.4f}"

            # Determine trend indicator
            if abs(delta) < 0.001:
                trend = "="
            elif (delta > 0) == higher_is_better:
                trend = "GOOD"
            else:
                trend = "BAD"
        else:
            delta_str = "-"
            trend = ""

        print(f"{label:<20} {str_a:>12} {str_b:>12} {delta_str:>12} {trend:>6}")

    print("=" * 70)

    # Per-benchmark details
    results_a = {r["id"]: r for r in run_a.get("results", [])}
    results_b = {r["id"]: r for r in run_b.get("results", [])}
    all_ids = sorted(set(results_a.keys()) | set(results_b.keys()))

    if len(all_ids) > 0:
        print("\nPER-BENCHMARK DETAILS:")
        print("-" * 70)

        for bench_id in all_ids:
            ra = results_a.get(bench_id)
            rb = results_b.get(bench_id)

            if ra is None:
                print(f"  {bench_id}: NEW in Run B")
                continue
            if rb is None:
                print(f"  {bench_id}: REMOVED in Run B")
                continue

            # Get key metrics
            exec_a = ra.get("execution", {})
            exec_b = rb.get("execution", {})
            gp_a = ra.get("graph_physics", {})
            gp_b = rb.get("graph_physics", {})

            cost_a = exec_a.get("total_cost", 0)
            cost_b = exec_b.get("total_cost", 0)
            tovs_a = gp_a.get("tovs")
            tovs_b = gp_b.get("tovs")

            changes = []

            # Cost change
            if cost_a > 0 and cost_b > 0:
                cost_change = ((cost_b - cost_a) / cost_a) * 100
                if abs(cost_change) > 5:
                    changes.append(f"cost {cost_change:+.0f}%")

            # TOVS change
            if tovs_a is not None and tovs_b is not None:
                tovs_delta = tovs_b - tovs_a
                if abs(tovs_delta) > 0.01:
                    direction = "↑" if tovs_delta > 0 else "↓"
                    changes.append(f"TOVS {direction}{abs(tovs_delta):.2f}")

            if changes:
                print(f"  {bench_id}: {', '.join(changes)}")
            elif verbose:
                print(f"  {bench_id}: no significant change")

    print("=" * 70)

    # Summary
    print("\nSUMMARY:")

    # Count improvements vs regressions
    improvements = 0
    regressions = 0

    for key, _, _, higher_is_better in metrics:
        val_a = sum_a.get(key)
        val_b = sum_b.get(key)
        if val_a is not None and val_b is not None:
            delta = val_b - val_a
            if abs(delta) > 0.001:
                if (delta > 0) == higher_is_better:
                    improvements += 1
                else:
                    regressions += 1

    print(f"  Improvements: {improvements}")
    print(f"  Regressions: {regressions}")

    if regressions > improvements:
        print("\n  Overall: REGRESSION detected")
        return 1
    elif improvements > regressions:
        print("\n  Overall: IMPROVEMENT detected")
        return 0
    else:
        print("\n  Overall: No significant change")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Compare two GAADP benchmark runs"
    )
    parser.add_argument(
        "run_a",
        nargs="?",
        help="Path to first run report (baseline)"
    )
    parser.add_argument(
        "run_b",
        nargs="?",
        help="Path to second run report (new)"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Compare the two most recent runs"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all benchmarks, not just changes"
    )

    args = parser.parse_args()

    if args.latest:
        reports = get_latest_reports(2)
        if len(reports) < 2:
            print("Error: Need at least 2 reports to compare")
            print(f"Found: {len(reports)} reports in reports/")
            sys.exit(1)
        run_a_path = str(reports[1])  # Older
        run_b_path = str(reports[0])  # Newer
        print(f"Comparing latest runs:")
        print(f"  A (older): {reports[1].name}")
        print(f"  B (newer): {reports[0].name}")
    else:
        if not args.run_a or not args.run_b:
            print("Error: Provide two report paths or use --latest")
            parser.print_help()
            sys.exit(1)
        run_a_path = args.run_a
        run_b_path = args.run_b

    try:
        run_a = load_report(run_a_path)
        run_b = load_report(run_b_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
        sys.exit(1)

    exit_code = compare_reports(run_a, run_b, verbose=args.verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
