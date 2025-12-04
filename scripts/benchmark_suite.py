#!/usr/bin/env python3
"""
BENCHMARK SUITE RUNNER - Rigorous Metrics Edition
==================================================
Runs benchmark tests and captures comprehensive metrics.
No targets, no thresholds - only measurement and observation.

Usage:
    python scripts/benchmark_suite.py                    # Run all benchmarks
    python scripts/benchmark_suite.py --tags simple      # Run benchmarks with tag
    python scripts/benchmark_suite.py --id hello_world   # Run specific benchmark
    python scripts/benchmark_suite.py --compare RUN_A RUN_B  # Compare two runs
"""
import asyncio
import argparse
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaadp_main import main as gaadp_main
from benchmarks.metrics import GraphMetrics, print_metrics_report
from benchmarks.trajectory import TrajectoryAnalyzer, print_trajectory_report


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Benchmark")


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


class BenchmarkRunner:
    """Runs benchmark tests with comprehensive metrics collection."""

    def __init__(self, suite_path: str = "benchmarks/suite.yaml"):
        self.suite_path = Path(suite_path)
        self.suite = self._load_suite()
        self.results: List[Dict[str, Any]] = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.commit = get_git_commit()

    def _load_suite(self) -> Dict:
        """Load benchmark suite configuration."""
        if not self.suite_path.exists():
            raise FileNotFoundError(f"Suite file not found: {self.suite_path}")

        with open(self.suite_path) as f:
            return yaml.safe_load(f)

    async def run_benchmark(self, benchmark: Dict) -> Dict[str, Any]:
        """
        Run a single benchmark and collect all metrics.

        Returns comprehensive result with:
        - Execution stats
        - Graph physics metrics (TOVS, ICR, GCR, DSR)
        - Trajectory metrics (oscillations, token efficiency, etc)
        """
        bench_id = benchmark["id"]
        prompt_file = benchmark["prompt_file"]

        logger.info("=" * 60)
        logger.info(f"BENCHMARK: {bench_id}")
        logger.info("=" * 60)

        # Read prompt
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            return {
                "id": bench_id,
                "status": "ERROR",
                "error": f"Prompt file not found: {prompt_file}",
                "timestamp": datetime.now().isoformat()
            }

        prompt = prompt_path.read_text()

        # Create unique output directory
        output_dir = Path(f"workspace/benchmark_{bench_id}_{self.run_id}")

        # Clear previous graph (each benchmark should start fresh)
        graph_path = Path(".gaadp/graph.json")
        if graph_path.exists():
            graph_path.unlink()
            logger.info("Cleared previous graph state")

        # Clear previous telemetry
        telemetry_path = Path(".gaadp/logs/telemetry.jsonl")
        if telemetry_path.exists():
            telemetry_path.unlink()

        # Reset telemetry singleton
        try:
            from core.telemetry import TelemetryRecorder
            TelemetryRecorder.reset_instance()
        except ImportError:
            pass

        # Run GAADP
        start_time = time.time()
        try:
            stats = await gaadp_main(
                requirement=prompt,
                enable_viz=False,
                output_dir=str(output_dir)
            )
            execution_time = time.time() - start_time
            status = "COMPLETED"
            error = None

        except Exception as e:
            execution_time = time.time() - start_time
            status = "ERROR"
            error = str(e)
            stats = {}
            logger.error(f"Benchmark failed: {e}")

        # Collect metrics
        result = {
            "id": bench_id,
            "run_id": self.run_id,
            "commit": self.commit,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt[:500],
            "execution": {
                "time_seconds": execution_time,
                "iterations": stats.get("iterations", 0),
                "nodes_processed": stats.get("nodes_processed", 0),
                "errors": stats.get("errors", 0),
                "total_cost": stats.get("total_cost", 0)
            },
            "output_dir": str(output_dir)
        }

        if error:
            result["error"] = error

        # Collect graph physics metrics
        if status == "COMPLETED":
            try:
                from infrastructure.graph_db import GraphDB

                graph_db = GraphDB(persistence_path=".gaadp/graph.json")
                graph_metrics = GraphMetrics(
                    graph_db,
                    telemetry_path=str(telemetry_path)
                )
                metrics_report = graph_metrics.calculate_all(
                    run_id=self.run_id,
                    prompt=prompt
                )

                result["graph_physics"] = {
                    "tovs": metrics_report.tovs.score,
                    "icr": metrics_report.icr.score,
                    "gcr": metrics_report.gcr.score,
                    "dsr": metrics_report.dsr.score,
                    "details": {
                        "tovs_violations": len(metrics_report.tovs.violations),
                        "phantom_imports": len(metrics_report.icr.phantom_imports),
                        "largest_module_loc": metrics_report.gcr.largest_module.get("loc", 0),
                        "unsatisfied_deps": len(metrics_report.dsr.unsatisfied)
                    }
                }
            except Exception as e:
                logger.warning(f"Failed to collect graph metrics: {e}")
                result["graph_physics"] = {"error": str(e)}

        # Collect trajectory metrics
        if status == "COMPLETED" and telemetry_path.exists():
            try:
                trajectory = TrajectoryAnalyzer(str(telemetry_path))
                traj_report = trajectory.analyze()

                result["trajectory"] = {
                    "oscillations": traj_report.oscillations.total_oscillations,
                    "token_efficiency": traj_report.token_efficiency.ratio,
                    "cost_per_verified_node": traj_report.cost_efficiency.cost_per_verified_node,
                    "iteration_efficiency": traj_report.iteration_efficiency.ratio,
                    "details": {
                        "tokens_total": traj_report.token_efficiency.tokens_total,
                        "tokens_wasted": traj_report.token_efficiency.tokens_wasted,
                        "llm_calls": traj_report.llm_calls,
                        "state_transitions": traj_report.state_transitions
                    }
                }
            except Exception as e:
                logger.warning(f"Failed to collect trajectory metrics: {e}")
                result["trajectory"] = {"error": str(e)}

        return result

    async def run_all(
        self,
        tags: Optional[List[str]] = None,
        benchmark_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Run all benchmarks, optionally filtered."""
        benchmarks = self.suite.get("benchmarks", [])

        if benchmark_ids:
            benchmarks = [b for b in benchmarks if b["id"] in benchmark_ids]
        elif tags:
            benchmarks = [
                b for b in benchmarks
                if any(t in b.get("tags", []) for t in tags)
            ]

        logger.info(f"Running {len(benchmarks)} benchmark(s)")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Commit: {self.commit or 'unknown'}")

        for benchmark in benchmarks:
            result = await self.run_benchmark(benchmark)
            self.results.append(result)
            self._print_result(result)

        return self.results

    def _print_result(self, result: Dict):
        """Print a benchmark result summary."""
        bench_id = result.get("id", "?")
        status = result.get("status", "UNKNOWN")

        logger.info("-" * 60)
        logger.info(f"[{status}] {bench_id}")

        if status == "COMPLETED":
            exec_data = result.get("execution", {})
            logger.info(f"  Time: {exec_data.get('time_seconds', 0):.1f}s")
            logger.info(f"  Cost: ${exec_data.get('total_cost', 0):.4f}")
            logger.info(f"  Iterations: {exec_data.get('iterations', 0)}")

            # Graph Physics
            gp = result.get("graph_physics", {})
            if "error" not in gp:
                logger.info(f"  TOVS: {gp.get('tovs', 'N/A'):.3f}")
                logger.info(f"  ICR:  {gp.get('icr', 'N/A'):.3f}")
                logger.info(f"  GCR:  {gp.get('gcr', 'N/A'):.2f}")
                logger.info(f"  DSR:  {gp.get('dsr', 'N/A'):.3f}")

            # Trajectory
            traj = result.get("trajectory", {})
            if "error" not in traj:
                logger.info(f"  Oscillations: {traj.get('oscillations', 'N/A')}")
                logger.info(f"  Token Efficiency: {traj.get('token_efficiency', 'N/A'):.3f}")

        elif status == "ERROR":
            logger.error(f"  Error: {result.get('error', 'Unknown')[:100]}")

    def save_report(self) -> Path:
        """Save complete run report to JSON."""
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)

        report = {
            "run_id": self.run_id,
            "commit": self.commit,
            "timestamp": datetime.now().isoformat(),
            "suite": str(self.suite_path),
            "results": self.results,
            "summary": self._compute_summary()
        }

        report_path = report_dir / f"benchmark_{self.run_id}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved: {report_path}")
        return report_path

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute aggregate summary of all results."""
        completed = [r for r in self.results if r.get("status") == "COMPLETED"]
        errors = [r for r in self.results if r.get("status") == "ERROR"]

        if not completed:
            return {
                "total": len(self.results),
                "completed": 0,
                "errors": len(errors)
            }

        # Aggregate metrics
        def avg(key, subkey=None):
            values = []
            for r in completed:
                data = r.get(key, {})
                if subkey:
                    val = data.get(subkey)
                else:
                    val = data
                if val is not None and not isinstance(val, str):
                    values.append(val)
            return sum(values) / len(values) if values else None

        return {
            "total": len(self.results),
            "completed": len(completed),
            "errors": len(errors),
            "avg_time_seconds": avg("execution", "time_seconds"),
            "avg_cost": avg("execution", "total_cost"),
            "total_cost": sum(
                r.get("execution", {}).get("total_cost", 0)
                for r in completed
            ),
            "avg_tovs": avg("graph_physics", "tovs"),
            "avg_icr": avg("graph_physics", "icr"),
            "avg_gcr": avg("graph_physics", "gcr"),
            "avg_dsr": avg("graph_physics", "dsr"),
            "avg_oscillations": avg("trajectory", "oscillations"),
            "avg_token_efficiency": avg("trajectory", "token_efficiency")
        }

    def print_summary(self):
        """Print summary of all results."""
        summary = self._compute_summary()

        logger.info("")
        logger.info("=" * 60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Commit: {self.commit or 'unknown'}")
        logger.info("")
        logger.info(f"Total: {summary['total']}")
        logger.info(f"Completed: {summary['completed']}")
        logger.info(f"Errors: {summary['errors']}")

        if summary['completed'] > 0:
            logger.info("")
            logger.info("Averages:")
            if summary.get("avg_time_seconds"):
                logger.info(f"  Time: {summary['avg_time_seconds']:.1f}s")
            if summary.get("avg_cost"):
                logger.info(f"  Cost: ${summary['avg_cost']:.4f}")
            if summary.get("total_cost"):
                logger.info(f"  Total Cost: ${summary['total_cost']:.4f}")

            logger.info("")
            logger.info("Graph Physics (avg):")
            if summary.get("avg_tovs") is not None:
                logger.info(f"  TOVS: {summary['avg_tovs']:.3f}")
            if summary.get("avg_icr") is not None:
                logger.info(f"  ICR:  {summary['avg_icr']:.3f}")
            if summary.get("avg_gcr") is not None:
                logger.info(f"  GCR:  {summary['avg_gcr']:.2f}")
            if summary.get("avg_dsr") is not None:
                logger.info(f"  DSR:  {summary['avg_dsr']:.3f}")

            logger.info("")
            logger.info("Trajectory (avg):")
            if summary.get("avg_oscillations") is not None:
                logger.info(f"  Oscillations: {summary['avg_oscillations']:.1f}")
            if summary.get("avg_token_efficiency") is not None:
                logger.info(f"  Token Efficiency: {summary['avg_token_efficiency']:.3f}")

        logger.info("=" * 60)


def compare_runs(run_a_path: str, run_b_path: str):
    """Compare two benchmark runs."""
    with open(run_a_path) as f:
        run_a = json.load(f)
    with open(run_b_path) as f:
        run_b = json.load(f)

    print("\n" + "=" * 60)
    print("RUN COMPARISON")
    print("=" * 60)
    print(f"Run A: {run_a.get('run_id')} (commit: {run_a.get('commit', '?')})")
    print(f"Run B: {run_b.get('run_id')} (commit: {run_b.get('commit', '?')})")
    print()

    # Compare summaries
    sum_a = run_a.get("summary", {})
    sum_b = run_b.get("summary", {})

    metrics = [
        ("avg_time_seconds", "Avg Time (s)", "{:.1f}"),
        ("total_cost", "Total Cost ($)", "{:.4f}"),
        ("avg_tovs", "TOVS", "{:.3f}"),
        ("avg_icr", "ICR", "{:.3f}"),
        ("avg_gcr", "GCR", "{:.2f}"),
        ("avg_dsr", "DSR", "{:.3f}"),
        ("avg_oscillations", "Oscillations", "{:.1f}"),
        ("avg_token_efficiency", "Token Efficiency", "{:.3f}")
    ]

    print(f"{'Metric':<20} {'Run A':>12} {'Run B':>12} {'Delta':>12}")
    print("-" * 58)

    for key, label, fmt in metrics:
        val_a = sum_a.get(key)
        val_b = sum_b.get(key)

        if val_a is None and val_b is None:
            continue

        str_a = fmt.format(val_a) if val_a is not None else "N/A"
        str_b = fmt.format(val_b) if val_b is not None else "N/A"

        if val_a is not None and val_b is not None:
            delta = val_b - val_a
            delta_str = f"{delta:+.3f}"
        else:
            delta_str = "-"

        print(f"{label:<20} {str_a:>12} {str_b:>12} {delta_str:>12}")

    print("=" * 60)

    # Per-benchmark comparison
    results_a = {r["id"]: r for r in run_a.get("results", [])}
    results_b = {r["id"]: r for r in run_b.get("results", [])}

    all_ids = set(results_a.keys()) | set(results_b.keys())

    if len(all_ids) > 1:
        print("\nPer-Benchmark Changes:")
        for bench_id in sorted(all_ids):
            ra = results_a.get(bench_id, {})
            rb = results_b.get(bench_id, {})

            gp_a = ra.get("graph_physics", {})
            gp_b = rb.get("graph_physics", {})

            tovs_a = gp_a.get("tovs")
            tovs_b = gp_b.get("tovs")

            if tovs_a is not None and tovs_b is not None:
                delta = tovs_b - tovs_a
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
                print(f"  {bench_id}: TOVS {tovs_a:.3f} → {tovs_b:.3f} ({arrow})")


async def main():
    parser = argparse.ArgumentParser(
        description="GAADP Benchmark Suite - Rigorous Metrics Edition"
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Filter benchmarks by tags"
    )
    parser.add_argument(
        "--id",
        nargs="+",
        dest="benchmark_ids",
        help="Run specific benchmark(s) by ID"
    )
    parser.add_argument(
        "--suite",
        default="benchmarks/suite.yaml",
        help="Path to suite configuration file"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RUN_A", "RUN_B"),
        help="Compare two run report files"
    )

    args = parser.parse_args()

    # Compare mode
    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
        return

    # Run mode
    runner = BenchmarkRunner(suite_path=args.suite)

    await runner.run_all(
        tags=args.tags,
        benchmark_ids=args.benchmark_ids
    )

    runner.print_summary()
    report_path = runner.save_report()

    print(f"\nTo compare with another run:")
    print(f"  python scripts/benchmark_suite.py --compare {report_path} <other_report>")


if __name__ == "__main__":
    asyncio.run(main())
