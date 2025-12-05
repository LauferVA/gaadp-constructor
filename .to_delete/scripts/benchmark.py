#!/usr/bin/env python3
"""
GAADP REGRESSION ENGINE - The Scientist
========================================
Runs A/B tests against Golden Sets and automatically detects regression.

Usage:
    python scripts/benchmark.py                    # Run benchmark (with viz)
    python scripts/benchmark.py --no-viz           # Run without visualization
    python scripts/benchmark.py --baseline         # Save as new baseline
    python scripts/benchmark.py --compare-only     # Compare without running
    python scripts/benchmark.py --task "Build X"   # Run single task

Output:
    - Results saved to .gaadp/benchmarks/latest.json
    - Compared against .gaadp/benchmarks/baseline.json
    - Red warnings if regression detected
    - Visualization at http://localhost:8765 (default)
"""
import os
import sys
import json
import yaml
import time
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ontology import NodeType, NodeStatus
from core.telemetry import TelemetryRecorder, get_recorder

logger = logging.getLogger("GAADP.Benchmark")


# =============================================================================
# CONFIGURATION
# =============================================================================

BENCHMARK_DIR = Path(".gaadp/benchmarks")
LATEST_FILE = BENCHMARK_DIR / "latest.json"
BASELINE_FILE = BENCHMARK_DIR / "baseline.json"
GOLDEN_SET_PATH = Path("tests/golden_set.yaml")

# Regression thresholds
REGRESSION_THRESHOLDS = {
    "success_rate_drop": 0.10,      # Alert if success rate drops by 10%
    "cycle_increase": 0.25,          # Alert if cycles increase by 25%
    "cost_increase": 0.50,           # Alert if cost increases by 50%
    "error_increase": 0.20,          # Alert if error rate increases by 20%
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class TaskType(str, Enum):
    """Types of benchmark tasks."""
    BUILDER = "builder"      # Code generation
    ARCHITECT = "architect"  # Decomposition
    GRAPH = "graph"          # Multi-file dependencies


@dataclass
class GoldenTask:
    """A task from the golden set."""
    id: str
    name: str
    task_type: TaskType
    input_requirement: str
    expected_criteria: List[str]
    max_cycles: int = 10
    max_cost_usd: float = 0.50


@dataclass
class TaskResult:
    """Result of running a single task."""
    task_id: str
    task_name: str
    success: bool
    criteria_met: Dict[str, bool]
    cycles: int
    cost_usd: float
    duration_ms: float
    nodes_created: int
    nodes_verified: int
    nodes_failed: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Complete benchmark run report."""
    timestamp: str
    commit_hash: Optional[str]
    tasks_run: int
    tasks_passed: int
    tasks_failed: int
    total_cycles: int
    total_cost_usd: float
    total_duration_ms: float
    success_rate: float
    avg_cycles_per_task: float
    task_results: List[TaskResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['task_results'] = [r.to_dict() if hasattr(r, 'to_dict') else r for r in self.task_results]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkReport":
        results = data.pop('task_results', [])
        report = cls(**data)
        report.task_results = [
            TaskResult(**r) if isinstance(r, dict) else r
            for r in results
        ]
        return report


# =============================================================================
# GOLDEN SET LOADER
# =============================================================================

def load_golden_set(path: Path = GOLDEN_SET_PATH) -> List[GoldenTask]:
    """
    Load tasks from golden_set.yaml.

    Expected format:
        tasks:
          - id: task_001
            name: Fibonacci with error handling
            type: builder
            input: "Create fib.py that..."
            criteria:
              - "VERIFIED status"
              - "contains 'raise ValueError'"
    """
    if not path.exists():
        logger.warning(f"Golden set not found: {path}")
        return []

    with open(path) as f:
        data = yaml.safe_load(f)

    tasks = []
    for task_data in data.get('tasks', []):
        task = GoldenTask(
            id=task_data['id'],
            name=task_data['name'],
            task_type=TaskType(task_data.get('type', 'builder')),
            input_requirement=task_data['input'],
            expected_criteria=task_data.get('criteria', []),
            max_cycles=task_data.get('max_cycles', 10),
            max_cost_usd=task_data.get('max_cost_usd', 0.50)
        )
        tasks.append(task)

    logger.info(f"Loaded {len(tasks)} tasks from golden set")
    return tasks


# =============================================================================
# TASK RUNNER
# =============================================================================

async def run_task(task: GoldenTask, dry_run: bool = False, viz_server=None) -> TaskResult:
    """
    Execute a single golden set task.

    Args:
        task: The task to run
        dry_run: If True, simulate without API calls
    """
    logger.info(f"Running task: {task.name}")
    start_time = time.time()

    # Initialize components
    from infrastructure.graph_db import GraphDB
    from infrastructure.graph_runtime import GraphRuntime

    # Use isolated graph for each task
    graph_path = BENCHMARK_DIR / f"task_{task.id}.json"
    db = GraphDB(persistence_path=str(graph_path))

    # Get telemetry
    TelemetryRecorder.reset_instance()
    recorder = TelemetryRecorder.get_instance()

    try:
        if dry_run:
            # Simulate result
            return TaskResult(
                task_id=task.id,
                task_name=task.name,
                success=True,
                criteria_met={c: True for c in task.expected_criteria},
                cycles=3,
                cost_usd=0.01,
                duration_ms=1000.0,
                nodes_created=5,
                nodes_verified=3,
                nodes_failed=0
            )

        # Initialize runtime with LLM gateway
        from infrastructure.llm_gateway import LLMGateway
        try:
            gateway = LLMGateway()
        except Exception as e:
            logger.warning(f"LLM Gateway not available: {e}")
            gateway = None

        runtime = GraphRuntime(db, llm_gateway=gateway, viz_server=viz_server)

        # Inject requirement
        db.add_node(
            node_id=f"req_{task.id}",
            node_type=NodeType.REQ,
            content=task.input_requirement,
            metadata={"cost_limit": task.max_cost_usd}
        )

        # Run until complete or max cycles
        stats = await runtime.run_until_complete(max_iterations=task.max_cycles)

        # Evaluate criteria
        criteria_met = evaluate_criteria(db, task.expected_criteria)

        # Calculate success
        success = all(criteria_met.values()) and stats['errors'] == 0

        duration_ms = (time.time() - start_time) * 1000
        metrics = recorder.get_session_metrics()

        return TaskResult(
            task_id=task.id,
            task_name=task.name,
            success=success,
            criteria_met=criteria_met,
            cycles=stats['iterations'],
            cost_usd=metrics.total_cost_usd,
            duration_ms=duration_ms,
            nodes_created=metrics.nodes_created,
            nodes_verified=metrics.nodes_verified,
            nodes_failed=metrics.nodes_failed
        )

    except Exception as e:
        logger.error(f"Task {task.id} failed with error: {e}")
        duration_ms = (time.time() - start_time) * 1000

        return TaskResult(
            task_id=task.id,
            task_name=task.name,
            success=False,
            criteria_met={c: False for c in task.expected_criteria},
            cycles=0,
            cost_usd=0.0,
            duration_ms=duration_ms,
            nodes_created=0,
            nodes_verified=0,
            nodes_failed=0,
            error=str(e)
        )

    finally:
        recorder.flush()
        # Cleanup task graph
        try:
            graph_path.unlink()
        except:
            pass


def evaluate_criteria(db, criteria: List[str]) -> Dict[str, bool]:
    """
    Evaluate expected criteria against the graph.

    Supported criteria:
    - "VERIFIED status" - At least one CODE node is VERIFIED
    - "contains 'X'" - Some CODE node contains string X
    - "SPEC node created" - At least one SPEC node exists
    - "DEPENDS_ON edge exists" - At least one DEPENDS_ON edge exists
    """
    results = {}

    for criterion in criteria:
        criterion_lower = criterion.lower()

        if "verified status" in criterion_lower:
            # Check for verified CODE nodes
            verified = [
                n for n, d in db.graph.nodes(data=True)
                if d.get('type') == NodeType.CODE.value
                and d.get('status') == NodeStatus.VERIFIED.value
            ]
            results[criterion] = len(verified) > 0

        elif criterion_lower.startswith("contains '"):
            # Check if any CODE node contains the string
            search_str = criterion.split("'")[1]
            found = False
            for n, d in db.graph.nodes(data=True):
                if d.get('type') == NodeType.CODE.value:
                    content = d.get('content', '')
                    if search_str.lower() in content.lower():
                        found = True
                        break
            results[criterion] = found

        elif "spec node" in criterion_lower:
            # Check for SPEC nodes
            specs = [
                n for n, d in db.graph.nodes(data=True)
                if d.get('type') == NodeType.SPEC.value
            ]
            results[criterion] = len(specs) > 0

        elif "depends_on" in criterion_lower:
            # Check for DEPENDS_ON edges
            from core.ontology import EdgeType
            found = False
            for u, v, d in db.graph.edges(data=True):
                if d.get('type') == EdgeType.DEPENDS_ON.value:
                    found = True
                    break
            results[criterion] = found

        else:
            # Unknown criterion - mark as not evaluated
            logger.warning(f"Unknown criterion: {criterion}")
            results[criterion] = False

    return results


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

async def run_benchmark(
    tasks: Optional[List[GoldenTask]] = None,
    dry_run: bool = False,
    viz_server=None
) -> BenchmarkReport:
    """
    Run the complete benchmark suite.

    Args:
        tasks: Optional list of tasks (loads from golden set if not provided)
        dry_run: If True, simulate without API calls
    """
    if tasks is None:
        tasks = load_golden_set()

    if not tasks:
        logger.warning("No tasks to run")
        return BenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            commit_hash=get_git_commit(),
            tasks_run=0,
            tasks_passed=0,
            tasks_failed=0,
            total_cycles=0,
            total_cost_usd=0.0,
            total_duration_ms=0.0,
            success_rate=0.0,
            avg_cycles_per_task=0.0
        )

    # Ensure benchmark directory exists
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    results: List[TaskResult] = []
    total_cycles = 0
    total_cost = 0.0
    total_duration = 0.0

    for task in tasks:
        result = await run_task(task, dry_run=dry_run, viz_server=viz_server)
        results.append(result)
        total_cycles += result.cycles
        total_cost += result.cost_usd
        total_duration += result.duration_ms

    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed

    report = BenchmarkReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        commit_hash=get_git_commit(),
        tasks_run=len(results),
        tasks_passed=passed,
        tasks_failed=failed,
        total_cycles=total_cycles,
        total_cost_usd=total_cost,
        total_duration_ms=total_duration,
        success_rate=passed / len(results) if results else 0.0,
        avg_cycles_per_task=total_cycles / len(results) if results else 0.0,
        task_results=results
    )

    return report


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except:
        return None


# =============================================================================
# COMPARISON & REGRESSION DETECTION
# =============================================================================

def compare_reports(
    current: BenchmarkReport,
    baseline: BenchmarkReport
) -> Dict[str, Any]:
    """
    Compare current results with baseline.

    Returns:
        Dictionary with comparison results and regression flags
    """
    comparison = {
        "baseline_commit": baseline.commit_hash,
        "current_commit": current.commit_hash,
        "baseline_timestamp": baseline.timestamp,
        "current_timestamp": current.timestamp,
        "metrics": {},
        "regressions": [],
        "improvements": []
    }

    # Success rate comparison
    sr_delta = current.success_rate - baseline.success_rate
    comparison["metrics"]["success_rate"] = {
        "baseline": baseline.success_rate,
        "current": current.success_rate,
        "delta": sr_delta,
        "delta_pct": sr_delta / baseline.success_rate if baseline.success_rate > 0 else 0
    }
    if sr_delta < -REGRESSION_THRESHOLDS["success_rate_drop"]:
        comparison["regressions"].append(
            f"SUCCESS RATE DROPPED: {baseline.success_rate:.1%} → {current.success_rate:.1%}"
        )
    elif sr_delta > REGRESSION_THRESHOLDS["success_rate_drop"]:
        comparison["improvements"].append(
            f"Success rate improved: {baseline.success_rate:.1%} → {current.success_rate:.1%}"
        )

    # Cycle efficiency comparison
    cycle_delta = current.avg_cycles_per_task - baseline.avg_cycles_per_task
    cycle_delta_pct = cycle_delta / baseline.avg_cycles_per_task if baseline.avg_cycles_per_task > 0 else 0
    comparison["metrics"]["avg_cycles"] = {
        "baseline": baseline.avg_cycles_per_task,
        "current": current.avg_cycles_per_task,
        "delta": cycle_delta,
        "delta_pct": cycle_delta_pct
    }
    if cycle_delta_pct > REGRESSION_THRESHOLDS["cycle_increase"]:
        comparison["regressions"].append(
            f"CYCLES INCREASED: {baseline.avg_cycles_per_task:.1f} → {current.avg_cycles_per_task:.1f}"
        )
    elif cycle_delta_pct < -0.10:  # 10% improvement
        comparison["improvements"].append(
            f"Cycles reduced: {baseline.avg_cycles_per_task:.1f} → {current.avg_cycles_per_task:.1f}"
        )

    # Cost comparison
    cost_delta_pct = (current.total_cost_usd - baseline.total_cost_usd) / baseline.total_cost_usd if baseline.total_cost_usd > 0 else 0
    comparison["metrics"]["total_cost"] = {
        "baseline": baseline.total_cost_usd,
        "current": current.total_cost_usd,
        "delta": current.total_cost_usd - baseline.total_cost_usd,
        "delta_pct": cost_delta_pct
    }
    if cost_delta_pct > REGRESSION_THRESHOLDS["cost_increase"]:
        comparison["regressions"].append(
            f"COST INCREASED: ${baseline.total_cost_usd:.2f} → ${current.total_cost_usd:.2f}"
        )

    # Per-task comparison
    comparison["task_comparison"] = []
    baseline_by_id = {r.task_id: r for r in baseline.task_results}
    for result in current.task_results:
        baseline_result = baseline_by_id.get(result.task_id)
        if baseline_result:
            comparison["task_comparison"].append({
                "task_id": result.task_id,
                "baseline_success": baseline_result.success,
                "current_success": result.success,
                "regressed": baseline_result.success and not result.success,
                "improved": not baseline_result.success and result.success
            })

    return comparison


def print_comparison(comparison: Dict[str, Any]):
    """Print comparison results with colored output."""
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)

    print(f"\nBaseline: {comparison['baseline_commit']} ({comparison['baseline_timestamp'][:10]})")
    print(f"Current:  {comparison['current_commit']} ({comparison['current_timestamp'][:10]})")

    print("\n--- Metrics ---")
    for metric, data in comparison["metrics"].items():
        delta_str = f"+{data['delta_pct']:.1%}" if data['delta_pct'] > 0 else f"{data['delta_pct']:.1%}"
        print(f"  {metric}: {data['baseline']:.2f} → {data['current']:.2f} ({delta_str})")

    if comparison["regressions"]:
        print("\n\033[91m⚠️  REGRESSIONS DETECTED:\033[0m")
        for reg in comparison["regressions"]:
            print(f"  \033[91m• {reg}\033[0m")

    if comparison["improvements"]:
        print("\n\033[92m✅ IMPROVEMENTS:\033[0m")
        for imp in comparison["improvements"]:
            print(f"  \033[92m• {imp}\033[0m")

    if not comparison["regressions"] and not comparison["improvements"]:
        print("\n\033[93m→ No significant changes detected\033[0m")

    print("\n" + "=" * 70)


# =============================================================================
# FILE I/O
# =============================================================================

def save_report(report: BenchmarkReport, path: Path):
    """Save report to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"Saved report to {path}")


def load_report(path: Path) -> Optional[BenchmarkReport]:
    """Load report from JSON file."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return BenchmarkReport.from_dict(data)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GAADP Regression Benchmark")
    parser.add_argument("--baseline", action="store_true",
                       help="Save results as new baseline")
    parser.add_argument("--compare-only", action="store_true",
                       help="Compare existing results without running")
    parser.add_argument("--dry-run", action="store_true",
                       help="Simulate without API calls")
    parser.add_argument("--task", type=str,
                       help="Run single task by name")
    parser.add_argument("--no-viz", action="store_true",
                       help="Disable visualization server (enabled by default)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    if args.compare_only:
        # Just compare existing reports
        current = load_report(LATEST_FILE)
        baseline = load_report(BASELINE_FILE)

        if not current:
            print("No current results found. Run benchmark first.")
            return 1
        if not baseline:
            print("No baseline found. Run with --baseline to create one.")
            return 1

        comparison = compare_reports(current, baseline)
        print_comparison(comparison)
        return 0 if not comparison["regressions"] else 1

    # Run benchmark
    print("\n🧪 GAADP BENCHMARK RUNNER\n")

    tasks = load_golden_set()
    if args.task:
        tasks = [t for t in tasks if args.task.lower() in t.name.lower()]
        if not tasks:
            print(f"No task matching '{args.task}' found")
            return 1

    print(f"Running {len(tasks)} task(s)...")
    if args.dry_run:
        print("(DRY RUN - no API calls)")

    # Run with or without viz server
    async def run_with_viz():
        viz_server = None
        if not args.no_viz:
            try:
                from infrastructure.viz_server import VizServer
                viz_server = VizServer(port=8765)
                await viz_server.start()
                print("🔮 Visualization: http://localhost:8765")
            except Exception as e:
                logger.warning(f"Could not start viz server: {e}")
                viz_server = None

        try:
            return await run_benchmark(tasks, dry_run=args.dry_run, viz_server=viz_server)
        finally:
            if viz_server:
                try:
                    await viz_server.stop()
                except:
                    pass

    report = asyncio.run(run_with_viz())

    # Print summary
    print("\n--- RESULTS ---")
    print(f"Tasks Run:    {report.tasks_run}")
    print(f"Passed:       {report.tasks_passed}")
    print(f"Failed:       {report.tasks_failed}")
    print(f"Success Rate: {report.success_rate:.1%}")
    print(f"Avg Cycles:   {report.avg_cycles_per_task:.1f}")
    print(f"Total Cost:   ${report.total_cost_usd:.4f}")
    print(f"Duration:     {report.total_duration_ms/1000:.1f}s")

    # Save results
    save_report(report, LATEST_FILE)

    if args.baseline:
        save_report(report, BASELINE_FILE)
        print(f"\n✅ Saved as baseline: {BASELINE_FILE}")
    else:
        # Compare with baseline if exists
        baseline = load_report(BASELINE_FILE)
        if baseline:
            comparison = compare_reports(report, baseline)
            print_comparison(comparison)
            if comparison["regressions"]:
                return 1
        else:
            print("\nℹ️  No baseline exists. Run with --baseline to create one.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
