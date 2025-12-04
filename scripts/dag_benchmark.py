#!/usr/bin/env python3
"""
DAG BENCHMARK RUNNER - Generate DAGs for All 30 Tasks
======================================================

Runs all 30 tasks from the golden set and exports DAGs for each.

Usage:
    python scripts/dag_benchmark.py                    # Run all 30 tasks
    python scripts/dag_benchmark.py --start 11        # Start from task 11
    python scripts/dag_benchmark.py --count 5         # Run only 5 tasks
    python scripts/dag_benchmark.py --task-id builder_003  # Run specific task
"""
import os
import sys
import json
import yaml
import asyncio
import argparse
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.graph_db import GraphDB
from infrastructure.graph_runtime import GraphRuntime
from infrastructure.llm_gateway import LLMGateway
from infrastructure.viz_server import (
    VizServer, start_viz_server, stop_viz_server,
    get_viz_server, export_dag_to_json, DAGExporter
)
from core.ontology import NodeType, NodeStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("DAG.Benchmark")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TaskResult:
    """Result of running a single task."""
    task_id: str
    task_name: str
    task_type: str
    success: bool
    nodes_created: int
    nodes_verified: int
    nodes_failed: int
    edges_created: int
    cycles: int
    cost_usd: float
    duration_seconds: float
    dag_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Summary of the full benchmark run."""
    run_id: str
    timestamp: str
    total_tasks: int
    tasks_completed: int
    tasks_failed: int
    total_cost_usd: float
    total_duration_seconds: float
    dag_dir: str
    results: List[TaskResult] = field(default_factory=list)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class DAGBenchmarkRunner:
    """Runs tasks and exports DAGs."""

    def __init__(
        self,
        golden_set_path: Path = None,
        output_dir: Path = None,
        enable_viz: bool = False
    ):
        self.golden_set_path = golden_set_path or PROJECT_ROOT / "tests" / "golden_set_30.yaml"
        self.output_dir = output_dir or PROJECT_ROOT / "logs" / "dag_benchmark"
        self.enable_viz = enable_viz

        # Create output directories
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.run_id}"
        self.dag_dir = self.run_dir / "dags"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.dag_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[TaskResult] = []
        self.dag_exporter = DAGExporter(self.dag_dir)

    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks from golden set YAML."""
        with open(self.golden_set_path) as f:
            data = yaml.safe_load(f)
        return data.get("tasks", [])

    async def run_single_task(self, task: Dict[str, Any]) -> TaskResult:
        """Run a single task and export its DAG."""
        task_id = task["id"]
        task_name = task["name"]
        task_type = task.get("type", "builder")
        requirement = task["input"]

        logger.info(f"[{task_id}] Starting: {task_name}")
        start_time = datetime.now()

        try:
            # Create output directory for this task
            task_output_dir = self.run_dir / "outputs" / task_id
            task_output_dir.mkdir(parents=True, exist_ok=True)

            # Create fresh graph, gateway and runtime for this task
            graph = GraphDB()
            gateway = LLMGateway()

            # Connect to viz server if running
            viz = get_viz_server()
            if viz:
                # Reset viz state for new task
                viz.graph_state = {
                    "nodes": {},
                    "edges": [],
                    "agents": {},
                    "events": [],
                    "stats": {
                        "total_cost": 0.0,
                        "nodes_created": 0,
                        "nodes_verified": 0,
                        "nodes_failed": 0,
                        "iterations": 0
                    }
                }

            # Create runtime with proper initialization
            runtime = GraphRuntime(
                graph_db=graph,
                llm_gateway=gateway,
                viz_server=viz,
                output_dir=task_output_dir
            )

            # Add initial REQ node
            req_id = uuid.uuid4().hex
            graph.add_node(
                node_id=req_id,
                node_type=NodeType.REQ,
                content=requirement,
                metadata={
                    'cost_limit': 5.0,
                    'max_attempts': 3,
                    'task_id': task_id
                }
            )
            logger.info(f"[{task_id}] Created REQ node: {req_id[:8]}")

            # Emit to visualization
            if viz:
                await viz.on_node_created(
                    req_id, 'REQ', requirement,
                    {'cost_limit': 5.0, 'max_attempts': 3}
                )

            # Run the task
            max_cycles = task.get("max_cycles", 20)
            result = await runtime.run_until_complete(max_iterations=max_cycles)

            # Collect stats
            duration = (datetime.now() - start_time).total_seconds()
            # Compute stats from graph directly
            stats = {
                "total_nodes": graph.graph.number_of_nodes(),
                "total_edges": graph.graph.number_of_edges(),
                "total_cost": result.get("total_cost", 0.0) if isinstance(result, dict) else 0.0
            }

            # Export DAG
            dag_filename = f"dag_{task_id}_{task_name.replace(' ', '_').replace('/', '_')[:30]}.json"
            dag_path = self.dag_dir / dag_filename

            # Build DAG export data
            dag_data = {
                "metadata": {
                    "task_id": task_id,
                    "task_name": task_name,
                    "task_type": task_type,
                    "requirement": requirement[:500],
                    "exported_at": datetime.now().isoformat(),
                    "duration_seconds": duration,
                    "stats": stats
                },
                "nodes": [],
                "edges": []
            }

            # Export nodes
            for node_id in graph.graph.nodes():
                node_data = graph.graph.nodes[node_id]
                dag_data["nodes"].append({
                    "id": node_id,
                    "type": node_data.get("type"),
                    "status": node_data.get("status"),
                    "content": node_data.get("content", "")[:500],
                    "metadata": node_data.get("metadata", {})
                })

            # Export edges
            for source, target, edge_data in graph.graph.edges(data=True):
                dag_data["edges"].append({
                    "source": source,
                    "target": target,
                    "type": edge_data.get("type", "UNKNOWN")
                })

            # Save DAG
            with open(dag_path, "w") as f:
                json.dump(dag_data, f, indent=2, default=str)

            logger.info(f"[{task_id}] DAG exported to {dag_path}")

            # Determine success
            verified_count = sum(1 for n in graph.graph.nodes()
                               if graph.graph.nodes[n].get("status") == NodeStatus.VERIFIED.value)
            success = verified_count > 0

            task_result = TaskResult(
                task_id=task_id,
                task_name=task_name,
                task_type=task_type,
                success=success,
                nodes_created=stats.get("total_nodes", 0),
                nodes_verified=verified_count,
                nodes_failed=sum(1 for n in graph.graph.nodes()
                               if graph.graph.nodes[n].get("status") == NodeStatus.FAILED.value),
                edges_created=stats.get("total_edges", 0),
                cycles=result.get("iterations", 0) if isinstance(result, dict) else 0,
                cost_usd=result.get("total_cost", 0.0) if isinstance(result, dict) else 0.0,
                duration_seconds=duration,
                dag_path=str(dag_path)
            )

            logger.info(f"[{task_id}] {'PASS' if success else 'FAIL'} - "
                       f"{task_result.nodes_created} nodes, "
                       f"{task_result.nodes_verified} verified, "
                       f"${task_result.cost_usd:.4f}")

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"[{task_id}] Error: {e}")

            task_result = TaskResult(
                task_id=task_id,
                task_name=task_name,
                task_type=task_type,
                success=False,
                nodes_created=0,
                nodes_verified=0,
                nodes_failed=0,
                edges_created=0,
                cycles=0,
                cost_usd=0.0,
                duration_seconds=duration,
                error=str(e)
            )

        return task_result

    async def run_benchmark(
        self,
        start_index: int = 0,
        count: Optional[int] = None,
        task_id: Optional[str] = None
    ) -> BenchmarkSummary:
        """Run the full benchmark."""
        tasks = self.load_tasks()

        # Filter tasks
        if task_id:
            tasks = [t for t in tasks if t["id"] == task_id]
            if not tasks:
                logger.error(f"Task {task_id} not found")
                return None
        else:
            tasks = tasks[start_index:]
            if count:
                tasks = tasks[:count]

        logger.info(f"Running {len(tasks)} tasks")
        logger.info(f"Output directory: {self.run_dir}")

        # Start viz server if enabled
        if self.enable_viz:
            await start_viz_server()
            logger.info("Visualization server started at http://localhost:8766")

        start_time = datetime.now()

        for i, task in enumerate(tasks):
            logger.info(f"\n{'='*60}")
            logger.info(f"Task {i+1}/{len(tasks)}: {task['name']}")
            logger.info(f"{'='*60}")

            result = await self.run_single_task(task)
            self.results.append(result)

            # Small delay between tasks
            await asyncio.sleep(1)

        # Stop viz server
        if self.enable_viz:
            await stop_viz_server()

        total_duration = (datetime.now() - start_time).total_seconds()

        # Create summary
        summary = BenchmarkSummary(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            total_tasks=len(tasks),
            tasks_completed=sum(1 for r in self.results if r.success),
            tasks_failed=sum(1 for r in self.results if not r.success),
            total_cost_usd=sum(r.cost_usd for r in self.results),
            total_duration_seconds=total_duration,
            dag_dir=str(self.dag_dir),
            results=self.results
        )

        # Save summary
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(asdict(summary), f, indent=2, default=str)

        logger.info(f"\n{'='*60}")
        logger.info("BENCHMARK COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Tasks: {summary.tasks_completed}/{summary.total_tasks} passed")
        logger.info(f"Total cost: ${summary.total_cost_usd:.4f}")
        logger.info(f"Duration: {summary.total_duration_seconds:.1f}s")
        logger.info(f"DAGs saved to: {self.dag_dir}")
        logger.info(f"Summary: {summary_path}")

        return summary


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Run DAG benchmark on golden set")
    parser.add_argument("--start", type=int, default=0, help="Start from task index")
    parser.add_argument("--count", type=int, help="Number of tasks to run")
    parser.add_argument("--task-id", type=str, help="Run specific task by ID")
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    parser.add_argument("--golden-set", type=str, help="Path to golden set YAML")
    args = parser.parse_args()

    golden_set_path = Path(args.golden_set) if args.golden_set else None

    runner = DAGBenchmarkRunner(
        golden_set_path=golden_set_path,
        enable_viz=args.viz
    )

    await runner.run_benchmark(
        start_index=args.start,
        count=args.count,
        task_id=args.task_id
    )


if __name__ == "__main__":
    asyncio.run(main())
