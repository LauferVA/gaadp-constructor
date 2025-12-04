#!/usr/bin/env python3
"""
BENCHMARK COMPARISON SCRIPT

Runs the same task with Gen-1 (BASELINE) and Gen-2 (TREATMENT) pipelines,
displaying results side-by-side in the dev dashboard.

Usage:
    python scripts/benchmark_compare.py "Create a fibonacci function"
    python scripts/benchmark_compare.py -f test_prompts/fibonacci.txt

This script:
1. Starts VizServer in dev mode (comparison dashboard)
2. Runs Gen-1 pipeline (baseline) - RESEARCHER → VERIFIER only
3. Runs Gen-2 pipeline (treatment) - RESEARCHER → SPEC → BUILDER → TESTER → VERIFIER
4. Displays comparison metrics in the dashboard
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.graph_db import GraphDB
from infrastructure.graph_runtime import GraphRuntime
from infrastructure.llm_gateway import LLMGateway
from infrastructure.viz_server import start_viz_server, stop_viz_server, get_viz_server
from core.ontology import NodeType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("GAADP.Compare")


async def run_pipeline(
    session: str,
    requirement: str,
    viz_server,
    output_dir: Path,
    use_tdd: bool = False
) -> dict:
    """
    Run a single pipeline execution.

    Args:
        session: "baseline" or "treatment"
        requirement: The requirement to process
        viz_server: VizServer instance
        output_dir: Output directory for generated files
        use_tdd: If True, use Gen-2 TDD pipeline (BUILDER → TESTER → VERIFIER)

    Returns:
        Execution statistics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING {session.upper()} PIPELINE")
    if use_tdd:
        logger.info("Mode: Gen-2 TDD (BUILDER → TESTER → VERIFIER)")
    else:
        logger.info("Mode: Gen-1 (RESEARCHER → VERIFIER only)")
    logger.info(f"{'='*60}")

    # Switch to this session in VizServer
    viz_server.set_active_session(session)
    viz_server.reset_session(session)

    # Create isolated graph DB for this session
    session_output = output_dir / session
    session_output.mkdir(parents=True, exist_ok=True)

    graph_db = GraphDB(persistence_path=str(session_output / "graph.json"))
    gateway = LLMGateway()

    # TODO: Pass use_tdd flag to GraphRuntime when Gen-2 is implemented
    # For now, both pipelines use the same runtime
    runtime = GraphRuntime(
        graph_db=graph_db,
        llm_gateway=gateway,
        viz_server=viz_server,
        output_dir=session_output
    )

    # Create initial REQ node
    import uuid
    req_id = uuid.uuid4().hex
    graph_db.add_node(
        node_id=req_id,
        node_type=NodeType.REQ,
        content=requirement,
        metadata={
            'cost_limit': 5.0,
            'max_attempts': 3,
            'pipeline': 'gen2_tdd' if use_tdd else 'gen1'
        }
    )
    logger.info(f"Created REQ node: {req_id[:8]}")

    # Emit to visualization
    await viz_server.on_node_created(
        req_id, 'REQ', requirement,
        {'cost_limit': 5.0, 'max_attempts': 3, 'pipeline': 'gen2_tdd' if use_tdd else 'gen1'}
    )

    # Run execution
    logger.info(f"Starting {session} execution...")
    stats = await runtime.run_until_complete(max_iterations=50)

    # Collect results
    verified = graph_db.get_by_status("VERIFIED")
    failed = graph_db.get_by_status("FAILED")
    code_nodes = graph_db.get_materializable_nodes()

    stats['session'] = session
    stats['verified_count'] = len(verified)
    stats['failed_count'] = len(failed)
    stats['code_count'] = len(code_nodes) if code_nodes else 0

    logger.info(f"\n{session.upper()} RESULTS:")
    logger.info(f"  Iterations: {stats['iterations']}")
    logger.info(f"  Nodes processed: {stats['nodes_processed']}")
    logger.info(f"  Total cost: ${stats['total_cost']:.4f}")
    logger.info(f"  Verified: {stats['verified_count']}")
    logger.info(f"  Failed: {stats['failed_count']}")
    logger.info(f"  CODE nodes: {stats['code_count']}")

    # Broadcast completion
    await viz_server.on_complete(stats)

    return stats


async def main(requirement: str = None):
    """
    Main comparison runner.
    """
    if not requirement:
        logger.error("No requirement provided")
        return

    logger.info("="*60)
    logger.info("GAADP COMPARISON BENCHMARK")
    logger.info("Gen-1 (Baseline) vs Gen-2 (Treatment)")
    logger.info("="*60)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"workspace/compare_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_dir}")

    # Start VizServer in dev mode
    logger.info("Starting visualization server (comparison mode)...")
    viz_server = await start_viz_server(dev_mode=True)
    logger.info("Dashboard: http://localhost:8766 (comparison view)")
    await asyncio.sleep(2)  # Give browser time to open

    try:
        # Run baseline (Gen-1)
        baseline_stats = await run_pipeline(
            session="baseline",
            requirement=requirement,
            viz_server=viz_server,
            output_dir=output_dir,
            use_tdd=False
        )

        # Brief pause between runs
        await asyncio.sleep(1)

        # Run treatment (Gen-2 with TDD)
        treatment_stats = await run_pipeline(
            session="treatment",
            requirement=requirement,
            viz_server=viz_server,
            output_dir=output_dir,
            use_tdd=True
        )

        # Print comparison summary
        logger.info("\n" + "="*60)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*60)
        logger.info(f"{'Metric':<25} {'Baseline':>15} {'Treatment':>15} {'Delta':>15}")
        logger.info("-"*70)

        metrics = [
            ('Iterations', 'iterations'),
            ('Nodes Processed', 'nodes_processed'),
            ('Verified Nodes', 'verified_count'),
            ('Failed Nodes', 'failed_count'),
            ('CODE Nodes', 'code_count'),
            ('Total Cost ($)', 'total_cost'),
            ('Errors', 'errors'),
        ]

        for label, key in metrics:
            b_val = baseline_stats.get(key, 0)
            t_val = treatment_stats.get(key, 0)

            if isinstance(b_val, float):
                delta = t_val - b_val
                logger.info(f"{label:<25} {b_val:>15.4f} {t_val:>15.4f} {delta:>+15.4f}")
            else:
                delta = t_val - b_val
                logger.info(f"{label:<25} {b_val:>15} {t_val:>15} {delta:>+15}")

        logger.info("="*60)

        # Save comparison report
        report = {
            'timestamp': datetime.now().isoformat(),
            'requirement': requirement[:500],
            'output_dir': str(output_dir),
            'baseline': baseline_stats,
            'treatment': treatment_stats,
            'delta': {
                key: treatment_stats.get(key, 0) - baseline_stats.get(key, 0)
                for _, key in metrics
            }
        }

        report_path = output_dir / "comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved: {report_path}")

        # Keep visualization running
        logger.info("\nVisualization running. Press Ctrl+C to exit.")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass

    finally:
        logger.info("Shutting down...")
        await stop_viz_server()


def cli():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Run GAADP comparison benchmark (Gen-1 vs Gen-2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_compare.py "Create a fibonacci function"
  python scripts/benchmark_compare.py -f test_prompts/fibonacci.txt
        """
    )
    parser.add_argument(
        "requirement",
        nargs="?",
        help="The requirement to process"
    )
    parser.add_argument(
        "-f", "--file",
        help="Read requirement from file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get requirement
    requirement = None
    if args.file:
        requirement = Path(args.file).read_text()
    elif args.requirement:
        requirement = args.requirement
    else:
        parser.print_help()
        sys.exit(1)

    # Run
    try:
        asyncio.run(main(requirement))
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    cli()
