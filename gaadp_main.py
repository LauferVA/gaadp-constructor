#!/usr/bin/env python3
"""
GAADP MAIN - Minimal Entry Point

This is the new graph-first entry point that uses:
- GraphRuntime (the execution engine)
- GenericAgent (universal agent)
- TransitionMatrix (the physics)

Usage:
    # Production mode (default) - lean execution
    python gaadp_main.py "Create a hello world function"

    # Development mode - full instrumentation for factory improvement
    python gaadp_main.py --dev "Create a hello world function"

Modes:
    Production (default):
        - Output to current directory (or --output-dir)
        - Normal logging
        - No visualization
        - Optimized for widget manufacturing

    Development (--dev):
        - Output to workspace/ (isolated)
        - Verbose logging
        - Real-time visualization
        - Telemetry recording
        - Run reports saved to reports/
        - Used when improving the factory itself
"""
import asyncio
import logging
import sys
from pathlib import Path

from infrastructure.graph_db import GraphDB
from infrastructure.graph_runtime import GraphRuntime
from infrastructure.llm_gateway import LLMGateway
from core.ontology import NodeType


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("GAADP.Main")


async def main(requirement: str = None, enable_viz: bool = False, output_dir: str = None, dev_mode: bool = False) -> dict:
    """
    Main entry point for GAADP.

    Args:
        requirement: The user requirement to process.
        enable_viz: If True, start the visualization server.
        output_dir: Directory to write generated code files.
        dev_mode: If True, enable all development instrumentation.

    Returns:
        Execution statistics
    """
    from datetime import datetime
    import json

    # Dev mode enables multiple features
    if dev_mode:
        enable_viz = True  # Always enable viz in dev mode
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("[DEV MODE] Factory improvement mode enabled")

    # Set output directory based on mode
    if output_dir is not None:
        output_dir = Path(output_dir)
    elif dev_mode:
        # Dev mode: isolated workspace with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"workspace/run_{timestamp}")
    else:
        # Prod mode: current directory
        output_dir = Path(".")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    viz_server = None

    # Start visualization server if requested
    if enable_viz:
        try:
            from infrastructure.viz_server import start_viz_server
            logger.info("Starting visualization server...")
            # In dev mode, use comparison dashboard with session support
            viz_server = await start_viz_server(dev_mode=dev_mode)
            if dev_mode:
                logger.info("Dashboard: http://localhost:8766 (comparison mode)")
            else:
                logger.info("Dashboard: http://localhost:8766")
            # Give browser time to open
            await asyncio.sleep(1)
        except ImportError as e:
            logger.warning(f"Visualization not available: {e}")
            logger.warning("Install websockets: pip install websockets")

    logger.info("=" * 60)
    logger.info("GAADP - Graph-First Architecture")
    if enable_viz:
        logger.info("Real-time visualization enabled")
    logger.info("=" * 60)

    # Initialize components
    graph_db = GraphDB(persistence_path=".gaadp/graph.json")
    gateway = LLMGateway()
    runtime = GraphRuntime(graph_db=graph_db, llm_gateway=gateway, viz_server=viz_server, output_dir=output_dir)

    # Create initial REQ node if requirement provided
    if requirement:
        import uuid
        req_id = uuid.uuid4().hex
        graph_db.add_node(
            node_id=req_id,
            node_type=NodeType.REQ,
            content=requirement,
            metadata={
                'cost_limit': 5.0,  # $5 max for this requirement
                'max_attempts': 3
            }
        )
        logger.info(f"Created REQ node: {req_id[:8]}")
        logger.info(f"Requirement: {requirement[:100]}...")

        # Emit to visualization
        if viz_server:
            await viz_server.on_node_created(
                req_id, 'REQ', requirement,
                {'cost_limit': 5.0, 'max_attempts': 3}
            )

    # Show initial state
    pending = graph_db.get_by_status("PENDING")
    logger.info(f"Pending nodes: {len(pending)}")

    # Run until complete
    logger.info("-" * 60)
    logger.info("Starting execution...")
    stats = await runtime.run_until_complete(max_iterations=50)

    # Report results
    logger.info("-" * 60)
    logger.info("Execution complete")
    logger.info(f"  Iterations: {stats['iterations']}")
    logger.info(f"  Nodes processed: {stats['nodes_processed']}")
    logger.info(f"  Total cost: ${stats['total_cost']:.4f}")
    logger.info(f"  Errors: {stats['errors']}")

    # Show final state
    verified = graph_db.get_by_status("VERIFIED")
    failed = graph_db.get_by_status("FAILED")
    logger.info(f"  Verified nodes: {len(verified)}")
    logger.info(f"  Failed nodes: {len(failed)}")

    # Materialize code
    code_nodes = graph_db.get_materializable_nodes()
    if code_nodes:
        logger.info(f"\nMaterializable code artifacts: {len(code_nodes)}")
        for node in code_nodes:
            logger.info(f"  - {node['file_path']}")

    # Dev mode: save run report
    if dev_mode:
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"run_{timestamp}.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "mode": "dev",
            "requirement": requirement[:500] if requirement else None,
            "output_dir": str(output_dir),
            "stats": stats,
            "verified_nodes": len(verified),
            "failed_nodes": len(failed),
            "code_artifacts": [n.get('file_path', 'unknown') for n in code_nodes] if code_nodes else []
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"[DEV MODE] Run report saved: {report_path}")

    # Keep visualization running if enabled
    if viz_server and stats['nodes_processed'] > 0:
        logger.info("\nVisualization server still running. Press Ctrl+C to exit.")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down visualization server...")
            from infrastructure.viz_server import stop_viz_server
            await stop_viz_server()

    return stats


def cli():
    """Command line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GAADP - Graph-Aware Autonomous Development Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production mode - output to current directory
  python gaadp_main.py "Create a hello world function"

  # Development mode - workspace isolation, viz, telemetry
  python gaadp_main.py --dev "Create a hello world function"

  # Read from file
  python gaadp_main.py --dev -f test_prompts/asteroid_game.txt

Modes:
  Production (default): Lean execution, output to current dir
  Development (--dev):  Full instrumentation for factory improvement
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
        "--dev",
        action="store_true",
        help="Development mode: viz, verbose logging, workspace isolation, run reports"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging (automatic in --dev mode)"
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable visualization only (automatic in --dev mode)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Override output directory (default: '.' in prod, 'workspace/run_*' in dev)"
    )

    args = parser.parse_args()

    # Verbose can be set independently of dev mode
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get requirement
    requirement = None
    if args.file:
        requirement = Path(args.file).read_text()
    elif args.requirement:
        requirement = args.requirement

    # Run
    try:
        stats = asyncio.run(main(
            requirement,
            enable_viz=args.viz,
            output_dir=args.output_dir,
            dev_mode=args.dev
        ))
        sys.exit(1 if stats['errors'] > 0 else 0)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    cli()
