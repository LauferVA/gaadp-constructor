#!/usr/bin/env python3
"""
GAADP MAIN - Minimal Entry Point

This is the new graph-first entry point that uses:
- GraphRuntime (the execution engine)
- GenericAgent (universal agent)
- TransitionMatrix (the physics)

Usage:
    python main.py "your requirement here"
    python main.py --viz "your requirement here"  # With real-time visualization
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


async def main(requirement: str = None, enable_viz: bool = False) -> dict:
    """
    Main entry point for GAADP.

    Args:
        requirement: The user requirement to process.
        enable_viz: If True, start the visualization server.

    Returns:
        Execution statistics
    """
    viz_server = None

    # Start visualization server if requested
    if enable_viz:
        try:
            from infrastructure.viz_server import start_viz_server
            logger.info("Starting visualization server...")
            viz_server = await start_viz_server()
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
    runtime = GraphRuntime(graph_db=graph_db, llm_gateway=gateway, viz_server=viz_server)

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
  python main.py "Create a hello world function"
  python main.py --viz "Create a REST API for user management"
  python main.py -f requirements.txt --viz
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
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable real-time visualization dashboard"
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

    # Run
    try:
        stats = asyncio.run(main(requirement, enable_viz=args.viz))
        sys.exit(1 if stats['errors'] > 0 else 0)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    cli()
