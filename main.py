#!/usr/bin/env python3
"""
GAADP MAIN - Minimal Entry Point

This is the new graph-first entry point that uses:
- GraphRuntime (the execution engine)
- GenericAgent (universal agent)
- TransitionMatrix (the physics)

~50 lines replacing ~500+ lines of orchestration code.
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


async def main(requirement: str = None) -> dict:
    """
    Main entry point for GAADP.

    Args:
        requirement: The user requirement to process.
                    If None, uses a default test requirement.

    Returns:
        Execution statistics
    """
    logger.info("=" * 60)
    logger.info("GAADP - Graph-First Architecture")
    logger.info("=" * 60)

    # Initialize components
    graph_db = GraphDB(persistence_path=".gaadp/graph.json")
    gateway = LLMGateway()
    runtime = GraphRuntime(graph_db=graph_db, llm_gateway=gateway)

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

    return stats


def cli():
    """Command line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GAADP - Graph-Aware Autonomous Development Platform"
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

    # Run
    stats = asyncio.run(main(requirement))

    # Exit code based on errors
    sys.exit(1 if stats['errors'] > 0 else 0)


if __name__ == "__main__":
    cli()
