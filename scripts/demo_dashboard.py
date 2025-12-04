#!/usr/bin/env python3
"""
UNIFIED DASHBOARD DEMO

One script, multiple prompts. Complexity emerges from the task.

Usage:
    python scripts/demo_dashboard.py                           # Interactive prompt selection
    python scripts/demo_dashboard.py --prompt "Create X"       # Inline prompt
    python scripts/demo_dashboard.py -f prompts/asteroid.md    # From file

The demo simulates what the pipeline WOULD produce for that prompt.
"""
import asyncio
import argparse
import logging
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.viz_server import start_viz_server, stop_viz_server

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("GAADP.Demo")


# =============================================================================
# PROMPT ANALYSIS - Infer structure from the prompt itself
# =============================================================================

def analyze_prompt(prompt: str) -> dict:
    """
    Analyze prompt to infer project structure.
    Returns dict with files, dependencies, complexity signals.
    """
    prompt_lower = prompt.lower()

    # Extract explicit file mentions
    files = []
    file_pattern = r'(\w+\.py)'
    file_matches = re.findall(file_pattern, prompt)
    if file_matches:
        files = list(dict.fromkeys(file_matches))  # Dedupe, preserve order

    # Detect architecture keywords
    is_multi_file = any(kw in prompt_lower for kw in [
        'multiple files', 'separate files', 'decomposed', 'architecture',
        'package', 'module', 'imports', 'depends on'
    ])

    # Detect domain
    domain = 'utility'
    if any(kw in prompt_lower for kw in ['game', 'pygame', 'player', 'asteroid']):
        domain = 'game'
    elif any(kw in prompt_lower for kw in ['api', 'endpoint', 'rest', 'http', 'server']):
        domain = 'api'
    elif any(kw in prompt_lower for kw in ['cli', 'command', 'terminal', 'argparse']):
        domain = 'cli'
    elif any(kw in prompt_lower for kw in ['calculator', 'math', 'compute']):
        domain = 'calculator'

    # Estimate complexity
    if len(files) >= 5 or 'complex' in prompt_lower:
        complexity = 'high'
    elif len(files) >= 3 or is_multi_file:
        complexity = 'medium'
    else:
        complexity = 'low'

    # If no explicit files, infer from domain
    if not files:
        if domain == 'game':
            files = ['entities.py', 'player.py', 'main.py']
        elif domain == 'api':
            files = ['models.py', 'routes.py', 'app.py']
        elif domain == 'cli':
            files = ['core.py', 'cli.py']
        elif domain == 'calculator':
            files = ['math_ops.py', 'calculator.py']
        else:
            files = ['main.py']

    return {
        'files': files,
        'domain': domain,
        'complexity': complexity,
        'is_multi_file': is_multi_file or len(files) > 1,
        'raw_prompt': prompt[:200]
    }


def infer_dependencies(files: list) -> list:
    """
    Infer DEPENDS_ON relationships from file list.
    Returns list of (from_file, to_file) tuples.
    """
    deps = []

    # Common patterns
    main_files = {'main.py', 'app.py', 'cli.py', '__main__.py'}
    base_files = {'entities.py', 'models.py', 'base.py', 'core.py', 'utils.py'}

    main = next((f for f in files if f in main_files), None)
    bases = [f for f in files if f in base_files]
    others = [f for f in files if f not in main_files and f not in base_files]

    # Base files have no deps (leaf nodes)
    # Others depend on bases
    for other in others:
        for base in bases:
            deps.append((other, base))

    # Main depends on others (and bases if no others)
    if main:
        targets = others if others else bases
        for target in targets:
            deps.append((main, target))

    return deps


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

async def simulate_from_prompt(viz_server, session: str, prompt: str, use_tdd: bool = False):
    """Simulate pipeline execution based on prompt analysis."""

    analysis = analyze_prompt(prompt)
    files = analysis['files']
    deps = infer_dependencies(files)

    logger.info(f"Simulating {session.upper()} (TDD={use_tdd})")
    logger.info(f"  Domain: {analysis['domain']}")
    logger.info(f"  Files: {files}")
    logger.info(f"  Dependencies: {len(deps)}")

    viz_server.set_active_session(session)
    viz_server.reset_session(session)

    # ========== REQ ==========
    await viz_server.on_node_created("req_001", "REQ", prompt[:300], analysis)
    await asyncio.sleep(0.3)

    # ========== RESEARCH ==========
    await viz_server.on_agent_started("RESEARCHER", "req_001", ["req_001"])
    await asyncio.sleep(0.2)

    research_content = f"""Research Artifact v1.0
Domain: {analysis['domain']}
Files: {', '.join(files)}
Complexity: {analysis['complexity']}

Typed Contracts: [inferred from prompt]
Examples: [3 examples generated]
Security: Standard trust boundary"""

    await viz_server.on_node_created("research_001", "RESEARCH", research_content, {
        'domain': analysis['domain'],
        'file_count': len(files)
    })
    await viz_server.on_edge_created("research_001", "req_001", "TRACES_TO")
    await asyncio.sleep(0.2)

    await viz_server.on_agent_finished("RESEARCHER", "req_001", True, 0.002)
    await viz_server.on_node_status_changed("research_001", "PENDING", "VERIFIED", "Research complete")
    await asyncio.sleep(0.3)

    # ========== ARCHITECTURE (SPECs) ==========
    await viz_server.on_agent_started("ARCHITECT", "research_001", ["req_001", "research_001"])
    await asyncio.sleep(0.2)

    spec_ids = {}
    for i, filename in enumerate(files):
        spec_id = f"spec_{filename.replace('.py', '')}"
        spec_ids[filename] = spec_id

        file_deps = [d[1] for d in deps if d[0] == filename]

        await viz_server.on_node_created(
            spec_id, "SPEC",
            f"SPEC: {filename} - Implementation specification",
            {'file': filename, 'depends_on': file_deps}
        )
        await viz_server.on_edge_created(spec_id, "research_001", "TRACES_TO")

        # Add DEPENDS_ON edges
        for dep_file in file_deps:
            if dep_file in spec_ids:
                await viz_server.on_edge_created(spec_id, spec_ids[dep_file], "DEPENDS_ON")

        await asyncio.sleep(0.1)

    await viz_server.on_agent_finished("ARCHITECT", "research_001", True, 0.003)

    for spec_id in spec_ids.values():
        await viz_server.on_node_status_changed(spec_id, "PENDING", "VERIFIED", "Spec approved")
    await asyncio.sleep(0.3)

    # ========== BUILD (in dependency order) ==========
    # Topological sort - build leaves first
    built = set()
    code_ids = {}

    def can_build(filename):
        file_deps = [d[1] for d in deps if d[0] == filename]
        return all(d in built for d in file_deps)

    remaining = list(files)
    while remaining:
        for filename in remaining[:]:
            if can_build(filename):
                spec_id = spec_ids[filename]
                code_id = f"code_{filename.replace('.py', '')}"
                code_ids[filename] = code_id

                await viz_server.on_agent_started("BUILDER", spec_id, [spec_id])
                await asyncio.sleep(0.15)

                await viz_server.on_node_created(
                    code_id, "CODE",
                    f"# {filename}\n# Generated implementation",
                    {'file_path': filename}
                )
                await viz_server.on_edge_created(code_id, spec_id, "TRACES_TO")

                # Add code DEPENDS_ON edges
                for dep_file in [d[1] for d in deps if d[0] == filename]:
                    if dep_file in code_ids:
                        await viz_server.on_edge_created(code_id, code_ids[dep_file], "DEPENDS_ON")

                await viz_server.on_agent_finished("BUILDER", spec_id, True, 0.002)

                built.add(filename)
                remaining.remove(filename)
                await asyncio.sleep(0.15)

    # ========== TDD TESTING ==========
    if use_tdd:
        for filename in files:
            code_id = code_ids[filename]
            test_id = f"test_{filename.replace('.py', '')}"

            await viz_server.on_node_status_changed(code_id, "PENDING", "TESTING", f"Testing {filename}...")
            await viz_server.on_agent_started("TESTER", code_id, [code_id])
            await asyncio.sleep(0.1)

            await viz_server.on_node_created(
                test_id, "TEST_SUITE",
                f"Tests for {filename}: 5 tests, 90% coverage\n✓ All tests passed",
                {'tests': 5, 'passed': 5, 'coverage': 0.90}
            )
            await viz_server.on_edge_created(test_id, code_id, "TESTS")

            await viz_server.on_agent_finished("TESTER", code_id, True, 0.0015)
            await viz_server.on_node_status_changed(code_id, "TESTING", "TESTED", "Tests pass")
            await viz_server.on_node_status_changed(test_id, "PENDING", "VERIFIED", "Test suite validated")
            await asyncio.sleep(0.1)

    # ========== VERIFICATION ==========
    for filename in files:
        code_id = code_ids[filename]
        status_before = "TESTED" if use_tdd else "PENDING"

        await viz_server.on_node_status_changed(code_id, status_before, "PROCESSING", "Verifying...")
        await viz_server.on_agent_started("VERIFIER", code_id, [code_id])
        await asyncio.sleep(0.08)

        await viz_server.on_agent_finished("VERIFIER", code_id, True, 0.001)
        await viz_server.on_node_status_changed(code_id, "PROCESSING", "VERIFIED", "Verified")
        await asyncio.sleep(0.08)

    # ========== COMPLETE ==========
    node_count = 2 + len(files) * 2  # REQ, RESEARCH, SPECs, CODEs
    if use_tdd:
        node_count += len(files)  # TEST_SUITEs

    stats = {
        "iterations": node_count,
        "nodes_processed": node_count,
        "total_cost": node_count * 0.002,
        "errors": 0,
        "files_generated": len(files)
    }
    await viz_server.on_complete(stats)

    logger.info(f"{session.upper()} complete: {node_count} nodes, {len(files)} files")
    return stats


# =============================================================================
# SAMPLE PROMPTS
# =============================================================================

SAMPLE_PROMPTS = {
    "fibonacci": "Create a fibonacci function that returns the nth fibonacci number",

    "calculator": """Create a Python calculator package with:
- math_ops.py: add, subtract, multiply, divide functions with ZeroDivisionError handling
- calculator.py: Calculator class that imports math_ops, with history tracking
- cli.py: Command-line interface that imports Calculator""",

    "asteroid": """Create an Asteroid game in Python with Pygame:
- entities.py: Base Entity class with Vector2D, position, velocity
- physics.py: Collision detection, wrap_position (depends on entities)
- player.py: Player class with thrust, rotation (depends on entities)
- asteroid.py: Asteroid class with split, spawn (depends on entities, physics)
- renderer.py: Pygame rendering (depends on entities, player, asteroid)
- main.py: Game loop (depends on player, asteroid, renderer, physics)""",

    "api": """Create a REST API with FastAPI:
- models.py: Pydantic models for User, Item
- database.py: SQLAlchemy setup (depends on models)
- routes.py: API endpoints (depends on models, database)
- app.py: FastAPI application (depends on routes)"""
}


async def main():
    parser = argparse.ArgumentParser(description="GAADP Dashboard Demo")
    parser.add_argument("--prompt", "-p", help="Inline prompt")
    parser.add_argument("--file", "-f", help="Read prompt from file")
    parser.add_argument("--sample", "-s", choices=SAMPLE_PROMPTS.keys(),
                       help="Use a sample prompt")
    parser.add_argument("--list-samples", action="store_true",
                       help="List available sample prompts")
    args = parser.parse_args()

    if args.list_samples:
        print("\nAvailable sample prompts:")
        for name, prompt in SAMPLE_PROMPTS.items():
            print(f"\n  {name}:")
            print(f"    {prompt[:80]}...")
        return

    # Get prompt
    prompt = None
    if args.file:
        prompt = Path(args.file).read_text()
    elif args.prompt:
        prompt = args.prompt
    elif args.sample:
        prompt = SAMPLE_PROMPTS[args.sample]
    else:
        # Interactive selection
        print("\nSelect a sample prompt:")
        for i, name in enumerate(SAMPLE_PROMPTS.keys(), 1):
            print(f"  {i}. {name}")
        print(f"  {len(SAMPLE_PROMPTS)+1}. Enter custom prompt")

        choice = input("\nChoice: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(SAMPLE_PROMPTS):
                prompt = list(SAMPLE_PROMPTS.values())[idx]
            else:
                prompt = input("Enter prompt: ").strip()
        else:
            prompt = choice

    if not prompt:
        print("No prompt provided")
        return

    # Run demo
    logger.info("=" * 70)
    logger.info("GAADP DASHBOARD DEMO")
    logger.info("=" * 70)
    logger.info(f"Prompt: {prompt[:100]}...")

    viz_server = await start_viz_server(dev_mode=True)
    logger.info("Dashboard: http://localhost:8766")
    await asyncio.sleep(3)

    try:
        # Baseline
        logger.info("\n" + "=" * 70)
        logger.info("BASELINE (Gen-1)")
        logger.info("=" * 70)
        await simulate_from_prompt(viz_server, "baseline", prompt, use_tdd=False)

        await asyncio.sleep(1)

        # Treatment
        logger.info("\n" + "=" * 70)
        logger.info("TREATMENT (Gen-2 TDD)")
        logger.info("=" * 70)
        await simulate_from_prompt(viz_server, "treatment", prompt, use_tdd=True)

        logger.info("\n" + "=" * 70)
        logger.info("DEMO COMPLETE - Press Ctrl+C to exit")
        logger.info("=" * 70)

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        await stop_viz_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
