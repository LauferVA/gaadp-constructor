#!/usr/bin/env python3
"""
Test script for new GAADP Mission Control (graph-native).

This demonstrates:
1. Single mode (production monitoring)
2. Comparison mode (baseline vs treatment)
3. DAG playback via event log clicking
4. All graph-native types from ontology

Usage:
    python scripts/test_mission_control.py --mode single
    python scripts/test_mission_control.py --mode comparison
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.mission_control import (
    MissionControl,
    start_mission_control,
    get_node_types,
    get_edge_types,
    get_agent_types,
)
from core.ontology import NodeType, EdgeType, NodeStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SAMPLE DAG SCENARIOS
# =============================================================================

FIBONACCI_SCENARIO = {
    "name": "Fibonacci with Error Handling",
    "events": [
        # Initial requirement
        {"action": "create_node", "id": "req_001", "type": "REQ",
         "content": "Create a Fibonacci function with error handling for negative inputs"},

        # Research phase (graph-native - includes RESEARCHER agent)
        {"action": "agent_start", "agent": "RESEARCHER", "node": "req_001"},
        {"action": "create_node", "id": "res_001", "type": "RESEARCH",
         "content": "Research: Fibonacci implementations - iterative vs recursive, memoization options"},
        {"action": "create_edge", "source": "res_001", "target": "req_001", "type": "RESEARCH_FOR"},
        {"action": "agent_finish", "agent": "RESEARCHER", "node": "req_001", "success": True, "cost": 0.012},

        # Specification
        {"action": "agent_start", "agent": "ARCHITECT", "node": "req_001"},
        {"action": "create_node", "id": "spec_001", "type": "SPEC",
         "content": "SPEC: fibonacci(n: int) -> int. Raises ValueError for n < 0. Uses iterative approach."},
        {"action": "create_edge", "source": "spec_001", "target": "req_001", "type": "TRACES_TO"},
        {"action": "create_edge", "source": "spec_001", "target": "res_001", "type": "DEPENDS_ON"},
        {"action": "agent_finish", "agent": "ARCHITECT", "node": "req_001", "success": True, "cost": 0.015},

        # Plan
        {"action": "create_node", "id": "plan_001", "type": "PLAN",
         "content": "PLAN: 1. Validate input 2. Handle base cases 3. Iterate to compute Fib(n)"},
        {"action": "create_edge", "source": "plan_001", "target": "spec_001", "type": "IMPLEMENTS"},

        # Build code
        {"action": "agent_start", "agent": "BUILDER", "node": "plan_001"},
        {"action": "create_node", "id": "code_001", "type": "CODE",
         "content": "def fibonacci(n):\n    if n < 0:\n        raise ValueError('n must be >= 0')\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"},
        {"action": "create_edge", "source": "code_001", "target": "plan_001", "type": "IMPLEMENTS"},
        {"action": "status_change", "node": "code_001", "old": "PENDING", "new": "PROCESSING"},
        {"action": "agent_finish", "agent": "BUILDER", "node": "plan_001", "success": True, "cost": 0.018},

        # Test
        {"action": "agent_start", "agent": "TESTER", "node": "code_001"},
        {"action": "create_node", "id": "test_001", "type": "TEST",
         "content": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55"},
        {"action": "create_edge", "source": "test_001", "target": "code_001", "type": "TESTS"},
        {"action": "status_change", "node": "code_001", "old": "PROCESSING", "new": "TESTING"},
        {"action": "status_change", "node": "code_001", "old": "TESTING", "new": "TESTED"},
        {"action": "agent_finish", "agent": "TESTER", "node": "code_001", "success": True, "cost": 0.008},

        # Verification
        {"action": "agent_start", "agent": "VERIFIER", "node": "code_001"},
        {"action": "status_change", "node": "code_001", "old": "TESTED", "new": "VERIFIED"},
        {"action": "agent_finish", "agent": "VERIFIER", "node": "code_001", "success": True, "cost": 0.005},
    ]
}

CALCULATOR_SCENARIO = {
    "name": "Calculator Package",
    "events": [
        # REQ
        {"action": "create_node", "id": "req_001", "type": "REQ",
         "content": "Create a calculator package with math_ops.py and calculator.py"},

        # Research
        {"action": "agent_start", "agent": "RESEARCHER", "node": "req_001"},
        {"action": "create_node", "id": "res_001", "type": "RESEARCH",
         "content": "Research: Python package structure, __init__.py, module imports"},
        {"action": "create_edge", "source": "res_001", "target": "req_001", "type": "RESEARCH_FOR"},
        {"action": "agent_finish", "agent": "RESEARCHER", "node": "req_001", "success": True, "cost": 0.010},

        # Dialector (Socratic) - clarification
        {"action": "agent_start", "agent": "DIALECTOR", "node": "req_001"},
        {"action": "create_node", "id": "clar_001", "type": "CLARIFICATION",
         "content": "Should the calculator support floating point or just integers?"},
        {"action": "create_edge", "source": "clar_001", "target": "req_001", "type": "FEEDBACK"},
        {"action": "agent_finish", "agent": "DIALECTOR", "node": "req_001", "success": True, "cost": 0.008},

        # SPEC for math_ops
        {"action": "agent_start", "agent": "ARCHITECT", "node": "req_001"},
        {"action": "create_node", "id": "spec_001", "type": "SPEC",
         "content": "math_ops.py: add(a,b), subtract(a,b), multiply(a,b), divide(a,b) with ZeroDivisionError handling"},
        {"action": "create_edge", "source": "spec_001", "target": "req_001", "type": "TRACES_TO"},
        {"action": "create_edge", "source": "spec_001", "target": "res_001", "type": "DEPENDS_ON"},
        {"action": "agent_finish", "agent": "ARCHITECT", "node": "req_001", "success": True, "cost": 0.012},

        # SPEC for calculator.py
        {"action": "create_node", "id": "spec_002", "type": "SPEC",
         "content": "calculator.py: Calculator class with history list, calculate(op, a, b) method"},
        {"action": "create_edge", "source": "spec_002", "target": "req_001", "type": "TRACES_TO"},
        {"action": "create_edge", "source": "spec_002", "target": "spec_001", "type": "DEPENDS_ON"},

        # PLANs
        {"action": "create_node", "id": "plan_001", "type": "PLAN",
         "content": "PLAN math_ops: Define 4 arithmetic functions with type hints"},
        {"action": "create_edge", "source": "plan_001", "target": "spec_001", "type": "IMPLEMENTS"},

        {"action": "create_node", "id": "plan_002", "type": "PLAN",
         "content": "PLAN calculator: Import math_ops, create Calculator class"},
        {"action": "create_edge", "source": "plan_002", "target": "spec_002", "type": "IMPLEMENTS"},
        {"action": "create_edge", "source": "plan_002", "target": "plan_001", "type": "DEPENDS_ON"},

        # CODE - math_ops
        {"action": "agent_start", "agent": "BUILDER", "node": "plan_001"},
        {"action": "create_node", "id": "code_001", "type": "CODE",
         "content": "# math_ops.py\ndef add(a, b): return a + b\ndef subtract(a, b): return a - b\ndef multiply(a, b): return a * b\ndef divide(a, b):\n    if b == 0: raise ZeroDivisionError\n    return a / b"},
        {"action": "create_edge", "source": "code_001", "target": "plan_001", "type": "IMPLEMENTS"},
        {"action": "status_change", "node": "code_001", "old": "PENDING", "new": "PROCESSING"},
        {"action": "agent_finish", "agent": "BUILDER", "node": "plan_001", "success": True, "cost": 0.015},

        # CODE - calculator
        {"action": "agent_start", "agent": "BUILDER", "node": "plan_002"},
        {"action": "create_node", "id": "code_002", "type": "CODE",
         "content": "# calculator.py\nfrom math_ops import *\n\nclass Calculator:\n    def __init__(self):\n        self.history = []\n    \n    def calculate(self, op, a, b):\n        ops = {'add': add, 'sub': subtract, 'mul': multiply, 'div': divide}\n        result = ops[op](a, b)\n        self.history.append((op, a, b, result))\n        return result"},
        {"action": "create_edge", "source": "code_002", "target": "plan_002", "type": "IMPLEMENTS"},
        {"action": "create_edge", "source": "code_002", "target": "code_001", "type": "DEPENDS_ON"},
        {"action": "status_change", "node": "code_002", "old": "PENDING", "new": "PROCESSING"},
        {"action": "agent_finish", "agent": "BUILDER", "node": "plan_002", "success": True, "cost": 0.018},

        # TEST - math_ops
        {"action": "agent_start", "agent": "TESTER", "node": "code_001"},
        {"action": "create_node", "id": "test_001", "type": "TEST",
         "content": "assert add(2, 3) == 5\nassert divide(10, 2) == 5\ntry:\n    divide(1, 0)\n    assert False\nexcept ZeroDivisionError:\n    pass"},
        {"action": "create_edge", "source": "test_001", "target": "code_001", "type": "TESTS"},
        {"action": "status_change", "node": "code_001", "old": "PROCESSING", "new": "TESTING"},
        {"action": "status_change", "node": "code_001", "old": "TESTING", "new": "TESTED"},
        {"action": "agent_finish", "agent": "TESTER", "node": "code_001", "success": True, "cost": 0.008},

        # TEST - calculator
        {"action": "agent_start", "agent": "TESTER", "node": "code_002"},
        {"action": "create_node", "id": "test_002", "type": "TEST",
         "content": "calc = Calculator()\nassert calc.calculate('add', 2, 3) == 5\nassert len(calc.history) == 1"},
        {"action": "create_edge", "source": "test_002", "target": "code_002", "type": "TESTS"},
        {"action": "status_change", "node": "code_002", "old": "PROCESSING", "new": "TESTING"},
        {"action": "status_change", "node": "code_002", "old": "TESTING", "new": "TESTED"},
        {"action": "agent_finish", "agent": "TESTER", "node": "code_002", "success": True, "cost": 0.010},

        # VERIFY
        {"action": "agent_start", "agent": "VERIFIER", "node": "code_001"},
        {"action": "status_change", "node": "code_001", "old": "TESTED", "new": "VERIFIED"},
        {"action": "agent_finish", "agent": "VERIFIER", "node": "code_001", "success": True, "cost": 0.005},

        {"action": "agent_start", "agent": "VERIFIER", "node": "code_002"},
        {"action": "status_change", "node": "code_002", "old": "TESTED", "new": "VERIFIED"},
        {"action": "agent_finish", "agent": "VERIFIER", "node": "code_002", "success": True, "cost": 0.005},
    ]
}

# Smaller scenario for baseline (simulates prior commit version)
CALCULATOR_BASELINE = {
    "name": "Calculator Package (Baseline)",
    "events": [
        {"action": "create_node", "id": "req_001", "type": "REQ",
         "content": "Create a basic calculator with add/subtract"},
        {"action": "create_node", "id": "spec_001", "type": "SPEC",
         "content": "math_ops.py: add(a,b), subtract(a,b)"},
        {"action": "create_edge", "source": "spec_001", "target": "req_001", "type": "TRACES_TO"},
        {"action": "create_node", "id": "code_001", "type": "CODE",
         "content": "def add(a, b): return a + b\ndef subtract(a, b): return a - b"},
        {"action": "create_edge", "source": "code_001", "target": "spec_001", "type": "IMPLEMENTS"},
        {"action": "status_change", "node": "code_001", "old": "PENDING", "new": "VERIFIED"},
    ]
}


# =============================================================================
# DEMO RUNNER
# =============================================================================

async def run_scenario(mc: MissionControl, scenario: dict, session: str = None, delay: float = 0.3):
    """Run a scenario's events through Mission Control."""
    logger.info(f"Running scenario: {scenario['name']} (session: {session or 'default'})")

    for event in scenario["events"]:
        action = event["action"]

        if action == "create_node":
            await mc.on_node_created(
                event["id"], event["type"], event["content"], {},
                session=session
            )
        elif action == "create_edge":
            await mc.on_edge_created(
                event["source"], event["target"], event["type"],
                session=session
            )
        elif action == "status_change":
            await mc.on_node_status_changed(
                event["node"], event["old"], event["new"],
                session=session
            )
        elif action == "agent_start":
            await mc.on_agent_started(event["agent"], event["node"], session=session)
        elif action == "agent_finish":
            await mc.on_agent_finished(
                event["agent"], event["node"], event["success"],
                cost=event.get("cost", 0.0), session=session
            )

        await asyncio.sleep(delay)

    logger.info(f"Scenario complete: {scenario['name']}")


async def run_single_mode():
    """Demo single mode (production monitoring)."""
    print("\n" + "=" * 60)
    print("MISSION CONTROL - SINGLE MODE (Production)")
    print("=" * 60)

    mc = await start_mission_control(mode="single")

    print(f"\nDashboard: http://localhost:8766")
    print(f"WebSocket: ws://localhost:8765")
    print("\nOpen the dashboard and watch the DAG build in real-time!")
    print("Click on events in the log to replay DAG state.")
    print("\nRunning Fibonacci scenario...\n")

    await asyncio.sleep(2)  # Let user open dashboard

    await run_scenario(mc, FIBONACCI_SCENARIO)

    await mc.on_complete({
        "total_nodes": len(mc.get_session().nodes),
        "total_edges": len(mc.get_session().edges),
        "status": "success",
    })

    print("\nScenario complete! Dashboard remains active.")
    print("Try clicking events in the log to see DAG playback.")
    print("Press Ctrl+C to exit.\n")

    # Keep running until interrupted (don't call run_forever which tries to start again)
    try:
        await asyncio.Future()  # Run forever
    except asyncio.CancelledError:
        pass
    finally:
        await mc.stop()


async def run_comparison_mode():
    """Demo comparison mode (baseline vs treatment)."""
    print("\n" + "=" * 60)
    print("MISSION CONTROL - COMPARISON MODE (Dev)")
    print("=" * 60)

    mc = await start_mission_control(mode="comparison")

    # Set git info for sessions
    mc.set_session_git_info("baseline", commit="abc1234")  # Simulated prior commit
    mc.set_session_git_info("treatment")  # Current HEAD

    print(f"\nDashboard: http://localhost:8766?mode=comparison")
    print(f"WebSocket: ws://localhost:8765")
    print("\nOpen the dashboard to see side-by-side comparison!")
    print("BASELINE = prior commit, TREATMENT = current dev")
    print("\nRunning baseline scenario...\n")

    await asyncio.sleep(2)

    # Run baseline (smaller, simpler)
    await run_scenario(mc, CALCULATOR_BASELINE, session="baseline", delay=0.2)

    print("\nBaseline complete. Running treatment scenario...\n")
    await asyncio.sleep(1)

    # Run treatment (full scenario)
    await run_scenario(mc, CALCULATOR_SCENARIO, session="treatment", delay=0.25)

    # Show delta
    baseline_sess = mc.get_session("baseline")
    treatment_sess = mc.get_session("treatment")

    delta = {
        "nodes_delta": len(treatment_sess.nodes) - len(baseline_sess.nodes),
        "edges_delta": len(treatment_sess.edges) - len(baseline_sess.edges),
        "cost_delta": treatment_sess.metrics["total_cost"] - baseline_sess.metrics["total_cost"],
    }

    print("\n" + "-" * 40)
    print("DELTA (Treatment vs Baseline):")
    print(f"  Nodes: +{delta['nodes_delta']}")
    print(f"  Edges: +{delta['edges_delta']}")
    print(f"  Cost: +${delta['cost_delta']:.4f}")
    print("-" * 40)

    print("\nBoth scenarios complete! Dashboard remains active.")
    print("Press Ctrl+C to exit.\n")

    # Keep running until interrupted
    try:
        await asyncio.Future()  # Run forever
    except asyncio.CancelledError:
        pass
    finally:
        await mc.stop()


async def main():
    parser = argparse.ArgumentParser(description="Test GAADP Mission Control")
    parser.add_argument(
        "--mode",
        choices=["single", "comparison"],
        default="single",
        help="Dashboard mode to test"
    )
    parser.add_argument(
        "--scenario",
        choices=["fibonacci", "calculator"],
        default="fibonacci",
        help="Scenario to run (single mode only)"
    )
    args = parser.parse_args()

    # Show graph-native info
    print("\n" + "=" * 60)
    print("GAADP MISSION CONTROL - Graph-Native Test")
    print("=" * 60)
    print(f"\nNode types from ontology: {len(get_node_types())}")
    print(f"Edge types from ontology: {len(get_edge_types())}")
    print(f"Agent types from AGENT_DISPATCH: {len(get_agent_types())}")
    print(f"\nAgents: {', '.join(get_agent_types())}")

    if args.mode == "single":
        await run_single_mode()
    else:
        await run_comparison_mode()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
