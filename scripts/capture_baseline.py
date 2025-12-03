#!/usr/bin/env python3
"""
BASELINE METRICS CAPTURE
Runs a set of test prompts and captures quantitative metrics for regression testing.

Usage:
    python scripts/capture_baseline.py
    python scripts/capture_baseline.py --output baseline_v1.json
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import main as run_gaadp


# Test cases with expected outcomes
TEST_CASES = [
    {
        "id": "simple_function",
        "prompt": "Create a Python function called add_numbers that takes two integers and returns their sum.",
        "expected": {
            "should_pass": True,
            "min_code_length": 20,
        }
    },
    {
        "id": "docstring_function",
        "prompt": "Create a Python function called greet that takes a name parameter and returns a greeting string with a docstring explaining the function.",
        "expected": {
            "should_pass": True,
            "min_code_length": 50,
        }
    },
    {
        "id": "class_implementation",
        "prompt": "Create a Python class called Counter with an __init__ method that sets count to 0, an increment method, and a get_count method.",
        "expected": {
            "should_pass": True,
            "min_code_length": 100,
        }
    },
]


def parse_log_for_metrics(log_path: str = ".gaadp/logs/gaadp.jsonl") -> Dict:
    """Parse the JSONL log file to extract metrics."""
    metrics = {
        "architect_calls": 0,
        "builder_calls": 0,
        "verifier_calls": 0,
        "tool_calls": [],
        "llm_costs": [],
        "verdicts": [],
        "phases": [],
        "errors": [],
    }

    if not os.path.exists(log_path):
        return metrics

    with open(log_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                msg = entry.get("message", "")
                phase = entry.get("phase", "")

                # Track phases
                if phase and phase not in metrics["phases"]:
                    metrics["phases"].append(phase)

                # Count agent calls
                if "Architect processing" in msg:
                    metrics["architect_calls"] += 1
                elif "Builder processing" in msg:
                    metrics["builder_calls"] += 1
                elif "Verifier processing" in msg:
                    metrics["verifier_calls"] += 1

                # Track verdicts
                if "Verifier verdict:" in msg:
                    verdict = msg.split("verdict:")[-1].strip()
                    metrics["verdicts"].append(verdict)

                # Track costs
                if "Cost:" in msg:
                    try:
                        cost_part = msg.split("Cost:")[-1].split("|")[0].strip()
                        cost = float(cost_part.replace("$", ""))
                        metrics["llm_costs"].append(cost)
                    except:
                        pass

                # Track errors
                if entry.get("level") == "ERROR":
                    metrics["errors"].append(msg[:200])

            except json.JSONDecodeError:
                continue

    return metrics


def analyze_graph(graph_path: str = ".gaadp/live_graph.json") -> Dict:
    """Analyze the graph state."""
    if not os.path.exists(graph_path):
        return {"nodes": 0, "edges": 0}

    with open(graph_path, "r") as f:
        data = json.load(f)

    graph = data.get("graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # Count by type
    from collections import Counter
    node_types = Counter(n.get("type", "UNKNOWN") for n in nodes)
    edge_types = Counter(e.get("type", "UNKNOWN") for e in edges)
    statuses = Counter(n.get("status", "UNKNOWN") for n in nodes)

    return {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
        "statuses": dict(statuses),
        "verified_count": statuses.get("VERIFIED", 0),
        "failed_count": statuses.get("FAILED", 0),
    }


async def run_test_case(test_case: Dict) -> Dict:
    """Run a single test case and capture metrics."""
    print(f"\n{'='*60}")
    print(f"Running: {test_case['id']}")
    print(f"{'='*60}")

    # Write prompt
    with open("prompt.md", "w") as f:
        f.write(test_case["prompt"])

    # Clear previous logs
    log_path = ".gaadp/logs/gaadp.jsonl"
    if os.path.exists(log_path):
        # Truncate log file
        open(log_path, "w").close()

    # Run GAADP
    start_time = time.time()
    try:
        await main(interactive=False)
        success = True
        error = None
    except Exception as e:
        success = False
        error = str(e)
    elapsed = time.time() - start_time

    # Collect metrics
    log_metrics = parse_log_for_metrics(log_path)
    graph_metrics = analyze_graph()

    # Determine pass/fail
    passed_verification = "PASS" in log_metrics.get("verdicts", [])
    expected_pass = test_case["expected"].get("should_pass", True)
    test_passed = passed_verification == expected_pass

    result = {
        "test_id": test_case["id"],
        "prompt": test_case["prompt"],
        "success": success,
        "error": error,
        "elapsed_seconds": round(elapsed, 2),
        "test_passed": test_passed,
        "verification_passed": passed_verification,
        "expected_pass": expected_pass,
        "metrics": {
            "architect_calls": log_metrics["architect_calls"],
            "builder_calls": log_metrics["builder_calls"],
            "verifier_calls": log_metrics["verifier_calls"],
            "total_llm_cost": round(sum(log_metrics["llm_costs"]), 4),
            "verdicts": log_metrics["verdicts"],
            "phases_visited": log_metrics["phases"],
            "errors": log_metrics["errors"][:5],  # First 5 errors
        },
        "graph": graph_metrics,
    }

    return result


async def run_baseline():
    """Run all test cases and generate baseline report."""
    print("="*60)
    print("GAADP BASELINE METRICS CAPTURE")
    print("="*60)

    results = []
    total_start = time.time()

    for test_case in TEST_CASES:
        result = await run_test_case(test_case)
        results.append(result)

        # Summary
        status = "✅ PASS" if result["test_passed"] else "❌ FAIL"
        print(f"\n{status} - {test_case['id']} ({result['elapsed_seconds']}s, ${result['metrics']['total_llm_cost']:.4f})")

    total_elapsed = time.time() - total_start

    # Aggregate metrics
    baseline = {
        "version": "1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_elapsed_seconds": round(total_elapsed, 2),
        "summary": {
            "total_tests": len(results),
            "passed": sum(1 for r in results if r["test_passed"]),
            "failed": sum(1 for r in results if not r["test_passed"]),
            "total_cost_usd": round(sum(r["metrics"]["total_llm_cost"] for r in results), 4),
            "avg_elapsed_seconds": round(sum(r["elapsed_seconds"] for r in results) / len(results), 2),
        },
        "tests": results,
    }

    # Print summary
    print("\n" + "="*60)
    print("BASELINE SUMMARY")
    print("="*60)
    print(f"Tests: {baseline['summary']['passed']}/{baseline['summary']['total_tests']} passed")
    print(f"Total Cost: ${baseline['summary']['total_cost_usd']}")
    print(f"Total Time: {baseline['total_elapsed_seconds']}s")
    print(f"Avg Time per Test: {baseline['summary']['avg_elapsed_seconds']}s")

    return baseline


def save_baseline(baseline: Dict, output_path: str = "baselines/baseline_latest.json"):
    """Save baseline to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\n📊 Baseline saved to: {output_path}")

    # Also save timestamped version
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = f"baselines/baseline_{ts}.json"
    with open(versioned_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"📊 Versioned copy: {versioned_path}")


def compare_baselines(current: Dict, previous_path: str) -> Dict:
    """Compare current baseline against a previous one."""
    if not os.path.exists(previous_path):
        return {"comparison": "no_previous_baseline"}

    with open(previous_path, "r") as f:
        previous = json.load(f)

    comparison = {
        "current_version": current.get("version"),
        "previous_version": previous.get("version"),
        "tests": {
            "current_passed": current["summary"]["passed"],
            "previous_passed": previous["summary"]["passed"],
            "delta": current["summary"]["passed"] - previous["summary"]["passed"],
        },
        "cost": {
            "current_usd": current["summary"]["total_cost_usd"],
            "previous_usd": previous["summary"]["total_cost_usd"],
            "delta": round(current["summary"]["total_cost_usd"] - previous["summary"]["total_cost_usd"], 4),
        },
        "time": {
            "current_avg_seconds": current["summary"]["avg_elapsed_seconds"],
            "previous_avg_seconds": previous["summary"]["avg_elapsed_seconds"],
            "delta": round(current["summary"]["avg_elapsed_seconds"] - previous["summary"]["avg_elapsed_seconds"], 2),
        },
        "regression": current["summary"]["passed"] < previous["summary"]["passed"],
    }

    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capture GAADP baseline metrics")
    parser.add_argument("--output", default="baselines/baseline_latest.json", help="Output path for baseline")
    parser.add_argument("--compare", default=None, help="Path to previous baseline for comparison")
    args = parser.parse_args()

    # Run baseline capture
    baseline = asyncio.run(run_baseline())

    # Save
    save_baseline(baseline, args.output)

    # Compare if previous baseline provided
    if args.compare:
        comparison = compare_baselines(baseline, args.compare)
        print("\n" + "="*60)
        print("REGRESSION COMPARISON")
        print("="*60)
        print(json.dumps(comparison, indent=2))

        if comparison.get("regression"):
            print("\n⚠️  WARNING: REGRESSION DETECTED")
            sys.exit(1)
