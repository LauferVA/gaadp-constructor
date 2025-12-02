#!/usr/bin/env python3
"""
CODE RECONSTRUCTION BENCHMARK
Tests whether GAADP can recreate existing code from its extracted specification.

This is a key benchmark for evaluating agent quality:
1. Take an existing Python file
2. Extract its specification (what it does, not how)
3. Hide the original implementation
4. Run GAADP to generate code from the spec
5. Compare: Does the generated code have the same behavior?

Usage:
    python scripts/run_reconstruction.py path/to/file.py
    python scripts/run_reconstruction.py --suite core  # Run on core/ directory
    python scripts/run_reconstruction.py --compare baseline.json
"""
import asyncio
import argparse
import json
import os
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.graph_db import GraphDB
from requirements.socratic_agent import DevelopmentSocraticPhase, SocraticConfig


def extract_spec(file_path: str, source_path: str = ".") -> Dict:
    """
    Extract specification from a file without running full pipeline.

    Returns a dict with:
    - public_interface: What the code exposes
    - behavior_spec: What it should do (from docstrings)
    - dependencies: What it imports
    """
    db = GraphDB(persistence_path=":memory:")
    phase = DevelopmentSocraticPhase(db, source_path)

    # Use synchronous extraction
    loop = asyncio.new_event_loop()
    spec = loop.run_until_complete(phase._extract_spec_from_file(file_path))
    loop.close()

    return spec


def create_reconstruction_prompt(spec: Dict) -> str:
    """
    Create a prompt.md from extracted specification.

    This prompt should contain ONLY what the code does,
    not how it's implemented.
    """
    prompt_parts = [
        f"# CODE RECONSTRUCTION TASK",
        f"",
        f"## Target File: {spec.get('file_path', 'unknown')}",
        f"",
    ]

    # Module purpose
    if spec.get("module_docstring"):
        prompt_parts.extend([
            "## Module Purpose",
            spec["module_docstring"],
            ""
        ])

    # Dependencies (these ARE allowed - they're part of the spec)
    if spec.get("imports"):
        prompt_parts.extend([
            "## Required Imports",
            "The implementation should use these modules:",
            "```python"
        ])
        for imp in spec["imports"][:20]:  # Limit to avoid noise
            prompt_parts.append(f"import {imp}" if "." not in imp else f"from {imp.rsplit('.', 1)[0]} import {imp.rsplit('.', 1)[1]}")
        prompt_parts.extend(["```", ""])

    # Public interface
    prompt_parts.extend([
        "## Required Interface",
        "Implement the following public interface:",
        ""
    ])

    for component in spec.get("components", []):
        if component["type"] == "class":
            prompt_parts.append(f"### Class: `{component['name']}`")
            if component.get("docstring"):
                prompt_parts.append(f"*{component['docstring']}*")
            prompt_parts.append("")
            prompt_parts.append("Methods:")
            for method in component.get("methods", []):
                if not method["name"].startswith("_") or method["name"] in ["__init__", "__str__", "__repr__"]:
                    args_str = ", ".join(method.get("args", []))
                    prompt_parts.append(f"- `{method['name']}({args_str})`")
                    if method.get("docstring"):
                        prompt_parts.append(f"  - {method['docstring'][:100]}")
            prompt_parts.append("")

        elif component["type"] == "function":
            if not component["name"].startswith("_"):
                args_str = ", ".join(component.get("args", []))
                prompt_parts.append(f"### Function: `{component['name']}({args_str})`")
                if component.get("docstring"):
                    prompt_parts.append(f"*{component['docstring']}*")
                prompt_parts.append("")

    prompt_parts.extend([
        "## Constraints",
        "- Implement all listed classes and functions",
        "- Match the method signatures exactly",
        "- The implementation details may differ from the original",
        "- Focus on correctness over optimization",
        ""
    ])

    return "\n".join(prompt_parts)


async def run_reconstruction_test(
    file_path: str,
    source_path: str = ".",
    timeout: int = 180
) -> Dict:
    """
    Run a full reconstruction test on a single file.

    1. Extract spec
    2. Generate prompt
    3. Run GAADP
    4. Compare results
    """
    from production_main import main as run_gaadp

    print(f"\n{'='*60}")
    print(f"RECONSTRUCTION TEST: {file_path}")
    print(f"{'='*60}")

    result = {
        "file": file_path,
        "timestamp": datetime.utcnow().isoformat(),
        "spec_extraction": None,
        "generation": None,
        "comparison": None,
        "success": False
    }

    # 1. Extract spec
    print("\n📋 Extracting specification...")
    spec = extract_spec(file_path, source_path)

    if "error" in spec:
        print(f"❌ Spec extraction failed: {spec['error']}")
        result["spec_extraction"] = {"success": False, "error": spec["error"]}
        return result

    result["spec_extraction"] = {
        "success": True,
        "components": len(spec.get("components", [])),
        "imports": len(spec.get("imports", []))
    }
    print(f"   Found {result['spec_extraction']['components']} components")

    # 2. Generate prompt
    print("\n📝 Generating reconstruction prompt...")
    prompt = create_reconstruction_prompt(spec)

    # Write to prompt.md
    with open("prompt.md", "w") as f:
        f.write(prompt)
    print(f"   Written to prompt.md ({len(prompt)} chars)")

    # 3. Backup original file (we'll compare later)
    original_content = None
    full_path = os.path.join(source_path, file_path)
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            original_content = f.read()

    # 4. Run GAADP
    print("\n🔨 Running GAADP...")
    try:
        await run_gaadp(interactive=False)
        result["generation"] = {"success": True}
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        result["generation"] = {"success": False, "error": str(e)}
        return result

    # 5. Compare results
    print("\n📊 Comparing results...")
    comparison = compare_implementations(spec, original_content)
    result["comparison"] = comparison
    result["success"] = comparison.get("overall_match", False)

    # Print summary
    print(f"\n{'='*60}")
    if result["success"]:
        print("✅ RECONSTRUCTION SUCCESS")
    else:
        print("❌ RECONSTRUCTION FAILED")
    print(f"   Interface match: {comparison.get('interface_match', 0)*100:.0f}%")
    print(f"   Components found: {comparison.get('components_found', 0)}/{comparison.get('components_expected', 0)}")
    print(f"{'='*60}")

    return result


def compare_implementations(spec: Dict, original_content: Optional[str]) -> Dict:
    """
    Compare generated code against specification.

    Checks:
    - Are all required classes/functions present?
    - Do method signatures match?
    - (Optional) Do test cases pass?
    """
    comparison = {
        "interface_match": 0.0,
        "components_expected": len(spec.get("components", [])),
        "components_found": 0,
        "missing": [],
        "extra": [],
        "signature_mismatches": []
    }

    # Find the generated file
    # GAADP writes to the path specified in the prompt or a default
    generated_file = None
    for candidate in ["generated_output.py", spec.get("file_path", "")]:
        if os.path.exists(candidate):
            generated_file = candidate
            break

    if not generated_file:
        comparison["error"] = "Generated file not found"
        return comparison

    # Extract spec from generated code
    generated_spec = extract_spec(generated_file)

    if "error" in generated_spec:
        comparison["error"] = f"Failed to parse generated code: {generated_spec['error']}"
        return comparison

    # Compare components
    expected_names = {c["name"] for c in spec.get("components", [])}
    found_names = {c["name"] for c in generated_spec.get("components", [])}

    comparison["components_found"] = len(expected_names & found_names)
    comparison["missing"] = list(expected_names - found_names)
    comparison["extra"] = list(found_names - expected_names)

    # Calculate interface match percentage
    if comparison["components_expected"] > 0:
        comparison["interface_match"] = comparison["components_found"] / comparison["components_expected"]

    # Check method signatures for matching classes
    for expected_comp in spec.get("components", []):
        if expected_comp["type"] == "class":
            # Find matching generated class
            generated_class = next(
                (c for c in generated_spec.get("components", [])
                 if c["name"] == expected_comp["name"] and c["type"] == "class"),
                None
            )

            if generated_class:
                expected_methods = {m["name"] for m in expected_comp.get("methods", [])}
                found_methods = {m["name"] for m in generated_class.get("methods", [])}

                for missing in expected_methods - found_methods:
                    comparison["signature_mismatches"].append(
                        f"Class {expected_comp['name']}: missing method {missing}"
                    )

    # Overall match
    comparison["overall_match"] = (
        comparison["interface_match"] >= 0.8 and
        len(comparison["signature_mismatches"]) == 0
    )

    return comparison


def run_test_suite(suite_name: str, source_path: str = ".") -> List[Dict]:
    """Run reconstruction tests on a predefined suite of files."""
    suites = {
        "core": [
            "core/ontology.py",
            "core/protocols.py",
            "core/state_machine.py",
        ],
        "infrastructure": [
            "infrastructure/graph_db.py",
            "infrastructure/event_bus.py",
        ],
        "simple": [
            "log_validator.py",  # Simple standalone file
        ]
    }

    if suite_name not in suites:
        print(f"Unknown suite: {suite_name}")
        print(f"Available: {list(suites.keys())}")
        return []

    results = []
    for file_path in suites[suite_name]:
        result = asyncio.run(run_reconstruction_test(file_path, source_path))
        results.append(result)

    return results


def print_suite_summary(results: List[Dict]):
    """Print summary of test suite results."""
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.get("success"))
    total = len(results)

    print(f"Results: {passed}/{total} passed ({passed/total*100:.0f}%)")
    print()

    for result in results:
        status = "✅" if result.get("success") else "❌"
        file_name = result.get("file", "unknown")
        interface_match = result.get("comparison", {}).get("interface_match", 0)
        print(f"  {status} {file_name} ({interface_match*100:.0f}% interface match)")

        if result.get("comparison", {}).get("missing"):
            print(f"      Missing: {result['comparison']['missing']}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run code reconstruction benchmark")
    parser.add_argument("file", nargs="?", help="File to reconstruct")
    parser.add_argument("--suite", help="Run predefined test suite (core, infrastructure, simple)")
    parser.add_argument("--source", default=".", help="Source path for file lookup")
    parser.add_argument("--output", help="Output file for results JSON")
    parser.add_argument("--compare", help="Previous results to compare against")
    parser.add_argument("--spec-only", action="store_true", help="Only extract spec, don't run GAADP")

    args = parser.parse_args()

    if args.suite:
        results = run_test_suite(args.suite, args.source)
        print_suite_summary(results)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    elif args.file:
        if args.spec_only:
            # Just extract and print spec
            spec = extract_spec(args.file, args.source)
            print(json.dumps(spec, indent=2))
        else:
            result = asyncio.run(run_reconstruction_test(args.file, args.source))

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"\nResult saved to: {args.output}")

    else:
        parser.print_help()
