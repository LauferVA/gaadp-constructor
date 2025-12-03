#!/usr/bin/env python3
"""
GAADP Test Runner
Streamlines testing by accepting prompts as arguments or files.

Usage:
    python scripts/run_test.py "Create a function that adds two numbers"
    python scripts/run_test.py --file .prompts/test_protocols.md
    python scripts/run_test.py --file .prompts/test_protocols.md --dry-run
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path

# Ensure we're in the project root
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


def write_prompt(content: str):
    """Write content to prompt.md"""
    with open("prompt.md", "w") as f:
        f.write(content)
    print(f"[TEST] Wrote {len(content)} chars to prompt.md")


def run_pipeline(timeout: int = 300) -> dict:
    """Run the GAADP pipeline and capture output."""
    print("[TEST] Starting GAADP pipeline...")
    print("=" * 60)

    try:
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
            timeout=timeout,
            input="n\nn\n",  # Auto-answer "no" to domain discovery and data loading
            cwd=PROJECT_ROOT
        )

        print(result.stdout)
        if result.stderr:
            print("[STDERR]", result.stderr)

        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Pipeline timed out after {timeout}s")
        return {"exit_code": -1, "stdout": "", "stderr": "Timeout", "success": False}
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        return {"exit_code": -1, "stdout": "", "stderr": str(e), "success": False}


def analyze_output(result: dict) -> dict:
    """Analyze the pipeline output for test validation."""
    stdout = result.get("stdout", "")

    analysis = {
        "architect_ran": "Architect is thinking" in stdout,
        "builder_ran": "Builder is coding" in stdout,
        "verifier_ran": "Verifier is judging" in stdout,
        "verdict_pass": "Verdict: PASS" in stdout,
        "verdict_fail": "Verdict: FAIL" in stdout,
        "success_chain": "SUCCESS: Code Verified" in stdout,
        "graph_nodes": None,
        "graph_edges": None,
        "errors": [],
    }

    # Extract graph stats
    if "KNOWLEDGE GRAPH STATE" in stdout:
        for line in stdout.split("\n"):
            if "Nodes:" in line and "Edges:" in line:
                try:
                    parts = line.split("|")
                    analysis["graph_nodes"] = int(parts[0].split(":")[1].strip())
                    analysis["graph_edges"] = int(parts[1].split(":")[1].strip())
                except:
                    pass

    # Check for errors
    if "Traceback" in stdout or "Traceback" in result.get("stderr", ""):
        analysis["errors"].append("Python traceback detected")
    if "parse_error" in stdout.lower():
        analysis["errors"].append("Parse error detected")
    if "Protocol output failed" in stdout:
        analysis["errors"].append("Protocol output failed")

    return analysis


def print_analysis(analysis: dict):
    """Print test analysis in a readable format."""
    print("\n" + "=" * 60)
    print("TEST ANALYSIS")
    print("=" * 60)

    checks = [
        ("Architect ran", analysis["architect_ran"]),
        ("Builder ran", analysis["builder_ran"]),
        ("Verifier ran", analysis["verifier_ran"]),
        ("Verdict PASS", analysis["verdict_pass"]),
        ("Success chain", analysis["success_chain"]),
    ]

    for name, passed in checks:
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}")

    if analysis["graph_nodes"] is not None:
        print(f"  📊 Graph: {analysis['graph_nodes']} nodes, {analysis['graph_edges']} edges")

    if analysis["errors"]:
        print("\n  ⚠️ ERRORS DETECTED:")
        for err in analysis["errors"]:
            print(f"     - {err}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="GAADP Test Runner")
    parser.add_argument("prompt", nargs="?", help="Prompt text (inline)")
    parser.add_argument("--file", "-f", help="Path to prompt file")
    parser.add_argument("--dry-run", action="store_true", help="Just write prompt, don't run")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")

    args = parser.parse_args()

    # Get prompt content
    if args.file:
        with open(args.file, "r") as f:
            prompt_content = f.read()
        print(f"[TEST] Loaded prompt from: {args.file}")
    elif args.prompt:
        prompt_content = args.prompt
    else:
        parser.print_help()
        sys.exit(1)

    # Write prompt
    write_prompt(prompt_content)

    if args.dry_run:
        print("[TEST] Dry run - not executing pipeline")
        print("\nPrompt content:")
        print("-" * 40)
        print(prompt_content[:500])
        if len(prompt_content) > 500:
            print(f"... [{len(prompt_content) - 500} more chars]")
        return

    # Run pipeline
    result = run_pipeline(timeout=args.timeout)

    # Analyze and report
    analysis = analyze_output(result)
    print_analysis(analysis)

    # Exit with appropriate code
    if not analysis["errors"] and (analysis["verdict_pass"] or analysis["verdict_fail"]):
        print("\n✅ TEST COMPLETED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("\n❌ TEST HAD ISSUES")
        sys.exit(1)


if __name__ == "__main__":
    main()
