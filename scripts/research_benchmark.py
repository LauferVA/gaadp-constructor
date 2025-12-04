#!/usr/bin/env python3
"""
RESEARCH BENCHMARK - Test Research Standard v1.0 with Integrated Infrastructure
================================================================================

This script tests the Research→Verify pipeline using the integrated:
- ResearcherOutput and ResearchVerifierOutput protocols (core/protocols.py)
- RESEARCHER and RESEARCH_VERIFIER agents (config/agent_manifest.yaml)
- RESEARCH node type and transitions (core/ontology.py)

Usage:
    python scripts/research_benchmark.py
    python scripts/research_benchmark.py --tasks 5
    python scripts/research_benchmark.py --task "Smart CSV Parser"
"""
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.llm_providers import AnthropicAPIProvider
from core.protocols import (
    ResearcherOutput, ResearchVerifierOutput,
    get_agent_tools, protocol_to_tool_schema, load_agent_manifest
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Research.Benchmark")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TaskResult:
    """Complete result for a task."""
    task_id: str
    task_name: str
    success: bool
    research_output: Optional[Dict[str, Any]] = None
    verification_output: Optional[Dict[str, Any]] = None
    attempts: int = 0
    total_cost: float = 0.0
    total_tokens: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    error: Optional[str] = None


# =============================================================================
# DETERMINISTIC VERIFICATION (Criteria 1-9)
# =============================================================================

def verify_structure(artifact: Dict[str, Any]) -> tuple[int, List[str], Dict[str, bool]]:
    """
    Deterministic verification of Research Artifact structure.
    Checks criteria 1-9 programmatically. Criterion 10 (pronouns) needs LLM.

    Returns: (criteria_passed, issues, criterion_flags)
    """
    issues = []
    flags = {}

    # Criterion 1: Input types fully specified
    inputs = artifact.get("inputs", [])
    c1_pass = (
        len(inputs) > 0 and
        all(i.get("type") and i.get("validation") for i in inputs)
    )
    flags["criterion_1_input_types"] = c1_pass
    if not c1_pass:
        issues.append("Criterion 1: Inputs missing or lack type/validation")

    # Criterion 2: Output types fully specified
    outputs = artifact.get("outputs", [])
    c2_pass = len(outputs) > 0 and all(o.get("type") for o in outputs)
    flags["criterion_2_output_types"] = c2_pass
    if not c2_pass:
        issues.append("Criterion 2: Outputs missing or lack type annotation")

    # Criterion 3: At least 3 examples (happy, edge, error)
    c3_pass = (
        len(artifact.get("happy_path_examples", [])) >= 1 and
        len(artifact.get("edge_case_examples", [])) >= 1 and
        len(artifact.get("error_case_examples", [])) >= 1
    )
    flags["criterion_3_examples"] = c3_pass
    if not c3_pass:
        issues.append("Criterion 3: Missing happy_path, edge_case, or error_case examples")

    # Criterion 4: Complexity bounds stated
    c4_pass = bool(
        artifact.get("complexity_time") and
        artifact.get("complexity_space") and
        artifact.get("complexity_justification")
    )
    flags["criterion_4_complexity"] = c4_pass
    if not c4_pass:
        issues.append("Criterion 4: Missing complexity_time, complexity_space, or justification")

    # Criterion 5: Dependencies declared (can be empty list)
    c5_pass = "dependencies" in artifact
    flags["criterion_5_dependencies"] = c5_pass
    if not c5_pass:
        issues.append("Criterion 5: Missing dependencies field")

    # Criterion 6: Security posture defined
    c6_pass = bool(
        artifact.get("forbidden_patterns") is not None and
        artifact.get("trust_boundary")
    )
    flags["criterion_6_security"] = c6_pass
    if not c6_pass:
        issues.append("Criterion 6: Missing forbidden_patterns or trust_boundary")

    # Criterion 7: File structure (conditional - single file is OK)
    task_category = artifact.get("task_category", "")
    if task_category in ["algorithmic", "debug"]:
        c7_pass = True  # Single file acceptable for algorithmic tasks
    else:
        c7_pass = bool(artifact.get("files") or artifact.get("entry_point"))
    flags["criterion_7_files"] = c7_pass
    if not c7_pass:
        issues.append("Criterion 7: Multi-file project needs files array")

    # Criterion 8: Acceptance tests with traces_to_criterion
    unit_tests = artifact.get("unit_tests", [])
    c8_pass = (
        len(unit_tests) >= 1 and
        all(t.get("traces_to_criterion") is not None for t in unit_tests)
    )
    flags["criterion_8_tests"] = c8_pass
    if not c8_pass:
        issues.append("Criterion 8: Missing unit_tests or traces_to_criterion links")

    # Criterion 9: Research rationale documented
    c9_pass = bool(artifact.get("reasoning"))
    flags["criterion_9_rationale"] = c9_pass
    if not c9_pass:
        issues.append("Criterion 9: Missing reasoning field")

    # Criterion 10: Pronouns - CANNOT check deterministically, assume pass
    # LLM verifier will override if it finds issues
    flags["criterion_10_no_ambiguity"] = True  # Optimistic, LLM will correct

    criteria_passed = sum(flags.values())
    return criteria_passed, issues, flags


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class ResearchBenchmarkRunner:
    """
    Runs Research→Verify pipeline using integrated infrastructure.
    """

    # Model configuration
    MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 4000

    def __init__(
        self,
        provider: AnthropicAPIProvider = None,
        max_attempts: int = 3,
        log_dir: Path = None
    ):
        self.provider = provider or AnthropicAPIProvider()
        self.max_attempts = max_attempts
        self.log_dir = log_dir or Path("logs/research_benchmark")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Load agent configs
        self.manifest = load_agent_manifest()

        # Get tool schemas from integrated protocols
        self.researcher_tools = get_agent_tools("RESEARCHER")
        self.verifier_tools = get_agent_tools("RESEARCH_VERIFIER")

        logger.info(f"Initialized with model {self.MODEL}")
        logger.info(f"Researcher tool: {self.researcher_tools[0]['name']}")
        logger.info(f"Verifier tool: {self.verifier_tools[0]['name']}")

    async def run_researcher(self, prompt: str) -> tuple[Dict[str, Any], Dict[str, int], float]:
        """
        Run the Researcher agent using the integrated protocol.

        Returns: (output_dict, token_counts, cost)
        """
        # Load system prompt from manifest (inline in YAML)
        researcher_config = self.manifest.get_config("RESEARCHER")
        system_prompt = researcher_config.system_prompt

        user_prompt = f"Transform this prompt into a Research Artifact:\n\n{prompt}"

        # Model config with forced tool choice
        model_config = {
            "model": self.MODEL,
            "max_tokens": self.MAX_TOKENS,
            "temperature": 0.7,
            "tool_choice": {"type": "tool", "name": "submit_research"}
        }

        # Track cost before call
        cost_before = self.provider._cost_session

        # Run synchronously via asyncio.to_thread
        response_str = await asyncio.to_thread(
            self.provider.call,
            system_prompt,
            user_prompt,
            model_config,
            self.researcher_tools
        )

        # Parse tool response (provider returns JSON with tool_calls array)
        tool_output = None
        try:
            response_data = json.loads(response_str)
            if isinstance(response_data, dict):
                # Extract from tool_calls[0].input structure
                tool_calls = response_data.get("tool_calls", [])
                if tool_calls and len(tool_calls) > 0:
                    tool_output = tool_calls[0].get("input", {})
                else:
                    # Fallback: maybe it's a direct input
                    tool_output = response_data.get("input", response_data)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse response as JSON: {response_str[:200]}")

        # Calculate cost delta
        cost_after = self.provider._cost_session
        cost = cost_after - cost_before

        tokens = {
            "input": self.provider._token_usage.get("input", 0),
            "output": self.provider._token_usage.get("output", 0)
        }

        return tool_output, tokens, cost

    async def run_verifier(self, research_artifact: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, int], float]:
        """
        Run the Research Verifier using the integrated protocol.

        Returns: (output_dict, token_counts, cost)
        """
        # Load system prompt from manifest (inline in YAML)
        verifier_config = self.manifest.get_config("RESEARCH_VERIFIER")
        system_prompt = verifier_config.system_prompt

        user_prompt = f"Verify this Research Artifact against the 10-criterion checklist:\n\n{json.dumps(research_artifact, indent=2)}"

        # Model config with forced tool choice
        model_config = {
            "model": self.MODEL,
            "max_tokens": self.MAX_TOKENS,
            "temperature": 0.7,
            "tool_choice": {"type": "tool", "name": "submit_research_verdict"}
        }

        # Track cost before call
        cost_before = self.provider._cost_session

        # Run synchronously via asyncio.to_thread
        response_str = await asyncio.to_thread(
            self.provider.call,
            system_prompt,
            user_prompt,
            model_config,
            self.verifier_tools
        )

        # Parse tool response (provider returns JSON with tool_calls array)
        tool_output = None
        try:
            response_data = json.loads(response_str)
            if isinstance(response_data, dict):
                # Extract from tool_calls[0].input structure
                tool_calls = response_data.get("tool_calls", [])
                if tool_calls and len(tool_calls) > 0:
                    tool_output = tool_calls[0].get("input", {})
                else:
                    # Fallback: maybe it's a direct input
                    tool_output = response_data.get("input", response_data)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse verifier response as JSON: {response_str[:200]}")

        # Calculate cost delta
        cost_after = self.provider._cost_session
        cost = cost_after - cost_before

        tokens = {
            "input": self.provider._token_usage.get("input", 0),
            "output": self.provider._token_usage.get("output", 0)
        }

        return tool_output, tokens, cost

    async def run_tiebreaker(self, research_artifact: Dict[str, Any], det_score: int, llm_score: int) -> tuple[Dict[str, Any], Dict[str, int], float]:
        """
        Run independent tiebreaker verification when deterministic and LLM disagree.
        Uses different temperature for diversity.
        """
        system_prompt = """You are an INDEPENDENT Research Artifact Tiebreaker.

Two verification methods disagreed:
- Deterministic check: {det_score}/10 criteria
- LLM verifier: {llm_score}/10 criteria

Your job is to provide a THIRD OPINION. Be rigorous and impartial.
Focus especially on borderline cases. Check each criterion carefully.

Use the submit_research_verdict tool with your independent assessment.""".format(
            det_score=det_score, llm_score=llm_score
        )

        user_prompt = f"Provide independent verification of this Research Artifact:\n\n{json.dumps(research_artifact, indent=2)}"

        # Different temperature for diversity
        model_config = {
            "model": self.MODEL,
            "max_tokens": self.MAX_TOKENS,
            "temperature": 0.3,  # Lower temp for more deterministic tiebreaker
            "tool_choice": {"type": "tool", "name": "submit_research_verdict"}
        }

        cost_before = self.provider._cost_session

        response_str = await asyncio.to_thread(
            self.provider.call,
            system_prompt,
            user_prompt,
            model_config,
            self.verifier_tools
        )

        tool_output = None
        try:
            response_data = json.loads(response_str)
            if isinstance(response_data, dict):
                tool_calls = response_data.get("tool_calls", [])
                if tool_calls and len(tool_calls) > 0:
                    tool_output = tool_calls[0].get("input", {})
                else:
                    tool_output = response_data.get("input", response_data)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse tiebreaker response as JSON")

        cost_after = self.provider._cost_session
        cost = cost_after - cost_before
        tokens = {
            "input": self.provider._token_usage.get("input", 0),
            "output": self.provider._token_usage.get("output", 0)
        }

        return tool_output, tokens, cost

    async def process_task(self, task_id: str, task_name: str, prompt: str) -> TaskResult:
        """Process a single task through Research→Verify pipeline."""
        start_time = datetime.now()
        result = TaskResult(task_id=task_id, task_name=task_name, success=False)

        try:
            for attempt in range(1, self.max_attempts + 1):
                result.attempts = attempt
                logger.info(f"[{task_name}] Attempt {attempt}/{self.max_attempts}")

                # Run Researcher
                logger.info(f"[{task_name}] Running Researcher...")
                research_output, research_tokens, research_cost = await self.run_researcher(prompt)

                if not research_output:
                    logger.warning(f"[{task_name}] Researcher returned no output")
                    continue

                result.research_output = research_output
                result.total_cost += research_cost
                result.total_tokens["input"] = result.total_tokens.get("input", 0) + research_tokens["input"]
                result.total_tokens["output"] = result.total_tokens.get("output", 0) + research_tokens["output"]

                # === HYBRID VERIFICATION ===
                # Step 1: Deterministic structural check (fast, free)
                logger.info(f"[{task_name}] Running structural check...")
                det_passed, det_issues, det_flags = verify_structure(research_output)
                logger.info(f"[{task_name}] Structural check: {det_passed}/10 criteria")

                # Step 2: If structural check fails badly, skip LLM
                if det_passed < 6:
                    logger.warning(f"[{task_name}] Structural FAIL ({det_passed}/10) - skipping LLM verifier")
                    result.verification_output = {
                        "verdict": "FAIL",
                        "criteria_passed": det_passed,
                        "completeness_score": det_passed / 10.0,
                        "issues": det_issues,
                        "verification_method": "deterministic_only",
                        **det_flags
                    }
                    continue

                # Step 3: Run LLM verifier for semantic checks
                logger.info(f"[{task_name}] Running LLM Verifier...")
                verify_output, verify_tokens, verify_cost = await self.run_verifier(research_output)

                if not verify_output:
                    logger.warning(f"[{task_name}] Verifier returned no output")
                    continue

                result.total_cost += verify_cost
                result.total_tokens["input"] = result.total_tokens.get("input", 0) + verify_tokens["input"]
                result.total_tokens["output"] = result.total_tokens.get("output", 0) + verify_tokens["output"]

                # Recalculate LLM score from boolean flags (Admonition C)
                llm_flags = [
                    verify_output.get("criterion_1_input_types", False),
                    verify_output.get("criterion_2_output_types", False),
                    verify_output.get("criterion_3_examples", False),
                    verify_output.get("criterion_4_complexity", False),
                    verify_output.get("criterion_5_dependencies", False),
                    verify_output.get("criterion_6_security", False),
                    verify_output.get("criterion_7_files", False),
                    verify_output.get("criterion_8_tests", False),
                    verify_output.get("criterion_9_rationale", False),
                    verify_output.get("criterion_10_no_ambiguity", False),
                ]
                llm_passed = sum(llm_flags)

                # Step 4: Check for discordance (≥2 criteria difference)
                discordance = abs(det_passed - llm_passed)
                final_passed = llm_passed
                final_method = "llm"

                if discordance >= 2:
                    logger.warning(f"[{task_name}] DISCORDANCE: Deterministic={det_passed}, LLM={llm_passed} (diff={discordance})")
                    logger.info(f"[{task_name}] Running Tiebreaker...")

                    tiebreaker_output, tie_tokens, tie_cost = await self.run_tiebreaker(
                        research_output, det_passed, llm_passed
                    )
                    result.total_cost += tie_cost
                    result.total_tokens["input"] = result.total_tokens.get("input", 0) + tie_tokens["input"]
                    result.total_tokens["output"] = result.total_tokens.get("output", 0) + tie_tokens["output"]

                    if tiebreaker_output:
                        tie_flags = [
                            tiebreaker_output.get(f"criterion_{i}_{['input_types','output_types','examples','complexity','dependencies','security','files','tests','rationale','no_ambiguity'][i-1]}", False)
                            for i in range(1, 11)
                        ]
                        tie_passed = sum(tie_flags)
                        logger.info(f"[{task_name}] Tiebreaker: {tie_passed}/10")

                        # Use median of three scores
                        scores = sorted([det_passed, llm_passed, tie_passed])
                        final_passed = scores[1]  # Median
                        final_method = "tiebreaker_median"
                        logger.info(f"[{task_name}] Final score (median): {final_passed}/10")

                # Build final verification output
                final_score = final_passed / 10.0
                final_verdict = "PASS" if final_passed >= 8 else "FAIL"

                verify_output["criteria_passed"] = final_passed
                verify_output["completeness_score"] = final_score
                verify_output["verdict"] = final_verdict
                verify_output["verification_method"] = final_method
                verify_output["deterministic_score"] = det_passed
                verify_output["llm_score"] = llm_passed

                result.verification_output = verify_output

                if final_verdict == "PASS":
                    result.success = True
                    logger.info(f"[{task_name}] PASS ({final_passed}/10 criteria)")
                    break
                else:
                    logger.warning(f"[{task_name}] FAIL ({final_passed}/10 criteria)")
                    issues = verify_output.get("issues", [])
                    for issue in issues[:3]:
                        logger.warning(f"  - {issue}")

        except Exception as e:
            logger.error(f"[{task_name}] Error: {e}")
            result.error = str(e)

        result.duration_seconds = (datetime.now() - start_time).total_seconds()

        # Save task log (sanitize filename to remove slashes)
        safe_name = task_name.replace(' ', '_').replace('/', '_')
        log_path = self.log_dir / f"task_{task_id}_{safe_name}.json"
        with open(log_path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        return result


# =============================================================================
# BENCHMARK TASKS
# =============================================================================

BENCHMARK_TASKS = [
    ("task_01", "Smart CSV Parser", "Create a Python function that parses a CSV file and infers the type of each column (int, float, string, bool, date)."),
    ("task_02", "ls Replica", "Create a Python script that replicates the behavior of the Unix 'ls -la' command."),
    ("task_03", "URL Shortener", "Create a URL shortener service with a hash function and in-memory storage."),
    ("task_04", "Markdown Parser", "Create a Markdown parser that converts markdown text to HTML."),
    ("task_05", "JSON Schema Validator", "Create a JSON schema validator that validates JSON documents against a schema."),
    ("task_06", "Rate Limiter", "Create a rate limiter using the token bucket algorithm."),
    ("task_07", "LRU Cache", "Create an LRU cache with O(1) get and put operations."),
    ("task_08", "Binary Search Tree", "Create a binary search tree with insert, delete, and search operations."),
    ("task_09", "Graph BFS/DFS", "Create graph traversal algorithms for breadth-first and depth-first search."),
    ("task_10", "Merkle Tree", "Create a Merkle tree implementation for data integrity verification."),
]


# =============================================================================
# MAIN
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Research Standard v1.0 Benchmark")
    parser.add_argument("--tasks", type=int, default=None, help="Number of tasks to run")
    parser.add_argument("--task", type=str, default=None, help="Run single task by name")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent workers")
    args = parser.parse_args()

    # Filter tasks
    tasks = BENCHMARK_TASKS
    if args.task:
        tasks = [(tid, name, prompt) for tid, name, prompt in tasks if args.task.lower() in name.lower()]
        if not tasks:
            logger.error(f"No task matching '{args.task}'")
            return
    elif args.tasks:
        tasks = tasks[:args.tasks]

    logger.info(f"Running {len(tasks)} tasks with concurrency {args.concurrency}")

    # Create runner
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/research_benchmark/run_{run_id}")
    runner = ResearchBenchmarkRunner(log_dir=log_dir)

    # Process tasks with semaphore
    semaphore = asyncio.Semaphore(args.concurrency)
    results = []

    async def process_with_semaphore(task_id, task_name, prompt):
        async with semaphore:
            return await runner.process_task(task_id, task_name, prompt)

    # Run all tasks concurrently
    start_time = datetime.now()
    task_coros = [process_with_semaphore(tid, name, prompt) for tid, name, prompt in tasks]
    results = await asyncio.gather(*task_coros)

    # Summary
    total_duration = (datetime.now() - start_time).total_seconds()
    successes = sum(1 for r in results if r.success)
    total_cost = sum(r.total_cost for r in results)

    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Tasks: {len(results)}")
    logger.info(f"Success rate: {successes}/{len(results)} ({100*successes/len(results):.1f}%)")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Duration: {total_duration:.1f}s")

    # Save summary
    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "tasks": {
            "total": len(results),
            "completed": successes,
            "failed": len(results) - successes,
            "success_rate": successes / len(results) if results else 0
        },
        "cost": {"total_usd": total_cost},
        "duration_seconds": total_duration
    }

    summary_path = log_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Logs saved to: {log_dir}")


if __name__ == "__main__":
    asyncio.run(main())
