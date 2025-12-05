#!/usr/bin/env python3
"""
GAADP RUNNER - Orchestration Harness for Sufficient Statistic Experiment
==========================================================================

This script processes 30 coding tasks through a Research→Verify pipeline
to test the Research Standard v1.0 (the "Sufficient Statistic").

Architecture:
    - Queue: 30 tasks loaded from benchmarks/test_prompts/
    - Pool: 10 concurrent Researcher+Verifier pairs
    - Logging: Full telemetry to gaadp_experiment/logs/

Workflow per task:
    1. RESEARCHER: Takes raw prompt → produces Research Artifact (JSON)
    2. VERIFIER: Validates artifact against Research Standard v1.0
    3. ARCHITECT: Resolves conflicts, escalates if needed
    4. Log and aggregate results

Usage:
    python gaadp_experiment/gaadp_runner.py
    python gaadp_experiment/gaadp_runner.py --tasks 5  # Run first 5 only
    python gaadp_experiment/gaadp_runner.py --concurrency 5  # Lower concurrency
"""
import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.llm_providers import AnthropicAPIProvider, LLMProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("GAADP.Runner")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Task:
    """A single coding task to process."""
    id: str
    name: str
    prompt_path: str
    prompt_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchArtifact:
    """Output from the Researcher role - follows Research Standard v1.0."""
    task_id: str
    artifact_id: str
    maturity_level: str  # DRAFT, REVIEWABLE, EXECUTABLE, PRODUCTION
    completeness_score: float
    domain: Dict[str, Any]
    contracts: Dict[str, Any]
    verification: Dict[str, Any]
    enforcement: Dict[str, Any]
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """Output from the Verifier role."""
    task_id: str
    artifact_id: str
    verdict: str  # PASS, FAIL, NEEDS_REVISION
    completeness_score: float
    criteria_passed: List[str]
    criteria_failed: List[str]
    feedback: str
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskResult:
    """Complete result for a task."""
    task: Task
    research_artifact: Optional[ResearchArtifact]
    verification_result: Optional[VerificationResult]
    status: str  # COMPLETED, FAILED, TIMEOUT
    attempts: int
    total_cost: float
    total_tokens: Dict[str, int]
    duration_seconds: float
    error: Optional[str] = None


# =============================================================================
# PROMPTS
# =============================================================================

RESEARCHER_SYSTEM_PROMPT = """You are a RESEARCH AGENT in the GAADP (Graph-Aware Autonomous Development Platform).

Your role is to transform raw user prompts into fully-specified Research Artifacts that enable autonomous code generation without further human clarification.

## The Research Standard v1.0

A complete Research Artifact must include:

1. **DOMAIN** - Business context
   - task_category: greenfield | brownfield | algorithmic | systems | debug
   - why: Business context explaining WHY this exists (min 20 chars)
   - success_criteria: Measurable outcomes with test methods

2. **CONTRACTS** - Interface specification
   - interface.inputs: List of typed inputs with validation rules
   - interface.outputs: List of typed outputs with postconditions
   - examples.happy_path: At least 1 normal case
   - examples.edge_cases: At least 1 boundary condition
   - examples.error_cases: At least 1 error scenario
   - ambiguities: Explicitly captured ambiguities (if any)

3. **VERIFICATION** - How to prove correctness
   - complexity_bounds: Time and space complexity with justification
   - test_oracle.unit_tests: Tests that trace to success criteria
   - test_oracle.property_tests: Structural invariants

4. **ENFORCEMENT** - Governance boundaries
   - security.forbidden_patterns: Blocklist of disallowed patterns
   - security.required_clearance: 0-3 clearance level
   - governance.cost_limit: USD budget
   - governance.max_attempts: Retry limit
   - escalation.triggers: When to involve humans

## Your Output Format

You MUST respond with a valid JSON object matching the Research Standard schema.
Do NOT include any text before or after the JSON.

```json
{
  "metadata": {
    "artifact_id": "<uuid>",
    "maturity_level": "REVIEWABLE",
    "completeness_score": 0.75
  },
  "domain": { ... },
  "contracts": { ... },
  "verification": { ... },
  "enforcement": { ... }
}
```

## Key Principles

1. NO AMBIGUOUS PRONOUNS - Every "it", "this", "that" must refer to something named
2. TYPED EXAMPLES REQUIRED - At least 3 examples (happy, edge, error)
3. TESTABLE ASSERTIONS - Every success criterion maps to a test
4. EXPLICIT AMBIGUITIES - Capture unknowns explicitly, don't hide them
5. SECURITY FIRST - Define forbidden patterns, not just "be secure"
"""

VERIFIER_SYSTEM_PROMPT = """You are a VERIFIER AGENT in the GAADP (Graph-Aware Autonomous Development Platform).

Your role is to validate Research Artifacts against the Research Standard v1.0 checklist.

## The Sufficient Statistic Checklist

A Research Artifact is COMPLETE if and only if:

| # | Criterion | Check |
|---|-----------|-------|
| 1 | Input types fully specified | Every input has Python type annotation |
| 2 | Output types fully specified | Every output has Python type annotation |
| 3 | At least 3 examples provided | 1 happy path + 1 edge case + 1 error case minimum |
| 4 | Complexity bounds stated | Time AND space complexity (or N/A with justification) |
| 5 | Dependencies declared | Required imports listed, forbidden patterns noted |
| 6 | Security posture defined | Trust boundary and clearance level set |
| 7 | File structure mapped | All files listed with dependency edges |
| 8 | Acceptance tests defined | At least 1 test per success criterion |
| 9 | Research rationale documented | Approach chosen with justification |
| 10 | No ambiguous pronouns | Every reference is named |

**Threshold:** 8/10 criteria must pass for PASS verdict.

## Your Output Format

Respond with a JSON object:

```json
{
  "verdict": "PASS" | "FAIL" | "NEEDS_REVISION",
  "completeness_score": 0.0-1.0,
  "criteria_passed": ["criterion_1", "criterion_2", ...],
  "criteria_failed": ["criterion_x", ...],
  "feedback": "Detailed feedback for the Researcher if revision needed"
}
```

Be strict. A Builder Agent will use this artifact to generate code.
If the artifact is ambiguous, the Builder will fail or produce wrong code.
"""

ARCHITECT_SYSTEM_PROMPT = """You are an ARCHITECT AGENT in the GAADP (Graph-Aware Autonomous Development Platform).

Your role is to resolve conflicts between Researcher and Verifier, synthesize feedback, and determine next actions.

## Your Responsibilities

1. If Verifier says PASS: Confirm completion
2. If Verifier says FAIL with recoverable feedback: Guide revision
3. If Verifier says FAIL with fundamental issues: Escalate to human
4. If max attempts reached: Summarize blockers and escalate

## Your Output Format

```json
{
  "decision": "COMPLETE" | "REVISE" | "ESCALATE",
  "rationale": "Why this decision",
  "next_action": {
    "actor": "RESEARCHER" | "HUMAN",
    "instruction": "What to do next"
  },
  "blockers": ["list of unresolved issues if escalating"]
}
```
"""


# =============================================================================
# TASK PROCESSOR
# =============================================================================

class TaskProcessor:
    """
    Processes a single task through the Research→Verify pipeline.

    Roles:
        - RESEARCHER: Produces Research Artifact from raw prompt
        - VERIFIER: Validates artifact against Research Standard
        - ARCHITECT: Resolves conflicts, decides next action
    """

    MAX_ATTEMPTS = 3

    def __init__(
        self,
        task: Task,
        provider: LLMProvider,
        log_dir: Path,
        model_config: Dict[str, Any] = None
    ):
        self.task = task
        self.provider = provider
        self.log_dir = log_dir
        self.model_config = model_config or {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4000,
            "temperature": 0.3
        }

        # Create task-specific log file
        self.log_file = log_dir / f"task_{task.id}.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.attempts = 0
        self.total_cost = 0.0
        self.total_tokens = {"input": 0, "output": 0}
        self.events = []

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event to the task log file."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "task_id": self.task.id,
            "data": data
        }
        self.events.append(event)

        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _call_llm(self, role: str, system_prompt: str, user_prompt: str) -> str:
        """Call LLM and track cost."""
        self._log_event("llm_call_start", {"role": role})

        try:
            response = self.provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_config=self.model_config
            )

            # Track cost
            stats = self.provider.get_usage_stats()
            self.total_cost = stats.get("cost", 0.0)
            self.total_tokens = {
                "input": stats.get("tokens_input", 0),
                "output": stats.get("tokens_output", 0)
            }

            self._log_event("llm_call_complete", {
                "role": role,
                "response_length": len(response),
                "cost": self.total_cost
            })

            return response

        except Exception as e:
            self._log_event("llm_call_error", {"role": role, "error": str(e)})
            raise

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        # Try direct parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding { ... } block
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(response[start:end+1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from response: {response[:200]}...")

    async def _run_researcher(self, feedback: str = "") -> ResearchArtifact:
        """Run the Researcher role to produce a Research Artifact."""
        user_prompt = f"""## Raw User Prompt
```
{self.task.prompt_content}
```

## Task Information
- Task ID: {self.task.id}
- Task Name: {self.task.name}

{f"## Feedback from Previous Attempt{chr(10)}{feedback}" if feedback else ""}

Transform this raw prompt into a complete Research Artifact following the Research Standard v1.0.
"""

        response = self._call_llm("RESEARCHER", RESEARCHER_SYSTEM_PROMPT, user_prompt)

        try:
            data = self._parse_json_response(response)

            return ResearchArtifact(
                task_id=self.task.id,
                artifact_id=data.get("metadata", {}).get("artifact_id", str(uuid.uuid4())),
                maturity_level=data.get("metadata", {}).get("maturity_level", "DRAFT"),
                completeness_score=data.get("metadata", {}).get("completeness_score", 0.0),
                domain=data.get("domain", {}),
                contracts=data.get("contracts", {}),
                verification=data.get("verification", {}),
                enforcement=data.get("enforcement", {}),
                raw_response=response
            )
        except Exception as e:
            logger.error(f"Failed to parse Researcher response: {e}")
            return ResearchArtifact(
                task_id=self.task.id,
                artifact_id=str(uuid.uuid4()),
                maturity_level="DRAFT",
                completeness_score=0.0,
                domain={},
                contracts={},
                verification={},
                enforcement={},
                raw_response=response
            )

    async def _run_verifier(self, artifact: ResearchArtifact) -> VerificationResult:
        """Run the Verifier role to validate the Research Artifact."""
        artifact_json = json.dumps({
            "metadata": {
                "artifact_id": artifact.artifact_id,
                "maturity_level": artifact.maturity_level,
                "completeness_score": artifact.completeness_score
            },
            "domain": artifact.domain,
            "contracts": artifact.contracts,
            "verification": artifact.verification,
            "enforcement": artifact.enforcement
        }, indent=2)

        user_prompt = f"""## Research Artifact to Verify

```json
{artifact_json}
```

## Original Task
Task ID: {self.task.id}
Task Name: {self.task.name}

Original prompt:
```
{self.task.prompt_content[:1000]}
```

Validate this artifact against the Research Standard v1.0 checklist.
"""

        response = self._call_llm("VERIFIER", VERIFIER_SYSTEM_PROMPT, user_prompt)

        try:
            data = self._parse_json_response(response)

            return VerificationResult(
                task_id=self.task.id,
                artifact_id=artifact.artifact_id,
                verdict=data.get("verdict", "FAIL"),
                completeness_score=data.get("completeness_score", 0.0),
                criteria_passed=data.get("criteria_passed", []),
                criteria_failed=data.get("criteria_failed", []),
                feedback=data.get("feedback", ""),
                raw_response=response
            )
        except Exception as e:
            logger.error(f"Failed to parse Verifier response: {e}")
            return VerificationResult(
                task_id=self.task.id,
                artifact_id=artifact.artifact_id,
                verdict="FAIL",
                completeness_score=0.0,
                criteria_passed=[],
                criteria_failed=["parse_error"],
                feedback=f"Could not parse verification response: {e}",
                raw_response=response
            )

    async def _run_architect(
        self,
        artifact: ResearchArtifact,
        verification: VerificationResult
    ) -> Dict[str, Any]:
        """Run the Architect to decide next action."""
        user_prompt = f"""## Current State

### Research Artifact
- Artifact ID: {artifact.artifact_id}
- Maturity Level: {artifact.maturity_level}
- Completeness Score: {artifact.completeness_score}

### Verification Result
- Verdict: {verification.verdict}
- Completeness Score: {verification.completeness_score}
- Criteria Passed: {verification.criteria_passed}
- Criteria Failed: {verification.criteria_failed}
- Feedback: {verification.feedback}

### Meta
- Task ID: {self.task.id}
- Attempt: {self.attempts} / {self.MAX_ATTEMPTS}

Decide the next action.
"""

        response = self._call_llm("ARCHITECT", ARCHITECT_SYSTEM_PROMPT, user_prompt)

        try:
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"Failed to parse Architect response: {e}")
            return {
                "decision": "ESCALATE",
                "rationale": f"Could not parse architect decision: {e}",
                "blockers": ["architect_parse_error"]
            }

    async def process(self) -> TaskResult:
        """
        Process the task through the full pipeline.

        Returns:
            TaskResult with final status and artifacts
        """
        start_time = datetime.now()

        self._log_event("task_start", {"task_name": self.task.name})

        artifact = None
        verification = None
        feedback = ""

        try:
            while self.attempts < self.MAX_ATTEMPTS:
                self.attempts += 1
                self._log_event("attempt_start", {"attempt": self.attempts})

                # 1. Run Researcher
                logger.info(f"[{self.task.id[:8]}] Researcher attempt {self.attempts}")
                artifact = await self._run_researcher(feedback)
                self._log_event("artifact_produced", {
                    "artifact_id": artifact.artifact_id,
                    "maturity_level": artifact.maturity_level,
                    "completeness_score": artifact.completeness_score
                })

                # 2. Run Verifier
                logger.info(f"[{self.task.id[:8]}] Verifier checking artifact")
                verification = await self._run_verifier(artifact)
                self._log_event("verification_complete", {
                    "verdict": verification.verdict,
                    "completeness_score": verification.completeness_score
                })

                # 3. Check verdict
                if verification.verdict == "PASS":
                    logger.info(f"[{self.task.id[:8]}] PASS - artifact verified")
                    self._log_event("task_complete", {"status": "COMPLETED"})

                    duration = (datetime.now() - start_time).total_seconds()
                    return TaskResult(
                        task=self.task,
                        research_artifact=artifact,
                        verification_result=verification,
                        status="COMPLETED",
                        attempts=self.attempts,
                        total_cost=self.total_cost,
                        total_tokens=self.total_tokens,
                        duration_seconds=duration
                    )

                # 4. Run Architect for revision decision
                logger.info(f"[{self.task.id[:8]}] Architect deciding next action")
                architect_decision = await self._run_architect(artifact, verification)
                self._log_event("architect_decision", architect_decision)

                if architect_decision.get("decision") == "ESCALATE":
                    logger.warning(f"[{self.task.id[:8]}] Escalated to human")
                    break

                # Prepare feedback for next iteration
                feedback = verification.feedback

            # Max attempts reached or escalated
            duration = (datetime.now() - start_time).total_seconds()
            self._log_event("task_complete", {"status": "FAILED", "reason": "max_attempts"})

            return TaskResult(
                task=self.task,
                research_artifact=artifact,
                verification_result=verification,
                status="FAILED",
                attempts=self.attempts,
                total_cost=self.total_cost,
                total_tokens=self.total_tokens,
                duration_seconds=duration,
                error="Max attempts reached or escalated"
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"[{self.task.id[:8]}] Error: {e}")
            self._log_event("task_error", {"error": str(e)})

            return TaskResult(
                task=self.task,
                research_artifact=artifact,
                verification_result=verification,
                status="FAILED",
                attempts=self.attempts,
                total_cost=self.total_cost,
                total_tokens=self.total_tokens,
                duration_seconds=duration,
                error=str(e)
            )


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class GADPRunner:
    """
    Orchestrates parallel task processing.

    Architecture:
        - Queue: Tasks loaded from benchmarks/test_prompts/
        - Pool: Semaphore-controlled concurrency
        - Aggregation: Results collected and summarized
    """

    def __init__(
        self,
        prompts_dir: Path = None,
        log_dir: Path = None,
        max_concurrency: int = 10,
        max_tasks: int = None
    ):
        self.prompts_dir = prompts_dir or PROJECT_ROOT / "benchmarks" / "test_prompts"
        self.log_dir = log_dir or PROJECT_ROOT / "gaadp_experiment" / "logs"
        self.max_concurrency = max_concurrency
        self.max_tasks = max_tasks

        # Initialize provider
        self.provider = AnthropicAPIProvider()
        if not self.provider.is_available():
            raise RuntimeError("Anthropic API key not found. Set ANTHROPIC_API_KEY.")

        # Semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrency)

        # Results
        self.results: List[TaskResult] = []

        logger.info(f"GADPRunner initialized:")
        logger.info(f"  Prompts: {self.prompts_dir}")
        logger.info(f"  Logs: {self.log_dir}")
        logger.info(f"  Concurrency: {max_concurrency}")

    def load_tasks(self) -> List[Task]:
        """Load tasks from prompt files."""
        tasks = []

        prompt_files = sorted(self.prompts_dir.glob("prompt.*.md"))

        for i, prompt_path in enumerate(prompt_files):
            if self.max_tasks and i >= self.max_tasks:
                break

            # Parse filename: prompt.01.smart_csv.md
            parts = prompt_path.stem.split(".")
            task_num = parts[1] if len(parts) > 1 else str(i+1)
            task_name = parts[2] if len(parts) > 2 else prompt_path.stem

            task = Task(
                id=f"task_{task_num}_{uuid.uuid4().hex[:8]}",
                name=task_name,
                prompt_path=str(prompt_path),
                prompt_content=prompt_path.read_text(),
                metadata={"file": prompt_path.name, "index": i}
            )
            tasks.append(task)

        logger.info(f"Loaded {len(tasks)} tasks from {self.prompts_dir}")
        return tasks

    async def process_task(self, task: Task) -> TaskResult:
        """Process a single task with semaphore control."""
        async with self.semaphore:
            processor = TaskProcessor(
                task=task,
                provider=self.provider,
                log_dir=self.log_dir
            )
            return await processor.process()

    async def run(self) -> Dict[str, Any]:
        """
        Run all tasks and return aggregated results.

        Returns:
            Summary statistics and results
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting run: {run_id}")

        # Load tasks
        tasks = self.load_tasks()
        if not tasks:
            logger.error("No tasks found!")
            return {"error": "No tasks found"}

        # Reset provider stats
        self.provider.reset_usage_stats()

        # Process all tasks concurrently
        start_time = datetime.now()

        logger.info(f"Processing {len(tasks)} tasks with {self.max_concurrency} concurrent workers...")

        results = await asyncio.gather(
            *[self.process_task(task) for task in tasks],
            return_exceptions=True
        )

        # Handle exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed with exception: {result}")
                results[i] = TaskResult(
                    task=tasks[i],
                    research_artifact=None,
                    verification_result=None,
                    status="FAILED",
                    attempts=0,
                    total_cost=0.0,
                    total_tokens={"input": 0, "output": 0},
                    duration_seconds=0.0,
                    error=str(result)
                )

        self.results = results

        # Aggregate statistics
        total_duration = (datetime.now() - start_time).total_seconds()
        provider_stats = self.provider.get_usage_stats()

        completed = sum(1 for r in results if r.status == "COMPLETED")
        failed = sum(1 for r in results if r.status == "FAILED")

        summary = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "tasks": {
                "total": len(tasks),
                "completed": completed,
                "failed": failed,
                "success_rate": completed / len(tasks) if tasks else 0
            },
            "cost": {
                "total_usd": provider_stats.get("cost", 0.0),
                "per_task_avg": provider_stats.get("cost", 0.0) / len(tasks) if tasks else 0
            },
            "tokens": {
                "input": provider_stats.get("tokens_input", 0),
                "output": provider_stats.get("tokens_output", 0)
            },
            "duration_seconds": total_duration,
            "concurrency": self.max_concurrency
        }

        # Save summary
        summary_path = self.log_dir / f"run_{run_id}_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save detailed results
        results_path = self.log_dir / f"run_{run_id}_results.json"
        with open(results_path, "w") as f:
            json.dump([self._result_to_dict(r) for r in results], f, indent=2)

        # Print summary
        logger.info("=" * 60)
        logger.info("RUN COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Run ID: {run_id}")
        logger.info(f"  Tasks: {completed}/{len(tasks)} completed ({summary['tasks']['success_rate']:.1%})")
        logger.info(f"  Cost: ${summary['cost']['total_usd']:.4f}")
        logger.info(f"  Duration: {total_duration:.1f}s")
        logger.info(f"  Summary: {summary_path}")
        logger.info("=" * 60)

        return summary

    def _result_to_dict(self, result: TaskResult) -> Dict[str, Any]:
        """Convert TaskResult to JSON-serializable dict."""
        return {
            "task_id": result.task.id,
            "task_name": result.task.name,
            "status": result.status,
            "attempts": result.attempts,
            "total_cost": result.total_cost,
            "total_tokens": result.total_tokens,
            "duration_seconds": result.duration_seconds,
            "error": result.error,
            "artifact": result.research_artifact.to_dict() if result.research_artifact else None,
            "verification": result.verification_result.to_dict() if result.verification_result else None
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GAADP Runner - Sufficient Statistic Experiment"
    )
    parser.add_argument(
        "--tasks", "-t",
        type=int,
        default=None,
        help="Maximum number of tasks to process (default: all)"
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        help="Maximum concurrent task processors (default: 10)"
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=None,
        help="Directory containing prompt files"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for log files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        runner = GADPRunner(
            prompts_dir=args.prompts_dir,
            log_dir=args.log_dir,
            max_concurrency=args.concurrency,
            max_tasks=args.tasks
        )

        summary = asyncio.run(runner.run())

        # Exit code based on success rate
        success_rate = summary.get("tasks", {}).get("success_rate", 0)
        sys.exit(0 if success_rate >= 0.8 else 1)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
