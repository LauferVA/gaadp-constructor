"""
ESCALATION CONTROLLER
Handles failure escalation when Builder/Verifier loops exhaust retries.

The escalation flow is:
1. Builder attempts to implement SPEC (up to MAX_BUILD_RETRIES)
2. If all attempts fail, escalate to Architect for re-planning
3. Architect analyzes failures and produces new/modified SPEC
4. Builder retries with new SPEC
5. If escalation also fails, escalate to human intervention

Escalation Levels:
- Level 0: Normal retry loop (Builder retries with feedback)
- Level 1: Architect re-planning (strategy change required)
- Level 2: Human intervention (system cannot solve this)

This module provides:
- EscalationContext: Tracks failure history for a spec
- EscalationController: Decides when/how to escalate
- Hooks for custom escalation handlers
"""
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = logging.getLogger("GAADP.Escalation")


class EscalationLevel(Enum):
    """Escalation severity levels."""
    RETRY = 0           # Normal retry with feedback
    ARCHITECT = 1       # Escalate to Architect for re-planning
    HUMAN = 2           # Escalate to human intervention


@dataclass
class FailureRecord:
    """Record of a single failure attempt."""
    attempt: int
    code_id: str
    verdict: str
    critique: str
    issues: List[Dict]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        return {
            "attempt": self.attempt,
            "code_id": self.code_id,
            "verdict": self.verdict,
            "critique": self.critique,
            "issues": self.issues,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EscalationContext:
    """
    Tracks the failure history for a single spec.

    Used to build context for escalation decisions and
    to provide rich feedback to the Architect during re-planning.
    """
    spec_id: str
    spec_content: str
    requirement_id: str
    failures: List[FailureRecord] = field(default_factory=list)
    escalation_level: EscalationLevel = EscalationLevel.RETRY
    architect_attempts: int = 0  # How many times Architect re-planned

    def add_failure(self, code_id: str, verdict: str, critique: str, issues: List[Dict]) -> None:
        """Record a new failure."""
        self.failures.append(FailureRecord(
            attempt=len(self.failures) + 1,
            code_id=code_id,
            verdict=verdict,
            critique=critique,
            issues=issues
        ))
        logger.info(f"Recorded failure {len(self.failures)} for spec {self.spec_id[:8]}")

    def get_failure_summary(self) -> str:
        """Build a summary of all failures for escalation prompt."""
        if not self.failures:
            return "No failures recorded."

        summary_parts = [
            f"FAILURE HISTORY FOR SPEC {self.spec_id[:8]}:",
            f"Total attempts: {len(self.failures)}",
            f"Current escalation level: {self.escalation_level.name}",
            ""
        ]

        for failure in self.failures:
            summary_parts.extend([
                f"--- Attempt {failure.attempt} ---",
                f"Verdict: {failure.verdict}",
                f"Critique: {failure.critique[:200]}{'...' if len(failure.critique) > 200 else ''}",
            ])
            if failure.issues:
                summary_parts.append("Issues:")
                for issue in failure.issues[:5]:
                    severity = issue.get('severity', 'unknown')
                    desc = issue.get('description', 'no description')
                    summary_parts.append(f"  - [{severity}] {desc}")
            summary_parts.append("")

        return "\n".join(summary_parts)

    def get_common_failure_patterns(self) -> List[str]:
        """Analyze failures to identify common patterns."""
        patterns = []

        # Collect all issues
        all_issues = []
        all_critiques = []
        for f in self.failures:
            all_issues.extend(f.issues)
            all_critiques.append(f.critique.lower())

        # Check for common patterns
        import_issues = sum(1 for i in all_issues if i.get('category') == 'import')
        type_issues = sum(1 for i in all_issues if i.get('category') == 'type')
        logic_issues = sum(1 for i in all_issues if i.get('category') == 'logic')

        if import_issues >= 2:
            patterns.append("REPEATED_IMPORT_ERRORS: Multiple import/dependency issues")

        if type_issues >= 2:
            patterns.append("REPEATED_TYPE_ERRORS: Multiple type mismatch issues")

        if logic_issues >= 2:
            patterns.append("REPEATED_LOGIC_ERRORS: Multiple logic/implementation issues")

        # Check critique text for patterns
        critique_text = " ".join(all_critiques)
        if "incomplete" in critique_text or "missing" in critique_text:
            patterns.append("INCOMPLETE_IMPLEMENTATION: Code is missing required functionality")

        if "syntax" in critique_text:
            patterns.append("SYNTAX_ISSUES: Code has syntax errors")

        if "security" in critique_text:
            patterns.append("SECURITY_CONCERNS: Security vulnerabilities detected")

        return patterns if patterns else ["NO_CLEAR_PATTERN: Failures don't follow obvious pattern"]

    def should_escalate(self, max_retries: int, escalate_after: int) -> EscalationLevel:
        """
        Determine if escalation is needed and at what level.

        Args:
            max_retries: Maximum number of Builder retry attempts
            escalate_after: After how many failures to escalate to Architect

        Returns:
            The recommended escalation level
        """
        failure_count = len(self.failures)

        if failure_count < escalate_after:
            return EscalationLevel.RETRY

        if failure_count >= max_retries:
            if self.architect_attempts >= 2:
                # Architect has tried twice, escalate to human
                return EscalationLevel.HUMAN
            else:
                return EscalationLevel.ARCHITECT

        # Between escalate_after and max_retries
        return EscalationLevel.ARCHITECT


# Type alias for escalation handlers
EscalationHandler = Callable[[EscalationContext], Dict]


class EscalationController:
    """
    Controls the escalation flow for failed builds.

    Provides hooks for custom escalation handlers and manages
    the decision-making process for escalation levels.
    """

    def __init__(
        self,
        max_retries: int = 3,
        escalate_after: int = 2,
        max_architect_attempts: int = 2
    ):
        """
        Initialize the escalation controller.

        Args:
            max_retries: Maximum Builder retry attempts before escalation
            escalate_after: Failures before escalating to Architect
            max_architect_attempts: Max Architect re-planning attempts before human escalation
        """
        self.max_retries = max_retries
        self.escalate_after = escalate_after
        self.max_architect_attempts = max_architect_attempts

        # Track escalation contexts by spec_id
        self.contexts: Dict[str, EscalationContext] = {}

        # Escalation handlers (can be customized)
        self._architect_handler: Optional[EscalationHandler] = None
        self._human_handler: Optional[EscalationHandler] = None

    def get_or_create_context(
        self,
        spec_id: str,
        spec_content: str,
        requirement_id: str
    ) -> EscalationContext:
        """Get existing context or create new one."""
        if spec_id not in self.contexts:
            self.contexts[spec_id] = EscalationContext(
                spec_id=spec_id,
                spec_content=spec_content,
                requirement_id=requirement_id
            )
        return self.contexts[spec_id]

    def record_failure(
        self,
        spec_id: str,
        code_id: str,
        verify_output: Dict
    ) -> EscalationLevel:
        """
        Record a build/verify failure and determine next action.

        Args:
            spec_id: The spec that failed
            code_id: The failed code artifact
            verify_output: The verification output

        Returns:
            The recommended escalation level
        """
        if spec_id not in self.contexts:
            logger.warning(f"No context for spec {spec_id}, creating placeholder")
            self.contexts[spec_id] = EscalationContext(
                spec_id=spec_id,
                spec_content="(unknown)",
                requirement_id="(unknown)"
            )

        ctx = self.contexts[spec_id]

        # Record the failure
        ctx.add_failure(
            code_id=code_id,
            verdict=verify_output.get('verdict', 'UNKNOWN'),
            critique=str(verify_output.get('critique', 'No critique')),
            issues=verify_output.get('issues', [])
        )

        # Determine escalation level
        level = ctx.should_escalate(self.max_retries, self.escalate_after)
        ctx.escalation_level = level

        logger.info(f"Spec {spec_id[:8]} failure #{len(ctx.failures)}: escalation={level.name}")
        return level

    def build_architect_escalation_prompt(self, spec_id: str) -> str:
        """
        Build the escalation prompt for the Architect.

        This prompt includes:
        - Original spec content
        - Failure history
        - Common failure patterns
        - Instructions for re-planning
        """
        ctx = self.contexts.get(spec_id)
        if not ctx:
            return "ERROR: No context found for spec"

        patterns = ctx.get_common_failure_patterns()

        prompt = f"""
ESCALATION: SPEC IMPLEMENTATION FAILED REPEATEDLY

{ctx.get_failure_summary()}

FAILURE PATTERN ANALYSIS:
{chr(10).join(f"- {p}" for p in patterns)}

ORIGINAL SPECIFICATION:
{ctx.spec_content}

REQUIRED ACTION:
The Builder has failed {len(ctx.failures)} times to implement this specification.
You must RE-PLAN with a DIFFERENT approach.

Consider:
1. Is the specification too complex? Break it into smaller specs.
2. Are there missing dependencies? Add them explicitly.
3. Is the approach fundamentally flawed? Try a different strategy.
4. Are there implicit requirements not captured in the spec?

DO NOT simply re-state the original spec. You MUST change your approach.
"""
        return prompt

    def register_architect_handler(self, handler: EscalationHandler) -> None:
        """Register a custom handler for Architect escalation."""
        self._architect_handler = handler

    def register_human_handler(self, handler: EscalationHandler) -> None:
        """Register a custom handler for human escalation."""
        self._human_handler = handler

    async def execute_escalation(
        self,
        spec_id: str,
        level: EscalationLevel,
        architect_agent=None,
        db=None
    ) -> Dict:
        """
        Execute the escalation action.

        Args:
            spec_id: The spec being escalated
            level: The escalation level
            architect_agent: The Architect agent (for ARCHITECT level)
            db: The graph database

        Returns:
            Result dict with new_spec or action instructions
        """
        ctx = self.contexts.get(spec_id)
        if not ctx:
            return {"error": "No context found"}

        if level == EscalationLevel.RETRY:
            # Just return feedback for retry
            return {
                "action": "RETRY",
                "feedback": ctx.failures[-1].critique if ctx.failures else ""
            }

        elif level == EscalationLevel.ARCHITECT:
            if self._architect_handler:
                return self._architect_handler(ctx)

            if architect_agent:
                # Call Architect with escalation context
                escalation_prompt = self.build_architect_escalation_prompt(spec_id)
                ctx.architect_attempts += 1

                result = await architect_agent.process({
                    "nodes": [{
                        "content": ctx.spec_content,
                        "id": spec_id,
                        "escalation_context": escalation_prompt
                    }]
                })

                return {
                    "action": "REPLAN",
                    "architect_output": result,
                    "attempt": ctx.architect_attempts
                }

            return {
                "action": "ARCHITECT_REQUIRED",
                "prompt": self.build_architect_escalation_prompt(spec_id)
            }

        elif level == EscalationLevel.HUMAN:
            if self._human_handler:
                return self._human_handler(ctx)

            return {
                "action": "HUMAN_INTERVENTION_REQUIRED",
                "message": f"Spec {spec_id} has failed {len(ctx.failures)} times after {ctx.architect_attempts} Architect re-plans.",
                "failure_summary": ctx.get_failure_summary(),
                "patterns": ctx.get_common_failure_patterns()
            }

        return {"error": f"Unknown escalation level: {level}"}

    def get_stats(self) -> Dict:
        """Get escalation statistics."""
        total_failures = sum(len(ctx.failures) for ctx in self.contexts.values())
        architect_escalations = sum(1 for ctx in self.contexts.values() if ctx.architect_attempts > 0)
        human_escalations = sum(1 for ctx in self.contexts.values() if ctx.escalation_level == EscalationLevel.HUMAN)

        return {
            "total_specs_tracked": len(self.contexts),
            "total_failures": total_failures,
            "architect_escalations": architect_escalations,
            "human_escalations": human_escalations,
            "success_after_escalation": sum(
                1 for ctx in self.contexts.values()
                if ctx.architect_attempts > 0 and ctx.escalation_level != EscalationLevel.HUMAN
            )
        }

    def clear(self) -> None:
        """Clear all tracked contexts."""
        self.contexts.clear()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.DEBUG)

    print("=== Test: Escalation Flow ===")

    controller = EscalationController(max_retries=3, escalate_after=2)

    # Create context
    ctx = controller.get_or_create_context(
        spec_id="spec_001",
        spec_content="Implement a user authentication system",
        requirement_id="req_001"
    )

    # Simulate failures
    print("\nSimulating 3 failures...")

    # Failure 1 - should continue retrying
    level = controller.record_failure("spec_001", "code_001", {
        "verdict": "FAIL",
        "critique": "Missing import for hashlib",
        "issues": [{"severity": "error", "category": "import", "description": "hashlib not imported"}]
    })
    print(f"After failure 1: {level.name}")
    assert level == EscalationLevel.RETRY

    # Failure 2 - should escalate to Architect
    level = controller.record_failure("spec_001", "code_002", {
        "verdict": "FAIL",
        "critique": "Type mismatch in password comparison",
        "issues": [{"severity": "error", "category": "type", "description": "Comparing bytes to str"}]
    })
    print(f"After failure 2: {level.name}")
    assert level == EscalationLevel.ARCHITECT

    # Failure 3 - max retries reached, architect escalation
    level = controller.record_failure("spec_001", "code_003", {
        "verdict": "FAIL",
        "critique": "Logic error in password hashing",
        "issues": [{"severity": "error", "category": "logic", "description": "Hash not salted"}]
    })
    print(f"After failure 3: {level.name}")
    assert level == EscalationLevel.ARCHITECT

    # Test escalation prompt
    print("\n=== Escalation Prompt ===")
    prompt = controller.build_architect_escalation_prompt("spec_001")
    print(prompt[:500] + "...")

    # Test patterns
    print("\n=== Failure Patterns ===")
    patterns = ctx.get_common_failure_patterns()
    for p in patterns:
        print(f"  - {p}")

    # Test stats
    print("\n=== Stats ===")
    stats = controller.get_stats()
    print(f"  Total specs: {stats['total_specs_tracked']}")
    print(f"  Total failures: {stats['total_failures']}")

    print("\n✅ All tests passed!")
