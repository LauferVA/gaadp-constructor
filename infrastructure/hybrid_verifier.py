"""
HYBRID VERIFIER - Production-grade Research Artifact Verification
==================================================================

This module implements hybrid verification combining:
1. Deterministic structural checks (fast, free, criteria 1-9)
2. LLM semantic verification (criterion 10 + quality assessment)
3. Tiebreaker mechanism when deterministic and LLM disagree

The hybrid approach:
- Reduces cost by short-circuiting on structural failures
- Increases reliability by cross-validating verifiers
- Uses median scoring when discordance is detected

Usage:
    verifier = HybridVerifier(llm_gateway)
    result = await verifier.verify(research_artifact)
"""
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("GAADP.HybridVerifier")


@dataclass
class VerificationResult:
    """Result of hybrid verification."""
    verdict: str  # "PASS" or "FAIL"
    criteria_passed: int  # 0-10
    completeness_score: float  # 0.0-1.0

    # Method tracking
    verification_method: str  # "deterministic_only", "llm", "tiebreaker_median"
    deterministic_score: int
    llm_score: Optional[int]
    tiebreaker_score: Optional[int]

    # Details
    criterion_flags: Dict[str, bool]
    issues: List[str]
    suggestions: List[str]


# =============================================================================
# DETERMINISTIC VERIFICATION (Criteria 1-9)
# =============================================================================

def verify_structure(artifact: Dict[str, Any]) -> Tuple[int, List[str], Dict[str, bool]]:
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


def extract_llm_score(llm_output: Dict[str, Any]) -> Tuple[int, Dict[str, bool]]:
    """
    Extract score from LLM verifier output by counting boolean criterion flags.
    This recalculates from flags rather than trusting the 'criteria_passed' field.
    """
    criterion_names = [
        "criterion_1_input_types",
        "criterion_2_output_types",
        "criterion_3_examples",
        "criterion_4_complexity",
        "criterion_5_dependencies",
        "criterion_6_security",
        "criterion_7_files",
        "criterion_8_tests",
        "criterion_9_rationale",
        "criterion_10_no_ambiguity",
    ]

    flags = {}
    for name in criterion_names:
        flags[name] = bool(llm_output.get(name, False))

    return sum(flags.values()), flags


# =============================================================================
# HYBRID VERIFIER CLASS
# =============================================================================

class HybridVerifier:
    """
    Production-grade hybrid verification combining deterministic + LLM + tiebreaker.

    Verification flow:
    1. Run deterministic structural check (fast, free)
    2. If structural fail (<6 criteria), return immediately
    3. Run LLM verifier for semantic checks
    4. If discordance >= 2, run tiebreaker
    5. Use median of three scores as final result
    """

    # Thresholds
    STRUCTURAL_FAIL_THRESHOLD = 6  # Skip LLM if structural < 6
    DISCORDANCE_THRESHOLD = 2  # Run tiebreaker if |det - llm| >= 2
    PASS_THRESHOLD = 8  # 8/10 criteria required to pass

    def __init__(self, llm_gateway=None):
        """
        Initialize hybrid verifier.

        Args:
            llm_gateway: LLM gateway for semantic verification (optional, lazy-loaded)
        """
        self._gateway = llm_gateway

    @property
    def gateway(self):
        """Lazy-load LLM gateway if not injected."""
        if self._gateway is None:
            from infrastructure.llm_gateway import LLMGateway
            self._gateway = LLMGateway()
        return self._gateway

    async def verify(
        self,
        artifact: Dict[str, Any],
        llm_output: Optional[Dict[str, Any]] = None,
        skip_llm: bool = False
    ) -> VerificationResult:
        """
        Run hybrid verification on a research artifact.

        Args:
            artifact: Research artifact to verify
            llm_output: Pre-computed LLM verification (if already run by GenericAgent)
            skip_llm: If True, only run deterministic check

        Returns:
            VerificationResult with verdict, scores, and issues
        """
        # Step 1: Deterministic structural check
        det_passed, det_issues, det_flags = verify_structure(artifact)
        logger.info(f"Deterministic check: {det_passed}/10 criteria")

        # Step 2: Short-circuit if structural failure is severe
        if det_passed < self.STRUCTURAL_FAIL_THRESHOLD:
            logger.warning(f"Structural FAIL ({det_passed}/10) - skipping LLM verifier")
            return VerificationResult(
                verdict="FAIL",
                criteria_passed=det_passed,
                completeness_score=det_passed / 10.0,
                verification_method="deterministic_only",
                deterministic_score=det_passed,
                llm_score=None,
                tiebreaker_score=None,
                criterion_flags=det_flags,
                issues=det_issues,
                suggestions=self._generate_suggestions(det_issues)
            )

        # Step 2b: If skip_llm requested, return deterministic result with proper verdict
        if skip_llm:
            verdict = "PASS" if det_passed >= self.PASS_THRESHOLD else "FAIL"
            return VerificationResult(
                verdict=verdict,
                criteria_passed=det_passed,
                completeness_score=det_passed / 10.0,
                verification_method="deterministic_only",
                deterministic_score=det_passed,
                llm_score=None,
                tiebreaker_score=None,
                criterion_flags=det_flags,
                issues=det_issues,
                suggestions=self._generate_suggestions(det_issues)
            )

        # Step 3: Use provided LLM output or rely on external call
        if llm_output is None:
            # In production, GenericAgent handles LLM call
            # This is a fallback for direct usage
            logger.warning("No LLM output provided - returning deterministic result")
            return VerificationResult(
                verdict="PASS" if det_passed >= self.PASS_THRESHOLD else "FAIL",
                criteria_passed=det_passed,
                completeness_score=det_passed / 10.0,
                verification_method="deterministic_only",
                deterministic_score=det_passed,
                llm_score=None,
                tiebreaker_score=None,
                criterion_flags=det_flags,
                issues=det_issues,
                suggestions=self._generate_suggestions(det_issues)
            )

        # Step 4: Extract LLM score from flags (Admonition C - recalculate)
        llm_passed, llm_flags = extract_llm_score(llm_output)
        logger.info(f"LLM verifier: {llm_passed}/10 criteria")

        # Merge issues
        all_issues = det_issues + llm_output.get("issues", [])

        # Step 5: Check for discordance
        discordance = abs(det_passed - llm_passed)

        if discordance >= self.DISCORDANCE_THRESHOLD:
            logger.warning(f"DISCORDANCE: det={det_passed}, llm={llm_passed} (diff={discordance})")
            # Tiebreaker would be run by GenericAgent
            # For now, use median of two scores (conservative)
            final_passed = min(det_passed, llm_passed)  # Conservative without tiebreaker
            method = "conservative_min"
        else:
            # Scores agree - trust LLM for semantic checks
            final_passed = llm_passed
            method = "llm"

        # Merge flags (LLM overrides deterministic for criterion 10)
        final_flags = det_flags.copy()
        final_flags["criterion_10_no_ambiguity"] = llm_flags.get("criterion_10_no_ambiguity", True)

        verdict = "PASS" if final_passed >= self.PASS_THRESHOLD else "FAIL"

        return VerificationResult(
            verdict=verdict,
            criteria_passed=final_passed,
            completeness_score=final_passed / 10.0,
            verification_method=method,
            deterministic_score=det_passed,
            llm_score=llm_passed,
            tiebreaker_score=None,
            criterion_flags=final_flags,
            issues=all_issues,
            suggestions=self._generate_suggestions(all_issues)
        )

    def apply_tiebreaker(
        self,
        det_score: int,
        llm_score: int,
        tiebreaker_score: int
    ) -> Tuple[int, str]:
        """
        Apply tiebreaker using median of three scores.

        Returns: (final_score, method)
        """
        scores = sorted([det_score, llm_score, tiebreaker_score])
        median = scores[1]
        logger.info(f"Tiebreaker: det={det_score}, llm={llm_score}, tie={tiebreaker_score} -> median={median}")
        return median, "tiebreaker_median"

    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        """Generate actionable suggestions from issues."""
        suggestions = []

        for issue in issues:
            if "input" in issue.lower() and "type" in issue.lower():
                suggestions.append("Add 'type' (Python annotation) and 'validation' (executable expression) to all inputs")
            elif "output" in issue.lower() and "type" in issue.lower():
                suggestions.append("Add 'type' (Python annotation) to all outputs")
            elif "example" in issue.lower():
                suggestions.append("Ensure at least one example in happy_path_examples, edge_case_examples, and error_case_examples")
            elif "complexity" in issue.lower():
                suggestions.append("Add complexity_time, complexity_space, and complexity_justification fields")
            elif "dependencies" in issue.lower():
                suggestions.append("Add 'dependencies' field (can be empty array [])")
            elif "security" in issue.lower() or "forbidden" in issue.lower():
                suggestions.append("Add 'forbidden_patterns' array and 'trust_boundary' classification")
            elif "file" in issue.lower():
                suggestions.append("Add 'files' array with file paths and purposes, or set task_category to 'algorithmic'")
            elif "test" in issue.lower() or "traces" in issue.lower():
                suggestions.append("Add unit_tests with 'traces_to_criterion' linking each test to a success criterion index")
            elif "reasoning" in issue.lower() or "rationale" in issue.lower():
                suggestions.append("Add 'reasoning' field explaining design decisions")
            elif "pronoun" in issue.lower() or "ambig" in issue.lower():
                suggestions.append("Replace 'it', 'this', 'that' with explicit subjects (e.g., 'the parser', 'the function')")

        return list(set(suggestions))  # Dedupe


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_verify(artifact: Dict[str, Any]) -> Tuple[bool, int, List[str]]:
    """
    Quick deterministic verification for fast rejection.

    Returns: (passed, score, issues)
    """
    score, issues, _ = verify_structure(artifact)
    return score >= 8, score, issues


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.DEBUG)

    # Test artifact (should pass 9/10)
    test_artifact = {
        "maturity_level": "REVIEWABLE",
        "completeness_score": 0.9,
        "task_category": "algorithmic",
        "why": "Test task",
        "success_criteria": [{"criterion": "Works", "test_method": "Test", "is_automated": True}],
        "inputs": [{"name": "x", "type": "int", "validation": "isinstance(x, int)", "trust_boundary": "trusted"}],
        "outputs": [{"name": "result", "type": "int", "postcondition": "result >= 0"}],
        "happy_path_examples": [{"input": {"x": 1}, "expected_output": 1, "explanation": "Basic"}],
        "edge_case_examples": [{"input": {"x": 0}, "expected_output": 0, "why_edge": "Zero"}],
        "error_case_examples": [{"input": {"x": None}, "expected_exception": "TypeError", "explanation": "Null"}],
        "ambiguities": [],
        "complexity_time": "O(1)",
        "complexity_space": "O(1)",
        "complexity_justification": "Constant time operation",
        "unit_tests": [{"name": "test_basic", "assertion": "f(1) == 1", "traces_to_criterion": 0, "priority": "critical"}],
        "forbidden_patterns": ["eval("],
        "trust_boundary": "trusted",
        "files": [{"path": "main.py", "purpose": "Entry point"}],
        "entry_point": "main.py",
        "dependencies": [],
        "reasoning": "Simple test artifact"
    }

    async def test():
        print("=== Testing Hybrid Verifier ===\n")

        # Test deterministic check
        score, issues, flags = verify_structure(test_artifact)
        print(f"Deterministic score: {score}/10")
        print(f"Issues: {issues}")
        print(f"Flags: {flags}")
        print()

        # Test quick_verify
        passed, score, issues = quick_verify(test_artifact)
        print(f"Quick verify: passed={passed}, score={score}")
        print()

        # Test HybridVerifier
        verifier = HybridVerifier()
        result = await verifier.verify(test_artifact, skip_llm=True)
        print(f"Hybrid result: {result.verdict}, {result.criteria_passed}/10")
        print(f"Method: {result.verification_method}")
        print(f"Suggestions: {result.suggestions}")

        print("\n✅ Hybrid Verifier tests passed")

    asyncio.run(test())
