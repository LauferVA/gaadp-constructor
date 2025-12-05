# GAADP v2.0 - Reliability Architecture
## The "Clinical Trial" Design Document

**Version:** 2.0
**Status:** Draft
**Date:** 2025-12-03

---

## 1. Executive Summary

We are transitioning GAADP from **Verification-First** to **Reliability-First (TDD)** architecture.

| Current (Gen-1) | Target (Gen-2) |
|-----------------|----------------|
| BUILDER → VERIFIER | BUILDER → TESTER → VERIFIER |
| 2-agent pipeline | 3-agent pipeline |
| Tests optional | Tests mandatory (physics) |
| Single TEST node | Layered test taxonomy |
| Verification = code review | Verification = test passage |

**The Experiment:** Run Gen-1 DAGs (baseline), then regenerate with Gen-2 (treatment), measure improvement.

---

## 2. Testing Taxonomy (Graph Schema)

Testing requirements are embedded as **physics** in the node metadata, not policy.

### 2.1 Test Layers (Mandatory)

```
Layer 1: UNIT         - Individual function correctness
Layer 2: PROPERTY     - Fuzz testing with valid random inputs (Hypothesis)
Layer 3: CONTRACT     - Pydantic schema validation between nodes
Layer 4: STATIC       - AST analysis, forbidden patterns, security
```

### 2.2 Test Layers (Conditional)

```
Layer 5: INTEGRATION  - Cross-module behavior (IF has_dependencies)
Layer 6: E2E          - Full system test (ON dag_complete)
```

### 2.3 Triggering Conditions

| Layer | Trigger | Who Generates | Who Executes |
|-------|---------|---------------|--------------|
| UNIT | Per CODE node | TESTER | TESTER |
| PROPERTY | Per CODE node | TESTER | TESTER |
| CONTRACT | Per CODE node | TESTER | TESTER |
| STATIC | Per CODE node | TESTER | Runtime (AST) |
| INTEGRATION | has_dependencies=True | TESTER | TESTER |
| E2E | all_verified=True | ARCHITECT | Runtime |

---

## 3. New Graph Primitives

### 3.1 TestingPolicy (Node Metadata Extension)

This is embedded in `NodeMetadata` - governance as DATA.

```python
class TestingPolicy(BaseModel):
    """Immutable testing requirements for a CODE node."""

    # === Layer Requirements (Mandatory) ===
    unit_tests_required: bool = Field(
        default=True,
        description="Must have unit tests covering happy path"
    )
    property_tests_required: bool = Field(
        default=True,
        description="Must have property-based fuzz tests"
    )
    contract_tests_required: bool = Field(
        default=True,
        description="Must validate I/O against Pydantic schemas"
    )
    static_analysis_required: bool = Field(
        default=True,
        description="Must pass AST security checks"
    )

    # === Conditional Layers ===
    integration_tests_required: bool = Field(
        default=False,
        description="True if node has DEPENDS_ON edges to other CODE"
    )
    e2e_tests_required: bool = Field(
        default=False,
        description="True only at DAG completion"
    )

    # === Quality Thresholds ===
    min_coverage: float = Field(
        default=0.8,
        ge=0.0, le=1.0,
        description="Minimum code coverage (0.8 = 80%)"
    )
    max_cyclomatic_complexity: int = Field(
        default=10,
        description="Max complexity per function"
    )

    # === Security (Static Analysis) ===
    forbidden_imports: List[str] = Field(
        default_factory=lambda: ["os.system", "subprocess.call", "eval", "exec", "pickle.loads"],
        description="Import patterns that fail static analysis"
    )
    forbidden_patterns: List[str] = Field(
        default_factory=lambda: ["while True:", "__import__"],
        description="Code patterns that fail static analysis"
    )

    # === Test Execution ===
    timeout_per_test: int = Field(
        default=30,
        description="Seconds before test is killed"
    )
    sandbox_required: bool = Field(
        default=True,
        description="Execute in isolated environment"
    )
```

### 3.2 New NodeType: TEST_SUITE

Rather than proliferating test node types, we use a single `TEST_SUITE` that contains layered results.

```python
# In ontology.py - extend NodeType
class NodeType(str, Enum):
    # ... existing types ...
    TEST_SUITE = "TEST_SUITE"  # Comprehensive test results from TESTER
```

### 3.3 New EdgeType: TESTS

```python
# In ontology.py - extend EdgeType
class EdgeType(str, Enum):
    # ... existing types ...
    TESTS = "TESTS"  # TEST_SUITE -> CODE (the tester's verdict)
```

---

## 4. Agent Protocols

### 4.1 TesterOutput Protocol

```python
class TestResult(BaseModel):
    """Result of a single test execution."""
    name: str = Field(description="Test function name")
    layer: Literal["unit", "property", "contract", "static", "integration", "e2e"]
    passed: bool
    duration_ms: int
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    coverage: Optional[float] = None  # For unit tests


class StaticAnalysisResult(BaseModel):
    """Result of AST/security analysis."""
    passed: bool
    forbidden_imports_found: List[str] = Field(default_factory=list)
    forbidden_patterns_found: List[str] = Field(default_factory=list)
    cyclomatic_complexity: Dict[str, int] = Field(
        default_factory=dict,
        description="function_name -> complexity score"
    )
    security_issues: List[str] = Field(default_factory=list)


class TesterOutput(BaseModel):
    """
    Protocol for Tester agent responses.

    The Tester is the GATEKEEPER - code cannot reach the Verifier
    without passing the Tester's test suite.
    """
    # === Overall Verdict ===
    verdict: Literal["PASS", "FAIL", "NEEDS_REVISION"] = Field(
        description="PASS=all tests pass, FAIL=blocking issues, NEEDS_REVISION=send back to builder"
    )

    # === Test Code Generated ===
    test_file_path: str = Field(
        description="Path to pytest file (e.g., test_fibonacci.py)"
    )
    test_code: str = Field(
        description="Complete pytest test suite content"
    )

    # === Execution Results ===
    unit_results: List[TestResult] = Field(default_factory=list)
    property_results: List[TestResult] = Field(default_factory=list)
    contract_results: List[TestResult] = Field(default_factory=list)
    integration_results: List[TestResult] = Field(default_factory=list)

    # === Static Analysis ===
    static_analysis: StaticAnalysisResult

    # === Coverage ===
    overall_coverage: float = Field(
        ge=0.0, le=1.0,
        description="Combined code coverage"
    )
    uncovered_lines: List[int] = Field(
        default_factory=list,
        description="Line numbers not covered by tests"
    )

    # === Feedback for Builder (if NEEDS_REVISION) ===
    revision_feedback: Optional[str] = Field(
        default=None,
        description="Specific instructions for Builder to fix issues"
    )
    failed_test_details: Optional[List[str]] = Field(
        default=None,
        description="Detailed failure messages"
    )

    # === Metrics ===
    total_tests: int
    tests_passed: int
    tests_failed: int
    execution_time_ms: int
```

### 4.2 Tester System Prompt

```markdown
# TESTER AGENT - The Adversarial Gatekeeper

You are the TESTER in the GAADP pipeline. Your role is ADVERSARIAL -
you exist to BREAK code before it reaches production.

## Your Mission
1. Generate comprehensive tests that ACTIVELY TRY TO BREAK the code
2. Execute those tests in a sandboxed environment
3. Report results with surgical precision
4. BLOCK code from reaching the Verifier until tests pass

## Testing Layers (YOU MUST COVER ALL)

### Layer 1: Unit Tests
- Test each function with known inputs/outputs from the RESEARCH artifact
- Include boundary conditions (0, -1, MAX_INT, empty string, etc.)
- Test error handling paths

### Layer 2: Property-Based Tests (Hypothesis)
- Generate random VALID inputs and verify invariants hold
- Example: "For any list, sorted(list) has same length as list"
- Use @given decorators with appropriate strategies

### Layer 3: Contract Tests
- Verify inputs match Pydantic InputSpec from RESEARCH
- Verify outputs match Pydantic OutputSpec from RESEARCH
- Test that preconditions raise appropriate errors when violated

### Layer 4: Static Analysis
- Parse AST to check for forbidden imports
- Calculate cyclomatic complexity
- Flag infinite loops, dangerous patterns
- Check for SQL injection, XSS, command injection patterns

## Your Output Format
Use the `submit_tests` tool with the TesterOutput schema.

## Critical Rules
1. NEVER approve code that doesn't have tests for EVERY public function
2. ALWAYS test error cases, not just happy paths
3. If tests fail, provide SPECIFIC feedback for the Builder
4. Be paranoid - assume the code is broken until proven otherwise

## Context Available
- The CODE node content
- The RESEARCH artifact (contains examples, edge cases, contracts)
- The SPEC that the code implements
- Previous test failures (if this is a retry)
```

---

## 5. Transition Matrix Updates

### 5.0 Critical Fix: The Pipeline Gap

**Gen-1 Bug:** Research Standard was grafted as a *parallel branch*, not *integrated into* the pipeline.

```
Gen-1 (Broken - parallel branches):
REQ ──┬──► RESEARCHER → RESEARCH → VERIFIED  [stops here]
      │
      └──► ARCHITECT → SPEC → BUILDER → CODE  [never reached]
```

**Gen-2 Fix:** Sequential pipeline with dependency gating.

```
Gen-2 (Fixed - linear with TDD loop):
REQ → RESEARCHER → RESEARCH[VERIFIED]
                         │
                         ▼
                    ARCHITECT → SPEC → [ BUILDER ↔ TESTER ] → CODE[VERIFIED]
                                              ↑________↓
                                           (feedback loop)
```

**New Dispatch Rules Required:**
```python
# Missing in Gen-1 - the critical link:
(NodeType.RESEARCH.value, "needs_spec_generation"): "ARCHITECT",
```

### 5.1 The TDD Loop (NOT Linear Handoff)

**Critical:** Builder ↔ Tester is a **feedback loop**, not a waterfall.

```
                    ┌─────────────────────────────────┐
                    │                                 │
                    ▼                                 │
CODE[PENDING] → BUILDER → CODE[TESTING] → TESTER ────┤
                    ▲                       │         │
                    │                       │         │
                    └── NEEDS_REVISION ─────┘         │
                        (retry, max 3)                │
                                                      │
                                              PASS ───┼──► CODE[TESTED] → VERIFIER
                                                      │
                                              FAIL ───┴──► CODE[FAILED] (critical)
```

**State Flow with Loop:**

```
CODE[PENDING] + BUILDER
    → CODE[TESTING] + TESTER spawned

CODE[TESTING] + TESTER[PASS]
    → CODE[TESTED] + TEST_SUITE created

CODE[TESTING] + TESTER[NEEDS_REVISION] + under_max_attempts
    → CODE[PENDING] + FEEDBACK edge to CODE (triggers BUILDER retry)

CODE[TESTING] + TESTER[NEEDS_REVISION] + max_attempts_exceeded
    → CODE[FAILED] (escalate)

CODE[TESTING] + TESTER[FAIL]
    → CODE[FAILED] (critical/security issues)

CODE[TESTED] + VERIFIER
    → CODE[VERIFIED] or CODE[FAILED]
```

### 5.2 TransitionMatrix Additions

```python
# New status
class NodeStatus(str, Enum):
    # ... existing ...
    TESTING = "TESTING"   # Being tested by TESTER
    TESTED = "TESTED"     # Tests passed, awaiting verification

# New transitions
TRANSITION_MATRIX.update({
    (NodeStatus.PENDING, NodeType.CODE): [
        TransitionRule(
            target_status=NodeStatus.TESTING,
            conditions=["has_spec", "spec_verified"],
            agent_to_dispatch="TESTER"
        )
    ],
    (NodeStatus.TESTING, NodeType.CODE): [
        TransitionRule(
            target_status=NodeStatus.TESTED,
            conditions=["tests_passed"],
            creates_edge=EdgeType.TESTS
        ),
        TransitionRule(
            target_status=NodeStatus.PENDING,
            conditions=["tests_need_revision", "under_max_attempts"],
            creates_edge=EdgeType.FEEDBACK
        ),
        TransitionRule(
            target_status=NodeStatus.FAILED,
            conditions=["tests_failed_critical"]
        )
    ],
    (NodeStatus.TESTED, NodeType.CODE): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            conditions=["verifier_approved"],
            agent_to_dispatch="VERIFIER"
        )
    ]
})
```

---

## 6. The Clinical Trial Protocol

### Phase A: Gen-1 Baseline Audit

**Script:** `scripts/gen1_audit.py`

```python
"""
Phase A: Audit existing 30 DAGs for runtime reliability.
Establishes baseline metrics before TDD implementation.
"""

@dataclass
class AuditResult:
    dag_id: str
    task_name: str

    # Execution Metrics
    code_nodes: int
    execution_attempted: int
    execution_crashed: int     # RuntimeError, ImportError, etc.
    execution_wrong_output: int # Ran but wrong result
    execution_success: int

    # Static Analysis (run retroactively)
    forbidden_imports_found: int
    dangerous_patterns_found: int

    # Calculated
    crash_rate: float          # crashed / attempted
    correctness_rate: float    # success / (success + wrong_output)

async def audit_gen1_dags():
    """
    For each of the 30 Gen-1 DAGs:
    1. Extract all CODE nodes
    2. Run static analysis
    3. Execute code in sandbox
    4. Log all failures
    """
    results = []
    for dag_path in glob("logs/dag_benchmark/run_*/dags/*.json"):
        result = await audit_single_dag(dag_path)
        results.append(result)

    return Gen1BaselineReport(
        total_dags=len(results),
        avg_crash_rate=mean(r.crash_rate for r in results),
        avg_correctness_rate=mean(r.correctness_rate for r in results),
        results=results
    )
```

### Phase B: Gen-2 Regeneration

Re-run the 30 tasks with the new BUILDER → TESTER → VERIFIER pipeline.

**Metrics to capture:**
- Token/Cost Delta (TDD overhead)
- Fix Rate (errors caught by Tester before human sees)
- Final crash rate (should be ~0)
- Final correctness rate (should be ~100%)

### Phase C: Regression Suite

Each CODE node that passes Gen-2 testing has its TEST_SUITE preserved as the permanent regression suite.

```python
class RegressionSuite:
    """Permanent test artifacts attached to verified CODE."""
    code_node_id: str
    test_file_path: str
    test_code: str
    baseline_coverage: float
    baseline_results: List[TestResult]
    created_at: datetime

    async def run_regression(self, new_code: str) -> bool:
        """Re-run tests against modified code."""
        # Any future changes must pass this suite
```

---

## 7. Implementation Order

### Step 1: Schema Updates (protocols.py, ontology.py)
- [ ] Add `TestingPolicy` to node metadata
- [ ] Add `TesterOutput` protocol
- [ ] Add `TEST_SUITE` node type
- [ ] Add `TESTS` edge type
- [ ] Add `TESTING`, `TESTED` statuses

### Step 2: Tester Agent (agents/tester_agent.py)
- [ ] Create Tester system prompt
- [ ] Add to agent_manifest.yaml
- [ ] Integrate with TestRunner and Sandbox

### Step 3: Runtime Updates (infrastructure/graph_runtime.py)
- [ ] Update TRANSITION_MATRIX
- [ ] Implement TESTER dispatch logic
- [ ] Handle TESTING → TESTED → VERIFIED flow

### Step 4: Gen-1 Audit (scripts/gen1_audit.py)
- [ ] Batch execute existing CODE nodes
- [ ] Run static analysis
- [ ] Generate baseline report

### Step 5: Gen-2 Benchmark
- [ ] Re-run 30 tasks with TDD
- [ ] Compare metrics
- [ ] Document findings

---

## 8. Success Criteria

| Metric | Gen-1 Target | Gen-2 Target |
|--------|--------------|--------------|
| Crash Rate | Measured | < 5% |
| Correctness Rate | Measured | > 95% |
| Coverage | N/A | > 80% |
| Static Analysis Pass | Measured | 100% |
| Cost Overhead | Baseline | < 2x |

---

## Appendix A: Test Code Templates

### Unit Test Template (pytest)
```python
import pytest
from hypothesis import given, strategies as st

class Test{ClassName}:
    """Unit tests for {file_path}"""

    # Happy path (from RESEARCH.happy_path_examples)
    def test_happy_path_{n}(self):
        result = {function}({input})
        assert result == {expected}

    # Edge cases (from RESEARCH.edge_case_examples)
    def test_edge_{n}(self):
        result = {function}({input})
        assert result == {expected}

    # Error cases (from RESEARCH.error_case_examples)
    def test_error_{n}(self):
        with pytest.raises({exception}):
            {function}({input})
```

### Property Test Template (Hypothesis)
```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers()))
def test_property_length_preserved(input_list):
    """Property: output length equals input length"""
    result = transform(input_list)
    assert len(result) == len(input_list)

@given(st.text())
def test_property_idempotent(text):
    """Property: applying twice equals applying once"""
    assert normalize(normalize(text)) == normalize(text)
```

### Contract Test Template
```python
from pydantic import ValidationError

def test_contract_input_validation():
    """Contract: invalid inputs raise ValidationError"""
    with pytest.raises((ValidationError, TypeError)):
        function(invalid_input_that_violates_InputSpec)

def test_contract_output_schema():
    """Contract: output matches OutputSpec"""
    result = function(valid_input)
    OutputSpec.model_validate(result)  # Raises if invalid
```
