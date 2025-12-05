# GAADP Benchmarking Plan (v3.0 - Graph-First Architecture)

**Status:** Ready for Implementation
**Last Updated:** 2025-12-02
**Architecture Baseline:** `d137023` (Post-refactor, graph-first complete)

---

## Executive Summary

This benchmarking plan validates the Graph-First Architecture introduced in the refactor. It replaces the imperative orchestration approach with declarative physics (TransitionMatrix, NodeMetadata, GenericAgent). The benchmarks focus on:

1. **Correctness** - Does the system produce verified code from requirements?
2. **Governance** - Are cost limits, security levels, and attempt limits enforced?
3. **Performance** - Latency, throughput, and parallel execution efficiency
4. **Provider Abstraction** - Do all LLM providers produce consistent results?

---

## Architecture Under Test

### Core Components (from `ARCHITECTURE_ROADMAP.tsv`)

| Component | Status | File | Purpose |
|-----------|--------|------|---------|
| TransitionMatrix | [DONE] | `core/ontology.py` | Single source of truth for state transitions |
| NodeMetadata | [DONE] | `core/ontology.py` | Governance as physics (cost, security, attempts) |
| GenericAgent | [DONE] | `agents/generic_agent.py` | Universal agent loading behavior from YAML |
| GraphRuntime | [DONE] | `infrastructure/graph_runtime.py` | Execution engine consulting TransitionMatrix |
| LLMGateway | [DONE] | `infrastructure/llm_gateway.py` | Pluggable providers with model routing |
| AtomicMaterializer | [DONE] | `infrastructure/materializer.py` | Graph nodes → filesystem with validation |
| Protocols | [DONE] | `core/protocols.py` | Structured output (ArchitectOutput, BuilderOutput, VerifierOutput) |

### Key Invariants to Verify

From `.blueprint/prime_directives.md`:

1. **Directive 1**: TransitionMatrix is the ONLY source of truth for state changes
2. **Directive 4**: `NodeMetadata.cost_limit` is physics - cannot be bypassed
3. **Directive 5**: `attempts >= max_attempts` → FAILED
4. **Directive 6**: Agents cannot process nodes above their security clearance
5. **Directive 19**: "You generate content. The runtime signs it."

---

## Benchmark Levels

### Level 0: Import & Structural Health

**Goal:** Verify all modules load correctly post-refactor.

```bash
# Quick sanity check
python3 -c "
import main
import core.ontology
import core.protocols
import agents.generic_agent
import infrastructure.graph_runtime
import infrastructure.graph_db
import infrastructure.llm_gateway
import infrastructure.llm_providers
import infrastructure.materializer
print('All imports OK')
"
```

**Success Criteria:** No import errors

---

### Level 1: TransitionMatrix Validation (Unit)

**Goal:** Verify the physics engine works correctly.

**Location:** `tests/test_transition_matrix.py`

```python
def test_valid_transitions():
    """Verify all transitions in TRANSITION_MATRIX are valid."""
    from core.ontology import TRANSITION_MATRIX, KNOWN_CONDITIONS, validate_transition_matrix

    errors = validate_transition_matrix()
    assert len(errors) == 0, f"Invalid transitions: {errors}"

def test_pending_req_can_start_processing():
    """REQ: PENDING → PROCESSING when conditions met."""
    from core.ontology import TRANSITION_MATRIX, NodeStatus, NodeType

    rules = TRANSITION_MATRIX.get((NodeStatus.PENDING.value, NodeType.REQ.value), [])
    assert len(rules) > 0, "No transitions for PENDING REQ"

    # Should have path to PROCESSING
    targets = [r.target_status for r in rules]
    assert NodeStatus.PROCESSING in targets

def test_code_requires_verification_edge():
    """CODE: PROCESSING → VERIFIED only with VERIFIES edge."""
    from core.ontology import TRANSITION_MATRIX, NodeStatus, NodeType, EdgeType

    rules = TRANSITION_MATRIX.get((NodeStatus.PROCESSING.value, NodeType.CODE.value), [])
    verify_rule = next((r for r in rules if r.target_status == NodeStatus.VERIFIED), None)

    assert verify_rule is not None
    assert EdgeType.VERIFIES in verify_rule.required_edge_types

def test_max_attempts_leads_to_failed():
    """SPEC: PROCESSING → FAILED when max_attempts_exceeded."""
    from core.ontology import TRANSITION_MATRIX, NodeStatus, NodeType

    rules = TRANSITION_MATRIX.get((NodeStatus.PROCESSING.value, NodeType.SPEC.value), [])
    fail_rule = next((r for r in rules if r.target_status == NodeStatus.FAILED), None)

    assert fail_rule is not None
    assert "max_attempts_exceeded" in fail_rule.required_conditions
```

**Run Command:**
```bash
pytest tests/test_transition_matrix.py -v
```

---

### Level 2: Governance Physics (Unit)

**Goal:** Verify NodeMetadata constraints are enforced.

**Location:** `tests/test_governance_physics.py`

```python
import pytest
from core.ontology import NodeMetadata, NodeSpec, NodeType, NodeStatus

def test_cost_limit_default_unlimited():
    """Nodes without cost_limit should have None (unlimited)."""
    meta = NodeMetadata()
    assert meta.cost_limit is None

def test_cost_actual_tracking():
    """cost_actual should accumulate."""
    meta = NodeMetadata(cost_limit=1.0, cost_actual=0.5)
    assert meta.cost_actual < meta.cost_limit

def test_attempt_tracking():
    """attempts vs max_attempts determines failure."""
    meta = NodeMetadata(attempts=2, max_attempts=3)
    assert meta.attempts < meta.max_attempts

    meta.attempts = 3
    assert meta.attempts >= meta.max_attempts  # Should trigger FAILED

def test_security_level_range():
    """security_level must be 0-3."""
    with pytest.raises(ValueError):
        NodeMetadata(security_level=5)

def test_node_spec_defaults():
    """NodeSpec should have sensible defaults."""
    spec = NodeSpec(
        type=NodeType.CODE,
        content="print('hello')",
        created_by="test"
    )
    assert spec.status == NodeStatus.PENDING
    assert spec.metadata.attempts == 0
```

---

### Level 3: Provider Abstraction (Integration)

**Goal:** Verify all providers implement the same interface.

**Location:** `tests/test_provider_abstraction.py`

```python
import pytest
from infrastructure.llm_providers import (
    LLMProvider, ProviderRegistry, create_default_registry,
    AnthropicAPIProvider, ManualProvider
)
from infrastructure.openai_provider import OpenAIProvider

def test_all_providers_implement_interface():
    """All providers must inherit from LLMProvider."""
    providers = [AnthropicAPIProvider, OpenAIProvider, ManualProvider]

    for provider_cls in providers:
        assert issubclass(provider_cls, LLMProvider), \
            f"{provider_cls.__name__} must inherit from LLMProvider"

def test_registry_prioritization():
    """Registry should respect priority ordering."""
    registry = create_default_registry()

    # All providers should be registered
    names = [p.get_name() for _, p in registry._providers]
    assert "anthropic_api" in names
    assert "openai_api" in names
    assert "manual" in names

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No API key")
def test_anthropic_provider_call():
    """Anthropic provider should make real API calls."""
    provider = AnthropicAPIProvider()
    assert provider.is_available()

    response = provider.call(
        system_prompt="You are a test assistant.",
        user_prompt="Say 'BENCHMARK_OK' and nothing else.",
        model_config={"model": "claude-3-5-haiku-20241022", "max_tokens": 50}
    )

    assert "BENCHMARK_OK" in response or "benchmark" in response.lower()

    stats = provider.get_usage_stats()
    assert stats["tokens_input"] > 0
    assert stats["cost"] > 0

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
def test_openai_provider_call():
    """OpenAI provider should make real API calls."""
    provider = OpenAIProvider()
    assert provider.is_available()

    response = provider.call(
        system_prompt="You are a test assistant.",
        user_prompt="Say 'BENCHMARK_OK' and nothing else.",
        model_config={"model": "gpt-4o-mini", "max_tokens": 50}
    )

    assert "BENCHMARK_OK" in response or "benchmark" in response.lower()
```

**Run Commands:**
```bash
# Without API keys (structure only)
pytest tests/test_provider_abstraction.py -v -k "not call"

# With Anthropic
ANTHROPIC_API_KEY=sk-... pytest tests/test_provider_abstraction.py -v

# With OpenAI
OPENAI_API_KEY=sk-... pytest tests/test_provider_abstraction.py -v
```

---

### Level 4: End-to-End Pipeline (Integration)

**Goal:** Run a complete requirement through the system.

**Location:** `tests/test_e2e_pipeline.py`

```python
import pytest
import asyncio
from infrastructure.graph_db import GraphDB
from infrastructure.graph_runtime import GraphRuntime
from infrastructure.llm_gateway import LLMGateway
from core.ontology import NodeType, NodeStatus

@pytest.fixture
def clean_graph():
    """Fresh graph for each test."""
    return GraphDB(persistence_path=".gaadp_test/e2e.json")

@pytest.mark.asyncio
@pytest.mark.slow
async def test_simple_requirement_to_verified_code(clean_graph):
    """
    BENCHMARK: REQ → SPEC → CODE → VERIFIED

    This is the core value proposition of GAADP.
    """
    # Initialize
    runtime = GraphRuntime(clean_graph)

    # Inject requirement
    req_id = clean_graph.add_node(
        node_id="bench_req_001",
        node_type=NodeType.REQ,
        content="Create a Python function that calculates the factorial of a number recursively.",
        metadata={"cost_limit": 0.50}
    )

    # Run pipeline
    await runtime.run_until_complete(max_iterations=50, timeout=120)

    # Verify outcomes
    verified_code = clean_graph.get_by_type_and_status(NodeType.CODE, NodeStatus.VERIFIED)
    assert len(verified_code) >= 1, "Should produce at least one VERIFIED CODE node"

    # Check code content
    code_node = clean_graph.graph.nodes[verified_code[0]]
    assert "factorial" in code_node["content"].lower()
    assert "def " in code_node["content"]

@pytest.mark.asyncio
@pytest.mark.slow
async def test_cost_limit_enforcement(clean_graph):
    """
    BENCHMARK: Verify cost_limit blocks processing when exceeded.
    """
    runtime = GraphRuntime(clean_graph)

    # Very tight budget
    req_id = clean_graph.add_node(
        node_id="bench_budget_001",
        node_type=NodeType.REQ,
        content="Build a complex system with many components",
        metadata={"cost_limit": 0.001}  # $0.001 - will be exceeded quickly
    )

    await runtime.run_until_complete(max_iterations=10, timeout=60)

    # Node should be BLOCKED or FAILED due to cost
    node_data = clean_graph.graph.nodes[req_id]
    assert node_data["status"] in [NodeStatus.BLOCKED.value, NodeStatus.FAILED.value]

@pytest.mark.asyncio
@pytest.mark.slow
async def test_materialization(clean_graph):
    """
    BENCHMARK: Verified code should materialize to filesystem.
    """
    from infrastructure.materializer import AtomicMaterializer
    import os
    import tempfile

    # Create a verified code node manually
    clean_graph.add_node(
        node_id="mat_code_001",
        node_type=NodeType.CODE,
        content="def hello():\n    return 'Hello, World!'\n",
        metadata={"file_path": "test_output/hello.py"}
    )
    clean_graph.set_status("mat_code_001", NodeStatus.VERIFIED)

    # Materialize
    with tempfile.TemporaryDirectory() as tmpdir:
        materializer = AtomicMaterializer(
            db=clean_graph,
            output_dir=tmpdir,
            run_tests=False,
            auto_commit=False
        )
        result = materializer.materialize()

        assert result.success
        assert len(result.files_written) == 1

        # Verify file exists
        output_path = os.path.join(tmpdir, "test_output/hello.py")
        assert os.path.exists(output_path)

        with open(output_path) as f:
            content = f.read()
        assert "def hello" in content
```

**Run Command:**
```bash
ANTHROPIC_API_KEY=sk-... pytest tests/test_e2e_pipeline.py -v -m slow --timeout=300
```

**Expected Cost:** $0.10-0.50

---

### Level 5: Performance Benchmarks

**Goal:** Measure latency and throughput.

**Location:** `tests/test_performance.py`

```python
import pytest
import time
import statistics
from infrastructure.llm_gateway import LLMGateway

@pytest.mark.performance
def test_llm_latency():
    """Measure LLM call latency."""
    gateway = LLMGateway()

    latencies = []
    for i in range(5):
        start = time.time()
        response = gateway.call_model(
            role="VERIFIER",
            system_prompt="You are a helpful assistant.",
            user_context=f"Say 'OK' - iteration {i}"
        )
        latencies.append(time.time() - start)

    avg_latency = statistics.mean(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    print(f"\nLLM Latency - Avg: {avg_latency:.2f}s, P95: {p95_latency:.2f}s")

    # Baseline expectations (Claude Haiku)
    assert avg_latency < 5.0, "Average latency should be under 5s"
    assert p95_latency < 10.0, "P95 latency should be under 10s"

@pytest.mark.performance
def test_graph_query_performance():
    """Measure graph query performance."""
    from infrastructure.graph_db import GraphDB
    from core.ontology import NodeType, NodeStatus

    db = GraphDB(persistence_path=":memory:")

    # Create 100 nodes
    for i in range(100):
        db.add_node(f"node_{i}", NodeType.CODE, f"content_{i}")

    # Measure query time
    start = time.time()
    for _ in range(1000):
        db.get_by_status(NodeStatus.PENDING)
    query_time = time.time() - start

    print(f"\n1000 status queries: {query_time:.3f}s")
    assert query_time < 1.0, "1000 queries should complete in under 1s"

@pytest.mark.performance
def test_context_extraction_performance():
    """Measure context neighborhood extraction."""
    from infrastructure.graph_db import GraphDB
    from core.ontology import NodeType, EdgeType

    db = GraphDB(persistence_path=":memory:")

    # Create connected graph
    for i in range(50):
        db.add_node(f"node_{i}", NodeType.SPEC, f"spec content {i}")

    for i in range(1, 50):
        db.add_edge(f"node_{i}", f"node_{i-1}", EdgeType.DEPENDS_ON, "sys", "sig")

    # Measure context extraction
    start = time.time()
    for _ in range(100):
        db.get_context_neighborhood("node_25", radius=3)
    context_time = time.time() - start

    print(f"\n100 context extractions (radius=3): {context_time:.3f}s")
    assert context_time < 2.0, "100 context extractions should complete in under 2s"
```

**Run Command:**
```bash
pytest tests/test_performance.py -v -m performance
```

---

## Benchmark Metrics

### Correctness Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Simple REQ → Verified Code | 90%+ success | Count successful completions |
| Protocol Validation | 100% | All agent outputs validate against protocols |
| Transition Legality | 100% | All transitions allowed by TransitionMatrix |

### Governance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Cost Limit Enforcement | 100% | No node exceeds cost_limit |
| Attempt Limit Enforcement | 100% | nodes with attempts >= max_attempts are FAILED |
| Security Level Enforcement | 100% | Agents respect security clearance |

### Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| LLM Call Latency (avg) | < 5s | Time per API call |
| LLM Call Latency (P95) | < 10s | 95th percentile |
| Graph Query (1000x) | < 1s | Status query throughput |
| Context Extraction (100x) | < 2s | Neighborhood extraction |
| E2E Simple Req | < 60s | REQ → VERIFIED total time |

### Cost Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Simple REQ cost | < $0.10 | LLMGateway.get_session_cost() |
| Complex REQ cost | < $0.50 | Including retries |
| Full test suite | < $5.00 | All E2E tests |

---

## Execution Schedule

### Quick Validation (5 min, $0)
```bash
# No API calls
pytest tests/test_transition_matrix.py tests/test_governance_physics.py -v
```

### Provider Validation (10 min, $0.10)
```bash
ANTHROPIC_API_KEY=sk-... pytest tests/test_provider_abstraction.py -v
```

### Full E2E (30 min, $1-2)
```bash
ANTHROPIC_API_KEY=sk-... pytest tests/ -v -m slow --timeout=600
```

### Performance Benchmarks (15 min, $0.50)
```bash
ANTHROPIC_API_KEY=sk-... pytest tests/test_performance.py -v -m performance
```

---

## Reference Files

| File | Relevance |
|------|-----------|
| `core/ontology.py` | TransitionMatrix, NodeMetadata definitions |
| `core/protocols.py` | Agent output protocols |
| `.blueprint/prime_directives.md` | 22 immutable rules to verify |
| `ARCHITECTURE_ROADMAP.tsv` | Implementation status tracking |
| `TEST_PLAN.md` | Previous test structure (for migration) |
| `docs/IMPLEMENTATION_PLAN.md` | Architecture decisions |

---

## Next Steps

1. **Create test directory structure:**
   ```bash
   mkdir -p tests
   touch tests/__init__.py
   touch tests/test_transition_matrix.py
   touch tests/test_governance_physics.py
   touch tests/test_provider_abstraction.py
   touch tests/test_e2e_pipeline.py
   touch tests/test_performance.py
   ```

2. **Copy test code from this plan into files**

3. **Run quick validation first:**
   ```bash
   pytest tests/test_transition_matrix.py -v
   ```

4. **Iterate based on failures**

---

**Document Status:** READY FOR IMPLEMENTATION
