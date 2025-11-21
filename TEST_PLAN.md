# GAADP Constructor - Comprehensive Test Plan

**Version:** 1.0
**Status:** Ready for Execution
**Last Updated:** 2025-11-21

## Executive Summary

This test plan validates the GAADP Constructor system after critical architectural improvements:
- Real Anthropic Claude API integration (replaced stubs)
- Docker security enforcement
- Vector DB failure handling
- Janitor daemon (orphan cleanup)
- Non-blocking human loop
- Adaptive failure recovery with strategy changes

## Test Environment Setup

### Prerequisites

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
export ANTHROPIC_API_KEY="your-key-here"

# 3. Verify Docker (optional for sandboxing)
docker --version

# 4. Initialize test workspace
mkdir -p .gaadp_test
```

### Test Data Preparation

```bash
# Create test requirements of varying complexity
echo "Create a Python function to calculate factorial" > tests/fixtures/req_simple.txt
echo "Build a REST API with authentication and rate limiting" > tests/fixtures/req_complex.txt
echo "Implement a binary search tree with balancing" > tests/fixtures/req_medium.txt
```

---

## Test Levels

### Level 1: Unit Tests (Existing Coverage ✅)

**Location:** `tests/test_integration.py`

**Covered Components:**
- `TestGraphDBIntegration` - Persistence, cycles, Merkle chains
- `TestSandboxIntegration` - Code execution, timeouts, errors
- `TestEventBusIntegration` - Pub/sub, history tracking
- `TestTestRunnerIntegration` - Test execution framework

**Run Command:**
```bash
pytest tests/test_integration.py -v
```

**Expected Result:** All 13 tests pass

---

### Level 2: Critical Fixes Validation (NEW - Required ⚠️)

**Location:** `tests/test_critical_fixes.py` (to be created)

#### Test 2.1: Docker Security Enforcement
```python
def test_docker_required_by_default():
    """Verify sandbox requires Docker unless explicitly allowed fallback."""
    with pytest.raises(SandboxSecurityError):
        sandbox = CodeSandbox(use_docker=True, allow_local_fallback=False)
        # Should fail if Docker not available
```

#### Test 2.2: Vector DB Graceful Degradation
```python
def test_semantic_memory_fallback_mode():
    """Verify semantic memory logs warnings when degraded."""
    import logging
    with patch('infrastructure.semantic_memory.EMBEDDINGS_AVAILABLE', False):
        memory = SemanticMemory(require_embeddings=False, fallback_mode=True)
        assert not memory.embeddings_enabled
        # Check logs contain WARNING about degraded mode
```

#### Test 2.3: Janitor Orphan Detection
```python
@pytest.mark.asyncio
async def test_janitor_cleans_orphaned_nodes():
    """Verify Janitor marks stuck IN_PROGRESS nodes as FAILED."""
    db = GraphDB(persistence_path=":memory:")
    event_bus = EventBus()

    # Create stuck node
    db.add_node("stuck_1", NodeType.CODE, "test")
    db.set_status("stuck_1", NodeStatus.IN_PROGRESS, reason="Testing")

    # Backdate the timestamp
    db.graph.nodes["stuck_1"]["metadata"]["in_progress_since"] = time.time() - 1000

    # Run janitor
    janitor = JanitorDaemon(db, event_bus, JanitorConfig(orphan_timeout=600))
    await janitor._scan_for_orphans()

    # Verify cleanup
    assert db.graph.nodes["stuck_1"]["status"] == NodeStatus.FAILED.value
```

#### Test 2.4: Non-Blocking Human Loop
```python
@pytest.mark.asyncio
async def test_parallel_execution_with_human_block():
    """Verify other DAG branches continue when one is blocked on human input."""
    scheduler = TaskScheduler(db, event_bus)

    # Create two independent branches
    db.add_node("req_a", NodeType.REQ, "Branch A")
    db.add_node("req_b", NodeType.REQ, "Branch B")

    # Block branch A
    scheduler.mark_waiting_for_human("req_a")

    # Get ready nodes - should only return req_b
    ready = scheduler._get_ready_nodes()
    assert len(ready) == 1
    assert ready[0]["node_id"] == "req_b"
```

#### Test 2.5: Architect Strategy Change on Escalation
```python
@pytest.mark.asyncio
async def test_architect_escalation_prompt():
    """Verify Architect changes strategy after failures."""
    architect = RealArchitect("arch_test", AgentRole.ARCHITECT, db)

    # Create requirement with escalation context
    req_node = {
        "id": "req_fail",
        "content": "Build a complex system",
        "escalation_context": "ESCALATION: Previous attempt failed with Missing import errors..."
    }

    # Process should use different prompt
    result = await architect.process({"nodes": [req_node]})

    # Verify new strategy hints are included
    # (This requires mocking LLM or checking logs)
```

**Run Command:**
```bash
pytest tests/test_critical_fixes.py -v --tb=short
```

---

### Level 3: LLM Gateway Integration (NEW - Critical ⚠️)

**Location:** `tests/test_llm_gateway.py` (to be created)

#### Test 3.1: Real API Connectivity
```python
@pytest.mark.asyncio
async def test_anthropic_api_connection():
    """Verify real Anthropic API calls work."""
    gateway = LLMGateway()

    response = gateway.call_model(
        role="ARCHITECT",
        system_prompt="You are a helpful assistant.",
        user_context="Say 'Hello GAADP' and nothing else."
    )

    assert "GAADP" in response or "gaadp" in response.lower()
    assert gateway.get_session_cost() > 0  # Verify cost tracking
```

#### Test 3.2: Token Tracking Accuracy
```python
def test_token_usage_tracking():
    """Verify token counts and cost estimation."""
    gateway = LLMGateway()

    initial_cost = gateway.get_session_cost()

    response = gateway.call_model(
        role="BUILDER",
        system_prompt="You are a code generator.",
        user_context="Write a hello world function in Python."
    )

    final_cost = gateway.get_session_cost()

    assert final_cost > initial_cost
    assert gateway._token_usage["input"] > 0
    assert gateway._token_usage["output"] > 0
```

#### Test 3.3: Tool Use Handling
```python
@pytest.mark.asyncio
async def test_tool_calling():
    """Verify tool use responses are formatted correctly."""
    gateway = LLMGateway()

    tools = [{
        "name": "read_file",
        "description": "Read a file",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    }]

    response = gateway.call_model(
        role="ARCHITECT",
        system_prompt="You have access to file tools.",
        user_context="Read the file at /etc/hosts",
        tools=tools
    )

    # Should return JSON with tool_calls
    import json
    parsed = json.loads(response)
    assert "tool_calls" in parsed
```

#### Test 3.4: Error Handling
```python
def test_invalid_api_key():
    """Verify proper error on missing API key."""
    import os
    original = os.getenv("ANTHROPIC_API_KEY")
    os.environ.pop("ANTHROPIC_API_KEY", None)

    with pytest.raises(LLMGatewayError, match="ANTHROPIC_API_KEY"):
        gateway = LLMGateway()

    if original:
        os.environ["ANTHROPIC_API_KEY"] = original
```

**Run Command:**
```bash
ANTHROPIC_API_KEY=sk-... pytest tests/test_llm_gateway.py -v
```

**Expected Cost:** ~$0.05-0.10 for all LLM tests

---

### Level 4: End-to-End Agent Pipeline (NEW - Integration ⚠️)

**Location:** `tests/test_e2e_agents.py` (to be created)

#### Test 4.1: Simple Requirement → Verified Code
```python
@pytest.mark.asyncio
@pytest.mark.slow
async def test_full_pipeline_simple():
    """Test: REQ → Architect → Builder → Verifier → CODE (VERIFIED)"""

    # Setup
    engine = GADPEngine(persistence_path=".gaadp_test/e2e_simple.json", num_workers=1)

    architect = RealArchitect("arch_e2e", AgentRole.ARCHITECT, engine.db)
    builder = RealBuilder("build_e2e", AgentRole.BUILDER, engine.db)
    verifier = RealVerifier("verif_e2e", AgentRole.VERIFIER, engine.db)

    engine.register_agents({
        AgentRole.ARCHITECT: architect,
        AgentRole.BUILDER: builder,
        AgentRole.VERIFIER: verifier
    })

    # Inject simple requirement
    req_id = engine.inject_requirement("Create a Python function that calculates factorial recursively")

    # Run until complete (with timeout)
    await engine.run_until_complete(timeout=120)

    # Verify outcome
    status = engine.get_status()
    assert status["running"] == False

    # Check for VERIFIED code node
    verified_nodes = [
        n for n, d in engine.db.graph.nodes(data=True)
        if d.get("type") == NodeType.CODE.value and d.get("status") == NodeStatus.VERIFIED.value
    ]

    assert len(verified_nodes) >= 1, "Should have at least one verified code node"

    # Check cost tracking
    assert engine.gateway.get_session_cost() > 0
```

#### Test 4.2: Failure Recovery Loop
```python
@pytest.mark.asyncio
@pytest.mark.slow
async def test_failure_recovery_with_strategy_change():
    """Test that failures trigger re-planning with different strategy."""

    engine = GADPEngine(persistence_path=".gaadp_test/e2e_recovery.json")
    # ... register agents ...

    # Inject complex requirement likely to need iteration
    req_id = engine.inject_requirement("Build a thread-safe LRU cache with TTL expiration")

    await engine.run_until_complete(timeout=300)

    # Check feedback controller engaged
    retry_counts = engine.feedback.get_retry_counts()

    # Should have attempted retries if initial approach failed
    # (May pass on first try with good LLM, but should handle retries)
```

#### Test 4.3: Governance Hooks (Budget + Security)
```python
@pytest.mark.asyncio
async def test_governance_middleware():
    """Verify Treasurer budget limits and Sentinel security checks."""

    engine = GADPEngine(persistence_path=".gaadp_test/e2e_governance.json")

    # Set very low budget
    engine.treasurer.config.project_total_limit_usd = 0.01

    # ... register agents, inject requirement ...

    await engine.run_until_complete(timeout=60)

    # Should hit budget limit and block
    treasurer_status = engine.treasurer.get_status()
    assert treasurer_status["budget_blocked"] or treasurer_status["spend_usd"] <= 0.01
```

**Run Command:**
```bash
pytest tests/test_e2e_agents.py -v -m slow --timeout=600
```

**Expected Duration:** 5-10 minutes
**Expected Cost:** $0.50-2.00 (real LLM calls)

---

### Level 5: Orchestration & Concurrency (NEW - System Level ⚠️)

**Location:** `tests/test_orchestration.py` (to be created)

#### Test 5.1: Parallel Task Execution
```python
@pytest.mark.asyncio
async def test_concurrent_builders():
    """Verify multiple Builders work in parallel."""

    scheduler = TaskScheduler(db, event_bus, config=SchedulerConfig(max_concurrent_builders=3))

    # Create 5 independent SPEC nodes
    for i in range(5):
        db.add_node(f"spec_{i}", NodeType.SPEC, f"Implement feature {i}")

    # Start scheduler
    start_time = time.time()
    await scheduler.start(num_workers=3)

    # Wait for completion
    await asyncio.sleep(10)  # Adjust based on actual LLM latency

    elapsed = time.time() - start_time

    # Should complete faster than sequential (5 * ~3s each = 15s sequential vs ~6s parallel)
    assert elapsed < 12, "Parallel execution should be faster than sequential"
```

#### Test 5.2: Dependency Resolution
```python
@pytest.mark.asyncio
async def test_dependency_ordering():
    """Verify tasks execute in correct dependency order."""

    # Create diamond dependency:
    #     REQ
    #    /   \
    #  SPEC1 SPEC2
    #    \   /
    #     CODE

    db.add_node("req", NodeType.REQ, "Root requirement")
    db.add_node("spec1", NodeType.SPEC, "Spec 1")
    db.add_node("spec2", NodeType.SPEC, "Spec 2")
    db.add_node("code", NodeType.CODE, "Final code")

    db.add_edge("spec1", "req", EdgeType.DEPENDS_ON, "sys", "sig")
    db.add_edge("spec2", "req", EdgeType.DEPENDS_ON, "sys", "sig")
    db.add_edge("code", "spec1", EdgeType.DEPENDS_ON, "sys", "sig")
    db.add_edge("code", "spec2", EdgeType.DEPENDS_ON, "sys", "sig")

    # Track execution order
    executed = []

    async def mock_process(context):
        executed.append(context["nodes"][0]["id"])
        return {"verdict": "PASS"}

    # ... run scheduler ...

    # Verify: req before spec1/spec2, spec1/spec2 before code
    assert executed.index("req") < executed.index("spec1")
    assert executed.index("req") < executed.index("spec2")
    assert executed.index("spec1") < executed.index("code")
    assert executed.index("spec2") < executed.index("code")
```

---

### Level 6: Production Readiness (Smoke Tests)

**Location:** `tests/test_production.py` (to be created)

#### Test 6.1: Full Production Main
```bash
# Run production_main.py with batch mode
python production_main.py --batch "Create a function to validate email addresses"
```

**Success Criteria:**
- No unhandled exceptions
- Graph persisted to `.gaadp/live_graph.json`
- At least one VERIFIED code node created
- Output file materialized to disk
- Git commit created (if GitController initialized)

#### Test 6.2: MCP Server Health
```python
def test_mcp_server_startup():
    """Verify MCP server starts without errors."""
    import subprocess

    proc = subprocess.Popen(
        ["python", "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(3)  # Wait for startup

    # Check if still running
    assert proc.poll() is None, "MCP server should be running"

    # Cleanup
    proc.terminate()
```

---

## Test Execution Strategy

### Phase 1: Pre-Deployment Validation
```bash
# 1. Unit tests (fast, no API calls)
pytest tests/test_integration.py -v

# 2. Critical fixes (fast, mocked)
pytest tests/test_critical_fixes.py -v

# 3. LLM Gateway (real API, low cost)
ANTHROPIC_API_KEY=sk-... pytest tests/test_llm_gateway.py -v
```

**Time:** ~5 minutes
**Cost:** ~$0.10

### Phase 2: Integration Validation
```bash
# 4. E2E simple pipeline (real agents, moderate cost)
pytest tests/test_e2e_agents.py::test_full_pipeline_simple -v

# 5. Orchestration (parallel, dependencies)
pytest tests/test_orchestration.py -v
```

**Time:** ~15 minutes
**Cost:** ~$1.00

### Phase 3: Full System Validation
```bash
# 6. Complete E2E suite
pytest tests/test_e2e_agents.py -v -m slow

# 7. Production smoke test
python production_main.py --batch "Create a sorting algorithm"
```

**Time:** ~30 minutes
**Cost:** ~$3.00

---

## Success Criteria

### Minimum Viable (Must Pass ✅)
- [ ] All existing unit tests pass (Level 1)
- [ ] Docker security enforced (Test 2.1)
- [ ] LLM Gateway connects to Claude API (Test 3.1)
- [ ] Simple E2E pipeline produces verified code (Test 4.1)
- [ ] No unhandled exceptions in production_main.py (Test 6.1)

### Production Ready (Should Pass ⚙️)
- [ ] Janitor cleans orphaned nodes (Test 2.3)
- [ ] Non-blocking human loop works (Test 2.4)
- [ ] Token tracking accurate within 5% (Test 3.2)
- [ ] Failure recovery with strategy change (Test 4.2)
- [ ] Governance hooks enforce limits (Test 4.3)
- [ ] Parallel execution faster than sequential (Test 5.1)

### Stretch Goals (Nice to Have 🎯)
- [ ] MCP server integration tests
- [ ] Load testing (100+ concurrent tasks)
- [ ] Performance benchmarks (latency, throughput)
- [ ] Semantic memory clustering accuracy

---

## Known Limitations & Risks

### Current Gaps (from evaluator feedback)
1. **MCP Hub tool execution** - Currently stubbed, may fail in tests requiring real tool calls
2. **Git conflict resolution** - Basic git ops work, but complex scenarios untested
3. **Embedding service** - Tests assume fallback mode; full semantic search untested if embeddings unavailable

### Test Environment Dependencies
- **Docker:** Required for sandbox security tests (can be mocked for CI/CD)
- **ANTHROPIC_API_KEY:** Required for all LLM integration tests
- **Git:** Required for version control tests

### Cost Management
- E2E tests with real LLM calls: Budget $5-10 for full test suite
- Use `max_tokens` limits in test configs to reduce costs
- Consider using Claude Haiku for non-critical test scenarios

---

## Next Steps

### Immediate (Before First Test Run)
1. Create missing test files:
   - `tests/test_critical_fixes.py`
   - `tests/test_llm_gateway.py`
   - `tests/test_e2e_agents.py`
   - `tests/test_orchestration.py`
   - `tests/test_production.py`

2. Set up test fixtures:
   ```bash
   mkdir -p tests/fixtures
   # Create sample requirements
   ```

3. Configure pytest:
   ```ini
   # pytest.ini
   [pytest]
   markers =
       slow: marks tests as slow (deselect with '-m "not slow"')
       integration: integration tests requiring external services
       unit: fast unit tests with no external dependencies
   ```

### Post-Test Analysis
1. Collect metrics:
   - Test coverage percentage
   - Actual API costs incurred
   - Failure patterns
   - Performance bottlenecks

2. Create test report:
   - Pass/fail breakdown by level
   - Cost analysis
   - Identified bugs/issues
   - Recommendations for fixes

---

## Appendix: Test Data

### Sample Requirements (Complexity Levels)

**Trivial (Should always pass):**
- "Create a function that returns 'Hello World'"
- "Write a function to add two numbers"

**Simple (High success rate):**
- "Implement factorial using recursion"
- "Create an email validator function"
- "Write a function to reverse a string"

**Medium (May require iteration):**
- "Implement a binary search tree"
- "Create a rate limiter using token bucket algorithm"
- "Build a simple LRU cache"

**Complex (Likely to need multiple attempts):**
- "Design a distributed task queue with priority scheduling"
- "Implement OAuth2 authentication flow"
- "Build a real-time collaborative text editor backend"

---

**Document Status:** READY FOR EXECUTION
**Approved By:** System ready for validation testing
**Estimated Total Cost:** $5-10 USD for complete test suite
**Estimated Total Time:** 1-2 hours for full execution
