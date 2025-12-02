# GRAPH-FIRST ARCHITECTURE SPECIFICATION

## Executive Summary

This document specifies a refactored GAADP architecture where **everything emerges from the graph**. The goal is a system that can be described in ~500 lines of code, not 5000, while maintaining full capability.

---

## Core Principle

> **The graph is the program. Execution is traversal. Agents are node processors.**

All state, all workflow, all governance flows from nodes, edges, and their properties. There is no separate "orchestration layer" - orchestration IS graph traversal.

---

## Part 1: The Ontology (Single Source of Truth)

### 1.1 Node Types

| Type | Purpose | Processor | Terminal? |
|------|---------|-----------|-----------|
| `REQ` | User requirement | Human/Socratic | No |
| `CLARIFICATION` | Ambiguity needing resolution | Socratic | Blocking |
| `SPEC` | Atomic specification | Architect | No |
| `PLAN` | Decomposition strategy | Architect | No |
| `CODE` | Implementation artifact | Builder | No |
| `TEST` | Verification result | Verifier | Yes |
| `DOC` | Documentation artifact | Builder | Yes |
| `ESCALATION` | Failure requiring intervention | Architect/Human | Blocking |

### 1.2 Node Properties (Universal)

Every node has:

```yaml
id: string           # Unique identifier
type: NodeType       # From enum above
content: string      # The actual artifact
status: Status       # PENDING | PROCESSING | BLOCKED | VERIFIED | FAILED
created_by: AgentID  # Who created this node
created_at: ISO8601  # When
signature: Hash      # Content hash for integrity
metadata: Dict       # Type-specific additional data
```

### 1.3 Status State Machine

```
PENDING ──────────────────────────────────────┐
    │                                         │
    ▼                                         │
PROCESSING ───► BLOCKED ───► PENDING ─────────┤
    │               │                         │
    │               ▼                         │
    │          (wait for blocker resolution)  │
    │                                         │
    ├───────────► VERIFIED ───────────────────┤
    │                                         │
    └───────────► FAILED ─────────────────────┘
                     │
                     ▼
              (creates ESCALATION node)
```

### 1.4 Edge Types

| Type | Semantics | From → To | Created By |
|------|-----------|-----------|------------|
| `TRACES_TO` | Provenance | Any → REQ | Any |
| `DEPENDS_ON` | Must complete first | SPEC → SPEC | Architect |
| `IMPLEMENTS` | Realizes | CODE → SPEC | Builder |
| `VERIFIES` | Validates | TEST → CODE | Verifier |
| `DEFINES` | Decomposes into | PLAN → SPEC | Architect |
| `BLOCKS` | Cannot proceed until | Any → CLARIFICATION/ESCALATION | System |
| `FEEDBACK` | Critique for retry | CODE(failed) → SPEC | Verifier |
| `RESOLVED_BY` | Answer to question | CLARIFICATION → REQ | Human/Socratic |

### 1.5 Edge Properties

```yaml
source: NodeID
target: NodeID
type: EdgeType
created_by: AgentID
created_at: ISO8601
signature: Hash
weight: float        # Optional: for prioritization
metadata: Dict       # Type-specific
```

---

## Part 2: The Runtime (Graph Traversal)

### 2.1 The Core Loop

```python
async def run(graph: Graph) -> None:
    """The entire runtime in ~20 lines."""
    while True:
        # 1. Find processable nodes
        ready = graph.query(
            status=PENDING,
            not_blocked=True,
            dependencies_met=True
        )

        if not ready:
            if graph.has_status(PROCESSING):
                await asyncio.sleep(0.1)  # Wait for in-flight
                continue
            else:
                break  # Nothing left to do

        # 2. Process in parallel waves
        waves = topological_waves(ready)
        for wave in waves:
            await asyncio.gather(*[
                process_node(graph, node) for node in wave
            ])
```

### 2.2 Node Processing

```python
async def process_node(graph: Graph, node: Node) -> None:
    """Process a single node based on its type."""
    graph.set_status(node.id, PROCESSING)

    # Get the appropriate agent
    agent = AGENT_REGISTRY[node.type]

    # Build context from graph neighborhood
    context = graph.get_context(node.id, radius=2)

    # Call the agent
    result = await agent.process(node, context)

    # Apply result to graph
    graph.apply(result)
```

### 2.3 Context Building (Graph Query)

```python
def get_context(self, node_id: str, radius: int = 2) -> Context:
    """Build context by traversing graph neighborhood."""
    return Context(
        node=self.get_node(node_id),
        parents=self.predecessors(node_id),
        children=self.successors(node_id),
        siblings=self.siblings(node_id),
        requirement=self.trace_to_requirement(node_id),
        feedback=self.get_edges(target=node_id, type=FEEDBACK),
        blocking=self.get_blocking_nodes(node_id),
    )
```

### 2.4 Blocking Logic

A node is **blocked** if:
```python
def is_blocked(self, node_id: str) -> bool:
    # Has unresolved CLARIFICATION in ancestry
    if self.has_ancestor(node_id, type=CLARIFICATION, status=PENDING):
        return True
    # Has unresolved ESCALATION for this spec
    if self.has_edge(source=node_id, type=BLOCKS, target_status=PENDING):
        return True
    # Dependencies not met
    if not self.dependencies_verified(node_id):
        return True
    return False
```

---

## Part 3: Agents as Node Processors

### 3.1 Agent Registry

```python
AGENT_REGISTRY = {
    NodeType.REQ: None,           # Entry point, no processor
    NodeType.CLARIFICATION: SocraticAgent,
    NodeType.SPEC: None,          # Created by Architect, processed by Builder
    NodeType.PLAN: None,          # Created by Architect, informational
    NodeType.CODE: BuilderAgent,  # Wait, CODE is output not input...
    NodeType.TEST: None,          # Created by Verifier
    NodeType.ESCALATION: ArchitectAgent,  # Re-planning
}

# Actually, agents process based on WHAT NEEDS TO HAPPEN:
WORKFLOW_TRIGGERS = {
    # When REQ is pending with no children → Architect decomposes
    (REQ, PENDING, no_children): ArchitectAgent,

    # When SPEC is pending with dependencies met → Builder implements
    (SPEC, PENDING, deps_met): BuilderAgent,

    # When CODE is pending → Verifier checks
    (CODE, PENDING): VerifierAgent,

    # When CLARIFICATION is pending → Socratic resolves
    (CLARIFICATION, PENDING): SocraticAgent,

    # When ESCALATION is pending → Architect re-plans
    (ESCALATION, PENDING): ArchitectAgent,
}
```

### 3.2 Agent Interface

```python
class Agent(Protocol):
    """All agents implement this interface."""

    async def process(self, node: Node, context: Context) -> Result:
        """
        Process a node and return graph mutations.

        Returns:
            Result containing:
            - new_nodes: List[NodeSpec]
            - new_edges: List[EdgeSpec]
            - status_updates: Dict[NodeID, Status]
            - artifacts: Dict[str, bytes]  # Files to write
        """
        ...
```

### 3.3 Agent Behavior (Prompt-Driven)

Each agent's behavior is defined by a single prompt template:

```yaml
# agents/architect.yaml
role: ARCHITECT
trigger: (REQ, PENDING, no_children) | (ESCALATION, PENDING)

system_prompt: |
  You decompose requirements into atomic specifications.

  INPUTS (from graph context):
  - requirement: {context.requirement.content}
  - existing_specs: {context.children | filter(type=SPEC)}
  - feedback: {context.feedback}

  OUTPUTS (as structured tool call):
  - new_nodes: SPEC and PLAN nodes
  - new_edges: DEPENDS_ON relationships between specs

  RULES:
  - Each SPEC must be atomic (single Builder can implement)
  - Emit DEPENDS_ON edges when specs have dependencies
  - If requirement is ambiguous, emit CLARIFICATION node instead

output_protocol: ArchitectOutput
```

---

## Part 4: Governance as Graph Properties

### 4.1 Cost Tracking

Cost is a **node property**, not a separate system:

```yaml
# Every node tracks its cost
node.metadata.cost:
  input_tokens: int
  output_tokens: int
  model: string
  usd: float

# Total cost is a graph query
def total_cost(graph):
    return sum(n.metadata.cost.usd for n in graph.nodes())
```

### 4.2 Failure Tracking

Failure count is a **node property**:

```yaml
node.metadata.attempts: int        # How many times processed
node.metadata.max_attempts: int    # Before escalation (default: 3)
```

Escalation is automatic:
```python
if node.metadata.attempts >= node.metadata.max_attempts:
    if node.status == FAILED:
        graph.add_node(type=ESCALATION, content=build_escalation_context(node))
        graph.add_edge(node.id, escalation_id, type=BLOCKS)
```

### 4.3 Audit Trail

The graph IS the audit trail:
- Every node has `created_by`, `created_at`, `signature`
- Every edge has the same
- Status changes are logged as node property updates
- No separate logging system needed - query the graph

---

## Part 5: What Gets Deleted

### Current Modules → Fate

| Current | Lines | Fate |
|---------|-------|------|
| `production_main.py` | ~750 | **DELETE** - becomes 50-line `runtime.py` |
| `orchestration/gaadp_engine.py` | ~300 | **DELETE** - traversal is in runtime |
| `orchestration/dependency_resolver.py` | ~250 | **SIMPLIFY** - becomes graph query |
| `orchestration/escalation_controller.py` | ~300 | **DELETE** - becomes node type + property |
| `requirements/socratic_agent.py` | ~400 | **SIMPLIFY** - becomes agent config |
| `agents/concrete_agents.py` | ~650 | **SIMPLIFY** - becomes prompt configs |
| `agents/base_agent.py` | ~300 | **KEEP** - but simplify |
| `core/ontology.py` | ~100 | **KEEP** - this is the source of truth |
| `core/protocols.py` | ~350 | **KEEP** - structured output |
| `infrastructure/graph_db.py` | ~200 | **ENHANCE** - add query methods |
| `infrastructure/llm_providers.py` | ~400 | **KEEP** |

### Estimated Reduction

- Current: ~5000+ lines across orchestration/agents/runtime
- Target: ~800 lines total
  - `core/ontology.py`: 150 lines (expanded with properties)
  - `core/graph.py`: 200 lines (enhanced GraphDB)
  - `core/runtime.py`: 100 lines (the traversal loop)
  - `agents/`: 200 lines (prompt configs + base class)
  - `infrastructure/llm.py`: 150 lines (provider abstraction)

---

## Part 6: File Structure (Target)

```
gaadp/
├── core/
│   ├── ontology.py      # Node types, edge types, properties, state machine
│   ├── graph.py         # Graph with query methods
│   └── runtime.py       # The traversal loop
├── agents/
│   ├── base.py          # Agent protocol + LLM wrapper
│   ├── architect.yaml   # Architect prompt/config
│   ├── builder.yaml     # Builder prompt/config
│   ├── verifier.yaml    # Verifier prompt/config
│   └── socratic.yaml    # Socratic prompt/config
├── infrastructure/
│   ├── llm.py           # LLM provider abstraction
│   └── sandbox.py       # Code execution (keep as-is)
└── main.py              # CLI entry point (~50 lines)
```

---

## Part 7: Migration Path

### Phase 1: Enhance Ontology
- Add all node properties to `ontology.py`
- Add blocking/dependency logic to graph queries
- Keep existing code working

### Phase 2: Unify Orchestration
- Create `runtime.py` with the core loop
- Route existing code through it
- Deprecate `gaadp_engine.py`, `dependency_resolver.py`

### Phase 3: Simplify Agents
- Move agent logic to YAML configs
- Reduce `concrete_agents.py` to thin wrappers
- Escalation becomes node creation, not separate module

### Phase 4: Delete Sprawl
- Remove `production_main.py` orchestration
- Remove `escalation_controller.py`
- Remove redundant modules

### Phase 5: Validate
- Run reconstruction benchmarks
- Ensure same behavior with 1/10th the code

---

## Part 8: Key Insight

The current codebase has this pattern:
```
User → production_main.py → [orchestration modules] → [agents] → [graph]
```

The refactored architecture inverts this:
```
User → main.py → graph.traverse() → agents respond to graph state
```

**The graph drives execution, not Python control flow.**

This is why LLMs can work with it more economically - they just need to understand:
1. What nodes exist
2. What edges mean
3. What state transitions are valid

They don't need to understand 5000 lines of Python orchestration.

---

## Questions to Resolve

1. **Interactive mode**: How does human input fit? (Probably: CLARIFICATION nodes that block until resolved via stdin/API)

2. **MCP tools**: Keep as agent capability, or model as graph? (Probably: keep as capability, tool results become node metadata)

3. **Persistence**: JSON file? SQLite? (Probably: start with JSON, graph is small)

4. **Parallelism**: How aggressive? (Probably: process entire waves in parallel, respect DEPENDS_ON)

5. **Streaming**: Do we need real-time output? (Probably: yes for UX, but orthogonal to architecture)
