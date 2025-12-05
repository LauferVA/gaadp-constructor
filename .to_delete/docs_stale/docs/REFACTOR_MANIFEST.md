# REFACTOR MANIFEST: Specific Intended Changes

This document provides a line-by-line accounting of what changes, what stays, and why.

---

## 1. FILES TO DELETE (After Migration)

### 1.1 `production_main.py` (ENTIRE FILE)
**Reason:** This file is 750 lines of imperative orchestration that should be 50 lines of graph traversal.

**What it currently does:**
- Lines 1-150: Logging setup (KEEP, move to `infrastructure/logging.py`)
- Lines 150-210: Imports and requirement loading (KEEP, move to `main.py`)
- Lines 210-330: Socratic phase (REPLACE with CLARIFICATION node processing)
- Lines 330-450: Architect phase + edge creation (REPLACE with graph traversal triggering Architect)
- Lines 450-650: Build/verify loop with escalation (REPLACE with graph traversal)
- Lines 650-750: Post-build summary, git commit (KEEP, simplify)

**Replacement:** `core/runtime.py` (~100 lines) + `main.py` (~50 lines)

---

### 1.2 `orchestration/gaadp_engine.py`
**Reason:** Wavefront execution is just topological traversal of the graph.

**What to preserve:**
- The concept of "waves" for parallel execution
- Move to `core/graph.py` as `Graph.get_execution_waves()`

---

### 1.3 `orchestration/dependency_resolver.py`
**Reason:** Dependency resolution is a graph query, not a separate module.

**What to preserve:**
- `topological_sort()` logic → becomes `Graph.topological_order()`
- `get_build_waves()` logic → becomes `Graph.get_execution_waves()`
- Cycle detection → becomes `Graph.validate_dag()`

---

### 1.4 `orchestration/escalation_controller.py`
**Reason:** Escalation is not a controller - it's a node type with automatic creation rules.

**What to preserve:**
- `EscalationContext` data → becomes ESCALATION node content
- `build_architect_escalation_prompt()` → becomes part of Architect agent's context building
- Failure pattern detection → becomes graph query on FEEDBACK edges

**Replacement pattern:**
```python
# Instead of EscalationController.record_failure():
if node.metadata.attempts >= node.metadata.max_attempts:
    graph.add_node(
        type=ESCALATION,
        content=graph.build_escalation_context(node),
        metadata={"original_spec": node.id, "failure_count": node.metadata.attempts}
    )
    graph.add_edge(node.id, escalation.id, EdgeType.BLOCKS)
```

---

### 1.5 `requirements/socratic_agent.py` (PARTIAL DELETE)
**Reason:** 400 lines that should be ~50 lines of agent config + CLARIFICATION node handling.

**What to preserve:**
- `_extract_spec_from_file()` → useful utility, move to `infrastructure/code_analysis.py`
- `AmbiguityAnalysis` concept → becomes CLARIFICATION node content structure

**What to delete:**
- `InteractiveSocraticPhase` class → replaced by Socratic agent processing CLARIFICATION nodes
- `DevelopmentSocraticPhase` class → replaced by pre-processing that creates enriched REQ nodes
- `SocraticConfig` → becomes node metadata

---

## 2. FILES TO SIGNIFICANTLY SIMPLIFY

### 2.1 `agents/concrete_agents.py` (650 → ~100 lines)

**Current state:** Each agent is 100-150 lines of:
- ReAct loop management
- Tool call parsing
- Protocol validation
- Prompt building

**Target state:** Each agent is:
```python
class ArchitectAgent(BaseAgent):
    config = "agents/architect.yaml"
```

All the logic moves to:
- `BaseAgent` handles ReAct loop, tool calls, validation
- YAML config defines prompts and output protocol
- Graph context is passed in, not manually built

---

### 2.2 `agents/base_agent.py` (300 → ~150 lines)

**Keep:**
- `_hydrate_prompt()` - template filling
- `sign_content()` - integrity
- `execute_tool_calls()` - MCP integration
- `_parse_nested_json()` - LLM output parsing

**Remove:**
- Manual context building (graph provides this)
- Role-specific logic (moves to config)

---

### 2.3 `core/protocols.py` (350 → ~200 lines)

**Keep:**
- All Pydantic models (ArchitectOutput, BuilderOutput, etc.)
- `protocol_to_tool_schema()` utility
- `get_agent_tools()` utility

**Simplify:**
- Remove redundant docstrings
- Consolidate similar patterns

---

## 3. FILES TO ENHANCE

### 3.1 `core/ontology.py` (100 → ~200 lines)

**Add:**
```python
# Node property schemas
@dataclass
class NodeProperties:
    """Universal properties for all nodes."""
    id: str
    type: NodeType
    content: str
    status: NodeStatus
    created_by: str
    created_at: datetime
    signature: str
    metadata: Dict[str, Any]

    # Governance
    attempts: int = 0
    max_attempts: int = 3
    cost_usd: float = 0.0

    # Computed
    @property
    def is_terminal(self) -> bool:
        return self.status in (NodeStatus.VERIFIED, NodeStatus.FAILED)

# State transition rules
VALID_TRANSITIONS = {
    NodeStatus.PENDING: {NodeStatus.PROCESSING, NodeStatus.BLOCKED},
    NodeStatus.PROCESSING: {NodeStatus.VERIFIED, NodeStatus.FAILED, NodeStatus.BLOCKED},
    NodeStatus.BLOCKED: {NodeStatus.PENDING},
    NodeStatus.VERIFIED: set(),  # Terminal
    NodeStatus.FAILED: set(),    # Terminal
}

# Which agent processes which trigger condition
PROCESSING_RULES = {
    # (node_type, status, condition) → agent_type
    (NodeType.REQ, NodeStatus.PENDING, "no_children"): AgentRole.ARCHITECT,
    (NodeType.SPEC, NodeStatus.PENDING, "deps_met"): AgentRole.BUILDER,
    (NodeType.CODE, NodeStatus.PENDING, None): AgentRole.VERIFIER,
    (NodeType.CLARIFICATION, NodeStatus.PENDING, None): AgentRole.SOCRATES,
    (NodeType.ESCALATION, NodeStatus.PENDING, None): AgentRole.ARCHITECT,
}
```

---

### 3.2 `infrastructure/graph_db.py` (200 → ~350 lines)

**Add these query methods:**

```python
class Graph:
    # === Status queries ===
    def get_by_status(self, status: NodeStatus) -> List[Node]: ...
    def get_pending(self) -> List[Node]: ...
    def get_blocked(self) -> List[Node]: ...

    # === Dependency queries ===
    def dependencies_met(self, node_id: str) -> bool: ...
    def get_blocking_nodes(self, node_id: str) -> List[Node]: ...
    def is_blocked(self, node_id: str) -> bool: ...

    # === Traversal ===
    def topological_order(self, nodes: List[Node] = None) -> List[Node]: ...
    def get_execution_waves(self) -> List[List[Node]]: ...
    def validate_dag(self) -> Tuple[bool, Optional[List[str]]]: ...

    # === Context building ===
    def get_context(self, node_id: str, radius: int = 2) -> Context: ...
    def trace_to_requirement(self, node_id: str) -> Node: ...
    def get_feedback_for(self, node_id: str) -> List[Edge]: ...

    # === Mutations ===
    def apply_result(self, result: AgentResult) -> None:
        """Apply an agent's output to the graph."""
        for node_spec in result.new_nodes:
            self.add_node(...)
        for edge_spec in result.new_edges:
            self.add_edge(...)
        for node_id, status in result.status_updates.items():
            self.set_status(node_id, status)

    # === Governance ===
    def increment_attempts(self, node_id: str) -> int: ...
    def should_escalate(self, node_id: str) -> bool: ...
    def total_cost(self) -> float: ...
```

---

## 4. NEW FILES TO CREATE

### 4.1 `core/runtime.py` (~100 lines)

```python
"""
The entire GAADP runtime.

This replaces 750 lines of production_main.py with graph traversal.
"""
import asyncio
from typing import Optional
from core.graph import Graph
from core.ontology import NodeStatus, PROCESSING_RULES
from agents import get_agent

async def run(graph: Graph, max_iterations: int = 1000) -> None:
    """Execute the graph until completion or max iterations."""
    for _ in range(max_iterations):
        # Find processable nodes
        ready = [
            node for node in graph.get_pending()
            if not graph.is_blocked(node.id)
            and graph.dependencies_met(node.id)
        ]

        if not ready:
            if graph.get_by_status(NodeStatus.PROCESSING):
                await asyncio.sleep(0.1)
                continue
            break  # Done

        # Process in waves
        for wave in graph.get_execution_waves(ready):
            await asyncio.gather(*[
                process_node(graph, node) for node in wave
            ])

async def process_node(graph: Graph, node: Node) -> None:
    """Process a single node."""
    graph.set_status(node.id, NodeStatus.PROCESSING)
    graph.increment_attempts(node.id)

    # Determine which agent handles this
    agent = get_agent_for_node(graph, node)
    if not agent:
        return  # No processor for this node type/state

    # Build context from graph
    context = graph.get_context(node.id)

    # Process
    try:
        result = await agent.process(node, context)
        graph.apply_result(result)
    except Exception as e:
        graph.set_status(node.id, NodeStatus.FAILED)
        graph.add_node(type=NodeType.ESCALATION, ...)

    # Check for escalation
    if graph.should_escalate(node.id):
        create_escalation_node(graph, node)
```

---

### 4.2 `agents/config/architect.yaml`

```yaml
name: architect
role: ARCHITECT
triggers:
  - node_type: REQ
    status: PENDING
    condition: no_children
  - node_type: ESCALATION
    status: PENDING

system_prompt: |
  You are the Architect agent. You decompose requirements into atomic specifications.

  ## Your Inputs
  - Requirement: {context.requirement.content}
  - Existing work: {context.children}
  - Previous feedback: {context.feedback}

  ## Your Outputs
  Use the submit_architecture tool to output:
  - new_nodes: SPEC nodes (atomic, single-builder implementable)
  - new_edges: DEPENDS_ON relationships between specs

  ## Rules
  1. Each SPEC must be atomic
  2. Emit DEPENDS_ON edges for dependencies
  3. If ambiguous, emit CLARIFICATION node instead
  4. On ESCALATION, analyze failures and try different approach

output_protocol: ArchitectOutput
output_tool: submit_architecture

tools:
  - read_file
  - list_directory
  - search_web
```

---

### 4.3 `main.py` (~50 lines)

```python
#!/usr/bin/env python3
"""GAADP: Graph-Augmented Autonomous Development Platform"""
import asyncio
import argparse
from core.graph import Graph
from core.runtime import run
from core.ontology import NodeType

def main():
    parser = argparse.ArgumentParser(description="GAADP")
    parser.add_argument("--requirement", "-r", help="Requirement text or file")
    parser.add_argument("--graph", "-g", default=".gaadp/graph.json", help="Graph file")
    parser.add_argument("--interactive", "-i", action="store_true")
    args = parser.parse_args()

    # Load or create graph
    graph = Graph.load(args.graph) if Path(args.graph).exists() else Graph()

    # Inject requirement if provided
    if args.requirement:
        content = Path(args.requirement).read_text() if Path(args.requirement).exists() else args.requirement
        graph.add_node(type=NodeType.REQ, content=content)

    # Run
    asyncio.run(run(graph))

    # Save
    graph.save(args.graph)

    # Summary
    print(f"Nodes: {len(graph.nodes)}, Verified: {len(graph.get_by_status(NodeStatus.VERIFIED))}")

if __name__ == "__main__":
    main()
```

---

## 5. MIGRATION SEQUENCE

### Step 1: Enhance without breaking
- Add new methods to `graph_db.py`
- Add new properties to `ontology.py`
- Keep existing code working

### Step 2: Create runtime.py
- Implement core loop
- Test alongside existing `production_main.py`
- Verify same behavior

### Step 3: Convert agents to config-driven
- Create YAML configs
- Simplify `concrete_agents.py` to use them
- Keep `base_agent.py` logic

### Step 4: Route through runtime
- Replace `production_main.py` calls with `runtime.run()`
- Keep `production_main.py` as thin wrapper during transition

### Step 5: Delete sprawl
- Remove `orchestration/` modules
- Remove `production_main.py`
- Remove redundant code from agents

### Step 6: Validate
- Run all benchmarks
- Verify reconstruction tests pass
- Measure code reduction

---

## 6. SUCCESS METRICS

| Metric | Current | Target |
|--------|---------|--------|
| Total lines (core + agents + runtime) | ~5000 | ~800 |
| Files in orchestration/ | 4 | 0 |
| Lines in production_main.py | 750 | 0 (deleted) |
| Agent definition lines | 650 | 100 + YAML |
| Time to explain system | 30 min | 5 min |
| Context needed for LLM to work with | 10K tokens | 2K tokens |

---

## 7. RISKS AND MITIGATIONS

### Risk: Loss of functionality during migration
**Mitigation:** Keep old code running in parallel until new path proven

### Risk: YAML configs become complex
**Mitigation:** Keep configs minimal, complex logic stays in Python base classes

### Risk: Graph queries become performance bottleneck
**Mitigation:** Add caching, but graphs are small (<1000 nodes typically)

### Risk: Debugging harder without imperative flow
**Mitigation:** Rich logging in runtime.py, graph state is inspectable

---

## 8. OPEN QUESTIONS

1. **Where does MCP tool registration live?** Currently in agents. Move to config?

2. **How do we handle streaming output?** Orthogonal to architecture, but need hooks.

3. **What about the Socratic "development mode"?** Becomes pre-processing that enriches REQ content before adding to graph.

4. **Git integration?** Keep as post-processing hook, triggered by VERIFIED status.

5. **Cost limits?** Graph property `max_cost_usd`, checked in runtime before each agent call.
