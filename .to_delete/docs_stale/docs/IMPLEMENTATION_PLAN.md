# GRAPH-FIRST REFACTOR: Implementation Plan

**Checkpoint Commit:** `d26aabf` (Apogee of imperative architecture)

---

## Locked Decisions

These are non-negotiable architectural constraints:

| Decision | Rationale |
|----------|-----------|
| **TransitionMatrix is Truth** | Kills spaghetti logic. No `if status == 'verified'` anywhere. |
| **NodeMetadata.cost_limit as Physics** | Governance in Data, not Middleware. Cannot be bypassed. |
| **GenericAgent** | Only way to scale agent types without scaling code complexity. |
| **CLARIFICATION/ESCALATION as Nodes** | Human questions are just nodes. No special controllers. |
| **Strangler Fig Migration** | Freeze old code, build new core, migrate, purge. |

---

## Phase 1: Data Definitions (Schema Only)

**Goal:** Define the "Physics" without any execution logic.

**Constraint:** Do NOT write `if/else` or loops. Only Pydantic models and constants.

### 1.1 Create `core/new_ontology.py`

```python
"""
GAADP New Ontology - The Physics of the System

This module defines the declarative schema that governs all behavior.
The runtime CONSULTS this schema; it does not contain logic itself.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Optional, Literal, Set
from enum import Enum
from datetime import datetime
import uuid


# =============================================================================
# ENUMS (Keep simple, behavior defined elsewhere)
# =============================================================================

class NodeType(str, Enum):
    REQ = "REQ"                    # User requirement
    CLARIFICATION = "CLARIFICATION"  # Ambiguity needing human input
    SPEC = "SPEC"                  # Atomic specification
    PLAN = "PLAN"                  # Decomposition strategy
    CODE = "CODE"                  # Implementation artifact
    TEST = "TEST"                  # Verification result
    DOC = "DOC"                    # Documentation
    ESCALATION = "ESCALATION"      # Failure requiring intervention


class EdgeType(str, Enum):
    TRACES_TO = "TRACES_TO"        # Provenance to requirement
    DEPENDS_ON = "DEPENDS_ON"      # Must complete first
    IMPLEMENTS = "IMPLEMENTS"      # CODE -> SPEC
    VERIFIES = "VERIFIES"          # TEST -> CODE
    DEFINES = "DEFINES"            # PLAN -> SPEC
    BLOCKS = "BLOCKS"              # Any -> CLARIFICATION/ESCALATION
    FEEDBACK = "FEEDBACK"          # Failed CODE -> SPEC
    RESOLVED_BY = "RESOLVED_BY"    # CLARIFICATION -> answer


class NodeStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    BLOCKED = "BLOCKED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


# =============================================================================
# NODE METADATA (Governance as Data)
# =============================================================================

class NodeMetadata(BaseModel):
    """
    Governance properties embedded in every node.

    These are PHYSICS, not policy. A node physically cannot be
    processed if it violates these constraints.
    """
    # Cost governance
    cost_limit: Optional[float] = Field(
        default=None,
        description="Max USD cost allowed for processing this node. None = unlimited."
    )
    cost_actual: float = Field(
        default=0.0,
        description="Actual cost incurred so far."
    )

    # Security
    security_level: int = Field(
        default=0,
        description="0=public, 1=internal, 2=confidential, 3=restricted"
    )

    # Versioning
    version: str = Field(default="1.0")

    # Approval workflow
    approval_status: Literal["pending", "approved", "rejected"] = "pending"
    approved_by: Optional[str] = None

    # Retry tracking
    attempts: int = Field(default=0, description="Processing attempts so far")
    max_attempts: int = Field(default=3, description="Max attempts before escalation")

    # Arbitrary extension
    extra: Dict[str, any] = Field(default_factory=dict)


# =============================================================================
# NODE SPECIFICATION (The Atomic Unit)
# =============================================================================

class NodeSpec(BaseModel):
    """
    The atomic unit of the graph.

    Every artifact in the system is a NodeSpec.
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: NodeType
    content: str
    status: NodeStatus = NodeStatus.PENDING
    metadata: NodeMetadata = Field(default_factory=NodeMetadata)

    # Provenance
    created_by: str = Field(description="Agent or user who created this")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Integrity
    signature: Optional[str] = Field(
        default=None,
        description="Hash of content for integrity verification"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# EDGE SPECIFICATION
# =============================================================================

class EdgeSpec(BaseModel):
    """Edge between two nodes."""
    source_id: str
    target_id: str
    type: EdgeType
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    weight: float = Field(default=1.0, description="For prioritization")
    metadata: Dict[str, any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


# =============================================================================
# TRANSITION RULES (The Physics)
# =============================================================================

class TransitionRule(BaseModel):
    """
    Defines what is required for a status transition.

    This is the "physics" - you cannot violate these rules.
    """
    target_status: NodeStatus
    required_edge_types: List[EdgeType] = Field(
        default_factory=list,
        description="Must have incoming edges of these types"
    )
    required_conditions: List[str] = Field(
        default_factory=list,
        description="Named conditions that must be true"
    )

    class Config:
        use_enum_values = True


# The Transition Matrix: (CurrentStatus, NodeType) -> List[TransitionRule]
# This is the SINGLE SOURCE OF TRUTH for what state changes are allowed.

TRANSITION_MATRIX: Dict[Tuple[str, str], List[TransitionRule]] = {
    # REQ can become PROCESSING (being decomposed by Architect)
    (NodeStatus.PENDING.value, NodeType.REQ.value): [
        TransitionRule(
            target_status=NodeStatus.PROCESSING,
            required_conditions=["cost_under_limit"]
        ),
    ],

    # REQ becomes VERIFIED when it has SPEC children that are all VERIFIED
    (NodeStatus.PROCESSING.value, NodeType.REQ.value): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_conditions=["all_children_verified"]
        ),
        TransitionRule(
            target_status=NodeStatus.BLOCKED,
            required_conditions=["has_clarification"]
        ),
    ],

    # SPEC can be processed by Builder
    (NodeStatus.PENDING.value, NodeType.SPEC.value): [
        TransitionRule(
            target_status=NodeStatus.PROCESSING,
            required_conditions=["cost_under_limit", "dependencies_met"]
        ),
    ],

    # SPEC becomes VERIFIED when its CODE is VERIFIED
    (NodeStatus.PROCESSING.value, NodeType.SPEC.value): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_edge_types=[EdgeType.IMPLEMENTS],
            required_conditions=["implementor_verified"]
        ),
        TransitionRule(
            target_status=NodeStatus.FAILED,
            required_conditions=["max_attempts_exceeded"]
        ),
    ],

    # CODE can be processed by Verifier
    (NodeStatus.PENDING.value, NodeType.CODE.value): [
        TransitionRule(
            target_status=NodeStatus.PROCESSING,
            required_conditions=["cost_under_limit"]
        ),
    ],

    # CODE becomes VERIFIED when it has a passing TEST
    (NodeStatus.PROCESSING.value, NodeType.CODE.value): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_edge_types=[EdgeType.VERIFIES],
            required_conditions=["verifier_passed"]
        ),
        TransitionRule(
            target_status=NodeStatus.FAILED,
            required_conditions=["verifier_failed"]
        ),
    ],

    # CLARIFICATION blocks until resolved
    (NodeStatus.PENDING.value, NodeType.CLARIFICATION.value): [
        TransitionRule(
            target_status=NodeStatus.PROCESSING,
            required_conditions=[]  # Can always process (ask human)
        ),
    ],

    (NodeStatus.PROCESSING.value, NodeType.CLARIFICATION.value): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_edge_types=[EdgeType.RESOLVED_BY],
            required_conditions=["has_answer"]
        ),
    ],

    # ESCALATION triggers Architect re-planning
    (NodeStatus.PENDING.value, NodeType.ESCALATION.value): [
        TransitionRule(
            target_status=NodeStatus.PROCESSING,
            required_conditions=["cost_under_limit"]
        ),
    ],

    (NodeStatus.PROCESSING.value, NodeType.ESCALATION.value): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_conditions=["replan_complete"]
        ),
        TransitionRule(
            target_status=NodeStatus.FAILED,
            required_conditions=["max_escalations_exceeded"]
        ),
    ],
}


# =============================================================================
# AGENT DISPATCH RULES
# =============================================================================

# Maps (NodeType, Condition) -> AgentRole that should process it
# This replaces hardcoded "if node.type == CODE: use Verifier"

AGENT_DISPATCH: Dict[Tuple[str, str], str] = {
    # REQ with no children -> Architect decomposes
    (NodeType.REQ.value, "no_children"): "ARCHITECT",

    # SPEC with dependencies met -> Builder implements
    (NodeType.SPEC.value, "dependencies_met"): "BUILDER",

    # CODE pending verification -> Verifier checks
    (NodeType.CODE.value, "needs_verification"): "VERIFIER",

    # CLARIFICATION pending -> Socratic resolves (or human)
    (NodeType.CLARIFICATION.value, "pending"): "SOCRATES",

    # ESCALATION -> Architect re-plans
    (NodeType.ESCALATION.value, "pending"): "ARCHITECT",
}


# =============================================================================
# CONDITION EVALUATORS (Names only - implementation in runtime)
# =============================================================================

# These are the named conditions referenced in TransitionRules.
# The runtime must implement evaluation functions for each.

KNOWN_CONDITIONS: Set[str] = {
    "cost_under_limit",       # node.metadata.cost_actual < node.metadata.cost_limit
    "dependencies_met",       # All DEPENDS_ON targets are VERIFIED
    "all_children_verified",  # All child nodes are VERIFIED
    "has_clarification",      # Node has CLARIFICATION child
    "implementor_verified",   # CODE implementing this SPEC is VERIFIED
    "verifier_passed",        # TEST verdict is PASS
    "verifier_failed",        # TEST verdict is FAIL
    "max_attempts_exceeded",  # node.metadata.attempts >= max_attempts
    "has_answer",             # CLARIFICATION has been answered
    "replan_complete",        # Architect produced new plan
    "max_escalations_exceeded",
    "no_children",            # Node has no child nodes
    "needs_verification",     # CODE has no TEST
    "pending",                # Generic: status is PENDING
}
```

### 1.2 Create `config/agent_manifest.yaml`

```yaml
# Agent Configuration Manifest
#
# Each agent is defined by:
# - What node types it processes
# - What node types it produces
# - Its system prompt
# - Its allowed tools

architect:
  role_name: ARCHITECT
  description: "Decomposes requirements into atomic specifications"
  input_node_types:
    - REQ
    - ESCALATION
  output_node_types:
    - SPEC
    - PLAN
    - CLARIFICATION
  system_prompt_path: .blueprint/prompts/architect.md
  allowed_tools:
    - read_file
    - list_directory
    - search_web
    - fetch_url
  output_protocol: ArchitectOutput
  output_tool_name: submit_architecture

builder:
  role_name: BUILDER
  description: "Implements code from specifications"
  input_node_types:
    - SPEC
  output_node_types:
    - CODE
  system_prompt_path: .blueprint/prompts/builder.md
  allowed_tools:
    - read_file
    - write_file
    - list_directory
  output_protocol: BuilderOutput
  output_tool_name: submit_code

verifier:
  role_name: VERIFIER
  description: "Reviews and verifies code correctness"
  input_node_types:
    - CODE
  output_node_types:
    - TEST
  system_prompt_path: .blueprint/prompts/verifier.md
  allowed_tools:
    - read_file
    - list_directory
  output_protocol: VerifierOutput
  output_tool_name: submit_verdict

socrates:
  role_name: SOCRATES
  description: "Resolves ambiguity through clarifying questions"
  input_node_types:
    - CLARIFICATION
  output_node_types:
    - REQ  # Enriched requirement
  system_prompt_path: .blueprint/prompts/socrates.md
  allowed_tools:
    - read_file
    - search_web
  output_protocol: SocratesOutput
  output_tool_name: submit_questions
```

### 1.3 Update `core/protocols.py`

Add these new models:

```python
# === AGENT CONFIGURATION ===

class AgentConfig(BaseModel):
    """Configuration for a GenericAgent loaded from YAML."""
    role_name: str
    description: str
    input_node_types: List[str]
    output_node_types: List[str]
    system_prompt_path: str
    allowed_tools: List[str]
    output_protocol: str
    output_tool_name: str


class AgentOutput(BaseModel):
    """
    Standardized output structure for ALL agents.

    This replaces the agent-specific output handling.
    """
    thought: Optional[str] = Field(
        default=None,
        description="Agent's reasoning process"
    )
    plan: Optional[str] = Field(
        default=None,
        description="What the agent intends to do"
    )
    new_nodes: List[NodeSpec] = Field(
        default_factory=list,
        description="Nodes to create"
    )
    new_edges: List[EdgeSpec] = Field(
        default_factory=list,
        description="Edges to create"
    )
    status_updates: Dict[str, str] = Field(
        default_factory=dict,
        description="Node ID -> new status"
    )
    artifacts: Dict[str, str] = Field(
        default_factory=dict,
        description="Files to write: path -> content"
    )
```

---

## Phase 2: Runtime Logic

**Goal:** Build the generic engine that enforces the schemas.

**Constraint:** Runtime must ONLY consult `new_ontology.py` for decisions. No hardcoded type checks.

### 2.1 Create `agents/generic_agent.py`

- Single class that loads personality from `agent_manifest.yaml`
- Implements ReAct loop
- Uses `AgentOutput` for all responses
- Does not know what it's building - follows config

### 2.2 Create `infrastructure/graph_runtime.py`

- `can_transition(node_id, target_status)` - consults TransitionMatrix
- `evaluate_condition(node_id, condition_name)` - evaluates named conditions
- `execute_step(node_id)` - process one node if physics allows
- `run_until_complete()` - main loop
- `prune_orphans()` - replaces Janitor

### 2.3 Enhance `infrastructure/graph_db.py`

Add query methods:
- `get_by_status(status)`
- `dependencies_met(node_id)`
- `is_blocked(node_id)`
- `get_context(node_id, radius)`
- `apply_result(agent_output)`
- `topological_order()`
- `get_execution_waves()`

---

## Phase 3: Cleanup

**Goal:** Delete legacy code, wire up new entry point.

### 3.1 Delete Files

```
rm agents/concrete_agents.py
rm orchestration/gaadp_engine.py
rm orchestration/dependency_resolver.py
rm orchestration/escalation_controller.py
rm requirements/socratic_agent.py
rm core/state_machine.py  # if exists
```

### 3.2 Refactor Entry Point

Replace `production_main.py` with minimal `main.py`:
- ~50 lines
- Initialize GraphRuntime
- Main loop calls `runtime.run_until_complete()`

### 3.3 Verify

- Run reconstruction benchmarks
- Ensure same behavior
- Measure code reduction

---

## File Structure (Target)

```
gaadp/
├── core/
│   ├── new_ontology.py    # 250 lines - The Physics
│   ├── protocols.py       # 200 lines - Pydantic models
│   └── graph.py           # 300 lines - Enhanced GraphDB
├── agents/
│   ├── generic_agent.py   # 150 lines - Universal agent
│   └── base.py            # 100 lines - LLM wrapper
├── config/
│   └── agent_manifest.yaml
├── infrastructure/
│   ├── graph_runtime.py   # 200 lines - The Engine
│   ├── llm.py             # 150 lines - Provider abstraction
│   └── sandbox.py         # Keep as-is
├── .blueprint/
│   └── prompts/
│       ├── architect.md
│       ├── builder.md
│       ├── verifier.md
│       └── socrates.md
└── main.py                # 50 lines - Entry point

Total: ~1400 lines (down from ~5000+)
```

---

## Rollback Plan

If refactor fails:
```bash
git checkout d26aabf -- .
```

This restores the complete imperative architecture.

---

## Next Action

Begin Phase 1: Create `core/new_ontology.py` with the schemas above.
