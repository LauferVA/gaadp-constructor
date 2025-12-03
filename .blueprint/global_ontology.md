# GAADP GLOBAL ONTOLOGY
> VERSION: 3.0.0 (Graph-First Architecture)
> AUTHORITY: ABSOLUTE
> STATUS: IMMUTABLE

## I. AGENT ROLES

Agent roles are defined in `config/agent_manifest.yaml`. All agents use `GenericAgent` class.

| Role | Responsibility | Dispatched For |
| :--- | :--- | :--- |
| **ARCHITECT** | Decomposes REQ into SPECs. Defines execution plan. | `(REQ, needs_decomposition)` |
| **BUILDER** | Implements CODE from SPEC. | `(SPEC, ready_for_build)` |
| **VERIFIER** | Reviews CODE. Signs verification edges. | `(CODE, needs_verification)` |

## II. NODE TYPES

Defined in `core/ontology.py` as `NodeType` enum.

### Primary Workflow Types

| Node Type | Definition |
| :--- | :--- |
| **`REQ`** | User requirement (entry point). Only node without TRACES_TO. |
| **`CLARIFICATION`** | Ambiguity requiring human input. Blocks dependents. |
| **`SPEC`** | Atomic technical specification. |
| **`PLAN`** | Decomposition strategy from ARCHITECT. |
| **`CODE`** | Implementation artifact. |
| **`TEST`** | Verification result/test code. |
| **`DOC`** | Documentation artifact. |
| **`ESCALATION`** | Failure requiring intervention. Terminal unless resolved. |

### CPG (Code Property Graph) Types

| Node Type | Definition |
| :--- | :--- |
| **`CLASS`** | Class definition (from static analysis). |
| **`FUNCTION`** | Function/method definition. |
| **`CALL`** | Function call site. |

## III. EDGE TYPES

Defined in `core/ontology.py` as `EdgeType` enum.

| Edge Type | Meaning | Constraint |
| :--- | :--- | :--- |
| **`TRACES_TO`** | Provenance to source REQ. | Mandatory for all non-REQ nodes. |
| **`DEPENDS_ON`** | Must complete before processing. | **NO CYCLES.** |
| **`IMPLEMENTS`** | CODE satisfies SPEC. | CODE → SPEC |
| **`VERIFIES`** | Attestation of correctness. | Requires signature. |
| **`DEFINES`** | PLAN produces SPECs. | PLAN → SPEC |
| **`BLOCKS`** | Waiting on resolution. | → CLARIFICATION/ESCALATION |
| **`FEEDBACK`** | Critique from failed verification. | CODE → SPEC |
| **`RESOLVED_BY`** | Answer to clarification. | CLARIFICATION → response |
| **`CONTAINS`** | Parent contains child. | For CPG hierarchies. |
| **`REFERENCES`** | Caller references callee. | For CPG analysis. |
| **`INHERITS`** | Subclass extends superclass. | For CPG analysis. |

## IV. NODE STATUSES

Defined in `core/ontology.py` as `NodeStatus` enum.

| Status | Description | Terminal? |
| :--- | :--- | :--- |
| **`PENDING`** | Awaiting processing. | No |
| **`PROCESSING`** | Agent actively working. | No |
| **`BLOCKED`** | Waiting on CLARIFICATION/ESCALATION. | No |
| **`VERIFIED`** | Successfully completed. | **Yes** |
| **`FAILED`** | Unrecoverable failure. | **Yes** |

## V. TRANSITION MATRIX

State transitions are governed by `TRANSITION_MATRIX` in `core/ontology.py`.

Format: `(current_status, node_type) -> List[TransitionRule]`

Each `TransitionRule` specifies:
- `target_status`: The status to transition to
- `required_conditions`: List of condition names that must all be true
- `priority`: Higher priority rules are evaluated first

## VI. NODE METADATA

Defined in `core/ontology.py` as `NodeMetadata` Pydantic model.

| Field | Type | Purpose |
| :--- | :--- | :--- |
| `cost_limit` | `Optional[float]` | Max cost for this node (physics) |
| `cost_actual` | `float` | Actual cost incurred |
| `security_level` | `int` | Required clearance level |
| `attempts` | `int` | Processing attempt count |
| `max_attempts` | `int` | Max retries before FAILED |
| `created_at` | `datetime` | Creation timestamp |
| `updated_at` | `datetime` | Last modification |

## VII. GOVERNANCE AS DATA

Governance is embedded in `NodeMetadata`, not middleware:

1. **Budget**: `cost_actual > cost_limit` → Cannot transition to PROCESSING
2. **Retries**: `attempts >= max_attempts` → Transitions to FAILED
3. **Security**: Agent clearance checked against `security_level`

This is **physics**, not **prompts**. The runtime cannot bypass these constraints.
