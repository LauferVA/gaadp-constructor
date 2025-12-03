"""
GAADP NEW ONTOLOGY - The Physics of the System

This module defines the declarative schema that governs all behavior.
The runtime CONSULTS this schema; it does not contain execution logic.

Key Principles:
1. TransitionMatrix is the SINGLE SOURCE OF TRUTH for state changes
2. NodeMetadata contains governance as DATA, not middleware
3. AGENT_DISPATCH determines which agent processes which node
4. Named conditions are evaluated by the runtime, defined here

This replaces imperative if/else orchestration with declarative rules.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Optional, Literal, Set, Any
from enum import Enum
from datetime import datetime, timezone
import uuid


# =============================================================================
# ENUMS (Simple labels - behavior defined in TransitionMatrix)
# =============================================================================

class NodeType(str, Enum):
    """Types of nodes in the graph."""
    REQ = "REQ"                      # User requirement (entry point)
    CLARIFICATION = "CLARIFICATION"  # Ambiguity needing human input
    SPEC = "SPEC"                    # Atomic specification
    PLAN = "PLAN"                    # Decomposition strategy
    CODE = "CODE"                    # Implementation artifact
    TEST = "TEST"                    # Verification result
    DOC = "DOC"                      # Documentation artifact
    ESCALATION = "ESCALATION"        # Failure requiring intervention


class EdgeType(str, Enum):
    """Types of edges between nodes."""
    TRACES_TO = "TRACES_TO"          # Provenance: any -> REQ
    DEPENDS_ON = "DEPENDS_ON"        # Ordering: SPEC -> SPEC (must complete first)
    IMPLEMENTS = "IMPLEMENTS"        # Realization: CODE -> SPEC
    VERIFIES = "VERIFIES"            # Validation: TEST -> CODE
    DEFINES = "DEFINES"              # Decomposition: PLAN -> SPEC
    BLOCKS = "BLOCKS"                # Blocking: any -> CLARIFICATION/ESCALATION
    FEEDBACK = "FEEDBACK"            # Critique: failed CODE -> SPEC
    RESOLVED_BY = "RESOLVED_BY"      # Answer: CLARIFICATION -> response
    # Extended relationships (for AST/CPG features)
    CONTAINS = "CONTAINS"            # Containment: parent -> child
    REFERENCES = "REFERENCES"        # Reference: caller -> callee
    INHERITS = "INHERITS"            # Inheritance: subclass -> superclass


class NodeStatus(str, Enum):
    """Status states for nodes."""
    PENDING = "PENDING"              # Waiting to be processed
    PROCESSING = "PROCESSING"        # Currently being processed
    BLOCKED = "BLOCKED"              # Waiting on another node
    VERIFIED = "VERIFIED"            # Successfully completed
    FAILED = "FAILED"                # Terminal failure


# =============================================================================
# NODE METADATA (Governance as Data)
# =============================================================================

class NodeMetadata(BaseModel):
    """
    Governance properties embedded in every node.

    These are PHYSICS, not policy. A node physically cannot be
    processed if it violates these constraints. This moves governance
    from "middleware that can be bypassed" to "data that cannot."
    """
    # === Cost Governance ===
    cost_limit: Optional[float] = Field(
        default=None,
        description="Maximum USD cost allowed for processing this node. None = unlimited."
    )
    cost_actual: float = Field(
        default=0.0,
        description="Actual cost incurred processing this node."
    )

    # === Security ===
    security_level: int = Field(
        default=0,
        ge=0,
        le=3,
        description="0=public, 1=internal, 2=confidential, 3=restricted"
    )
    required_clearance: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Minimum clearance level to process this node"
    )

    # === Versioning ===
    version: str = Field(default="1.0.0")
    parent_version: Optional[str] = Field(
        default=None,
        description="Version of node this was derived from"
    )

    # === Approval Workflow ===
    approval_status: Literal["none", "pending", "approved", "rejected"] = Field(
        default="none",
        description="Approval state for gated nodes"
    )
    approved_by: Optional[str] = Field(
        default=None,
        description="Agent/user who approved"
    )
    approval_required: bool = Field(
        default=False,
        description="Whether this node requires approval before processing"
    )

    # === Retry Tracking ===
    attempts: int = Field(
        default=0,
        ge=0,
        description="Number of processing attempts so far"
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum attempts before escalation"
    )
    last_error: Optional[str] = Field(
        default=None,
        description="Error message from last failed attempt"
    )

    # === Timing ===
    timeout_seconds: Optional[int] = Field(
        default=None,
        description="Max processing time in seconds. None = no limit."
    )
    deadline: Optional[datetime] = Field(
        default=None,
        description="Absolute deadline for completion"
    )

    # === Extension Point ===
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional metadata"
    )

    class Config:
        extra = "allow"


# =============================================================================
# NODE SPECIFICATION (The Atomic Unit)
# =============================================================================

class NodeSpec(BaseModel):
    """
    The atomic unit of the graph.

    Every artifact in GAADP is a NodeSpec. The graph is a collection
    of NodeSpecs connected by EdgeSpecs.
    """
    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier"
    )
    type: NodeType = Field(description="Node type from ontology")
    content: str = Field(description="The actual artifact content")
    status: NodeStatus = Field(
        default=NodeStatus.PENDING,
        description="Current processing status"
    )
    metadata: NodeMetadata = Field(
        default_factory=NodeMetadata,
        description="Governance and tracking properties"
    )

    # === Provenance ===
    created_by: str = Field(
        description="Agent ID or 'human' who created this node"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last modification timestamp"
    )

    # === Integrity ===
    signature: Optional[str] = Field(
        default=None,
        description="Content hash for integrity verification"
    )
    previous_hash: Optional[str] = Field(
        default=None,
        description="Hash of previous node in chain (for audit trail)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# EDGE SPECIFICATION
# =============================================================================

class EdgeSpec(BaseModel):
    """
    Edge connecting two nodes in the graph.

    Edges define relationships and constraints between nodes.
    """
    source_id: str = Field(description="ID of source node")
    target_id: str = Field(description="ID of target node")
    type: EdgeType = Field(description="Edge type from ontology")

    # === Provenance ===
    created_by: str = Field(description="Agent ID who created this edge")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # === Properties ===
    weight: float = Field(
        default=1.0,
        description="Edge weight for prioritization/ranking"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional edge properties"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# TRANSITION RULES (The Physics)
# =============================================================================

class TransitionRule(BaseModel):
    """
    Defines requirements for a status transition.

    This is the "physics" of the system - transitions that don't
    satisfy these rules are physically impossible.
    """
    target_status: NodeStatus = Field(
        description="Status to transition to"
    )
    required_edge_types: List[EdgeType] = Field(
        default_factory=list,
        description="Must have incoming edges of these types"
    )
    required_conditions: List[str] = Field(
        default_factory=list,
        description="Named conditions that must evaluate to True"
    )
    priority: int = Field(
        default=0,
        description="Higher priority rules are checked first"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# THE TRANSITION MATRIX (Single Source of Truth)
# =============================================================================

# Key: (current_status, node_type) -> List of possible transitions
# The runtime checks these rules to determine valid state changes.
# NO OTHER CODE should contain status transition logic.

TRANSITION_MATRIX: Dict[Tuple[str, str], List[TransitionRule]] = {

    # =========================================================================
    # REQ (Requirement) Transitions
    # =========================================================================

    # REQ: PENDING -> PROCESSING (Architect starts decomposition)
    (NodeStatus.PENDING.value, NodeType.REQ.value): [
        TransitionRule(
            target_status=NodeStatus.PROCESSING,
            required_conditions=["cost_under_limit", "not_blocked"],
            priority=10
        ),
        TransitionRule(
            target_status=NodeStatus.BLOCKED,
            required_conditions=["has_pending_clarification"],
            priority=20  # Check blocking first
        ),
    ],

    # REQ: PROCESSING -> VERIFIED/BLOCKED
    (NodeStatus.PROCESSING.value, NodeType.REQ.value): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_conditions=["all_specs_verified"],
            priority=10
        ),
        TransitionRule(
            target_status=NodeStatus.BLOCKED,
            required_conditions=["has_pending_clarification"],
            priority=20
        ),
    ],

    # REQ: BLOCKED -> PENDING (clarification resolved)
    (NodeStatus.BLOCKED.value, NodeType.REQ.value): [
        TransitionRule(
            target_status=NodeStatus.PENDING,
            required_conditions=["no_pending_clarifications"],
            priority=10
        ),
    ],

    # =========================================================================
    # SPEC (Specification) Transitions
    # =========================================================================

    # SPEC: PENDING -> PROCESSING (Builder starts implementation)
    (NodeStatus.PENDING.value, NodeType.SPEC.value): [
        TransitionRule(
            target_status=NodeStatus.PROCESSING,
            required_conditions=["cost_under_limit", "dependencies_verified", "not_blocked"],
            priority=10
        ),
        TransitionRule(
            target_status=NodeStatus.BLOCKED,
            required_conditions=["has_unmet_dependencies"],
            priority=20
        ),
    ],

    # SPEC: PROCESSING -> VERIFIED/FAILED
    (NodeStatus.PROCESSING.value, NodeType.SPEC.value): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_edge_types=[EdgeType.IMPLEMENTS],
            required_conditions=["implementation_verified"],
            priority=10
        ),
        TransitionRule(
            target_status=NodeStatus.FAILED,
            required_conditions=["max_attempts_exceeded"],
            priority=5
        ),
    ],

    # SPEC: BLOCKED -> PENDING
    (NodeStatus.BLOCKED.value, NodeType.SPEC.value): [
        TransitionRule(
            target_status=NodeStatus.PENDING,
            required_conditions=["dependencies_verified"],
            priority=10
        ),
    ],

    # SPEC: FAILED -> (creates ESCALATION, handled by runtime)
    # No transition rule - FAILED is terminal, but triggers escalation node creation

    # =========================================================================
    # CODE Transitions
    # =========================================================================

    # CODE: PENDING -> PROCESSING (Verifier starts review)
    (NodeStatus.PENDING.value, NodeType.CODE.value): [
        TransitionRule(
            target_status=NodeStatus.PROCESSING,
            required_conditions=["cost_under_limit"],
            priority=10
        ),
    ],

    # CODE: PROCESSING -> VERIFIED/FAILED
    (NodeStatus.PROCESSING.value, NodeType.CODE.value): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_edge_types=[EdgeType.VERIFIES],
            required_conditions=["verification_passed"],
            priority=10
        ),
        TransitionRule(
            target_status=NodeStatus.FAILED,
            required_conditions=["verification_failed"],
            priority=10
        ),
    ],

    # =========================================================================
    # TEST Transitions
    # =========================================================================

    # TEST is created in VERIFIED or FAILED state (terminal)
    # No transitions needed - TEST nodes don't change status

    # =========================================================================
    # CLARIFICATION Transitions
    # =========================================================================

    # CLARIFICATION: PENDING -> PROCESSING (Socrates or Human engagement)
    (NodeStatus.PENDING.value, NodeType.CLARIFICATION.value): [
        TransitionRule(
            target_status=NodeStatus.PROCESSING,
            required_conditions=[],  # Always processable
            priority=10
        ),
    ],

    # CLARIFICATION: PROCESSING -> VERIFIED (answered)
    (NodeStatus.PROCESSING.value, NodeType.CLARIFICATION.value): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_edge_types=[EdgeType.RESOLVED_BY],
            required_conditions=["has_resolution"],
            priority=10
        ),
    ],

    # =========================================================================
    # ESCALATION Transitions
    # =========================================================================

    # ESCALATION: PENDING -> PROCESSING (Architect re-planning)
    (NodeStatus.PENDING.value, NodeType.ESCALATION.value): [
        TransitionRule(
            target_status=NodeStatus.PROCESSING,
            required_conditions=["cost_under_limit"],
            priority=10
        ),
    ],

    # ESCALATION: PROCESSING -> VERIFIED/FAILED
    (NodeStatus.PROCESSING.value, NodeType.ESCALATION.value): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_conditions=["replan_produced"],
            priority=10
        ),
        TransitionRule(
            target_status=NodeStatus.FAILED,
            required_conditions=["max_escalations_exceeded"],
            priority=5
        ),
    ],

    # =========================================================================
    # PLAN Transitions
    # =========================================================================

    # PLAN nodes are informational - created VERIFIED, no transitions
    (NodeStatus.PENDING.value, NodeType.PLAN.value): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_conditions=[],  # Plans are immediately verified
            priority=10
        ),
    ],

    # =========================================================================
    # DOC Transitions
    # =========================================================================

    # DOC nodes follow similar pattern to PLAN
    (NodeStatus.PENDING.value, NodeType.DOC.value): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_conditions=[],
            priority=10
        ),
    ],
}


# =============================================================================
# AGENT DISPATCH RULES
# =============================================================================

# Maps (NodeType, ConditionName) -> AgentRole
# This replaces hardcoded "if node.type == CODE: use Verifier" logic

AGENT_DISPATCH: Dict[Tuple[str, str], str] = {
    # REQ with no specs -> Architect decomposes
    (NodeType.REQ.value, "needs_decomposition"): "ARCHITECT",

    # SPEC ready for implementation -> Builder
    (NodeType.SPEC.value, "ready_for_build"): "BUILDER",

    # CODE needs verification -> Verifier
    (NodeType.CODE.value, "needs_verification"): "VERIFIER",

    # CLARIFICATION pending -> Socrates (or human)
    (NodeType.CLARIFICATION.value, "needs_resolution"): "SOCRATES",

    # ESCALATION -> Architect re-plans
    (NodeType.ESCALATION.value, "needs_replan"): "ARCHITECT",
}


# =============================================================================
# KNOWN CONDITIONS
# =============================================================================

# These are the named conditions referenced in TransitionRules.
# The runtime implements evaluation functions for each.
# This set allows validation that rules only use known conditions.

KNOWN_CONDITIONS: Set[str] = {
    # Cost governance
    "cost_under_limit",           # metadata.cost_actual < metadata.cost_limit

    # Dependency tracking
    "dependencies_verified",      # All DEPENDS_ON targets are VERIFIED
    "has_unmet_dependencies",     # Has DEPENDS_ON edges to non-VERIFIED nodes

    # Blocking
    "not_blocked",                # No BLOCKS edges to PENDING nodes
    "has_pending_clarification",  # Has CLARIFICATION child in PENDING/PROCESSING

    # Completion checks
    "all_specs_verified",         # All SPEC children are VERIFIED
    "implementation_verified",    # CODE implementing this SPEC is VERIFIED
    "verification_passed",        # Verifier verdict is PASS
    "verification_failed",        # Verifier verdict is FAIL

    # Retry/escalation
    "max_attempts_exceeded",      # metadata.attempts >= metadata.max_attempts
    "max_escalations_exceeded",   # Too many escalation attempts

    # Resolution
    "has_resolution",             # CLARIFICATION has RESOLVED_BY edge
    "no_pending_clarifications",  # No unresolved CLARIFICATION nodes
    "replan_produced",            # Architect produced new SPEC nodes

    # Dispatch conditions
    "needs_decomposition",        # REQ has no SPEC children
    "ready_for_build",            # SPEC dependencies met, no CODE yet
    "needs_verification",         # CODE has no TEST
    "needs_resolution",           # CLARIFICATION not yet answered
    "needs_replan",               # ESCALATION not yet addressed
}


# =============================================================================
# VALIDATION
# =============================================================================

def validate_transition_matrix() -> List[str]:
    """
    Validate that all conditions in TRANSITION_MATRIX are known.

    Returns list of errors (empty if valid).
    """
    errors = []
    for (status, node_type), rules in TRANSITION_MATRIX.items():
        for rule in rules:
            for condition in rule.required_conditions:
                if condition not in KNOWN_CONDITIONS:
                    errors.append(
                        f"Unknown condition '{condition}' in rule "
                        f"({status}, {node_type}) -> {rule.target_status}"
                    )
    return errors


# Run validation on module load
_validation_errors = validate_transition_matrix()
if _validation_errors:
    import warnings
    for err in _validation_errors:
        warnings.warn(f"TransitionMatrix validation: {err}")
