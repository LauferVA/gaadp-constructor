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
    # Primary workflow types
    REQ = "REQ"                      # User requirement (entry point)
    RESEARCH = "RESEARCH"            # Research artifact (Research Standard v1.0)
    CLARIFICATION = "CLARIFICATION"  # Ambiguity needing human input
    SPEC = "SPEC"                    # Atomic specification
    PLAN = "PLAN"                    # Decomposition strategy
    CODE = "CODE"                    # Implementation artifact
    TEST = "TEST"                    # Verification result (legacy)
    TEST_SUITE = "TEST_SUITE"        # Gen-2 TDD: Comprehensive test results from TESTER
    DOC = "DOC"                      # Documentation artifact
    ESCALATION = "ESCALATION"        # Failure requiring intervention
    # CPG (Code Property Graph) types for static analysis
    CLASS = "CLASS"                  # Class definition
    FUNCTION = "FUNCTION"            # Function/method definition
    CALL = "CALL"                    # Function call site


class EdgeType(str, Enum):
    """Types of edges between nodes."""
    TRACES_TO = "TRACES_TO"          # Provenance: any -> REQ
    DEPENDS_ON = "DEPENDS_ON"        # Ordering: SPEC -> SPEC (must complete first)
    IMPLEMENTS = "IMPLEMENTS"        # Realization: CODE -> SPEC
    VERIFIES = "VERIFIES"            # Validation: TEST -> CODE
    TESTS = "TESTS"                  # TDD: TEST_SUITE -> CODE (Gen-2)
    DEFINES = "DEFINES"              # Decomposition: PLAN -> SPEC
    BLOCKS = "BLOCKS"                # Blocking: any -> CLARIFICATION/ESCALATION
    FEEDBACK = "FEEDBACK"            # Critique: failed CODE -> SPEC (or Tester -> Builder)
    RESOLVED_BY = "RESOLVED_BY"      # Answer: CLARIFICATION -> response
    RESEARCH_FOR = "RESEARCH_FOR"    # Research artifact for: RESEARCH -> REQ
    # Extended relationships (for AST/CPG features)
    CONTAINS = "CONTAINS"            # Containment: parent -> child
    REFERENCES = "REFERENCES"        # Reference: caller -> callee
    INHERITS = "INHERITS"            # Inheritance: subclass -> superclass


class NodeStatus(str, Enum):
    """Status states for nodes."""
    PENDING = "PENDING"              # Waiting to be processed
    PROCESSING = "PROCESSING"        # Currently being processed
    BLOCKED = "BLOCKED"              # Waiting on another node
    TESTING = "TESTING"              # Being tested by TESTER (Gen-2 TDD)
    TESTED = "TESTED"                # Tests passed, awaiting verification
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
        # Block only for BLOCKING clarifications (not clarifying-only)
        TransitionRule(
            target_status=NodeStatus.BLOCKED,
            required_conditions=["has_blocking_clarifications"],
            priority=20  # Check blocking first
        ),
        # Proceed even with clarifying-only ambiguities (use defaults)
        TransitionRule(
            target_status=NodeStatus.PROCESSING,
            required_conditions=["has_clarifying_only", "cost_under_limit"],
            priority=15
        ),
    ],

    # REQ: PROCESSING -> VERIFIED/BLOCKED
    (NodeStatus.PROCESSING.value, NodeType.REQ.value): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_conditions=["all_specs_verified"],
            priority=10
        ),
        # Block only for BLOCKING clarifications
        TransitionRule(
            target_status=NodeStatus.BLOCKED,
            required_conditions=["has_blocking_clarifications"],
            priority=20
        ),
    ],

    # REQ: BLOCKED -> PENDING (all blocking clarifications resolved)
    (NodeStatus.BLOCKED.value, NodeType.REQ.value): [
        TransitionRule(
            target_status=NodeStatus.PENDING,
            required_conditions=["all_blocking_resolved"],
            priority=10
        ),
        # Legacy: also unblock if no clarifications at all
        TransitionRule(
            target_status=NodeStatus.PENDING,
            required_conditions=["no_pending_clarifications"],
            priority=5
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
    # CODE Transitions (Gen-2: TDD Loop)
    # =========================================================================

    # CODE: PENDING -> TESTING (Tester starts, NOT directly to Verifier)
    (NodeStatus.PENDING.value, NodeType.CODE.value): [
        TransitionRule(
            target_status=NodeStatus.TESTING,
            required_conditions=["cost_under_limit", "needs_testing"],
            priority=10
        ),
    ],

    # CODE: TESTING -> TESTED/PENDING/FAILED (TDD Feedback Loop)
    (NodeStatus.TESTING.value, NodeType.CODE.value): [
        # PASS: Tests passed, move to TESTED (awaits Verifier)
        TransitionRule(
            target_status=NodeStatus.TESTED,
            required_conditions=["tests_passed"],
            priority=10
        ),
        # NEEDS_REVISION + under max attempts: Back to PENDING (Builder retry)
        TransitionRule(
            target_status=NodeStatus.PENDING,
            required_conditions=["tests_need_revision", "under_max_attempts"],
            priority=15  # Higher priority - check retry before fail
        ),
        # NEEDS_REVISION + max attempts exceeded: FAILED (escalate)
        TransitionRule(
            target_status=NodeStatus.FAILED,
            required_conditions=["tests_need_revision", "max_attempts_exceeded"],
            priority=10
        ),
        # FAIL (critical/security): Immediate FAILED
        TransitionRule(
            target_status=NodeStatus.FAILED,
            required_conditions=["tests_failed_critical"],
            priority=20  # Highest priority - security issues block immediately
        ),
    ],

    # CODE: TESTED -> VERIFIED/FAILED (Verifier final review)
    (NodeStatus.TESTED.value, NodeType.CODE.value): [
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

    # CODE: PROCESSING -> VERIFIED/FAILED (Legacy path for non-TDD)
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

    # CLARIFICATION: PROCESSING -> VERIFIED (answered by user or agent)
    (NodeStatus.PROCESSING.value, NodeType.CLARIFICATION.value): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_edge_types=[EdgeType.RESOLVED_BY],
            required_conditions=["has_resolution"],
            priority=10
        ),
        # User requested pause - move to BLOCKED
        TransitionRule(
            target_status=NodeStatus.BLOCKED,
            required_conditions=["user_requested_pause"],
            priority=5
        ),
        # Non-blocking clarification timed out - use default and verify
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_conditions=["clarification_timed_out", "can_use_default"],
            priority=3
        ),
    ],

    # CLARIFICATION: BLOCKED -> PROCESSING (user ready to resume)
    (NodeStatus.BLOCKED.value, NodeType.CLARIFICATION.value): [
        TransitionRule(
            target_status=NodeStatus.PROCESSING,
            required_conditions=["user_ready_to_answer"],
            priority=10
        ),
        # Timeout on blocked clarification - escalate or use default
        TransitionRule(
            target_status=NodeStatus.FAILED,
            required_conditions=["clarification_timed_out"],
            priority=5
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
    # RESEARCH Transitions (Research Standard v1.0)
    # =========================================================================

    # RESEARCH: PENDING -> PROCESSING (Researcher transforms raw prompt)
    (NodeStatus.PENDING.value, NodeType.RESEARCH.value): [
        TransitionRule(
            target_status=NodeStatus.PROCESSING,
            required_conditions=["cost_under_limit"],
            priority=10
        ),
    ],

    # RESEARCH: PROCESSING -> VERIFIED/FAILED
    (NodeStatus.PROCESSING.value, NodeType.RESEARCH.value): [
        TransitionRule(
            target_status=NodeStatus.VERIFIED,
            required_conditions=["research_verified"],
            priority=10
        ),
        TransitionRule(
            target_status=NodeStatus.FAILED,
            required_conditions=["max_attempts_exceeded"],
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
    # ==========================================================================
    # DIALECTIC PIPELINE (Pre-Research Ambiguity Detection)
    # ==========================================================================

    # REQ that hasn't been analyzed for ambiguity -> Dialector checks first
    (NodeType.REQ.value, "needs_dialectic"): "DIALECTOR",

    # ==========================================================================
    # RESEARCH PIPELINE (Gen-2: Sequential, not parallel)
    # ==========================================================================

    # REQ with no research artifact (and dialectic passed) -> Researcher transforms
    (NodeType.REQ.value, "needs_research"): "RESEARCHER",

    # RESEARCH needs verification -> Research Verifier
    (NodeType.RESEARCH.value, "needs_research_verification"): "RESEARCH_VERIFIER",

    # CRITICAL FIX (Gen-2): Verified RESEARCH triggers SPEC generation
    # This was the missing link that caused Gen-1 to stop at RESEARCH
    (NodeType.RESEARCH.value, "needs_spec_generation"): "ARCHITECT",

    # ==========================================================================
    # SPEC → CODE PIPELINE
    # ==========================================================================

    # REQ with no specs (legacy path, kept for non-research tasks)
    (NodeType.REQ.value, "needs_decomposition"): "ARCHITECT",

    # SPEC ready for implementation -> Builder
    (NodeType.SPEC.value, "ready_for_build"): "BUILDER",

    # ==========================================================================
    # TDD LOOP (Gen-2: Builder ↔ Tester feedback loop)
    # ==========================================================================

    # CODE needs testing -> Tester (Gen-2 TDD)
    (NodeType.CODE.value, "needs_testing"): "TESTER",

    # CODE passed tests, needs final verification -> Verifier
    (NodeType.CODE.value, "needs_verification"): "VERIFIER",

    # ==========================================================================
    # CLARIFICATION & ESCALATION
    # ==========================================================================

    # CLARIFICATION pending -> Socrates (or human via VizServer)
    (NodeType.CLARIFICATION.value, "needs_resolution"): "SOCRATES",

    # CLARIFICATION answered by user -> Socrates confirms/processes answer
    (NodeType.CLARIFICATION.value, "user_provided_answer"): "SOCRATES",

    # ESCALATION -> Architect re-plans
    (NodeType.ESCALATION.value, "needs_replan"): "ARCHITECT",

    # ==========================================================================
    # Graph-Native Socratic Q&A Dispatch
    # ==========================================================================

    # REQ blocked by clarifications -> hold for SOCRATES to manage Q&A
    # (SOCRATES will ask questions via VizServer and process answers)
    (NodeType.REQ.value, "has_blocking_clarifications"): "SOCRATES",
}


# =============================================================================
# KNOWN CONDITIONS
# =============================================================================

# These are the named conditions referenced in TransitionRules.
# The runtime implements evaluation functions for each.
# This set allows validation that rules only use known conditions.

KNOWN_CONDITIONS: Set[str] = {
    # ==========================================================================
    # Cost governance
    # ==========================================================================
    "cost_under_limit",           # metadata.cost_actual < metadata.cost_limit

    # ==========================================================================
    # Dependency tracking
    # ==========================================================================
    "dependencies_verified",      # All DEPENDS_ON targets are VERIFIED
    "has_unmet_dependencies",     # Has DEPENDS_ON edges to non-VERIFIED nodes

    # ==========================================================================
    # Blocking
    # ==========================================================================
    "not_blocked",                # No BLOCKS edges to PENDING nodes
    "has_pending_clarification",  # Has CLARIFICATION child in PENDING/PROCESSING

    # ==========================================================================
    # Completion checks
    # ==========================================================================
    "all_specs_verified",         # All SPEC children are VERIFIED
    "implementation_verified",    # CODE implementing this SPEC is VERIFIED
    "verification_passed",        # Verifier verdict is PASS
    "verification_failed",        # Verifier verdict is FAIL

    # ==========================================================================
    # Retry/escalation
    # ==========================================================================
    "max_attempts_exceeded",      # metadata.attempts >= metadata.max_attempts
    "max_escalations_exceeded",   # Too many escalation attempts
    "under_max_attempts",         # metadata.attempts < metadata.max_attempts (TDD loop)

    # ==========================================================================
    # Resolution
    # ==========================================================================
    "has_resolution",             # CLARIFICATION has RESOLVED_BY edge
    "no_pending_clarifications",  # No unresolved CLARIFICATION nodes
    "replan_produced",            # Architect produced new SPEC nodes

    # ==========================================================================
    # Dispatch conditions (original)
    # ==========================================================================
    "needs_decomposition",        # REQ has no SPEC children
    "ready_for_build",            # SPEC dependencies met, no CODE yet
    "needs_verification",         # CODE passed tests, needs final review
    "needs_resolution",           # CLARIFICATION not yet answered
    "needs_replan",               # ESCALATION not yet addressed

    # ==========================================================================
    # Research Standard v1.0 conditions
    # ==========================================================================
    "needs_research",             # REQ has no RESEARCH artifact yet
    "needs_research_verification", # RESEARCH artifact needs verification
    "research_verified",          # RESEARCH artifact passed 8/10 criteria

    # ==========================================================================
    # Gen-2 Pipeline Gap Fix
    # ==========================================================================
    "needs_spec_generation",      # RESEARCH is VERIFIED but no SPEC created yet
    "has_verified_research",      # REQ has a VERIFIED RESEARCH child

    # ==========================================================================
    # Gen-2 TDD Loop conditions
    # ==========================================================================
    "needs_testing",              # CODE created, needs TESTER review
    "tests_passed",               # TESTER verdict is PASS
    "tests_need_revision",        # TESTER verdict is NEEDS_REVISION (retry)
    "tests_failed_critical",      # TESTER verdict is FAIL (security/critical)
    "code_tested",                # CODE has passed all tests (status=TESTED)

    # ==========================================================================
    # Dialectic Pipeline conditions (Pre-Research Ambiguity Detection)
    # ==========================================================================
    "needs_dialectic",            # REQ hasn't been analyzed for ambiguity yet
    "dialectic_passed",           # Dialector found no blocking ambiguities
    "dialectic_blocked",          # Dialector found blocking ambiguities, waiting for user

    # ==========================================================================
    # Graph-Native Socratic Q&A conditions
    # ==========================================================================
    "has_blocking_clarifications",     # Has CLARIFICATION children with impact_level="blocking"
    "has_clarifying_only",             # Has only non-blocking CLARIFICATION children
    "all_blocking_resolved",           # All blocking CLARIFICATIONs are VERIFIED
    "user_provided_answer",            # VizServer received user response for this CLARIFICATION
    "user_requested_pause",            # User asked to defer answering this CLARIFICATION
    "user_ready_to_answer",            # User signaled ready to resume paused CLARIFICATION
    "clarification_timed_out",         # Timeout expired waiting for user response
    "can_use_default",                 # CLARIFICATION is non-blocking, default can be used
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
