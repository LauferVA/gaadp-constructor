"""
NODE STATE MACHINE
Enforces valid state transitions for graph nodes.
"""
import logging
from typing import Dict, Set, Optional
from core.ontology import NodeStatus, NodeType

logger = logging.getLogger("StateMachine")


# Valid transitions: from_state -> {to_states}
VALID_TRANSITIONS: Dict[NodeStatus, Set[NodeStatus]] = {
    NodeStatus.PENDING: {
        NodeStatus.IN_PROGRESS,  # Task starts
        NodeStatus.BLOCKED,      # Pre-hook rejection or missing deps
    },
    NodeStatus.IN_PROGRESS: {
        NodeStatus.COMPLETE,     # Task finishes successfully
        NodeStatus.FAILED,       # Task fails
        NodeStatus.BLOCKED,      # Blocked mid-execution
    },
    NodeStatus.BLOCKED: {
        NodeStatus.PENDING,      # Unblocked, ready for retry
        NodeStatus.FAILED,       # Permanent block
    },
    NodeStatus.COMPLETE: {
        NodeStatus.VERIFIED,     # Verification passes (CODE nodes)
        NodeStatus.FAILED,       # Verification fails (triggers feedback)
        NodeStatus.PENDING,      # Re-process (rare, for feedback loop)
    },
    NodeStatus.FAILED: {
        NodeStatus.PENDING,      # Retry via feedback loop
    },
    NodeStatus.VERIFIED: set(),  # Terminal state - no transitions out
}

# Node types that can be verified
VERIFIABLE_TYPES = {NodeType.CODE, NodeType.TEST}


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class NodeStateMachine:
    """
    Enforces valid state transitions for nodes.
    Integrated with GraphDB to validate all status changes.
    """

    def __init__(self):
        self._transition_history: Dict[str, list] = {}  # node_id -> [transitions]

    def validate_transition(
        self,
        node_id: str,
        from_status: NodeStatus,
        to_status: NodeStatus,
        node_type: Optional[NodeType] = None
    ) -> bool:
        """
        Validate a state transition.

        Args:
            node_id: The node ID
            from_status: Current status
            to_status: Target status
            node_type: Optional node type for type-specific rules

        Returns:
            True if valid

        Raises:
            StateTransitionError if invalid
        """
        # Same state is always valid (no-op)
        if from_status == to_status:
            return True

        # Convert strings to enums if needed
        if isinstance(from_status, str):
            from_status = NodeStatus(from_status)
        if isinstance(to_status, str):
            to_status = NodeStatus(to_status)

        # Check valid transitions
        valid_targets = VALID_TRANSITIONS.get(from_status, set())
        if to_status not in valid_targets:
            raise StateTransitionError(
                f"Invalid transition for node {node_id}: "
                f"{from_status.value} → {to_status.value}. "
                f"Valid targets: {[s.value for s in valid_targets]}"
            )

        # Type-specific rules
        if to_status == NodeStatus.VERIFIED:
            if node_type and node_type not in VERIFIABLE_TYPES:
                raise StateTransitionError(
                    f"Node type {node_type.value} cannot be VERIFIED. "
                    f"Only {[t.value for t in VERIFIABLE_TYPES]} can be verified."
                )

        return True

    def transition(
        self,
        node_id: str,
        from_status: NodeStatus,
        to_status: NodeStatus,
        node_type: Optional[NodeType] = None,
        reason: str = ""
    ) -> bool:
        """
        Execute a state transition with validation and logging.

        Args:
            node_id: The node ID
            from_status: Current status
            to_status: Target status
            node_type: Optional node type
            reason: Reason for transition (for audit)

        Returns:
            True if transition executed

        Raises:
            StateTransitionError if invalid
        """
        self.validate_transition(node_id, from_status, to_status, node_type)

        # Record in history
        if node_id not in self._transition_history:
            self._transition_history[node_id] = []

        self._transition_history[node_id].append({
            'from': from_status.value if isinstance(from_status, NodeStatus) else from_status,
            'to': to_status.value if isinstance(to_status, NodeStatus) else to_status,
            'reason': reason
        })

        logger.debug(f"State transition: {node_id} {from_status} → {to_status} ({reason})")
        return True

    def get_history(self, node_id: str) -> list:
        """Get transition history for a node."""
        return self._transition_history.get(node_id, [])

    def can_transition(
        self,
        from_status: NodeStatus,
        to_status: NodeStatus,
        node_type: Optional[NodeType] = None
    ) -> bool:
        """
        Check if a transition is valid without raising.

        Returns:
            True if valid, False otherwise
        """
        try:
            self.validate_transition("check", from_status, to_status, node_type)
            return True
        except StateTransitionError:
            return False

    def get_valid_transitions(self, from_status: NodeStatus) -> Set[NodeStatus]:
        """Get all valid target states from a given state."""
        return VALID_TRANSITIONS.get(from_status, set())


# Global instance for convenience
_state_machine = NodeStateMachine()


def validate_transition(node_id: str, from_status, to_status, node_type=None) -> bool:
    """Module-level convenience function."""
    return _state_machine.validate_transition(node_id, from_status, to_status, node_type)


def transition(node_id: str, from_status, to_status, node_type=None, reason="") -> bool:
    """Module-level convenience function."""
    return _state_machine.transition(node_id, from_status, to_status, node_type, reason)
