"""
GAADP TELEMETRY LAYER - The Flight Recorder
============================================
Captures granular execution data for measuring Trajectory Validity,
Token Efficiency, and enabling regression detection.

Key Metrics:
- Trajectory Validity: Did the system follow a valid path through the graph?
- Token Efficiency: Tokens used per successful outcome
- Cycle Efficiency: Number of iterations to reach VERIFIED status

Usage:
    from core.telemetry import TelemetryRecorder, TelemetryEvent

    recorder = TelemetryRecorder.get_instance()
    recorder.log_event("ARCHITECT", "decompose", {"specs_created": 3})

    # At end of session
    recorder.flush()
    report = recorder.get_session_report()
"""
import os
import json
import uuid
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger("GAADP.Telemetry")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class EventType(str, Enum):
    """Types of telemetry events."""
    # Lifecycle events
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # Agent events
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"

    # State transition events
    STATE_TRANSITION = "state_transition"
    TRANSITION_BLOCKED = "transition_blocked"

    # Execution events
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"

    # Node events
    NODE_CREATED = "node_created"
    EDGE_CREATED = "edge_created"

    # LLM events
    LLM_CALL_START = "llm_call_start"
    LLM_CALL_END = "llm_call_end"

    # Metrics
    COST_UPDATE = "cost_update"
    TOKENS_USED = "tokens_used"

    # Custom
    CUSTOM = "custom"


@dataclass
class TelemetryEvent:
    """
    Structured telemetry event.

    All events are recorded with:
    - session_id: Unique session identifier
    - timestamp: ISO format UTC timestamp
    - event_type: Type of event (see EventType enum)
    - role: Agent role or "system"
    - step: Execution step/iteration number
    - payload: Event-specific data
    """
    session_id: str
    event_type: str
    role: str
    step: int
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryEvent":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SessionMetrics:
    """Aggregated metrics for a session."""
    session_id: str
    started_at: str
    ended_at: Optional[str] = None

    # Execution metrics
    total_iterations: int = 0
    nodes_created: int = 0
    edges_created: int = 0
    nodes_verified: int = 0
    nodes_failed: int = 0

    # Agent metrics
    agent_calls: int = 0
    agent_errors: int = 0

    # Cost metrics
    total_cost_usd: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0

    # Trajectory metrics
    transitions_attempted: int = 0
    transitions_blocked: int = 0
    transitions_successful: int = 0

    # LLM metrics
    llm_calls: int = 0
    llm_latency_total_ms: float = 0.0

    def get_success_rate(self) -> float:
        """Calculate verification success rate."""
        total = self.nodes_verified + self.nodes_failed
        if total == 0:
            return 0.0
        return self.nodes_verified / total

    def get_avg_llm_latency_ms(self) -> float:
        """Calculate average LLM latency."""
        if self.llm_calls == 0:
            return 0.0
        return self.llm_latency_total_ms / self.llm_calls

    def get_tokens_per_verified_node(self) -> float:
        """Calculate token efficiency."""
        if self.nodes_verified == 0:
            return float('inf')
        return (self.tokens_input + self.tokens_output) / self.nodes_verified

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['success_rate'] = self.get_success_rate()
        d['avg_llm_latency_ms'] = self.get_avg_llm_latency_ms()
        d['tokens_per_verified_node'] = self.get_tokens_per_verified_node()
        return d


# =============================================================================
# SINGLETON RECORDER
# =============================================================================

class TelemetryRecorder:
    """
    Singleton telemetry recorder.

    Writes to .gaadp/logs/telemetry.jsonl in append mode.
    Thread-safe for concurrent agent execution.
    """

    _instance: Optional["TelemetryRecorder"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TelemetryRecorder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._session_id = uuid.uuid4().hex
        self._step = 0
        self._events: List[TelemetryEvent] = []
        self._metrics = SessionMetrics(
            session_id=self._session_id,
            started_at=datetime.now(timezone.utc).isoformat()
        )

        # Setup log directory
        self._log_dir = Path(".gaadp/logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "telemetry.jsonl"

        # Write lock for file operations
        self._write_lock = threading.Lock()

        logger.info(f"TelemetryRecorder initialized: session={self._session_id[:8]}")

        # Log session start
        self._log_internal(
            EventType.SESSION_START,
            "system",
            {"log_file": str(self._log_file)}
        )

    @classmethod
    def get_instance(cls) -> "TelemetryRecorder":
        """Get the singleton instance."""
        return cls()

    @classmethod
    def reset_instance(cls):
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    @property
    def session_id(self) -> str:
        """Get current session ID."""
        return self._session_id

    @property
    def current_step(self) -> int:
        """Get current step number."""
        return self._step

    def increment_step(self):
        """Increment the step counter."""
        self._step += 1

    # =========================================================================
    # LOGGING METHODS
    # =========================================================================

    def _log_internal(
        self,
        event_type: EventType,
        role: str,
        payload: Dict[str, Any]
    ) -> TelemetryEvent:
        """Internal logging method."""
        event = TelemetryEvent(
            session_id=self._session_id,
            event_type=event_type.value,
            role=role,
            step=self._step,
            payload=payload
        )

        self._events.append(event)

        # Write to file
        with self._write_lock:
            try:
                with open(self._log_file, "a") as f:
                    f.write(event.to_json() + "\n")
            except Exception as e:
                logger.error(f"Failed to write telemetry: {e}")

        return event

    def log_event(
        self,
        role: str,
        action: str,
        payload: Optional[Dict[str, Any]] = None
    ) -> TelemetryEvent:
        """
        Log a custom event.

        Args:
            role: Agent role or component name
            action: Action being performed
            payload: Additional data
        """
        return self._log_internal(
            EventType.CUSTOM,
            role,
            {"action": action, **(payload or {})}
        )

    def log_state_transition(
        self,
        node_id: str,
        node_type: str,
        from_status: str,
        to_status: str,
        reason: Optional[str] = None
    ):
        """Log a state transition."""
        self._metrics.transitions_successful += 1

        return self._log_internal(
            EventType.STATE_TRANSITION,
            "runtime",
            {
                "node_id": node_id[:8] if node_id else None,
                "node_type": node_type,
                "from_status": from_status,
                "to_status": to_status,
                "reason": reason
            }
        )

    def log_transition_blocked(
        self,
        node_id: str,
        node_type: str,
        target_status: str,
        reason: str
    ):
        """Log a blocked transition."""
        self._metrics.transitions_blocked += 1

        return self._log_internal(
            EventType.TRANSITION_BLOCKED,
            "runtime",
            {
                "node_id": node_id[:8] if node_id else None,
                "node_type": node_type,
                "target_status": target_status,
                "reason": reason
            }
        )

    def log_agent_start(self, role: str, node_id: str):
        """Log agent starting processing."""
        self._metrics.agent_calls += 1

        return self._log_internal(
            EventType.AGENT_START,
            role,
            {"node_id": node_id[:8] if node_id else None}
        )

    def log_agent_end(
        self,
        role: str,
        node_id: str,
        success: bool,
        cost: float = 0.0,
        duration_ms: float = 0.0
    ):
        """Log agent finishing processing."""
        if not success:
            self._metrics.agent_errors += 1

        self._metrics.total_cost_usd += cost

        return self._log_internal(
            EventType.AGENT_END,
            role,
            {
                "node_id": node_id[:8] if node_id else None,
                "success": success,
                "cost_usd": cost,
                "duration_ms": duration_ms
            }
        )

    def log_agent_error(self, role: str, node_id: str, error: str):
        """Log an agent error."""
        self._metrics.agent_errors += 1

        return self._log_internal(
            EventType.AGENT_ERROR,
            role,
            {
                "node_id": node_id[:8] if node_id else None,
                "error": error[:500]  # Truncate long errors
            }
        )

    def log_iteration(self, iteration: int, nodes_processed: int):
        """Log an iteration."""
        self._metrics.total_iterations = iteration

        return self._log_internal(
            EventType.ITERATION_END,
            "runtime",
            {
                "iteration": iteration,
                "nodes_processed": nodes_processed
            }
        )

    def log_node_created(self, node_id: str, node_type: str):
        """Log node creation."""
        self._metrics.nodes_created += 1

        return self._log_internal(
            EventType.NODE_CREATED,
            "runtime",
            {
                "node_id": node_id[:8] if node_id else None,
                "node_type": node_type
            }
        )

    def log_edge_created(self, source_id: str, target_id: str, edge_type: str):
        """Log edge creation."""
        self._metrics.edges_created += 1

        return self._log_internal(
            EventType.EDGE_CREATED,
            "runtime",
            {
                "source_id": source_id[:8] if source_id else None,
                "target_id": target_id[:8] if target_id else None,
                "edge_type": edge_type
            }
        )

    def log_llm_call(
        self,
        role: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost: float,
        latency_ms: float
    ):
        """Log an LLM API call."""
        self._metrics.llm_calls += 1
        self._metrics.tokens_input += tokens_in
        self._metrics.tokens_output += tokens_out
        self._metrics.total_cost_usd += cost
        self._metrics.llm_latency_total_ms += latency_ms

        return self._log_internal(
            EventType.LLM_CALL_END,
            role,
            {
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost,
                "latency_ms": latency_ms
            }
        )

    def log_verification_result(self, node_id: str, passed: bool):
        """Log a verification result."""
        if passed:
            self._metrics.nodes_verified += 1
        else:
            self._metrics.nodes_failed += 1

        return self._log_internal(
            EventType.STATE_TRANSITION,
            "verifier",
            {
                "node_id": node_id[:8] if node_id else None,
                "verification_passed": passed
            }
        )

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_session_metrics(self) -> SessionMetrics:
        """Get current session metrics."""
        return self._metrics

    def get_session_report(self) -> Dict[str, Any]:
        """Get a complete session report."""
        self._metrics.ended_at = datetime.now(timezone.utc).isoformat()

        return {
            "session_id": self._session_id,
            "metrics": self._metrics.to_dict(),
            "event_count": len(self._events),
            "log_file": str(self._log_file)
        }

    def flush(self):
        """Flush any pending writes and finalize session."""
        self._log_internal(
            EventType.SESSION_END,
            "system",
            {
                "metrics": self._metrics.to_dict()
            }
        )

        logger.info(
            f"Telemetry session complete: {self._metrics.nodes_verified} verified, "
            f"{self._metrics.nodes_failed} failed, ${self._metrics.total_cost_usd:.4f} cost"
        )

    def get_events(self) -> List[TelemetryEvent]:
        """Get all events from this session."""
        return self._events.copy()

    def get_events_by_type(self, event_type: EventType) -> List[TelemetryEvent]:
        """Get events filtered by type."""
        return [e for e in self._events if e.event_type == event_type.value]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_recorder() -> TelemetryRecorder:
    """Get the singleton telemetry recorder."""
    return TelemetryRecorder.get_instance()


def log_transition(node_id: str, node_type: str, from_status: str, to_status: str):
    """Convenience function to log a state transition."""
    return get_recorder().log_state_transition(node_id, node_type, from_status, to_status)


def log_agent(role: str, node_id: str, success: bool, cost: float = 0.0):
    """Convenience function to log agent completion."""
    return get_recorder().log_agent_end(role, node_id, success, cost)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Testing Telemetry Layer ===\n")

    # Reset for fresh test
    TelemetryRecorder.reset_instance()

    recorder = TelemetryRecorder.get_instance()
    print(f"Session ID: {recorder.session_id[:8]}")

    # Log some events
    recorder.log_event("ARCHITECT", "decompose", {"specs": 3})
    recorder.increment_step()
    recorder.log_state_transition("node123", "REQ", "PENDING", "PROCESSING")
    recorder.log_agent_start("BUILDER", "node456")
    recorder.log_llm_call("BUILDER", "claude-3-haiku", 1000, 500, 0.002, 1500.0)
    recorder.log_agent_end("BUILDER", "node456", True, 0.002, 1500.0)
    recorder.log_verification_result("node456", True)
    recorder.increment_step()
    recorder.log_iteration(2, 3)

    # Get report
    report = recorder.get_session_report()
    print("\nSession Report:")
    print(json.dumps(report, indent=2))

    # Flush
    recorder.flush()

    print(f"\n✅ Telemetry test complete. Log file: {recorder._log_file}")
