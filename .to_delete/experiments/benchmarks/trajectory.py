"""
GAADP TRAJECTORY ANALYSIS - Execution Path Measurement
=======================================================
Extracts signal from telemetry to understand how the system behaved
during execution, not just what it produced.

Metrics:
    - Oscillation Count: State flip-flops indicating instability
    - Token Efficiency Ratio: Useful tokens vs wasted tokens
    - Cost Per Verified Node: Economic efficiency
    - Iteration Efficiency: Actual vs optimal iteration count

Usage:
    from benchmarks.trajectory import TrajectoryAnalyzer

    analyzer = TrajectoryAnalyzer(".gaadp/logs/telemetry.jsonl")
    report = analyzer.analyze()
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger("GAADP.Trajectory")


@dataclass
class OscillationResult:
    """State oscillation analysis result."""
    total_oscillations: int
    oscillating_nodes: List[Dict[str, Any]]  # Nodes that oscillated and their patterns
    max_oscillations_per_node: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenEfficiencyResult:
    """Token efficiency analysis result."""
    ratio: float  # tokens_in_verified / tokens_total
    tokens_total: int
    tokens_in_verified: int
    tokens_wasted: int  # In failed/retried artifacts
    tokens_by_role: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CostEfficiencyResult:
    """Cost efficiency analysis result."""
    cost_per_verified_node: float
    total_cost: float
    verified_nodes: int
    cost_by_role: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IterationEfficiencyResult:
    """Iteration efficiency analysis result."""
    ratio: float  # optimal / actual
    actual_iterations: int
    optimal_iterations: int  # = unique nodes processed
    nodes_processed_multiple_times: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryReport:
    """Complete trajectory analysis report."""
    session_id: str
    timestamp: str
    duration_seconds: float

    # Trajectory Metrics
    oscillations: OscillationResult
    token_efficiency: TokenEfficiencyResult
    cost_efficiency: CostEfficiencyResult
    iteration_efficiency: IterationEfficiencyResult

    # Raw stats
    total_events: int
    state_transitions: int
    llm_calls: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "metrics": {
                "oscillation_count": self.oscillations.total_oscillations,
                "token_efficiency_ratio": self.token_efficiency.ratio,
                "cost_per_verified_node": self.cost_efficiency.cost_per_verified_node,
                "iteration_efficiency": self.iteration_efficiency.ratio
            },
            "details": {
                "oscillations": self.oscillations.to_dict(),
                "token_efficiency": self.token_efficiency.to_dict(),
                "cost_efficiency": self.cost_efficiency.to_dict(),
                "iteration_efficiency": self.iteration_efficiency.to_dict()
            },
            "raw": {
                "total_events": self.total_events,
                "state_transitions": self.state_transitions,
                "llm_calls": self.llm_calls
            }
        }


class TrajectoryAnalyzer:
    """
    Analyze telemetry data to understand execution trajectory.

    Reads from telemetry.jsonl and extracts patterns that indicate
    system health and efficiency.
    """

    def __init__(self, telemetry_path: str):
        """
        Initialize analyzer with path to telemetry file.

        Args:
            telemetry_path: Path to telemetry.jsonl file
        """
        self.telemetry_path = Path(telemetry_path)
        self._events = None
        self._session_id = None

    def _load_events(self) -> List[Dict]:
        """Load all events from telemetry file."""
        if self._events is not None:
            return self._events

        if not self.telemetry_path.exists():
            logger.warning(f"Telemetry file not found: {self.telemetry_path}")
            return []

        events = []
        with open(self.telemetry_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        self._events = events

        # Extract session ID from first event
        if events:
            self._session_id = events[0].get("session_id", "unknown")

        return events

    def analyze_oscillations(self) -> OscillationResult:
        """
        Detect state oscillations in the execution.

        An oscillation is when a node goes:
        PENDING -> BUILDING -> PENDING -> BUILDING (or similar patterns)

        This indicates the system is thrashing or retrying excessively.
        """
        events = self._load_events()

        # Track state history per node
        node_states: Dict[str, List[str]] = defaultdict(list)

        for event in events:
            if event.get("event_type") == "state_transition":
                payload = event.get("payload", {})
                node_id = payload.get("node_id")
                to_status = payload.get("to_status")

                if node_id and to_status:
                    node_states[node_id].append(to_status)

        # Detect oscillations
        oscillating_nodes = []
        total_oscillations = 0
        max_per_node = 0

        for node_id, states in node_states.items():
            oscillation_count = self._count_oscillations(states)
            if oscillation_count > 0:
                oscillating_nodes.append({
                    "node": node_id,
                    "oscillations": oscillation_count,
                    "state_sequence": states
                })
                total_oscillations += oscillation_count
                max_per_node = max(max_per_node, oscillation_count)

        return OscillationResult(
            total_oscillations=total_oscillations,
            oscillating_nodes=oscillating_nodes,
            max_oscillations_per_node=max_per_node
        )

    def _count_oscillations(self, states: List[str]) -> int:
        """
        Count oscillations in a state sequence.

        An oscillation is defined as going back to a previous state
        that should be "before" the current state in the lifecycle.
        """
        # Define the expected forward progression
        FORWARD_ORDER = ["PENDING", "PROCESSING", "BUILDING", "VERIFYING", "VERIFIED"]

        oscillations = 0
        for i in range(1, len(states)):
            prev = states[i - 1]
            curr = states[i]

            # Skip if either state not in our tracking list
            if prev not in FORWARD_ORDER or curr not in FORWARD_ORDER:
                continue

            prev_idx = FORWARD_ORDER.index(prev)
            curr_idx = FORWARD_ORDER.index(curr)

            # Going backwards (except for VERIFIED which is terminal)
            if curr_idx < prev_idx and prev != "VERIFIED":
                oscillations += 1

        return oscillations

    def analyze_token_efficiency(self) -> TokenEfficiencyResult:
        """
        Calculate token efficiency.

        TER = tokens_in_verified_artifacts / tokens_total

        We estimate tokens in verified artifacts by tracking which
        LLM calls produced nodes that eventually became VERIFIED.
        """
        events = self._load_events()

        # Track tokens per role
        tokens_by_role: Dict[str, int] = defaultdict(int)
        total_tokens = 0

        # Track which nodes got verified
        verified_nodes: set = set()
        node_tokens: Dict[str, int] = defaultdict(int)

        for event in events:
            event_type = event.get("event_type")
            payload = event.get("payload", {})
            role = event.get("role", "unknown")

            if event_type == "llm_call_end":
                tokens_in = payload.get("tokens_in", 0)
                tokens_out = payload.get("tokens_out", 0)
                tokens = tokens_in + tokens_out

                total_tokens += tokens
                tokens_by_role[role] += tokens

            elif event_type == "state_transition":
                node_id = payload.get("node_id")
                to_status = payload.get("to_status")

                if to_status == "VERIFIED" and node_id:
                    verified_nodes.add(node_id)

            elif event_type == "agent_end":
                node_id = payload.get("node_id")
                # Rough estimate: associate agent's tokens with node
                # This is imperfect but gives a signal

        # Calculate tokens in verified vs wasted
        # Since we can't perfectly track token-to-node mapping,
        # we use verification rate as a proxy
        verification_events = [
            e for e in events
            if e.get("event_type") == "state_transition"
            and e.get("payload", {}).get("to_status") == "VERIFIED"
        ]
        failure_events = [
            e for e in events
            if e.get("event_type") == "state_transition"
            and e.get("payload", {}).get("to_status") == "FAILED"
        ]

        total_outcomes = len(verification_events) + len(failure_events)
        verification_rate = len(verification_events) / total_outcomes if total_outcomes > 0 else 0

        tokens_in_verified = int(total_tokens * verification_rate)
        tokens_wasted = total_tokens - tokens_in_verified

        ratio = tokens_in_verified / total_tokens if total_tokens > 0 else 0.0

        return TokenEfficiencyResult(
            ratio=ratio,
            tokens_total=total_tokens,
            tokens_in_verified=tokens_in_verified,
            tokens_wasted=tokens_wasted,
            tokens_by_role=dict(tokens_by_role)
        )

    def analyze_cost_efficiency(self) -> CostEfficiencyResult:
        """
        Calculate cost efficiency.

        CPVN = total_cost / verified_nodes
        """
        events = self._load_events()

        total_cost = 0.0
        cost_by_role: Dict[str, float] = defaultdict(float)
        verified_count = 0

        for event in events:
            event_type = event.get("event_type")
            payload = event.get("payload", {})
            role = event.get("role", "unknown")

            if event_type == "llm_call_end":
                cost = payload.get("cost_usd", 0.0)
                total_cost += cost
                cost_by_role[role] += cost

            elif event_type == "state_transition":
                if payload.get("to_status") == "VERIFIED":
                    verified_count += 1

        cpvn = total_cost / verified_count if verified_count > 0 else float('inf')

        return CostEfficiencyResult(
            cost_per_verified_node=cpvn,
            total_cost=total_cost,
            verified_nodes=verified_count,
            cost_by_role=dict(cost_by_role)
        )

    def analyze_iteration_efficiency(self) -> IterationEfficiencyResult:
        """
        Calculate iteration efficiency.

        IE = optimal_iterations / actual_iterations

        Optimal = number of unique nodes (if each node processed once)
        Actual = total iteration count from telemetry
        """
        events = self._load_events()

        # Find max iteration from events
        max_iteration = 0
        nodes_seen: set = set()
        nodes_processed_counts: Dict[str, int] = defaultdict(int)

        for event in events:
            step = event.get("step", 0)
            max_iteration = max(max_iteration, step)

            # Count how many times each node was processed
            if event.get("event_type") == "agent_start":
                node_id = event.get("payload", {}).get("node_id")
                if node_id:
                    nodes_seen.add(node_id)
                    nodes_processed_counts[node_id] += 1

        # Get iteration count from iteration_end events
        iteration_events = [
            e for e in events if e.get("event_type") == "iteration_end"
        ]
        actual_iterations = len(iteration_events) if iteration_events else max_iteration

        # Optimal = unique nodes
        optimal_iterations = len(nodes_seen)

        # Count nodes processed multiple times
        multiple_processed = sum(1 for count in nodes_processed_counts.values() if count > 1)

        ratio = optimal_iterations / actual_iterations if actual_iterations > 0 else 1.0

        return IterationEfficiencyResult(
            ratio=min(ratio, 1.0),  # Cap at 1.0
            actual_iterations=actual_iterations,
            optimal_iterations=optimal_iterations,
            nodes_processed_multiple_times=multiple_processed
        )

    def analyze(self) -> TrajectoryReport:
        """
        Run all trajectory analyses and return complete report.

        Returns:
            TrajectoryReport with all metrics
        """
        events = self._load_events()

        # Get session timing
        start_time = None
        end_time = None

        for event in events:
            ts = event.get("timestamp")
            if ts:
                if start_time is None:
                    start_time = ts
                end_time = ts

        duration = 0.0
        if start_time and end_time:
            try:
                start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                duration = (end_dt - start_dt).total_seconds()
            except Exception:
                pass

        # Count event types
        state_transitions = sum(
            1 for e in events if e.get("event_type") == "state_transition"
        )
        llm_calls = sum(
            1 for e in events if e.get("event_type") == "llm_call_end"
        )

        return TrajectoryReport(
            session_id=self._session_id or "unknown",
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            oscillations=self.analyze_oscillations(),
            token_efficiency=self.analyze_token_efficiency(),
            cost_efficiency=self.analyze_cost_efficiency(),
            iteration_efficiency=self.analyze_iteration_efficiency(),
            total_events=len(events),
            state_transitions=state_transitions,
            llm_calls=llm_calls
        )


def print_trajectory_report(report: TrajectoryReport):
    """Pretty-print a trajectory report."""
    print("\n" + "=" * 60)
    print("TRAJECTORY REPORT")
    print("=" * 60)
    print(f"Session: {report.session_id[:8] if report.session_id else 'unknown'}")
    print(f"Duration: {report.duration_seconds:.1f}s")
    print()

    print("Trajectory Metrics:")
    print(f"  Oscillations: {report.oscillations.total_oscillations}")
    print(f"  Token Efficiency: {report.token_efficiency.ratio:.3f}")
    print(f"  Cost/Verified Node: ${report.cost_efficiency.cost_per_verified_node:.4f}")
    print(f"  Iteration Efficiency: {report.iteration_efficiency.ratio:.3f}")

    print()
    print("Raw Stats:")
    print(f"  Total events: {report.total_events}")
    print(f"  State transitions: {report.state_transitions}")
    print(f"  LLM calls: {report.llm_calls}")
    print(f"  Total cost: ${report.cost_efficiency.total_cost:.4f}")
    print(f"  Verified nodes: {report.cost_efficiency.verified_nodes}")

    if report.oscillations.oscillating_nodes:
        print("\nOscillating Nodes:")
        for node in report.oscillations.oscillating_nodes[:3]:
            print(f"  - {node['node']}: {node['oscillations']} oscillations")
            print(f"    Path: {' -> '.join(node['state_sequence'][:8])}")

    if report.token_efficiency.tokens_by_role:
        print("\nTokens by Role:")
        for role, tokens in sorted(
            report.token_efficiency.tokens_by_role.items(),
            key=lambda x: -x[1]
        )[:5]:
            print(f"  {role}: {tokens:,}")

    print("=" * 60)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_telemetry(telemetry_path: str) -> Dict[str, Any]:
    """
    Convenience function to analyze telemetry.

    Args:
        telemetry_path: Path to telemetry JSONL file

    Returns:
        Dict with trajectory metrics
    """
    analyzer = TrajectoryAnalyzer(telemetry_path)
    report = analyzer.analyze()
    return report.to_dict()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO)

    print("=== Testing Trajectory Analyzer ===\n")

    telemetry_path = ".gaadp/logs/telemetry.jsonl"
    if Path(telemetry_path).exists():
        analyzer = TrajectoryAnalyzer(telemetry_path)
        report = analyzer.analyze()
        print_trajectory_report(report)

        # Save report
        report_path = Path("reports") / f"trajectory_{report.session_id[:8]}.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved: {report_path}")
    else:
        print(f"No telemetry found at {telemetry_path}")
        print("Run a GAADP task first to generate telemetry.")
