"""
METRICS COLLECTOR
Tracks failures, successes, and performance patterns for testing analysis.
Provides instrumentation to understand what works and what doesn't.
"""
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum

from core.ontology import NodeType, NodeStatus

logger = logging.getLogger("Metrics")


class MetricCategory(str, Enum):
    """Categories of metrics."""
    NODE_LIFECYCLE = "NODE_LIFECYCLE"
    VERIFICATION = "VERIFICATION"
    CONSENSUS = "CONSENSUS"
    CONTEXT_PRUNING = "CONTEXT_PRUNING"
    PERFORMANCE = "PERFORMANCE"
    FAILURE = "FAILURE"


@dataclass
class NodeMetric:
    """Metrics for a single node with full traceability."""
    node_id: str
    node_type: str
    status: str
    created_at: str
    status_updated_at: Optional[str] = None
    processing_time: Optional[float] = None
    token_count: Optional[int] = None
    retry_count: int = 0
    failure_reason: Optional[str] = None
    verifier_count: Optional[int] = None
    consensus_verdict: Optional[str] = None

    # Agent attribution
    agent_id: Optional[str] = None
    agent_role: Optional[str] = None  # "architect", "builder", "verifier", etc.
    operation: Optional[str] = None   # "planning", "building", "verifying", "materializing"

    # Traceability chain (graph lineage)
    traces_to_req: Optional[str] = None      # REQ node this traces back to
    traces_to_spec: Optional[str] = None     # SPEC node this implements
    implements_spec: Optional[str] = None     # Direct IMPLEMENTS edge target
    depends_on: List[str] = field(default_factory=list)  # DEPENDS_ON nodes

    # Context snapshot
    context_node_ids: List[str] = field(default_factory=list)  # Nodes in context
    context_token_count: Optional[int] = None
    context_pruning_ratio: Optional[str] = None

    # Git linkage
    materialized_commit: Optional[str] = None
    materialized_files: List[str] = field(default_factory=list)
    materialization_status: Optional[str] = None  # "success", "failed", "pending"


@dataclass
class FailurePattern:
    """Aggregated failure pattern with traceability."""
    category: str
    count: int
    node_types: List[str] = field(default_factory=list)
    common_reasons: List[str] = field(default_factory=list)
    avg_retry_count: float = 0.0
    examples: List[str] = field(default_factory=list)

    # Agent attribution
    agent_roles: List[str] = field(default_factory=list)  # Which agents hit this failure
    operations: List[str] = field(default_factory=list)   # During which operations

    # Context patterns
    avg_context_size: float = 0.0
    context_token_ranges: str = ""  # e.g., "1000-2000 tokens"


@dataclass
class SuccessPattern:
    """Aggregated success pattern."""
    node_type: str
    count: int
    avg_processing_time: float = 0.0
    avg_token_count: float = 0.0
    avg_verifier_count: float = 0.0


class MetricsCollector:
    """
    Collects and analyzes execution metrics for testing and debugging.

    Features:
    - Node lifecycle tracking (creation → verification → success/failure)
    - Failure pattern analysis (by type, reason, retry count)
    - Success rate tracking (by node type, domain)
    - Performance metrics (processing time, token usage)
    - Consensus effectiveness (agreement rates, verifier patterns)
    - Context pruning efficiency
    """

    def __init__(self):
        # Node-level metrics
        self.node_metrics: Dict[str, NodeMetric] = {}

        # Aggregated counters
        self.status_transitions: Dict[str, int] = defaultdict(int)
        self.node_type_counts: Dict[str, int] = defaultdict(int)
        self.failure_reasons: Dict[str, int] = defaultdict(int)
        self.retry_counts: List[int] = []

        # Performance tracking
        self.processing_times: Dict[str, List[float]] = defaultdict(list)
        self.token_usage: Dict[str, List[int]] = defaultdict(list)

        # Verification/Consensus tracking
        self.consensus_verdicts: Dict[str, int] = defaultdict(int)
        self.verifier_agreement_rates: List[float] = []

        # Context pruning tracking
        self.pruning_ratios: List[float] = []
        self.context_sizes: List[int] = []

        # Temporal tracking
        self.session_start = datetime.utcnow().isoformat()
        self.event_log: List[Dict] = []

    def record_node_created(
        self,
        node_id: str,
        node_type: NodeType,
        token_count: Optional[int] = None,
        agent_id: Optional[str] = None,
        agent_role: Optional[str] = None,
        traces_to_req: Optional[str] = None,
        traces_to_spec: Optional[str] = None,
        implements_spec: Optional[str] = None,
        depends_on: Optional[List[str]] = None
    ):
        """Record node creation with full traceability."""
        metric = NodeMetric(
            node_id=node_id,
            node_type=node_type.value,
            status=NodeStatus.PENDING.value,
            created_at=datetime.utcnow().isoformat(),
            token_count=token_count,
            agent_id=agent_id,
            agent_role=agent_role,
            traces_to_req=traces_to_req,
            traces_to_spec=traces_to_spec,
            implements_spec=implements_spec,
            depends_on=depends_on or []
        )
        self.node_metrics[node_id] = metric
        self.node_type_counts[node_type.value] += 1

        self._log_event("NODE_CREATED", {
            "node_id": node_id,
            "node_type": node_type.value,
            "agent_id": agent_id,
            "agent_role": agent_role
        })

    def record_status_change(
        self,
        node_id: str,
        old_status: NodeStatus,
        new_status: NodeStatus,
        reason: str = ""
    ):
        """Record node status transition."""
        if node_id not in self.node_metrics:
            logger.warning(f"Status change for untracked node: {node_id}")
            return

        metric = self.node_metrics[node_id]
        metric.status = new_status.value
        metric.status_updated_at = datetime.utcnow().isoformat()

        transition_key = f"{old_status.value}→{new_status.value}"
        self.status_transitions[transition_key] += 1

        # Track failures
        if new_status == NodeStatus.FAILED:
            metric.failure_reason = reason
            self.failure_reasons[reason] += 1

        self._log_event("STATUS_CHANGE", {
            "node_id": node_id,
            "transition": transition_key,
            "reason": reason
        })

    def record_retry(self, node_id: str, retry_number: int):
        """Record a retry attempt."""
        if node_id in self.node_metrics:
            self.node_metrics[node_id].retry_count = retry_number
            self.retry_counts.append(retry_number)

        self._log_event("RETRY", {
            "node_id": node_id,
            "retry_number": retry_number
        })

    def record_processing_time(
        self,
        node_id: str,
        processing_time: float,
        node_type: Optional[str] = None
    ):
        """Record node processing time."""
        if node_id in self.node_metrics:
            self.node_metrics[node_id].processing_time = processing_time
            node_type = self.node_metrics[node_id].node_type

        if node_type:
            self.processing_times[node_type].append(processing_time)

        self._log_event("PROCESSING_TIME", {
            "node_id": node_id,
            "time_seconds": processing_time
        })

    def record_consensus_result(
        self,
        node_id: str,
        verdict: str,
        pass_count: int,
        fail_count: int,
        verifier_count: int
    ):
        """Record consensus verification result."""
        if node_id in self.node_metrics:
            self.node_metrics[node_id].consensus_verdict = verdict
            self.node_metrics[node_id].verifier_count = verifier_count

        self.consensus_verdicts[verdict] += 1

        # Calculate agreement rate
        if verifier_count > 0:
            agreement_rate = max(pass_count, fail_count) / verifier_count
            self.verifier_agreement_rates.append(agreement_rate)

        self._log_event("CONSENSUS", {
            "node_id": node_id,
            "verdict": verdict,
            "pass_count": pass_count,
            "fail_count": fail_count
        })

    def record_context_pruning(
        self,
        nodes_considered: int,
        nodes_selected: int,
        token_budget: int,
        tokens_used: int
    ):
        """Record context pruning effectiveness."""
        if nodes_considered > 0:
            pruning_ratio = nodes_selected / nodes_considered
            self.pruning_ratios.append(pruning_ratio)

        self.context_sizes.append(nodes_selected)

        self._log_event("CONTEXT_PRUNING", {
            "nodes_considered": nodes_considered,
            "nodes_selected": nodes_selected,
            "pruning_ratio": f"{nodes_selected}/{nodes_considered}",
            "token_usage": f"{tokens_used}/{token_budget}"
        })

    def record_operation(
        self,
        node_id: str,
        operation: str,
        agent_id: Optional[str] = None,
        agent_role: Optional[str] = None
    ):
        """Record the operation being performed on a node."""
        if node_id in self.node_metrics:
            self.node_metrics[node_id].operation = operation
            if agent_id:
                self.node_metrics[node_id].agent_id = agent_id
            if agent_role:
                self.node_metrics[node_id].agent_role = agent_role

        self._log_event("OPERATION", {
            "node_id": node_id,
            "operation": operation,
            "agent_id": agent_id,
            "agent_role": agent_role
        })

    def record_context_snapshot(
        self,
        node_id: str,
        context_node_ids: List[str],
        context_token_count: int,
        pruning_ratio: str
    ):
        """Record the context snapshot used for a node."""
        if node_id in self.node_metrics:
            self.node_metrics[node_id].context_node_ids = context_node_ids
            self.node_metrics[node_id].context_token_count = context_token_count
            self.node_metrics[node_id].context_pruning_ratio = pruning_ratio

        self._log_event("CONTEXT_SNAPSHOT", {
            "node_id": node_id,
            "context_size": len(context_node_ids),
            "token_count": context_token_count
        })

    def record_materialization(
        self,
        node_id: str,
        commit_hash: Optional[str],
        files: List[str],
        status: str
    ):
        """Record git materialization result."""
        if node_id in self.node_metrics:
            self.node_metrics[node_id].materialized_commit = commit_hash
            self.node_metrics[node_id].materialized_files = files
            self.node_metrics[node_id].materialization_status = status

        self._log_event("MATERIALIZATION", {
            "node_id": node_id,
            "commit": commit_hash,
            "file_count": len(files),
            "status": status
        })

    def update_traceability(
        self,
        node_id: str,
        traces_to_req: Optional[str] = None,
        traces_to_spec: Optional[str] = None,
        implements_spec: Optional[str] = None,
        depends_on: Optional[List[str]] = None
    ):
        """Update traceability information for a node (can be called after creation)."""
        if node_id not in self.node_metrics:
            logger.warning(f"Traceability update for untracked node: {node_id}")
            return

        metric = self.node_metrics[node_id]
        if traces_to_req:
            metric.traces_to_req = traces_to_req
        if traces_to_spec:
            metric.traces_to_spec = traces_to_spec
        if implements_spec:
            metric.implements_spec = implements_spec
        if depends_on:
            metric.depends_on = depends_on

    def _log_event(self, event_type: str, data: Dict):
        """Log event for temporal analysis."""
        self.event_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "data": data
        })

    def get_failure_patterns(self) -> List[FailurePattern]:
        """Analyze and return failure patterns."""
        patterns = []

        # Group failures by reason
        reason_groups = defaultdict(list)
        for node_id, metric in self.node_metrics.items():
            if metric.status == NodeStatus.FAILED.value and metric.failure_reason:
                reason_groups[metric.failure_reason].append(metric)

        # Create pattern summaries
        for reason, metrics in reason_groups.items():
            node_types = [m.node_type for m in metrics]
            retry_counts = [m.retry_count for m in metrics if m.retry_count > 0]
            agent_roles = list(set([m.agent_role for m in metrics if m.agent_role]))
            operations = list(set([m.operation for m in metrics if m.operation]))
            context_sizes = [len(m.context_node_ids) for m in metrics if m.context_node_ids]
            context_tokens = [m.context_token_count for m in metrics if m.context_token_count]

            # Calculate token range
            if context_tokens:
                min_tokens = min(context_tokens)
                max_tokens = max(context_tokens)
                token_range = f"{min_tokens}-{max_tokens} tokens"
            else:
                token_range = "N/A"

            patterns.append(FailurePattern(
                category=reason,
                count=len(metrics),
                node_types=list(set(node_types)),
                common_reasons=[reason],
                avg_retry_count=sum(retry_counts) / len(retry_counts) if retry_counts else 0,
                examples=[m.node_id for m in metrics[:3]],
                agent_roles=agent_roles,
                operations=operations,
                avg_context_size=sum(context_sizes) / len(context_sizes) if context_sizes else 0,
                context_token_ranges=token_range
            ))

        return sorted(patterns, key=lambda x: x.count, reverse=True)

    def get_success_patterns(self) -> List[SuccessPattern]:
        """Analyze and return success patterns."""
        patterns = []

        # Group successes by node type
        type_groups = defaultdict(list)
        for node_id, metric in self.node_metrics.items():
            if metric.status == NodeStatus.VERIFIED.value:
                type_groups[metric.node_type].append(metric)

        # Create pattern summaries
        for node_type, metrics in type_groups.items():
            processing_times = [m.processing_time for m in metrics if m.processing_time]
            token_counts = [m.token_count for m in metrics if m.token_count]
            verifier_counts = [m.verifier_count for m in metrics if m.verifier_count]

            patterns.append(SuccessPattern(
                node_type=node_type,
                count=len(metrics),
                avg_processing_time=sum(processing_times) / len(processing_times) if processing_times else 0,
                avg_token_count=sum(token_counts) / len(token_counts) if token_counts else 0,
                avg_verifier_count=sum(verifier_counts) / len(verifier_counts) if verifier_counts else 0
            ))

        return sorted(patterns, key=lambda x: x.count, reverse=True)

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics summary."""
        total_nodes = len(self.node_metrics)

        # Status breakdown
        status_counts = defaultdict(int)
        for metric in self.node_metrics.values():
            status_counts[metric.status] += 1

        # Success rate by node type
        success_rates = {}
        for node_type in set(m.node_type for m in self.node_metrics.values()):
            type_metrics = [m for m in self.node_metrics.values() if m.node_type == node_type]
            verified = sum(1 for m in type_metrics if m.status == NodeStatus.VERIFIED.value)
            total = len(type_metrics)
            success_rates[node_type] = verified / total if total > 0 else 0

        # Average metrics
        all_processing_times = [m.processing_time for m in self.node_metrics.values() if m.processing_time]
        all_token_counts = [m.token_count for m in self.node_metrics.values() if m.token_count]

        return {
            "session_start": self.session_start,
            "total_nodes": total_nodes,
            "status_breakdown": dict(status_counts),
            "success_rates_by_type": success_rates,
            "status_transitions": dict(self.status_transitions),
            "failure_reasons": dict(sorted(
                self.failure_reasons.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            "retry_stats": {
                "total_retries": len(self.retry_counts),
                "avg_retries": sum(self.retry_counts) / len(self.retry_counts) if self.retry_counts else 0,
                "max_retries": max(self.retry_counts) if self.retry_counts else 0
            },
            "performance": {
                "avg_processing_time": sum(all_processing_times) / len(all_processing_times) if all_processing_times else 0,
                "avg_token_count": sum(all_token_counts) / len(all_token_counts) if all_token_counts else 0
            },
            "consensus": {
                "verdicts": dict(self.consensus_verdicts),
                "avg_agreement_rate": sum(self.verifier_agreement_rates) / len(self.verifier_agreement_rates) if self.verifier_agreement_rates else 0
            },
            "context_pruning": {
                "avg_pruning_ratio": sum(self.pruning_ratios) / len(self.pruning_ratios) if self.pruning_ratios else 0,
                "avg_context_size": sum(self.context_sizes) / len(self.context_sizes) if self.context_sizes else 0
            },
            "failure_patterns": [asdict(p) for p in self.get_failure_patterns()],
            "success_patterns": [asdict(p) for p in self.get_success_patterns()]
        }

    def export_detailed_report(self, filepath: str):
        """Export detailed metrics to JSON file."""
        import json

        report = {
            "summary": self.get_summary_report(),
            "node_metrics": {
                node_id: asdict(metric)
                for node_id, metric in self.node_metrics.items()
            },
            "event_log": self.event_log
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Metrics exported to {filepath}")

    def get_node_metric(self, node_id: str) -> Optional[NodeMetric]:
        """Get metrics for a specific node."""
        return self.node_metrics.get(node_id)

    def query_metrics(
        self,
        node_type: Optional[str] = None,
        status: Optional[str] = None,
        has_retries: Optional[bool] = None
    ) -> List[NodeMetric]:
        """Query metrics with filters."""
        results = list(self.node_metrics.values())

        if node_type:
            results = [m for m in results if m.node_type == node_type]

        if status:
            results = [m for m in results if m.status == status]

        if has_retries is not None:
            if has_retries:
                results = [m for m in results if m.retry_count > 0]
            else:
                results = [m for m in results if m.retry_count == 0]

        return results

    def query_by_traceability(
        self,
        req_id: Optional[str] = None,
        spec_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[NodeMetric]:
        """Query metrics by traceability chain."""
        results = list(self.node_metrics.values())

        if req_id:
            results = [m for m in results if m.traces_to_req == req_id]

        if spec_id:
            results = [m for m in results if m.traces_to_spec == spec_id or m.implements_spec == spec_id]

        if status:
            results = [m for m in results if m.status == status]

        return results

    def query_by_agent(
        self,
        agent_role: Optional[str] = None,
        operation: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[NodeMetric]:
        """Query metrics by agent and operation."""
        results = list(self.node_metrics.values())

        if agent_role:
            results = [m for m in results if m.agent_role == agent_role]

        if operation:
            results = [m for m in results if m.operation == operation]

        if status:
            results = [m for m in results if m.status == status]

        return results

    def get_traceability_report(self, req_id: str) -> Dict[str, Any]:
        """Get full traceability report for a requirement."""
        metrics = self.query_by_traceability(req_id=req_id)

        if not metrics:
            return {"error": f"No metrics found for requirement {req_id}"}

        # Group by node type
        by_type = defaultdict(list)
        for m in metrics:
            by_type[m.node_type].append(m)

        # Calculate stats
        total = len(metrics)
        verified = sum(1 for m in metrics if m.status == NodeStatus.VERIFIED.value)
        failed = sum(1 for m in metrics if m.status == NodeStatus.FAILED.value)
        pending = sum(1 for m in metrics if m.status == NodeStatus.PENDING.value)

        return {
            "requirement_id": req_id,
            "total_nodes": total,
            "status_breakdown": {
                "verified": verified,
                "failed": failed,
                "pending": pending
            },
            "by_node_type": {
                node_type: len(nodes) for node_type, nodes in by_type.items()
            },
            "failures": [
                {
                    "node_id": m.node_id,
                    "node_type": m.node_type,
                    "reason": m.failure_reason,
                    "agent_role": m.agent_role,
                    "operation": m.operation,
                    "retry_count": m.retry_count
                }
                for m in metrics if m.status == NodeStatus.FAILED.value
            ],
            "successes": [
                {
                    "node_id": m.node_id,
                    "node_type": m.node_type,
                    "processing_time": m.processing_time,
                    "verifier_count": m.verifier_count,
                    "materialized_commit": m.materialized_commit
                }
                for m in metrics if m.status == NodeStatus.VERIFIED.value
            ]
        }

    def reset(self):
        """Clear all metrics (for testing)."""
        self.__init__()
