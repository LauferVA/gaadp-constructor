"""
METRICS EVENT SUBSCRIBER
Automatically collects metrics by subscribing to event bus.
"""
import logging
from typing import Dict, Any

from infrastructure.metrics import MetricsCollector
from infrastructure.event_bus import EventBus
from core.ontology import NodeType, NodeStatus

logger = logging.getLogger("MetricsSubscriber")


class MetricsSubscriber:
    """
    Subscribes to event bus and automatically collects metrics.
    """

    def __init__(self, metrics: MetricsCollector, event_bus: EventBus):
        self.metrics = metrics
        self.event_bus = event_bus
        self._subscribe_to_events()

    def _subscribe_to_events(self):
        """Subscribe to relevant event topics."""
        self.event_bus.subscribe("node_lifecycle", self._handle_node_lifecycle)
        self.event_bus.subscribe("consensus", self._handle_consensus)
        self.event_bus.subscribe("context", self._handle_context)
        self.event_bus.subscribe("performance", self._handle_performance)

        logger.info("MetricsSubscriber: Subscribed to event bus topics")

    def _handle_node_lifecycle(self, event: Dict[str, Any]):
        """Handle node lifecycle events."""
        msg_type = event.get("message_type")
        payload = event.get("payload", {})

        if msg_type == "NODE_CREATED":
            node_id = payload.get("node_id")
            node_type_str = payload.get("node_type")
            token_count = payload.get("token_count")
            agent_id = payload.get("agent_id")
            agent_role = payload.get("agent_role")
            traces_to_req = payload.get("traces_to_req")
            traces_to_spec = payload.get("traces_to_spec")
            implements_spec = payload.get("implements_spec")
            depends_on = payload.get("depends_on")

            if node_id and node_type_str:
                try:
                    node_type = NodeType(node_type_str)
                    self.metrics.record_node_created(
                        node_id, node_type, token_count,
                        agent_id=agent_id,
                        agent_role=agent_role,
                        traces_to_req=traces_to_req,
                        traces_to_spec=traces_to_spec,
                        implements_spec=implements_spec,
                        depends_on=depends_on
                    )
                except ValueError:
                    logger.warning(f"Invalid node type: {node_type_str}")

        elif msg_type == "STATUS_CHANGED":
            node_id = payload.get("node_id")
            old_status_str = payload.get("old_status")
            new_status_str = payload.get("new_status")
            reason = payload.get("reason", "")

            if node_id and old_status_str and new_status_str:
                try:
                    old_status = NodeStatus(old_status_str)
                    new_status = NodeStatus(new_status_str)
                    self.metrics.record_status_change(node_id, old_status, new_status, reason)
                except ValueError as e:
                    logger.warning(f"Invalid status: {e}")

        elif msg_type == "RETRY_ATTEMPT":
            node_id = payload.get("node_id")
            retry_number = payload.get("retry_number")

            if node_id and retry_number is not None:
                self.metrics.record_retry(node_id, retry_number)

    def _handle_consensus(self, event: Dict[str, Any]):
        """Handle consensus events."""
        msg_type = event.get("message_type")
        payload = event.get("payload", {})

        if msg_type == "CONSENSUS_ACHIEVED":
            node_id = payload.get("node_id")
            verdict = payload.get("verdict")
            pass_count = payload.get("pass_count", 0)
            fail_count = payload.get("fail_count", 0)
            verifier_count = payload.get("verifier_count", 0)

            if node_id and verdict:
                self.metrics.record_consensus_result(
                    node_id, verdict, pass_count, fail_count, verifier_count
                )

    def _handle_context(self, event: Dict[str, Any]):
        """Handle context pruning events."""
        msg_type = event.get("message_type")
        payload = event.get("payload", {})

        if msg_type == "CONTEXT_PRUNED":
            nodes_considered = payload.get("nodes_considered", 0)
            nodes_selected = payload.get("nodes_selected", 0)
            token_budget = payload.get("token_budget", 0)
            tokens_used = payload.get("tokens_used", 0)

            self.metrics.record_context_pruning(
                nodes_considered, nodes_selected, token_budget, tokens_used
            )

        elif msg_type == "CONTEXT_SNAPSHOT":
            node_id = payload.get("node_id")
            context_node_ids = payload.get("context_node_ids", [])
            context_token_count = payload.get("context_token_count", 0)
            pruning_ratio = payload.get("pruning_ratio", "")

            if node_id:
                self.metrics.record_context_snapshot(
                    node_id, context_node_ids, context_token_count, pruning_ratio
                )

    def _handle_performance(self, event: Dict[str, Any]):
        """Handle performance events."""
        msg_type = event.get("message_type")
        payload = event.get("payload", {})

        if msg_type == "PROCESSING_COMPLETE":
            node_id = payload.get("node_id")
            processing_time = payload.get("processing_time")
            node_type = payload.get("node_type")

            if node_id and processing_time is not None:
                self.metrics.record_processing_time(node_id, processing_time, node_type)

        elif msg_type == "OPERATION":
            node_id = payload.get("node_id")
            operation = payload.get("operation")
            agent_id = payload.get("agent_id")
            agent_role = payload.get("agent_role")

            if node_id and operation:
                self.metrics.record_operation(node_id, operation, agent_id, agent_role)

        elif msg_type == "MATERIALIZATION":
            node_id = payload.get("node_id")
            commit_hash = payload.get("commit")
            files = payload.get("files", [])
            status = payload.get("status")

            if node_id and status:
                self.metrics.record_materialization(node_id, commit_hash, files, status)
