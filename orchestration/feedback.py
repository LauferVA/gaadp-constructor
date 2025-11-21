"""
FEEDBACK CONTROLLER
Handles reject → replan → rebuild cycles.
"""
import asyncio
import logging
import uuid
from typing import Dict, Optional
from dataclasses import dataclass

from infrastructure.graph_db import GraphDB
from infrastructure.event_bus import EventBus, MessageType
from core.ontology import NodeType, NodeStatus, EdgeType

logger = logging.getLogger("FeedbackController")


@dataclass
class FeedbackConfig:
    """Feedback loop configuration."""
    max_retries: int = 3
    escalate_to_architect_after: int = 2  # After N builder failures, re-plan
    cooldown_seconds: float = 1.0


class FeedbackController:
    """
    Manages the feedback loop when Verifier rejects code.

    Flow:
    1. Verifier marks CODE as FAILED with critique
    2. FeedbackController detects FAILED CODE nodes
    3. Creates FEEDBACK edge from CODE → SPEC with critique
    4. Resets SPEC to PENDING (triggers re-implementation)
    5. Builder sees critique in context via get_context_neighborhood
    6. After max_retries, escalates to Architect for re-planning
    """

    def __init__(
        self,
        db: GraphDB,
        event_bus: EventBus,
        config: Optional[FeedbackConfig] = None
    ):
        self.db = db
        self.event_bus = event_bus
        self.config = config or FeedbackConfig()
        self._running = False
        self._retry_counts: Dict[str, int] = {}  # spec_id → retry count

    def _get_failed_code_nodes(self):
        """Find CODE nodes that failed verification."""
        failed = []
        for node_id, data in self.db.graph.nodes(data=True):
            if (data.get('type') == NodeType.CODE.value and
                data.get('status') == NodeStatus.FAILED.value and
                not data.get('feedback_processed', False)):
                failed.append({
                    'node_id': node_id,
                    'critique': data.get('critique', 'No critique provided'),
                    'metadata': data.get('metadata', {})
                })
        return failed

    def _find_parent_spec(self, code_node_id: str) -> Optional[str]:
        """Find the SPEC node that this CODE implements."""
        for _, target, edge_data in self.db.graph.out_edges(code_node_id, data=True):
            if edge_data.get('type') == EdgeType.IMPLEMENTS.value:
                target_type = self.db.graph.nodes[target].get('type')
                if target_type == NodeType.SPEC.value:
                    return target
        return None

    def _find_parent_req(self, spec_node_id: str) -> Optional[str]:
        """Find the REQ node that this SPEC traces to."""
        for _, target, edge_data in self.db.graph.out_edges(spec_node_id, data=True):
            if edge_data.get('type') == EdgeType.TRACES_TO.value:
                target_type = self.db.graph.nodes[target].get('type')
                if target_type == NodeType.REQ.value:
                    return target
        return None

    async def _process_failure(self, failed_node: Dict):
        """Process a single failed CODE node."""
        code_id = failed_node['node_id']
        critique = failed_node['critique']

        # Find parent SPEC
        spec_id = self._find_parent_spec(code_id)
        if not spec_id:
            logger.warning(f"No parent SPEC found for CODE {code_id}, marking as DEAD_END")
            self.db.graph.nodes[code_id]['type'] = NodeType.DEAD_END.value
            self.db._persist()
            return

        # Track retries
        current_retries = self._retry_counts.get(spec_id, 0)
        self._retry_counts[spec_id] = current_retries + 1

        logger.info(f"Processing failure for {code_id}, SPEC {spec_id} retry #{current_retries + 1}")

        # Emit metrics event for retry
        await self.event_bus.publish(
            topic="node_lifecycle",
            message_type="RETRY_ATTEMPT",
            payload={
                "node_id": spec_id,
                "retry_number": current_retries + 1
            },
            source_id="feedback_controller"
        )

        # Create FEEDBACK edge with critique
        feedback_edge_id = f"feedback_{uuid.uuid4().hex[:8]}"
        self.db.add_edge(
            code_id, spec_id,
            EdgeType.FEEDBACK,
            signed_by="feedback_controller",
            signature=feedback_edge_id
        )

        # Store critique on the edge for context retrieval
        self.db.graph.edges[code_id, spec_id]['critique'] = critique
        self.db.graph.edges[code_id, spec_id]['retry_number'] = current_retries + 1

        # Mark CODE as processed (won't be picked up again)
        self.db.graph.nodes[code_id]['feedback_processed'] = True
        self.db.graph.nodes[code_id]['type'] = NodeType.DEAD_END.value

        # Decide: retry Builder or escalate to Architect
        if current_retries + 1 >= self.config.escalate_to_architect_after:
            # Escalate: Mark SPEC for re-planning
            await self._escalate_to_architect(spec_id, critique, current_retries + 1)
        else:
            # Retry: Reset SPEC to PENDING for Builder retry
            self.db.graph.nodes[spec_id]['status'] = NodeStatus.PENDING.value
            self.db.graph.nodes[spec_id]['last_failure_critique'] = critique
            logger.info(f"Reset SPEC {spec_id} to PENDING for retry")

        self.db._persist()

        # Publish feedback event
        await self.event_bus.publish(
            topic="feedback",
            message_type="FEEDBACK_CREATED",
            payload={
                "code_id": code_id,
                "spec_id": spec_id,
                "critique": critique,
                "retry_count": current_retries + 1
            },
            source_id="feedback_controller"
        )

    async def _escalate_to_architect(self, spec_id: str, critique: str, retry_count: int):
        """
        Escalate to Architect for re-planning after multiple Builder failures.
        Creates a new SPEC node with accumulated context.
        """
        logger.warning(f"Escalating SPEC {spec_id} to Architect after {retry_count} failures")

        # Find parent REQ
        req_id = self._find_parent_req(spec_id)
        if not req_id:
            logger.error(f"No parent REQ found for SPEC {spec_id}, marking as DEAD_END")
            self.db.graph.nodes[spec_id]['type'] = NodeType.DEAD_END.value
            return

        # Mark old SPEC as DEAD_END
        self.db.graph.nodes[spec_id]['type'] = NodeType.DEAD_END.value

        # Reset REQ to PENDING with escalation context
        original_content = self.db.graph.nodes[req_id].get('content', '')
        escalation_context = f"""
[ESCALATION: Previous implementation attempts failed {retry_count} times]

Original Requirement:
{original_content}

Last Failure Critique:
{critique}

Instruction: Re-plan this requirement with a different approach. The previous specification led to implementation failures.
"""
        self.db.graph.nodes[req_id]['escalation_context'] = escalation_context
        self.db.graph.nodes[req_id]['status'] = NodeStatus.PENDING.value

        # Clear retry count for fresh start
        del self._retry_counts[spec_id]

        await self.event_bus.publish(
            topic="feedback",
            message_type="ESCALATED_TO_ARCHITECT",
            payload={
                "spec_id": spec_id,
                "req_id": req_id,
                "retry_count": retry_count,
                "critique": critique
            },
            source_id="feedback_controller"
        )

    async def _feedback_loop(self):
        """Main loop that monitors for failures and processes feedback."""
        while self._running:
            failed_nodes = self._get_failed_code_nodes()

            for failed in failed_nodes:
                try:
                    await self._process_failure(failed)
                except Exception as e:
                    logger.error(f"Error processing failure for {failed['node_id']}: {e}")

            await asyncio.sleep(self.config.cooldown_seconds)

    async def start(self):
        """Start the feedback controller."""
        logger.info("Starting feedback controller")
        self._running = True
        await self._feedback_loop()

    def stop(self):
        """Stop the feedback controller."""
        logger.info("Stopping feedback controller")
        self._running = False

    def get_retry_counts(self) -> Dict[str, int]:
        """Get current retry counts (for MCP queries)."""
        return dict(self._retry_counts)
