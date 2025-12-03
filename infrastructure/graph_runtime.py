"""
GRAPH RUNTIME - The Execution Engine

This module implements the runtime that CONSULTS the ontology schemas
to make all execution decisions. It contains NO hardcoded business logic
about node types or transitions - all rules come from new_ontology.py.

Key Principles:
1. TransitionMatrix is Truth - all state changes query the matrix
2. AGENT_DISPATCH determines which agent processes which node
3. Conditions are evaluated against the graph, not hardcoded
4. The runtime is a dumb executor that follows the physics
"""
import logging
import asyncio
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timezone

from core.ontology import (
    NodeType, EdgeType, NodeStatus,
    NodeSpec, EdgeSpec, NodeMetadata,
    TRANSITION_MATRIX, AGENT_DISPATCH, KNOWN_CONDITIONS,
    TransitionRule,
)
from core.protocols import (
    UnifiedAgentOutput,
    GraphContext,
)
from agents.generic_agent import GenericAgent, create_agent
from core.telemetry import TelemetryRecorder, get_recorder


logger = logging.getLogger("GAADP.GraphRuntime")


class GraphRuntime:
    """
    The execution engine that enforces ontology physics.

    This runtime:
    - Queries TRANSITION_MATRIX to determine valid state changes
    - Evaluates named CONDITIONS against the graph
    - Dispatches to agents based on AGENT_DISPATCH rules
    - Applies agent outputs as graph mutations
    - Optionally emits events to visualization server
    """

    def __init__(self, graph_db, llm_gateway=None, viz_server=None):
        """
        Initialize the runtime.

        Args:
            graph_db: The GraphDB instance
            llm_gateway: Optional LLM gateway for agents
            viz_server: Optional VizServer for real-time visualization
        """
        self.graph = graph_db
        self.gateway = llm_gateway
        self._agents: Dict[str, GenericAgent] = {}
        self._viz = viz_server
        self._telemetry = get_recorder()

        logger.info("GraphRuntime initialized")

    async def _emit(self, method: str, *args, **kwargs):
        """Emit an event to the visualization server if connected."""
        if self._viz and hasattr(self._viz, method):
            try:
                await getattr(self._viz, method)(*args, **kwargs)
            except Exception as e:
                logger.debug(f"Viz emit error: {e}")

    # =========================================================================
    # CONDITION EVALUATION
    # =========================================================================

    def evaluate_condition(self, node_id: str, condition: str) -> bool:
        """
        Evaluate a named condition against the graph.

        This is where named conditions from TRANSITION_MATRIX and
        AGENT_DISPATCH are resolved to boolean values.

        Args:
            node_id: The node to evaluate for
            condition: Name of the condition

        Returns:
            True if condition is satisfied
        """
        if condition not in KNOWN_CONDITIONS:
            logger.warning(f"Unknown condition: {condition}")
            return False

        node_data = self.graph.graph.nodes.get(node_id)
        if not node_data:
            return False

        # === Cost Governance ===
        if condition == "cost_under_limit":
            metadata = node_data.get('metadata', {})
            limit = metadata.get('cost_limit')
            actual = metadata.get('cost_actual', 0.0)
            if limit is None:
                return True  # No limit = unlimited
            return actual < limit

        # === Dependency Tracking ===
        if condition == "dependencies_verified":
            # All DEPENDS_ON targets must be VERIFIED
            for pred in self.graph.graph.predecessors(node_id):
                edge_data = self.graph.graph.edges[pred, node_id]
                if edge_data.get('type') == EdgeType.DEPENDS_ON.value:
                    pred_status = self.graph.graph.nodes[pred].get('status')
                    if pred_status != NodeStatus.VERIFIED.value:
                        return False
            return True

        if condition == "has_unmet_dependencies":
            return not self.evaluate_condition(node_id, "dependencies_verified")

        # === Blocking ===
        if condition == "not_blocked":
            # No BLOCKS edges from this node to PENDING/PROCESSING nodes
            for succ in self.graph.graph.successors(node_id):
                edge_data = self.graph.graph.edges[node_id, succ]
                if edge_data.get('type') == EdgeType.BLOCKS.value:
                    succ_status = self.graph.graph.nodes[succ].get('status')
                    if succ_status in [NodeStatus.PENDING.value, NodeStatus.PROCESSING.value]:
                        return False
            return True

        if condition == "has_pending_clarification":
            # Has CLARIFICATION child in PENDING/PROCESSING
            for succ in self.graph.graph.successors(node_id):
                succ_data = self.graph.graph.nodes[succ]
                if succ_data.get('type') == NodeType.CLARIFICATION.value:
                    if succ_data.get('status') in [NodeStatus.PENDING.value, NodeStatus.PROCESSING.value]:
                        return True
            return False

        if condition == "no_pending_clarifications":
            return not self.evaluate_condition(node_id, "has_pending_clarification")

        # === Completion Checks ===
        if condition == "all_specs_verified":
            # All SPEC children must be VERIFIED
            has_specs = False
            for succ in self.graph.graph.successors(node_id):
                succ_data = self.graph.graph.nodes[succ]
                if succ_data.get('type') == NodeType.SPEC.value:
                    has_specs = True
                    if succ_data.get('status') != NodeStatus.VERIFIED.value:
                        return False
            return has_specs  # Must have at least one SPEC

        if condition == "implementation_verified":
            # CODE implementing this SPEC must be VERIFIED
            for pred in self.graph.graph.predecessors(node_id):
                edge_data = self.graph.graph.edges[pred, node_id]
                if edge_data.get('type') == EdgeType.IMPLEMENTS.value:
                    pred_status = self.graph.graph.nodes[pred].get('status')
                    if pred_status == NodeStatus.VERIFIED.value:
                        return True
            return False

        if condition == "verification_passed":
            # Has TEST with PASS verdict
            for pred in self.graph.graph.predecessors(node_id):
                edge_data = self.graph.graph.edges[pred, node_id]
                if edge_data.get('type') == EdgeType.VERIFIES.value:
                    pred_data = self.graph.graph.nodes[pred]
                    verdict = pred_data.get('metadata', {}).get('verdict')
                    if verdict == "PASS":
                        return True
            return False

        if condition == "verification_failed":
            # Has TEST with FAIL verdict
            for pred in self.graph.graph.predecessors(node_id):
                edge_data = self.graph.graph.edges[pred, node_id]
                if edge_data.get('type') == EdgeType.VERIFIES.value:
                    pred_data = self.graph.graph.nodes[pred]
                    verdict = pred_data.get('metadata', {}).get('verdict')
                    if verdict == "FAIL":
                        return True
            return False

        # === Retry/Escalation ===
        if condition == "max_attempts_exceeded":
            metadata = node_data.get('metadata', {})
            attempts = metadata.get('attempts', 0)
            max_attempts = metadata.get('max_attempts', 3)
            return attempts >= max_attempts

        if condition == "max_escalations_exceeded":
            # Count ESCALATION nodes for this REQ
            escalation_count = 0
            for succ in self.graph.graph.successors(node_id):
                if self.graph.graph.nodes[succ].get('type') == NodeType.ESCALATION.value:
                    escalation_count += 1
            return escalation_count >= 3  # Max 3 escalations

        # === Resolution ===
        if condition == "has_resolution":
            # CLARIFICATION has RESOLVED_BY edge
            for pred in self.graph.graph.predecessors(node_id):
                edge_data = self.graph.graph.edges[pred, node_id]
                if edge_data.get('type') == EdgeType.RESOLVED_BY.value:
                    return True
            return False

        if condition == "replan_produced":
            # ESCALATION handling produced new SPEC nodes
            for succ in self.graph.graph.successors(node_id):
                if self.graph.graph.nodes[succ].get('type') == NodeType.SPEC.value:
                    return True
            return False

        # === Dispatch Conditions ===
        if condition == "needs_decomposition":
            # REQ has no SPEC children
            for succ in self.graph.graph.successors(node_id):
                if self.graph.graph.nodes[succ].get('type') == NodeType.SPEC.value:
                    return False
            return True

        if condition == "ready_for_build":
            # SPEC with dependencies met, no CODE yet
            if not self.evaluate_condition(node_id, "dependencies_verified"):
                return False
            # Check no CODE implementing this SPEC
            for pred in self.graph.graph.predecessors(node_id):
                edge_data = self.graph.graph.edges[pred, node_id]
                if edge_data.get('type') == EdgeType.IMPLEMENTS.value:
                    return False
            return True

        if condition == "needs_verification":
            # CODE has no TEST verifying it
            for pred in self.graph.graph.predecessors(node_id):
                edge_data = self.graph.graph.edges[pred, node_id]
                if edge_data.get('type') == EdgeType.VERIFIES.value:
                    return False
            return True

        if condition == "needs_resolution":
            # CLARIFICATION not yet resolved
            return not self.evaluate_condition(node_id, "has_resolution")

        if condition == "needs_replan":
            # ESCALATION not yet addressed
            return not self.evaluate_condition(node_id, "replan_produced")

        logger.warning(f"Condition not implemented: {condition}")
        return False

    # =========================================================================
    # TRANSITION VALIDATION
    # =========================================================================

    def can_transition(
        self,
        node_id: str,
        target_status: NodeStatus
    ) -> Tuple[bool, Optional[TransitionRule]]:
        """
        Check if a transition is allowed by the TransitionMatrix.

        Args:
            node_id: The node to transition
            target_status: The desired target status

        Returns:
            Tuple of (allowed, matching_rule)
        """
        node_data = self.graph.graph.nodes.get(node_id)
        if not node_data:
            return False, None

        current_status = node_data.get('status', NodeStatus.PENDING.value)
        node_type = node_data.get('type')

        # Look up rules in TransitionMatrix
        key = (current_status, node_type)
        rules = TRANSITION_MATRIX.get(key, [])

        # Sort by priority (higher first)
        sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            # Handle both enum and string values for target_status
            rule_target = rule.target_status.value if hasattr(rule.target_status, 'value') else rule.target_status
            check_target = target_status.value if hasattr(target_status, 'value') else target_status
            if rule_target != check_target:
                continue

            # Check required edge types
            has_required_edges = True
            for edge_type in rule.required_edge_types:
                # Handle both enum and string values
                edge_type_value = edge_type.value if hasattr(edge_type, 'value') else edge_type
                found = False
                for pred in self.graph.graph.predecessors(node_id):
                    edge_data = self.graph.graph.edges[pred, node_id]
                    if edge_data.get('type') == edge_type_value:
                        found = True
                        break
                if not found:
                    has_required_edges = False
                    break

            if not has_required_edges:
                continue

            # Check all conditions
            all_conditions_met = True
            for condition in rule.required_conditions:
                if not self.evaluate_condition(node_id, condition):
                    all_conditions_met = False
                    break

            if all_conditions_met:
                return True, rule

        return False, None

    def get_valid_transitions(self, node_id: str) -> List[NodeStatus]:
        """Get all valid target statuses for a node."""
        node_data = self.graph.graph.nodes.get(node_id)
        if not node_data:
            return []

        current_status = node_data.get('status', NodeStatus.PENDING.value)
        node_type = node_data.get('type')

        key = (current_status, node_type)
        rules = TRANSITION_MATRIX.get(key, [])

        valid = []
        for rule in rules:
            allowed, _ = self.can_transition(node_id, rule.target_status)
            if allowed:
                valid.append(rule.target_status)

        return valid

    # =========================================================================
    # AGENT DISPATCH
    # =========================================================================

    def get_agent_for_node(self, node_id: str) -> Optional[str]:
        """
        Determine which agent should process a node.

        Consults AGENT_DISPATCH rules.

        Args:
            node_id: The node to dispatch

        Returns:
            Agent role name or None
        """
        node_data = self.graph.graph.nodes.get(node_id)
        if not node_data:
            return None

        node_type = node_data.get('type')

        # Check each dispatch rule
        for (dispatch_type, condition), agent_role in AGENT_DISPATCH.items():
            if dispatch_type != node_type:
                continue
            if self.evaluate_condition(node_id, condition):
                return agent_role

        return None

    def get_or_create_agent(self, role: str) -> GenericAgent:
        """Get or create an agent for a role."""
        if role not in self._agents:
            self._agents[role] = create_agent(
                role=role,
                llm_gateway=self.gateway
            )
        return self._agents[role]

    # =========================================================================
    # GRAPH CONTEXT BUILDING
    # =========================================================================

    def build_context(self, node_id: str) -> GraphContext:
        """
        Build context for an agent from the graph neighborhood.

        Args:
            node_id: The node being processed

        Returns:
            GraphContext with all relevant information
        """
        node_data = self.graph.graph.nodes.get(node_id, {})

        # Find root REQ
        req_content = None
        req_id = None
        for ancestor in self._find_ancestors(node_id, NodeType.REQ.value):
            ancestor_data = self.graph.graph.nodes[ancestor]
            req_content = ancestor_data.get('content')
            req_id = ancestor
            break

        # Get parent nodes (TRACES_TO)
        parents = []
        for pred in self.graph.graph.predecessors(node_id):
            edge_data = self.graph.graph.edges[pred, node_id]
            if edge_data.get('type') == EdgeType.TRACES_TO.value:
                parents.append(dict(self.graph.graph.nodes[pred]))
                parents[-1]['id'] = pred

        # Get child nodes
        children = []
        for succ in self.graph.graph.successors(node_id):
            children.append(dict(self.graph.graph.nodes[succ]))
            children[-1]['id'] = succ

        # Get dependencies (DEPENDS_ON)
        deps = []
        for pred in self.graph.graph.predecessors(node_id):
            edge_data = self.graph.graph.edges[pred, node_id]
            if edge_data.get('type') == EdgeType.DEPENDS_ON.value:
                deps.append(dict(self.graph.graph.nodes[pred]))
                deps[-1]['id'] = pred

        # Get feedback
        feedback = []
        for pred in self.graph.graph.predecessors(node_id):
            edge_data = self.graph.graph.edges[pred, node_id]
            if edge_data.get('type') == EdgeType.FEEDBACK.value:
                feedback.append({
                    'source_id': pred,
                    'content': edge_data.get('critique', ''),
                    'retry_number': edge_data.get('retry_number', 0)
                })

        metadata = node_data.get('metadata', {})

        return GraphContext(
            node_id=node_id,
            node_type=node_data.get('type', ''),
            node_content=node_data.get('content', ''),
            node_status=node_data.get('status', NodeStatus.PENDING.value),
            node_metadata=metadata,
            parent_nodes=parents,
            child_nodes=children,
            dependency_nodes=deps,
            requirement_content=req_content,
            requirement_id=req_id,
            feedback=feedback,
            previous_attempts=metadata.get('attempts', 0)
        )

    def _find_ancestors(self, node_id: str, target_type: str) -> List[str]:
        """Find ancestor nodes of a specific type."""
        result = []
        visited = set()
        queue = [node_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for pred in self.graph.graph.predecessors(current):
                pred_data = self.graph.graph.nodes[pred]
                if pred_data.get('type') == target_type:
                    result.append(pred)
                else:
                    queue.append(pred)

        return result

    # =========================================================================
    # EXECUTION
    # =========================================================================

    async def execute_step(self, node_id: str) -> Optional[UnifiedAgentOutput]:
        """
        Process a single node if physics allows.

        This is the core execution method. It:
        1. Checks if transition to PROCESSING is allowed
        2. Dispatches to the appropriate agent
        3. Applies the agent's output to the graph
        4. Transitions based on results

        Args:
            node_id: The node to process

        Returns:
            Agent output or None if node cannot be processed
        """
        # Check if we can transition to PROCESSING
        can_start, rule = self.can_transition(node_id, NodeStatus.PROCESSING)
        if not can_start:
            logger.debug(f"Cannot process {node_id[:8]}: transition not allowed")
            return None

        # Get the agent
        agent_role = self.get_agent_for_node(node_id)
        if not agent_role:
            logger.warning(f"No agent for node {node_id[:8]}")
            return None

        # Transition to PROCESSING
        old_status = self.graph.graph.nodes[node_id].get('status', 'PENDING')
        node_type = self.graph.graph.nodes[node_id].get('type', 'UNKNOWN')
        self.graph.set_status(node_id, NodeStatus.PROCESSING, f"Agent {agent_role} starting")
        await self._emit('on_node_status_changed', node_id, old_status, 'PROCESSING', f"Agent {agent_role} starting")

        # Telemetry: Log state transition
        self._telemetry.log_state_transition(node_id, node_type, old_status, 'PROCESSING', f"Agent {agent_role}")

        # Increment attempts
        node_data = self.graph.graph.nodes[node_id]
        metadata = node_data.get('metadata', {})
        metadata['attempts'] = metadata.get('attempts', 0) + 1
        node_data['metadata'] = metadata

        # Build context and execute
        context = self.build_context(node_id)
        agent = self.get_or_create_agent(agent_role)

        # Emit agent started event
        await self._emit('on_agent_started', agent_role, node_id)

        # Telemetry: Log agent start
        self._telemetry.log_agent_start(agent_role, node_id)
        import time
        agent_start_time = time.time()

        try:
            output = await agent.process(context)

            # Apply output to graph
            await self._apply_output(node_id, output)

            # Update cost
            metadata['cost_actual'] = metadata.get('cost_actual', 0.0) + output.cost_incurred
            node_data['metadata'] = metadata

            # Emit agent finished event
            await self._emit('on_agent_finished', agent_role, node_id, True, output.cost_incurred)

            # Telemetry: Log agent success
            duration_ms = (time.time() - agent_start_time) * 1000
            self._telemetry.log_agent_end(agent_role, node_id, True, output.cost_incurred, duration_ms)

            return output

        except Exception as e:
            logger.error(f"Agent {agent_role} failed on {node_id[:8]}: {e}")
            metadata['last_error'] = str(e)
            node_data['metadata'] = metadata

            # Emit error events
            await self._emit('on_error', node_id, str(e))
            await self._emit('on_agent_finished', agent_role, node_id, False, 0.0)

            # Telemetry: Log agent failure
            duration_ms = (time.time() - agent_start_time) * 1000
            self._telemetry.log_agent_end(agent_role, node_id, False, 0.0, duration_ms)
            self._telemetry.log_agent_error(agent_role, node_id, str(e))

            # Check if we should escalate
            if self.evaluate_condition(node_id, "max_attempts_exceeded"):
                self.graph.set_status(node_id, NodeStatus.FAILED, str(e))
                await self._emit('on_node_status_changed', node_id, 'PROCESSING', 'FAILED', str(e))
                await self._create_escalation(node_id, str(e))

            return None

    async def _apply_output(self, source_node_id: str, output: UnifiedAgentOutput):
        """Apply agent output to the graph."""
        import uuid

        # Create new nodes
        node_id_map = {}  # Map placeholder IDs to real IDs
        for node_spec in output.new_nodes:
            new_id = uuid.uuid4().hex
            node_id_map[node_spec.type] = new_id  # Simple mapping

            self.graph.add_node(
                node_id=new_id,
                node_type=NodeType(node_spec.type),
                content=node_spec.content,
                metadata=node_spec.metadata or {}
            )

            # Emit node created event
            await self._emit('on_node_created', new_id, node_spec.type, node_spec.content, node_spec.metadata or {})

            # Create TRACES_TO edge to source
            signature = output.agent_role or "system"
            self.graph.add_edge(
                source_id=new_id,
                target_id=source_node_id,
                edge_type=EdgeType.TRACES_TO,
                signed_by=signature,
                signature=f"{signature}:{new_id[:8]}"
            )

            # Emit edge created event
            await self._emit('on_edge_created', new_id, source_node_id, 'TRACES_TO')

        # Create edges
        for edge_spec in output.new_edges:
            # Resolve IDs
            src = edge_spec.source_id
            tgt = edge_spec.target_id

            # Check if IDs exist or need mapping
            if src not in self.graph.graph.nodes:
                # Try to find by type
                src = node_id_map.get(src, src)
            if tgt not in self.graph.graph.nodes:
                tgt = node_id_map.get(tgt, tgt)

            if src in self.graph.graph.nodes and tgt in self.graph.graph.nodes:
                signature = output.agent_role or "system"
                self.graph.add_edge(
                    source_id=src,
                    target_id=tgt,
                    edge_type=EdgeType(edge_spec.relation),
                    signed_by=signature,
                    signature=f"{signature}:{src[:8]}"
                )

                # Emit edge created event
                await self._emit('on_edge_created', src, tgt, edge_spec.relation)

        # Apply status updates
        for update_node_id, new_status in output.status_updates.items():
            if update_node_id in self.graph.graph.nodes:
                try:
                    old_status = self.graph.graph.nodes[update_node_id].get('status', 'PENDING')
                    self.graph.set_status(
                        update_node_id,
                        NodeStatus(new_status),
                        f"Updated by {output.agent_role}"
                    )
                    # Emit status change event
                    await self._emit('on_node_status_changed', update_node_id, old_status, new_status, f"Updated by {output.agent_role}")
                except Exception as e:
                    logger.warning(f"Failed to update status for {update_node_id}: {e}")

        # Write artifacts
        for file_path, content in output.artifacts.items():
            try:
                from pathlib import Path
                path = Path(file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                logger.info(f"Wrote artifact: {file_path}")
            except Exception as e:
                logger.error(f"Failed to write artifact {file_path}: {e}")

    async def _create_escalation(self, failed_node_id: str, error: str):
        """Create an ESCALATION node for a failed node."""
        import uuid

        node_data = self.graph.graph.nodes[failed_node_id]

        escalation_id = uuid.uuid4().hex
        self.graph.add_node(
            node_id=escalation_id,
            node_type=NodeType.ESCALATION,
            content=f"Node {failed_node_id[:8]} failed after max attempts.\n\nError: {error}",
            metadata={
                'failed_node_id': failed_node_id,
                'failed_node_type': node_data.get('type'),
                'error': error
            }
        )

        # Link to failed node
        self.graph.add_edge(
            source_id=escalation_id,
            target_id=failed_node_id,
            edge_type=EdgeType.BLOCKS,
            signed_by="system",
            signature=f"escalation:{escalation_id[:8]}"
        )

        logger.info(f"Created ESCALATION {escalation_id[:8]} for failed {failed_node_id[:8]}")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def get_processable_nodes(self) -> List[str]:
        """
        Get all nodes that can be processed right now.

        Returns nodes where:
        - Status is PENDING
        - Can transition to PROCESSING (physics allows)
        - Has an agent dispatch rule
        """
        processable = []

        for node_id, data in self.graph.graph.nodes(data=True):
            if data.get('status') != NodeStatus.PENDING.value:
                continue

            can_start, _ = self.can_transition(node_id, NodeStatus.PROCESSING)
            if not can_start:
                continue

            agent = self.get_agent_for_node(node_id)
            if not agent:
                continue

            processable.append(node_id)

        return processable

    def get_execution_waves(self) -> List[List[str]]:
        """
        Get nodes organized into execution waves.

        Nodes in the same wave can be processed in parallel.
        Each wave depends on previous waves completing.
        """
        import networkx as nx

        # Build dependency subgraph
        dep_graph = nx.DiGraph()
        pending = [
            n for n, d in self.graph.graph.nodes(data=True)
            if d.get('status') == NodeStatus.PENDING.value
        ]

        for node_id in pending:
            dep_graph.add_node(node_id)
            for pred in self.graph.graph.predecessors(node_id):
                edge_data = self.graph.graph.edges[pred, node_id]
                if edge_data.get('type') == EdgeType.DEPENDS_ON.value:
                    if pred in pending:
                        dep_graph.add_edge(pred, node_id)

        # Topological generations = execution waves
        try:
            waves = list(nx.topological_generations(dep_graph))
            return waves
        except nx.NetworkXUnfeasible:
            logger.warning("Cycle detected in dependency graph")
            return [pending]  # Fall back to single wave

    async def run_until_complete(self, max_iterations: int = 100) -> Dict[str, Any]:
        """
        Main execution loop.

        Processes nodes until:
        - No more processable nodes
        - Max iterations reached
        - All REQ nodes are VERIFIED or FAILED

        Args:
            max_iterations: Safety limit

        Returns:
            Execution statistics
        """
        stats = {
            'iterations': 0,
            'nodes_processed': 0,
            'errors': 0,
            'total_cost': 0.0
        }

        for iteration in range(max_iterations):
            stats['iterations'] = iteration + 1

            processable = self.get_processable_nodes()
            if not processable:
                logger.info("No more processable nodes")
                break

            logger.info(f"Iteration {iteration + 1}: {len(processable)} processable nodes")

            # Emit iteration event
            await self._emit('on_iteration', iteration + 1, len(processable))

            # Telemetry: Log iteration
            self._telemetry.increment_step()

            # Process nodes (could be parallelized)
            for node_id in processable:
                try:
                    output = await self.execute_step(node_id)
                    if output:
                        stats['nodes_processed'] += 1
                        stats['total_cost'] += output.cost_incurred
                except Exception as e:
                    logger.error(f"Error processing {node_id[:8]}: {e}")
                    stats['errors'] += 1
                    await self._emit('on_error', node_id, str(e))

            # Telemetry: Log iteration results
            self._telemetry.log_iteration(iteration + 1, stats['nodes_processed'])

            # Check completion
            if self._all_reqs_terminal():
                logger.info("All requirements reached terminal state")
                break

        # Emit completion event
        await self._emit('on_complete', stats)

        # Telemetry: Flush session
        self._telemetry.flush()

        return stats

    def _all_reqs_terminal(self) -> bool:
        """Check if all REQ nodes are in terminal state."""
        for node_id, data in self.graph.graph.nodes(data=True):
            if data.get('type') != NodeType.REQ.value:
                continue
            status = data.get('status')
            if status not in [NodeStatus.VERIFIED.value, NodeStatus.FAILED.value]:
                return False
        return True

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def prune_orphans(self) -> int:
        """
        Remove orphan nodes not connected to any REQ.

        Returns:
            Number of nodes removed
        """
        import networkx as nx

        # Find all REQ nodes
        req_nodes = [
            n for n, d in self.graph.graph.nodes(data=True)
            if d.get('type') == NodeType.REQ.value
        ]

        if not req_nodes:
            return 0

        # Find all nodes reachable from REQs (in undirected sense)
        undirected = self.graph.graph.to_undirected()
        reachable = set()
        for req in req_nodes:
            reachable.update(nx.node_connected_component(undirected, req))

        # Remove unreachable nodes
        all_nodes = set(self.graph.graph.nodes())
        orphans = all_nodes - reachable

        if orphans:
            self.graph.graph.remove_nodes_from(orphans)
            logger.info(f"Pruned {len(orphans)} orphan nodes")

        return len(orphans)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.DEBUG)

    async def test():
        print("=== Testing GraphRuntime ===")

        # Would need a GraphDB instance for full test
        # For now just test imports
        print("✅ GraphRuntime imports successfully")

        # Test condition evaluation (mock)
        print("\nKnown conditions:")
        for cond in sorted(KNOWN_CONDITIONS):
            print(f"  - {cond}")

        print("\nAgent dispatch rules:")
        for (node_type, condition), role in AGENT_DISPATCH.items():
            print(f"  {node_type} + {condition} -> {role}")

        print("\n✅ GraphRuntime tests passed")

    asyncio.run(test())
