"""
GRAPH DATABASE BACKEND (Hardened)
Features: Atomic Persistence, Token Limits, Merkle Linking.
"""
import networkx as nx
import logging
import datetime
import json
import hashlib
import os
import time
from typing import List, Dict, Any, Optional
# Core ontology - the physics of the system
from core.ontology import NodeType, EdgeType, NodeStatus
# Legacy imports - will be removed in Phase 3
try:
    from core.state_machine import NodeStateMachine, StateTransitionError
except ImportError:
    # Fallback: state machine functionality now in new_ontology TransitionMatrix
    NodeStateMachine = None
    StateTransitionError = Exception
try:
    from core.token_counter import TokenCounter
except ImportError:
    TokenCounter = None
try:
    from core.context_pruner import ContextPruner
except ImportError:
    ContextPruner = None

# Schema version for persistence format
SCHEMA_VERSION = "1.0"

class GraphDB:
    def __init__(
        self,
        persistence_path: str = ".gaadp/graph.json",
        model: str = "claude-3-sonnet",
        semantic_memory=None,
        event_bus=None
    ):
        self.logger = logging.getLogger("GraphDB")
        # Auto-convert .pkl to .json for backward compatibility
        if persistence_path.endswith('.pkl'):
            persistence_path = persistence_path.replace('.pkl', '.json')
        self.persistence_path = persistence_path
        self.graph = nx.DiGraph()
        # Legacy components (optional during migration)
        self._state_machine = NodeStateMachine() if NodeStateMachine else None
        self._token_counter = TokenCounter(default_model=model) if TokenCounter else None
        self._context_pruner = ContextPruner(semantic_memory=semantic_memory) if ContextPruner else None
        self._event_bus = event_bus
        self._load()

    def _calculate_checksum(self, data: Dict) -> str:
        """Calculate SHA256 checksum of graph data."""
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _serialize_graph(self) -> Dict:
        """Serialize graph to JSON-compatible dict."""
        graph_data = nx.node_link_data(self.graph)

        return {
            'version': SCHEMA_VERSION,
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'graph': graph_data,
            'metadata': {
                'node_count': self.graph.number_of_nodes(),
                'edge_count': self.graph.number_of_edges()
            }
        }

    def _persist(self):
        """Atomic Write to Disk (Crash Recovery) - JSON format"""
        temp_path = self.persistence_path + ".tmp"

        # Serialize graph
        data = self._serialize_graph()

        # Add checksum for integrity
        data['checksum'] = self._calculate_checksum(data['graph'])

        # Atomic write
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        os.replace(temp_path, self.persistence_path)

    def _load_from_json(self, path: str) -> bool:
        """Load graph from JSON format."""
        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Validate schema version
            version = data.get('version', 'unknown')
            if version != SCHEMA_VERSION:
                self.logger.warning(f"Schema version mismatch: {version} != {SCHEMA_VERSION}")

            # Validate checksum if present
            if 'checksum' in data:
                stored_checksum = data['checksum']
                calculated_checksum = self._calculate_checksum(data['graph'])
                if stored_checksum != calculated_checksum:
                    self.logger.error("Checksum mismatch - graph may be corrupted!")
                    return False

            # Reconstruct graph
            self.graph = nx.node_link_graph(data['graph'], directed=True)
            self.logger.info(f"Loaded Graph: {self.graph.number_of_nodes()} nodes (JSON)")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load JSON graph: {e}")
            return False

    def _load_from_pickle(self, path: str) -> bool:
        """Legacy: Load graph from pickle format (for migration)."""
        try:
            import pickle
            with open(path, "rb") as f:
                self.graph = pickle.load(f)
            self.logger.warning(f"Loaded legacy pickle graph: {self.graph.number_of_nodes()} nodes")
            self.logger.warning("Migrating to JSON format on next persist...")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load pickle graph: {e}")
            return False

    def _load(self):
        """Load state on startup with automatic format detection."""
        # Try JSON format first
        if os.path.exists(self.persistence_path):
            if self._load_from_json(self.persistence_path):
                return

        # Try legacy pickle format
        pkl_path = self.persistence_path.replace('.json', '.pkl')
        if os.path.exists(pkl_path):
            if self._load_from_pickle(pkl_path):
                # Immediately migrate to JSON
                self._persist()
                self.logger.info(f"Migrated graph from pickle to JSON: {self.persistence_path}")
                return

        # No existing graph found
        self.logger.info("No existing graph found, starting fresh")

    def get_last_node_hash(self) -> str:
        """Retrieves the signature of the last verified edge for Merkle Chaining."""
        try:
            # Get the most recently created verified edge
            verified_edges = [
                (u, v, d) for u, v, d in self.graph.edges(data=True)
                if d.get('type') == EdgeType.VERIFIES.value
            ]
            if not verified_edges: return "GENESIS"

            # Sort by timestamp descending
            last_edge = sorted(verified_edges, key=lambda x: x[2].get('created_at', ''))[-1]
            return last_edge[2].get('signature', 'GENESIS')
        except Exception:
            return "GENESIS"

    def add_node(self, node_id: str, node_type: NodeType, content: Any, metadata: Dict = None):
        if not isinstance(node_type, NodeType):
            raise ValueError(f"Invalid Node Type: {node_type}")

        # Count tokens if content is string (if token counter available)
        token_count = None
        if isinstance(content, str) and self._token_counter:
            token_count = self._token_counter.count_tokens(content)

        self.graph.add_node(
            node_id,
            type=node_type.value,
            status=NodeStatus.PENDING.value,
            content=content,
            metadata=metadata or {},
            created_at=datetime.datetime.utcnow().isoformat()
        )
        self._persist()
        self.logger.info(f"Node Created: {node_id} ({node_type})")

        # Emit metrics event
        if self._event_bus:
            import asyncio
            try:
                asyncio.create_task(self._event_bus.publish(
                    topic="node_lifecycle",
                    message_type="NODE_CREATED",
                    payload={
                        "node_id": node_id,
                        "node_type": node_type.value,
                        "token_count": token_count
                    },
                    source_id="graph_db"
                ))
            except RuntimeError:
                # No event loop running, skip event
                pass

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, signed_by: str, signature: str, previous_hash: str = None):
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            raise ValueError("Cannot link non-existent nodes")

        if edge_type == EdgeType.DEPENDS_ON:
            if nx.has_path(self.graph, target_id, source_id):
                raise ValueError(f"Cycle detected! Cannot link {source_id} -> {target_id}")

        self.graph.add_edge(
            source_id,
            target_id,
            type=edge_type.value,
            signed_by=signed_by,
            signature=signature,
            previous_hash=previous_hash,
            created_at=datetime.datetime.utcnow().isoformat()
        )
        self._persist()

    def set_status(self, node_id: str, new_status: NodeStatus, reason: str = "") -> bool:
        """
        Set node status with state machine validation.

        Args:
            node_id: The node ID
            new_status: Target status (NodeStatus enum or string)
            reason: Reason for transition (for audit trail)

        Returns:
            True if transition succeeded

        Raises:
            StateTransitionError if transition is invalid
            ValueError if node doesn't exist
        """
        if node_id not in self.graph:
            raise ValueError(f"Node {node_id} does not exist")

        node_data = self.graph.nodes[node_id]
        current_status = node_data.get('status', NodeStatus.PENDING.value)
        node_type_str = node_data.get('type')

        # Convert to enums
        if isinstance(current_status, str):
            current_status = NodeStatus(current_status)
        if isinstance(new_status, str):
            new_status = NodeStatus(new_status)

        node_type = NodeType(node_type_str) if node_type_str else None

        # Validate and record transition (if legacy state machine available)
        if self._state_machine:
            self._state_machine.transition(
                node_id, current_status, new_status, node_type, reason
            )
        # Note: In new architecture, validation happens via TransitionMatrix in GraphRuntime

        # Apply the change
        self.graph.nodes[node_id]['status'] = new_status.value
        self.graph.nodes[node_id]['status_updated_at'] = datetime.datetime.utcnow().isoformat()
        if reason:
            self.graph.nodes[node_id]['status_reason'] = reason

        self._persist()
        self.logger.info(f"Status: {node_id} {current_status.value} → {new_status.value}")

        # Emit metrics event
        if self._event_bus:
            import asyncio
            try:
                asyncio.create_task(self._event_bus.publish(
                    topic="node_lifecycle",
                    message_type="STATUS_CHANGED",
                    payload={
                        "node_id": node_id,
                        "old_status": current_status.value,
                        "new_status": new_status.value,
                        "reason": reason
                    },
                    source_id="graph_db"
                ))
            except RuntimeError:
                # No event loop running, skip event
                pass

        return True

    def get_status_history(self, node_id: str) -> list:
        """Get the state transition history for a node."""
        if self._state_machine:
            return self._state_machine.get_history(node_id)
        return []  # No history tracking without legacy state machine

    def get_context_neighborhood(self, center_node_id: str, radius: int, filter_domain: str = None, max_tokens: int = 6000) -> Dict:
        """Semantic Relevance-Based Context Pruner with Feedback Integration."""
        if center_node_id not in self.graph: return {}

        feedback_critiques = []
        feedback_tokens = 0

        # Check for FEEDBACK edges pointing to this node (previous failure critiques)
        for pred in self.graph.predecessors(center_node_id):
            edge_data = self.graph.edges[pred, center_node_id]
            if edge_data.get('type') == EdgeType.FEEDBACK.value:
                critique = edge_data.get('critique', '')
                retry_num = edge_data.get('retry_number', 0)
                if critique:
                    feedback_critiques.append({
                        'retry': retry_num,
                        'critique': critique
                    })
                    # Count feedback critique tokens (if token counter available)
                    if self._token_counter:
                        feedback_tokens += self._token_counter.count_tokens(critique)

        # Adjust token budget to account for feedback
        available_tokens = max_tokens - feedback_tokens

        # Gather ALL candidates within radius (don't break early)
        candidates = {center_node_id}
        for u, v in nx.bfs_edges(self.graph, center_node_id, depth_limit=radius):
            # Apply domain filter if specified
            if filter_domain:
                domain = self.graph.nodes[v].get('metadata', {}).get('domain')
                if domain and domain != filter_domain:
                    continue
            candidates.add(v)

        # Use semantic pruner to select most relevant nodes within budget (if available)
        if self._context_pruner and self._token_counter:
            pruned_nodes, total_tokens = self._context_pruner.prune_to_token_budget(
                candidates=list(candidates),
                center_node_id=center_node_id,
                graph=self.graph,
                token_counter=self._token_counter,
                max_tokens=available_tokens
            )
        else:
            # Fallback: use all candidates without pruning
            pruned_nodes = list(candidates)
            total_tokens = 0

        # Build subgraph from pruned nodes
        subgraph = self.graph.subgraph(pruned_nodes)
        result = nx.node_link_data(subgraph)

        # Inject feedback critiques into context
        if feedback_critiques:
            result['feedback_history'] = sorted(feedback_critiques, key=lambda x: x['retry'])

        # Include token usage stats
        result['_token_stats'] = {
            'total_tokens': total_tokens + feedback_tokens,
            'max_tokens': max_tokens,
            'feedback_tokens': feedback_tokens,
            'content_tokens': total_tokens,
            'nodes_included': len(pruned_nodes),
            'nodes_considered': len(candidates),
            'pruning_ratio': f"{len(pruned_nodes)}/{len(candidates)}"
        }

        # Emit metrics event
        if self._event_bus:
            import asyncio
            try:
                asyncio.create_task(self._event_bus.publish(
                    topic="context",
                    message_type="CONTEXT_PRUNED",
                    payload={
                        "center_node_id": center_node_id,
                        "nodes_considered": len(candidates),
                        "nodes_selected": len(pruned_nodes),
                        "token_budget": max_tokens,
                        "tokens_used": total_tokens + feedback_tokens
                    },
                    source_id="graph_db"
                ))
            except RuntimeError:
                # No event loop running, skip event
                pass

        return result

    def prune_dead_ends(self):
        """Garbage Collection for failed branches"""
        to_remove = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == NodeType.DEAD_END.value
        ]
        self.graph.remove_nodes_from(to_remove)
        if to_remove:
            self._persist()
            self.logger.info(f"GC: Removed {len(to_remove)} dead nodes")

    def get_materializable_nodes(self) -> List[Dict]:
        nodes = []
        for n, data in self.graph.nodes(data=True):
            if (data.get('type') == NodeType.CODE.value and
                data.get('status') == NodeStatus.VERIFIED.value):
                nodes.append({
                    'id': n,
                    'file_path': data['metadata'].get('file_path'),
                    'content': data['content']
                })
        return nodes

    # =========================================================================
    # NEW QUERY METHODS (For GraphRuntime)
    # =========================================================================

    def get_by_status(self, status: NodeStatus) -> List[str]:
        """
        Get all node IDs with a specific status.

        Args:
            status: The status to filter by (NodeStatus enum or string)

        Returns:
            List of node IDs
        """
        status_value = status.value if isinstance(status, NodeStatus) else status
        return [
            n for n, d in self.graph.nodes(data=True)
            if d.get('status') == status_value
        ]

    def get_by_type(self, node_type: NodeType) -> List[str]:
        """
        Get all node IDs of a specific type.

        Args:
            node_type: The type to filter by (NodeType enum or string)

        Returns:
            List of node IDs
        """
        type_value = node_type.value if isinstance(node_type, NodeType) else node_type
        return [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == type_value
        ]

    def get_by_type_and_status(self, node_type: NodeType, status: NodeStatus) -> List[str]:
        """
        Get all node IDs matching both type and status.

        Args:
            node_type: The type to filter by
            status: The status to filter by

        Returns:
            List of node IDs
        """
        type_value = node_type.value if isinstance(node_type, NodeType) else node_type
        status_value = status.value if isinstance(status, NodeStatus) else status
        return [
            n for n, d in self.graph.nodes(data=True)
            if d.get('type') == type_value and d.get('status') == status_value
        ]

    def dependencies_met(self, node_id: str) -> bool:
        """
        Check if all DEPENDS_ON targets are VERIFIED.

        Args:
            node_id: The node to check

        Returns:
            True if all dependencies are met
        """
        if node_id not in self.graph:
            return False

        for pred in self.graph.predecessors(node_id):
            edge_data = self.graph.edges[pred, node_id]
            if edge_data.get('type') == EdgeType.DEPENDS_ON.value:
                pred_status = self.graph.nodes[pred].get('status')
                if pred_status != NodeStatus.VERIFIED.value:
                    return False
        return True

    def is_blocked(self, node_id: str) -> bool:
        """
        Check if a node is blocked by CLARIFICATION or ESCALATION.

        Args:
            node_id: The node to check

        Returns:
            True if node is blocked
        """
        if node_id not in self.graph:
            return False

        for succ in self.graph.successors(node_id):
            succ_data = self.graph.nodes[succ]
            succ_type = succ_data.get('type')
            succ_status = succ_data.get('status')

            if succ_type in [NodeType.CLARIFICATION.value, NodeType.ESCALATION.value]:
                if succ_status in [NodeStatus.PENDING.value, NodeStatus.PROCESSING.value]:
                    return True
        return False

    def get_children(self, node_id: str, edge_type: EdgeType = None) -> List[str]:
        """
        Get child nodes (successors) optionally filtered by edge type.

        Args:
            node_id: The node to get children for
            edge_type: Optional edge type to filter by

        Returns:
            List of child node IDs
        """
        if node_id not in self.graph:
            return []

        children = []
        for succ in self.graph.successors(node_id):
            if edge_type:
                edge_data = self.graph.edges[node_id, succ]
                if edge_data.get('type') == edge_type.value:
                    children.append(succ)
            else:
                children.append(succ)
        return children

    def get_parents(self, node_id: str, edge_type: EdgeType = None) -> List[str]:
        """
        Get parent nodes (predecessors) optionally filtered by edge type.

        Args:
            node_id: The node to get parents for
            edge_type: Optional edge type to filter by

        Returns:
            List of parent node IDs
        """
        if node_id not in self.graph:
            return []

        parents = []
        for pred in self.graph.predecessors(node_id):
            if edge_type:
                edge_data = self.graph.edges[pred, node_id]
                if edge_data.get('type') == edge_type.value:
                    parents.append(pred)
            else:
                parents.append(pred)
        return parents

    def topological_order(self) -> List[str]:
        """
        Get nodes in topological order based on DEPENDS_ON edges.

        Returns:
            List of node IDs in dependency order (dependencies first)
        """
        # Build subgraph with only DEPENDS_ON edges
        dep_graph = nx.DiGraph()
        for n in self.graph.nodes():
            dep_graph.add_node(n)

        for u, v, d in self.graph.edges(data=True):
            if d.get('type') == EdgeType.DEPENDS_ON.value:
                dep_graph.add_edge(u, v)

        try:
            return list(nx.topological_sort(dep_graph))
        except nx.NetworkXUnfeasible:
            self.logger.warning("Cycle detected in dependency graph")
            return list(self.graph.nodes())

    def get_execution_waves(self) -> List[List[str]]:
        """
        Get PENDING nodes organized into parallel execution waves.

        Nodes in the same wave can be processed in parallel.
        Each wave depends on previous waves completing.

        Returns:
            List of waves, each wave is a list of node IDs
        """
        # Build dependency subgraph for PENDING nodes only
        dep_graph = nx.DiGraph()
        pending = self.get_by_status(NodeStatus.PENDING)

        for node_id in pending:
            dep_graph.add_node(node_id)
            for pred in self.graph.predecessors(node_id):
                edge_data = self.graph.edges[pred, node_id]
                if edge_data.get('type') == EdgeType.DEPENDS_ON.value:
                    if pred in pending:
                        dep_graph.add_edge(pred, node_id)

        try:
            return list(nx.topological_generations(dep_graph))
        except nx.NetworkXUnfeasible:
            self.logger.warning("Cycle detected, returning single wave")
            return [pending]

    def get_root_requirement(self, node_id: str) -> Optional[str]:
        """
        Find the root REQ node for any node.

        Traverses TRACES_TO edges upward to find the originating requirement.

        Args:
            node_id: Starting node

        Returns:
            REQ node ID or None
        """
        if node_id not in self.graph:
            return None

        # Check if this is already a REQ
        if self.graph.nodes[node_id].get('type') == NodeType.REQ.value:
            return node_id

        # BFS to find REQ ancestor
        visited = set()
        queue = [node_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            for pred in self.graph.predecessors(current):
                pred_data = self.graph.nodes[pred]
                if pred_data.get('type') == NodeType.REQ.value:
                    return pred
                queue.append(pred)

        return None

    def get_full_context(self, node_id: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        Get comprehensive context for a node including neighborhood.

        This is used by agents to understand the full context around a node.

        Args:
            node_id: The center node
            max_depth: How many hops to include

        Returns:
            Dict with nodes, edges, and metadata
        """
        if node_id not in self.graph:
            return {}

        # Gather nodes within radius
        nodes_in_context = {node_id}
        current_frontier = {node_id}

        for _ in range(max_depth):
            next_frontier = set()
            for n in current_frontier:
                # Add predecessors
                for pred in self.graph.predecessors(n):
                    if pred not in nodes_in_context:
                        nodes_in_context.add(pred)
                        next_frontier.add(pred)
                # Add successors
                for succ in self.graph.successors(n):
                    if succ not in nodes_in_context:
                        nodes_in_context.add(succ)
                        next_frontier.add(succ)
            current_frontier = next_frontier

        # Build subgraph
        subgraph = self.graph.subgraph(nodes_in_context)

        # Find root REQ
        root_req = self.get_root_requirement(node_id)
        req_content = None
        if root_req:
            req_content = self.graph.nodes[root_req].get('content')

        return {
            'center_node': node_id,
            'center_data': dict(self.graph.nodes[node_id]),
            'nodes': {n: dict(d) for n, d in subgraph.nodes(data=True)},
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'type': d.get('type'),
                    'metadata': d
                }
                for u, v, d in subgraph.edges(data=True)
            ],
            'root_requirement': root_req,
            'requirement_content': req_content,
            'node_count': len(nodes_in_context)
        }

    def apply_agent_output(self, source_node_id: str, output: Dict[str, Any]) -> Dict[str, str]:
        """
        Apply agent output to the graph.

        Creates new nodes, edges, and updates statuses as specified.

        Args:
            source_node_id: The node that was processed
            output: Dict with new_nodes, new_edges, status_updates, artifacts

        Returns:
            Mapping of placeholder IDs to real IDs
        """
        import uuid

        id_map = {}

        # Create new nodes
        for node_spec in output.get('new_nodes', []):
            new_id = uuid.uuid4().hex
            node_type = node_spec.get('type')
            id_map[node_type] = new_id  # Simple mapping by type

            self.add_node(
                node_id=new_id,
                node_type=NodeType(node_type),
                content=node_spec.get('content', ''),
                metadata=node_spec.get('metadata', {})
            )

            # Create TRACES_TO edge
            self.add_edge(
                source_id=new_id,
                target_id=source_node_id,
                edge_type=EdgeType.TRACES_TO,
                signed_by='system',
                signature=f"auto:{new_id[:8]}"
            )

        # Create explicit edges
        for edge_spec in output.get('new_edges', []):
            src = edge_spec.get('source_id')
            tgt = edge_spec.get('target_id')

            # Resolve IDs
            if src in id_map:
                src = id_map[src]
            if tgt in id_map:
                tgt = id_map[tgt]

            if src in self.graph.nodes and tgt in self.graph.nodes:
                self.add_edge(
                    source_id=src,
                    target_id=tgt,
                    edge_type=EdgeType(edge_spec.get('relation')),
                    signed_by='system',
                    signature=f"auto:{src[:8]}"
                )

        # Apply status updates
        for update_id, new_status in output.get('status_updates', {}).items():
            if update_id in self.graph.nodes:
                try:
                    self.set_status(update_id, NodeStatus(new_status))
                except Exception as e:
                    self.logger.warning(f"Failed to update status: {e}")

        return id_map
