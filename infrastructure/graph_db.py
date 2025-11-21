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
from core.ontology import NodeType, EdgeType, NodeStatus
from core.state_machine import NodeStateMachine, StateTransitionError
from core.token_counter import TokenCounter
from core.context_pruner import ContextPruner

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
        self._state_machine = NodeStateMachine()
        self._token_counter = TokenCounter(default_model=model)
        self._context_pruner = ContextPruner(semantic_memory=semantic_memory)
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

        # Count tokens if content is string
        token_count = None
        if isinstance(content, str):
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

        # Validate and record transition
        self._state_machine.transition(
            node_id, current_status, new_status, node_type, reason
        )

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
        return self._state_machine.get_history(node_id)

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
                    # Count feedback critique tokens
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

        # Use semantic pruner to select most relevant nodes within budget
        pruned_nodes, total_tokens = self._context_pruner.prune_to_token_budget(
            candidates=list(candidates),
            center_node_id=center_node_id,
            graph=self.graph,
            token_counter=self._token_counter,
            max_tokens=available_tokens
        )

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
