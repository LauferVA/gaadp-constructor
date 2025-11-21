"""
GRAPH DATABASE BACKEND (Hardened)
Features: Atomic Persistence, Token Limits, Merkle Linking.
"""
import networkx as nx
import logging
import datetime
import pickle
import os
import time
from typing import List, Dict, Any
from core.ontology import NodeType, EdgeType, NodeStatus

class GraphDB:
    def __init__(self, persistence_path: str = ".gaadp/graph.pkl"):
        self.logger = logging.getLogger("GraphDB")
        self.persistence_path = persistence_path
        self.graph = nx.DiGraph()
        self._load()

    def _persist(self):
        """Atomic Write to Disk (Crash Recovery)"""
        temp_path = self.persistence_path + ".tmp"
        with open(temp_path, "wb") as f:
            pickle.dump(self.graph, f)
        os.replace(temp_path, self.persistence_path)

    def _load(self):
        """Load state on startup"""
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, "rb") as f:
                    self.graph = pickle.load(f)
                self.logger.info(f"Loaded Graph: {self.graph.number_of_nodes()} nodes")
            except Exception as e:
                self.logger.error(f"Corrupt Graph DB, starting fresh: {e}")

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

    def get_context_neighborhood(self, center_node_id: str, radius: int, filter_domain: str = None, max_tokens: int = 6000) -> Dict:
        """Token-Aware Context Pruner"""
        if center_node_id not in self.graph: return {}

        # BFS to prioritize close neighbors
        subgraph_nodes = {center_node_id}
        current_tokens = 0

        # Calculate center node tokens first
        center_content = str(self.graph.nodes[center_node_id].get('content', ''))
        current_tokens += len(center_content) / 4

        for u, v in nx.bfs_edges(self.graph, center_node_id, depth_limit=radius):
            # Simple heuristic: 4 chars ~= 1 token
            node_content = str(self.graph.nodes[v].get('content', ''))
            tokens = len(node_content) / 4

            if current_tokens + tokens > max_tokens:
                self.logger.warning(f"Context truncated at {current_tokens} tokens")
                break

            if filter_domain:
                domain = self.graph.nodes[v].get('metadata', {}).get('domain')
                if domain and domain != filter_domain:
                    continue

            subgraph_nodes.add(v)
            current_tokens += tokens

        subgraph = self.graph.subgraph(subgraph_nodes)
        return nx.node_link_data(subgraph)

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
