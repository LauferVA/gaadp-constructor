"""
TRACEABILITY HELPER
Utilities for extracting traceability information from the graph.
"""
import networkx as nx
from typing import Optional, List, Dict
from core.ontology import EdgeType, NodeType


class TraceabilityHelper:
    """
    Helper class to extract traceability chains from the graph.

    Used by agents to provide traceability information when creating nodes.
    """

    @staticmethod
    def trace_to_requirement(graph: nx.DiGraph, node_id: str) -> Optional[str]:
        """
        Trace a node back to its originating REQ node.

        Follows edges: TRACES_TO, IMPLEMENTS, DEPENDS_ON backwards
        """
        if node_id not in graph:
            return None

        # If this is already a REQ, return it
        if graph.nodes[node_id].get('type') == NodeType.REQ.value:
            return node_id

        # BFS backwards through predecessors
        visited = set()
        queue = [node_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            node_type = graph.nodes[current].get('type')
            if node_type == NodeType.REQ.value:
                return current

            # Check all incoming edges
            for pred in graph.predecessors(current):
                if pred not in visited:
                    queue.append(pred)

        return None

    @staticmethod
    def trace_to_specification(graph: nx.DiGraph, node_id: str) -> Optional[str]:
        """
        Trace a node back to its SPEC node.

        Follows IMPLEMENTS and TRACES_TO edges backwards.
        """
        if node_id not in graph:
            return None

        # If this is already a SPEC, return it
        if graph.nodes[node_id].get('type') == NodeType.SPEC.value:
            return node_id

        # Check direct IMPLEMENTS edge
        for pred in graph.predecessors(node_id):
            edge_data = graph.edges[pred, node_id]
            if edge_data.get('type') == EdgeType.IMPLEMENTS.value:
                if graph.nodes[pred].get('type') == NodeType.SPEC.value:
                    return pred

        # BFS backwards
        visited = set()
        queue = [node_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            node_type = graph.nodes[current].get('type')
            if node_type == NodeType.SPEC.value:
                return current

            for pred in graph.predecessors(current):
                if pred not in visited:
                    queue.append(pred)

        return None

    @staticmethod
    def get_implements_target(graph: nx.DiGraph, node_id: str) -> Optional[str]:
        """
        Get the direct IMPLEMENTS edge target.

        Returns the SPEC node that this node directly implements.
        """
        if node_id not in graph:
            return None

        for pred in graph.predecessors(node_id):
            edge_data = graph.edges[pred, node_id]
            if edge_data.get('type') == EdgeType.IMPLEMENTS.value:
                return pred

        return None

    @staticmethod
    def get_dependencies(graph: nx.DiGraph, node_id: str) -> List[str]:
        """
        Get all DEPENDS_ON dependencies for a node.

        Returns list of node IDs this node depends on.
        """
        if node_id not in graph:
            return []

        dependencies = []
        for pred in graph.predecessors(node_id):
            edge_data = graph.edges[pred, node_id]
            if edge_data.get('type') == EdgeType.DEPENDS_ON.value:
                dependencies.append(pred)

        return dependencies

    @staticmethod
    def extract_full_lineage(graph: nx.DiGraph, node_id: str) -> Dict[str, any]:
        """
        Extract complete traceability lineage for a node.

        Returns dict with:
        - traces_to_req
        - traces_to_spec
        - implements_spec
        - depends_on
        """
        return {
            'traces_to_req': TraceabilityHelper.trace_to_requirement(graph, node_id),
            'traces_to_spec': TraceabilityHelper.trace_to_specification(graph, node_id),
            'implements_spec': TraceabilityHelper.get_implements_target(graph, node_id),
            'depends_on': TraceabilityHelper.get_dependencies(graph, node_id)
        }

    @staticmethod
    def get_traceability_for_new_node(
        graph: nx.DiGraph,
        parent_node_id: Optional[str] = None,
        implements_node_id: Optional[str] = None,
        depends_on_ids: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Calculate traceability for a new node being created.

        Args:
            graph: The graph
            parent_node_id: Parent node (e.g., SPEC for a CODE node)
            implements_node_id: Node this implements (usually same as parent)
            depends_on_ids: List of dependency node IDs

        Returns:
            Dict with traceability info ready for metrics
        """
        lineage = {
            'traces_to_req': None,
            'traces_to_spec': None,
            'implements_spec': implements_node_id,
            'depends_on': depends_on_ids or []
        }

        # Trace backwards from parent
        if parent_node_id and parent_node_id in graph:
            parent_type = graph.nodes[parent_node_id].get('type')

            # If parent is REQ, use it directly
            if parent_type == NodeType.REQ.value:
                lineage['traces_to_req'] = parent_node_id

            # If parent is SPEC, trace its REQ
            elif parent_type == NodeType.SPEC.value:
                lineage['traces_to_spec'] = parent_node_id
                lineage['traces_to_req'] = TraceabilityHelper.trace_to_requirement(
                    graph, parent_node_id
                )

            # If parent is CODE/TEST, inherit its lineage
            else:
                parent_lineage = TraceabilityHelper.extract_full_lineage(graph, parent_node_id)
                lineage['traces_to_req'] = parent_lineage['traces_to_req']
                lineage['traces_to_spec'] = parent_lineage['traces_to_spec']

        return lineage
