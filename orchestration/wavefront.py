"""
WAVEFRONT EXECUTION ENGINE
"""
import networkx as nx
from typing import List, Set
from core.ontology import NodeStatus, EdgeType

class WavefrontExecutor:
    def identify_waves(self, graph: nx.DiGraph) -> List[Set[str]]:
        waves = []
        pending_nodes = {
            n for n, d in graph.nodes(data=True)
            if d.get('status') in [NodeStatus.PENDING.value, NodeStatus.BLOCKED.value]
        }
        # Nodes considered "complete" for dependency resolution:
        # VERIFIED = successfully completed, TESTED = tests passed (Gen-2 TDD)
        completed_nodes = {
            n for n, d in graph.nodes(data=True)
            if d.get('status') in [NodeStatus.VERIFIED.value, NodeStatus.TESTED.value]
        }

        while pending_nodes:
            current_wave = set()
            for node in pending_nodes:
                dependencies = [
                    u for u, v, d in graph.in_edges(node, data=True)
                    if d.get('type') == EdgeType.DEPENDS_ON.value
                ]
                if set(dependencies).issubset(completed_nodes):
                    current_wave.add(node)

            if not current_wave:
                break

            waves.append(current_wave)
            completed_nodes.update(current_wave)
            pending_nodes.difference_update(current_wave)

        return waves
