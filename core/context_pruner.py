"""
CONTEXT PRUNER
Semantic relevance-based context pruning for token-constrained LLM calls.
"""
import logging
from typing import List, Dict, Tuple, Optional
import networkx as nx

logger = logging.getLogger("ContextPruner")


class ContextPruner:
    """
    Prunes graph context using semantic relevance scoring.

    Strategies:
    1. Semantic similarity to center node (primary)
    2. Temporal decay (older nodes less relevant)
    3. Critical path preservation (REQ → SPEC → CODE chains)
    4. Edge type weighting (IMPLEMENTS > DEPENDS_ON)
    """

    def __init__(self, semantic_memory=None):
        """
        Args:
            semantic_memory: Optional SemanticMemory instance for embeddings
        """
        self.semantic_memory = semantic_memory

        # Edge type weights (higher = more important)
        self.edge_weights = {
            'IMPLEMENTS': 1.0,
            'VERIFIES': 0.9,
            'TRACES_TO': 0.8,
            'FEEDBACK': 0.7,
            'DEPENDS_ON': 0.6,
            'DEFINES': 0.5,
        }

    def score_node_relevance(
        self,
        node_id: str,
        center_node_id: str,
        graph: nx.DiGraph,
        center_embedding: Optional[List[float]] = None
    ) -> float:
        """
        Calculate relevance score for a node relative to center node.

        Returns:
            Score between 0.0 (irrelevant) and 1.0 (highly relevant)
        """
        if node_id == center_node_id:
            return 1.0

        scores = []

        # 1. Semantic similarity (if embeddings available)
        if (self.semantic_memory and
            hasattr(self.semantic_memory, 'embeddings_enabled') and
            self.semantic_memory.embeddings_enabled and
            center_embedding):
            try:
                node_embedding = self.semantic_memory.vectors.get(node_id)
                if node_embedding is not None:
                    similarity = self._cosine_similarity(center_embedding, node_embedding)
                    scores.append(('semantic', similarity, 0.5))
            except Exception as e:
                logger.debug(f"Failed to get semantic similarity: {e}")

        # 2. Graph distance (inverse BFS depth)
        try:
            shortest_path_len = nx.shortest_path_length(
                graph.to_undirected(),
                center_node_id,
                node_id
            )
            # Normalize: 1.0 at distance 0, 0.5 at distance 2, 0.25 at distance 4
            distance_score = 1.0 / (2 ** (shortest_path_len / 2))
            scores.append(('distance', distance_score, 0.3))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            scores.append(('distance', 0.0, 0.3))

        # 3. Edge type importance
        edge_score = self._score_edges_to_center(node_id, center_node_id, graph)
        scores.append(('edge_type', edge_score, 0.2))

        # Weighted average
        if not scores:
            return 0.5  # Neutral if no signals

        total_weight = sum(weight for _, _, weight in scores)
        weighted_sum = sum(score * weight for _, score, weight in scores)
        return weighted_sum / total_weight

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 0.0
        return float(dot_product / norm_product)

    def _score_edges_to_center(
        self,
        node_id: str,
        center_node_id: str,
        graph: nx.DiGraph
    ) -> float:
        """Score based on edge types connecting to center."""
        try:
            path = nx.shortest_path(graph.to_undirected(), node_id, center_node_id)
            if len(path) < 2:
                return 1.0

            # Get edge types along path
            edge_scores = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]

                # Check both directions
                if graph.has_edge(u, v):
                    edge_type = graph.edges[u, v].get('type', 'UNKNOWN')
                elif graph.has_edge(v, u):
                    edge_type = graph.edges[v, u].get('type', 'UNKNOWN')
                else:
                    edge_type = 'UNKNOWN'

                edge_scores.append(self.edge_weights.get(edge_type, 0.3))

            return sum(edge_scores) / len(edge_scores) if edge_scores else 0.3

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 0.0

    def prune_by_relevance(
        self,
        candidates: List[str],
        center_node_id: str,
        graph: nx.DiGraph,
        target_count: int
    ) -> List[str]:
        """
        Prune candidates to target count using relevance scoring.

        Args:
            candidates: List of node IDs to consider
            center_node_id: The center node for relevance calculation
            graph: The graph
            target_count: Maximum number of nodes to keep

        Returns:
            Pruned list of node IDs (most relevant)
        """
        if len(candidates) <= target_count:
            return candidates

        # Get center node embedding if available
        center_embedding = None
        if (self.semantic_memory and
            hasattr(self.semantic_memory, 'embeddings_enabled') and
            self.semantic_memory.embeddings_enabled):
            center_embedding = self.semantic_memory.vectors.get(center_node_id)

        # Score all candidates
        scored_candidates = []
        for node_id in candidates:
            if node_id == center_node_id:
                # Always keep center node
                scored_candidates.append((node_id, 1.0))
                continue

            score = self.score_node_relevance(
                node_id,
                center_node_id,
                graph,
                center_embedding
            )
            scored_candidates.append((node_id, score))

        # Sort by score descending and take top N
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        pruned = [node_id for node_id, _ in scored_candidates[:target_count]]

        logger.debug(
            f"Pruned context from {len(candidates)} to {len(pruned)} nodes "
            f"(relevance range: {scored_candidates[-1][1]:.3f} - {scored_candidates[0][1]:.3f})"
        )

        return pruned

    def prune_to_token_budget(
        self,
        candidates: List[str],
        center_node_id: str,
        graph: nx.DiGraph,
        token_counter,
        max_tokens: int
    ) -> Tuple[List[str], int]:
        """
        Prune candidates to fit within token budget using relevance.

        Args:
            candidates: List of node IDs to consider
            center_node_id: The center node
            graph: The graph
            token_counter: TokenCounter instance
            max_tokens: Maximum tokens allowed

        Returns:
            (pruned_node_ids, total_tokens_used)
        """
        # Get center node embedding if available
        center_embedding = None
        if (self.semantic_memory and
            hasattr(self.semantic_memory, 'embeddings_enabled') and
            self.semantic_memory.embeddings_enabled):
            center_embedding = self.semantic_memory.vectors.get(center_node_id)

        # Score and sort all candidates by relevance
        scored_candidates = []
        for node_id in candidates:
            score = self.score_node_relevance(
                node_id,
                center_node_id,
                graph,
                center_embedding
            )

            content = str(graph.nodes[node_id].get('content', ''))
            tokens = token_counter.count_tokens(content)

            scored_candidates.append((node_id, score, tokens))

        # Sort by relevance (highest first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Greedy selection to fit budget
        selected = []
        total_tokens = 0

        for node_id, score, tokens in scored_candidates:
            if total_tokens + tokens <= max_tokens:
                selected.append(node_id)
                total_tokens += tokens
            else:
                # Check if we can fit a summarized version
                # (future enhancement)
                break

        logger.info(
            f"Pruned to {len(selected)}/{len(candidates)} nodes "
            f"({total_tokens}/{max_tokens} tokens)"
        )

        return selected, total_tokens
