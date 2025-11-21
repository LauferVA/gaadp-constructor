"""
SEMANTIC MEMORY LAYER
Stores vector embeddings of code for semantic retrieval.

DEPENDENCY:
- Requires sentence-transformers for embedding generation
- Install: pip install sentence-transformers
"""
import logging
import numpy as np

logger = logging.getLogger("SemanticMemory")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None


class SemanticMemoryError(Exception):
    """Raised when semantic memory operations fail due to missing dependencies."""
    pass


class SemanticMemory:
    def __init__(self, require_embeddings: bool = False, fallback_mode: bool = True):
        """
        Initialize semantic memory.

        Args:
            require_embeddings: If True, raise error if sentence-transformers missing
            fallback_mode: If True, operate in degraded mode without embeddings

        Raises:
            SemanticMemoryError: If embeddings required but not available
        """
        self.vectors = {}  # node_id -> numpy array
        self.model = None
        self.embeddings_enabled = False

        if not EMBEDDINGS_AVAILABLE:
            if require_embeddings:
                raise SemanticMemoryError(
                    "sentence-transformers is required but not installed. "
                    "Install: pip install sentence-transformers"
                )
            elif not fallback_mode:
                raise SemanticMemoryError(
                    "Embeddings not available and fallback_mode=False"
                )
            else:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Semantic pruning will degrade to graph distance only. "
                    "Install: pip install sentence-transformers"
                )
                return

        # Initialize model
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embeddings_enabled = True
            logger.info("Semantic embeddings enabled (all-MiniLM-L6-v2)")
        except Exception as e:
            if require_embeddings:
                raise SemanticMemoryError(f"Failed to load embedding model: {e}")
            else:
                logger.error(f"Failed to load embedding model: {e}. Continuing without embeddings.")
                self.model = None

    def embed_node(self, node_id: str, content: str):
        if not self.model: return
        embedding = self.model.encode(content)
        self.vectors[node_id] = embedding

    def find_similar(self, query: str, top_k=3):
        if not self.model or not self.vectors: return []
        query_vec = self.model.encode(query)

        results = []
        for nid, vec in self.vectors.items():
            score = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            results.append((score, nid))

        return sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
