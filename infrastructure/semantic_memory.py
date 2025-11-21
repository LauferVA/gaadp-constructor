"""
SEMANTIC MEMORY LAYER
Stores vector embeddings of code for semantic retrieval.
"""
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class SemanticMemory:
    def __init__(self):
        if SentenceTransformer:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            print("⚠️ SentenceTransformers not installed. Semantic layer disabled.")
            self.model = None
        self.vectors = {} # node_id -> numpy array

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
