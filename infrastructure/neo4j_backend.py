"""
NEO4J GRAPH DATABASE BACKEND
Production-scale alternative to NetworkX.
"""
import logging
import datetime
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger("Neo4jBackend")

# Try to import neo4j, gracefully degrade if not available
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j driver not installed. Neo4j backend unavailable.")


class GraphBackend(ABC):
    """Abstract interface for graph backends."""

    @abstractmethod
    def add_node(self, node_id: str, node_type: str, content: Any, metadata: Dict) -> None:
        pass

    @abstractmethod
    def add_edge(self, source: str, target: str, edge_type: str, signed_by: str, signature: str, previous_hash: str) -> None:
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict]:
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str, radius: int) -> List[Dict]:
        pass

    @abstractmethod
    def has_path(self, source: str, target: str) -> bool:
        pass


class Neo4jBackend(GraphBackend):
    """
    Neo4j implementation of the graph backend.
    Designed for production-scale deployments.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j driver not installed. Run: pip install neo4j")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Create indexes for performance."""
        with self.driver.session(database=self.database) as session:
            # Index on node_id for fast lookups
            session.run("CREATE INDEX node_id_idx IF NOT EXISTS FOR (n:Node) ON (n.node_id)")
            # Index on type for filtering
            session.run("CREATE INDEX node_type_idx IF NOT EXISTS FOR (n:Node) ON (n.type)")

    def close(self):
        """Close the driver connection."""
        self.driver.close()

    def add_node(self, node_id: str, node_type: str, content: Any, metadata: Dict = None) -> None:
        """Add a node to Neo4j."""
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (n:Node {node_id: $node_id})
                SET n.type = $type,
                    n.content = $content,
                    n.status = 'PENDING',
                    n.metadata = $metadata,
                    n.created_at = $created_at
                """,
                node_id=node_id,
                type=node_type,
                content=str(content),
                metadata=str(metadata or {}),
                created_at=datetime.datetime.utcnow().isoformat()
            )
        logger.info(f"Neo4j: Added node {node_id}")

    def add_edge(self, source: str, target: str, edge_type: str, signed_by: str,
                 signature: str, previous_hash: str = None) -> None:
        """Add an edge to Neo4j."""
        # First check for cycles if this is a DEPENDS_ON edge
        if edge_type == "DEPENDS_ON" and self.has_path(target, source):
            raise ValueError(f"Cycle detected! Cannot link {source} -> {target}")

        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (s:Node {node_id: $source})
                MATCH (t:Node {node_id: $target})
                MERGE (s)-[r:EDGE {type: $type}]->(t)
                SET r.signed_by = $signed_by,
                    r.signature = $signature,
                    r.previous_hash = $previous_hash,
                    r.created_at = $created_at
                """,
                source=source,
                target=target,
                type=edge_type,
                signed_by=signed_by,
                signature=signature,
                previous_hash=previous_hash,
                created_at=datetime.datetime.utcnow().isoformat()
            )

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Retrieve a node by ID."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (n:Node {node_id: $node_id}) RETURN n",
                node_id=node_id
            )
            record = result.single()
            if record:
                node = record["n"]
                return dict(node)
            return None

    def get_neighbors(self, node_id: str, radius: int, max_tokens: int = 6000) -> List[Dict]:
        """Get neighborhood within radius, respecting token limits."""
        with self.driver.session(database=self.database) as session:
            # Use variable-length path pattern for radius
            result = session.run(
                """
                MATCH (center:Node {node_id: $node_id})
                MATCH path = (center)-[*0..$radius]-(neighbor:Node)
                RETURN DISTINCT neighbor
                LIMIT 100
                """,
                node_id=node_id,
                radius=radius
            )

            neighbors = []
            current_tokens = 0

            for record in result:
                node = dict(record["neighbor"])
                content = node.get("content", "")
                tokens = len(str(content)) // 4

                if current_tokens + tokens > max_tokens:
                    break

                neighbors.append(node)
                current_tokens += tokens

            return neighbors

    def has_path(self, source: str, target: str) -> bool:
        """Check if a path exists between two nodes."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (s:Node {node_id: $source}), (t:Node {node_id: $target})
                RETURN EXISTS((s)-[*]->(t)) as path_exists
                """,
                source=source,
                target=target
            )
            record = result.single()
            return record["path_exists"] if record else False

    def get_last_verified_signature(self) -> str:
        """Get the signature of the most recent VERIFIES edge."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH ()-[r:EDGE {type: 'VERIFIES'}]->()
                RETURN r.signature as sig
                ORDER BY r.created_at DESC
                LIMIT 1
                """
            )
            record = result.single()
            return record["sig"] if record else "GENESIS"

    def get_materializable_nodes(self) -> List[Dict]:
        """Get all CODE nodes with VERIFIED status."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (n:Node {type: 'CODE', status: 'VERIFIED'})
                RETURN n.node_id as id, n.content as content, n.metadata as metadata
                """
            )
            return [dict(record) for record in result]

    def update_node_status(self, node_id: str, status: str) -> None:
        """Update a node's status."""
        with self.driver.session(database=self.database) as session:
            session.run(
                "MATCH (n:Node {node_id: $node_id}) SET n.status = $status",
                node_id=node_id,
                status=status
            )


class HybridGraphDB:
    """
    Hybrid backend that uses NetworkX for development and Neo4j for production.
    Automatically selects based on configuration.
    """

    def __init__(self, use_neo4j: bool = False, neo4j_config: Dict = None):
        self.use_neo4j = use_neo4j and NEO4J_AVAILABLE

        if self.use_neo4j:
            config = neo4j_config or {}
            self.backend = Neo4jBackend(
                uri=config.get("uri", "bolt://localhost:7687"),
                user=config.get("user", "neo4j"),
                password=config.get("password", "password"),
                database=config.get("database", "neo4j")
            )
            logger.info("Using Neo4j backend")
        else:
            # Fall back to NetworkX-based GraphDB
            from infrastructure.graph_db import GraphDB
            self.backend = GraphDB()
            logger.info("Using NetworkX backend")

    def __getattr__(self, name):
        """Delegate to the underlying backend."""
        return getattr(self.backend, name)
