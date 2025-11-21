#!/usr/bin/env python3
"""
GAADP MCP SERVER
Exposes Graph-Native Physics Engine as MCP tools for Claude Code.
"""
import os
import json
import uuid
from typing import Optional
from fastmcp import FastMCP

from infrastructure.graph_db import GraphDB
from infrastructure.sandbox import CodeSandbox
from infrastructure.version_control import GitController
from core.ontology import NodeType, EdgeType, NodeStatus

# Initialize MCP Server
mcp = FastMCP("GAADP-Physics-Engine")

# Initialize shared infrastructure
db = GraphDB(persistence_path=".gaadp/live_graph.pkl")
sandbox = CodeSandbox(use_docker=False)
git = GitController()


@mcp.tool()
def create_node(node_type: str, content: str, metadata_json: str = "{}") -> str:
    """
    Creates a new node in the GAADP knowledge graph.

    Args:
        node_type: One of REQ, SPEC, PLAN, CODE, TEST, DOC
        content: The node content (requirement text, code, etc.)
        metadata_json: Optional JSON string with metadata

    Returns:
        Node ID on success, error message on failure
    """
    try:
        # Validate node type
        valid_types = [t.value for t in NodeType]
        if node_type not in valid_types:
            return f"PHYSICS VIOLATION: Invalid node_type '{node_type}'. Must be one of: {valid_types}"

        node_id = f"{node_type.lower()}_{uuid.uuid4().hex[:8]}"
        metadata = json.loads(metadata_json)

        db.add_node(node_id, NodeType(node_type), content, metadata=metadata)

        return f"Created node: {node_id}"
    except Exception as e:
        return f"Error creating node: {str(e)}"


@mcp.tool()
def link_nodes(source_id: str, target_id: str, edge_type: str, justification: str) -> str:
    """
    Creates a directed edge between nodes. GraphDB enforces cycle detection.

    Args:
        source_id: Source node ID
        target_id: Target node ID
        edge_type: One of TRACES_TO, DEPENDS_ON, IMPLEMENTS, VERIFIES, DEFINES
        justification: Why this link exists

    Returns:
        Success message or PHYSICS VIOLATION error
    """
    try:
        valid_edges = [e.value for e in EdgeType]
        if edge_type not in valid_edges:
            return f"PHYSICS VIOLATION: Invalid edge_type '{edge_type}'. Must be one of: {valid_edges}"

        # Get previous hash for Merkle chaining
        prev_hash = db.get_last_node_hash()

        # Create a simple signature (in production, use proper agent signing)
        signature = f"mcp_signed_{uuid.uuid4().hex[:8]}"

        db.add_edge(
            source_id, target_id,
            EdgeType(edge_type),
            signed_by="mcp_orchestrator",
            signature=signature,
            previous_hash=prev_hash
        )

        return f"Linked {source_id} --[{edge_type}]--> {target_id}"
    except ValueError as e:
        # Cycle detection triggers ValueError
        return f"PHYSICS VIOLATION: {str(e)}"
    except Exception as e:
        return f"Error linking nodes: {str(e)}"


@mcp.tool()
def get_node(node_id: str) -> str:
    """
    Retrieves a node's content and metadata from the graph.

    Args:
        node_id: The node ID to retrieve

    Returns:
        JSON string with node data or error message
    """
    try:
        if node_id not in db.graph.nodes:
            return f"Node '{node_id}' not found"

        node_data = dict(db.graph.nodes[node_id])
        return json.dumps(node_data, indent=2, default=str)
    except Exception as e:
        return f"Error getting node: {str(e)}"


@mcp.tool()
def query_graph(node_type: Optional[str] = None, status: Optional[str] = None) -> str:
    """
    Queries the graph for nodes matching criteria.

    Args:
        node_type: Filter by node type (REQ, SPEC, CODE, etc.)
        status: Filter by status (PENDING, VERIFIED, REJECTED)

    Returns:
        JSON list of matching node IDs and summaries
    """
    try:
        results = []
        for node_id, data in db.graph.nodes(data=True):
            if node_type and data.get('type') != node_type:
                continue
            if status and data.get('status') != status:
                continue

            content_preview = str(data.get('content', ''))[:100]
            results.append({
                "id": node_id,
                "type": data.get('type'),
                "status": data.get('status'),
                "preview": content_preview
            })

        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error querying graph: {str(e)}"


@mcp.tool()
def run_verification_sandbox(code_node_id: str) -> str:
    """
    Extracts code from a node and runs it in an isolated sandbox.

    Args:
        code_node_id: ID of the CODE node to execute

    Returns:
        Execution results (stdout, stderr, exit_code)
    """
    try:
        if code_node_id not in db.graph.nodes:
            return f"Node '{code_node_id}' not found"

        node_data = db.graph.nodes[code_node_id]
        if node_data.get('type') != NodeType.CODE.value:
            return f"Node '{code_node_id}' is not a CODE node"

        code_content = node_data.get('content', '')

        # Write to temp file and execute
        temp_path = f".gaadp/sandbox/{code_node_id}.py"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)

        with open(temp_path, "w") as f:
            f.write(code_content)

        result = sandbox.run_code(temp_path)

        return json.dumps({
            "exit_code": result['exit_code'],
            "stdout": result['stdout'][:2000],
            "stderr": result['stderr'][:2000]
        }, indent=2)
    except Exception as e:
        return f"Error running sandbox: {str(e)}"


@mcp.tool()
def update_node_status(node_id: str, new_status: str) -> str:
    """
    Updates the status of a node (PENDING, VERIFIED, REJECTED).

    Args:
        node_id: The node to update
        new_status: New status value

    Returns:
        Success message or error
    """
    try:
        valid_statuses = [s.value for s in NodeStatus]
        if new_status not in valid_statuses:
            return f"Invalid status '{new_status}'. Must be one of: {valid_statuses}"

        if node_id not in db.graph.nodes:
            return f"Node '{node_id}' not found"

        db.graph.nodes[node_id]['status'] = new_status
        db._persist()

        return f"Updated {node_id} status to {new_status}"
    except Exception as e:
        return f"Error updating status: {str(e)}"


@mcp.tool()
def git_commit_work(agent_name: str, node_id: str, message: str) -> str:
    """
    Commits current work to Git with proper attribution.

    Args:
        agent_name: Name of the agent/tool making the commit
        node_id: Related node ID for traceability
        message: Commit message

    Returns:
        Commit hash or error message
    """
    try:
        result = git.commit_work(agent_name, node_id, message)
        return f"Committed: {result}" if result else "Commit created"
    except Exception as e:
        return f"Error committing: {str(e)}"


@mcp.tool()
def get_graph_stats() -> str:
    """
    Returns statistics about the current knowledge graph.

    Returns:
        JSON with node count, edge count, type distribution
    """
    try:
        type_counts = {}
        status_counts = {}

        for _, data in db.graph.nodes(data=True):
            node_type = data.get('type', 'UNKNOWN')
            status = data.get('status', 'UNKNOWN')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1

        return json.dumps({
            "total_nodes": db.graph.number_of_nodes(),
            "total_edges": db.graph.number_of_edges(),
            "nodes_by_type": type_counts,
            "nodes_by_status": status_counts
        }, indent=2)
    except Exception as e:
        return f"Error getting stats: {str(e)}"


if __name__ == "__main__":
    mcp.run()
