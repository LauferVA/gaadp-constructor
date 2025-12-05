"""
REAL-TIME VISUALIZATION SERVER

WebSocket server that broadcasts graph state changes to connected dashboards.
Provides a "mission control" view of DAG construction in real-time.

Supports two modes:
- Production mode (--viz): Single DAG visualization
- Dev mode (--dev): Side-by-side comparison of BASELINE vs TREATMENT DAGs

Usage:
    python gaadp_main.py --viz "your requirement here"    # Production
    python gaadp_main.py --dev "your requirement here"    # Dev comparison
    Then open http://localhost:8766 in your browser
"""
import asyncio
import json
import logging
import webbrowser
from pathlib import Path
from typing import Set, Dict, Any, Optional, List
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

logger = logging.getLogger("GAADP.VizServer")

# Global state for the visualization server
_viz_server: Optional['VizServer'] = None


def _create_empty_graph_state() -> Dict[str, Any]:
    """Create an empty graph state structure."""
    return {
        "nodes": {},
        "edges": [],
        "agents": {},
        "events": [],
        "stats": {
            "total_cost": 0.0,
            "nodes_created": 0,
            "nodes_verified": 0,
            "nodes_failed": 0,
            "iterations": 0
        }
    }


class VizServer:
    """
    WebSocket server for real-time graph visualization.

    Broadcasts events to all connected dashboard clients.
    Supports session-based tracking for comparison mode (baseline/treatment).
    """

    def __init__(self, host: str = "localhost", port: int = 8765, dev_mode: bool = False):
        self.host = host
        self.port = port
        self.dev_mode = dev_mode
        self.clients: Set = set()

        # Session-based graph states for comparison mode
        # In dev mode, tracks both "baseline" and "treatment"
        # In prod mode, uses "default" session
        self.graph_states: Dict[str, Dict[str, Any]] = {
            "default": _create_empty_graph_state()
        }
        if dev_mode:
            self.graph_states["baseline"] = _create_empty_graph_state()
            self.graph_states["treatment"] = _create_empty_graph_state()

        # Backward compatibility: alias for single-session mode
        self.graph_state = self.graph_states["default"]

        self._server = None
        self._http_server = None
        self._http_thread = None
        self._active_session: str = "default"  # Current session being populated

    async def start(self):
        """Start the WebSocket server."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets not installed. Run: pip install websockets")
            return

        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port
        )

        # Start HTTP server for dashboard in a thread
        self._start_http_server()

        logger.info(f"Visualization server started at ws://{self.host}:{self.port}")
        logger.info(f"Dashboard available at http://{self.host}:{self.port + 1}")

        # Open browser
        webbrowser.open(f"http://{self.host}:{self.port + 1}")

    def _start_http_server(self):
        """Start HTTP server to serve the dashboard HTML."""
        dashboard_dir = Path(__file__).parent
        dev_mode = self.dev_mode

        class DashboardHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(dashboard_dir), **kwargs)

            def do_GET(self):
                # Redirect root to proper dashboard URL with mode parameter
                if self.path == "/" or self.path == "/index.html":
                    mode = "comparison" if dev_mode else "single"
                    redirect_url = f"/viz_dashboard_unified.html?mode={mode}"
                    self.send_response(302)  # HTTP redirect
                    self.send_header('Location', redirect_url)
                    self.end_headers()
                    return
                return super().do_GET()

            def log_message(self, format, *args):
                pass  # Suppress HTTP logs

        self._http_server = HTTPServer((self.host, self.port + 1), DashboardHandler)
        self._http_thread = threading.Thread(target=self._http_server.serve_forever)
        self._http_thread.daemon = True
        self._http_thread.start()

    async def stop(self):
        """Stop the server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._http_server:
            self._http_server.shutdown()

    async def _handle_client(self, websocket, path=None):
        """Handle a new WebSocket client connection.

        Note: path parameter is optional for websockets 10.0+ compatibility.
        """
        self.clients.add(websocket)
        logger.debug(f"Client connected. Total clients: {len(self.clients)}")

        try:
            # Send current state to new client
            if self.dev_mode:
                # Dev mode: send both baseline and treatment states
                await websocket.send(json.dumps({
                    "type": "full_state",
                    "dev_mode": True,
                    "data": {
                        "baseline": self.graph_states.get("baseline", _create_empty_graph_state()),
                        "treatment": self.graph_states.get("treatment", _create_empty_graph_state())
                    },
                    "timestamp": datetime.now().isoformat()
                }))
            else:
                # Prod mode: send single state
                await websocket.send(json.dumps({
                    "type": "full_state",
                    "dev_mode": False,
                    "data": self.graph_state,
                    "timestamp": datetime.now().isoformat()
                }))

            # Keep connection alive and handle incoming messages
            async for message in websocket:
                # Handle client requests (e.g., requesting full state)
                try:
                    msg = json.loads(message)
                    msg_type = msg.get("type")

                    if msg_type == "request_state":
                        await websocket.send(json.dumps({
                            "type": "full_state",
                            "data": self.graph_state,
                            "timestamp": datetime.now().isoformat()
                        }))

                    elif msg_type == "user_response":
                        # User responding to a question from the model
                        question_id = msg.get("question_id")
                        response = msg.get("response")
                        if question_id and response:
                            self.receive_user_response(question_id, response)

                    elif msg_type == "user_query":
                        # User querying about nodes/edges/agents
                        query = msg.get("query", "")
                        target_type = msg.get("target_type")
                        target_id = msg.get("target_id")
                        routing = msg.get("routing", "auto")
                        await self.on_user_query(query, target_type, target_id, routing)

                    elif msg_type == "export_dag":
                        # User requesting DAG export
                        task_name = msg.get("task_name", "manual_export")
                        from infrastructure.viz_server import export_dag_to_json
                        result = export_dag_to_json(task_name=task_name)
                        await websocket.send(json.dumps({
                            "type": "dag_exported",
                            "path": str(result) if result else None,
                            "timestamp": datetime.now().isoformat()
                        }))

                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.debug(f"Client disconnected: {e}")
        finally:
            self.clients.discard(websocket)

    async def broadcast(self, event_type: str, data: Dict[str, Any], session: str = None):
        """Broadcast an event to all connected clients.

        Args:
            event_type: Type of event (node_created, edge_created, etc.)
            data: Event data payload
            session: Session ID for dev mode ("baseline" or "treatment")
        """
        if not self.clients:
            return

        # Include session info for dev mode
        message_data = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

        if self.dev_mode and session:
            message_data["session"] = session

        message = json.dumps(message_data)

        # Send to all clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except Exception:
                disconnected.add(client)

        # Remove disconnected clients
        self.clients -= disconnected

    # =========================================================================
    # SESSION MANAGEMENT (for dev mode comparisons)
    # =========================================================================

    def set_active_session(self, session: str):
        """Set the active session for subsequent events.

        Args:
            session: "baseline", "treatment", or "default"
        """
        if session not in self.graph_states:
            if self.dev_mode and session in ("baseline", "treatment"):
                self.graph_states[session] = _create_empty_graph_state()
            else:
                logger.warning(f"Unknown session: {session}, using default")
                session = "default"

        self._active_session = session
        self.graph_state = self.graph_states[session]
        logger.debug(f"Active session set to: {session}")

    def reset_session(self, session: str):
        """Reset a session's graph state (for starting a new run).

        Args:
            session: Session ID to reset
        """
        if session in self.graph_states:
            self.graph_states[session] = _create_empty_graph_state()
            if session == self._active_session:
                self.graph_state = self.graph_states[session]
            logger.debug(f"Session {session} reset")

    def get_session_state(self, session: str) -> Dict[str, Any]:
        """Get the graph state for a specific session."""
        return self.graph_states.get(session, _create_empty_graph_state())

    # =========================================================================
    # EVENT METHODS (called by GraphRuntime)
    # =========================================================================

    async def on_node_created(self, node_id: str, node_type: str, content: str, metadata: Dict):
        """Called when a new node is created."""
        self.graph_state["nodes"][node_id] = {
            "id": node_id,
            "type": node_type,
            "status": "PENDING",
            "content": content[:200] + "..." if len(content) > 200 else content,
            "metadata": metadata,
            "created_at": datetime.now().isoformat()
        }
        self.graph_state["stats"]["nodes_created"] += 1

        await self.broadcast("node_created", {
            "node": self.graph_state["nodes"][node_id]
        }, session=self._active_session if self.dev_mode else None)

        self._add_event("node_created", f"Created {node_type} node {node_id[:8]}")

    async def on_node_status_changed(self, node_id: str, old_status: str, new_status: str, reason: str = ""):
        """Called when a node's status changes."""
        if node_id in self.graph_state["nodes"]:
            self.graph_state["nodes"][node_id]["status"] = new_status
            self.graph_state["nodes"][node_id]["status_reason"] = reason

            if new_status == "VERIFIED":
                self.graph_state["stats"]["nodes_verified"] += 1
            elif new_status == "FAILED":
                self.graph_state["stats"]["nodes_failed"] += 1

        await self.broadcast("node_status_changed", {
            "node_id": node_id,
            "old_status": old_status,
            "new_status": new_status,
            "reason": reason
        }, session=self._active_session if self.dev_mode else None)

        self._add_event("status_change", f"{node_id[:8]}: {old_status} → {new_status}")

    async def on_edge_created(self, source_id: str, target_id: str, edge_type: str):
        """Called when a new edge is created."""
        edge = {
            "source": source_id,
            "target": target_id,
            "type": edge_type,
            "created_at": datetime.now().isoformat()
        }
        self.graph_state["edges"].append(edge)

        await self.broadcast("edge_created", {"edge": edge}, session=self._active_session if self.dev_mode else None)

        self._add_event("edge_created", f"{edge_type}: {source_id[:8]} → {target_id[:8]}")

    async def on_agent_started(self, agent_role: str, node_id: str, context_node_ids: list = None):
        """Called when an agent starts processing a node.

        Args:
            agent_role: The agent type (ARCHITECT, BUILDER, VERIFIER, etc.)
            node_id: The node being processed
            context_node_ids: List of node IDs visible to this agent (its "field of view")
        """
        context_node_ids = context_node_ids or [node_id]

        self.graph_state["agents"][agent_role] = {
            "role": agent_role,
            "status": "working",
            "current_node": node_id,
            "context_nodes": context_node_ids,
            "started_at": datetime.now().isoformat()
        }

        await self.broadcast("agent_started", {
            "agent_role": agent_role,
            "node_id": node_id,
            "context_nodes": context_node_ids
        }, session=self._active_session if self.dev_mode else None)

        self._add_event("agent_started", f"{agent_role} started on {node_id[:8]} (context: {len(context_node_ids)} nodes)")

    async def on_agent_finished(self, agent_role: str, node_id: str, success: bool, cost: float):
        """Called when an agent finishes processing."""
        if agent_role in self.graph_state["agents"]:
            self.graph_state["agents"][agent_role]["status"] = "idle"
            self.graph_state["agents"][agent_role]["current_node"] = None
            self.graph_state["agents"][agent_role]["context_nodes"] = []
            self.graph_state["agents"][agent_role]["last_result"] = "success" if success else "failed"

        # Store which agent processed this node (for tooltip display)
        if node_id in self.graph_state["nodes"]:
            self.graph_state["nodes"][node_id]["processed_by"] = agent_role
            self.graph_state["nodes"][node_id]["process_cost"] = cost

        self.graph_state["stats"]["total_cost"] += cost

        await self.broadcast("agent_finished", {
            "agent_role": agent_role,
            "node_id": node_id,
            "success": success,
            "cost": cost
        }, session=self._active_session if self.dev_mode else None)

        status = "✓" if success else "✗"
        self._add_event("agent_finished", f"{agent_role} {status} on {node_id[:8]} (${cost:.4f})")

    async def on_iteration(self, iteration: int, processable_count: int):
        """Called at the start of each execution iteration."""
        self.graph_state["stats"]["iterations"] = iteration

        await self.broadcast("iteration", {
            "iteration": iteration,
            "processable_count": processable_count
        }, session=self._active_session if self.dev_mode else None)

    async def on_error(self, node_id: str, error: str):
        """Called when an error occurs."""
        await self.broadcast("error", {
            "node_id": node_id,
            "error": error
        }, session=self._active_session if self.dev_mode else None)

        self._add_event("error", f"Error on {node_id[:8]}: {error[:50]}")

    async def on_complete(self, stats: Dict[str, Any]):
        """Called when execution completes."""
        await self.broadcast("complete", {"stats": stats}, session=self._active_session if self.dev_mode else None)
        self._add_event("complete", f"Execution complete: {stats['nodes_processed']} nodes processed")

    # =========================================================================
    # Q&A METHODS (bidirectional communication)
    # =========================================================================

    async def ask_user(self, question: str, context: Dict[str, Any] = None,
                       options: List[str] = None, timeout: float = 300.0) -> Optional[str]:
        """
        Ask the user a question via the dashboard and wait for response.

        Args:
            question: The question to ask
            context: Optional context (node_id, agent_role, etc.)
            options: Optional list of predefined answer options
            timeout: How long to wait for response (seconds)

        Returns:
            User's response string, or None if timeout/no response
        """
        question_id = f"q_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Store pending question
        if not hasattr(self, '_pending_questions'):
            self._pending_questions = {}
            self._question_responses = {}

        self._pending_questions[question_id] = {
            "question": question,
            "context": context or {},
            "options": options,
            "asked_at": datetime.now().isoformat()
        }

        # Broadcast question to dashboard
        await self.broadcast("question_from_model", {
            "question_id": question_id,
            "question": question,
            "context": context or {},
            "options": options
        })

        self._add_event("question", f"Model asks: {question[:50]}...")

        # Wait for response (with timeout)
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            if question_id in self._question_responses:
                response = self._question_responses.pop(question_id)
                del self._pending_questions[question_id]
                return response
            await asyncio.sleep(0.5)

        # Timeout - remove pending question
        self._pending_questions.pop(question_id, None)
        logger.warning(f"Question timeout: {question[:50]}")
        return None

    def receive_user_response(self, question_id: str, response: str):
        """
        Receive a response from the user via dashboard.

        For graph-native Socratic Q&A, this also stores context about
        which CLARIFICATION node the answer resolves, enabling the runtime
        to create RESOLVED_BY edges.

        The answer is stored with metadata for later processing:
        - response: The user's answer text
        - clarification_node_id: The CLARIFICATION node this answers (if any)
        - timestamp: When the answer was received
        """
        if not hasattr(self, '_question_responses'):
            self._question_responses = {}

        # Get context from pending question (includes clarification_node_id if set)
        context = {}
        if hasattr(self, '_pending_questions') and question_id in self._pending_questions:
            context = self._pending_questions[question_id].get('context', {})

        # Store response with metadata for graph integration
        self._question_responses[question_id] = {
            "response": response,
            "clarification_node_id": context.get('clarification_node_id'),
            "feature": context.get('feature'),
            "question_id_canonical": context.get('question_id'),  # L2.01, L3.03, etc.
            "timestamp": datetime.now().isoformat(),
            "source": "user"
        }

        self._add_event("answer", f"User responds: {response[:50]}...")

    def get_pending_answer(self, question_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a pending answer for a question (for graph-native Q&A integration).

        Returns dict with:
        - response: The answer text
        - clarification_node_id: Node ID to create RESOLVED_BY edge to
        - timestamp: When answered

        Returns None if no answer pending.
        """
        if not hasattr(self, '_question_responses'):
            return None

        answer_data = self._question_responses.get(question_id)
        if answer_data:
            # Return and remove (consumed)
            del self._question_responses[question_id]
            return answer_data
        return None

    def get_all_pending_answers(self) -> List[Dict[str, Any]]:
        """
        Get all pending answers (for batch processing by runtime).

        Returns list of answer dicts, each with:
        - question_id: The question this answers
        - response: The answer text
        - clarification_node_id: Node ID to create RESOLVED_BY edge to

        Clears the pending answers after returning.
        """
        if not hasattr(self, '_question_responses'):
            return []

        answers = []
        for qid, answer_data in self._question_responses.items():
            if isinstance(answer_data, dict):
                answers.append({
                    "question_id": qid,
                    **answer_data
                })
            else:
                # Legacy format (just string response)
                answers.append({
                    "question_id": qid,
                    "response": answer_data,
                    "clarification_node_id": None,
                    "timestamp": datetime.now().isoformat()
                })

        self._question_responses = {}
        return answers

    async def on_user_query(self, query: str, target_type: str = None,
                           target_id: str = None, routing: str = "auto") -> Dict[str, Any]:
        """
        Handle a query from the user about the graph state.

        Args:
            query: The user's query text
            target_type: "node", "edge", "agent", or None for general
            target_id: Specific ID to query about
            routing: Where to route the query ("auto", "architect", "socrates", "researcher", "builder", "system")

        Returns:
            Response dict with relevant information
        """
        response = {
            "query": query,
            "target_type": target_type,
            "target_id": target_id,
            "routing": routing,
            "timestamp": datetime.now().isoformat()
        }

        # If routing to a specific agent, store for agent pickup
        if routing not in ("auto", "system", None):
            # Store the query for the specified agent to pick up
            if not hasattr(self, '_pending_user_queries'):
                self._pending_user_queries = []

            self._pending_user_queries.append({
                "query": query,
                "routing": routing,
                "timestamp": datetime.now().isoformat()
            })

            response["data"] = {
                "status": "queued",
                "message": f"Query routed to {routing.upper()}. Response will appear when the agent processes it.",
                "routing": routing
            }
            await self.broadcast("query_response", response)
            return response

        # System query - handle directly
        if target_type == "node" and target_id:
            node = self.graph_state["nodes"].get(target_id)
            if node:
                response["data"] = node
                response["edges"] = [e for e in self.graph_state["edges"]
                                    if e["source"] == target_id or e["target"] == target_id]
            else:
                response["error"] = f"Node {target_id} not found"

        elif target_type == "agent" and target_id:
            agent = self.graph_state["agents"].get(target_id)
            if agent:
                response["data"] = agent
            else:
                response["error"] = f"Agent {target_id} not found"

        elif target_type == "edge":
            response["data"] = self.graph_state["edges"]

        else:
            # General query - return summary
            response["data"] = {
                "nodes_count": len(self.graph_state["nodes"]),
                "edges_count": len(self.graph_state["edges"]),
                "stats": self.graph_state["stats"],
                "active_agents": [a for a in self.graph_state["agents"].values()
                                 if a.get("status") == "working"]
            }

        await self.broadcast("query_response", response)
        return response

    async def send_chat_message(self, message: str, agent_role: str = "SYSTEM"):
        """
        Send a chat message from an agent to the dashboard.

        Args:
            message: The message text
            agent_role: The agent sending the message
        """
        await self.broadcast("chat_message", {
            "message": message,
            "agent_role": agent_role,
            "timestamp": datetime.now().isoformat()
        })
        self._add_event("chat", f"{agent_role}: {message[:50]}...")

    def get_pending_user_queries(self, agent_role: str = None) -> List[Dict[str, Any]]:
        """
        Get pending user queries, optionally filtered by target agent.

        Args:
            agent_role: Filter to queries routed to this agent (case-insensitive)

        Returns:
            List of pending query dicts
        """
        if not hasattr(self, '_pending_user_queries'):
            return []

        if agent_role:
            agent_lower = agent_role.lower()
            matching = [q for q in self._pending_user_queries
                       if q.get("routing", "").lower() == agent_lower]
            # Remove matched queries from pending
            self._pending_user_queries = [q for q in self._pending_user_queries
                                         if q.get("routing", "").lower() != agent_lower]
            return matching

        # Return all and clear
        queries = self._pending_user_queries
        self._pending_user_queries = []
        return queries

    def _add_event(self, event_type: str, message: str):
        """Add an event to the event log."""
        self.graph_state["events"].append({
            "type": event_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 100 events
        if len(self.graph_state["events"]) > 100:
            self.graph_state["events"] = self.graph_state["events"][-100:]


def get_viz_server() -> Optional[VizServer]:
    """Get the global visualization server instance."""
    global _viz_server
    return _viz_server


async def start_viz_server(host: str = "localhost", port: int = 8765, dev_mode: bool = False) -> VizServer:
    """Start the global visualization server.

    Args:
        host: Host to bind to
        port: WebSocket port (HTTP will be port + 1)
        dev_mode: If True, enables comparison dashboard with baseline/treatment sessions
    """
    global _viz_server
    _viz_server = VizServer(host, port, dev_mode=dev_mode)
    await _viz_server.start()
    return _viz_server


async def stop_viz_server():
    """Stop the global visualization server."""
    global _viz_server
    if _viz_server:
        await _viz_server.stop()
        _viz_server = None


# =============================================================================
# DAG EXPORT FUNCTIONALITY
# =============================================================================

def export_dag_to_json(output_path: Path = None, task_name: str = None) -> Optional[Path]:
    """
    Export the current DAG state to a JSON file.

    Args:
        output_path: Path to save the JSON file. If None, auto-generates in logs/dags/
        task_name: Name of the task (used for filename if output_path is None)

    Returns:
        Path to the saved file, or None if no server running
    """
    global _viz_server
    if not _viz_server:
        logger.warning("No visualization server running - cannot export DAG")
        return None

    # Prepare output directory
    if output_path is None:
        dag_dir = Path("logs/dags")
        dag_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = (task_name or "unknown").replace(" ", "_").replace("/", "_")[:50]
        output_path = dag_dir / f"dag_{timestamp}_{safe_name}.json"

    # Build export data
    state = _viz_server.graph_state
    export_data = {
        "metadata": {
            "task_name": task_name,
            "exported_at": datetime.now().isoformat(),
            "stats": state["stats"]
        },
        "nodes": list(state["nodes"].values()),
        "edges": state["edges"],
        "events": state["events"][-50:],  # Last 50 events
        "agents": state["agents"]
    }

    # Save to file
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    logger.info(f"DAG exported to {output_path}")
    return output_path


class DAGExporter:
    """Helper class for exporting multiple DAGs during benchmark runs."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("logs/dags")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exported_dags = []

    def export(self, task_name: str, task_id: str = None) -> Optional[Path]:
        """Export current DAG state for a specific task."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = task_name.replace(" ", "_").replace("/", "_")[:50]
        task_id_part = f"_{task_id}" if task_id else ""

        output_path = self.output_dir / f"dag{task_id_part}_{safe_name}_{timestamp}.json"

        result = export_dag_to_json(output_path, task_name)
        if result:
            self.exported_dags.append({
                "task_name": task_name,
                "task_id": task_id,
                "path": str(result),
                "timestamp": timestamp
            })
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all exported DAGs."""
        return {
            "total_exported": len(self.exported_dags),
            "output_dir": str(self.output_dir),
            "dags": self.exported_dags
        }
