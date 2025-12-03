"""
REAL-TIME VISUALIZATION SERVER

WebSocket server that broadcasts graph state changes to connected dashboards.
Provides a "mission control" view of DAG construction in real-time.

Usage:
    python main.py --viz "your requirement here"
    Then open http://localhost:8765 in your browser
"""
import asyncio
import json
import logging
import webbrowser
from pathlib import Path
from typing import Set, Dict, Any, Optional
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

logger = logging.getLogger("GAADP.VizServer")

# Global state for the visualization server
_viz_server: Optional['VizServer'] = None


class VizServer:
    """
    WebSocket server for real-time graph visualization.

    Broadcasts events to all connected dashboard clients.
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set = set()
        self.graph_state: Dict[str, Any] = {
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
        self._server = None
        self._http_server = None
        self._http_thread = None

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

        class DashboardHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(dashboard_dir), **kwargs)

            def do_GET(self):
                if self.path == "/" or self.path == "/index.html":
                    self.path = "/viz_dashboard.html"
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

    async def _handle_client(self, websocket, path):
        """Handle a new WebSocket client connection."""
        self.clients.add(websocket)
        logger.debug(f"Client connected. Total clients: {len(self.clients)}")

        try:
            # Send current state to new client
            await websocket.send(json.dumps({
                "type": "full_state",
                "data": self.graph_state,
                "timestamp": datetime.now().isoformat()
            }))

            # Keep connection alive and handle incoming messages
            async for message in websocket:
                # Handle client requests (e.g., requesting full state)
                try:
                    msg = json.loads(message)
                    if msg.get("type") == "request_state":
                        await websocket.send(json.dumps({
                            "type": "full_state",
                            "data": self.graph_state,
                            "timestamp": datetime.now().isoformat()
                        }))
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.debug(f"Client disconnected: {e}")
        finally:
            self.clients.discard(websocket)

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        """Broadcast an event to all connected clients."""
        if not self.clients:
            return

        message = json.dumps({
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })

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
        })

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
        })

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

        await self.broadcast("edge_created", {"edge": edge})

        self._add_event("edge_created", f"{edge_type}: {source_id[:8]} → {target_id[:8]}")

    async def on_agent_started(self, agent_role: str, node_id: str):
        """Called when an agent starts processing a node."""
        self.graph_state["agents"][agent_role] = {
            "role": agent_role,
            "status": "working",
            "current_node": node_id,
            "started_at": datetime.now().isoformat()
        }

        await self.broadcast("agent_started", {
            "agent_role": agent_role,
            "node_id": node_id
        })

        self._add_event("agent_started", f"{agent_role} started on {node_id[:8]}")

    async def on_agent_finished(self, agent_role: str, node_id: str, success: bool, cost: float):
        """Called when an agent finishes processing."""
        if agent_role in self.graph_state["agents"]:
            self.graph_state["agents"][agent_role]["status"] = "idle"
            self.graph_state["agents"][agent_role]["current_node"] = None
            self.graph_state["agents"][agent_role]["last_result"] = "success" if success else "failed"

        self.graph_state["stats"]["total_cost"] += cost

        await self.broadcast("agent_finished", {
            "agent_role": agent_role,
            "node_id": node_id,
            "success": success,
            "cost": cost
        })

        status = "✓" if success else "✗"
        self._add_event("agent_finished", f"{agent_role} {status} on {node_id[:8]} (${cost:.4f})")

    async def on_iteration(self, iteration: int, processable_count: int):
        """Called at the start of each execution iteration."""
        self.graph_state["stats"]["iterations"] = iteration

        await self.broadcast("iteration", {
            "iteration": iteration,
            "processable_count": processable_count
        })

    async def on_error(self, node_id: str, error: str):
        """Called when an error occurs."""
        await self.broadcast("error", {
            "node_id": node_id,
            "error": error
        })

        self._add_event("error", f"Error on {node_id[:8]}: {error[:50]}")

    async def on_complete(self, stats: Dict[str, Any]):
        """Called when execution completes."""
        await self.broadcast("complete", {"stats": stats})
        self._add_event("complete", f"Execution complete: {stats['nodes_processed']} nodes processed")

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


async def start_viz_server(host: str = "localhost", port: int = 8765) -> VizServer:
    """Start the global visualization server."""
    global _viz_server
    _viz_server = VizServer(host, port)
    await _viz_server.start()
    return _viz_server


async def stop_viz_server():
    """Stop the global visualization server."""
    global _viz_server
    if _viz_server:
        await _viz_server.stop()
        _viz_server = None
