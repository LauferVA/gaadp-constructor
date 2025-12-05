"""
GAADP Mission Control - Graph-Native Visualization Server

This is the SINGLE SOURCE OF TRUTH for real-time dashboard visualization.
All types, colors, and behaviors are derived from core.ontology.

Key Design Principles:
1. GRAPH-NATIVE: Everything reads from ontology, nothing hardcoded
2. UNIFIED: Single implementation for both single and comparison modes
3. PLAYBACK: Full DAG history stored for event-log-based replay
4. GIT-BASED: Baseline/treatment tracks actual git commits

Usage:
    from infrastructure.mission_control import MissionControl, start_mission_control

    mc = await start_mission_control(mode="single")  # or "comparison"
    await mc.on_node_created(node_id, node_type, content, metadata)
    await mc.stop()
"""

import asyncio
import json
import logging
import subprocess
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Literal

import websockets
from websockets.server import serve

# Graph-Native imports - THE SINGLE SOURCE OF TRUTH
from core.ontology import NodeType, EdgeType, NodeStatus, AGENT_DISPATCH

# Integration with checkpoint system for stored results
from infrastructure.checkpoint import RunResultStore, RunResult

logger = logging.getLogger(__name__)


# =============================================================================
# GRAPH-NATIVE CONFIGURATION (derived from ontology)
# =============================================================================

def load_agent_manifest() -> Dict[str, Any]:
    """Load agent definitions from manifest (graph-native source)."""
    manifest_path = Path(__file__).parent.parent / "config" / "agent_manifest.yaml"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def get_node_types() -> List[str]:
    """Get all node types from ontology."""
    return [nt.value for nt in NodeType]


def get_edge_types() -> List[str]:
    """Get all edge types from ontology."""
    return [et.value for et in EdgeType]


def get_node_statuses() -> List[str]:
    """Get all node statuses from ontology."""
    return [ns.value for ns in NodeStatus]


def get_agent_types() -> List[str]:
    """Get all agent types from AGENT_DISPATCH (graph-native)."""
    agents = set()
    for (node_type, condition), agent in AGENT_DISPATCH.items():
        agents.add(agent)
    return sorted(agents)


# Node type colors (derived from ontology types)
NODE_COLORS: Dict[str, str] = {
    NodeType.REQ.value: "#FF6B6B",           # Red - Requirements
    NodeType.RESEARCH.value: "#9B59B6",      # Purple - Research
    NodeType.CLARIFICATION.value: "#FFB347", # Orange - Clarification
    NodeType.SPEC.value: "#4ECDC4",          # Teal - Specification
    NodeType.PLAN.value: "#95E1D3",          # Light teal - Plan
    NodeType.CODE.value: "#45B7D1",          # Blue - Code
    NodeType.TEST.value: "#98D8C8",          # Mint - Test
    NodeType.TEST_SUITE.value: "#7DCEA0",    # Green - Test Suite
    NodeType.DOC.value: "#A8E6CF",           # Light green - Documentation
    NodeType.ESCALATION.value: "#E74C3C",    # Dark red - Escalation
    NodeType.CLASS.value: "#85C1E9",         # Light blue - Class
    NodeType.FUNCTION.value: "#82E0AA",      # Light green - Function
    NodeType.CALL.value: "#F9E79F",          # Yellow - Call
}

# Node status colors
STATUS_COLORS: Dict[str, str] = {
    NodeStatus.PENDING.value: "#95A5A6",     # Gray
    NodeStatus.PROCESSING.value: "#F39C12",  # Orange
    NodeStatus.BLOCKED.value: "#E74C3C",     # Red
    NodeStatus.TESTING.value: "#9B59B6",     # Purple
    NodeStatus.TESTED.value: "#3498DB",      # Blue
    NodeStatus.VERIFIED.value: "#27AE60",    # Green
    NodeStatus.FAILED.value: "#C0392B",      # Dark red
}

# Edge type colors
EDGE_COLORS: Dict[str, str] = {
    EdgeType.TRACES_TO.value: "#7F8C8D",
    EdgeType.DEPENDS_ON.value: "#E74C3C",
    EdgeType.IMPLEMENTS.value: "#27AE60",
    EdgeType.VERIFIES.value: "#3498DB",
    EdgeType.TESTS.value: "#9B59B6",
    EdgeType.DEFINES.value: "#F39C12",
    EdgeType.BLOCKS.value: "#C0392B",
    EdgeType.FEEDBACK.value: "#E67E22",
    EdgeType.RESOLVED_BY.value: "#1ABC9C",
    EdgeType.RESEARCH_FOR.value: "#8E44AD",
    EdgeType.CONTAINS.value: "#BDC3C7",
    EdgeType.REFERENCES.value: "#95A5A6",
    EdgeType.INHERITS.value: "#2980B9",
}

# Agent colors (derived from manifest)
AGENT_COLORS: Dict[str, str] = {
    "ARCHITECT": "#E74C3C",
    "BUILDER": "#3498DB",
    "VERIFIER": "#27AE60",
    "TESTER": "#9B59B6",
    "RESEARCHER": "#F39C12",
    "RESEARCH_VERIFIER": "#E67E22",
    "DIALECTOR": "#1ABC9C",
    "SOCRATES": "#8E44AD",
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DAGSnapshot:
    """Snapshot of DAG state at a point in time (for playback)."""
    timestamp: str
    event_index: int
    nodes: Dict[str, Dict[str, Any]]
    edges: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionState:
    """State for a single session (baseline or treatment)."""
    name: str
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    agents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=lambda: {
        "total_cost": 0.0,
        "iterations": 0,
        "nodes_created": 0,
        "nodes_verified": 0,
        "nodes_failed": 0,
        "edges_created": 0,
    })
    # DAG history for playback
    snapshots: List[DAGSnapshot] = field(default_factory=list)
    # Git tracking
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": self.nodes,
            "edges": self.edges,
            "events": self.events,
            "agents": self.agents,
            "metrics": self.metrics,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
        }

    def take_snapshot(self) -> DAGSnapshot:
        """Create a snapshot for playback."""
        snapshot = DAGSnapshot(
            timestamp=datetime.now().isoformat(),
            event_index=len(self.events),
            nodes={k: dict(v) for k, v in self.nodes.items()},
            edges=[dict(e) for e in self.edges],
        )
        self.snapshots.append(snapshot)
        return snapshot


# =============================================================================
# MISSION CONTROL SERVER
# =============================================================================

class MissionControl:
    """
    Graph-native visualization server for GAADP Mission Control.

    Supports two modes:
    - single: One DAG panel (production monitoring)
    - comparison: Two DAG panels side-by-side (baseline vs treatment)
    """

    def __init__(
        self,
        mode: Literal["single", "comparison"] = "single",
        ws_port: int = 8765,
        http_port: int = 8766,
    ):
        self.mode = mode
        self.ws_port = ws_port
        self.http_port = http_port

        # Sessions
        self.sessions: Dict[str, SessionState] = {}
        if mode == "single":
            self.sessions["default"] = SessionState(name="default")
            self.active_session = "default"
        else:
            self.sessions["baseline"] = SessionState(name="baseline")
            self.sessions["treatment"] = SessionState(name="treatment")
            self.active_session = "treatment"

        # WebSocket clients
        self.clients: Set[websockets.WebSocketServerProtocol] = set()

        # Servers
        self.ws_server = None
        self.http_server = None

        # Agent manifest (graph-native)
        self.agent_manifest = load_agent_manifest()

        # Chat/dialogue messages
        self.chat_messages: List[Dict[str, Any]] = []

        # Question queue (for Socratic dialogue)
        self.pending_questions: List[Dict[str, Any]] = []
        self.question_responses: Dict[str, Any] = {}

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def get_session(self, name: Optional[str] = None) -> SessionState:
        """Get session by name or active session."""
        name = name or self.active_session
        return self.sessions.get(name, self.sessions[self.active_session])

    def set_active_session(self, name: str):
        """Switch active session (for comparison mode)."""
        if name in self.sessions:
            self.active_session = name
            logger.info(f"Switched active session to: {name}")

    def reset_session(self, name: str):
        """Reset a session to empty state."""
        if name in self.sessions:
            self.sessions[name] = SessionState(name=name)
            logger.info(f"Reset session: {name}")

    # =========================================================================
    # GIT TRACKING
    # =========================================================================

    def _get_git_info(self) -> Dict[str, Optional[str]]:
        """Get current git commit and branch."""
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return {"commit": commit, "branch": branch}
        except Exception:
            return {"commit": None, "branch": None}

    def _get_prior_commit(self, steps_back: int = 1) -> Optional[str]:
        """Get a prior commit SHA (for baseline)."""
        try:
            result = subprocess.check_output(
                ["git", "rev-parse", f"HEAD~{steps_back}"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return result
        except Exception:
            return None

    def _get_commit_message(self, commit: str) -> str:
        """Get commit message for display."""
        try:
            result = subprocess.check_output(
                ["git", "log", "-1", "--format=%s", commit],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return result[:50] + "..." if len(result) > 50 else result
        except Exception:
            return "unknown"

    def set_session_git_info(self, session_name: str, commit: Optional[str] = None):
        """Set git info for a session."""
        session = self.get_session(session_name)
        if commit:
            session.git_commit = commit
        else:
            info = self._get_git_info()
            session.git_commit = info["commit"]
            session.git_branch = info["branch"]

    def setup_git_comparison(self, baseline_steps_back: int = 1):
        """
        Set up git-based baseline vs treatment comparison.

        - baseline: Loads ACTUAL STORED RESULTS from HEAD~{steps_back} commit
        - treatment: Uses current HEAD (will be populated by current run)

        This allows comparing execution between prior and current codebase versions
        using real data, not just metadata labels.

        Gen-3 Enhancement: Now integrates with RunResultStore to load actual
        stored results from prior commits, enabling meaningful regression testing.
        """
        if self.mode != "comparison":
            logger.warning("Git comparison only works in comparison mode")
            return

        # Get commits
        current_info = self._get_git_info()
        prior_commit = self._get_prior_commit(baseline_steps_back)

        # Set treatment session (current run)
        treatment = self.get_session("treatment")
        treatment.git_commit = current_info["commit"]
        treatment.git_branch = current_info["branch"]

        # Set baseline session
        baseline = self.get_session("baseline")
        baseline.git_commit = prior_commit
        baseline.git_branch = f"HEAD~{baseline_steps_back}"

        # Gen-3: Try to load actual stored results for baseline
        baseline_result = None
        if prior_commit:
            run_store = RunResultStore()
            baseline_result = run_store.load_result(prior_commit)

            if baseline_result:
                # Populate baseline metrics from stored results
                baseline.metrics = {
                    "total_cost": baseline_result.total_cost,
                    "iterations": baseline_result.iterations,
                    "nodes_created": baseline_result.nodes_processed,
                    "nodes_verified": baseline_result.verified_count,
                    "nodes_failed": baseline_result.failed_count,
                    "edges_created": 0,  # Not stored in RunResult
                    # Extended metrics from stored data
                    "success": baseline_result.success,
                    "error_count": baseline_result.error_count,
                    "duration_seconds": baseline_result.duration_seconds,
                    "errors_by_category": baseline_result.errors_by_category,
                }

                # Add event indicating baseline loaded from storage
                self._add_event(
                    baseline,
                    "baseline_loaded",
                    f"Loaded stored results from commit {prior_commit[:8]}: "
                    f"{baseline_result.verified_count} verified, "
                    f"{baseline_result.failed_count} failed, "
                    f"${baseline_result.total_cost:.4f} cost"
                )

                logger.info(f"Baseline loaded from stored results:")
                logger.info(f"  Commit: {prior_commit[:8]}")
                logger.info(f"  Success: {baseline_result.success}")
                logger.info(f"  Verified: {baseline_result.verified_count}")
                logger.info(f"  Failed: {baseline_result.failed_count}")
                logger.info(f"  Cost: ${baseline_result.total_cost:.4f}")
                logger.info(f"  Errors: {baseline_result.error_count}")
            else:
                logger.warning(f"No stored results found for baseline commit {prior_commit[:8]}")
                logger.warning("Baseline will be empty - this is expected for the first run after implementing this feature")
                self._add_event(
                    baseline,
                    "baseline_empty",
                    f"No stored results for commit {prior_commit[:8]} - baseline will be populated by current run"
                )

        logger.info(f"Git comparison configured:")
        logger.info(f"  BASELINE: {prior_commit[:8] if prior_commit else 'N/A'} ({self._get_commit_message(prior_commit) if prior_commit else 'N/A'})")
        logger.info(f"  TREATMENT: {current_info['commit'][:8] if current_info['commit'] else 'N/A'} ({current_info['branch']})")
        if baseline_result:
            logger.info(f"  BASELINE DATA: Loaded from storage (meaningful comparison available)")
        else:
            logger.info(f"  BASELINE DATA: Empty (first run - will establish baseline for future comparisons)")

    # =========================================================================
    # EVENT EMITTERS (called by GraphRuntime)
    # =========================================================================

    async def on_node_created(
        self,
        node_id: str,
        node_type: str,
        content: str,
        metadata: Dict[str, Any],
        session: Optional[str] = None,
    ):
        """Handle node creation event."""
        sess = self.get_session(session)
        sess.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "content": content[:500],  # Truncate for display
            "status": NodeStatus.PENDING.value,
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
        }
        sess.metrics["nodes_created"] += 1

        # Add event
        self._add_event(sess, "node_created", f"Created {node_type} node: {node_id[:8]}")

        # Take snapshot for playback
        sess.take_snapshot()

        # Broadcast
        await self._broadcast({
            "type": "node_created",
            "session": sess.name,
            "node": sess.nodes[node_id],
        })

    async def on_node_status_changed(
        self,
        node_id: str,
        old_status: str,
        new_status: str,
        session: Optional[str] = None,
    ):
        """Handle node status change event."""
        sess = self.get_session(session)
        if node_id in sess.nodes:
            sess.nodes[node_id]["status"] = new_status
            sess.nodes[node_id]["updated_at"] = datetime.now().isoformat()

            # Update metrics
            if new_status == NodeStatus.VERIFIED.value:
                sess.metrics["nodes_verified"] += 1
            elif new_status == NodeStatus.FAILED.value:
                sess.metrics["nodes_failed"] += 1

            # Add event
            self._add_event(
                sess,
                "status_change",
                f"{sess.nodes[node_id]['type']} {node_id[:8]}: {old_status} -> {new_status}",
            )

            # Take snapshot
            sess.take_snapshot()

            # Broadcast
            await self._broadcast({
                "type": "node_status_changed",
                "session": sess.name,
                "node_id": node_id,
                "old_status": old_status,
                "new_status": new_status,
            })

    async def on_edge_created(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        session: Optional[str] = None,
    ):
        """Handle edge creation event."""
        sess = self.get_session(session)
        edge = {
            "source": source_id,
            "target": target_id,
            "type": edge_type,
            "created_at": datetime.now().isoformat(),
        }
        sess.edges.append(edge)
        sess.metrics["edges_created"] += 1

        # Add event
        self._add_event(sess, "edge_created", f"Created {edge_type} edge: {source_id[:8]} -> {target_id[:8]}")

        # Take snapshot
        sess.take_snapshot()

        # Broadcast
        await self._broadcast({
            "type": "edge_created",
            "session": sess.name,
            "edge": edge,
        })

    async def on_agent_started(
        self,
        agent_role: str,
        node_id: str,
        session: Optional[str] = None,
    ):
        """Handle agent start event."""
        sess = self.get_session(session)
        sess.agents[agent_role] = {
            "role": agent_role,
            "status": "active",
            "current_node": node_id,
            "started_at": datetime.now().isoformat(),
        }

        # Add event
        self._add_event(sess, "agent_started", f"{agent_role} started processing {node_id[:8]}")

        # Broadcast
        await self._broadcast({
            "type": "agent_started",
            "session": sess.name,
            "agent": sess.agents[agent_role],
        })

    async def on_agent_finished(
        self,
        agent_role: str,
        node_id: str,
        success: bool,
        cost: float = 0.0,
        session: Optional[str] = None,
    ):
        """Handle agent finish event."""
        sess = self.get_session(session)
        if agent_role in sess.agents:
            sess.agents[agent_role]["status"] = "completed" if success else "failed"
            sess.agents[agent_role]["finished_at"] = datetime.now().isoformat()
            sess.agents[agent_role]["cost"] = cost

        sess.metrics["total_cost"] += cost

        # Add event
        status = "completed" if success else "failed"
        self._add_event(sess, "agent_finished", f"{agent_role} {status} on {node_id[:8]} (${cost:.4f})")

        # Broadcast
        await self._broadcast({
            "type": "agent_finished",
            "session": sess.name,
            "agent_role": agent_role,
            "node_id": node_id,
            "success": success,
            "cost": cost,
        })

    async def on_iteration(self, iteration: int, session: Optional[str] = None):
        """Handle iteration event."""
        sess = self.get_session(session)
        sess.metrics["iterations"] = iteration

        # Broadcast
        await self._broadcast({
            "type": "iteration",
            "session": sess.name,
            "iteration": iteration,
        })

    async def on_error(self, error: str, session: Optional[str] = None):
        """Handle error event."""
        sess = self.get_session(session)
        self._add_event(sess, "error", f"Error: {error}")

        await self._broadcast({
            "type": "error",
            "session": sess.name,
            "error": error,
        })

    async def on_complete(self, stats: Dict[str, Any], session: Optional[str] = None):
        """Handle completion event."""
        sess = self.get_session(session)
        sess.metrics.update(stats)
        self._add_event(sess, "complete", f"Execution complete: {stats}")

        await self._broadcast({
            "type": "complete",
            "session": sess.name,
            "stats": stats,
        })

    async def on_chat_message(
        self,
        role: str,
        content: str,
        agent: Optional[str] = None,
    ):
        """Handle chat/dialogue message."""
        message = {
            "role": role,
            "content": content,
            "agent": agent,
            "timestamp": datetime.now().isoformat(),
        }
        self.chat_messages.append(message)

        await self._broadcast({
            "type": "chat_message",
            "message": message,
        })

    # =========================================================================
    # SOCRATIC Q&A (Human-in-the-loop)
    # =========================================================================

    async def ask_user(
        self,
        question_id: str,
        question: str,
        options: Optional[List[str]] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """Ask user a question via the dashboard."""
        q = {
            "id": question_id,
            "question": question,
            "options": options,
            "timestamp": datetime.now().isoformat(),
        }
        self.pending_questions.append(q)

        await self._broadcast({
            "type": "question_from_model",
            "question": q,
        })

        # Wait for response
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            if question_id in self.question_responses:
                return self.question_responses.pop(question_id)
            await asyncio.sleep(0.1)

        return None

    def receive_user_response(self, question_id: str, response: str):
        """Receive user response to a question."""
        self.question_responses[question_id] = response
        # Remove from pending
        self.pending_questions = [
            q for q in self.pending_questions if q["id"] != question_id
        ]

    # =========================================================================
    # PLAYBACK SUPPORT
    # =========================================================================

    def get_snapshot_at_event(
        self,
        event_index: int,
        session: Optional[str] = None,
    ) -> Optional[DAGSnapshot]:
        """Get DAG snapshot at specific event index (for playback)."""
        sess = self.get_session(session)
        for snapshot in reversed(sess.snapshots):
            if snapshot.event_index <= event_index:
                return snapshot
        return None

    def get_all_snapshots(self, session: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all snapshots for playback."""
        sess = self.get_session(session)
        return [s.to_dict() for s in sess.snapshots]

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _add_event(self, session: SessionState, event_type: str, message: str):
        """Add event to session log."""
        session.events.append({
            "type": event_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })

    async def _broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.clients:
            return

        data = json.dumps(message)
        disconnected = set()

        for client in self.clients:
            try:
                await client.send(data)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                disconnected.add(client)

        self.clients -= disconnected

    def _get_full_state(self) -> Dict[str, Any]:
        """Get full state for new client connections."""
        state = {
            "mode": self.mode,
            "sessions": {name: sess.to_dict() for name, sess in self.sessions.items()},
            "active_session": self.active_session,
            "chat_messages": self.chat_messages,
            "pending_questions": self.pending_questions,
            # Graph-native configuration
            "config": {
                "node_types": get_node_types(),
                "edge_types": get_edge_types(),
                "node_statuses": get_node_statuses(),
                "agent_types": get_agent_types(),
                "node_colors": NODE_COLORS,
                "status_colors": STATUS_COLORS,
                "edge_colors": EDGE_COLORS,
                "agent_colors": AGENT_COLORS,
            },
        }
        return state

    # =========================================================================
    # WEBSOCKET HANDLER
    # =========================================================================

    async def _handle_websocket(self, websocket: websockets.WebSocketServerProtocol):
        """Handle WebSocket connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected ({len(self.clients)} total)")

        try:
            # Send full state on connect
            await websocket.send(json.dumps({
                "type": "full_state",
                **self._get_full_state(),
            }))

            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(websocket, data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Message handling error: {e}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client disconnected ({len(self.clients)} total)")

    async def _handle_message(
        self,
        websocket: websockets.WebSocketServerProtocol,
        data: Dict[str, Any],
    ):
        """Handle incoming WebSocket message."""
        msg_type = data.get("type")

        if msg_type == "request_state":
            await websocket.send(json.dumps({
                "type": "full_state",
                **self._get_full_state(),
            }))

        elif msg_type == "user_response":
            question_id = data.get("question_id")
            response = data.get("response")
            if question_id and response:
                self.receive_user_response(question_id, response)

        elif msg_type == "set_active_session":
            session_name = data.get("session")
            if session_name:
                self.set_active_session(session_name)

        elif msg_type == "request_snapshot":
            event_index = data.get("event_index", 0)
            session_name = data.get("session")
            snapshot = self.get_snapshot_at_event(event_index, session_name)
            await websocket.send(json.dumps({
                "type": "snapshot",
                "snapshot": snapshot.to_dict() if snapshot else None,
            }))

        elif msg_type == "export_dag":
            session_name = data.get("session")
            sess = self.get_session(session_name)
            await websocket.send(json.dumps({
                "type": "dag_export",
                "dag": sess.to_dict(),
            }))

    # =========================================================================
    # HTTP SERVER (serves dashboard)
    # =========================================================================

    async def _handle_http(self, reader, writer):
        """Simple HTTP server for dashboard."""
        request = await reader.read(4096)
        request_line = request.decode().split("\r\n")[0]
        path = request_line.split(" ")[1] if " " in request_line else "/"

        # Parse query params
        if "?" in path:
            path, query = path.split("?", 1)
            params = dict(p.split("=") for p in query.split("&") if "=" in p)
        else:
            params = {}

        # Route
        if path == "/" or path == "/mission_control.html":
            # Serve dashboard
            dashboard_path = Path(__file__).parent / "mission_control.html"
            if dashboard_path.exists():
                content = dashboard_path.read_text()
                content_type = "text/html"
            else:
                content = "<h1>Dashboard not found</h1>"
                content_type = "text/html"

        elif path == "/api/config":
            # Return graph-native config as JSON
            content = json.dumps({
                "node_types": get_node_types(),
                "edge_types": get_edge_types(),
                "node_statuses": get_node_statuses(),
                "agent_types": get_agent_types(),
                "node_colors": NODE_COLORS,
                "status_colors": STATUS_COLORS,
                "edge_colors": EDGE_COLORS,
                "agent_colors": AGENT_COLORS,
            })
            content_type = "application/json"

        else:
            content = "Not Found"
            content_type = "text/plain"

        # Send response
        response = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(content.encode())}\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"\r\n"
            f"{content}"
        )
        writer.write(response.encode())
        await writer.drain()
        writer.close()

    # =========================================================================
    # SERVER LIFECYCLE
    # =========================================================================

    async def start(self):
        """Start WebSocket and HTTP servers."""
        # WebSocket server
        self.ws_server = await serve(
            self._handle_websocket,
            "localhost",
            self.ws_port,
        )
        logger.info(f"WebSocket server started on ws://localhost:{self.ws_port}")

        # HTTP server
        self.http_server = await asyncio.start_server(
            self._handle_http,
            "localhost",
            self.http_port,
        )
        logger.info(f"HTTP server started on http://localhost:{self.http_port}")
        logger.info(f"Mode: {self.mode}")

    async def stop(self):
        """Stop servers."""
        if self.ws_server:
            self.ws_server.close()
            await self.ws_server.wait_closed()

        if self.http_server:
            self.http_server.close()
            await self.http_server.wait_closed()

        logger.info("Mission Control stopped")

    async def run_forever(self):
        """Run servers until interrupted."""
        await self.start()
        try:
            await asyncio.Future()  # Run forever
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_mission_control: Optional[MissionControl] = None


async def start_mission_control(
    mode: Literal["single", "comparison"] = "single",
    ws_port: int = 8765,
    http_port: int = 8766,
) -> MissionControl:
    """Start Mission Control singleton."""
    global _mission_control

    if _mission_control is not None:
        logger.warning("Mission Control already running, returning existing instance")
        return _mission_control

    _mission_control = MissionControl(mode=mode, ws_port=ws_port, http_port=http_port)
    await _mission_control.start()
    return _mission_control


async def stop_mission_control():
    """Stop Mission Control singleton."""
    global _mission_control

    if _mission_control:
        await _mission_control.stop()
        _mission_control = None


def get_mission_control() -> Optional[MissionControl]:
    """Get Mission Control singleton."""
    return _mission_control


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GAADP Mission Control")
    parser.add_argument(
        "--mode",
        choices=["single", "comparison"],
        default="single",
        help="Dashboard mode",
    )
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--http-port", type=int, default=8766)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    async def main():
        mc = await start_mission_control(
            mode=args.mode,
            ws_port=args.ws_port,
            http_port=args.http_port,
        )
        print(f"\nMission Control running!")
        print(f"  Dashboard: http://localhost:{args.http_port}")
        print(f"  WebSocket: ws://localhost:{args.ws_port}")
        print(f"  Mode: {args.mode}")
        print("\nPress Ctrl+C to stop\n")

        await mc.run_forever()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
