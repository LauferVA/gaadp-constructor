"""
GAADP ENGINE
The unified orchestrator that ties together:
- TaskScheduler (work dispatch)
- FeedbackController (retry loops)
- Governance Middleware (Treasurer/Sentinel/Curator)
- Event Bus (observability)
"""
import asyncio
import logging
from typing import Optional, Dict, Any

from infrastructure.graph_db import GraphDB
from infrastructure.event_bus import EventBus
from infrastructure.llm_gateway import LLMGateway
from orchestration.scheduler import TaskScheduler, SchedulerConfig
from orchestration.feedback import FeedbackController, FeedbackConfig
from orchestration.governance import (
    create_governance_middleware,
    TreasurerMiddleware,
    SentinelMiddleware,
    CuratorDaemon
)
from core.ontology import AgentRole

logger = logging.getLogger("Engine")


class GADPEngine:
    """
    The Factory Floor Engine.

    This is the main entry point for autonomous code generation.
    It manages:
    - Task discovery and dispatch (TaskScheduler)
    - Failure handling and retries (FeedbackController)
    - Budget and security enforcement (Governance)
    - Background maintenance (Curator)

    Can run in two modes:
    1. Autonomous: Runs continuously until all tasks complete
    2. Step: Processes one wavefront at a time (for debugging)
    """

    def __init__(
        self,
        persistence_path: str = ".gaadp/live_graph.pkl",
        num_workers: int = 5
    ):
        # Core infrastructure
        self.db = GraphDB(persistence_path=persistence_path)
        self.event_bus = EventBus()
        self.gateway = LLMGateway()

        # Orchestration components
        self.scheduler = TaskScheduler(self.db, self.event_bus)
        self.feedback = FeedbackController(self.db, self.event_bus)

        # Governance middleware
        governance = create_governance_middleware(self.db, self.gateway, self.event_bus)
        self.treasurer: TreasurerMiddleware = governance['treasurer']
        self.sentinel: SentinelMiddleware = governance['sentinel']
        self.curator: CuratorDaemon = governance['curator']

        # Register governance hooks with scheduler
        for hook in governance['pre_hooks']:
            self.scheduler.register_pre_hook(hook)
        for hook in governance['post_hooks']:
            self.scheduler.register_post_hook(hook)

        # Configuration
        self.num_workers = num_workers
        self._running = False
        self._tasks = []

    def register_agents(self, agents: Dict[AgentRole, Any]):
        """
        Register agent instances with the scheduler.

        Args:
            agents: Dict mapping AgentRole to agent instance
                    e.g., {AgentRole.ARCHITECT: architect, AgentRole.BUILDER: builder, ...}
        """
        for role, agent in agents.items():
            self.scheduler.register_agent(role, agent)
        logger.info(f"Registered {len(agents)} agents")

    async def start(self):
        """
        Start the engine in autonomous mode.
        Runs until stop() is called or all tasks complete.
        """
        logger.info("🚀 Starting GAADP Engine")
        self._running = True

        # Start all components concurrently
        self._tasks = [
            asyncio.create_task(self.event_bus.start()),
            asyncio.create_task(self.scheduler.start(self.num_workers)),
            asyncio.create_task(self.feedback.start()),
            asyncio.create_task(self.curator.start()),
        ]

        logger.info("Engine components started:")
        logger.info(f"  - Scheduler: {self.num_workers} workers")
        logger.info(f"  - Feedback Controller: active")
        logger.info(f"  - Curator Daemon: {self.curator.config.interval_seconds}s interval")
        logger.info(f"  - Budget: ${self.treasurer.config.project_total_limit_usd}")

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Engine tasks cancelled")

    def stop(self):
        """Stop the engine gracefully."""
        logger.info("Stopping GAADP Engine")
        self._running = False

        self.event_bus.stop()
        self.scheduler.stop()
        self.feedback.stop()
        self.curator.stop()

        for task in self._tasks:
            task.cancel()

    async def run_until_complete(self, timeout: Optional[float] = None):
        """
        Run until all PENDING nodes are processed or timeout.

        Args:
            timeout: Maximum time to wait (seconds)
        """
        logger.info("Running until complete...")

        async def _wait_for_completion():
            while True:
                # Check if any work remains
                pending = [
                    n for n, d in self.db.graph.nodes(data=True)
                    if d.get('status') == 'PENDING'
                ]
                in_flight = list(self.scheduler._in_flight)

                if not pending and not in_flight:
                    logger.info("All tasks complete")
                    return

                await asyncio.sleep(1)

        # Start engine
        engine_task = asyncio.create_task(self.start())

        try:
            if timeout:
                await asyncio.wait_for(_wait_for_completion(), timeout=timeout)
            else:
                await _wait_for_completion()
        except asyncio.TimeoutError:
            logger.warning(f"Timeout after {timeout}s, stopping...")
        finally:
            self.stop()

    async def step(self) -> Dict:
        """
        Process one wavefront (for debugging/step-through mode).

        Returns:
            Dict with nodes processed and results
        """
        ready_nodes = self.scheduler._get_ready_nodes()

        if not ready_nodes:
            return {"status": "idle", "nodes_processed": 0}

        results = []
        for node in ready_nodes[:self.scheduler.config.max_wavefront_width]:
            task = self.scheduler._node_to_task(node)
            if task:
                await self.scheduler._process_task(task)
                results.append({
                    "node_id": node['node_id'],
                    "task_id": task.id,
                    "type": task.task_type.value
                })

        return {
            "status": "processed",
            "nodes_processed": len(results),
            "results": results
        }

    def get_status(self) -> Dict:
        """Get comprehensive engine status (for MCP queries)."""
        return {
            "running": self._running,
            "graph": {
                "nodes": self.db.graph.number_of_nodes(),
                "edges": self.db.graph.number_of_edges()
            },
            "scheduler": self.scheduler.get_status(),
            "feedback": {
                "retry_counts": self.feedback.get_retry_counts()
            },
            "governance": {
                "treasurer": self.treasurer.get_status(),
                "sentinel": self.sentinel.get_status(),
                "curator": self.curator.get_status()
            },
            "cost": {
                "session_spend": self.gateway.get_session_cost()
            }
        }

    def inject_requirement(self, requirement: str, metadata: Optional[Dict] = None) -> str:
        """
        Inject a new requirement into the graph.
        This is the entry point for starting work.

        Args:
            requirement: The requirement text
            metadata: Optional metadata (domain, priority, etc.)

        Returns:
            The created node ID
        """
        import uuid
        from core.ontology import NodeType

        node_id = f"req_{uuid.uuid4().hex[:8]}"
        self.db.add_node(node_id, NodeType.REQ, requirement, metadata=metadata or {})

        logger.info(f"Injected requirement: {node_id}")
        return node_id


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def run_factory(
    requirement: str,
    agents: Dict[AgentRole, Any],
    timeout: float = 300,
    persistence_path: str = ".gaadp/live_graph.pkl"
) -> Dict:
    """
    Convenience function to run the factory on a single requirement.

    Args:
        requirement: The requirement text
        agents: Dict of AgentRole -> agent instance
        timeout: Maximum runtime in seconds
        persistence_path: Path to persist graph

    Returns:
        Engine status after completion
    """
    engine = GADPEngine(persistence_path=persistence_path)
    engine.register_agents(agents)
    engine.inject_requirement(requirement)

    await engine.run_until_complete(timeout=timeout)

    return engine.get_status()
