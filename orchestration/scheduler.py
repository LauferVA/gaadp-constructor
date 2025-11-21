"""
TASK SCHEDULER
The Factory Floor - manages parallel agent dispatch with governance hooks.
"""
import asyncio
import logging
import uuid
import yaml
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from infrastructure.graph_db import GraphDB
from infrastructure.event_bus import EventBus, MessageType
from core.ontology import NodeType, NodeStatus, EdgeType, AgentRole

logger = logging.getLogger("Scheduler")


class TaskType(str, Enum):
    """Maps node types to processing tasks."""
    DECOMPOSE = "DECOMPOSE"  # REQ → SPEC/PLAN (Architect)
    IMPLEMENT = "IMPLEMENT"  # SPEC → CODE (Builder)
    VERIFY = "VERIFY"        # CODE → VERIFIED/FAILED (Verifier)
    TEST = "TEST"            # CODE → TEST results (TestRunner)


@dataclass
class Task:
    """A unit of work for the scheduler."""
    id: str
    node_id: str
    task_type: TaskType
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    context: Dict = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    """Configuration loaded from topology."""
    max_concurrent_builders: int = 10
    max_concurrent_verifiers: int = 10
    max_wavefront_width: int = 20
    plan_timeout_seconds: int = 300
    verification_timeout_seconds: int = 120
    curator_interval_seconds: int = 60

    @classmethod
    def from_topology(cls, topology_path: str = ".blueprint/topology_config.yaml") -> "SchedulerConfig":
        try:
            with open(topology_path, "r") as f:
                config = yaml.safe_load(f)

            parallel = config.get("parallel_limits", {})
            timeouts = config.get("timeouts", {})

            return cls(
                max_concurrent_builders=parallel.get("max_concurrent_builders", 10),
                max_concurrent_verifiers=parallel.get("max_concurrent_verifiers", 10),
                max_wavefront_width=parallel.get("max_wavefront_width", 20),
                plan_timeout_seconds=timeouts.get("plan_execution_seconds", 300),
                verification_timeout_seconds=timeouts.get("verification_seconds", 120),
            )
        except FileNotFoundError:
            logger.warning("Topology config not found, using defaults")
            return cls()


class TaskScheduler:
    """
    The Factory Floor Scheduler.

    Responsibilities:
    - Discover ready tasks (PENDING nodes with satisfied dependencies)
    - Dispatch to worker pool with concurrency limits
    - Handle completion/failure with state transitions
    - Run governance daemons (Curator) in background
    """

    def __init__(
        self,
        db: GraphDB,
        event_bus: EventBus,
        config: Optional[SchedulerConfig] = None,
    ):
        self.db = db
        self.event_bus = event_bus
        self.config = config or SchedulerConfig.from_topology()

        # Concurrency controls
        self._builder_semaphore = asyncio.Semaphore(self.config.max_concurrent_builders)
        self._verifier_semaphore = asyncio.Semaphore(self.config.max_concurrent_verifiers)
        self._wavefront_semaphore = asyncio.Semaphore(self.config.max_wavefront_width)

        # Agent registry (set by Engine)
        self._agents: Dict[AgentRole, Any] = {}

        # Governance hooks
        self._pre_hooks: List[Callable] = []  # Called before LLM call
        self._post_hooks: List[Callable] = []  # Called after node creation

        # Runtime state
        self._running = False
        self._in_flight: Set[str] = set()  # Node IDs currently being processed
        self._task_queue: asyncio.Queue = asyncio.Queue()

    def register_agent(self, role: AgentRole, agent: Any):
        """Register an agent instance for a role."""
        self._agents[role] = agent
        logger.info(f"Registered agent: {role.value}")

    def register_pre_hook(self, hook: Callable):
        """Register a pre-execution hook (e.g., Treasurer budget check)."""
        self._pre_hooks.append(hook)

    def register_post_hook(self, hook: Callable):
        """Register a post-execution hook (e.g., Sentinel security scan)."""
        self._post_hooks.append(hook)

    def _get_ready_nodes(self) -> List[Dict]:
        """
        Find nodes ready for processing:
        - Status = PENDING
        - All DEPENDS_ON predecessors are COMPLETE or VERIFIED
        - Not currently in flight
        """
        ready = []

        for node_id, data in self.db.graph.nodes(data=True):
            if data.get('status') != NodeStatus.PENDING.value:
                continue
            if node_id in self._in_flight:
                continue

            # Check dependencies
            deps_satisfied = True
            for pred in self.db.graph.predecessors(node_id):
                edge_data = self.db.graph.edges[pred, node_id]
                if edge_data.get('type') != EdgeType.DEPENDS_ON.value:
                    continue

                pred_status = self.db.graph.nodes[pred].get('status')
                if pred_status not in [NodeStatus.COMPLETE.value, NodeStatus.VERIFIED.value]:
                    deps_satisfied = False
                    break

            if deps_satisfied:
                ready.append({
                    'node_id': node_id,
                    'type': data.get('type'),
                    'content': data.get('content'),
                    'metadata': data.get('metadata', {})
                })

        return ready

    def _node_to_task(self, node: Dict) -> Optional[Task]:
        """Convert a ready node to a Task based on its type."""
        node_type = node['type']
        node_id = node['node_id']

        # Route based on node type
        if node_type == NodeType.REQ.value:
            return Task(
                id=f"task_{uuid.uuid4().hex[:8]}",
                node_id=node_id,
                task_type=TaskType.DECOMPOSE,
                context={'content': node['content']}
            )
        elif node_type == NodeType.SPEC.value:
            return Task(
                id=f"task_{uuid.uuid4().hex[:8]}",
                node_id=node_id,
                task_type=TaskType.IMPLEMENT,
                context={'content': node['content'], 'metadata': node['metadata']}
            )
        elif node_type == NodeType.CODE.value:
            return Task(
                id=f"task_{uuid.uuid4().hex[:8]}",
                node_id=node_id,
                task_type=TaskType.VERIFY,
                context={'content': node['content'], 'metadata': node['metadata']}
            )

        return None

    async def _run_pre_hooks(self, task: Task) -> bool:
        """
        Run pre-execution hooks. Returns False if any hook rejects.
        Used for: Treasurer budget check, rate limiting, etc.
        """
        for hook in self._pre_hooks:
            try:
                result = await hook(task) if asyncio.iscoroutinefunction(hook) else hook(task)
                if result is False:
                    logger.warning(f"Pre-hook rejected task {task.id}")
                    return False
            except Exception as e:
                logger.error(f"Pre-hook failed: {e}")
                return False
        return True

    async def _run_post_hooks(self, task: Task, result: Dict) -> bool:
        """
        Run post-execution hooks. Returns False if any hook rejects.
        Used for: Sentinel security scan, validation, etc.
        """
        for hook in self._post_hooks:
            try:
                hook_result = await hook(task, result) if asyncio.iscoroutinefunction(hook) else hook(task, result)
                if hook_result is False:
                    logger.warning(f"Post-hook rejected result for task {task.id}")
                    return False
            except Exception as e:
                logger.error(f"Post-hook failed: {e}")
                return False
        return True

    async def _execute_task(self, task: Task) -> Dict:
        """Execute a single task by dispatching to the appropriate agent."""
        # Select agent based on task type
        if task.task_type == TaskType.DECOMPOSE:
            agent = self._agents.get(AgentRole.ARCHITECT)
            semaphore = self._wavefront_semaphore
        elif task.task_type == TaskType.IMPLEMENT:
            agent = self._agents.get(AgentRole.BUILDER)
            semaphore = self._builder_semaphore
        elif task.task_type == TaskType.VERIFY:
            agent = self._agents.get(AgentRole.VERIFIER)
            semaphore = self._verifier_semaphore
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")

        if not agent:
            raise ValueError(f"No agent registered for task type: {task.task_type}")

        # Acquire semaphore for concurrency control
        async with semaphore:
            async with self._wavefront_semaphore:
                # Mark node as IN_PROGRESS (validated by state machine)
                self.db.set_status(task.node_id, NodeStatus.IN_PROGRESS, reason=f"Task {task.id} started")

                # Publish task start event
                await self.event_bus.publish(
                    topic="scheduler",
                    message_type=MessageType.TASK_ASSIGN,
                    payload={"task_id": task.id, "node_id": task.node_id, "type": task.task_type.value},
                    source_id="scheduler"
                )

                # Execute with timeout
                timeout = (
                    self.config.verification_timeout_seconds
                    if task.task_type == TaskType.VERIFY
                    else self.config.plan_timeout_seconds
                )

                try:
                    result = await asyncio.wait_for(
                        agent.process({"nodes": [{"id": task.node_id, "content": task.context.get('content', '')}]}),
                        timeout=timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    logger.error(f"Task {task.id} timed out after {timeout}s")
                    return {"verdict": "TIMEOUT", "error": f"Exceeded {timeout}s limit"}

    async def _process_task(self, task: Task):
        """Full task lifecycle: hooks → execute → state update."""
        try:
            self._in_flight.add(task.node_id)

            # Pre-hooks (e.g., Treasurer check)
            if not await self._run_pre_hooks(task):
                self.db.set_status(task.node_id, NodeStatus.BLOCKED, reason="Pre-hook rejection")
                await self.event_bus.publish(
                    topic="scheduler",
                    message_type=MessageType.ERROR,
                    payload={"task_id": task.id, "reason": "Pre-hook rejection"},
                    source_id="scheduler"
                )
                return

            # Execute
            result = await self._execute_task(task)

            # Post-hooks (e.g., Sentinel scan)
            if not await self._run_post_hooks(task, result):
                self.db.set_status(task.node_id, NodeStatus.FAILED, reason="Security rejection")
                return

            # Handle result based on task type
            await self._handle_result(task, result)

            # Publish completion
            await self.event_bus.publish(
                topic="scheduler",
                message_type=MessageType.RESULT_SUBMIT,
                payload={"task_id": task.id, "result": result},
                source_id="scheduler"
            )

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            self.db.set_status(task.node_id, NodeStatus.FAILED, reason=str(e))

            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.info(f"Retrying task {task.id} (attempt {task.retry_count})")
                await self._task_queue.put(task)
        finally:
            self._in_flight.discard(task.node_id)

    async def _handle_result(self, task: Task, result: Dict):
        """Process task result and update graph state."""
        node_id = task.node_id

        if task.task_type == TaskType.DECOMPOSE:
            # Architect produced new nodes (SPEC, PLAN)
            self.db.set_status(node_id, NodeStatus.COMPLETE, reason="Decomposition complete")

            # Create child nodes from result
            for new_node in result.get('new_nodes', []):
                child_id = f"{new_node['type'].lower()}_{uuid.uuid4().hex[:8]}"
                self.db.add_node(
                    child_id,
                    NodeType(new_node['type']),
                    new_node['content'],
                    metadata=new_node.get('metadata', {})
                )
                # Link child to parent
                self.db.add_edge(
                    child_id, node_id,
                    EdgeType.TRACES_TO,
                    signed_by="scheduler",
                    signature=f"auto_{uuid.uuid4().hex[:8]}"
                )

        elif task.task_type == TaskType.IMPLEMENT:
            # Builder produced CODE node
            self.db.set_status(node_id, NodeStatus.COMPLETE, reason="Implementation complete")

            code_id = f"code_{uuid.uuid4().hex[:8]}"
            self.db.add_node(
                code_id,
                NodeType.CODE,
                result.get('content', ''),
                metadata=result.get('metadata', {})
            )
            # CODE implements SPEC
            self.db.add_edge(
                code_id, node_id,
                EdgeType.IMPLEMENTS,
                signed_by="scheduler",
                signature=f"auto_{uuid.uuid4().hex[:8]}"
            )

        elif task.task_type == TaskType.VERIFY:
            # Verifier judged CODE node
            verdict = result.get('verdict', 'FAIL')

            if verdict == 'PASS':
                self.db.set_status(node_id, NodeStatus.VERIFIED, reason="Verification passed")
            elif verdict == 'CONDITIONAL':
                # Needs minor fixes but acceptable
                self.db.set_status(node_id, NodeStatus.VERIFIED, reason="Conditional pass")
                self.db.graph.nodes[node_id]['conditions'] = result.get('conditions', [])
                self.db._persist()
            else:
                # FAIL - triggers feedback loop (handled by FeedbackController)
                critique = result.get('critique', '')
                self.db.set_status(node_id, NodeStatus.FAILED, reason=f"Verification failed: {critique[:100]}")
                self.db.graph.nodes[node_id]['critique'] = critique
                self.db._persist()

    async def _discovery_loop(self):
        """Continuously discover ready tasks and queue them."""
        while self._running:
            ready_nodes = self._get_ready_nodes()

            for node in ready_nodes:
                task = self._node_to_task(node)
                if task:
                    await self._task_queue.put(task)
                    logger.debug(f"Queued task for node {node['node_id']}")

            await asyncio.sleep(1)  # Poll interval

    async def _worker_loop(self, worker_id: int):
        """Worker that pulls tasks from queue and processes them."""
        while self._running:
            try:
                task = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)
                logger.info(f"Worker {worker_id} processing task {task.id}")
                await self._process_task(task)
                self._task_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    async def start(self, num_workers: int = 5):
        """Start the scheduler with discovery loop and worker pool."""
        logger.info(f"Starting scheduler with {num_workers} workers")
        self._running = True

        # Start discovery loop
        discovery_task = asyncio.create_task(self._discovery_loop())

        # Start worker pool
        workers = [
            asyncio.create_task(self._worker_loop(i))
            for i in range(num_workers)
        ]

        await asyncio.gather(discovery_task, *workers)

    def stop(self):
        """Stop the scheduler gracefully."""
        logger.info("Stopping scheduler")
        self._running = False

    async def submit_task(self, task: Task):
        """Manually submit a task (for MCP integration)."""
        await self._task_queue.put(task)

    def get_status(self) -> Dict:
        """Get scheduler status (for MCP queries)."""
        return {
            "running": self._running,
            "in_flight": list(self._in_flight),
            "queue_size": self._task_queue.qsize(),
            "registered_agents": [r.value for r in self._agents.keys()]
        }
