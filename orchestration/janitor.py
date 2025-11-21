"""
JANITOR DAEMON
Cleans up orphaned tasks and crashed agent state.

Responsibilities:
- Detect nodes stuck IN_PROGRESS beyond timeout
- Verify if agent threads/tasks still alive
- Mark orphaned nodes as FAILED
- Emit cleanup events
- Prevent infinite blocking from crashed agents
"""
import asyncio
import logging
import time
from typing import Dict, Set, Optional
from datetime import datetime, timedelta

from infrastructure.graph_db import GraphDB
from infrastructure.event_bus import EventBus
from core.ontology import NodeStatus, NodeType

logger = logging.getLogger("Janitor")


class JanitorConfig:
    """Configuration for Janitor daemon."""
    def __init__(
        self,
        scan_interval: float = 60.0,  # Scan every 60 seconds
        orphan_timeout: float = 300.0,  # 5 minutes
        enabled: bool = True
    ):
        self.scan_interval = scan_interval
        self.orphan_timeout = orphan_timeout
        self.enabled = enabled


class JanitorDaemon:
    """
    Background daemon that cleans up orphaned tasks.

    An orphaned task is:
    - Status: IN_PROGRESS or PENDING_VERIFICATION
    - No activity for > orphan_timeout
    - Associated agent thread no longer running
    """

    def __init__(
        self,
        db: GraphDB,
        event_bus: EventBus,
        config: Optional[JanitorConfig] = None
    ):
        self.db = db
        self.event_bus = event_bus
        self.config = config or JanitorConfig()

        # Track active agents (agent_id -> last_heartbeat_time)
        self.agent_heartbeats: Dict[str, float] = {}

        # Track nodes being processed (node_id -> (agent_id, start_time))
        self.active_processing: Dict[str, tuple] = {}

        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the janitor daemon."""
        if not self.config.enabled:
            logger.info("Janitor daemon disabled by config")
            return

        if self._running:
            logger.warning("Janitor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Janitor daemon started (scan_interval={self.config.scan_interval}s, timeout={self.config.orphan_timeout}s)")

    async def stop(self):
        """Stop the janitor daemon."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Janitor daemon stopped")

    def record_agent_heartbeat(self, agent_id: str):
        """Record that an agent is alive."""
        self.agent_heartbeats[agent_id] = time.time()

    def record_node_processing(self, node_id: str, agent_id: str):
        """Record that a node is being processed by an agent."""
        self.active_processing[node_id] = (agent_id, time.time())

    def clear_node_processing(self, node_id: str):
        """Clear processing record for a node (on completion)."""
        self.active_processing.pop(node_id, None)

    async def _run_loop(self):
        """Main daemon loop."""
        while self._running:
            try:
                await self._scan_for_orphans()
                await asyncio.sleep(self.config.scan_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Janitor scan error: {e}", exc_info=True)
                await asyncio.sleep(self.config.scan_interval)

    async def _scan_for_orphans(self):
        """Scan for and clean up orphaned tasks."""
        current_time = time.time()
        orphans_found = 0

        logger.debug("Janitor: Scanning for orphans...")

        # Find nodes stuck in processing states
        stuck_nodes = []
        for node_id, data in self.db.graph.nodes(data=True):
            status = data.get('status')

            # Check if node is in a processing state
            if status not in [NodeStatus.IN_PROGRESS.value, NodeStatus.PENDING_VERIFICATION.value]:
                continue

            # Check last update time
            status_updated_at = data.get('status_updated_at')
            if not status_updated_at:
                # No timestamp, use created_at
                status_updated_at = data.get('created_at')

            if not status_updated_at:
                # Can't determine age, skip
                continue

            # Parse timestamp
            try:
                updated_time = datetime.fromisoformat(status_updated_at)
                age_seconds = (datetime.utcnow() - updated_time).total_seconds()
            except (ValueError, TypeError):
                logger.warning(f"Invalid timestamp for node {node_id}: {status_updated_at}")
                continue

            # Check if orphaned
            if age_seconds > self.config.orphan_timeout:
                stuck_nodes.append((node_id, status, age_seconds))

        # Process each stuck node
        for node_id, status, age in stuck_nodes:
            # Check if we have processing records
            if node_id in self.active_processing:
                agent_id, start_time = self.active_processing[node_id]

                # Check if agent is still alive
                agent_last_seen = self.agent_heartbeats.get(agent_id, 0)
                if current_time - agent_last_seen < self.config.orphan_timeout:
                    # Agent is still alive, not orphaned
                    continue

                logger.warning(
                    f"Orphaned node detected: {node_id} (agent {agent_id} dead for {current_time - agent_last_seen:.0f}s)"
                )
            else:
                logger.warning(
                    f"Orphaned node detected: {node_id} (status={status}, age={age:.0f}s, no processing record)"
                )

            # Clean up orphan
            await self._clean_orphan(node_id, status, age)
            orphans_found += 1

        if orphans_found > 0:
            logger.info(f"Janitor: Cleaned up {orphans_found} orphaned nodes")
        else:
            logger.debug("Janitor: No orphans found")

    async def _clean_orphan(self, node_id: str, old_status: str, age_seconds: float):
        """Clean up an orphaned node."""
        try:
            # Mark as FAILED with orphan reason
            reason = f"Orphaned (stuck in {old_status} for {age_seconds:.0f}s, agent crashed/timeout)"
            self.db.set_status(
                node_id,
                NodeStatus.FAILED,
                reason=reason
            )

            # Clear processing record
            self.clear_node_processing(node_id)

            # Emit cleanup event
            await self.event_bus.publish(
                topic="system",
                message_type="ORPHAN_CLEANED",
                payload={
                    "node_id": node_id,
                    "old_status": old_status,
                    "age_seconds": age_seconds,
                    "reason": reason
                },
                source_id="janitor"
            )

            logger.info(f"Cleaned orphaned node: {node_id}")

        except Exception as e:
            logger.error(f"Failed to clean orphan {node_id}: {e}", exc_info=True)

    def get_stats(self) -> Dict:
        """Get janitor statistics."""
        current_time = time.time()

        # Count active agents
        active_agents = sum(
            1 for last_seen in self.agent_heartbeats.values()
            if current_time - last_seen < self.config.orphan_timeout
        )

        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "scan_interval": self.config.scan_interval,
            "orphan_timeout": self.config.orphan_timeout,
            "active_agents": active_agents,
            "tracked_agents": len(self.agent_heartbeats),
            "nodes_processing": len(self.active_processing)
        }
