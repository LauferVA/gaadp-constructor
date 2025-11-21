"""
GOVERNANCE MIDDLEWARE
Enforces constraints at the infrastructure level, not agent level.
- Treasurer: Pre-hook for budget enforcement
- Sentinel: Post-hook for security scanning
- Curator: Background daemon for graph maintenance
"""
import asyncio
import logging
import re
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass

from infrastructure.graph_db import GraphDB
from infrastructure.event_bus import EventBus
from core.ontology import NodeType, NodeStatus, EdgeType

logger = logging.getLogger("Governance")


# =============================================================================
# TREASURER: Budget Enforcement (Pre-Hook)
# =============================================================================

@dataclass
class BudgetConfig:
    """Budget limits from llm_router.yaml."""
    project_total_limit_usd: float = 10.0
    per_call_limit_usd: float = 0.50
    warning_threshold: float = 0.8  # Warn at 80% budget


class TreasurerMiddleware:
    """
    Pre-execution hook that enforces budget limits.
    Rejects tasks if budget would be exceeded.
    """

    def __init__(self, gateway, config: Optional[BudgetConfig] = None):
        """
        Args:
            gateway: LLMGateway instance (has _cost_session)
            config: Budget configuration
        """
        self.gateway = gateway
        self.config = config or BudgetConfig()
        self._warned = False

    def get_current_spend(self) -> float:
        """Get current session spend."""
        return getattr(self.gateway, '_cost_session', 0.0)

    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        return self.config.project_total_limit_usd - self.get_current_spend()

    async def pre_hook(self, task: Any) -> bool:
        """
        Pre-execution hook. Returns False to reject task.
        Called by TaskScheduler before dispatching to agent.
        """
        current_spend = self.get_current_spend()
        budget = self.config.project_total_limit_usd

        # Hard limit
        if current_spend >= budget:
            logger.error(f"TREASURER HALT: Budget exhausted (${current_spend:.2f} >= ${budget:.2f})")
            await self._publish_halt_event(task, current_spend, budget)
            return False

        # Warning threshold
        if not self._warned and current_spend >= budget * self.config.warning_threshold:
            logger.warning(f"TREASURER WARNING: {self.config.warning_threshold*100}% budget consumed")
            self._warned = True

        # Estimate cost (rough heuristic)
        estimated_cost = self.config.per_call_limit_usd
        if current_spend + estimated_cost > budget:
            logger.warning(f"TREASURER: Task may exceed budget, proceeding with caution")

        return True

    async def _publish_halt_event(self, task: Any, spend: float, budget: float):
        """Publish budget halt event (if event bus available)."""
        # Event publishing would be wired by Engine
        pass

    def get_status(self) -> Dict:
        """Get Treasurer status for MCP queries."""
        return {
            "current_spend": self.get_current_spend(),
            "budget_limit": self.config.project_total_limit_usd,
            "remaining": self.get_remaining_budget(),
            "warning_issued": self._warned,
            "status": "SOLVENT" if self.get_remaining_budget() > 0 else "EXHAUSTED"
        }


# =============================================================================
# SENTINEL: Security Scanning (Post-Hook)
# =============================================================================

@dataclass
class SecurityConfig:
    """Security scanning configuration."""
    block_on_critical: bool = True
    block_on_high: bool = True
    block_on_medium: bool = False


class SentinelMiddleware:
    """
    Post-execution hook that scans generated code for security issues.
    Rejects nodes containing dangerous patterns.
    """

    # Pattern categories with severity
    PATTERNS = {
        'critical': [
            (r'eval\s*\(', 'eval() allows arbitrary code execution'),
            (r'exec\s*\(', 'exec() allows arbitrary code execution'),
            (r'__import__\s*\(', '__import__() can load arbitrary modules'),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', 'shell=True enables command injection'),
            (r'os\.system\s*\(', 'os.system() enables command injection'),
        ],
        'high': [
            (r'pickle\.loads?\s*\(', 'pickle can execute arbitrary code during deserialization'),
            (r'yaml\.load\s*\([^)]*Loader\s*=\s*yaml\.Loader', 'yaml.Loader is unsafe'),
            (r'yaml\.load\s*\([^)]*\)(?!.*Loader)', 'yaml.load without SafeLoader is unsafe'),
            (r'marshal\.loads?\s*\(', 'marshal can execute code during deserialization'),
            (r'input\s*\(', 'input() in Python 2 evaluates code'),
        ],
        'medium': [
            (r'subprocess\.(run|Popen|call)\s*\(', 'subprocess calls should be audited'),
            (r'open\s*\([^)]*["\']w["\']', 'file write operations should be audited'),
            (r'requests\.(get|post|put|delete)\s*\(', 'HTTP requests should be audited'),
            (r'sqlite3\.connect\s*\(', 'database operations should be audited'),
        ],
        'info': [
            (r'# TODO', 'TODO comment found'),
            (r'# FIXME', 'FIXME comment found'),
            (r'pass\s*$', 'empty pass statement'),
        ]
    }

    def __init__(self, db: GraphDB, config: Optional[SecurityConfig] = None):
        self.db = db
        self.config = config or SecurityConfig()
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List]:
        """Pre-compile regex patterns for performance."""
        compiled = {}
        for severity, patterns in self.PATTERNS.items():
            compiled[severity] = [
                (re.compile(pattern, re.MULTILINE | re.IGNORECASE), desc)
                for pattern, desc in patterns
            ]
        return compiled

    def scan_code(self, code: str) -> Dict:
        """
        Scan code for security issues.

        Returns:
            {
                'safe': bool,
                'issues': [{'severity': str, 'pattern': str, 'description': str, 'line': int}],
                'summary': {'critical': int, 'high': int, 'medium': int, 'info': int}
            }
        """
        issues = []
        summary = {'critical': 0, 'high': 0, 'medium': 0, 'info': 0}

        lines = code.split('\n')

        for severity, patterns in self._compiled_patterns.items():
            for pattern, description in patterns:
                for i, line in enumerate(lines, 1):
                    if pattern.search(line):
                        issues.append({
                            'severity': severity,
                            'pattern': pattern.pattern,
                            'description': description,
                            'line': i,
                            'content': line.strip()[:100]
                        })
                        summary[severity] += 1

        # Determine if safe based on config
        safe = True
        if self.config.block_on_critical and summary['critical'] > 0:
            safe = False
        if self.config.block_on_high and summary['high'] > 0:
            safe = False
        if self.config.block_on_medium and summary['medium'] > 0:
            safe = False

        return {
            'safe': safe,
            'issues': issues,
            'summary': summary
        }

    async def post_hook(self, task: Any, result: Dict) -> bool:
        """
        Post-execution hook. Returns False to reject result.
        Called by TaskScheduler after agent produces output.
        """
        # Only scan CODE outputs
        content = result.get('content', '')
        node_type = result.get('type', '')

        if node_type != NodeType.CODE.value and 'code' not in str(task.task_type).lower():
            return True  # Not code, skip scan

        if not content:
            return True

        scan_result = self.scan_code(content)

        if not scan_result['safe']:
            logger.error(f"SENTINEL BLOCK: Security issues detected in task {task.id}")
            for issue in scan_result['issues']:
                if issue['severity'] in ['critical', 'high']:
                    logger.error(f"  [{issue['severity'].upper()}] Line {issue['line']}: {issue['description']}")

            # Store scan results on the task for debugging
            result['security_scan'] = scan_result
            return False

        if scan_result['summary']['medium'] > 0 or scan_result['summary']['info'] > 0:
            logger.info(f"SENTINEL: {sum(scan_result['summary'].values())} issues found (non-blocking)")

        return True

    def get_status(self) -> Dict:
        """Get Sentinel status for MCP queries."""
        return {
            "config": {
                "block_on_critical": self.config.block_on_critical,
                "block_on_high": self.config.block_on_high,
                "block_on_medium": self.config.block_on_medium,
            },
            "patterns_loaded": sum(len(p) for p in self._compiled_patterns.values())
        }


# =============================================================================
# CURATOR: Graph Maintenance Daemon
# =============================================================================

@dataclass
class CuratorConfig:
    """Curator daemon configuration."""
    interval_seconds: int = 60
    prune_dead_ends: bool = True
    check_cycles: bool = True
    check_orphans: bool = True


class CuratorDaemon:
    """
    Background daemon that maintains graph integrity.
    - Prunes DEAD_END nodes
    - Detects cycles (shouldn't exist, but check anyway)
    - Finds orphan nodes
    - Validates traceability
    """

    def __init__(
        self,
        db: GraphDB,
        event_bus: EventBus,
        config: Optional[CuratorConfig] = None
    ):
        self.db = db
        self.event_bus = event_bus
        self.config = config or CuratorConfig()
        self._running = False
        self._last_run_result: Optional[Dict] = None

    def _run_maintenance(self) -> Dict:
        """Run a single maintenance pass."""
        import networkx as nx
        result = {
            'dead_ends_pruned': 0,
            'orphans_found': 0,
            'cycles_detected': 0,
            'traceability_issues': 0,
            'node_count': self.db.graph.number_of_nodes(),
            'edge_count': self.db.graph.number_of_edges()
        }

        # 1. Prune dead ends
        if self.config.prune_dead_ends:
            dead_ends = [
                n for n, d in self.db.graph.nodes(data=True)
                if d.get('type') == NodeType.DEAD_END.value
            ]
            if dead_ends:
                self.db.graph.remove_nodes_from(dead_ends)
                result['dead_ends_pruned'] = len(dead_ends)

        # 2. Check for cycles
        if self.config.check_cycles:
            try:
                cycles = list(nx.simple_cycles(self.db.graph))
                result['cycles_detected'] = len(cycles)
                if cycles:
                    logger.error(f"CURATOR: {len(cycles)} cycles detected!")
            except:
                pass

        # 3. Find orphans (no incoming edges, not REQ type)
        if self.config.check_orphans:
            for node in self.db.graph.nodes():
                if self.db.graph.in_degree(node) == 0:
                    node_type = self.db.graph.nodes[node].get('type')
                    if node_type != NodeType.REQ.value:
                        result['orphans_found'] += 1

        # 4. Traceability check (non-REQ nodes should trace to something)
        for node in self.db.graph.nodes():
            node_type = self.db.graph.nodes[node].get('type')
            if node_type in [NodeType.REQ.value, NodeType.STATE.value, NodeType.DEAD_END.value]:
                continue

            has_trace = any(
                d.get('type') == EdgeType.TRACES_TO.value
                for _, _, d in self.db.graph.out_edges(node, data=True)
            )
            has_implements = any(
                d.get('type') == EdgeType.IMPLEMENTS.value
                for _, _, d in self.db.graph.out_edges(node, data=True)
            )
            if not has_trace and not has_implements:
                result['traceability_issues'] += 1

        # Persist if changes made
        if result['dead_ends_pruned'] > 0:
            self.db._persist()

        return result

    async def _daemon_loop(self):
        """Main daemon loop."""
        while self._running:
            try:
                result = self._run_maintenance()
                self._last_run_result = result

                if result['dead_ends_pruned'] > 0:
                    logger.info(f"CURATOR: Pruned {result['dead_ends_pruned']} dead-end nodes")
                if result['cycles_detected'] > 0:
                    logger.error(f"CURATOR: {result['cycles_detected']} cycles detected!")
                if result['orphans_found'] > 0:
                    logger.warning(f"CURATOR: {result['orphans_found']} orphan nodes found")

                await self.event_bus.publish(
                    topic="curator",
                    message_type="MAINTENANCE_COMPLETE",
                    payload=result,
                    source_id="curator_daemon"
                )

            except Exception as e:
                logger.error(f"CURATOR error: {e}")

            await asyncio.sleep(self.config.interval_seconds)

    async def start(self):
        """Start the curator daemon."""
        logger.info(f"Starting curator daemon (interval: {self.config.interval_seconds}s)")
        self._running = True
        await self._daemon_loop()

    def stop(self):
        """Stop the curator daemon."""
        logger.info("Stopping curator daemon")
        self._running = False

    def run_once(self) -> Dict:
        """Run maintenance once (for MCP)."""
        return self._run_maintenance()

    def get_status(self) -> Dict:
        """Get curator status for MCP queries."""
        return {
            "running": self._running,
            "interval_seconds": self.config.interval_seconds,
            "last_run": self._last_run_result
        }


# =============================================================================
# FACTORY FUNCTION: Create all middleware
# =============================================================================

def create_governance_middleware(
    db: GraphDB,
    gateway,
    event_bus: EventBus
) -> Dict:
    """
    Factory function to create all governance middleware.

    Returns:
        {
            'treasurer': TreasurerMiddleware,
            'sentinel': SentinelMiddleware,
            'curator': CuratorDaemon,
            'pre_hooks': [callable],
            'post_hooks': [callable]
        }
    """
    treasurer = TreasurerMiddleware(gateway)
    sentinel = SentinelMiddleware(db)
    curator = CuratorDaemon(db, event_bus)

    return {
        'treasurer': treasurer,
        'sentinel': sentinel,
        'curator': curator,
        'pre_hooks': [treasurer.pre_hook],
        'post_hooks': [sentinel.post_hook]
    }
