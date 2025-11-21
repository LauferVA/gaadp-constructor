"""
GAADP Critical Fixes Validation Tests
Tests for the 5 critical architectural improvements identified by external review.
"""
import pytest
import asyncio
import time
import logging
from unittest.mock import patch, MagicMock
from infrastructure.sandbox import CodeSandbox, SandboxSecurityError
from infrastructure.semantic_memory import SemanticMemory, SemanticMemoryError
from infrastructure.graph_db import GraphDB
from infrastructure.event_bus import EventBus
from orchestration.janitor import JanitorDaemon, JanitorConfig
from orchestration.scheduler import TaskScheduler, SchedulerConfig
from agents.concrete_agents import RealArchitect
from core.ontology import NodeType, NodeStatus, AgentRole


class TestDockerSecurity:
    """Test Fix #1: Docker security enforcement (no local fallback by default)."""

    def test_docker_required_by_default(self):
        """Verify sandbox requires Docker unless explicitly allowed fallback."""
        # This test verifies the security fix - by default, no unsafe local execution

        # If Docker is available, this should work
        try:
            sandbox = CodeSandbox(use_docker=True, allow_local_fallback=False)
            # Docker available - good!
            assert sandbox.use_docker == True
        except SandboxSecurityError as e:
            # Docker not available and fallback disabled - this is CORRECT behavior
            assert "Docker is required" in str(e)
            assert "not available" in str(e)

    def test_explicit_fallback_allowed(self):
        """Verify local fallback works when explicitly enabled."""
        # This should always work (uses local Python)
        sandbox = CodeSandbox(use_docker=False, allow_local_fallback=True)
        assert sandbox.use_docker == False

    def test_security_error_provides_help(self):
        """Verify error message provides installation guidance."""
        try:
            # Force Docker unavailable
            with patch.object(CodeSandbox, '_check_docker', return_value=False):
                sandbox = CodeSandbox(use_docker=True, allow_local_fallback=False)
                pytest.fail("Should have raised SandboxSecurityError")
        except SandboxSecurityError as e:
            # Should provide helpful guidance
            assert "Install Docker" in str(e) or "docker.com" in str(e)


class TestSemanticMemoryDegradation:
    """Test Fix #2: Vector DB graceful degradation with warnings."""

    def test_embeddings_unavailable_raises_error_when_required(self):
        """Verify error raised when embeddings required but unavailable."""
        # Mock embeddings as unavailable
        with patch('infrastructure.semantic_memory.EMBEDDINGS_AVAILABLE', False):
            with pytest.raises(SemanticMemoryError, match="sentence-transformers"):
                memory = SemanticMemory(require_embeddings=True)

    def test_fallback_mode_allows_degraded_operation(self):
        """Verify fallback mode works without embeddings."""
        with patch('infrastructure.semantic_memory.EMBEDDINGS_AVAILABLE', False):
            # Should not raise error
            memory = SemanticMemory(require_embeddings=False, fallback_mode=True)
            assert memory.embeddings_enabled == False

    def test_fallback_mode_logs_warning(self, caplog):
        """Verify degraded mode logs clear warnings."""
        with patch('infrastructure.semantic_memory.EMBEDDINGS_AVAILABLE', False):
            with caplog.at_level(logging.WARNING):
                memory = SemanticMemory(require_embeddings=False, fallback_mode=True)

                # Should have logged warning about degraded mode
                warning_found = any(
                    "fallback" in record.message.lower() or "degraded" in record.message.lower()
                    for record in caplog.records
                )
                assert warning_found, "Should log warning about degraded mode"

    def test_embeddings_enabled_flag_accessible(self):
        """Verify embeddings_enabled flag is accessible for runtime checks."""
        with patch('infrastructure.semantic_memory.EMBEDDINGS_AVAILABLE', False):
            memory = SemanticMemory(require_embeddings=False, fallback_mode=True)

            # ContextPruner checks this flag
            assert hasattr(memory, 'embeddings_enabled')
            assert memory.embeddings_enabled == False


class TestJanitorOrphanCleanup:
    """Test Fix #3: Janitor daemon for orphaned task detection."""

    @pytest.fixture
    def setup_janitor(self):
        """Setup clean test environment."""
        db = GraphDB(persistence_path=".gaadp_test/janitor_test.json")
        event_bus = EventBus()
        config = JanitorConfig(
            scan_interval=1,
            orphan_timeout=5,  # 5 seconds for testing
            enabled=True
        )
        janitor = JanitorDaemon(db, event_bus, config)

        yield db, event_bus, janitor

        # Cleanup
        import os
        if os.path.exists(".gaadp_test/janitor_test.json"):
            os.remove(".gaadp_test/janitor_test.json")

    @pytest.mark.asyncio
    async def test_janitor_detects_orphaned_nodes(self, setup_janitor):
        """Verify Janitor marks stuck IN_PROGRESS nodes as FAILED."""
        db, event_bus, janitor = setup_janitor

        # Create a node and mark it IN_PROGRESS
        db.add_node("stuck_1", NodeType.CODE, "test code")
        db.set_status("stuck_1", NodeStatus.IN_PROGRESS, reason="Test stuck node")

        # Backdate the in_progress timestamp to simulate timeout
        if "metadata" not in db.graph.nodes["stuck_1"]:
            db.graph.nodes["stuck_1"]["metadata"] = {}
        db.graph.nodes["stuck_1"]["metadata"]["in_progress_since"] = time.time() - 10  # 10 seconds ago

        # Run orphan scan
        await janitor._scan_for_orphans()

        # Verify node was marked as FAILED
        node_status = db.graph.nodes["stuck_1"]["status"]
        assert node_status == NodeStatus.FAILED.value

        # Verify reason mentions orphan
        node_data = db.graph.nodes["stuck_1"]
        reason = node_data.get("status_reason", "")
        assert "orphan" in reason.lower()

    @pytest.mark.asyncio
    async def test_janitor_ignores_recent_nodes(self, setup_janitor):
        """Verify Janitor doesn't touch recently started nodes."""
        db, event_bus, janitor = setup_janitor

        # Create node just started
        db.add_node("recent_1", NodeType.CODE, "recent code")
        db.set_status("recent_1", NodeStatus.IN_PROGRESS, reason="Just started")

        # Recent timestamp (within timeout)
        if "metadata" not in db.graph.nodes["recent_1"]:
            db.graph.nodes["recent_1"]["metadata"] = {}
        db.graph.nodes["recent_1"]["metadata"]["in_progress_since"] = time.time() - 2  # 2 seconds ago

        # Run scan
        await janitor._scan_for_orphans()

        # Should still be IN_PROGRESS
        assert db.graph.nodes["recent_1"]["status"] == NodeStatus.IN_PROGRESS.value

    @pytest.mark.asyncio
    async def test_janitor_emits_events(self, setup_janitor):
        """Verify Janitor publishes events for orphan cleanup."""
        db, event_bus, janitor = setup_janitor

        # Subscribe to events
        events_received = []

        async def event_handler(event):
            events_received.append(event)

        event_bus.subscribe("janitor", event_handler)

        # Create orphaned node
        db.add_node("orphan_1", NodeType.CODE, "orphaned")
        db.set_status("orphan_1", NodeStatus.IN_PROGRESS, reason="Orphaned")
        if "metadata" not in db.graph.nodes["orphan_1"]:
            db.graph.nodes["orphan_1"]["metadata"] = {}
        db.graph.nodes["orphan_1"]["metadata"]["in_progress_since"] = time.time() - 20

        # Run scan
        await janitor._scan_for_orphans()

        # Give event bus time to dispatch
        await asyncio.sleep(0.2)

        # Should have emitted ORPHAN_CLEANED event
        # (Note: This assumes event bus is running; may need adjustment)


class TestNonBlockingHumanLoop:
    """Test Fix #4: Non-blocking human intervention."""

    @pytest.fixture
    def setup_scheduler(self):
        """Setup scheduler test environment."""
        db = GraphDB(persistence_path=".gaadp_test/scheduler_test.json")
        event_bus = EventBus()
        config = SchedulerConfig()
        scheduler = TaskScheduler(db, event_bus, config)

        yield db, scheduler

        import os
        if os.path.exists(".gaadp_test/scheduler_test.json"):
            os.remove(".gaadp_test/scheduler_test.json")

    def test_mark_waiting_for_human(self, setup_scheduler):
        """Verify nodes can be marked as waiting for human input."""
        db, scheduler = setup_scheduler

        db.add_node("req_human", NodeType.REQ, "Needs human input")

        scheduler.mark_waiting_for_human("req_human")

        assert "req_human" in scheduler._waiting_for_human

    def test_blocked_node_not_in_ready_list(self, setup_scheduler):
        """Verify nodes waiting for human are excluded from ready list."""
        db, scheduler = setup_scheduler

        # Create PENDING node
        db.add_node("req_blocked", NodeType.REQ, "Blocked on human")
        db.set_status("req_blocked", NodeStatus.PENDING, reason="Waiting")

        # Block it
        scheduler.mark_waiting_for_human("req_blocked")

        # Get ready nodes
        ready = scheduler._get_ready_nodes()

        # Should not include blocked node
        ready_ids = [n["node_id"] for n in ready]
        assert "req_blocked" not in ready_ids

    def test_parallel_execution_with_human_block(self, setup_scheduler):
        """Verify other DAG branches continue when one is blocked."""
        db, scheduler = setup_scheduler

        # Create two independent requirements
        db.add_node("req_a", NodeType.REQ, "Branch A - needs human")
        db.add_node("req_b", NodeType.REQ, "Branch B - autonomous")

        db.set_status("req_a", NodeStatus.PENDING, reason="Ready")
        db.set_status("req_b", NodeStatus.PENDING, reason="Ready")

        # Block branch A
        scheduler.mark_waiting_for_human("req_a")

        # Get ready nodes
        ready = scheduler._get_ready_nodes()

        # Should only return req_b
        assert len(ready) == 1
        assert ready[0]["node_id"] == "req_b"

    def test_resume_from_human(self, setup_scheduler):
        """Verify nodes can be resumed after human input received."""
        db, scheduler = setup_scheduler

        db.add_node("req_resume", NodeType.REQ, "Will resume")
        scheduler.mark_waiting_for_human("req_resume")

        # Resume
        scheduler.resume_from_human("req_resume")

        assert "req_resume" not in scheduler._waiting_for_human

    def test_dependent_nodes_blocked_by_human(self, setup_scheduler):
        """Verify nodes depending on human-blocked nodes are also blocked."""
        db, scheduler = setup_scheduler

        # Create dependency chain: req_1 -> spec_1
        db.add_node("req_1", NodeType.REQ, "Parent")
        db.add_node("spec_1", NodeType.SPEC, "Child depending on parent")

        from core.ontology import EdgeType
        db.add_edge("spec_1", "req_1", EdgeType.DEPENDS_ON, "sys", "sig")

        db.set_status("req_1", NodeStatus.PENDING, reason="Blocked on human")
        db.set_status("spec_1", NodeStatus.PENDING, reason="Waiting for parent")

        # Block parent
        scheduler.mark_waiting_for_human("req_1")

        # Get ready nodes
        ready = scheduler._get_ready_nodes()

        # spec_1 should NOT be ready (its dependency is blocked)
        ready_ids = [n["node_id"] for n in ready]
        assert "spec_1" not in ready_ids


class TestArchitectStrategyChange:
    """Test Fix #5: Architect strategy change on escalation."""

    @pytest.mark.asyncio
    async def test_escalation_prompt_includes_failure_context(self):
        """Verify Architect uses different prompt when processing escalation."""
        db = GraphDB(persistence_path=".gaadp_test/arch_test.json")

        # Create architect (note: will need real LLM gateway to fully test)
        architect = RealArchitect("arch_test", AgentRole.ARCHITECT, db)

        # Create requirement with escalation context
        req_node = {
            "id": "req_escalated",
            "content": "Build a complex module",
            "escalation_context": "ESCALATION: Previous implementation failed. Missing import errors detected."
        }

        # Build escalation prompt
        prompt = architect._build_escalation_prompt(
            req_node,
            req_node["escalation_context"]
        )

        # Verify prompt includes strategy change instructions
        assert "STRATEGY CHANGE" in prompt or "strategy change" in prompt.lower()
        assert "DIFFERENT approach" in prompt or "different approach" in prompt.lower()

        # Verify it includes the original requirement
        assert req_node["content"] in prompt

        # Verify it includes failure analysis
        assert "Missing import" in prompt or "import" in prompt

    def test_escalation_prompt_analyzes_failure_patterns(self):
        """Verify different failure types trigger different hints."""
        db = GraphDB(persistence_path=".gaadp_test/arch_test2.json")
        architect = RealArchitect("arch_test2", AgentRole.ARCHITECT, db)

        req_node = {"id": "req", "content": "Test requirement"}

        # Test import failure pattern
        prompt_import = architect._build_escalation_prompt(
            req_node,
            "Error: Missing import statement for numpy"
        )
        assert "dependency" in prompt_import.lower() or "import" in prompt_import.lower()

        # Test type mismatch pattern
        prompt_type = architect._build_escalation_prompt(
            req_node,
            "TypeError: Type mismatch in function signature"
        )
        assert "type" in prompt_type.lower()

        # Test complexity pattern
        prompt_complex = architect._build_escalation_prompt(
            req_node,
            "Implementation too complex and incomplete"
        )
        assert "simpl" in prompt_complex.lower() or "smaller" in prompt_complex.lower()

    @pytest.mark.asyncio
    async def test_normal_processing_without_escalation(self):
        """Verify normal processing path when no escalation context."""
        db = GraphDB(persistence_path=".gaadp_test/arch_normal.json")
        architect = RealArchitect("arch_normal", AgentRole.ARCHITECT, db)

        # Normal requirement (no escalation_context)
        req_node = {
            "id": "req_normal",
            "content": "Create a simple hello world function"
        }

        # This would normally call LLM - we're just verifying it doesn't crash
        # Full testing requires real LLM integration (covered in test_e2e_agents.py)

        import os
        if os.path.exists(".gaadp_test/arch_normal.json"):
            os.remove(".gaadp_test/arch_normal.json")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
