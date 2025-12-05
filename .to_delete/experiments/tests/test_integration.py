"""
GAADP Integration Tests
End-to-end tests for the complete pipeline.
"""
import pytest
import os
import shutil
import asyncio
from core.ontology import NodeType, EdgeType, NodeStatus
from infrastructure.graph_db import GraphDB
from infrastructure.sandbox import CodeSandbox
from infrastructure.version_control import GitController
from infrastructure.test_runner import TestRunner
from infrastructure.event_bus import EventBus, MessageType


class TestGraphDBIntegration:
    """Test GraphDB with persistence and Merkle chaining."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_dir = ".gaadp_integration_test"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        yield
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_persistence_roundtrip(self):
        """Test that graph persists and reloads correctly."""
        db_path = f"{self.test_dir}/graph.pkl"

        # Create and populate
        db1 = GraphDB(persistence_path=db_path)
        db1.add_node("req_1", NodeType.REQ, "Test requirement")
        db1.add_node("spec_1", NodeType.SPEC, "Test spec")
        db1.add_edge("spec_1", "req_1", EdgeType.TRACES_TO, "agent_1", "sig_1")

        # Reload
        db2 = GraphDB(persistence_path=db_path)
        assert db2.graph.number_of_nodes() == 2
        assert db2.graph.number_of_edges() == 1

    def test_cycle_prevention(self):
        """Test that cycles are prevented."""
        db = GraphDB(persistence_path=f"{self.test_dir}/graph.pkl")
        db.add_node("a", NodeType.CODE, "code a")
        db.add_node("b", NodeType.CODE, "code b")
        db.add_edge("a", "b", EdgeType.DEPENDS_ON, "agent", "sig")

        with pytest.raises(ValueError, match="Cycle detected"):
            db.add_edge("b", "a", EdgeType.DEPENDS_ON, "agent", "sig")

    def test_token_limited_context(self):
        """Test that context retrieval respects token limits."""
        db = GraphDB(persistence_path=f"{self.test_dir}/graph.pkl")
        db.add_node("center", NodeType.SPEC, "x" * 100)
        db.add_node("large", NodeType.CODE, "y" * 50000)
        db.add_edge("center", "large", EdgeType.DEPENDS_ON, "agent", "sig")

        context = db.get_context_neighborhood("center", radius=2, max_tokens=500)
        # Should include center but possibly truncate large node
        assert "center" in [n['id'] for n in context.get('nodes', [])]

    def test_merkle_hash_chain(self):
        """Test Merkle hash retrieval."""
        db = GraphDB(persistence_path=f"{self.test_dir}/graph.pkl")
        db.add_node("code_1", NodeType.CODE, "test code")

        # Initially should be GENESIS
        assert db.get_last_node_hash() == "GENESIS"

        # Add a verified edge
        db.add_edge("code_1", "code_1", EdgeType.VERIFIES, "verifier", "real_sig_123")

        # Now should return the signature
        assert db.get_last_node_hash() == "real_sig_123"


class TestSandboxIntegration:
    """Test code sandbox execution."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_dir = ".sandbox_test"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        yield
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_successful_execution(self):
        """Test running valid Python code."""
        test_file = f"{self.test_dir}/hello.py"
        with open(test_file, "w") as f:
            f.write("print('Hello GAADP')")

        sandbox = CodeSandbox(use_docker=False)
        result = sandbox.run_code(test_file)

        assert result["exit_code"] == 0
        assert "Hello GAADP" in result["stdout"]

    def test_timeout_handling(self):
        """Test that infinite loops are caught."""
        test_file = f"{self.test_dir}/infinite.py"
        with open(test_file, "w") as f:
            f.write("while True: pass")

        sandbox = CodeSandbox(use_docker=False)
        result = sandbox.run_code(test_file, timeout=1)

        assert result["exit_code"] == 124
        assert "Timed Out" in result["stderr"]

    def test_error_capture(self):
        """Test that errors are captured."""
        test_file = f"{self.test_dir}/error.py"
        with open(test_file, "w") as f:
            f.write("raise ValueError('test error')")

        sandbox = CodeSandbox(use_docker=False)
        result = sandbox.run_code(test_file)

        assert result["exit_code"] != 0
        assert "ValueError" in result["stderr"]


class TestEventBusIntegration:
    """Test async event bus."""

    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        """Test basic pub/sub functionality."""
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe("test_topic", handler)
        await bus.publish("test_topic", MessageType.TASK_ASSIGN, {"data": "test"}, "agent_1")

        # Process the queue
        event = await bus.queue.get()
        await bus._dispatch(event)

        await asyncio.sleep(0.1)  # Allow handler to complete
        assert len(received) == 1
        assert received[0]["payload"]["data"] == "test"

    def test_history_tracking(self):
        """Test that event history is maintained."""
        bus = EventBus()

        async def run():
            await bus.publish("topic1", "TYPE_A", {}, "agent")
            await bus.publish("topic2", "TYPE_B", {}, "agent")
            await bus.publish("topic1", "TYPE_C", {}, "agent")

        asyncio.run(run())

        assert len(bus.history) == 3
        topic1_events = bus.get_history("topic1")
        assert len(topic1_events) == 2


class TestTestRunnerIntegration:
    """Test the test runner itself."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_dir = ".test_runner_test"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        yield
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_inline_test_pass(self):
        """Test running inline test code that passes."""
        runner = TestRunner(work_dir=self.test_dir)
        test_code = """
def test_simple():
    assert 1 + 1 == 2
"""
        result = runner.run_inline_test(test_code)
        assert result["status"] == "PASS"
        assert result["passed"] >= 1

    def test_inline_test_fail(self):
        """Test running inline test code that fails."""
        runner = TestRunner(work_dir=self.test_dir)
        test_code = """
def test_failing():
    assert 1 == 2
"""
        result = runner.run_inline_test(test_code)
        assert result["status"] == "FAIL"
        assert result["failed"] >= 1


class TestEndToEndPipeline:
    """Test the complete generation pipeline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.test_dir = ".e2e_test"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        yield
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_graph_to_code_pipeline(self):
        """Test creating a node, materializing, and executing."""
        db = GraphDB(persistence_path=f"{self.test_dir}/graph.pkl")

        # 1. Create nodes
        db.add_node("req_1", NodeType.REQ, "Calculate sum")
        db.add_node("code_1", NodeType.CODE, "def add(a, b): return a + b", {
            "file_path": f"{self.test_dir}/math_utils.py",
            "language": "python"
        })
        db.add_edge("code_1", "req_1", EdgeType.IMPLEMENTS, "builder", "sig")

        # 2. Verify
        db.add_edge("code_1", "code_1", EdgeType.VERIFIES, "verifier", "verify_sig")
        db.graph.nodes["code_1"]["status"] = NodeStatus.VERIFIED.value

        # 3. Materialize
        nodes = db.get_materializable_nodes()
        assert len(nodes) == 1

        file_path = nodes[0]["file_path"]
        with open(file_path, "w") as f:
            f.write(nodes[0]["content"])

        # 4. Execute
        test_file = f"{self.test_dir}/test_math.py"
        with open(test_file, "w") as f:
            f.write(f"""
import sys
sys.path.insert(0, '{self.test_dir}')
from math_utils import add

def test_add():
    assert add(2, 3) == 5
""")

        runner = TestRunner(work_dir=self.test_dir)
        result = runner.run_pytest(test_file)
        assert result["status"] == "PASS"
