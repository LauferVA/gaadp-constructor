"""
GAADP PHYSICS VERIFICATION TESTS
================================
Deterministic unit tests proving the system mechanics work correctly.
These tests do NOT require API calls - they verify the "physics engine."

Levels:
- Level 0: Import health
- Level 1: TransitionMatrix validation
- Level 2: Governance/NodeMetadata enforcement
- Level 3: Protocol stress testing
"""
import pytest
import json
from datetime import datetime, timezone
from pydantic import ValidationError


# =============================================================================
# LEVEL 0: IMPORT HEALTH
# =============================================================================

class TestLevel0Imports:
    """Verify all core modules import without circular dependencies."""

    def test_core_ontology_imports(self):
        """core.ontology should import cleanly."""
        from core.ontology import (
            NodeType, EdgeType, NodeStatus,
            NodeSpec, EdgeSpec, NodeMetadata,
            TRANSITION_MATRIX, AGENT_DISPATCH, KNOWN_CONDITIONS,
            TransitionRule, validate_transition_matrix
        )
        assert NodeType.CODE is not None
        assert EdgeType.VERIFIES is not None
        assert NodeStatus.VERIFIED is not None

    def test_graph_runtime_imports(self):
        """infrastructure.graph_runtime should import cleanly."""
        from infrastructure.graph_runtime import GraphRuntime
        assert GraphRuntime is not None

    def test_protocols_imports(self):
        """core.protocols should import cleanly."""
        from core.protocols import (
            ArchitectOutput, BuilderOutput, VerifierOutput,
            UnifiedAgentOutput, GraphContext,
            protocol_to_tool_schema, validate_agent_output
        )
        assert ArchitectOutput is not None
        assert BuilderOutput is not None
        assert VerifierOutput is not None

    def test_graph_db_imports(self):
        """infrastructure.graph_db should import cleanly."""
        from infrastructure.graph_db import GraphDB
        assert GraphDB is not None

    def test_no_circular_imports(self):
        """Full import chain should not cause circular import errors."""
        # This would fail during import if there were circular deps
        import core.ontology
        import core.protocols
        import infrastructure.graph_db
        import infrastructure.graph_runtime
        import agents.generic_agent
        assert True  # If we get here, no circular imports


# =============================================================================
# LEVEL 1: TRANSITION MATRIX VALIDATION
# =============================================================================

class TestLevel1TransitionMatrix:
    """Verify TransitionMatrix physics are correctly defined."""

    def test_all_conditions_are_known(self):
        """Every condition in TransitionMatrix must be in KNOWN_CONDITIONS."""
        from core.ontology import TRANSITION_MATRIX, KNOWN_CONDITIONS, validate_transition_matrix

        errors = validate_transition_matrix()
        assert len(errors) == 0, f"Unknown conditions found: {errors}"

    def test_code_to_verified_requires_verifies_edge(self):
        """CODE -> VERIFIED is IMPOSSIBLE without a VERIFIES edge."""
        from core.ontology import (
            TRANSITION_MATRIX, NodeStatus, NodeType, EdgeType
        )

        key = (NodeStatus.PROCESSING.value, NodeType.CODE.value)
        rules = TRANSITION_MATRIX.get(key, [])

        # Find the rule for VERIFIED transition
        verified_rules = [r for r in rules if r.target_status == NodeStatus.VERIFIED]
        assert len(verified_rules) > 0, "Should have a rule for CODE -> VERIFIED"

        verify_rule = verified_rules[0]
        # Must require VERIFIES edge
        required_edges = [
            e.value if hasattr(e, 'value') else e
            for e in verify_rule.required_edge_types
        ]
        assert EdgeType.VERIFIES.value in required_edges, \
            "CODE -> VERIFIED must require VERIFIES edge"

    def test_req_to_processing_allowed_when_conditions_met(self):
        """REQ -> PROCESSING is allowed when cost_under_limit and not_blocked."""
        from core.ontology import (
            TRANSITION_MATRIX, NodeStatus, NodeType
        )

        key = (NodeStatus.PENDING.value, NodeType.REQ.value)
        rules = TRANSITION_MATRIX.get(key, [])

        # Should have a path to PROCESSING
        processing_rules = [r for r in rules if r.target_status == NodeStatus.PROCESSING]
        assert len(processing_rules) > 0, "Should have rule for REQ PENDING -> PROCESSING"

        # Check conditions
        rule = processing_rules[0]
        assert "cost_under_limit" in rule.required_conditions, \
            "REQ -> PROCESSING should require cost_under_limit"

    def test_spec_to_failed_on_max_attempts(self):
        """SPEC -> FAILED when max_attempts_exceeded."""
        from core.ontology import (
            TRANSITION_MATRIX, NodeStatus, NodeType
        )

        key = (NodeStatus.PROCESSING.value, NodeType.SPEC.value)
        rules = TRANSITION_MATRIX.get(key, [])

        failed_rules = [r for r in rules if r.target_status == NodeStatus.FAILED]
        assert len(failed_rules) > 0, "Should have rule for SPEC -> FAILED"

        rule = failed_rules[0]
        assert "max_attempts_exceeded" in rule.required_conditions, \
            "SPEC -> FAILED should require max_attempts_exceeded"

    def test_clarification_requires_resolved_by_edge(self):
        """CLARIFICATION -> VERIFIED requires RESOLVED_BY edge."""
        from core.ontology import (
            TRANSITION_MATRIX, NodeStatus, NodeType, EdgeType
        )

        key = (NodeStatus.PROCESSING.value, NodeType.CLARIFICATION.value)
        rules = TRANSITION_MATRIX.get(key, [])

        verified_rules = [r for r in rules if r.target_status == NodeStatus.VERIFIED]
        assert len(verified_rules) > 0, "Should have rule for CLARIFICATION -> VERIFIED"

        rule = verified_rules[0]
        required_edges = [
            e.value if hasattr(e, 'value') else e
            for e in rule.required_edge_types
        ]
        assert EdgeType.RESOLVED_BY.value in required_edges, \
            "CLARIFICATION -> VERIFIED must require RESOLVED_BY edge"

    def test_all_node_types_have_pending_rules(self):
        """Every processable NodeType should have PENDING -> * rules."""
        from core.ontology import TRANSITION_MATRIX, NodeStatus, NodeType

        processable_types = [
            NodeType.REQ, NodeType.SPEC, NodeType.CODE,
            NodeType.CLARIFICATION, NodeType.ESCALATION
        ]

        for node_type in processable_types:
            key = (NodeStatus.PENDING.value, node_type.value)
            rules = TRANSITION_MATRIX.get(key, [])
            assert len(rules) > 0, f"{node_type.value} should have PENDING transition rules"


# =============================================================================
# LEVEL 2: GOVERNANCE PHYSICS (NodeMetadata)
# =============================================================================

class TestLevel2GovernancePhysics:
    """Verify NodeMetadata constraints are enforced as physics."""

    @pytest.fixture
    def mock_graph_db(self):
        """Create a mock graph for testing."""
        from infrastructure.graph_db import GraphDB
        import tempfile
        import os

        # Use temp file for isolation
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        db = GraphDB(persistence_path=temp_path)
        yield db

        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass

    @pytest.fixture
    def runtime(self, mock_graph_db):
        """Create a runtime with mock graph."""
        from infrastructure.graph_runtime import GraphRuntime
        return GraphRuntime(mock_graph_db)

    def test_cost_limit_zero_blocks_processing(self, runtime, mock_graph_db):
        """Node with cost_limit=0.0 should be rejected by cost_under_limit."""
        from core.ontology import NodeType, NodeStatus

        # Create node with zero cost limit
        mock_graph_db.add_node(
            node_id="zero_cost_node",
            node_type=NodeType.REQ,
            content="Test requirement",
            metadata={"cost_limit": 0.0, "cost_actual": 0.0}
        )

        # Evaluate cost condition
        result = runtime.evaluate_condition("zero_cost_node", "cost_under_limit")

        # cost_actual (0.0) is NOT < cost_limit (0.0), so should be False
        assert result == False, "cost_under_limit should be False when cost_limit=0.0"

    def test_cost_unlimited_allows_processing(self, runtime, mock_graph_db):
        """Node with cost_limit=None (unlimited) should pass cost check."""
        from core.ontology import NodeType

        mock_graph_db.add_node(
            node_id="unlimited_node",
            node_type=NodeType.REQ,
            content="Unlimited budget requirement",
            metadata={"cost_limit": None, "cost_actual": 100.0}
        )

        result = runtime.evaluate_condition("unlimited_node", "cost_under_limit")
        assert result == True, "cost_under_limit should be True when cost_limit=None"

    def test_max_attempts_exceeded_triggers_failed(self, runtime, mock_graph_db):
        """Node with attempts >= max_attempts should trigger max_attempts_exceeded."""
        from core.ontology import NodeType

        # First attempt
        mock_graph_db.add_node(
            node_id="failing_node",
            node_type=NodeType.SPEC,
            content="Spec that fails",
            metadata={"attempts": 0, "max_attempts": 1}
        )

        # Not exceeded yet
        result = runtime.evaluate_condition("failing_node", "max_attempts_exceeded")
        assert result == False, "Should not be exceeded at attempts=0"

        # Simulate failure (increment attempts)
        mock_graph_db.graph.nodes["failing_node"]["metadata"]["attempts"] = 1

        # Now exceeded
        result = runtime.evaluate_condition("failing_node", "max_attempts_exceeded")
        assert result == True, "Should be exceeded at attempts=1 with max_attempts=1"

    def test_security_level_enforcement(self):
        """NodeMetadata.security_level must be 0-3."""
        from core.ontology import NodeMetadata

        # Valid levels
        for level in [0, 1, 2, 3]:
            meta = NodeMetadata(security_level=level)
            assert meta.security_level == level

        # Invalid levels should raise
        with pytest.raises(ValidationError):
            NodeMetadata(security_level=-1)

        with pytest.raises(ValidationError):
            NodeMetadata(security_level=4)

    def test_dependencies_verified_condition(self, runtime, mock_graph_db):
        """dependencies_verified should check all DEPENDS_ON targets."""
        from core.ontology import NodeType, EdgeType, NodeStatus

        # Create dependency chain: child depends on parent
        # Edge direction: parent -> child with DEPENDS_ON means child depends on parent
        mock_graph_db.add_node(
            node_id="parent_spec",
            node_type=NodeType.SPEC,
            content="Parent specification"
        )
        mock_graph_db.add_node(
            node_id="child_spec",
            node_type=NodeType.SPEC,
            content="Child specification"
        )
        # Edge from parent to child means child depends on parent
        mock_graph_db.add_edge(
            source_id="parent_spec",
            target_id="child_spec",
            edge_type=EdgeType.DEPENDS_ON,
            signed_by="test",
            signature="test_sig"
        )

        # Parent not verified yet - child's predecessor (parent) is not verified
        result = runtime.evaluate_condition("child_spec", "dependencies_verified")
        assert result == False, "Should be False when parent (predecessor) is PENDING"

        # Verify parent
        mock_graph_db.set_status("parent_spec", NodeStatus.VERIFIED)

        result = runtime.evaluate_condition("child_spec", "dependencies_verified")
        assert result == True, "Should be True when parent (predecessor) is VERIFIED"


# =============================================================================
# LEVEL 3: PROTOCOL STRESS TESTING
# =============================================================================

class TestLevel3ProtocolStress:
    """Test Pydantic protocol resilience to invalid input."""

    def test_architect_output_rejects_garbage(self):
        """ArchitectOutput.model_validate should raise ValidationError on garbage."""
        from core.protocols import ArchitectOutput

        garbage_inputs = [
            None,
            "just a string",
            12345,
            {"invalid": "structure"},
            {"specs": "not a list"},
            {"specs": [{"missing": "required fields"}]},
        ]

        for garbage in garbage_inputs:
            with pytest.raises((ValidationError, TypeError)):
                if garbage is None:
                    ArchitectOutput.model_validate(garbage)
                else:
                    ArchitectOutput.model_validate(garbage)

    def test_builder_output_rejects_garbage(self):
        """BuilderOutput.model_validate should raise ValidationError on garbage."""
        from core.protocols import BuilderOutput

        with pytest.raises(ValidationError):
            BuilderOutput.model_validate({"random": "garbage"})

        with pytest.raises(ValidationError):
            BuilderOutput.model_validate({"code": 12345})  # code must be string

    def test_verifier_output_rejects_garbage(self):
        """VerifierOutput.model_validate should raise ValidationError on garbage."""
        from core.protocols import VerifierOutput

        with pytest.raises(ValidationError):
            VerifierOutput.model_validate({"not": "valid"})

        with pytest.raises(ValidationError):
            # approved must be bool
            VerifierOutput.model_validate({"approved": "yes please"})

    def test_unified_agent_output_handles_malformed_json(self):
        """UnifiedAgentOutput should handle malformed input gracefully."""
        from core.protocols import UnifiedAgentOutput

        # Valid minimal output (all fields have defaults)
        valid = UnifiedAgentOutput(
            thought="I am thinking"
        )
        assert valid.thought == "I am thinking"

        # Empty dict is actually valid (all fields have defaults)
        empty = UnifiedAgentOutput.model_validate({})
        assert empty.thought is None

        # But wrong types should fail
        with pytest.raises(ValidationError):
            UnifiedAgentOutput.model_validate({"new_nodes": "not a list"})

        with pytest.raises(ValidationError):
            UnifiedAgentOutput.model_validate({"cost_incurred": "not a float"})

    def test_graph_context_validates_structure(self):
        """GraphContext should validate its structure."""
        from core.protocols import GraphContext

        # Valid context
        ctx = GraphContext(
            node_id="test123",
            node_type="CODE",
            node_content="print('hello')",
            node_status="PENDING"
        )
        assert ctx.node_id == "test123"

        # Invalid context - missing required fields
        with pytest.raises(ValidationError):
            GraphContext.model_validate({"node_id": "only_id"})

    def test_protocol_to_tool_schema_produces_valid_schema(self):
        """protocol_to_tool_schema should produce valid Anthropic tool schema."""
        from core.protocols import ArchitectOutput, protocol_to_tool_schema

        schema = protocol_to_tool_schema(ArchitectOutput, "submit_architecture")

        assert "name" in schema
        assert schema["name"] == "submit_architecture"
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"

    def test_edge_cases_dont_crash(self):
        """Various edge cases should not crash the system."""
        from core.ontology import NodeSpec, NodeMetadata, NodeType, NodeStatus

        # Empty content
        spec = NodeSpec(
            type=NodeType.CODE,
            content="",
            created_by="test"
        )
        assert spec.content == ""

        # Very long content
        long_content = "x" * 100000
        spec2 = NodeSpec(
            type=NodeType.DOC,
            content=long_content,
            created_by="test"
        )
        assert len(spec2.content) == 100000

        # Unicode content
        unicode_content = "def hello(): return '你好世界 🌍'"
        spec3 = NodeSpec(
            type=NodeType.CODE,
            content=unicode_content,
            created_by="test"
        )
        assert "你好" in spec3.content

        # Metadata with extra fields
        meta = NodeMetadata(
            cost_limit=1.0,
            extra={"custom_field": "custom_value", "nested": {"a": 1}}
        )
        assert meta.extra["custom_field"] == "custom_value"


# =============================================================================
# INTEGRATION: RUNTIME PHYSICS
# =============================================================================

class TestRuntimePhysics:
    """Test that GraphRuntime correctly enforces physics."""

    @pytest.fixture
    def isolated_runtime(self):
        """Create isolated runtime for testing."""
        from infrastructure.graph_db import GraphDB
        from infrastructure.graph_runtime import GraphRuntime
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        db = GraphDB(persistence_path=temp_path)
        runtime = GraphRuntime(db)

        yield runtime, db

        try:
            os.unlink(temp_path)
        except:
            pass

    def test_can_transition_enforces_rules(self, isolated_runtime):
        """can_transition should enforce TransitionMatrix rules."""
        runtime, db = isolated_runtime
        from core.ontology import NodeType, NodeStatus

        # Create a CODE node in PROCESSING state
        db.add_node(
            node_id="code_no_test",
            node_type=NodeType.CODE,
            content="print('hello')"
        )
        db.set_status("code_no_test", NodeStatus.PROCESSING)

        # Try to transition to VERIFIED without VERIFIES edge
        allowed, rule = runtime.can_transition("code_no_test", NodeStatus.VERIFIED)
        assert allowed == False, "Should NOT allow CODE -> VERIFIED without VERIFIES edge"

    def test_get_agent_for_node_uses_dispatch_rules(self, isolated_runtime):
        """get_agent_for_node should consult AGENT_DISPATCH."""
        runtime, db = isolated_runtime
        from core.ontology import NodeType

        # REQ with no children -> should dispatch to ARCHITECT
        db.add_node(
            node_id="new_req",
            node_type=NodeType.REQ,
            content="Build something"
        )

        agent = runtime.get_agent_for_node("new_req")
        assert agent == "ARCHITECT", f"REQ needing decomposition should get ARCHITECT, got {agent}"

    def test_get_processable_nodes_respects_physics(self, isolated_runtime):
        """get_processable_nodes should only return physics-allowed nodes."""
        runtime, db = isolated_runtime
        from core.ontology import NodeType, EdgeType, NodeStatus

        # Create a REQ with zero budget (blocked by cost)
        db.add_node(
            node_id="blocked_req",
            node_type=NodeType.REQ,
            content="This should be blocked",
            metadata={"cost_limit": 0.0}
        )

        # Create a REQ with budget (allowed)
        db.add_node(
            node_id="allowed_req",
            node_type=NodeType.REQ,
            content="This should be allowed",
            metadata={"cost_limit": 10.0}
        )

        processable = runtime.get_processable_nodes()

        assert "blocked_req" not in processable, "Zero-budget node should not be processable"
        assert "allowed_req" in processable, "Budgeted node should be processable"


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
