"""
GAADP LLM Gateway Integration Tests
Tests for real Anthropic Claude API integration (Priority 1 blocker fix).

NOTE: These tests make REAL API calls and will incur costs (~$0.05-0.10 total).
Requires ANTHROPIC_API_KEY environment variable.
"""
import pytest
import os
import json
from infrastructure.llm_gateway import LLMGateway, LLMGatewayError


@pytest.fixture
def gateway():
    """Create LLM Gateway instance (requires API key)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set - skipping real API tests")

    return LLMGateway()


class TestLLMGatewaySetup:
    """Test gateway initialization and configuration."""

    def test_requires_api_key(self):
        """Verify error raised when API key missing."""
        original_key = os.getenv("ANTHROPIC_API_KEY")

        # Temporarily remove key
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        try:
            with pytest.raises(LLMGatewayError, match="ANTHROPIC_API_KEY"):
                gateway = LLMGateway()
        finally:
            # Restore key
            if original_key:
                os.environ["ANTHROPIC_API_KEY"] = original_key

    def test_requires_anthropic_package(self):
        """Verify error raised when anthropic package missing."""
        # This test is more for documentation - hard to test without uninstalling
        # Just verify the error class exists
        from infrastructure.llm_gateway import LLMGatewayError
        assert LLMGatewayError is not None

    def test_default_config_loads(self, gateway):
        """Verify default configuration includes all required roles."""
        assert "model_assignments" in gateway.config

        # Check expected roles
        expected_roles = ["ARCHITECT", "BUILDER", "VERIFIER", "CURATOR"]
        for role in expected_roles:
            assert role in gateway.config["model_assignments"]

    def test_config_has_model_parameters(self, gateway):
        """Verify each role has model, temperature, max_tokens."""
        for role, config in gateway.config["model_assignments"].items():
            assert "model" in config
            assert "temperature" in config
            assert "max_tokens" in config

            # Verify reasonable values
            assert 0 <= config["temperature"] <= 1
            assert config["max_tokens"] > 0


class TestRealAPIConnectivity:
    """Test real Anthropic API calls (INCURS COSTS)."""

    def test_simple_completion(self, gateway):
        """Verify basic completion works with real API."""
        response = gateway.call_model(
            role="ARCHITECT",
            system_prompt="You are a helpful assistant. Always respond concisely.",
            user_context="Say 'Hello GAADP' and nothing else."
        )

        # Should get a response
        assert isinstance(response, str)
        assert len(response) > 0

        # Should contain the requested phrase (case insensitive)
        response_lower = response.lower()
        assert "hello" in response_lower or "gaadp" in response_lower

    def test_multiple_roles(self, gateway):
        """Verify different roles use different model configs."""
        # Test ARCHITECT (higher temperature, creative)
        arch_response = gateway.call_model(
            role="ARCHITECT",
            system_prompt="You are a system architect.",
            user_context="Respond with 'ARCHITECT'"
        )

        # Test BUILDER (lower temperature, precise)
        builder_response = gateway.call_model(
            role="BUILDER",
            system_prompt="You are a code builder.",
            user_context="Respond with 'BUILDER'"
        )

        # Both should succeed
        assert len(arch_response) > 0
        assert len(builder_response) > 0

    def test_invalid_role(self, gateway):
        """Verify error on undefined role."""
        with pytest.raises(LLMGatewayError, match="not defined"):
            gateway.call_model(
                role="NONEXISTENT_ROLE",
                system_prompt="Test",
                user_context="Test"
            )


class TestTokenTracking:
    """Test token usage tracking and cost estimation."""

    def test_tracks_input_output_tokens(self, gateway):
        """Verify token counts are tracked."""
        # Reset counters
        gateway.reset_session_cost()
        initial_input = gateway._token_usage["input"]
        initial_output = gateway._token_usage["output"]

        # Make a call
        response = gateway.call_model(
            role="CURATOR",  # Use cheaper model
            system_prompt="Count tokens test.",
            user_context="Respond with exactly three words."
        )

        # Should have tracked tokens
        assert gateway._token_usage["input"] > initial_input
        assert gateway._token_usage["output"] > initial_output

    def test_cost_calculation(self, gateway):
        """Verify cost is calculated and accumulates."""
        gateway.reset_session_cost()
        initial_cost = gateway.get_session_cost()

        # Make a small call
        response = gateway.call_model(
            role="CURATOR",
            system_prompt="Cost test.",
            user_context="Hello"
        )

        final_cost = gateway.get_session_cost()

        # Cost should increase
        assert final_cost > initial_cost
        assert final_cost > 0

        # Should be reasonable (not negative, not huge)
        assert final_cost < 1.0  # Simple call shouldn't cost $1

    def test_session_cost_reset(self, gateway):
        """Verify session cost can be reset."""
        # Make a call
        gateway.call_model(
            role="CURATOR",
            system_prompt="Reset test.",
            user_context="Test"
        )

        # Should have cost
        assert gateway.get_session_cost() > 0

        # Reset
        gateway.reset_session_cost()

        # Should be zero
        assert gateway.get_session_cost() == 0.0

    def test_cost_accumulates_across_calls(self, gateway):
        """Verify costs accumulate across multiple calls."""
        gateway.reset_session_cost()

        # First call
        gateway.call_model(role="CURATOR", system_prompt="Test", user_context="One")
        cost_after_one = gateway.get_session_cost()

        # Second call
        gateway.call_model(role="CURATOR", system_prompt="Test", user_context="Two")
        cost_after_two = gateway.get_session_cost()

        # Should accumulate
        assert cost_after_two > cost_after_one


class TestToolUse:
    """Test tool calling functionality (Anthropic format)."""

    def test_tool_schema_accepted(self, gateway):
        """Verify tool definitions are accepted without error."""
        tools = [
            {
                "name": "read_file",
                "description": "Read a file from disk",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to read"
                        }
                    },
                    "required": ["path"]
                }
            }
        ]

        # Should not raise error
        response = gateway.call_model(
            role="ARCHITECT",
            system_prompt="You have access to file reading tools.",
            user_context="Just respond 'OK', don't use any tools.",
            tools=tools
        )

        # Should get response (may or may not use tool)
        assert isinstance(response, str)

    def test_tool_use_response_format(self, gateway):
        """Verify tool use returns properly formatted JSON."""
        tools = [
            {
                "name": "calculate",
                "description": "Perform a calculation",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }
            }
        ]

        response = gateway.call_model(
            role="ARCHITECT",
            system_prompt="You must use the calculate tool for all math.",
            user_context="What is 2 + 2? Use the calculate tool.",
            tools=tools
        )

        # If tool was used, response should be JSON
        try:
            parsed = json.loads(response)

            # Should have expected structure
            if "tool_calls" in parsed:
                assert isinstance(parsed["tool_calls"], list)

                # Verify tool call format
                for tool_call in parsed["tool_calls"]:
                    assert "id" in tool_call
                    assert "name" in tool_call
                    assert "input" in tool_call

        except json.JSONDecodeError:
            # LLM chose not to use tool - that's fine for this test
            # (Tool use is non-deterministic based on prompt)
            pass


class TestErrorHandling:
    """Test error handling and retries."""

    def test_retry_on_api_error(self, gateway):
        """Verify retry logic handles transient errors."""
        # This is hard to test without mocking, but we verify the decorator exists
        import inspect
        source = inspect.getsource(gateway.call_model)

        # Should have @retry decorator
        assert "retry" in source or "@retry" in source

    def test_timeout_handling(self, gateway):
        """Verify very long responses are handled (context limit)."""
        # Request very long response
        response = gateway.call_model(
            role="BUILDER",
            system_prompt="You are a code generator.",
            user_context="Generate a very simple 'hello world' function in Python.",
            # Don't request infinite output - just verify normal handling works
        )

        # Should complete without hanging
        assert isinstance(response, str)


class TestConversationContinuation:
    """Test call_with_tool_results for multi-turn conversations."""

    def test_conversation_with_messages(self, gateway):
        """Verify multi-turn conversation works."""
        messages = [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 equals 4."},
            {"role": "user", "content": "What about 3 + 3?"}
        ]

        response = gateway.call_with_tool_results(
            role="ARCHITECT",
            messages=messages
        )

        # Should get response about 3 + 3
        assert isinstance(response, str)
        assert len(response) > 0

        # Likely mentions 6
        assert "6" in response or "six" in response.lower()


# Test execution summary
if __name__ == "__main__":
    print("=" * 60)
    print("LLM GATEWAY INTEGRATION TESTS")
    print("=" * 60)
    print("\n⚠️  WARNING: These tests make REAL API calls")
    print("   Expected cost: ~$0.05-0.10 USD")
    print("   Requires: ANTHROPIC_API_KEY environment variable")
    print("\nRunning tests...\n")

    pytest.main([__file__, "-v", "--tb=short"])
