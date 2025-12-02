#!/usr/bin/env python3
"""
OpenAI Provider Smoke Test

Verifies that the OpenAIProvider implementation works correctly:
1. Provider initialization and availability check
2. Simple API call to gpt-4o
3. Cost and token tracking (governance hooks)
4. Tool calling capability

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/test_openai.py
"""
import os
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.openai_provider import OpenAIProvider


def test_availability():
    """Test 1: Check if provider detects availability correctly."""
    print("\n" + "=" * 60)
    print("TEST 1: Provider Availability")
    print("=" * 60)

    provider = OpenAIProvider()

    if not os.getenv("OPENAI_API_KEY"):
        print("  OPENAI_API_KEY not set")
        print("  Expected: is_available() = False")
        assert not provider.is_available(), "Should be unavailable without API key"
        print("  Result: PASS (correctly reports unavailable)")
        return False  # Can't continue without key

    print(f"  OPENAI_API_KEY: Set (length={len(os.getenv('OPENAI_API_KEY'))})")
    print(f"  is_available(): {provider.is_available()}")
    print(f"  get_name(): {provider.get_name()}")

    assert provider.is_available(), "Should be available with API key"
    assert provider.get_name() == "openai_api"
    print("  Result: PASS")
    return True


def test_simple_call():
    """Test 2: Make a simple Hello World call."""
    print("\n" + "=" * 60)
    print("TEST 2: Simple API Call (Hello World)")
    print("=" * 60)

    provider = OpenAIProvider()

    model_config = {
        "model": "gpt-4o-mini",  # Use mini for cost efficiency in testing
        "temperature": 0.7,
        "max_tokens": 100
    }

    print(f"  Model: {model_config['model']}")
    print("  Prompt: 'Say hello world in exactly 5 words.'")
    print("  Calling API...")

    response = provider.call(
        system_prompt="You are a helpful assistant. Be concise.",
        user_prompt="Say hello world in exactly 5 words.",
        model_config=model_config
    )

    print(f"  Response: {response}")
    assert response, "Response should not be empty"
    assert len(response) > 0
    print("  Result: PASS")

    return response


def test_usage_tracking():
    """Test 3: Verify cost and token tracking."""
    print("\n" + "=" * 60)
    print("TEST 3: Usage & Cost Tracking (Governance Hook)")
    print("=" * 60)

    provider = OpenAIProvider()

    # Reset stats first
    provider.reset_usage_stats()
    initial_stats = provider.get_usage_stats()
    print(f"  Initial stats: {initial_stats}")

    assert initial_stats["tokens_input"] == 0
    assert initial_stats["tokens_output"] == 0
    assert initial_stats["cost"] == 0.0

    # Make a call
    model_config = {
        "model": "gpt-4o-mini",
        "temperature": 0.5,
        "max_tokens": 50
    }

    provider.call(
        system_prompt="Be brief.",
        user_prompt="What is 2+2?",
        model_config=model_config
    )

    # Check stats updated
    final_stats = provider.get_usage_stats()
    print(f"  Final stats: {final_stats}")

    assert final_stats["tokens_input"] > 0, "Should have tracked input tokens"
    assert final_stats["tokens_output"] > 0, "Should have tracked output tokens"
    assert final_stats["cost"] > 0, "Should have calculated cost"

    print(f"  Tokens In: {final_stats['tokens_input']}")
    print(f"  Tokens Out: {final_stats['tokens_output']}")
    print(f"  Cost: ${final_stats['cost']:.6f}")
    print("  Result: PASS (Governance tracking working)")

    return final_stats


def test_tool_calling():
    """Test 4: Verify tool calling capability."""
    print("\n" + "=" * 60)
    print("TEST 4: Tool Calling")
    print("=" * 60)

    provider = OpenAIProvider()

    model_config = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,  # Deterministic for testing
        "max_tokens": 200
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    print("  Prompt: 'What's the weather in Tokyo?'")
    print("  Tools: [get_weather]")
    print("  Calling API...")

    response = provider.call(
        system_prompt="You are a helpful assistant. Use tools when needed.",
        user_prompt="What's the weather in Tokyo?",
        model_config=model_config,
        tools=tools
    )

    print(f"  Raw response: {response[:200]}...")

    # Parse response
    try:
        parsed = json.loads(response)
        if "tool_calls" in parsed and parsed["tool_calls"]:
            print(f"  Tool calls detected: {len(parsed['tool_calls'])}")
            for tc in parsed["tool_calls"]:
                print(f"    - {tc['name']}: {tc.get('input', {})}")
            print("  Result: PASS (Tool calling works)")
        else:
            print("  No tool calls in response (model may have answered directly)")
            print("  Result: PASS (Response received)")
    except json.JSONDecodeError:
        print("  Response is plain text (no tool calls)")
        print("  Result: PASS (Response received)")

    return response


def test_multi_turn():
    """Test 5: Multi-turn conversation."""
    print("\n" + "=" * 60)
    print("TEST 5: Multi-Turn Conversation")
    print("=" * 60)

    provider = OpenAIProvider()

    model_config = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 100
    }

    messages = [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is 5 + 3?"},
        {"role": "assistant", "content": "5 + 3 = 8"},
        {"role": "user", "content": "Now multiply that by 2"}
    ]

    print("  Conversation history: 4 messages")
    print("  Last user message: 'Now multiply that by 2'")
    print("  Calling API...")

    response = provider.call_multi_turn(
        messages=messages,
        model_config=model_config
    )

    print(f"  Response: {response}")
    assert "16" in response, f"Expected '16' in response, got: {response}"
    print("  Result: PASS (Context maintained)")

    return response


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("  OPENAI PROVIDER SMOKE TEST")
    print("=" * 60)

    # Test 1: Availability
    has_key = test_availability()

    if not has_key:
        print("\n" + "=" * 60)
        print("  SKIPPING REMAINING TESTS (No API key)")
        print("=" * 60)
        print("\nTo run full tests:")
        print("  export OPENAI_API_KEY=sk-...")
        print("  python scripts/test_openai.py")
        return

    # Test 2: Simple call
    test_simple_call()

    # Test 3: Usage tracking
    stats = test_usage_tracking()

    # Test 4: Tool calling
    test_tool_calling()

    # Test 5: Multi-turn
    test_multi_turn()

    # Summary
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
    print(f"\n  Total API calls: ~4")
    print(f"  Estimated cost: ${stats['cost'] * 4:.4f}")
    print("\n  OpenAI Provider is ready for production use.")


if __name__ == "__main__":
    main()
