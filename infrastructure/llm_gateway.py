"""
GAADP LLM GATEWAY
Uses Anthropic Claude API for agent intelligence.
Supports standard completion and tool-augmented calls.
"""
import os
import yaml
import json
import logging
from typing import List, Dict, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from anthropic import Anthropic, AnthropicError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None
    AnthropicError = Exception

logger = logging.getLogger("LLM_Gateway")


class LLMGatewayError(Exception):
    """Raised when LLM Gateway encounters errors."""
    pass


class LLMGateway:
    """
    LLM Gateway using Anthropic Claude API.

    Requires:
    - anthropic package: pip install anthropic
    - ANTHROPIC_API_KEY environment variable
    """

    def __init__(self, config_path: str = ".blueprint/llm_router.yaml"):
        if not ANTHROPIC_AVAILABLE:
            raise LLMGatewayError(
                "anthropic package required. Install: pip install anthropic"
            )

        # Get API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMGatewayError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Get your key from: https://console.anthropic.com/settings/keys"
            )

        self.client = Anthropic(api_key=api_key)

        # Load config
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found: {config_path}. Using defaults.")
            self.config = self._default_config()

        # Cost tracking
        self._cost_session = 0.0
        self._token_usage = {"input": 0, "output": 0}

    def _default_config(self) -> Dict:
        """Default configuration if llm_router.yaml missing."""
        return {
            "model_assignments": {
                "ARCHITECT": {"model": "claude-3-5-sonnet-20241022", "temperature": 0.7, "max_tokens": 4000},
                "BUILDER": {"model": "claude-3-5-sonnet-20241022", "temperature": 0.3, "max_tokens": 8000},
                "VERIFIER": {"model": "claude-3-5-sonnet-20241022", "temperature": 0.1, "max_tokens": 4000},
                "CURATOR": {"model": "claude-3-5-haiku-20241022", "temperature": 0.5, "max_tokens": 2000},
            }
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_model(
        self,
        role: str,
        system_prompt: str,
        user_context: str,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """
        Call Claude with optional tool definitions.

        Args:
            role: Agent role (maps to model config)
            system_prompt: System message
            user_context: User message
            tools: Optional list of tool definitions for function calling

        Returns:
            Response content as string (or JSON if tool calls present)
        """
        role_config = self.config['model_assignments'].get(role)
        if not role_config:
            raise LLMGatewayError(f"Role {role} not defined in config")

        try:
            kwargs = {
                "model": role_config['model'],
                "max_tokens": role_config.get('max_tokens', 4000),
                "temperature": role_config.get('temperature', 0.7),
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_context}]
            }

            # Add tools if provided
            if tools:
                kwargs["tools"] = tools

            response = self.client.messages.create(**kwargs)

        except AnthropicError as e:
            logger.error(f"Anthropic API Error: {e}")
            raise LLMGatewayError(f"API call failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise LLMGatewayError(f"Unexpected error: {e}")

        # Track usage and costs
        self._track_usage(response)

        # Handle response
        if response.stop_reason == "tool_use":
            return self._format_tool_response(response)

        # Extract text content
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text

        return text_content

    def _track_usage(self, response):
        """Track token usage and estimate costs."""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        self._token_usage["input"] += input_tokens
        self._token_usage["output"] += output_tokens

        # Rough cost estimation (as of 2025, Claude 3.5 Sonnet pricing)
        # Input: $3/MTok, Output: $15/MTok
        input_cost = (input_tokens / 1_000_000) * 3.0
        output_cost = (output_tokens / 1_000_000) * 15.0
        call_cost = input_cost + output_cost

        self._cost_session += call_cost

        logger.info(
            f"Tokens: {input_tokens} in, {output_tokens} out | "
            f"Cost: ${call_cost:.4f} | Session Total: ${self._cost_session:.4f}"
        )

    def _format_tool_response(self, response) -> str:
        """Format a response containing tool calls (Anthropic format)."""
        result = {
            "content": "",
            "tool_calls": []
        }

        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })

        return json.dumps(result)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_with_tool_results(
        self,
        role: str,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None
    ) -> str:
        """
        Continue a conversation with tool results.

        Args:
            role: Agent role
            messages: Full conversation history including tool results (Anthropic format)
            tools: Optional tool definitions

        Returns:
            Response content
        """
        role_config = self.config['model_assignments'].get(role)
        if not role_config:
            raise LLMGatewayError(f"Role {role} not defined in config")

        try:
            kwargs = {
                "model": role_config['model'],
                "max_tokens": role_config.get('max_tokens', 4000),
                "temperature": role_config.get('temperature', 0.7),
                "messages": messages
            }

            if tools:
                kwargs["tools"] = tools

            response = self.client.messages.create(**kwargs)

        except AnthropicError as e:
            logger.error(f"Anthropic API Error: {e}")
            raise LLMGatewayError(f"API call failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise LLMGatewayError(f"Unexpected error: {e}")

        # Track usage
        self._track_usage(response)

        # Handle response
        if response.stop_reason == "tool_use":
            return self._format_tool_response(response)

        # Extract text
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text

        return text_content

    def get_session_cost(self) -> float:
        """Return total cost for this session."""
        return self._cost_session

    def reset_session_cost(self):
        """Reset the session cost counter."""
        self._cost_session = 0.0
