import os
import json
import logging
from typing import Dict, List, Optional

import openai  # Using v1.0+ OpenAI library

logger = logging.getLogger("LLM_Providers")


class OpenAIProvider:
    """
    Provider for OpenAI API (external, costs money).
    Supports GPT-4o and other OpenAI models.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI provider with optional API key.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
            self.client = openai
        else:
            self.client = None

        self._token_usage = {"input": 0, "output": 0}
        self._cost_session = 0.0

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        return self.api_key is not None

    def get_name(self) -> str:
        return "openai_api"

    def get_cost_per_call(self) -> float:
        """Return the average cost per call for the current session."""
        return self._cost_session

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_config: Dict,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """
        Make a call using OpenAI API.

        Args:
            system_prompt: System message
            user_prompt: User message
            model_config: Model configuration
            tools: Optional tool definitions

        Returns:
            Response string or JSON for tool calls
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        kwargs = {
            "model": model_config.get('model', 'gpt-4o'),
            "messages": messages,
            "max_tokens": model_config.get('max_tokens', 4000),
            "temperature": model_config.get('temperature', 0.7)
        }

        # Add tools if provided
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

        # Track usage
        self._track_usage(response)

        # Check for tool calls
        if response.choices[0].finish_reason == "tool_calls":
            return self._format_tool_response(response)

        # Extract text response
        return response.choices[0].message.content or ""

    def call_multi_turn(
        self,
        messages: List[Dict],
        model_config: Dict,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """
        Make a multi-turn conversation call using OpenAI API.

        Args:
            messages: Conversation history
            model_config: Model configuration
            tools: Optional tool definitions

        Returns:
            Response string or JSON for tool calls
        """
        kwargs = {
            "model": model_config.get('model', 'gpt-4o'),
            "messages": messages,
            "max_tokens": model_config.get('max_tokens', 4000),
            "temperature": model_config.get('temperature', 0.7)
        }

        # Add tools if provided
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

        # Track usage
        self._track_usage(response)

        # Check for tool calls
        if response.choices[0].finish_reason == "tool_calls":
            return self._format_tool_response(response)

        # Extract text response
        return response.choices[0].message.content or ""

    def _track_usage(self, response):
        """
        Track token usage and calculate costs for the session.
        Uses GPT-4o pricing as of 2024.
        """
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        self._token_usage["input"] += input_tokens
        self._token_usage["output"] += output_tokens

        # GPT-4o pricing (as of 2024)
        # $5.00 per 1M input tokens, $15.00 per 1M output tokens
        input_cost = (input_tokens / 1_000_000) * 5.0
        output_cost = (output_tokens / 1_000_000) * 15.0
        call_cost = input_cost + output_cost

        self._cost_session += call_cost

        logger.info(
            f"Tokens: {input_tokens} in, {output_tokens} out | "
            f"Cost: ${call_cost:.4f} | Session: ${self._cost_session:.4f}"
        )

    def _format_tool_response(self, response) -> str:
        """
        Format tool use response to match the expected JSON structure.

        Args:
            response: OpenAI API response object

        Returns:
            JSON-formatted string with content and tool calls
        """
        result = {
            "content": response.choices[0].message.content or "",
            "tool_calls": []
        }

        # Extract tool calls
        for tool_call in response.choices[0].message.tool_calls or []:
            result["tool_calls"].append({
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": json.loads(tool_call.function.arguments)
            })

        return json.dumps(result)

    def get_usage_stats(self) -> Dict:
        """
        Get current usage statistics for the session.

        Returns:
            Dictionary with usage and cost information
        """
        return {
            "provider": "openai_api",
            "tokens_input": self._token_usage["input"],
            "tokens_output": self._token_usage["output"],
            "cost": self._cost_session
        }

    def reset_usage_stats(self):
        """
        Reset usage statistics for a new session.
        """
        self._token_usage = {"input": 0, "output": 0}
        self._cost_session = 0.0