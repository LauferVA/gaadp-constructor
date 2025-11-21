"""
GAADP LLM GATEWAY
Supports standard completion and tool-augmented calls.
"""
import yaml
import json
import logging
from typing import List, Dict, Optional, Any
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("LLM_Gateway")


class LLMGateway:
    def __init__(self, config_path: str = ".blueprint/llm_router.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self._cost_session = 0.0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_model(
        self,
        role: str,
        system_prompt: str,
        user_context: str,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """
        Call the LLM with optional tool definitions.

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
            raise ValueError(f"Role {role} not defined in llm_router.yaml")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context}
        ]

        try:
            kwargs = {
                "model": role_config['model'],
                "messages": messages,
                "temperature": role_config['temperature'],
                "max_tokens": role_config.get('max_tokens', 4000)
            }

            # Add tools if provided
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = completion(**kwargs)

        except Exception as e:
            logger.error(f"API Failure: {e}")
            raise e

        # Track costs
        cost = getattr(response, '_hidden_params', {}).get("response_cost", 0.0)
        self._cost_session += cost
        logger.info(f"Call Cost: ${cost:.4f} | Total: ${self._cost_session:.4f}")

        # Handle response
        message = response.choices[0].message

        # Check for tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            return self._format_tool_response(message)

        return message.content

    def _format_tool_response(self, message) -> str:
        """Format a response containing tool calls."""
        result = {
            "content": message.content,
            "tool_calls": []
        }

        for tool_call in message.tool_calls:
            result["tool_calls"].append({
                "id": tool_call.id,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
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
            messages: Full conversation history including tool results
            tools: Optional tool definitions

        Returns:
            Response content
        """
        role_config = self.config['model_assignments'].get(role)
        if not role_config:
            raise ValueError(f"Role {role} not defined in llm_router.yaml")

        try:
            kwargs = {
                "model": role_config['model'],
                "messages": messages,
                "temperature": role_config['temperature'],
                "max_tokens": role_config.get('max_tokens', 4000)
            }

            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = completion(**kwargs)

        except Exception as e:
            logger.error(f"API Failure: {e}")
            raise e

        cost = getattr(response, '_hidden_params', {}).get("response_cost", 0.0)
        self._cost_session += cost

        message = response.choices[0].message

        if hasattr(message, 'tool_calls') and message.tool_calls:
            return self._format_tool_response(message)

        return message.content

    def get_session_cost(self) -> float:
        """Return total cost for this session."""
        return self._cost_session

    def reset_session_cost(self):
        """Reset the session cost counter."""
        self._cost_session = 0.0
