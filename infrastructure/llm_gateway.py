"""
GAADP LLM GATEWAY
Intelligently routes to Claude Code Agent SDK (headless) when available,
falls back to Anthropic API when running outside Claude Code.
"""
import os
import yaml
import json
import logging
import asyncio
from typing import List, Dict, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential

# Try to import Claude Agent SDK (available when running in Claude Code)
try:
    from claude_agent_sdk import query as claude_query
    from claude_agent_sdk import ClaudeAgentOptions
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    claude_query = None
    ClaudeAgentOptions = None

# Try to import Anthropic API (fallback for external execution)
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
    Intelligent LLM Gateway with dual backend support:

    1. Claude Code Agent SDK (headless, no cost) - when running in Claude Code
    2. Anthropic API (external, costs money) - fallback for non-Claude-Code environments

    Automatically detects which backend to use based on environment.
    """

    def __init__(self, config_path: str = ".blueprint/llm_router.yaml"):
        # Detect if running in Claude Code
        self.in_claude_code = self._detect_claude_code()

        if self.in_claude_code:
            logger.info("✅ Detected Claude Code session - using Agent SDK (headless, no cost)")
            self._init_claude_sdk()
        else:
            logger.info("⚠️  Not in Claude Code - using Anthropic API (costs money)")
            self._init_anthropic_api()

        # Load config
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found: {config_path}. Using defaults.")
            self.config = self._default_config()

        # Cost tracking (only relevant for Anthropic API)
        self._cost_session = 0.0
        self._token_usage = {"input": 0, "output": 0}

    def _detect_claude_code(self) -> bool:
        """Detect if running in a Claude Code session."""
        # CLAUDE_PROJECT_DIR is reliably set when running in Claude Code
        return 'CLAUDE_PROJECT_DIR' in os.environ

    def _init_claude_sdk(self):
        """Initialize Claude Code Agent SDK backend."""
        if not CLAUDE_SDK_AVAILABLE:
            raise LLMGatewayError(
                "Claude Agent SDK not available. Install: pip install claude-agent-sdk"
            )

        self.backend = "claude_sdk"
        self.project_dir = os.getenv("CLAUDE_PROJECT_DIR", os.getcwd())
        logger.info(f"Claude Code project directory: {self.project_dir}")

    def _init_anthropic_api(self):
        """Initialize Anthropic API backend."""
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
        self.backend = "anthropic_api"

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

        Automatically routes to:
        - Claude Agent SDK (if in Claude Code)
        - Anthropic API (if external)

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

        # Route to appropriate backend
        if self.backend == "claude_sdk":
            return self._call_with_agent_sdk(role_config, system_prompt, user_context, tools)
        else:
            return self._call_with_anthropic_api(role_config, system_prompt, user_context, tools)

    def _call_with_agent_sdk(
        self,
        role_config: Dict,
        system_prompt: str,
        user_context: str,
        tools: Optional[List[Dict]]
    ) -> str:
        """Call using Claude Code Agent SDK (headless, no cost)."""
        try:
            # Run the async query in sync context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, create new loop
                import asyncio
                return asyncio.run(self._async_agent_sdk_call(
                    role_config, system_prompt, user_context, tools
                ))
            else:
                return loop.run_until_complete(
                    self._async_agent_sdk_call(role_config, system_prompt, user_context, tools)
                )
        except Exception as e:
            logger.error(f"Claude Agent SDK error: {e}")
            raise LLMGatewayError(f"Agent SDK call failed: {e}")

    async def _async_agent_sdk_call(
        self,
        role_config: Dict,
        system_prompt: str,
        user_context: str,
        tools: Optional[List[Dict]]
    ) -> str:
        """Async implementation of Agent SDK call."""
        # Configure options
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            cwd=self.project_dir,
            permission_mode='acceptEdits',  # Auto-accept for headless
        )

        # Map temperature/max_tokens (Agent SDK uses different names)
        # Note: Agent SDK doesn't expose these directly, but uses model preset

        # Collect all messages from the generator
        all_messages = []
        async for message in claude_query(
            prompt=user_context,
            options=options
        ):
            all_messages.append(message)
            logger.debug(f"Agent SDK message type: {message.type}")

        # Extract the response content
        response_text = ""
        tool_calls = []

        for message in all_messages:
            # Check for assistant response
            if hasattr(message, 'type') and message.type == "assistant":
                if hasattr(message, 'message') and message.message:
                    # Extract text content
                    if hasattr(message.message, 'content'):
                        response_text += message.message.content

            # Check for tool use (if tools were provided)
            if tools and hasattr(message, 'type') and message.type == "tool_use":
                # Format tool calls similar to Anthropic API response
                tool_calls.append({
                    "name": getattr(message, 'name', 'unknown'),
                    "input": getattr(message, 'input', {})
                })

        # If tool calls occurred, return JSON format
        if tool_calls:
            return json.dumps({
                "content": response_text,
                "tool_calls": tool_calls
            })

        return response_text

    def _call_with_anthropic_api(
        self,
        role_config: Dict,
        system_prompt: str,
        user_context: str,
        tools: Optional[List[Dict]]
    ) -> str:
        """Call using Anthropic API (external, costs money)."""
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
        """Track token usage and estimate costs (Anthropic API only)."""
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
            messages: Full conversation history including tool results
            tools: Optional tool definitions

        Returns:
            Response content
        """
        role_config = self.config['model_assignments'].get(role)
        if not role_config:
            raise LLMGatewayError(f"Role {role} not defined in config")

        # Route to appropriate backend
        if self.backend == "claude_sdk":
            # For Agent SDK, we'd need to reconstruct the conversation
            # For now, just use the last user message
            # TODO: Implement proper multi-turn for Agent SDK
            last_user_msg = messages[-1].get("content", "") if messages else ""
            system_prompt = "Continue the conversation."
            return self._call_with_agent_sdk(role_config, system_prompt, last_user_msg, tools)
        else:
            return self._call_with_anthropic_api_multi_turn(role_config, messages, tools)

    def _call_with_anthropic_api_multi_turn(
        self,
        role_config: Dict,
        messages: List[Dict],
        tools: Optional[List[Dict]]
    ) -> str:
        """Multi-turn conversation using Anthropic API."""
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
        """Return total cost for this session (Anthropic API only, $0 for Agent SDK)."""
        if self.backend == "claude_sdk":
            return 0.0  # No cost when using Claude Code
        return self._cost_session

    def reset_session_cost(self):
        """Reset the session cost counter."""
        self._cost_session = 0.0

    def get_backend_info(self) -> Dict:
        """Get information about which backend is being used."""
        return {
            "backend": self.backend,
            "in_claude_code": self.in_claude_code,
            "cost_tracking": self.backend == "anthropic_api",
            "project_dir": getattr(self, 'project_dir', None)
        }
