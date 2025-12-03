"""
LLM PROVIDER ABSTRACTIONS
Base classes and concrete implementations for different LLM backends.
Supports: Claude Code Agent SDK, Anthropic API, OpenAI, local models, etc.
"""
import os
import json
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

logger = logging.getLogger("LLM_Providers")


# =============================================================================
# BASE PROVIDER INTERFACE
# =============================================================================

class LLMProvider(ABC):
    """
    Abstract base class for LLM provider implementations.
    Each provider must implement these methods.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider can be used in the current environment."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the provider name (e.g., 'claude_sdk', 'anthropic_api')."""
        pass

    @abstractmethod
    def get_cost_per_call(self) -> float:
        """Return the cost for this provider (0.0 if free)."""
        pass

    @abstractmethod
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_config: Dict,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """
        Make an LLM call.

        Args:
            system_prompt: System message
            user_prompt: User message
            model_config: Model configuration (temperature, max_tokens, etc.)
            tools: Optional tool definitions

        Returns:
            Response string (or JSON if tools used)
        """
        pass

    @abstractmethod
    def call_multi_turn(
        self,
        messages: List[Dict],
        model_config: Dict,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """
        Make a multi-turn conversation call.

        Args:
            messages: Conversation history
            model_config: Model configuration
            tools: Optional tool definitions

        Returns:
            Response string
        """
        pass

    @abstractmethod
    def get_usage_stats(self) -> Dict:
        """Return usage statistics (tokens, cost, etc.)."""
        pass

    @abstractmethod
    def reset_usage_stats(self):
        """Reset usage statistics."""
        pass


# =============================================================================
# CLAUDE CODE AGENT SDK PROVIDER
# =============================================================================

try:
    from claude_agent_sdk import query as claude_query
    from claude_agent_sdk import ClaudeAgentOptions
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    claude_query = None
    ClaudeAgentOptions = None


class ClaudeSDKProvider(LLMProvider):
    """
    Provider for Claude Code Agent SDK (headless, no cost).
    Used when code is running within a Claude Code session.
    """

    def __init__(self):
        self.project_dir = os.getenv("CLAUDE_PROJECT_DIR", os.getcwd())
        self._call_count = 0

    def is_available(self) -> bool:
        """Check if Claude SDK is available and we're in a Claude Code session."""
        return CLAUDE_SDK_AVAILABLE and 'CLAUDE_PROJECT_DIR' in os.environ

    def get_name(self) -> str:
        return "claude_sdk"

    def get_cost_per_call(self) -> float:
        return 0.0  # Free when using Claude Code

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_config: Dict,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """Make a call using Claude Agent SDK."""
        # Handle async in sync context
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - use nest_asyncio or create task
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(
                self._async_call(system_prompt, user_prompt, model_config, tools)
            )
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(self._async_call(system_prompt, user_prompt, model_config, tools))

    async def _async_call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_config: Dict,
        tools: Optional[List[Dict]]
    ) -> str:
        """Async implementation of SDK call."""
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            cwd=self.project_dir,
            permission_mode='acceptEdits',
        )

        # Collect messages
        all_messages = []
        async for message in claude_query(prompt=user_prompt, options=options):
            all_messages.append(message)
            logger.debug(f"SDK message type: {message.type}")

        # Extract response
        response_text = ""
        tool_calls = []

        for message in all_messages:
            if hasattr(message, 'type') and message.type == "assistant":
                if hasattr(message, 'message') and message.message:
                    if hasattr(message.message, 'content'):
                        response_text += message.message.content

            if tools and hasattr(message, 'type') and message.type == "tool_use":
                tool_calls.append({
                    "name": getattr(message, 'name', 'unknown'),
                    "input": getattr(message, 'input', {})
                })

        self._call_count += 1

        if tool_calls:
            return json.dumps({
                "content": response_text,
                "tool_calls": tool_calls
            })

        return response_text

    def call_multi_turn(
        self,
        messages: List[Dict],
        model_config: Dict,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """Multi-turn conversation (simplified for SDK)."""
        # For now, use the last user message
        # TODO: Implement proper conversation history
        last_user_msg = messages[-1].get("content", "") if messages else ""
        system_prompt = "Continue the conversation."
        return self.call(system_prompt, last_user_msg, model_config, tools)

    def get_usage_stats(self) -> Dict:
        return {
            "provider": "claude_sdk",
            "calls": self._call_count,
            "cost": 0.0,
            "tokens_input": 0,
            "tokens_output": 0
        }

    def reset_usage_stats(self):
        self._call_count = 0


# =============================================================================
# ANTHROPIC API PROVIDER
# =============================================================================

try:
    from anthropic import Anthropic, AnthropicError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None
    AnthropicError = Exception


class AnthropicAPIProvider(LLMProvider):
    """
    Provider for Anthropic API (external, costs money).
    Fallback for non-Claude-Code environments.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if self.api_key:
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None

        self._token_usage = {"input": 0, "output": 0}
        self._cost_session = 0.0

    def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        return ANTHROPIC_AVAILABLE and self.api_key is not None

    def get_name(self) -> str:
        return "anthropic_api"

    def get_cost_per_call(self) -> float:
        # Variable cost, return session average
        return self._cost_session

    def _convert_tools_to_anthropic(self, tools: List[Dict]) -> List[Dict]:
        """
        Convert OpenAI-formatted tools to Anthropic format.

        OpenAI format:
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": { ... JSON Schema ... }
                }
            }

        Anthropic format:
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": { ... JSON Schema ... }
            }

        Args:
            tools: List of tool definitions (OpenAI or Anthropic format)

        Returns:
            List of tools in Anthropic format
        """
        converted = []

        for tool in tools:
            # Check if it's OpenAI format (has 'type': 'function' and 'function' key)
            if tool.get('type') == 'function' and 'function' in tool:
                func = tool['function']
                anthropic_tool = {
                    "name": func.get('name', 'unknown'),
                    "description": func.get('description', ''),
                    "input_schema": func.get('parameters', {"type": "object", "properties": {}})
                }
                converted.append(anthropic_tool)
                logger.debug(f"Converted OpenAI tool '{anthropic_tool['name']}' to Anthropic format")
            else:
                # Already in Anthropic format or unknown - pass through
                converted.append(tool)

        return converted

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_config: Dict,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """
        Make a call using Anthropic API.

        Supports forced tool_choice for structured output:
            model_config["tool_choice"] = {"type": "tool", "name": "submit_X"}
        """
        kwargs = {
            "model": model_config.get('model', 'claude-3-5-sonnet-20241022'),
            "max_tokens": model_config.get('max_tokens', 4000),
            "temperature": model_config.get('temperature', 0.7),
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}]
        }

        if tools:
            # Convert OpenAI-formatted tools to Anthropic format
            kwargs["tools"] = self._convert_tools_to_anthropic(tools)

            # Check for forced tool_choice (for protocol-based output)
            tool_choice = model_config.get("tool_choice")
            if tool_choice:
                # Anthropic format: {"type": "tool", "name": "tool_name"}
                # or {"type": "any"} or {"type": "auto"}
                kwargs["tool_choice"] = tool_choice
                logger.debug(f"Forced tool_choice: {tool_choice}")

        try:
            response = self.client.messages.create(**kwargs)
        except AnthropicError as e:
            logger.error(f"Anthropic API error: {e}")
            raise

        # Track usage
        self._track_usage(response)

        # Handle tool use (either voluntary or forced)
        if response.stop_reason == "tool_use":
            return self._format_tool_response(response)

        # Extract text (shouldn't happen with forced tool_choice, but handle it)
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text

        return text_content

    def call_multi_turn(
        self,
        messages: List[Dict],
        model_config: Dict,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """
        Multi-turn conversation.

        Supports forced tool_choice for structured output:
            model_config["tool_choice"] = {"type": "tool", "name": "submit_X"}
        """
        kwargs = {
            "model": model_config.get('model', 'claude-3-5-sonnet-20241022'),
            "max_tokens": model_config.get('max_tokens', 4000),
            "temperature": model_config.get('temperature', 0.7),
            "messages": messages
        }

        if tools:
            # Convert OpenAI-formatted tools to Anthropic format
            kwargs["tools"] = self._convert_tools_to_anthropic(tools)

            # Check for forced tool_choice (for protocol-based output)
            tool_choice = model_config.get("tool_choice")
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
                logger.debug(f"Multi-turn forced tool_choice: {tool_choice}")

        try:
            response = self.client.messages.create(**kwargs)
        except AnthropicError as e:
            logger.error(f"Anthropic API error: {e}")
            raise

        self._track_usage(response)

        if response.stop_reason == "tool_use":
            return self._format_tool_response(response)

        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text

        return text_content

    def _track_usage(self, response):
        """Track token usage and costs."""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        self._token_usage["input"] += input_tokens
        self._token_usage["output"] += output_tokens

        # Claude 3.5 Sonnet pricing (2025)
        input_cost = (input_tokens / 1_000_000) * 3.0
        output_cost = (output_tokens / 1_000_000) * 15.0
        call_cost = input_cost + output_cost

        self._cost_session += call_cost

        logger.info(
            f"Tokens: {input_tokens} in, {output_tokens} out | "
            f"Cost: ${call_cost:.4f} | Session: ${self._cost_session:.4f}"
        )

    def _format_tool_response(self, response) -> str:
        """Format tool use response."""
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

    def get_usage_stats(self) -> Dict:
        return {
            "provider": "anthropic_api",
            "tokens_input": self._token_usage["input"],
            "tokens_output": self._token_usage["output"],
            "cost": self._cost_session
        }

    def reset_usage_stats(self):
        self._token_usage = {"input": 0, "output": 0}
        self._cost_session = 0.0


# =============================================================================
# OPENAI API PROVIDER (Imported from separate module)
# =============================================================================

# Import the real OpenAI provider implementation
try:
    from infrastructure.openai_provider import OpenAIProvider
except ImportError:
    # Fallback for relative imports
    try:
        from .openai_provider import OpenAIProvider
    except ImportError:
        # Create a stub if the file doesn't exist
        class OpenAIProvider(LLMProvider):
            """Fallback stub if openai_provider.py is missing."""
            def __init__(self, *args, **kwargs): pass
            def is_available(self) -> bool: return False
            def get_name(self) -> str: return "openai_api"
            def get_cost_per_call(self) -> float: return 0.0
            def call(self, *args, **kwargs) -> str: raise NotImplementedError()
            def call_multi_turn(self, *args, **kwargs) -> str: raise NotImplementedError()
            def get_usage_stats(self) -> Dict: return {}
            def reset_usage_stats(self): pass


# =============================================================================
# LOCAL MODEL PROVIDER (STUB - For Future Implementation)
# =============================================================================

class LocalModelProvider(LLMProvider):
    """
    Provider for local models (Ollama, llama.cpp, etc.).
    Stub implementation - can be filled in later.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def is_available(self) -> bool:
        return False  # Not implemented yet

    def get_name(self) -> str:
        return "local_model"

    def get_cost_per_call(self) -> float:
        return 0.0  # Free for local models

    def call(self, system_prompt: str, user_prompt: str, model_config: Dict, tools: Optional[List[Dict]] = None) -> str:
        raise NotImplementedError("Local model provider not yet implemented")

    def call_multi_turn(self, messages: List[Dict], model_config: Dict, tools: Optional[List[Dict]] = None) -> str:
        raise NotImplementedError("Local model provider not yet implemented")

    def get_usage_stats(self) -> Dict:
        return {"provider": "local_model", "status": "not_implemented"}

    def reset_usage_stats(self):
        pass


# =============================================================================
# MANUAL PROVIDER (Human-in-the-Loop Testing)
# =============================================================================

import sys

class ManualProvider(LLMProvider):
    """
    Provider for manual/interactive testing.
    Prints prompts to stdout and reads responses from stdin.
    Allows a human to act as the LLM for debugging agent logic.

    Usage:
        export LLM_PROVIDER=manual
        python main.py
    """

    def __init__(self):
        self._call_count = 0
        self._current_role = "UNKNOWN"

    def is_available(self) -> bool:
        """Manual provider is always available when explicitly requested."""
        return os.getenv("LLM_PROVIDER", "").lower() == "manual"

    def get_name(self) -> str:
        return "manual"

    def get_cost_per_call(self) -> float:
        return 0.0  # Free (human labor not counted)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_config: Dict,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """
        Display the prompt to the human and capture their response.
        """
        self._call_count += 1

        # Print clear separator and context
        print("\n" + "=" * 80)
        print(f"  MANUAL LLM REQUEST #{self._call_count}")
        print(f"  Role: {model_config.get('role', 'UNKNOWN')}")
        print(f"  Model Config: temp={model_config.get('temperature', 'N/A')}, max_tokens={model_config.get('max_tokens', 'N/A')}")
        print("=" * 80)

        print("\n┌─── SYSTEM PROMPT ───────────────────────────────────────────────────────────┐")
        # Truncate very long system prompts for readability
        if len(system_prompt) > 2000:
            print(system_prompt[:2000])
            print(f"\n... [TRUNCATED - {len(system_prompt)} chars total] ...")
        else:
            print(system_prompt)
        print("└─────────────────────────────────────────────────────────────────────────────┘")

        print("\n┌─── USER PROMPT / CONTEXT ───────────────────────────────────────────────────┐")
        # Show full user prompt (may include tool results from ReAct loop)
        print(user_prompt)
        print("└─────────────────────────────────────────────────────────────────────────────┘")

        if tools:
            print("\n┌─── AVAILABLE TOOLS ─────────────────────────────────────────────────────────┐")
            for tool in tools:
                func = tool.get('function', tool)
                tool_name = func.get('name', 'unknown')
                tool_desc = func.get('description', 'No description')[:60]
                print(f"  • {tool_name}: {tool_desc}")
            print("└─────────────────────────────────────────────────────────────────────────────┘")
            print("\n💡 TIP: To call a tool, respond with JSON like:")
            print('   {"tool_calls": [{"name": "read_file", "input": {"path": "some/file.py"}}]}')

        print("\n" + "=" * 80)
        print("  PASTE YOUR RESPONSE BELOW")
        print("  (Press Ctrl+D on empty line when finished, or Ctrl+C to abort)")
        print("=" * 80)
        print()

        # Capture multi-line input
        try:
            response = sys.stdin.read()
        except KeyboardInterrupt:
            print("\n\n⚠️  Input cancelled by user. Returning empty response.")
            response = '{"error": "User cancelled input"}'

        print("\n" + "─" * 80)
        print(f"  ✅ RESPONSE CAPTURED ({len(response)} chars)")
        print("─" * 80 + "\n")

        return response.strip()

    def call_multi_turn(
        self,
        messages: List[Dict],
        model_config: Dict,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """Multi-turn conversation - show full history."""
        self._call_count += 1

        print("\n" + "=" * 80)
        print(f"  MANUAL LLM REQUEST #{self._call_count} (Multi-Turn)")
        print("=" * 80)

        print("\n┌─── CONVERSATION HISTORY ────────────────────────────────────────────────────┐")
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            if len(content) > 500:
                content = content[:500] + f"... [{len(content)} chars total]"
            print(f"\n[{i+1}] {role}:")
            print(content)
        print("\n└─────────────────────────────────────────────────────────────────────────────┘")

        if tools:
            print("\n┌─── AVAILABLE TOOLS ─────────────────────────────────────────────────────────┐")
            for tool in tools:
                func = tool.get('function', tool)
                tool_name = func.get('name', 'unknown')
                tool_desc = func.get('description', 'No description')[:60]
                print(f"  • {tool_name}: {tool_desc}")
            print("└─────────────────────────────────────────────────────────────────────────────┘")

        print("\n" + "=" * 80)
        print("  PASTE YOUR RESPONSE BELOW")
        print("  (Press Ctrl+D on empty line when finished)")
        print("=" * 80)
        print()

        try:
            response = sys.stdin.read()
        except KeyboardInterrupt:
            print("\n\n⚠️  Input cancelled by user.")
            response = '{"error": "User cancelled input"}'

        print("\n" + "─" * 80)
        print(f"  ✅ RESPONSE CAPTURED ({len(response)} chars)")
        print("─" * 80 + "\n")

        return response.strip()

    def get_usage_stats(self) -> Dict:
        return {
            "provider": "manual",
            "calls": self._call_count,
            "cost": 0.0,
            "note": "Human-in-the-loop testing mode"
        }

    def reset_usage_stats(self):
        self._call_count = 0


# =============================================================================
# PROVIDER REGISTRY
# =============================================================================

class ProviderRegistry:
    """
    Registry for LLM providers.
    Automatically detects and selects the best available provider.
    """

    def __init__(self):
        self._providers: List[LLMProvider] = []
        self._active_provider: Optional[LLMProvider] = None

    def register(self, provider: LLMProvider, priority: int = 0):
        """Register a provider with priority (higher = preferred)."""
        self._providers.append((priority, provider))
        self._providers.sort(key=lambda x: x[0], reverse=True)

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [p.get_name() for _, p in self._providers if p.is_available()]

    def select_provider(self, preferred: Optional[str] = None) -> LLMProvider:
        """
        Select the best available provider.

        Args:
            preferred: Optional preferred provider name

        Returns:
            Selected provider instance

        Raises:
            RuntimeError: If no providers available
        """
        # Try preferred provider first
        if preferred:
            for _, provider in self._providers:
                if provider.get_name() == preferred and provider.is_available():
                    self._active_provider = provider
                    logger.info(f"✅ Selected preferred provider: {provider.get_name()}")
                    return provider

        # Fall back to highest priority available provider
        for _, provider in self._providers:
            if provider.is_available():
                self._active_provider = provider
                logger.info(f"✅ Auto-selected provider: {provider.get_name()}")
                return provider

        raise RuntimeError("No LLM providers available")

    def get_active_provider(self) -> Optional[LLMProvider]:
        """Get the currently active provider."""
        return self._active_provider


# =============================================================================
# DEFAULT REGISTRY FACTORY
# =============================================================================

def create_default_registry() -> ProviderRegistry:
    """
    Create a registry with all available providers.
    Priority order:
    1. Manual (if LLM_PROVIDER=manual) - Human-in-the-loop testing
    2. Claude SDK (if in Claude Code) - Free
    3. Anthropic API (if API key available) - Paid
    4. OpenAI (not yet implemented)
    5. Local models (not yet implemented)
    """
    registry = ProviderRegistry()

    # Register providers in priority order
    # ManualProvider has highest priority when LLM_PROVIDER=manual is set
    registry.register(ManualProvider(), priority=200)  # Highest when enabled
    registry.register(ClaudeSDKProvider(), priority=100)  # Free in Claude Code
    registry.register(AnthropicAPIProvider(), priority=50)  # Fallback (paid)
    registry.register(OpenAIProvider(), priority=25)  # Future
    registry.register(LocalModelProvider(), priority=10)  # Future

    return registry
