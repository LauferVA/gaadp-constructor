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
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create new event loop if already in async context
            return asyncio.run(self._async_call(system_prompt, user_prompt, model_config, tools))
        else:
            return loop.run_until_complete(
                self._async_call(system_prompt, user_prompt, model_config, tools)
            )

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

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_config: Dict,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """Make a call using Anthropic API."""
        kwargs = {
            "model": model_config.get('model', 'claude-3-5-sonnet-20241022'),
            "max_tokens": model_config.get('max_tokens', 4000),
            "temperature": model_config.get('temperature', 0.7),
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}]
        }

        if tools:
            kwargs["tools"] = tools

        try:
            response = self.client.messages.create(**kwargs)
        except AnthropicError as e:
            logger.error(f"Anthropic API error: {e}")
            raise

        # Track usage
        self._track_usage(response)

        # Handle tool use
        if response.stop_reason == "tool_use":
            return self._format_tool_response(response)

        # Extract text
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
        """Multi-turn conversation."""
        kwargs = {
            "model": model_config.get('model', 'claude-3-5-sonnet-20241022'),
            "max_tokens": model_config.get('max_tokens', 4000),
            "temperature": model_config.get('temperature', 0.7),
            "messages": messages
        }

        if tools:
            kwargs["tools"] = tools

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
# OPENAI API PROVIDER (STUB - For Future Implementation)
# =============================================================================

class OpenAIProvider(LLMProvider):
    """
    Provider for OpenAI API (GPT-4, etc.).
    Stub implementation - can be filled in later.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def is_available(self) -> bool:
        return False  # Not implemented yet

    def get_name(self) -> str:
        return "openai_api"

    def get_cost_per_call(self) -> float:
        return 0.0

    def call(self, system_prompt: str, user_prompt: str, model_config: Dict, tools: Optional[List[Dict]] = None) -> str:
        raise NotImplementedError("OpenAI provider not yet implemented")

    def call_multi_turn(self, messages: List[Dict], model_config: Dict, tools: Optional[List[Dict]] = None) -> str:
        raise NotImplementedError("OpenAI provider not yet implemented")

    def get_usage_stats(self) -> Dict:
        return {"provider": "openai_api", "status": "not_implemented"}

    def reset_usage_stats(self):
        pass


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
    1. Claude SDK (if in Claude Code) - Free
    2. Anthropic API (if API key available) - Paid
    3. OpenAI (not yet implemented)
    4. Local models (not yet implemented)
    """
    registry = ProviderRegistry()

    # Register providers in priority order
    registry.register(ClaudeSDKProvider(), priority=100)  # Highest priority (free)
    registry.register(AnthropicAPIProvider(), priority=50)  # Fallback (paid)
    registry.register(OpenAIProvider(), priority=25)  # Future
    registry.register(LocalModelProvider(), priority=10)  # Future

    return registry
