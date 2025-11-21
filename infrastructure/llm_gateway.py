"""
GAADP LLM GATEWAY
Pluggable LLM provider architecture supporting multiple backends.
Automatically selects the best available provider based on environment.
"""
import os
import yaml
import logging
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from infrastructure.llm_providers import (
    LLMProvider,
    ProviderRegistry,
    create_default_registry
)

logger = logging.getLogger("LLM_Gateway")


class LLMGatewayError(Exception):
    """Raised when LLM Gateway encounters errors."""
    pass


class LLMGateway:
    """
    LLM Gateway with pluggable provider architecture.

    Supports multiple LLM backends:
    - Claude Code Agent SDK (headless, no cost)
    - Anthropic API (external, costs money)
    - OpenAI API (future)
    - Local models (future)

    Automatically detects and selects the best available provider.
    Can override with specific provider via config or constructor.
    """

    def __init__(
        self,
        config_path: str = ".blueprint/llm_router.yaml",
        provider_registry: Optional[ProviderRegistry] = None,
        preferred_provider: Optional[str] = None
    ):
        """
        Initialize LLM Gateway.

        Args:
            config_path: Path to configuration file
            provider_registry: Optional custom provider registry (for testing)
            preferred_provider: Optional provider name to prefer (e.g., 'claude_sdk')
        """
        # Load config
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found: {config_path}. Using defaults.")
            self.config = self._default_config()

        # Initialize provider registry
        self.registry = provider_registry or create_default_registry()

        # Select provider
        try:
            self.provider = self.registry.select_provider(preferred=preferred_provider)
            logger.info(
                f"🚀 LLM Gateway initialized with provider: {self.provider.get_name()}"
            )

            # Log cost info
            cost = self.provider.get_cost_per_call()
            if cost == 0.0:
                logger.info("💰 Cost: FREE (no API charges)")
            else:
                logger.info(f"💰 Cost: Variable (API charges apply)")

        except RuntimeError as e:
            raise LLMGatewayError(f"No LLM providers available: {e}")

    def _default_config(self) -> Dict:
        """Default configuration if llm_router.yaml missing."""
        return {
            "model_assignments": {
                "ARCHITECT": {
                    "model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.7,
                    "max_tokens": 4000
                },
                "BUILDER": {
                    "model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.3,
                    "max_tokens": 8000
                },
                "VERIFIER": {
                    "model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.1,
                    "max_tokens": 4000
                },
                "CURATOR": {
                    "model": "claude-3-5-haiku-20241022",
                    "temperature": 0.5,
                    "max_tokens": 2000
                },
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
        Call LLM with optional tool definitions.

        Routes to the active provider (Claude SDK, Anthropic API, etc.)

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
            return self.provider.call(
                system_prompt=system_prompt,
                user_prompt=user_context,
                model_config=role_config,
                tools=tools
            )
        except Exception as e:
            logger.error(f"Provider {self.provider.get_name()} call failed: {e}")
            raise LLMGatewayError(f"LLM call failed: {e}")

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

        try:
            return self.provider.call_multi_turn(
                messages=messages,
                model_config=role_config,
                tools=tools
            )
        except Exception as e:
            logger.error(f"Provider {self.provider.get_name()} multi-turn call failed: {e}")
            raise LLMGatewayError(f"Multi-turn call failed: {e}")

    def get_session_cost(self) -> float:
        """Return total cost for this session."""
        stats = self.provider.get_usage_stats()
        return stats.get('cost', 0.0)

    def reset_session_cost(self):
        """Reset the session cost counter."""
        self.provider.reset_usage_stats()

    def get_backend_info(self) -> Dict:
        """Get information about active provider and usage."""
        stats = self.provider.get_usage_stats()
        available = self.registry.get_available_providers()

        return {
            "active_provider": self.provider.get_name(),
            "available_providers": available,
            "usage_stats": stats,
            "is_free": self.provider.get_cost_per_call() == 0.0
        }

    def switch_provider(self, provider_name: str) -> bool:
        """
        Switch to a different provider at runtime.

        Args:
            provider_name: Name of provider to switch to

        Returns:
            True if switch successful, False otherwise
        """
        try:
            self.provider = self.registry.select_provider(preferred=provider_name)
            logger.info(f"✅ Switched to provider: {provider_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to switch provider: {e}")
            return False
