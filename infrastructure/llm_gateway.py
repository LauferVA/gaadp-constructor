"""
GAADP LLM GATEWAY
Pluggable LLM provider architecture supporting multiple backends.
Automatically selects the best available provider based on environment.
Includes STRICT Model Router that forces all traffic to available models.
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


# =============================================================================
# STRICT MODEL ROUTER - FORCES ALL TRAFFIC TO WORKING MODELS
# =============================================================================
# This router intercepts ALL model requests and redirects them to models
# that are actually available on the account. No exceptions.

MODEL_MAP = {
    "anthropic_api": {
        # FORCE ALL ANTHROPIC TRAFFIC TO HAIKU (only available model)
        "heavy": "claude-3-5-haiku-20241022",
        "standard": "claude-3-5-haiku-20241022",
        "legacy": "claude-3-5-haiku-20241022",
    },
    "openai_api": {
        "heavy": "gpt-4o",
        "standard": "gpt-4o-mini",
        "legacy": "gpt-4",
    },
    "claude_sdk": {
        "heavy": "default",
        "standard": "default",
        "legacy": "default",
    },
    "manual": {
        "heavy": "manual",
        "standard": "manual",
        "legacy": "manual",
    },
    "local_model": {
        "heavy": "llama3:70b",
        "standard": "llama3:8b",
        "legacy": "llama2:13b",
    }
}

# The ONE TRUE MODEL for Anthropic (change this when you get access to others)
ANTHROPIC_FORCED_MODEL = "claude-3-5-haiku-20241022"

# Role-to-tier mapping: which model tier each agent role should use
ROLE_TIER_MAP = {
    "ARCHITECT": "heavy",
    "BUILDER": "heavy",
    "VERIFIER": "standard",
    "CURATOR": "standard",
    "SENTINEL": "standard",
    "TREASURER": "standard",
    "SOCRATES": "heavy",
    "LIBRARIAN": "standard",
}


class LLMGatewayError(Exception):
    """Raised when LLM Gateway encounters errors."""
    pass


class LLMGateway:
    """
    LLM Gateway with pluggable provider architecture and STRICT model routing.

    The Model Router intercepts ALL requests and forces them to use available models.
    For Anthropic, this means ALL requests go to claude-3-5-haiku-20241022.
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

        Environment Variables:
            LLM_PROVIDER: Force a specific provider (e.g., 'manual', 'anthropic_api')
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

        # Check for environment variable override
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider:
            logger.info(f"🔧 LLM_PROVIDER environment variable set: '{env_provider}'")
            preferred_provider = env_provider.lower()

        # Select provider
        try:
            self.provider = self.registry.select_provider(preferred=preferred_provider)
            logger.info(
                f"🚀 LLM Gateway initialized with provider: {self.provider.get_name()}"
            )

            # Log provider-specific info
            provider_name = self.provider.get_name()
            if provider_name == "manual":
                logger.info("🧑‍💻 MANUAL MODE: You will act as the LLM brain")
            elif provider_name == "anthropic_api":
                logger.info(f"🔒 STRICT ROUTING: All requests -> {ANTHROPIC_FORCED_MODEL}")
            elif self.provider.get_cost_per_call() == 0.0:
                logger.info("💰 Cost: FREE (no API charges)")
            else:
                logger.info("💰 Cost: Variable (API charges apply)")

        except RuntimeError as e:
            raise LLMGatewayError(f"No LLM providers available: {e}")

    def _default_config(self) -> Dict:
        """Default configuration if llm_router.yaml missing."""
        return {
            "model_assignments": {
                "ARCHITECT": {
                    "model": "heavy",
                    "temperature": 0.7,
                    "max_tokens": 4000
                },
                "BUILDER": {
                    "model": "heavy",
                    "temperature": 0.3,
                    "max_tokens": 8000
                },
                "VERIFIER": {
                    "model": "standard",
                    "temperature": 0.1,
                    "max_tokens": 4000
                },
                "CURATOR": {
                    "model": "standard",
                    "temperature": 0.5,
                    "max_tokens": 2000
                },
                "SOCRATES": {
                    "model": "heavy",
                    "temperature": 0.7,
                    "max_tokens": 4000
                },
            }
        }

    def _force_model_override(self, role: str, model_config: Dict) -> Dict:
        """
        STRICT MODEL ROUTER: Intercept and override ALL model requests.

        This is the nuclear option - it doesn't care what the config says,
        it FORCES the model to whatever is actually available.

        Args:
            role: Agent role name
            model_config: Original model configuration

        Returns:
            Modified config with forced model
        """
        provider_name = self.provider.get_name()
        config = model_config.copy() if model_config else {}

        # Store the original model for logging
        old_model = config.get("model", "default")

        # ---------------------------------------------------------
        # MODEL ROUTER: INTERCEPT AND OVERRIDE
        # ---------------------------------------------------------
        if provider_name == "anthropic_api":
            # FORCE ALL ANTHROPIC REQUESTS TO HAIKU
            config["model"] = ANTHROPIC_FORCED_MODEL
            if old_model != ANTHROPIC_FORCED_MODEL:
                logger.info(f"🔄 ROUTER: Swapped {old_model} -> {ANTHROPIC_FORCED_MODEL} (anthropic_api)")

        elif provider_name in MODEL_MAP:
            # For other providers, use tier-based routing
            target_tier = ROLE_TIER_MAP.get(role, "standard")
            overridden_model = MODEL_MAP[provider_name].get(target_tier, old_model)
            config["model"] = overridden_model
            if old_model != overridden_model:
                logger.info(f"🔄 ROUTER: Swapped {old_model} -> {overridden_model} ({provider_name})")

        # Add role to config for ManualProvider logging
        config["role"] = role

        return config

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

        Routes to the active provider with STRICT model enforcement.
        For Anthropic, ALL requests are forced to claude-3-5-haiku-20241022.

        Args:
            role: Agent role (maps to model config)
            system_prompt: System message
            user_context: User message
            tools: Optional list of tool definitions for function calling

        Returns:
            Response content as string (or JSON if tool calls present)
        """
        # Get role config from file or use defaults
        role_config = self.config.get('model_assignments', {}).get(role)
        if not role_config:
            tier = ROLE_TIER_MAP.get(role, 'standard')
            role_config = {
                'model': tier,
                'temperature': 0.7 if tier == 'heavy' else 0.5,
                'max_tokens': 4000
            }
            logger.warning(f"Role '{role}' not in config, using tier '{tier}'")

        # ---------------------------------------------------------
        # STRICT MODEL ROUTER: FORCE OVERRIDE BEFORE CALLING PROVIDER
        # ---------------------------------------------------------
        forced_config = self._force_model_override(role, role_config)

        try:
            return self.provider.call(
                system_prompt=system_prompt,
                user_prompt=user_context,
                model_config=forced_config,
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
        role_config = self.config.get('model_assignments', {}).get(role)
        if not role_config:
            tier = ROLE_TIER_MAP.get(role, 'standard')
            role_config = {
                'model': tier,
                'temperature': 0.7 if tier == 'heavy' else 0.5,
                'max_tokens': 4000
            }

        # ---------------------------------------------------------
        # STRICT MODEL ROUTER: FORCE OVERRIDE BEFORE CALLING PROVIDER
        # ---------------------------------------------------------
        forced_config = self._force_model_override(role, role_config)

        try:
            return self.provider.call_multi_turn(
                messages=messages,
                model_config=forced_config,
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
            "is_free": self.provider.get_cost_per_call() == 0.0,
            "forced_model": ANTHROPIC_FORCED_MODEL if self.provider.get_name() == "anthropic_api" else None
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
