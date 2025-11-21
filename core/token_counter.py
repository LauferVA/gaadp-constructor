"""
TOKEN COUNTER
Real token counting using tiktoken for accurate context management.
"""
import logging
from typing import Optional, Dict
from functools import lru_cache

logger = logging.getLogger("TokenCounter")

# Try to import tiktoken, fall back to approximation if not available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not installed, using character-based approximation")


# Model to encoding mapping
MODEL_ENCODINGS = {
    # OpenAI models
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-3.5-turbo": "cl100k_base",
    # Anthropic models (use cl100k as approximation)
    "claude-3-opus": "cl100k_base",
    "claude-3-sonnet": "cl100k_base",
    "claude-3-haiku": "cl100k_base",
    "claude-3.5-sonnet": "cl100k_base",
    # Default
    "default": "cl100k_base"
}

# Model context limits (approximate)
MODEL_CONTEXT_LIMITS = {
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 16385,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3.5-sonnet": 200000,
    "default": 8192
}


@lru_cache(maxsize=10)
def _get_encoding(model: str):
    """Get tiktoken encoding for a model (cached)."""
    if not TIKTOKEN_AVAILABLE:
        return None

    encoding_name = MODEL_ENCODINGS.get(model, MODEL_ENCODINGS["default"])
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to get encoding for {model}: {e}")
        return None


class TokenCounter:
    """
    Token counter with model-aware counting and context limit management.
    """

    def __init__(self, default_model: str = "claude-3-sonnet"):
        self.default_model = default_model
        self._encoding = _get_encoding(default_model)

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text.

        Args:
            text: The text to count
            model: Optional model override

        Returns:
            Token count
        """
        if not text:
            return 0

        model = model or self.default_model

        if TIKTOKEN_AVAILABLE:
            encoding = _get_encoding(model) or self._encoding
            if encoding:
                try:
                    return len(encoding.encode(text))
                except Exception as e:
                    logger.warning(f"Token encoding failed: {e}")

        # Fallback: character-based approximation
        # Average ~4 characters per token for English text
        return len(text) // 4

    def count_tokens_dict(self, data: Dict, model: Optional[str] = None) -> int:
        """
        Count tokens in a dictionary (serialized as JSON-like text).

        Args:
            data: Dictionary to count
            model: Optional model override

        Returns:
            Token count
        """
        import json
        try:
            text = json.dumps(data, default=str)
            return self.count_tokens(text, model)
        except Exception:
            # Fallback to string representation
            return self.count_tokens(str(data), model)

    def get_context_limit(self, model: Optional[str] = None) -> int:
        """
        Get the context limit for a model.

        Args:
            model: Model name

        Returns:
            Context limit in tokens
        """
        model = model or self.default_model
        return MODEL_CONTEXT_LIMITS.get(model, MODEL_CONTEXT_LIMITS["default"])

    def fits_in_context(
        self,
        text: str,
        model: Optional[str] = None,
        reserve_tokens: int = 1000
    ) -> bool:
        """
        Check if text fits in model context with reserved space for response.

        Args:
            text: Text to check
            model: Model name
            reserve_tokens: Tokens to reserve for response

        Returns:
            True if fits
        """
        model = model or self.default_model
        limit = self.get_context_limit(model) - reserve_tokens
        return self.count_tokens(text, model) <= limit

    def truncate_to_fit(
        self,
        text: str,
        max_tokens: int,
        model: Optional[str] = None,
        truncation_marker: str = "\n...[truncated]..."
    ) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            model: Model name
            truncation_marker: Marker to indicate truncation

        Returns:
            Truncated text
        """
        if not text:
            return text

        current_tokens = self.count_tokens(text, model)
        if current_tokens <= max_tokens:
            return text

        # Binary search for optimal truncation point
        marker_tokens = self.count_tokens(truncation_marker, model)
        target_tokens = max_tokens - marker_tokens

        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if self.count_tokens(text[:mid], model) <= target_tokens:
                low = mid
            else:
                high = mid - 1

        return text[:low] + truncation_marker


# Global instance for convenience
_counter = TokenCounter()


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Module-level convenience function."""
    return _counter.count_tokens(text, model)


def get_context_limit(model: Optional[str] = None) -> int:
    """Module-level convenience function."""
    return _counter.get_context_limit(model)


def truncate_to_fit(text: str, max_tokens: int, model: Optional[str] = None) -> str:
    """Module-level convenience function."""
    return _counter.truncate_to_fit(text, max_tokens, model)
