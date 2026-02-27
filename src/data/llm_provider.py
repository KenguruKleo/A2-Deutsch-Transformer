"""
llm_provider.py — Abstract base class for LLM completion providers.

Any new backend (Ollama, OpenAI, LM Studio, Anthropic…) should subclass
BaseLLMProvider and implement the single `complete()` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """
    Minimal interface for text-completion LLM backends.

    Implementations must be stateless with respect to prompt history —
    each call to `complete()` is an independent request.
    """

    @abstractmethod
    def complete(self, prompt: str) -> str | None:
        """
        Send *prompt* to the LLM and return the raw response text.

        Args:
            prompt: The full prompt string to send.

        Returns:
            The model's response as a plain string, or ``None`` if the
            request failed (connection error, timeout, API error, etc.).
        """

    def get_usage(self) -> dict[str, Any] | None:
        """
        Return accumulated token usage statistics, or ``None`` if this
        provider does not track usage.

        Providers that support billing/token tracking should override this
        and return a dict with at least:
            prompt_tokens     (int)
            completion_tokens (int)
            total_tokens      (int)
            cost_usd          (float)  — estimated cost in USD
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
