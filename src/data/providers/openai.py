"""
providers/openai.py — OpenAI-compatible LLM backend for the data generator.

Supports any OpenAI-compatible API endpoint, including:
  - OpenAI (api.openai.com)
  - LM Studio  (http://localhost:1234/v1)
  - Ollama OpenAI-compat layer (http://localhost:11434/v1)
  - Azure OpenAI, Groq, Together AI, etc.

API key is read from the environment variable specified in config.yaml
(`openai.api_key_env`, default: ``"OPENAI_API_KEY"``).
Never hardcode the key in config or code.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from src.data.llm_provider import BaseLLMProvider

# Pricing per 1M tokens (as of 2025-02).  Update when OpenAI changes prices.
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini":          {"input": 0.150, "output": 0.600},
    "gpt-4o":               {"input": 2.500, "output": 10.000},
    "gpt-4-turbo":          {"input": 10.00, "output": 30.000},
    "gpt-3.5-turbo":        {"input": 0.500, "output": 1.500},
    "gpt-4.1":              {"input": 2.000, "output": 8.000},
    "gpt-4.1-mini":         {"input": 0.400, "output": 1.600},
    "gpt-4.1-nano":         {"input": 0.100, "output": 0.400},
}
_DEFAULT_PRICING = {"input": 1.0, "output": 3.0}  # fallback for unknown models


class OpenAIProvider(BaseLLMProvider):
    """
    Completion provider that uses the OpenAI Python SDK.

    Tracks cumulative token usage across all requests so that a cost
    estimate can be printed at the end of a generation run.

    Args:
        base_url:    API base URL. Defaults to OpenAI's public endpoint.
        model:       Model identifier, e.g. ``"gpt-4o-mini"``.
        api_key_env: Name of the environment variable holding the API key.
        temperature: Sampling temperature.
        max_tokens:  Maximum tokens to generate.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Accumulated usage across all complete() calls
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0

    def _get_client(self):
        """Lazily import and build the OpenAI client."""
        # Load .env file if present (does not override real env vars).
        try:
            from dotenv import load_dotenv  # noqa: PLC0415
            load_dotenv(override=False)
        except ImportError:
            pass  # python-dotenv not installed — fall back to env vars only

        try:
            import openai  # noqa: PLC0415
        except ImportError:
            print(
                "❌ openai package not installed. Run: pip install openai",
                file=sys.stderr,
            )
            raise

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"❌ Environment variable '{self.api_key_env}' is not set. "
                "Set it in your shell or in a .env file in the project root."
            )

        return openai.OpenAI(base_url=self.base_url, api_key=api_key)

    def complete(self, prompt: str) -> str | None:
        """Send *prompt* to the OpenAI-compatible API and return the response."""
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            # Accumulate token usage
            if response.usage:
                self._prompt_tokens += response.usage.prompt_tokens
                self._completion_tokens += response.usage.completion_tokens

            content = response.choices[0].message.content
            return content.strip() if content else None
        except EnvironmentError:
            return None
        except ImportError:
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"\n⚠️  OpenAI provider error: {exc}", file=sys.stderr)
            return None

    def get_usage(self) -> dict[str, Any] | None:
        """Return cumulative token usage and estimated cost in USD."""
        if self._prompt_tokens == 0 and self._completion_tokens == 0:
            return None

        # Find per-million pricing by matching model prefix
        pricing = _DEFAULT_PRICING
        model_lower = self.model.lower()
        for name, prices in _PRICING.items():
            if model_lower.startswith(name):
                pricing = prices
                break

        cost = (
            self._prompt_tokens     / 1_000_000 * pricing["input"]
            + self._completion_tokens / 1_000_000 * pricing["output"]
        )

        return {
            "prompt_tokens":     self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens":      self._prompt_tokens + self._completion_tokens,
            "cost_usd":          cost,
            "model":             self.model,
        }

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self.model!r}, base_url={self.base_url!r})"
