"""
providers/ollama.py — Ollama LLM backend for the data generator.

Sends prompts to a locally running Ollama instance via its REST API
(non-streaming mode) and returns the raw response text.
"""

from __future__ import annotations

import sys

import requests

from src.data.llm_provider import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """
    Completion provider that calls a local Ollama instance.

    Args:
        host:        Ollama server URL, e.g. ``"http://localhost:11434"``.
        model:       Ollama model tag, e.g. ``"hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M"``.
        temperature: Sampling temperature (0 = deterministic).
        timeout:     Request timeout in seconds.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M",
        temperature: float = 0.7,
        timeout: int = 90,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._api_url = f"{self.host}/api/generate"

    def complete(self, prompt: str) -> str | None:
        """Send *prompt* to Ollama and return the generated text."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                # NOTE: do NOT set num_predict here — reasoning models (e.g. Qwen3)
                # use a large number of tokens internally for "thinking" before
                # emitting the actual response. Capping num_predict cuts off the
                # thought chain, resulting in an empty `response` field.
            },
        }
        try:
            resp = requests.post(self._api_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("response", "").strip() or None
        except requests.exceptions.ConnectionError:
            print(
                f"\n❌ Cannot connect to Ollama at {self.host}. "
                "Make sure `ollama serve` is running.",
                file=sys.stderr,
            )
            return None
        except requests.exceptions.Timeout:
            print(
                f"\n⚠️  Ollama request timed out after {self.timeout}s.",
                file=sys.stderr,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"\n⚠️  Ollama error: {exc}", file=sys.stderr)
            return None

    def __repr__(self) -> str:
        return f"OllamaProvider(model={self.model!r}, host={self.host!r})"
