"""
tests/test_llm_generator.py — Unit tests for LLMGenerator and helpers.

All LLM calls are mocked via the provider abstraction — tests run without
a real Ollama/OpenAI instance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.llm_prompts import TOPICS, build_prompt
from src.data.llm_generator import OllamaGenerator


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_provider(return_value: str | None = None, side_effect=None) -> MagicMock:
    """Build a mock BaseLLMProvider."""
    provider = MagicMock()
    if side_effect is not None:
        provider.complete.side_effect = side_effect
    else:
        provider.complete.return_value = return_value
    return provider


@pytest.fixture()
def gen() -> OllamaGenerator:
    """OllamaGenerator with a no-op mock provider."""
    return OllamaGenerator(provider=_make_provider(), ratio_incorrect=0.6, max_retries=2)


# ── Prompt tests ──────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_incorrect_prompt_contains_emoji(self):
        prompt = build_prompt("verb_conjugation", "incorrect")
        assert "❌" in prompt
        assert "📝" in prompt
        assert "✅" in prompt

    def test_correct_prompt_contains_emoji(self):
        prompt = build_prompt("nominativ", "correct")
        assert "✅" in prompt
        assert "📝" in prompt

    def test_prompt_contains_topic_description(self):
        prompt = build_prompt("modal_verbs", "incorrect")
        assert "modal verbs" in prompt.lower()

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            build_prompt("nominativ", "invalid_mode")

    def test_unknown_topic_raises(self):
        with pytest.raises(KeyError):
            build_prompt("nonexistent_topic", "incorrect")

    def test_all_topics_have_descriptions(self):
        for key, desc in TOPICS.items():
            assert isinstance(desc, str) and len(desc) > 5, f"Missing description for {key!r}"


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestParseResponse:
    def test_valid_incorrect_response(self, gen):
        raw = json.dumps({
            "input": "Er lerne Deutsch.",
            "output": "❌ Incorrect.\n✅ Correct: Er lernt Deutsch.\n📝 Пояснення: Правило відмінювання."
        }, ensure_ascii=False)
        result = gen._parse_response(raw, "incorrect")
        assert result is not None
        assert result["input"] == "Er lerne Deutsch."
        assert "❌" in result["output"]

    def test_valid_correct_response(self, gen):
        raw = json.dumps({
            "input": "Ich trinke Kaffee.",
            "output": "✅ Correct.\n📝 Пояснення: Правильна дієвідміна."
        }, ensure_ascii=False)
        result = gen._parse_response(raw, "correct")
        assert result is not None
        assert "✅" in result["output"]

    def test_strips_markdown_fences(self, gen):
        obj = {"input": "Er lerne.", "output": "❌ Incorrect.\n✅ Correct: Er lernt.\n📝 Тест."}
        raw = f"```json\n{json.dumps(obj, ensure_ascii=False)}\n```"
        result = gen._parse_response(raw, "incorrect")
        assert result is not None

    def test_invalid_json_returns_none(self, gen):
        assert gen._parse_response("this is not json", "incorrect") is None

    def test_missing_explanation_emoji_returns_none(self, gen):
        raw = json.dumps({
            "input": "Er lerne.",
            "output": "❌ Incorrect.\n✅ Correct: Er lernt."  # Missing 📝
        }, ensure_ascii=False)
        assert gen._parse_response(raw, "incorrect") is None

    def test_missing_incorrect_emoji_returns_none(self, gen):
        raw = json.dumps({
            "input": "Er lerne.",
            "output": "✅ Correct.\n📝 Test."  # ❌ missing but mode=incorrect
        }, ensure_ascii=False)
        assert gen._parse_response(raw, "incorrect") is None

    def test_empty_input_returns_none(self, gen):
        raw = json.dumps({"input": "", "output": "❌\n✅\n📝"}, ensure_ascii=False)
        assert gen._parse_response(raw, "incorrect") is None


# ── generate_one tests ────────────────────────────────────────────────────────

class TestGenerateOne:
    def test_returns_example_on_success(self):
        good_json = json.dumps({
            "input": "Er lerne Deutsch.",
            "output": "❌ Incorrect.\n✅ Correct: Er lernt Deutsch.\n📝 Пояснення: Тест."
        }, ensure_ascii=False)
        gen = OllamaGenerator(provider=_make_provider(good_json), max_retries=2)
        result = gen.generate_one(topic="verb_conjugation", mode="incorrect")
        assert result is not None
        assert result["input"] == "Er lerne Deutsch."

    def test_returns_none_on_connection_failure(self):
        gen = OllamaGenerator(provider=_make_provider(None), max_retries=2)
        assert gen.generate_one() is None

    def test_retries_on_bad_json(self):
        good_json = json.dumps({
            "input": "Ich gehe.",
            "output": "✅ Correct.\n📝 Тест."
        }, ensure_ascii=False)
        # First call: garbage; second call: valid
        gen = OllamaGenerator(
            provider=_make_provider(side_effect=["bad json", good_json]),
            max_retries=3,
        )
        result = gen.generate_one(topic="nominativ", mode="correct")
        assert result is not None

    def test_provider_complete_called_with_prompt(self):
        provider = _make_provider(None)
        gen = OllamaGenerator(provider=provider, max_retries=1)
        gen.generate_one(topic="nominativ", mode="correct")
        provider.complete.assert_called_once()
        prompt_arg = provider.complete.call_args[0][0]
        assert "nominativ" in prompt_arg.lower() or "Nominativ" in prompt_arg


# ── generate_batch tests ──────────────────────────────────────────────────────

def _make_unique_provider(mode: str = "incorrect") -> MagicMock:
    """Provider whose complete() returns a unique input sentence on every call."""
    counter = {"n": 0}

    def unique_response(*_args, **_kwargs):
        counter["n"] += 1
        if mode == "incorrect":
            return json.dumps({
                "input": f"Er lerne Deutsch {counter['n']}.",
                "output": "❌ Incorrect.\n✅ Correct: Er lernt Deutsch.\n📝 Тест."
            }, ensure_ascii=False)
        return json.dumps({
            "input": f"Er lernt Deutsch {counter['n']}.",
            "output": "✅ Correct.\n📝 Тест."
        }, ensure_ascii=False)

    provider = MagicMock()
    provider.complete.side_effect = unique_response
    return provider


class TestGenerateBatch:
    def test_writes_to_file(self, tmp_path):
        gen = OllamaGenerator(provider=_make_unique_provider(), max_retries=2)
        output = tmp_path / "out.jsonl"
        count = gen.generate_batch(n=3, output_path=output, append=False)
        assert count == 3
        lines = output.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_append_mode_adds_to_existing(self, tmp_path):
        seed_json = json.dumps({
            "input": "Er lerne.",
            "output": "❌ Incorrect.\n✅ Correct: Er lernt.\n📝 Тест."
        }, ensure_ascii=False)
        output = tmp_path / "out.jsonl"
        output.write_text(seed_json + "\n" + seed_json + "\n", encoding="utf-8")
        # The seed has 2 identical lines; after dedup only 1 unique input exists.
        gen = OllamaGenerator(provider=_make_unique_provider(), max_retries=2)
        gen.generate_batch(n=2, output_path=output, append=True)
        lines = output.read_text(encoding="utf-8").strip().splitlines()
        # 2 seed lines (unchanged) + 2 new unique lines = 4
        assert len(lines) == 4

    def test_overwrite_mode_replaces_file(self, tmp_path):
        seed_json = json.dumps({
            "input": "Er lerne.",
            "output": "❌ Incorrect.\n✅ Correct: Er lernt.\n📝 Тест."
        }, ensure_ascii=False)
        output = tmp_path / "out.jsonl"
        output.write_text(seed_json + "\n", encoding="utf-8")
        gen = OllamaGenerator(provider=_make_unique_provider(), max_retries=2)
        gen.generate_batch(n=2, output_path=output, append=False)
        lines = output.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2  # overwritten, not appended

    def test_dedup_skips_duplicate_input(self, tmp_path):
        """generate_batch must not write an input sentence it has already seen."""
        existing_json = json.dumps({
            "input": "Ich gehe nach Hause.",
            "output": "✅ Correct.\n📝 Тест."
        }, ensure_ascii=False)
        output = tmp_path / "out.jsonl"
        output.write_text(existing_json + "\n", encoding="utf-8")

        # Provider returns the duplicate first, then something new
        duplicate = json.dumps({
            "input": "Ich gehe nach Hause.",  # same as existing
            "output": "✅ Correct.\n📝 Тест."
        }, ensure_ascii=False)
        fresh = json.dumps({
            "input": "Sie arbeitet heute.",
            "output": "✅ Correct.\n📝 Тест."
        }, ensure_ascii=False)

        provider = _make_provider(side_effect=[duplicate, fresh])
        gen = OllamaGenerator(provider=provider, max_retries=1)
        count = gen.generate_batch(n=1, output_path=output, append=True)

        assert count == 1
        lines = output.read_text(encoding="utf-8").strip().splitlines()
        # original line + 1 new unique line = 2 total
        assert len(lines) == 2
        inputs = [json.loads(l)["input"] for l in lines]
        assert "Sie arbeitet heute." in inputs
        assert inputs.count("Ich gehe nach Hause.") == 1  # not duplicated
