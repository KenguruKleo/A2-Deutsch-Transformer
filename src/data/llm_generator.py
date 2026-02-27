"""
llm_generator.py — Core training example generator using a pluggable LLM provider.

Parses LLM responses into validated training JSONL examples, handles retries,
and supports resumable (append-mode) batch generation with a progress bar.

The actual LLM call is delegated to a BaseLLMProvider implementation
(OllamaProvider, OpenAIProvider, etc.) injected at construction time.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Literal

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.llm_provider import BaseLLMProvider
from src.data.llm_prompts import TOPICS, build_prompt
from src.data.vocabulary import pick_words

# ── Types ────────────────────────────────────────────────────────────────────

Mode = Literal["incorrect", "correct"]
Example = dict[str, str]  # {"input": ..., "output": ...}


class OllamaGenerator:
    """
    Generates German grammar training examples using a pluggable LLM provider.

    Args:
        provider:         Any BaseLLMProvider implementation.
        ratio_incorrect:  Fraction of examples that should contain a grammar error.
        max_retries:      Number of attempts per example when the response is invalid.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        ratio_incorrect: float = 0.6,
        max_retries: int = 3,
    ) -> None:
        self.provider = provider
        self.ratio_incorrect = ratio_incorrect
        self.max_retries = max_retries

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _pick_topic(self) -> str:
        return random.choice(list(TOPICS.keys()))

    def _pick_mode(self) -> Mode:
        return "incorrect" if random.random() < self.ratio_incorrect else "correct"

    def _parse_response(self, raw: str, mode: Mode) -> Example | None:
        """
        Parse and validate the raw LLM text as a training example.

        Handles:
          - Stripping markdown code fences (```json … ```)
          - Extracting the first JSON object from surrounding text
          - Validating required emoji markers (❌/✅ and 📝)
        """
        if not raw:
            return None

        cleaned = raw.strip()
        # Strip markdown fences the model might add despite instructions
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )

        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            return None

        try:
            obj = json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            return None

        if not isinstance(obj, dict):
            return None

        input_text = obj.get("input", "")
        output_text = obj.get("output", "")

        if not input_text or not output_text:
            return None
        if mode == "incorrect" and "❌" not in output_text:
            return None
        if mode == "correct" and "✅" not in output_text:
            return None
        if "📝" not in output_text:
            return None

        return {"input": str(input_text).strip(), "output": str(output_text).strip()}

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_one(
        self,
        topic: str | None = None,
        mode: Mode | None = None,
    ) -> Example | None:
        """
        Generate a single training example with retry logic.

        Each call samples 2 random vocabulary words and injects them into
        the prompt — this is the primary mechanism for preventing duplicate
        sentences when the same grammar topic is requested multiple times.

        Returns None if all retries fail or provider is unreachable.
        """
        topic = topic or self._pick_topic()
        mode = mode or self._pick_mode()
        words = pick_words(n=2)
        prompt = build_prompt(topic, mode, vocabulary_hint=words)

        for attempt in range(1, self.max_retries + 1):
            raw = self.provider.complete(prompt)
            if raw is None:
                return None  # Connection / API error — no point retrying

            example = self._parse_response(raw, mode)
            if example is not None:
                return example

            if attempt < self.max_retries:
                time.sleep(1)

        return None

    def generate_batch(
        self,
        n: int,
        output_path: str | Path,
        append: bool = True,
    ) -> int:
        """
        Generate *n* **unique** training examples and write them to a JSONL file.

        Deduplicates in real-time: maintains a ``seen`` set of already-written
        input sentences (loaded from the existing file when ``append=True``).
        Duplicate outputs from the LLM are silently discarded and a new example
        is requested instead, so the file always contains only fresh entries.

        Args:
            n:           Number of *new unique* examples to generate.
            output_path: Path to the output JSONL file.
            append:      Append to existing file (resumable) or overwrite.

        Returns:
            Number of successfully generated and written examples.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Load seen-inputs from existing file ───────────────────────────────
        seen: set[str] = set()
        if append and output_path.exists():
            with open(output_path, encoding="utf-8", errors="ignore") as f_existing:
                for raw_line in f_existing:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        key = obj.get("input", "").strip().lower()
                        if key:
                            seen.add(key)
                    except json.JSONDecodeError:
                        pass

        generated = 0
        failed = 0
        skipped = 0

        print(f"\n🚀 Generating {n} examples → {output_path} (append={append})")
        print(f"   Provider : {self.provider}")
        print(f"   Ratio    : {self.ratio_incorrect:.0%} incorrect / "
              f"{1 - self.ratio_incorrect:.0%} correct")
        if seen:
            print(f"   Dedup    : {len(seen):,} existing inputs loaded (real-time dedup ON)\n")
        else:
            print()

        write_mode = "a" if append else "w"
        with open(output_path, write_mode, encoding="utf-8") as f:
            pbar = tqdm(total=n, unit="example", dynamic_ncols=True)
            try:
                while generated < n:
                    topic = self._pick_topic()
                    mode = self._pick_mode()
                    pbar.set_description(f"[{mode:9s}] {topic:30s}")

                    example = self.generate_one(topic=topic, mode=mode)

                    if example is None:
                        failed += 1
                        pbar.set_postfix(ok=generated, fail=failed, skip=skipped)
                        if failed > 10 and generated == 0:
                            print(
                                "\n❌ Too many failures with 0 successes — "
                                "is the LLM provider reachable?",
                                file=sys.stderr,
                            )
                            break
                        continue

                    # ── Real-time dedup check ─────────────────────────────────
                    key = example["input"].strip().lower()
                    if key in seen:
                        skipped += 1
                        pbar.set_postfix(ok=generated, fail=failed, skip=skipped)
                        continue

                    seen.add(key)
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    f.flush()
                    generated += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=generated, fail=failed, skip=skipped)
            finally:
                pbar.close()

        suffix = f", {skipped} duplicate(s) skipped" if skipped else ""
        print(f"\n✅ Done: {generated} generated, {failed} failed{suffix} → {output_path}")
        return generated
