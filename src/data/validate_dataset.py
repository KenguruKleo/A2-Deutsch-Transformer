"""
validate_dataset.py — Validate and inspect a training JSONL dataset.

Checks each line for:
  - Valid JSON
  - Required keys ("input", "output")
  - Correct emoji markers (❌/✅ and 📝)
  - Duplicate inputs

Also prints statistics: total count, correct/incorrect ratio, avg output length.

Usage:
    python -m src.data.validate_dataset --input data/train_v3.jsonl
    python -m src.data.validate_dataset --input data/train_v3.jsonl --dedup --output data/train_v3_clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_and_validate(path: Path) -> tuple[list[dict], list[str]]:
    """
    Load a JSONL file and validate each line.

    Returns:
        (valid_examples, errors) — list of valid dicts and list of error messages.
    """
    valid: list[dict] = []
    errors: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            # JSON parse check
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_no}: Invalid JSON — {exc}")
                continue

            if not isinstance(obj, dict):
                errors.append(f"Line {line_no}: Expected object, got {type(obj).__name__}")
                continue

            # Required keys
            for key in ("input", "output"):
                if key not in obj:
                    errors.append(f"Line {line_no}: Missing key '{key}'")
                    break
            else:
                # Emoji content checks
                output = obj.get("output", "")
                has_incorrect = "❌" in output
                has_correct = "✅" in output
                has_explanation = "📝" in output

                if not (has_incorrect or has_correct):
                    errors.append(f"Line {line_no}: output missing ❌ or ✅")
                    continue
                if not has_explanation:
                    errors.append(f"Line {line_no}: output missing 📝 explanation")
                    continue

                valid.append(obj)

    return valid, errors


def deduplicate(examples: list[dict]) -> tuple[list[dict], int]:
    """Remove duplicate examples (matched by 'input' field)."""
    seen: set[str] = set()
    unique: list[dict] = []
    dupes = 0
    for ex in examples:
        key = ex["input"].strip().lower()
        if key in seen:
            dupes += 1
        else:
            seen.add(key)
            unique.append(ex)
    return unique, dupes


def print_stats(examples: list[dict]) -> None:
    """Print dataset statistics."""
    total = len(examples)
    if total == 0:
        print("⚠️  No valid examples found.")
        return

    incorrect_count = sum(1 for ex in examples if "❌" in ex["output"])
    correct_count = total - incorrect_count

    avg_input_len = sum(len(ex["input"]) for ex in examples) / total
    avg_output_len = sum(len(ex["output"]) for ex in examples) / total

    # Topic guessing (rough, based on common German grammar keywords in output)
    print("\n📊 Dataset Statistics")
    print("─" * 40)
    print(f"  Total examples  : {total:,}")
    print(f"  ❌ Incorrect     : {incorrect_count:,} ({incorrect_count / total:.1%})")
    print(f"  ✅ Correct       : {correct_count:,}  ({correct_count / total:.1%})")
    print(f"  Avg input len   : {avg_input_len:.1f} chars")
    print(f"  Avg output len  : {avg_output_len:.1f} chars")
    print("─" * 40)


def save_examples(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\n💾 Saved {len(examples)} examples → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and inspect a German grammar training JSONL dataset.",
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file to validate.")
    parser.add_argument("--dedup", action="store_true", help="Remove duplicate examples (by input text).")
    parser.add_argument("--output", "-o", default=None, help="Output path for cleaned dataset (requires --dedup).")
    parser.add_argument("--show-errors", action="store_true", help="Print all validation error messages.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 Validating: {input_path}")
    valid, errors = load_and_validate(input_path)

    print(f"\n✅ Valid   : {len(valid):,}")
    print(f"❌ Errors  : {len(errors):,}")

    if errors and args.show_errors:
        print("\n── Errors ──")
        for err in errors[:50]:  # Limit to first 50 errors
            print(f"  {err}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more.")

    if args.dedup:
        cleaned, dupes = deduplicate(valid)
        print(f"🗑️  Removed {dupes} duplicates → {len(cleaned)} unique examples")

        if args.output:
            save_examples(cleaned, Path(args.output))
        else:
            print("ℹ️  No --output specified; deduped data not saved.")
        print_stats(cleaned)
    else:
        print_stats(valid)


if __name__ == "__main__":
    main()
