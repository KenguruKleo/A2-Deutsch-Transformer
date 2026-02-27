"""
generate_data.py — CLI for LLM-backed training data generation.

Generates German grammar training examples using a pluggable LLM provider
(Ollama or OpenAI-compatible) and writes them to data/train.jsonl.

Supports resumable generation via --append; safe to interrupt with Ctrl+C.
After interruption a reminder is printed to retrain the tokenizer.

Usage:
    # Generate 500 examples with Ollama (default)
    python -m src.data.generate_data --count 500

    # Append to an existing file (resume a previous run)
    python -m src.data.generate_data --count 200 --append

    # Use OpenAI provider (requires OPENAI_API_KEY env var)
    python -m src.data.generate_data --count 500 --provider openai

    # Use a different model
    python -m src.data.generate_data --count 100 --model gpt-4o-mini --provider openai

    # Write to a custom path
    python -m src.data.generate_data --count 100 --output data/extra.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import load_config
from src.data.llm_provider import BaseLLMProvider
from src.data.llm_generator import OllamaGenerator

_TOKENIZER_REMINDER = """
⚠️  Generation interrupted.
   {saved} example(s) have been saved to {path}

   Remember to retrain the tokenizer before re-training the model:

       python src/tokenizer/train_tokenizer.py

   Then run training:

       python src/train.py
"""


# ── Provider factory ──────────────────────────────────────────────────────────

def build_provider(args: argparse.Namespace) -> BaseLLMProvider:
    """Instantiate the requested LLM provider from CLI args + config."""
    cfg = load_config()

    if args.provider == "ollama":
        from src.data.providers.ollama import OllamaProvider  # noqa: PLC0415
        ollama_cfg = cfg.ollama
        return OllamaProvider(
            host=args.host or ollama_cfg.host,
            model=args.model or ollama_cfg.model,
            temperature=ollama_cfg.timeout_seconds and 0.7,  # kept simple
            timeout=ollama_cfg.timeout_seconds,
        )

    elif args.provider == "openai":
        from src.data.providers.openai import OpenAIProvider  # noqa: PLC0415
        openai_cfg = cfg.openai  # type: ignore[attr-defined]
        return OpenAIProvider(
            base_url=args.host or openai_cfg.base_url,
            model=args.model or openai_cfg.model,
            api_key_env=openai_cfg.api_key_env,
            temperature=openai_cfg.temperature,
            max_tokens=openai_cfg.max_tokens,
        )

    else:
        print(f"❌ Unknown provider: {args.provider!r}", file=sys.stderr)
        sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def count_existing(path: Path) -> int:
    """Count lines in an existing JSONL file (0 if missing)."""
    if not path.exists():
        return 0
    return sum(
        1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    )


def _confirm_overwrite(path: Path) -> bool:
    """Ask the user to confirm overwriting an existing file. Returns True = proceed."""
    existing = count_existing(path)
    size_kb = path.stat().st_size / 1024
    print(
        f"\n⚠️  {path} already exists ({existing} examples, {size_kb:.1f} KB)."
        "\n   Use --append to resume, or confirm overwrite below."
    )
    try:
        answer = input("   Overwrite? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")


def _print_usage(provider) -> None:
    """Print token usage and cost estimate if the provider supports it."""
    usage = provider.get_usage()
    if usage is None:
        return
    print(
        f"\n💰 Token Usage ({usage['model']})"
        f"\n   Prompt tokens     : {usage['prompt_tokens']:,}"
        f"\n   Completion tokens : {usage['completion_tokens']:,}"
        f"\n   Total tokens      : {usage['total_tokens']:,}"
        f"\n   Estimated cost    : ${usage['cost_usd']:.4f} USD"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate German grammar training data via a local or cloud LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        required=True,
        help="Number of examples to generate.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/train.jsonl",
        help="Output JSONL file (default: data/train.jsonl).",
    )
    parser.add_argument(
        "--append", "-a",
        action="store_true",
        help="Append to an existing file instead of overwriting (enables resume).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM backend to use (default: ollama).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model name from config.yaml.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override the provider host/base_url from config.yaml.",
    )
    parser.add_argument(
        "--ratio-incorrect",
        type=float,
        default=None,
        dest="ratio_incorrect",
        help="Fraction of incorrect examples (0.0–1.0, default: from config.yaml).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_path = Path(args.output)

    # Confirm before overwriting an existing non-empty file
    if not args.append and output_path.exists() and count_existing(output_path) > 0:
        if not _confirm_overwrite(output_path):
            print("\nAborted. Use --append to continue adding to the existing file.")
            sys.exit(0)

    # Show existing count when appending
    if args.append:
        existing = count_existing(output_path)
        if existing > 0:
            print(
                f"📖 Found {existing} existing examples in {output_path}. "
                f"Appending {args.count} more."
            )

    provider = build_provider(args)

    cfg = load_config()
    ratio = (
        args.ratio_incorrect
        if args.ratio_incorrect is not None
        else cfg.ollama.ratio_incorrect
    )

    gen = OllamaGenerator(
        provider=provider,
        ratio_incorrect=ratio,
        max_retries=cfg.ollama.max_retries,
    )

    generated = 0
    try:
        generated = gen.generate_batch(
            n=args.count,
            output_path=output_path,
            append=args.append,
        )
    except KeyboardInterrupt:
        _print_usage(provider)
        print(
            _TOKENIZER_REMINDER.format(saved=generated, path=output_path),
            file=sys.stderr,
        )
        sys.exit(0)

    _print_usage(provider)

    if generated == 0:
        print("❌ No examples generated.", file=sys.stderr)
        sys.exit(1)

    # Remind user to retrain tokenizer after a successful full run too
    print(
        "\n📌 Next steps:\n"
        "   1. Retrain tokenizer:  python src/tokenizer/train_tokenizer.py\n"
        "   2. Train model:        python src/train.py"
    )


if __name__ == "__main__":
    main()
