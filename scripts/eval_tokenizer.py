"""
scripts/eval_tokenizer.py — Measures BPE tokenizer quality.

Usage:
    python scripts/eval_tokenizer.py
    python scripts/eval_tokenizer.py --max-seq-len 128
    python scripts/eval_tokenizer.py --tokenizer path/to/other/tokenizer.json

Metric descriptions: docs/tokenizer_metrics.md
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_JSONL = PROJECT_ROOT / "data" / "train.jsonl"
VAL_JSONL   = PROJECT_ROOT / "data" / "val.jsonl"
TOKENIZER   = PROJECT_ROOT / "src" / "tokenizer" / "tokenizer.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_texts() -> list[str]:
    texts: list[str] = []
    for path in [TRAIN_JSONL, VAL_JSONL]:
        if not path.exists():
            print(f"  ⚠️  {path.name} not found — skipping")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "input"  in entry: texts.append(entry["input"])
                    if "output" in entry: texts.append(entry["output"])
                except json.JSONDecodeError:
                    continue
    return texts


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def measure_fertility(texts: list[str], tok) -> float:
    """Fertility — average number of tokens per word.

    Measures how efficiently the vocabulary covers the language.
    Ideal = 1.0 (every word is a single token).
    Normal range for BPE = 1.3–1.7.
    """
    total_tokens = 0
    total_words  = 0
    for text in texts:
        tokens = tok.encode(text).tokens
        words  = text.split()
        total_tokens += len(tokens)
        total_words  += len(words) if words else 1
    return total_tokens / total_words if total_words else 0.0


def measure_unk_rate(texts: list[str], tok, sample: int = 10_000) -> float:
    """UNK rate — percentage of <UNK> tokens in the data.

    For Byte-level BPE this is always 0% — any character decomposes
    into bytes. Useful as a sanity check.
    """
    unk_id = tok.token_to_id("<UNK>")
    total  = 0
    unks   = 0
    for text in texts[:sample]:
        ids    = tok.encode(text).ids
        total += len(ids)
        unks  += ids.count(unk_id)
    return (unks / total * 100) if total else 0.0


def measure_continuation_rate(texts: list[str], tok, sample: int = 5_000) -> float:
    """Continuation rate — fraction of tokens that are mid-word subwords.

    Byte-level BPE marks word starts with Ġ (space prefix).
    Tokens without Ġ are continuations (subwords inside a word).
    Lower = better vocabulary (more whole words as single tokens).
    """
    special = {"<PAD>", "<BOS>", "<EOS>", "<UNK>"}
    total        = 0
    continuation = 0
    for text in texts[:sample]:
        for token in tok.encode(text).tokens:
            if token in special:
                continue
            total += 1
            if not token.startswith("Ġ") and not token[0].isupper() and not token[0].isdigit():
                continuation += 1
    return (continuation / total * 100) if total else 0.0


def measure_sequence_lengths(texts: list[str], tok,
                              max_seq_len: int = 64,
                              sample: int = 10_000) -> dict:
    """Sequence length distribution relative to max_seq_len.

    Critical for choosing max_seq_len in config.yaml.
    If > 5% of sequences are truncated — increase max_seq_len.
    """
    lengths = [len(tok.encode(t).ids) for t in texts[:sample]]
    over    = sum(1 for l in lengths if l > max_seq_len)
    return {
        "avg":      sum(lengths) / len(lengths),
        "max":      max(lengths),
        "min":      min(lengths),
        "over":     over,
        "over_pct": over / len(lengths) * 100,
        "total":    len(lengths),
    }


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def interpret_fertility(v: float) -> str:
    if v < 1.3: return "🟢 Excellent — most words are single tokens"
    if v < 1.7: return "🟢 Normal — vocabulary covers the data well"
    if v < 2.5: return "🟡 Acceptable — consider increasing vocab_size"
    return             "🔴 Vocabulary too small — many words are split"

def interpret_continuation(v: float) -> str:
    if v < 25:  return "🟢 Excellent — most words are whole tokens"
    if v < 40:  return "🟢 Normal"
    if v < 55:  return "🟡 Moderate — consider increasing vocabulary"
    return             "🔴 Too many splits — vocabulary too small"

def interpret_over(pct: float) -> str:
    if pct == 0: return "🟢 All sequences fit within max_seq_len"
    if pct < 5:  return "🟡 Small fraction is truncated"
    return              "🔴 Too many truncations — increase max_seq_len"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(tokenizer_path: Path, max_seq_len: int = 64) -> None:
    from tokenizers import Tokenizer

    print("=" * 60)
    print("  📊 Tokenizer Quality Metrics")
    print("=" * 60)

    if not tokenizer_path.exists():
        print(f"\n❌ tokenizer.json not found: {tokenizer_path}")
        print("   Run first: python src/tokenizer/train_tokenizer.py")
        sys.exit(1)

    tok = Tokenizer.from_file(str(tokenizer_path))

    print(f"\n  File     : {tokenizer_path}")
    print(f"  Vocab    : {tok.get_vocab_size():,} tokens")

    print("\n📂 Loading data...")
    texts = load_texts()
    if not texts:
        print("❌ No data found. Run: python src/data/generator.py")
        sys.exit(1)
    print(f"   Sentences: {len(texts):,}")

    # ── Fertility ─────────────────────────────────────────
    print("\n── 1. Fertility (tokens / word) ────────────────────")
    f = measure_fertility(texts, tok)
    print(f"   Value    : {f:.3f}")
    print(f"   Rating   : {interpret_fertility(f)}")

    # ── UNK rate ──────────────────────────────────────────
    print("\n── 2. UNK rate ─────────────────────────────────────")
    u = measure_unk_rate(texts, tok)
    print(f"   Value    : {u:.4f}%")
    print(f"   Rating   : {'🟢 Perfect' if u == 0 else '🔴 Unknown tokens present'}")

    # ── Continuation rate ─────────────────────────────────
    print("\n── 3. Continuation token rate ──────────────────────")
    c = measure_continuation_rate(texts, tok)
    print(f"   Value    : {c:.1f}%")
    print(f"   Rating   : {interpret_continuation(c)}")

    # ── Sequence lengths ──────────────────────────────────
    print(f"\n── 4. Sequence lengths (max_seq_len={max_seq_len}) ─────")
    s = measure_sequence_lengths(texts, tok, max_seq_len)
    print(f"   Average  : {s['avg']:.1f} tokens")
    print(f"   Minimum  : {s['min']} tokens")
    print(f"   Maximum  : {s['max']} tokens")
    print(f"   > {max_seq_len}      : {s['over']}/{s['total']} ({s['over_pct']:.1f}%)")
    print(f"   Rating   : {interpret_over(s['over_pct'])}")

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    if f < 1.7 and u == 0 and c < 40 and s["over_pct"] < 5:
        print("  ✅ Tokenizer quality is good — vocab_size is well chosen")
    elif f > 2.5:
        print("  ⚠️  Consider increasing vocab_size (16000+)")
    elif s["over_pct"] > 5:
        print(f"  ⚠️  Consider increasing max_seq_len (currently {max_seq_len})")
    print("=" * 60)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BPE tokenizer quality")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=TOKENIZER,
        help="Path to tokenizer.json",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=64,
        help="max_seq_len from config.yaml (default: 64)",
    )
    args = parser.parse_args()
    run(args.tokenizer, args.max_seq_len)
