"""
train_tokenizer.py — Trains a Byte-level BPE tokenizer on project data.

Text sources:
  1. data/train.jsonl + data/val.jsonl  — input and output fields
  2. data_raw/Begegnungen_А2.pdf        — textbook text (requires PyMuPDF)

Output:
  src/tokenizer/tokenizer.json  — HuggingFace-compatible BPE tokenizer

Usage (after generating data):
  python src/tokenizer/train_tokenizer.py

Also called automatically by generator.py after data generation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAIN_JSONL  = PROJECT_ROOT / "data" / "train.jsonl"
VAL_JSONL    = PROJECT_ROOT / "data" / "val.jsonl"
PDF_PATH     = PROJECT_ROOT / "data_raw" / "Begegnungen_А2.pdf"
OUTPUT_PATH  = PROJECT_ROOT / "src" / "tokenizer" / "tokenizer.json"

VOCAB_SIZE     = 8000
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]


# ---------------------------------------------------------------------------
# Text collection
# ---------------------------------------------------------------------------

def _iter_jsonl() -> list[str]:
    texts: list[str] = []
    for path in [TRAIN_JSONL, VAL_JSONL]:
        if not path.exists():
            print(f"  ⚠️  {path.name} not found — skipping")
            continue
        print(f"  📄 {path.name}...")
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "input"  in entry: texts.append(entry["input"])
                    if "output" in entry: texts.append(entry["output"])
                except json.JSONDecodeError:
                    continue
    return texts


def _iter_pdf() -> list[str]:
    if not PDF_PATH.exists():
        print(f"  ⚠️  PDF not found ({PDF_PATH.name}) — skipping")
        return []
    try:
        import fitz
    except ImportError:
        print("  ⚠️  PyMuPDF not installed (pip install pymupdf) — skipping PDF")
        return []

    print(f"  📚 {PDF_PATH.name}...")
    texts: list[str] = []
    doc = fitz.open(str(PDF_PATH))
    for page in doc:
        for line in str(page.get_text()).splitlines():
            line = line.strip()
            if len(line) > 5:
                texts.append(line)
    doc.close()
    return texts


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    print("\n🚀 Training BPE tokenizer")
    print(f"   vocab_size : {VOCAB_SIZE}")
    print(f"   output     : {OUTPUT_PATH}\n")

    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))

    # ByteLevel — handles any Unicode via bytes (like GPT-2).
    # No <UNK> for unknown characters — any character decomposes into bytes.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder       = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,       # token must appear at least twice
        show_progress=True,
    )

    print("📂 Collecting texts:")
    texts  = _iter_jsonl()
    texts += _iter_pdf()
    print(f"\n  📊 Total lines: {len(texts):,}")

    tokenizer.train_from_iterator(texts, trainer=trainer)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUTPUT_PATH))
    print(f"\n✅ Saved: {OUTPUT_PATH}")

    # --- Verification ---
    print("\n🧪 Verification:")
    test_cases = [
        "Heute gehe ich ins Kino.",
        "Ich habe nach Berlin gefahren.",
        "❌ Incorrect.\n✅ Correct: Heute gehe ich ins Kino.",
        "Дієслово fahren означає рух.",
        "Donaudampfschifffahrtsgesellschaft",
    ]
    for sentence in test_cases:
        enc = tokenizer.encode(sentence)
        dec = tokenizer.decode(enc.ids)
        print(f"  in : {sentence[:60]}")
        print(f"  tok: {enc.tokens[:10]}{'...' if len(enc.tokens) > 10 else ''}")
        print(f"  out: {dec[:60]}")
        print()

    print("📌 Special tokens:")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok:8s} → id {tokenizer.token_to_id(tok)}")
    print(f"\n📦 Final vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    train()
