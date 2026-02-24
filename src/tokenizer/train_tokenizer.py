"""
train_tokenizer.py — Тренує BPE токенізатор на даних проекту.

Джерела тексту:
  1. data/train.jsonl + data/val.jsonl  — input і output поля
  2. data_raw/Begegnungen_А2.pdf        — текст підручника (якщо є PyMuPDF)

Результат:
  src/tokenizer/tokenizer.json  — HF-сумісний BPE токенізатор

Запуск (після генерації даних):
  python src/tokenizer/train_tokenizer.py

Або автоматично з generator.py — він викликає train() після генерації.
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
# Збираємо текст з усіх джерел
# ---------------------------------------------------------------------------

def _iter_jsonl() -> list[str]:
    texts: list[str] = []
    for path in [TRAIN_JSONL, VAL_JSONL]:
        if not path.exists():
            print(f"  ⚠️  {path.name} не знайдено — пропускаємо")
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
        print(f"  ⚠️  PDF не знайдено ({PDF_PATH.name}) — пропускаємо")
        return []
    try:
        import fitz
    except ImportError:
        print("  ⚠️  PyMuPDF не встановлений (pip install pymupdf) — пропускаємо PDF")
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
# Тренування
# ---------------------------------------------------------------------------

def train() -> None:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    print("\n🚀 Тренуємо BPE токенізатор")
    print(f"   vocab_size : {VOCAB_SIZE}")
    print(f"   output     : {OUTPUT_PATH}\n")

    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))

    # ByteLevel — обробляє будь-який Unicode через байти (як GPT-2)
    # Ніяких <UNK> для нових символів — будь-який символ розкладається в байти
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder       = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,       # токен має зустрітись хоча б 2 рази
        show_progress=True,
    )

    print("📂 Збираємо тексти:")
    texts  = _iter_jsonl()
    texts += _iter_pdf()
    print(f"\n  📊 Всього рядків: {len(texts):,}")

    tokenizer.train_from_iterator(texts, trainer=trainer)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUTPUT_PATH))
    print(f"\n✅ Збережено: {OUTPUT_PATH}")

    # --- Перевірка ---
    print("\n🧪 Перевірка:")
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

    print("📌 Спеціальні токени:")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok:8s} → id {tokenizer.token_to_id(tok)}")
    print(f"\n📦 Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    train()
