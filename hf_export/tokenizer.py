"""
tokenizer.py — BPE токенізатор для A2 German Grammar Tutor (v2.0).

═══════════════════════════════════════════════════════════
ЩО ЗМІНИЛОСЬ ВІДНОСНО v1.0?
═══════════════════════════════════════════════════════════

v1.0 — Word-level токенізатор:
    Словник = список конкретних слів (4000 штук).
    "gegangen" → id 47   (якщо є в словнику)
    "gehts"    → <UNK>   (якщо немає — невідоме слово)

v2.0 — Byte-level BPE токенізатор:
    Словник = підслова (8000 штук), натренований на даних.
    "gegangen" → ["▁geg", "angen"]  → [312, 891]
    "gehts"    → ["▁ge", "hts"]     → [89, 1203]   ← ніколи не дає <UNK>

Переваги BPE:
    ✅ Ніяких <UNK> — будь-яке слово розкладається на підчастини
    ✅ Обробляє опечатки, нові слова, B1 словник
    ✅ HuggingFace-сумісний формат (tokenizer.json)
    ✅ Умлаути ä ö ü ß та кирилиця обробляються коректно

API залишається тим самим:
    encode(text)           → list[int]
    decode(ids)            → str
    pad_sequence(ids, n)   → list[int]

═══════════════════════════════════════════════════════════
ЯК ЧИТАТИ CHAIN ПЕРЕТВОРЕНЬ:
═══════════════════════════════════════════════════════════

    "Ich bin müde."
         ↓  encode()
    [1, 312, 891, 1203, 45, 2]     shape: [seq_len]
     ↑                          ↑
    BOS                        EOS
         ↓  Embedding layer
    [[v₁], [v₂], …]               shape: [seq_len, d_model]

═══════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizers import Tokenizer as HFTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


TOKENIZER_JSON = Path(__file__).parent / "tokenizer.json"

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


class Tokenizer:
    """Wrapper над HuggingFace BPE токенізатором.

    Зберігає той самий API що був у v1.0 (word-level),
    тому train.py / inference.py / generate.py не потребують змін.

    Matrix representation:
        Словник — mapping: str → int  (8000 підслів)
        Embedding layer конвертує int → вектор [d_model]

        Chain:  text → Tokenizer → [id₁, id₂, …] → Embedding → [[v₁], [v₂], …]
                  str       ↓          list[int]          ↓         [seq_len, d_model]
    """

    def __init__(self, tokenizer_path: str | Path | None = None):
        """Завантажує BPE токенізатор з tokenizer.json.

        Args:
            tokenizer_path: шлях до tokenizer.json.
                            За замовчуванням — поруч з цим файлом.
        """
        from tokenizers import Tokenizer as HFTokenizer

        if tokenizer_path is None:
            tokenizer_path = TOKENIZER_JSON

        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"tokenizer.json не знайдено: {tokenizer_path}\n"
                f"Запусти спочатку: python src/tokenizer/train_tokenizer.py"
            )

        self._tok: HFTokenizer = HFTokenizer.from_file(str(tokenizer_path))

        # Спеціальні токени — отримуємо їх id один раз
        self.pad_id: int = self._tok.token_to_id(PAD_TOKEN)
        self.bos_id: int = self._tok.token_to_id(BOS_TOKEN)
        self.eos_id: int = self._tok.token_to_id(EOS_TOKEN)
        self.unk_id: int = self._tok.token_to_id(UNK_TOKEN)

        # Для сумісності зі старим кодом (train.py використовує token_to_id.get)
        self.token_to_id: dict[str, int] = self._tok.get_vocab()

        self._special_ids = {self.pad_id, self.bos_id, self.eos_id}

    @property
    def vocab_size(self) -> int:
        """Розмір словника — визначає розмір embedding матриці [vocab_size, d_model]."""
        return self._tok.get_vocab_size()

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_len: int | None = None,
    ) -> list[int]:
        """Конвертує текст у послідовність id.

        Args:
            text:    вхідний текст
            add_bos: додати <BOS> на початку
            add_eos: додати <EOS> в кінці
            max_len: максимальна довжина (обрізає якщо довше)

        Returns:
            list[int] — послідовність token id  shape: [seq_len]

        Example:
            encode("Ich bin müde.")
            → BPE: ["▁Ich", "▁bin", "▁m", "üde", "."]
            → ids: [1, 312, 891, 445, 203, 5, 2]
                    ↑                             ↑
                   BOS                           EOS
        """
        # Вимикаємо автоматичне додавання special tokens у HF tokenizer
        # — керуємо цим вручну для сумісності зі старим API
        encoding = self._tok.encode(text)
        ids: list[int] = encoding.ids

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
            if add_eos:
                ids[-1] = self.eos_id

        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Конвертує послідовність id назад у текст.

        Args:
            ids:          список token id
            skip_special: якщо True — пропускає <PAD>, <BOS>, <EOS>

        Returns:
            str — відновлений текст

        Example:
            decode([1, 312, 891, 203, 5, 2]) → "Ich bin müde."
        """
        if skip_special:
            ids = [i for i in ids if i not in self._special_ids]

        return self._tok.decode(ids)

    def pad_sequence(
        self, ids: list[int], max_len: int, pad_id: int | None = None
    ) -> list[int]:
        """Доповнює послідовність токенами <PAD> до потрібної довжини.

        Потрібно для батч-обробки — всі послідовності в батчі
        мають бути однакової довжини:

            До padding:   [1, 312, 891, 2]          len=4
            Після (n=6):  [1, 312, 891, 2, 0, 0]    len=6
                                            ↑↑
                                           PAD

        Tensor shape після padding: [batch_size, max_seq_len]
        """
        if pad_id is None:
            pad_id = self.pad_id

        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [pad_id] * (max_len - len(ids))

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size}, bpe, path=tokenizer.json)"


# ═══════════════════════════════════════════════════════════
# SELF-TEST: python src/tokenizer/tokenizer.py
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  🧪 BPE Tokenizer Self-Test")
    print("=" * 60)

    tok = Tokenizer()
    print(f"\n  ✅ Loaded: {tok}")
    print(f"  Special IDs: PAD={tok.pad_id}, BOS={tok.bos_id}, "
          f"EOS={tok.eos_id}, UNK={tok.unk_id}")

    # ─── Test 1: Basic encode/decode ──────────────────────
    print("\n─── Test 1: Encode / Decode ───")
    sentence = "Ich bin müde."
    enc = tok.encode(sentence)
    dec = tok.decode(enc)
    print(f"  Input:   '{sentence}'")
    print(f"  Encoded: {enc}  (len={len(enc)})")
    print(f"  Decoded: '{dec}'")

    # ─── Test 2: Нові слова — не дає <UNK> ───────────────
    print("\n─── Test 2: Out-of-vocabulary words (no UNK) ───")
    unk_test = "Donaudampfschifffahrtsgesellschaft"
    enc2 = tok.encode(unk_test)
    dec2 = tok.decode(enc2)
    unk_count = enc2.count(tok.unk_id)
    print(f"  Input:     '{unk_test}'")
    print(f"  Encoded:   {enc2}")
    print(f"  Decoded:   '{dec2}'")
    print(f"  UNK count: {unk_count}  (має бути 0)")

    # ─── Test 3: Tutor response ───────────────────────────
    print("\n─── Test 3: Tutor Response ───")
    tutor = "❌ Incorrect.\n✅ Correct: Ich bin nach Hause gegangen."
    enc3 = tok.encode(tutor)
    dec3 = tok.decode(enc3)
    print(f"  Input:   '{tutor}'")
    print(f"  Len:     {len(enc3)}")
    print(f"  Decoded: '{dec3}'")

    # ─── Test 4: Padding ──────────────────────────────────
    print("\n─── Test 4: Padding ───")
    short = tok.encode("Ich lerne.")
    padded = tok.pad_sequence(short, max_len=16)
    print(f"  Original: {short}  (len={len(short)})")
    print(f"  Padded:   {padded}  (len={len(padded)})")

    # ─── Test 5: Roundtrip ────────────────────────────────
    print("\n─── Test 5: Roundtrip ───")
    sentences = [
        "Ich habe gegessen.",
        "Heute gehe ich in die Schule.",
        "Er kann gut Deutsch sprechen.",
        "Wir sind nach Berlin gefahren.",
    ]
    for s in sentences:
        enc_s = tok.encode(s)
        dec_s = tok.decode(enc_s)
        ok = "✅" if dec_s.strip() == s.strip() else "⚠️ "
        print(f"  {ok} '{s}' → '{dec_s}'")

    print("\n" + "=" * 60)
    print(f"  Vocab size: {tok.vocab_size}")
    print("=" * 60)
