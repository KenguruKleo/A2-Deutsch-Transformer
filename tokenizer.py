"""
tokenizer.py — Word-level токенізатор для A2 German Grammar Tutor.

═══════════════════════════════════════════════════════════
ЩО ТАКЕ ТОКЕНІЗАТОР?
═══════════════════════════════════════════════════════════

Модель працює лише з числами. Токенізатор — це «перекладач»
між людським текстом і числами (ID):

    encode("Ich bin müde.") → [4, 60, 469, 4]      текст → числа
    decode([4, 60, 469, 4]) → "Ich bin müde."       числа → текст

Як це працює:
    1. Текст розбивається на слова (токени)
    2. Кожне слово шукається у словнику (vocab.json)
    3. Якщо слово знайдене → повертаємо його ID
    4. Якщо не знайдене → повертаємо ID <UNK> (unknown)

Спеціальні токени:
    <PAD> (id=0) — заповнення коротких послідовностей до однієї довжини
    <BOS> (id=1) — "beginning of sequence" — маркер початку тексту
    <EOS> (id=2) — "end of sequence" — маркер кінця тексту
    <UNK> (id=3) — "unknown" — замінює будь-яке невідоме слово

═══════════════════════════════════════════════════════════
ЧОМУ WORD-LEVEL, А НЕ BPE/SENTENCEPIECE?
═══════════════════════════════════════════════════════════

Для великих моделей (GPT, LLaMA) використовують sub-word токенізатори
(BPE), які розбивають слова на частини: "gegangen" → "ge" + "gang" + "en".

Ми обрали word-level, тому що:
    ✅ Простіше для навчання та розуміння
    ✅ V=2000 достатньо для A2 (обмежена лексика)
    ✅ Кожен токен = ціле слово → легше інтерпретувати
    ❌ Мінус: невідомі слова → <UNK> (не може «вгадати» за частинами)

═══════════════════════════════════════════════════════════
"""

import json
import re
from pathlib import Path


# Тензорні форми (tensor shapes) для цього етапу:
#
#   Input text:     "Ich bin müde."
#   After encode:   [1, 60, 155, 469, 4, 2]          shape: [seq_len]
#                    ↑                    ↑
#                   BOS                  EOS
#
#   Batch (padding): [[1, 60, 155, 469, 4, 2],        shape: [batch_size, max_seq_len]
#                     [1, 22, 88,  4,   2, 0]]         ← 0 = PAD
#                                              ↑
#                                             PAD


class Tokenizer:
    """Word-level токенізатор з фіксованим словником.

    Матричне представлення:
        vocab — це відображення (mapping): str → int
        Embedding-шар потім перетворить int → vector [d_model]

        Ланцюг:  текст → Tokenizer → [id₁, id₂, …] → Embedding → [[v₁], [v₂], …]
                  str       ↓          list[int]          ↓         [seq_len, d_model]
    """

    # Спеціальні токени — константи
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, vocab_path: str | Path = "vocab.json"):
        """Завантажує словник з JSON-файлу.

        Args:
            vocab_path: шлях до vocab.json (token → id)

        Внутрішня структура:
            self.token_to_id: {"ich": 60, "bin": 155, ...}  — для encode
            self.id_to_token: {60: "ich", 155: "bin", ...}  — для decode
        """
        vocab_path = Path(vocab_path)
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Vocab file not found: {vocab_path}\n"
                f"Run 'python build_vocab.py' to create it."
            )

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token_to_id: dict[str, int] = json.load(f)

        # Зворотній маппінг: id → token (для decode)
        self.id_to_token: dict[int, str] = {
            idx: token for token, idx in self.token_to_id.items()
        }

        # Зберігаємо ID спеціальних токенів для швидкого доступу
        self.pad_id = self.token_to_id[self.PAD_TOKEN]   # 0
        self.bos_id = self.token_to_id[self.BOS_TOKEN]   # 1
        self.eos_id = self.token_to_id[self.EOS_TOKEN]   # 2
        self.unk_id = self.token_to_id[self.UNK_TOKEN]   # 3

    @property
    def vocab_size(self) -> int:
        """Розмір словника — кількість унікальних токенів.

        Це число визначає розмір embedding-матриці:
            Embedding matrix shape = [vocab_size, d_model] = [~2000, 128]
        """
        return len(self.token_to_id)

    def _tokenize(self, text: str) -> list[str]:
        """Розбиває текст на список токенів (слів + пунктуація).

        Використовує regex для розділення:
        - Слова (з умлаутами: ä, ö, ü, ß)
        - Пунктуація окремо (. , ! ? : ;)
        - Emoji-маркери (✅, ❌, 📝)
        - Спеціальні слова з двокрапкою (Correct:, Explanation:)
        - Символ нового рядка \\n

        Приклад:
            "Ich bin müde." → ["Ich", "bin", "müde", "."]
            "❌ Incorrect." → ["❌", "Incorrect", "."]
        """
        # Порядок альтернатив важливий: спершу довші патерни!
        pattern = (
            r"Correct:|Incorrect\.|Explanation:|Пояснення:"  # multi-char specials
            r"|\.\.\."                                        # три крапки (...)
            r"|[✅❌📝]"                                      # emoji-маркери
            r"|\n"                                            # новий рядок
            r"|[A-Za-zÄäÖöÜüß\u0400-\u04FF]+"               # слова
            r"|[.,!?;:\"'\-()]"                               # пунктуація
        )
        tokens = re.findall(pattern, text)
        return [t for t in tokens if t]  # filter empty strings just in case

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_len: int | None = None,
    ) -> list[int]:
        """Перетворює текст у послідовність ID.

        Tensor shape: [seq_len]  — одновимірний вектор

        Args:
            text: вхідний текст
            add_bos: чи додавати <BOS> на початок (зазвичай True)
            add_eos: чи додавати <EOS> в кінець (зазвичай True)
            max_len: максимальна довжина (обрізає, якщо довше)

        Returns:
            list[int] — послідовність token ID

        Приклад:
            encode("Ich bin müde.")
            → tokenize: ["Ich", "bin", "müde", "."]
            → lookup:   [60, 155, 469, 4]
            → + BOS/EOS: [1, 60, 155, 469, 4, 2]
        """
        raw_tokens = self._tokenize(text)

        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)

        for token in raw_tokens:
            # Шукаємо токен у словнику
            token_id = self.token_to_id.get(token)
            if token_id is not None:
                ids.append(token_id)
            else:
                # Спробуємо lowercase (якщо «Heute» не знайдено, шукаємо «heute»)
                token_id = self.token_to_id.get(token.lower())
                if token_id is not None:
                    ids.append(token_id)
                else:
                    ids.append(self.unk_id)  # невідоме слово → <UNK>

        if add_eos:
            ids.append(self.eos_id)

        # Обрізаємо до max_len (включно з BOS/EOS)
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
            # Переконуємось що останній токен — EOS
            if add_eos:
                ids[-1] = self.eos_id

        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Перетворює послідовність ID назад у текст.

        Args:
            ids: список token ID
            skip_special: якщо True, пропускає <PAD>, <BOS>, <EOS>, <UNK>

        Returns:
            str — відновлений текст

        Приклад:
            decode([1, 60, 155, 469, 4, 2]) → "Ich bin müde."
        """
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        tokens: list[str] = []

        for token_id in ids:
            if skip_special and token_id in special_ids:
                continue
            token = self.id_to_token.get(token_id, self.UNK_TOKEN)
            tokens.append(token)

        # Зʼєднуємо токени назад у текст
        # Пунктуація приєднується без пробілу перед нею
        if not tokens:
            return ""

        result = tokens[0]
        for token in tokens[1:]:
            if token in {".", ",", "!", "?", ":", ";", ")", '"', "'", "..."}:
                result += token
            elif token == "\n":
                result += token
            elif result.endswith("(") or result.endswith('"') or result.endswith("\n"):
                result += token
            else:
                result += " " + token

        return result

    def pad_sequence(
        self, ids: list[int], max_len: int, pad_id: int | None = None
    ) -> list[int]:
        """Доповнює послідовність PAD-токенами до потрібної довжини.

        Навіщо padding?
        Нейромережа обробляє дані батчами (пачками).
        Всі послідовності в батчі мають бути однієї довжини:

            Batch (до padding):
                [1, 60, 155, 469, 4, 2]        ← 6 токенів
                [1, 22, 88,  4,   2]            ← 5 токенів  ← РІЗНА ДОВЖИНА!

            Batch (після padding до max_len=6):
                [1, 60, 155, 469, 4, 2]         ← 6 токенів
                [1, 22, 88,  4,   2, 0]         ← 6 токенів  ← 0 = PAD

            Tensor shape: [batch_size=2, max_seq_len=6]
        """
        if pad_id is None:
            pad_id = self.pad_id

        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [pad_id] * (max_len - len(ids))

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size}, path=vocab.json)"


# ═══════════════════════════════════════════════════════════
# SELF-TEST: запусти python tokenizer.py для перевірки
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  🧪 Tokenizer Self-Test")
    print("=" * 60)

    tok = Tokenizer("vocab.json")
    print(f"\n  ✅ Loaded: {tok}")
    print(f"  Special IDs: PAD={tok.pad_id}, BOS={tok.bos_id}, EOS={tok.eos_id}, UNK={tok.unk_id}")

    # ─── Test 1: Basic encode/decode ──────────────────────
    print("\n─── Test 1: Encode / Decode ───")
    test_sentence = "Ich bin müde."
    encoded = tok.encode(test_sentence)
    decoded = tok.decode(encoded)
    print(f"  Input:   '{test_sentence}'")
    print(f"  Encoded: {encoded}  (shape: [{len(encoded)}])")
    print(f"  Decoded: '{decoded}'")

    # ─── Test 2: Unknown words ────────────────────────────
    print("\n─── Test 2: Unknown Words ───")
    test_unk = "Ich spiele Klavier."
    encoded_unk = tok.encode(test_unk)
    decoded_unk = tok.decode(encoded_unk, skip_special=False)
    print(f"  Input:   '{test_unk}'")
    print(f"  Encoded: {encoded_unk}")
    print(f"  Decoded: '{decoded_unk}'")
    unk_count = encoded_unk.count(tok.unk_id)
    print(f"  UNK tokens: {unk_count}")

    # ─── Test 3: Tutor response format ────────────────────
    print("\n─── Test 3: Tutor Response ───")
    tutor_output = "❌ Incorrect.\n✅ Correct: Ich bin nach Hause gegangen.\n📝 Explanation: gehen вживається з sein."
    encoded_tutor = tok.encode(tutor_output)
    decoded_tutor = tok.decode(encoded_tutor)
    print(f"  Input:   '{tutor_output}'")
    print(f"  Encoded: {encoded_tutor}  (len={len(encoded_tutor)})")
    print(f"  Decoded: '{decoded_tutor}'")

    # ─── Test 4: Padding ──────────────────────────────────
    print("\n─── Test 4: Padding ───")
    short = tok.encode("Ich lerne.")
    padded = tok.pad_sequence(short, max_len=10)
    print(f"  Original:  {short}  (len={len(short)})")
    print(f"  Padded:    {padded}  (len={len(padded)})")

    # ─── Test 5: Roundtrip ────────────────────────────────
    print("\n─── Test 5: Roundtrip ───")
    sentences = [
        "Ich habe gegessen.",
        "Heute gehe ich in die Schule.",
        "Er kann gut Deutsch sprechen.",
        "Wir sind nach Berlin gefahren.",
    ]
    all_ok = True
    for s in sentences:
        enc = tok.encode(s)
        dec = tok.decode(enc)
        ok = "✅" if dec == s else "❌"
        if dec != s:
            all_ok = False
        print(f"  {ok} '{s}' → {enc} → '{dec}'")

    # ─── Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅ All tests passed!")
    else:
        print("  ⚠️  Some roundtrip tests had differences (may be OK for punctuation)")
    print(f"  Vocab size: {tok.vocab_size}")
    print("=" * 60)
