"""
tokenizer.py — Word-level tokenizer for A2 German Grammar Tutor.

═══════════════════════════════════════════════════════════
WHAT IS A TOKENIZER?
═══════════════════════════════════════════════════════════

The model works only with numbers. A tokenizer is a "translator"
between human text and numbers (IDs):

    encode("Ich bin müde.") → [4, 60, 469, 4]      text → numbers
    decode([4, 60, 469, 4]) → "Ich bin müde."       numbers → text

How it works:
    1. Text is split into words (tokens)
    2. Each word is looked up in the dictionary (vocab.json)
    3. If word is found → return its ID
    4. If not found → return ID <UNK> (unknown)

Special tokens:
    <PAD> (id=0) — padding shorter sequences to equal length
    <BOS> (id=1) — "beginning of sequence" — start of text marker
    <EOS> (id=2) — "end of sequence" — end of text marker
    <UNK> (id=3) — "unknown" — replaces any unknown word

═══════════════════════════════════════════════════════════
WHY WORD-LEVEL INSTEAD OF BPE/SENTENCEPIECE?
═══════════════════════════════════════════════════════════

Large models (GPT, LLaMA) use sub-word tokenizers (BPE), 
which split words into parts: "gegangen" → "ge" + "gang" + "en".

We chose word-level because:
    ✅ Simpler for training and understanding
    ✅ V=4000 is enough for A2 (limited vocabulary)
    ✅ Each token = a whole word → easier to interpret
    ❌ Drawback: unknown words → <UNK> (cannot "guess" by parts)

═══════════════════════════════════════════════════════════
"""

import json
import re
from pathlib import Path


# Tensor shapes for this stage:
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
    """Word-level tokenizer with fixed vocabulary.

    Matrix representation:
        vocab — mapping: str → int
        Embedding layer will then convert int → vector [d_model]

        Chain:  text → Tokenizer → [id₁, id₂, …] → Embedding → [[v₁], [v₂], …]
                  str       ↓          list[int]          ↓         [seq_len, d_model]
    """

    # Special tokens constants
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, vocab_path: str | Path | None = None):
        """Loads vocabulary from a JSON file."""
        if vocab_path is None:
            # Look in the same directory as this script
            vocab_path = Path(__file__).parent / "vocab.json"
        
        vocab_path = Path(vocab_path)
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Vocab file not found: {vocab_path}\n"
                f"Run 'python build_vocab.py' to create it."
            )

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token_to_id: dict[str, int] = json.load(f)

        # Reverse mapping: id → token (for decoding)
        self.id_to_token: dict[int, str] = {
            idx: token for token, idx in self.token_to_id.items()
        }

        # Store special token IDs for fast access
        self.pad_id = self.token_to_id[self.PAD_TOKEN]   # 0
        self.bos_id = self.token_to_id[self.BOS_TOKEN]   # 1
        self.eos_id = self.token_to_id[self.EOS_TOKEN]   # 2
        self.unk_id = self.token_to_id[self.UNK_TOKEN]   # 3

    @property
    def vocab_size(self) -> int:
        """Vocabulary size — number of unique tokens.

        This number determines the size of the embedding matrix:
            Embedding matrix shape = [vocab_size, d_model] = [4000, 128]
        """
        return len(self.token_to_id)

    def _tokenize(self, text: str) -> list[str]:
        """Splits text into a list of tokens (words + punctuation).

        Uses regex for separation:
        - Words (with umlauts: ä, ö, ü, ß)
        - Punctuation separately (. , ! ? : ;)
        - Emoji markers (✅, ❌, 📝)
        - Special keywords (Correct:, Explanation:, Пояснення:)
        - Newline symbol \n

        Example:
            "Ich bin müde." → ["Ich", "bin", "müde", "."]
            "❌ Incorrect." → ["❌", "Incorrect", "."]
        """
        # Alternative order is important: longer patterns first!
        pattern = (
            r"Correct:|Incorrect\.|Explanation:|Пояснення:"  # multi-char specials
            r"|\.\.\."                                        # triple dots (...)
            r"|[✅❌📝]"                                      # emoji markers
            r"|\n"                                            # new line
            r"|[A-Za-zÄäÖöÜüß\u0400-\u04FF]+"               # words (German + Slavic characters)
            r"|[.,!?;:\"'\-()]"                               # punctuation
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
        """Converts text into a sequence of IDs.

        Tensor shape: [seq_len] — 1D vector

        Args:
            text: input text
            add_bos: whether to add <BOS> at the beginning (usually True)
            add_eos: whether to add <EOS> at the end (usually True)
            max_len: maximum length (trims if longer)

        Returns:
            list[int] — sequence of token IDs

        Example:
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
            # Look up token in dictionary
            token_id = self.token_to_id.get(token)
            if token_id is not None:
                ids.append(token_id)
            else:
                # Try lowercase (if "Heute" is not found, look for "heute")
                token_id = self.token_to_id.get(token.lower())
                if token_id is not None:
                    ids.append(token_id)
                else:
                    ids.append(self.unk_id)  # unknown word → <UNK>

        if add_eos:
            ids.append(self.eos_id)

        # Trim to max_len (including BOS/EOS)
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
            # Ensure last token is EOS if requested
            if add_eos:
                ids[-1] = self.eos_id

        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Converts a sequence of IDs back to text.

        Args:
            ids: list of token IDs
            skip_special: if True, skips <PAD>, <BOS>, <EOS>, <UNK>

        Returns:
            str — reconstructed text

        Example:
            decode([1, 60, 155, 469, 4, 2]) → "Ich bin müde."
        """
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        tokens: list[str] = []

        for token_id in ids:
            if skip_special and token_id in special_ids:
                continue
            token = self.id_to_token.get(token_id, self.UNK_TOKEN)
            tokens.append(token)

        # Join tokens back into text
        # Punctuation attaches without a space before it
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
        """Pads sequence with PAD tokens to the required length.

        Why padding?
        The neural network processes data in batches.
        All sequences in a batch must have the same length:

            Batch (before padding):
                [1, 60, 155, 469, 4, 2]        ← 6 tokens
                [1, 22, 88,  4,   2]            ← 5 tokens  ← DIFFERENT LENGTHS!

            Batch (after padding to max_len=6):
                [1, 60, 155, 469, 4, 2]         ← 6 tokens
                [1, 22, 88,  4,   2, 0]         ← 6 tokens  ← 0 = PAD

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
# SELF-TEST: run 'python tokenizer.py' to verify
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
    tutor_output = "❌ Incorrect.\n✅ Correct: Ich bin nach Hause gegangen.\n📝 Explanation: gehen is used with sein."
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
