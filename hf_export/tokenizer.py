"""
tokenizer.py — BPE tokenizer for A2 German Grammar Tutor (v2.0).

═══════════════════════════════════════════════════════════
WHAT CHANGED FROM v1.0?
═══════════════════════════════════════════════════════════

v1.0 — Word-level tokenizer:
    Vocabulary = fixed list of words (4000 entries).
    "gegangen" → id 47   (if in vocabulary)
    "gehts"    → <UNK>   (if not — unknown word)

v2.0 — Byte-level BPE tokenizer:
    Vocabulary = subwords (8000 entries), trained on project data.
    "gegangen" → ["Ġgeg", "angen"]  → [312, 891]
    "gehts"    → ["Ġge", "hts"]     → [89, 1203]   ← never <UNK>

Advantages of BPE:
    ✅ No <UNK> — any word decomposes into subparts
    ✅ Handles typos, new words, B1 vocabulary
    ✅ HuggingFace-compatible format (tokenizer.json)
    ✅ German umlauts ä ö ü ß and Cyrillic handled correctly

The public API is unchanged:
    encode(text)           → list[int]
    decode(ids)            → str
    pad_sequence(ids, n)   → list[int]

═══════════════════════════════════════════════════════════
DATA FLOW:
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
    """Wrapper around the HuggingFace BPE tokenizer.

    Preserves the same API as v1.0 (word-level tokenizer),
    so train.py / inference.py / generate.py require no changes.

    Matrix representation:
        Vocabulary — mapping: str → int  (8000 subwords)
        Embedding layer converts int → vector [d_model]

        Chain:  text → Tokenizer → [id₁, id₂, …] → Embedding → [[v₁], [v₂], …]
                  str       ↓          list[int]          ↓         [seq_len, d_model]
    """

    def __init__(self, tokenizer_path: str | Path | None = None):
        """Loads the BPE tokenizer from tokenizer.json.

        Args:
            tokenizer_path: path to tokenizer.json.
                            Defaults to the file next to this module.
        """
        from tokenizers import Tokenizer as HFTokenizer

        if tokenizer_path is None:
            tokenizer_path = TOKENIZER_JSON

        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found: {tokenizer_path}\n"
                f"Run first: python src/tokenizer/train_tokenizer.py"
            )

        self._tok: HFTokenizer = HFTokenizer.from_file(str(tokenizer_path))

        # Special token IDs — resolved once at init
        self.pad_id: int = self._tok.token_to_id(PAD_TOKEN)
        self.bos_id: int = self._tok.token_to_id(BOS_TOKEN)
        self.eos_id: int = self._tok.token_to_id(EOS_TOKEN)
        self.unk_id: int = self._tok.token_to_id(UNK_TOKEN)

        # For compatibility with old code that accesses token_to_id directly
        self.token_to_id: dict[str, int] = self._tok.get_vocab()

        self._special_ids = {self.pad_id, self.bos_id, self.eos_id}

    @property
    def vocab_size(self) -> int:
        """Vocabulary size — determines the embedding matrix shape [vocab_size, d_model]."""
        return self._tok.get_vocab_size()

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_len: int | None = None,
    ) -> list[int]:
        """Converts text into a sequence of token IDs.

        Args:
            text:    input text
            add_bos: prepend <BOS> token
            add_eos: append <EOS> token
            max_len: maximum length (truncates if longer)

        Returns:
            list[int] — token ID sequence,  shape: [seq_len]

        Example:
            encode("Ich bin müde.")
            → BPE: ["Ġ Ich", "Ġbin", "Ġm", "üde", "."]
            → ids: [1, 312, 891, 445, 203, 5, 2]
                    ↑                             ↑
                   BOS                           EOS
        """
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
        """Converts a sequence of token IDs back into text.

        Args:
            ids:          list of token IDs
            skip_special: if True, skips <PAD>, <BOS>, <EOS>

        Returns:
            str — reconstructed text

        Example:
            decode([1, 312, 891, 203, 5, 2]) → "Ich bin müde."
        """
        if skip_special:
            ids = [i for i in ids if i not in self._special_ids]

        return self._tok.decode(ids)

    def pad_sequence(
        self, ids: list[int], max_len: int, pad_id: int | None = None
    ) -> list[int]:
        """Pads a sequence with <PAD> tokens to a fixed length.

        Required for batch processing — all sequences in a batch
        must have the same length:

            Before padding:  [1, 312, 891, 2]          len=4
            After (n=6):     [1, 312, 891, 2, 0, 0]    len=6
                                              ↑↑
                                             PAD

        Tensor shape after padding: [batch_size, max_seq_len]
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

    # ─── Test 2: No UNK for out-of-vocabulary words ───────
    print("\n─── Test 2: Out-of-vocabulary words (no UNK) ───")
    unk_test = "Donaudampfschifffahrtsgesellschaft"
    enc2 = tok.encode(unk_test)
    dec2 = tok.decode(enc2)
    unk_count = enc2.count(tok.unk_id)
    print(f"  Input:     '{unk_test}'")
    print(f"  Encoded:   {enc2}")
    print(f"  Decoded:   '{dec2}'")
    print(f"  UNK count: {unk_count}  (expected: 0)")

    # ─── Test 3: Tutor response format ────────────────────
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
