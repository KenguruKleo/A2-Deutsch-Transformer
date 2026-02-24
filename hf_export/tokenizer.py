"""
tokenizer.py — Thin compatibility shim around PreTrainedTokenizerFast.

WHY THIS FILE EXISTS
─────────────────────
train.py, inference.py and generate.py call:
    tokenizer.encode(text, add_bos=True, add_eos=True, max_len=N)
    tokenizer.decode(ids)
    tokenizer.pad_sequence(ids, max_len)
    tokenizer.pad_id / .bos_id / .eos_id
    tokenizer.token_to_id  (dict-like)

PreTrainedTokenizerFast uses a slightly different API.
This shim adapts the HF tokenizer to the old API so we don't have to touch
every call site.  All the real work is delegated to PreTrainedTokenizerFast.

USAGE
─────
    from src.tokenizer.tokenizer import Tokenizer

    tok = Tokenizer("src/tokenizer/tokenizer.json")
    ids = tok.encode("Ich habe den Auto.", add_bos=True, add_eos=True)
    text = tok.decode(ids)
"""

from pathlib import Path
from transformers import PreTrainedTokenizerFast

# Special token strings (must match what train_tokenizer.py registered)
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


class Tokenizer:
    """Compatibility wrapper around PreTrainedTokenizerFast.

    Exposes the same API that train.py / inference.py / generate.py expect,
    while storing a standard HF tokenizer under self._tok.
    """

    def __init__(self, tokenizer_path: str | Path | None = None):
        if tokenizer_path is None:
            tokenizer_path = Path(__file__).parent / "tokenizer.json"

        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found: {tokenizer_path}\n"
                f"Run first: python src/tokenizer/train_tokenizer.py"
            )

        self._tok = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            bos_token=BOS_TOKEN,
            eos_token=EOS_TOKEN,
            pad_token=PAD_TOKEN,
            unk_token=UNK_TOKEN,
        )

        # Resolve special IDs once
        self.pad_id: int = self._tok.convert_tokens_to_ids(PAD_TOKEN)
        self.bos_id: int = self._tok.convert_tokens_to_ids(BOS_TOKEN)
        self.eos_id: int = self._tok.convert_tokens_to_ids(EOS_TOKEN)
        self.unk_id: int = self._tok.convert_tokens_to_ids(UNK_TOKEN)

        self._special_ids = {self.pad_id, self.bos_id, self.eos_id}

        # Legacy dict-style access: tokenizer.token_to_id["<BOS>"]
        self.token_to_id: dict[str, int] = self._tok.get_vocab()

    # ── Core API ─────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_len: int | None = None,
    ) -> list[int]:
        """Text → list of token IDs."""
        ids: list[int] = self._tok.encode(text, add_special_tokens=False)

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
        """List of token IDs → text."""
        if skip_special:
            ids = [i for i in ids if i not in self._special_ids]
        return self._tok.decode(ids)

    def pad_sequence(
        self, ids: list[int], max_len: int, pad_id: int | None = None
    ) -> list[int]:
        """Pad or truncate a sequence to exactly max_len."""
        if pad_id is None:
            pad_id = self.pad_id

        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [pad_id] * (max_len - len(ids))

    def __repr__(self) -> str:
        return (
            f"Tokenizer(vocab_size={self.vocab_size}, bpe, "
            f"pad={self.pad_id}, bos={self.bos_id}, eos={self.eos_id})"
        )
