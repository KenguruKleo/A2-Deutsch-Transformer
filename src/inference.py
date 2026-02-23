"""
inference.py — Shared inference utilities for A2 Deutsch Grammar Tutor.

Provides two reusable building blocks:
  - load_model()         → loads tokenizer + model from checkpoint
  - generate_response()  → autoregressive token generation

Used by:
  - src/generate.py      (CLI tool)
  - tests/evaluate_model.py (evaluation harness)
"""

import torch
import torch.nn.functional as F
from pathlib import Path

from src.config import Config, get_device, get_project_root
from src.model.model import TransformerModel
from src.tokenizer.tokenizer import Tokenizer


def load_model(
    model_path: str | Path | None = None,
    config: Config | None = None,
) -> tuple[TransformerModel, Tokenizer, str]:
    """Load tokenizer and model from a checkpoint.

    Args:
        model_path: Path to .pth checkpoint. Defaults to <project_root>/model_final.pth.
        config:     Already-loaded Config. If None, loads from project root config.yaml.

    Returns:
        (model, tokenizer, device) — model is in eval mode and on device.
    """
    from src.config import load_config  # local import to avoid circular issues

    if config is None:
        config = load_config()

    project_root = get_project_root()
    if model_path is None:
        model_path = project_root / "model_final.pth"

    device = get_device(config.training.device)
    tokenizer = Tokenizer(project_root / "src/tokenizer/vocab.json")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    model = TransformerModel(
        vocab_size=config.model.vocab_size,
        max_seq_len=config.model.max_seq_len,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, device


def generate_response(
    text: str,
    model: TransformerModel,
    tokenizer: Tokenizer,
    config: Config,
    device: str,
    max_new_tokens: int = 64,
) -> str:
    """Run autoregressive generation for a single input sentence.

    Args:
        text:           Input German sentence to check.
        model:          Loaded TransformerModel in eval mode.
        tokenizer:      Tokenizer with the project vocabulary.
        config:         Config with generation.temperature / top_k / model.max_seq_len.
        device:         Device string ("cpu", "mps", "cuda", …).
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        Decoded string — includes the original input text followed by the model response.
    """
    input_ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    temperature = config.generation.temperature
    top_k = config.generation.top_k

    for _ in range(max_new_tokens):
        idx_cond = input_tensor[:, -config.model.max_seq_len:]

        with torch.no_grad():
            logits = model(idx_cond)  # [1, seq_len, vocab_size]

        logits = logits[:, -1, :] / temperature  # [1, vocab_size]

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_tensor = torch.cat((input_tensor, next_token), dim=1)

        if next_token.item() == tokenizer.eos_id:
            break

    full_ids = input_tensor[0].tolist()
    if full_ids[0] == tokenizer.bos_id:
        full_ids = full_ids[1:]

    return tokenizer.decode(full_ids)
