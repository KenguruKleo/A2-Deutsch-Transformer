"""
inference.py — Shared inference utilities for A2 Deutsch Grammar Tutor v2.1 (HF BART).

Provides two reusable building blocks:
  - load_model()         → loads tokenizer + HF BART model from directory
  - generate_response()  → uses model.generate() for Seq2Seq inference
"""

import torch
from pathlib import Path
from transformers import BartForConditionalGeneration

from src.config import Config, get_device, get_project_root
from src.tokenizer.tokenizer import Tokenizer


def load_model(
    model_path: str | Path | None = None,
    config: Config | None = None,
) -> tuple[BartForConditionalGeneration, Tokenizer, str]:
    """
    Load tokenizer and HF BART model from a saved directory.

    The model directory should contain:
      - config.json          (BartConfig — architecture params)
      - model.safetensors    (weights: E, P, W_Q/K/V/O, W₁, W₂, LN per layer)

    Args:
        model_path: Path to model directory (default: project_root/model_final).
        config: Optional Config for device selection.

    Returns:
        (model, tokenizer, device)
    """
    from src.config import load_config

    if config is None:
        config = load_config()

    project_root = get_project_root()
    if model_path is None:
        model_path = project_root / "model_final"
    model_path = Path(model_path)

    device = get_device(config.training.device)
    tokenizer = Tokenizer(project_root / "src/tokenizer/tokenizer.json")

    # Load HF model directly — no manual state_dict manipulation needed
    model = BartForConditionalGeneration.from_pretrained(str(model_path))
    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_response(
    text: str,
    model: BartForConditionalGeneration,
    tokenizer: Tokenizer,
    config: Config,
    device: str,
    max_len: int = 64,
) -> str:
    """
    Run grammar check using HF's model.generate().

    Data flow:
      text → encode → input_ids [1, T_src]
           → Encoder → memory [1, T_src, d=256]
           → Decoder (greedy) → output_ids [1, T_out]
           → decode → response string

    Args:
        text: German sentence to check.
        model: BartForConditionalGeneration in eval mode.
        tokenizer: Tokenizer wrapper.
        config: Config for max_seq_len.
        device: Device string.
        max_len: Maximum generation length.

    Returns:
        Grammar check result as string.
    """
    model.eval()

    # Encode source: <BOS> + [tokens] + <EOS>
    src_ids = tokenizer.encode(text, add_bos=True, add_eos=True, max_len=max_len)
    input_ids = torch.tensor([src_ids], dtype=torch.long, device=device)
    attention_mask = (input_ids != tokenizer.pad_id).long()

    # Generate using HF standard pipeline
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_len,
            num_beams=1,
            do_sample=False,
        )

    # Decode result (skip <BOS>, <EOS>, <PAD>)
    result = tokenizer.decode(output_ids[0].tolist(), skip_special=True)
    return result.strip()
