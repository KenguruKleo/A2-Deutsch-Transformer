"""
inference.py — Shared inference utilities for A2 Deutsch Grammar Tutor v2.0 (Encoder-Decoder).

Provides two reusable building blocks:
  - load_model()         → loads tokenizer + model from checkpoint
  - generate_response()  → autoregressive token generation (Seq2Seq style)
"""

import torch
import torch.nn.functional as F
from pathlib import Path

from src.config import Config, get_device, get_project_root
from src.model.model import GrammarTransformer
from src.tokenizer.tokenizer import Tokenizer

def load_model(
    model_path: str | Path | None = None,
    config: Config | None = None,
) -> tuple[GrammarTransformer, Tokenizer, str]:
    """Load tokenizer and model from a checkpoint."""
    from src.config import load_config

    if config is None:
        config = load_config()

    project_root = get_project_root()
    if model_path is None:
        model_path = project_root / "model_final.pth"

    device = get_device(config.training.device)
    tokenizer = Tokenizer(project_root / "src/tokenizer/tokenizer.json")

    # weights_only=False because of the custom arch marker
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = GrammarTransformer(
        vocab_size=config.model.vocab_size,
        max_seq_len=config.model.max_seq_len,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_enc_layers=config.model.n_enc_layers,
        n_dec_layers=config.model.n_dec_layers,
        d_ff=config.model.d_ff,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, device

def generate_response(
    text: str,
    model: GrammarTransformer,
    tokenizer: Tokenizer,
    config: Config,
    device: str,
    max_len: int = 64,
) -> str:
    """Run autoregressive generation for v2.0 Seq2Seq model."""
    model.eval()
    
    # Encode source text (Encoder Input)
    src_ids = tokenizer.encode(text, add_bos=False, add_eos=True, max_len=max_len)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    
    # Generate using model's autoregressive loop
    with torch.no_grad():
        generated_ids = model.generate(
            src_tensor, 
            bos_id=tokenizer.bos_id, 
            eos_id=tokenizer.eos_id, 
            pad_id=tokenizer.pad_id,
            max_len=max_len
        )
    
    # Decode result (remove BOS)
    result = tokenizer.decode(generated_ids[0].tolist()[1:], skip_special=True)
    return result.strip()
