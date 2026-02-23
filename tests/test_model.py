import torch
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.model import TransformerModel
from src.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def model(config):
    return TransformerModel(
        vocab_size=config.model.vocab_size,
        max_seq_len=config.model.max_seq_len,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        weight_tying=config.model.weight_tying,
    )


def test_output_shape(model, config):
    """Logits shape must be [batch, seq_len, vocab_size]."""
    v = config.model.vocab_size
    t = config.model.max_seq_len
    batch_size = 2

    dummy_input = torch.randint(0, v, (batch_size, t))
    logits = model(dummy_input)

    assert logits.shape == (batch_size, t, v)


def test_weight_tying(model):
    """Embedding and LM head must share the same weight tensor."""
    assert model.token_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_device_compatibility(config):
    """Model must run forward pass on the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    v = config.model.vocab_size
    t = config.model.max_seq_len

    model = TransformerModel(
        vocab_size=v,
        max_seq_len=t,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        weight_tying=config.model.weight_tying,
    ).to(device)

    dummy_input = torch.randint(0, v, (4, t)).to(device)
    logits = model(dummy_input)

    assert logits.shape == (4, t, v)


def test_causal_mask_is_lower_triangular(model, config):
    """Causal mask must block future tokens (lower-triangular structure)."""
    t = config.model.max_seq_len
    mask = model._create_causal_mask(t, device="cpu")

    assert mask.shape == (1, 1, t, t)
    # Upper triangle (above diagonal) must be 0
    assert mask[0, 0].triu(diagonal=1).sum().item() == 0
    # Diagonal and below must be all 1s
    assert mask[0, 0].tril().sum().item() == t * (t + 1) / 2
