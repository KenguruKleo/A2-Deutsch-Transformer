import torch
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.model import create_model
from src.config import load_config
from transformers import BartForConditionalGeneration


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def model(config):
    # Create the model using the HF BART wrapper
    # We pass tokenizer=None here, so it uses the raw config values
    return create_model(config, tokenizer=None)


def test_is_hf_model(model):
    """Ensure the created model is a standard huggingface BartForConditionalGeneration."""
    assert isinstance(model, BartForConditionalGeneration)


def test_output_shape(model, config):
    """
    Logits shape must be [batch, seq_len, vocab_size]
    when passing input_ids and decoder_input_ids to HF BART.
    """
    v = config.model.vocab_size
    t = config.model.max_seq_len
    batch_size = 2

    dummy_src = torch.randint(0, v, (batch_size, t))
    dummy_tgt = torch.randint(0, v, (batch_size, t))
    
    # HF BART returns a Seq2SeqLMOutput object
    outputs = model(input_ids=dummy_src, decoder_input_ids=dummy_tgt)
    logits = outputs.logits

    assert logits.shape == (batch_size, t, v)


def test_weight_tying(model):
    """
    Embedding and LM head must share the same weight tensor,
    which is standard practice in tie_word_embeddings=True.
    """
    # In HF BART:
    # embed_tokens = model.model.shared.weight
    # lm_head = model.lm_head.weight
    assert model.model.shared.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_config_mapping(model, config):
    """Ensure the BartConfig matches our config.yaml params."""
    assert model.config.vocab_size == config.model.vocab_size
    assert model.config.d_model == config.model.d_model
    assert model.config.encoder_layers == config.model.n_enc_layers
    assert model.config.decoder_layers == config.model.n_dec_layers
    assert model.config.encoder_attention_heads == config.model.n_heads
    assert model.config.decoder_attention_heads == config.model.n_heads
    assert model.config.encoder_ffn_dim == config.model.d_ff
    assert model.config.decoder_ffn_dim == config.model.d_ff
    assert model.config.max_position_embeddings == config.model.max_seq_len
    
    # Custom tweaks for our training logic
    assert getattr(model.config, "scale_embedding", True) is False
    assert model.config.tie_word_embeddings is True
