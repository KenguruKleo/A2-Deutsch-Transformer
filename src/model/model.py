"""
model.py — A2 Deutsch Grammar Tutor v2.1 (Standard HF BART)

Uses HF's BartForConditionalGeneration directly.
No custom layers — training, inference, and export all share the same HF code.

Architecture overview (matrices & dimensions):
───────────────────────────────────────────────
  E  = Shared Embedding         ∈ ℝ^{V × d}        (token → vector)
  P  = Learned Positional Emb   ∈ ℝ^{(T+offset) × d}  (position → vector)
  LN = LayerNorm(d)             after embed sum

  Encoder Layer (× N_enc):
    Self-Attention:
      W_Q, W_K, W_V ∈ ℝ^{d × d}   →  split into H heads of d_k = d/H
      W_O            ∈ ℝ^{d × d}   →  concat + project
      LN₁(d)
    FFN:
      W₁ ∈ ℝ^{d × d_ff}   +  GELU activation
      W₂ ∈ ℝ^{d_ff × d}
      LN₂(d)

  Decoder Layer (× N_dec):
    Masked Self-Attention:  same as Encoder Self-Attn but causal (lower-triangular mask)
    Cross-Attention:        Q from decoder, K/V from encoder output
    FFN:                    same structure as encoder FFN

  LM Head  = Linear(d, V, bias=False)  — tied with E (shared weights)

Hyperparameters (config.yaml):
  V = 8000, d = 256, H = 4, d_k = 64, d_ff = 512
  N_enc = 3, N_dec = 3, T = 64
"""

from transformers import BartConfig, BartForConditionalGeneration, BartTokenizerFast
from pathlib import Path


def create_bart_config(config) -> BartConfig:
    """
    Build a BartConfig from our config.yaml structure.

    Args:
        config: Loaded config object with config.model.* attributes.

    Returns:
        BartConfig ready for model initialization.

    Matrix dimensions defined by this config:
        E ∈ ℝ^{vocab_size × d_model}  = ℝ^{8000 × 256}
        P ∈ ℝ^{(max_seq_len + 2) × d_model}  = ℝ^{66 × 256}  (offset=2 for BART)
        W_Q, W_K, W_V, W_O ∈ ℝ^{d_model × d_model} = ℝ^{256 × 256}
        W₁ ∈ ℝ^{d_model × d_ff}  = ℝ^{256 × 512}
        W₂ ∈ ℝ^{d_ff × d_model}  = ℝ^{512 × 256}
    """
    return BartConfig(
        # ── Vocabulary & Embeddings ──
        vocab_size=config.model.vocab_size,       # V = 8000 — E ∈ ℝ^{V×d}
        d_model=config.model.d_model,             # d = 256  — hidden dimension throughout

        # ── Encoder ──
        encoder_layers=config.model.n_enc_layers,          # N_enc = 3
        encoder_attention_heads=config.model.n_heads,      # H = 4  → d_k = d/H = 64
        encoder_ffn_dim=config.model.d_ff,                 # d_ff = 512  → W₁ ∈ ℝ^{d×d_ff}

        # ── Decoder ──
        decoder_layers=config.model.n_dec_layers,          # N_dec = 3
        decoder_attention_heads=config.model.n_heads,      # H = 4
        decoder_ffn_dim=config.model.d_ff,                 # d_ff = 512

        # ── Positional Embeddings ──
        max_position_embeddings=config.model.max_seq_len,  # T = 64  → P ∈ ℝ^{(T+2)×d}

        # ── Regularization ──
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,

        # ── Activation ──
        activation_function="gelu",       # FFN activation: GELU(x·W₁) · W₂

        # ── Embedding Behavior ──
        scale_embedding=False,            # Disable √d scaling (our training doesn't use it)
        tie_word_embeddings=True,         # LM Head shares weights with E

        # ── Architecture Flags ──
        is_encoder_decoder=True,

        # ── Special Token IDs (set later from tokenizer) ──
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,         # Decoder starts with <BOS>
    )


def create_model(config, tokenizer=None) -> BartForConditionalGeneration:
    """
    Create a fresh BartForConditionalGeneration model.

    If tokenizer is provided, special token IDs are synced from it,
    and vocab_size is adjusted to match len(tokenizer) (may differ
    from config due to extra special tokens added by HF).

    Args:
        config: Loaded config with config.model.* attributes.
        tokenizer: Optional BartTokenizerFast or Tokenizer instance.

    Returns:
        BartForConditionalGeneration — standard HF model, ready for training.
    """
    bart_config = create_bart_config(config)

    # Sync special token IDs from tokenizer if available
    if tokenizer is not None:
        # Our Tokenizer wrapper exposes .pad_id, .bos_id, .eos_id
        if hasattr(tokenizer, 'pad_id'):
            bart_config.pad_token_id = tokenizer.pad_id
            bart_config.bos_token_id = tokenizer.bos_id
            bart_config.eos_token_id = tokenizer.eos_id
            bart_config.decoder_start_token_id = tokenizer.bos_id
        # HF tokenizer API
        elif hasattr(tokenizer, 'pad_token_id'):
            bart_config.pad_token_id = tokenizer.pad_token_id
            bart_config.bos_token_id = tokenizer.bos_token_id
            bart_config.eos_token_id = tokenizer.eos_token_id
            bart_config.decoder_start_token_id = tokenizer.bos_token_id

        # Adjust vocab size to match tokenizer (HF may add extra tokens)
        tok_vocab = len(tokenizer) if hasattr(tokenizer, '__len__') else tokenizer.vocab_size
        if tok_vocab != bart_config.vocab_size:
            bart_config.vocab_size = tok_vocab

    # ── Create the model ──
    # All weight matrices are initialized by HF with N(0, 0.02):
    #   E ∈ ℝ^{V×d}, P ∈ ℝ^{(T+2)×d},
    #   W_Q, W_K, W_V, W_O ∈ ℝ^{d×d} per layer per head-group,
    #   W₁ ∈ ℝ^{d×d_ff}, W₂ ∈ ℝ^{d_ff×d} per layer,
    #   LN weights = 1, LN biases = 0 (per norm layer)
    model = BartForConditionalGeneration(bart_config)

    return model


def load_model_from_dir(model_dir: str | Path) -> BartForConditionalGeneration:
    """
    Load a trained model from a directory (HF format).

    The directory must contain:
      - config.json         (BartConfig)
      - model.safetensors   (weights)

    Args:
        model_dir: Path to directory with saved HF model.

    Returns:
        BartForConditionalGeneration in eval mode.
    """
    model = BartForConditionalGeneration.from_pretrained(str(model_dir))
    model.eval()
    return model
