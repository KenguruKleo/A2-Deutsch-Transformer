"""
modeling_custom.py — HuggingFace-compatible wrapper for TransformerModel.

DeutschA2Model inherits PreTrainedModel so the model can be:
  - loaded via AutoModelForCausalLM.from_pretrained()
  - used with HuggingFace generate() (or our custom generate override)
  - exported to safetensors via save_pretrained()

The tokenizer is intentionally NOT part of this class — it lives separately
as PreTrainedTokenizerFast and is loaded independently by the caller.
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from .configuration_custom import DeutschA2Config
from .model import TransformerModel


class DeutschA2Model(PreTrainedModel):
    config_class = DeutschA2Config

    # Weight tying: lm_head shares weights with token embedding wte
    _tied_weights_keys = ["model.lm_head.weight"]

    def __init__(self, config: DeutschA2Config):
        super().__init__(config)

        self.model = TransformerModel(
            vocab_size=config.vocab_size,
            max_seq_len=config.n_positions,   # GPT2Config uses n_positions
            d_model=config.n_embd,             # GPT2Config uses n_embd
            n_heads=config.n_head,             # GPT2Config uses n_head
            n_layers=config.n_layer,           # GPT2Config uses n_layer
            d_ff=config.n_inner,               # GPT2Config uses n_inner
            weight_tying=getattr(config, "weight_tying", True),
        )

        self.post_init()

    # ── Embedding tie hooks required by PreTrainedModel ──────────────────────

    def get_input_embeddings(self):
        return self.model.transformer["wte"]

    def set_input_embeddings(self, value):
        self.model.transformer["wte"] = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, input_ids, labels=None, **kwargs):
        """
        Args:
            input_ids: [batch, seq_len]
            labels:    optional [batch, seq_len] for loss computation

        Returns:
            If labels provided: (loss, logits)
            Otherwise:          logits  [batch, seq_len, vocab_size]
        """
        logits = self.model(input_ids)   # [B, T, vocab_size]

        loss = None
        if labels is not None:
            # Shift so we predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if loss is not None:
            return (loss, logits)
        return logits

    # ── Custom autoregressive generate (overrides HF generate) ───────────────

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        eos_token_id: int = 2,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive generation with temperature + top-k sampling.

        Args:
            input_ids:      [batch, prompt_len]
            max_new_tokens: max tokens to generate
            temperature:    sampling temperature
            top_k:          top-k filtering (0 = disabled)
            eos_token_id:   stop token (default 2 = <EOS>)

        Returns:
            [batch, prompt_len + generated_len]
        """
        self.eval()
        max_seq_len = self.model.max_seq_len

        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = input_ids[:, -max_seq_len:]
                logits = self.model(idx_cond)         # [B, T, vocab_size]
                logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat((input_ids, next_token), dim=1)

                if next_token.item() == eos_token_id:
                    break

        return input_ids
