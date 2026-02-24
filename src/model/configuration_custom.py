from transformers import GPT2Config


class DeutschA2Config(GPT2Config):
    """
    Config for A2 Deutsch Grammar Tutor.

    Inherits GPT2Config so HuggingFace treats the model as GPT-2 compatible.

    GPT-2 field mapping:
        n_embd      ← d_model
        n_layer     ← n_layers
        n_head      ← n_heads
        n_inner     ← d_ff
        n_positions ← max_seq_len
        n_ctx       ← max_seq_len
    """

    model_type = "gpt2"

    def __init__(
        self,
        vocab_size: int = 8000,
        n_positions: int = 64,
        n_embd: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        n_inner: int = 512,
        weight_tying: bool = True,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_ctx=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            **kwargs,
        )
        self.weight_tying = weight_tying
