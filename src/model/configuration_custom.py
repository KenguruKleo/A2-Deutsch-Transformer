from transformers import PretrainedConfig
from typing import Any

class DeutschA2Config(PretrainedConfig):
    model_type = "deutsch_a2_transformer"

    def __init__(
        self,
        vocab_size=4000,
        max_seq_len=64,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        weight_tying=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.weight_tying = weight_tying
        super().__init__(**kwargs)
