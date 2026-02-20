import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .configuration_custom import DeutschA2Config
from .model import TransformerModel

class DeutschA2Model(PreTrainedModel):
    config_class = DeutschA2Config

    def __init__(self, config):
        super().__init__(config)
        self.model = TransformerModel(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff
        )

    def forward(self, input_ids, **kwargs):
        # Hugging Face usually passes 'input_ids'
        return self.model(input_ids)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
