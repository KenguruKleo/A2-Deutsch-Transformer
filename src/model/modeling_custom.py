import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .configuration_custom import DeutschA2Config
from .model import TransformerModel
from .tokenizer import Tokenizer
import os

class DeutschA2Model(PreTrainedModel):
    config_class = DeutschA2Config
    _tied_weights_keys = ["model.lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = None
        
        self.model = TransformerModel(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff
        )
        
        # Robust tokenizer initialization
        self._init_tokenizer()
            
        self.post_init()

    def _init_tokenizer(self):
        """Attempts to find and load vocab.json from multiple possible locations."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, "vocab.json"),
            "vocab.json",
            "/app/vocab.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    self.tokenizer = Tokenizer(path)
                    print(f"✅ Tokenizer loaded from: {path}")
                    return
                except Exception as e:
                    print(f"⚠️ Failed to load vocab from {path}: {e}")

        # Fallback for Hugging Face Hub/Spaces: try to download if not found locally
        try:
            from huggingface_hub import hf_hub_download
            print("🌐 Attempting to download vocab.json from Hub...")
            vocab_path = hf_hub_download(repo_id=self.config._name_or_path, filename="vocab.json")
            self.tokenizer = Tokenizer(vocab_path)
            print(f"✅ Tokenizer loaded from Hub: {vocab_path}")
        except Exception as e:
            print(f"❌ Could not initialize tokenizer: {e}")

    def get_input_embeddings(self):
        return self.model.token_emb

    def set_input_embeddings(self, value):
        self.model.token_emb = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def tie_weights(self, *args, **kwargs):
        super().tie_weights(*args, **kwargs)
        if getattr(self.config, "weight_tying", True):
            self.model.lm_head.weight = self.model.token_emb.weight

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids)

    def generate(self, input_ids, max_new_tokens=64, temperature=0.7, top_k=50):
        self.eval()
        device = input_ids.device
        
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.model.max_seq_len else input_ids[:, -self.model.max_seq_len:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
            if next_token.item() == 2: # <EOS>
                break
                
        return input_ids
