import torch
import yaml
import json
import os
from safetensors.torch import save_file
from src.tokenizer.tokenizer import Tokenizer

def export_to_hf(model_path="model_final.pth", config_path="config.yaml", output_dir="hf_export"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"🚀 Loading config from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # 1. Create config.json
    # We use keys that are standard-ish for Transformer models
    hf_config = {
        "architectures": ["TransformerModel"],
        "vocab_size": config_data["model"]["vocab_size"],
        "max_position_embeddings": config_data["model"]["max_seq_len"],
        "hidden_size": config_data["model"]["d_model"],
        "num_hidden_layers": config_data["model"]["n_layers"],
        "num_attention_heads": config_data["model"]["n_heads"],
        "intermediate_size": config_data["model"]["d_ff"],
        "model_type": "deutsch_a2_transformer",
        "torch_dtype": "float32",
        "weight_tying": config_data["model"].get("weight_tying", True)
    }

    config_out = os.path.join(output_dir, "config.json")
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)
    print(f"✅ Saved HF config to {config_out}")

    # 2. Convert state_dict to Safetensors
    print(f"📦 Converting {model_path} to safetensors...")
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # Safetensors expects a flat dict of tensors
    # Since we use weight tying, some tensors share memory. 
    # We clone them to break memory sharing (safetensors requirement)
    tensors = {k: v.cpu().clone().contiguous() for k, v in state_dict.items()}
    
    safetensors_out = os.path.join(output_dir, "model.safetensors")
    save_file(tensors, safetensors_out)
    print(f"✅ Saved weights to {safetensors_out}")

    # 3. Copy/Save Vocabulary
    vocab_in = "src/tokenizer/vocab.json"
    vocab_out = os.path.join(output_dir, "vocab.json")
    if os.path.exists(vocab_in):
        import shutil
        shutil.copy(vocab_in, vocab_out)
        print(f"✅ Copied vocab.json to {vocab_out}")

    print("\n🎉 Export complete! You can now upload the 'hf_export' folder to Hugging Face Hub.")

if __name__ == "__main__":
    export_to_hf()
