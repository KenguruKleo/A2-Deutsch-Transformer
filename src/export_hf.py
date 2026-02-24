import torch
import yaml
import json
import os
import shutil
from pathlib import Path
from safetensors.torch import save_file

_PROJECT_ROOT = Path(__file__).parent.parent

def export_to_hf(model_path=None, config_path=None, output_dir=None):
    if model_path is None:
        model_path = _PROJECT_ROOT / "model_final.pth"
    if config_path is None:
        config_path = _PROJECT_ROOT / "config.yaml"
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "hf_export"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Loading config from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # 1. Create config.json
    hf_config = {
        "architectures": ["DeutschA2Model"],
        "vocab_size": config_data["model"]["vocab_size"],
        "max_seq_len": config_data["model"]["max_seq_len"],
        "hidden_size": config_data["model"]["d_model"],
        "num_hidden_layers": config_data["model"]["n_layers"],
        "num_attention_heads": config_data["model"]["n_heads"],
        "intermediate_size": config_data["model"]["d_ff"],
        "model_type": "deutsch_a2_transformer",
        "torch_dtype": "float32",
        "weight_tying": config_data["model"].get("weight_tying", True),
        "auto_map": {
            "AutoConfig": "configuration_custom.DeutschA2Config",
            "AutoModel": "modeling_custom.DeutschA2Model",
            "AutoModelForCausalLM": "modeling_custom.DeutschA2Model"
        }
    }

    config_out = os.path.join(output_dir, "config.json")
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)
    print(f"✅ Saved HF config to {config_out}")

    # 2. Convert state_dict to Safetensors
    print(f"📦 Converting {model_path} to safetensors...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]

    # In our wrapper DeutschA2Model, the TransformerModel is under self.model
    # So we must prefix all keys with 'model.'
    tensors = {f"model.{k}": v.cpu().clone().contiguous() for k, v in state_dict.items()}
    
    safetensors_out = os.path.join(output_dir, "model.safetensors")
    save_file(tensors, safetensors_out)
    print(f"✅ Saved weights to {safetensors_out}")

    # 3. Copy Source files and Vocabulary
    # Files needed for HF to reconstruct the model code
    src = _PROJECT_ROOT / "src"
    shutil.copy(src / "model/model.py", output_dir / "model.py")
    shutil.copy(src / "model/modeling_custom.py", output_dir / "modeling_custom.py")
    shutil.copy(src / "model/configuration_custom.py", output_dir / "configuration_custom.py")
    shutil.copy(src / "tokenizer/tokenizer.py", output_dir / "tokenizer.py")

    tok_in = src / "tokenizer/tokenizer.json"
    tok_out = output_dir / "tokenizer.json"
    if tok_in.exists():
        shutil.copy(tok_in, tok_out)
        print(f"✅ Copied tokenizer.json to {tok_out}")
    else:
        print("⚠️  tokenizer.json not found — run: python src/tokenizer/train_tokenizer.py")

    print("\n🎉 Custom Model Export complete! Folder 'hf_export' is ready for upload.")

if __name__ == "__main__":
    export_to_hf()
