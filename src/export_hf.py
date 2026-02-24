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

    # 1. Create config.json (standard GPT-2 format)
    hf_config = {
        "architectures": ["GPT2LMHeadModel"],
        "vocab_size": config_data["model"]["vocab_size"],
        "n_positions": config_data["model"]["max_seq_len"],
        "n_ctx": config_data["model"]["max_seq_len"],
        "n_embd": config_data["model"]["d_model"],
        "n_layer": config_data["model"]["n_layers"],
        "n_head": config_data["model"]["n_heads"],
        "n_inner": config_data["model"]["d_ff"],
        "activation_function": "gelu_new",
        "model_type": "gpt2",
        "torch_dtype": "float32",
        "tie_word_embeddings": config_data["model"].get("weight_tying", True),
    }

    config_out = output_dir / "config.json"
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)
    print(f"✅ Saved config.json")

    # 2. Convert state_dict to Safetensors
    print(f"📦 Converting {model_path} to safetensors...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]

    tensors = {k: v.cpu().clone().contiguous() for k, v in state_dict.items()}

    safetensors_out = output_dir / "model.safetensors"
    save_file(tensors, safetensors_out)
    print(f"✅ Saved model.safetensors")

    # 3. Copy model source files
    src = _PROJECT_ROOT / "src"
    shutil.copy(src / "model/model.py",                 output_dir / "model.py")
    shutil.copy(src / "model/modeling_custom.py",       output_dir / "modeling_custom.py")
    shutil.copy(src / "model/configuration_custom.py",  output_dir / "configuration_custom.py")
    shutil.copy(src / "tokenizer/tokenizer.py",         output_dir / "tokenizer.py")
    print("✅ Copied model source files")

    # 4. Save HuggingFace-compatible tokenizer (PreTrainedTokenizerFast)
    #    This makes the tokenizer recognized as GPT-2 style on the HF Hub.
    tok_json = src / "tokenizer/tokenizer.json"
    if not tok_json.exists():
        print("⚠️  tokenizer.json not found — run: python src/tokenizer/train_tokenizer.py")
    else:
        from transformers import PreTrainedTokenizerFast

        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tok_json),
            bos_token="<BOS>",
            eos_token="<EOS>",
            pad_token="<PAD>",
            unk_token="<UNK>",
        )
        hf_tokenizer.save_pretrained(str(output_dir))
        print("✅ Saved HF tokenizer (PreTrainedTokenizerFast)")

        # save_pretrained writes tokenizer_config.json with model_max_length=1e30 — fix it
        tc_path = output_dir / "tokenizer_config.json"
        with open(tc_path) as f:
            tc = json.load(f)
        tc["model_max_length"] = config_data["model"]["max_seq_len"]
        tc["tokenizer_class"]  = "PreTrainedTokenizerFast"
        with open(tc_path, "w") as f:
            json.dump(tc, f, indent=2)
        print("✅ Patched tokenizer_config.json (model_max_length, tokenizer_class)")

    print("\n🎉 Export complete! Folder 'hf_export' is ready for upload.")
    print(f"   Files: {sorted(p.name for p in output_dir.iterdir())}")

if __name__ == "__main__":
    export_to_hf()
