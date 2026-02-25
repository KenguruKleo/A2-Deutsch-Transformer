"""
export_hf.py — Export trained model to Hugging Face Hub format.

Since v2.1 trains directly with BartForConditionalGeneration,
the model is ALREADY in HF format. This script simply copies it
to the hf_export/ directory alongside the tokenizer.

No weight mapping, no offset hacks, no identity LayerNorm fixes needed.
"""

import os
import json
from pathlib import Path
from transformers import BartForConditionalGeneration, BartTokenizerFast
from src.config import load_config, get_project_root


def export_to_hf():
    config = load_config()
    project_root = get_project_root()

    # ── 1. Load the trained model (already HF format) ──
    model_dir = project_root / "model_final"
    if not model_dir.exists():
        print(f"❌ {model_dir} not found! Train the model first.")
        return

    print(f"📥 Loading model from {model_dir}...")
    model = BartForConditionalGeneration.from_pretrained(str(model_dir))
    model.eval()

    # ── 2. Load tokenizer ──
    tok_json = project_root / "src/tokenizer/tokenizer.json"
    if not tok_json.exists():
        print("❌ tokenizer.json not found!")
        return

    hf_tokenizer = BartTokenizerFast(
        tokenizer_file=str(tok_json),
        bos_token="<BOS>",
        eos_token="<EOS>",
        pad_token="<PAD>",
        unk_token="<UNK>",
        model_max_length=config.model.max_seq_len,
    )

    print(f"📊 Model vocab: {model.config.vocab_size}, Tokenizer vocab: {len(hf_tokenizer)}")

    # Resize embeddings if tokenizer has extra tokens (e.g. <mask>, <s>, </s>)
    if len(hf_tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(hf_tokenizer))
        print(f"📐 Resized model embeddings to {len(hf_tokenizer)}")

    # ── 3. Save to hf_export/ ──
    export_dir = project_root / "hf_export"
    os.makedirs(export_dir, exist_ok=True)

    model.save_pretrained(export_dir)
    hf_tokenizer.save_pretrained(export_dir)

    # Fix tokenizer_config.json class name
    config_path = export_dir / "tokenizer_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        t_config = json.load(f)
    t_config["tokenizer_class"] = "BartTokenizer"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(t_config, f, indent=2, ensure_ascii=False)

    print(f"✅ Export completed to {export_dir}")
    print(f"   Upload with: hf upload KenguruKleo/deutsch-a2-tutor-model {export_dir} . --repo-type model")


if __name__ == "__main__":
    export_to_hf()
