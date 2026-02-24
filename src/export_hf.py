import torch
import os
import json
from pathlib import Path
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizerFast
from src.config import load_config, get_project_root
from src.model.model import GrammarTransformer

def export_to_bart():
    config = load_config()
    project_root = get_project_root()
    
    # 1. Load our custom model
    checkpoint_path = project_root / "model_final.pth"
    if not checkpoint_path.exists():
        print("❌ model_final.pth not found!")
        return

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    custom_model = GrammarTransformer(
        vocab_size=config.model.vocab_size,
        max_seq_len=config.model.max_seq_len,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_enc_layers=config.model.n_enc_layers,
        n_dec_layers=config.model.n_dec_layers,
        d_ff=config.model.d_ff,
    )
    custom_model.load_state_dict(checkpoint["model_state_dict"])
    custom_model.eval()
    custom_sd = custom_model.state_dict()

    # 2. Prepare Tokenizer FIRST to get the correct vocabulary size
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
        model_max_length=config.model.max_seq_len
    )
    
    # BART tokenizer often adds extra special tokens like <s>, </s>, <mask>.
    # We must ensure the model's vocab_size matches len(hf_tokenizer).
    actual_vocab_size = len(hf_tokenizer)
    print(f"📊 Tokenizer vocab size: {actual_vocab_size} (Source model: {config.model.vocab_size})")

    # 3. Create BART config matching our architecture
    bart_config = BartConfig(
        vocab_size=actual_vocab_size,
        d_model=config.model.d_model,
        encoder_layers=config.model.n_enc_layers,
        decoder_layers=config.model.n_dec_layers,
        encoder_attention_heads=config.model.n_heads,
        decoder_attention_heads=config.model.n_heads,
        encoder_ffn_dim=config.model.d_ff,
        decoder_ffn_dim=config.model.d_ff,
        max_position_embeddings=config.model.max_seq_len,
        dropout=0.1,
        attention_dropout=0.1,
        activation_function="gelu",
        pad_token_id=hf_tokenizer.pad_token_id,
        bos_token_id=hf_tokenizer.bos_token_id,
        eos_token_id=hf_tokenizer.eos_token_id,
        decoder_start_token_id=hf_tokenizer.bos_token_id,
        is_encoder_decoder=True,
        tie_word_embeddings=True
    )

    # 4. Create HF BART model and map weights
    hf_model = BartForConditionalGeneration(bart_config)
    state_dict = hf_model.state_dict()
    
    def copy_weight(hf_key, custom_key):
        if custom_key in custom_sd:
            custom_w = custom_sd[custom_key]
            hf_w = state_dict[hf_key]
            
            # Special handling for embedding resizing
            if custom_w.shape != hf_w.shape:
                # Copy only the overlapping part
                min_dim0 = min(custom_w.shape[0], hf_w.shape[0])
                if len(custom_w.shape) > 1:
                    state_dict[hf_key][:min_dim0, :].copy_(custom_w[:min_dim0, :])
                else:
                    state_dict[hf_key][:min_dim0].copy_(custom_w[:min_dim0])
            else:
                state_dict[hf_key].copy_(custom_w)
            return True
        return False

    # Shared / Embeddings
    copy_weight("model.shared.weight", "shared.weight")
    copy_weight("model.encoder.embed_tokens.weight", "shared.weight")
    copy_weight("model.decoder.embed_tokens.weight", "shared.weight")
    copy_weight("lm_head.weight", "shared.weight")

    # Encoder Layers
    for i in range(config.model.n_enc_layers):
        pfx = f"model.encoder.layers.{i}."
        cpfx = f"encoder_layers.{i}."
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            copy_weight(pfx + f"self_attn.{proj}.weight", cpfx + f"self_attn.{proj}.weight")
            copy_weight(pfx + f"self_attn.{proj}.bias", cpfx + f"self_attn.{proj}.bias")
        copy_weight(pfx + "self_attn_layer_norm.weight", cpfx + "self_attn_layer_norm.weight")
        copy_weight(pfx + "self_attn_layer_norm.bias", cpfx + "self_attn_layer_norm.bias")
        copy_weight(pfx + "fc1.weight", cpfx + "fc1.weight")
        copy_weight(pfx + "fc1.bias", cpfx + "fc1.bias")
        copy_weight(pfx + "fc2.weight", cpfx + "fc2.weight")
        copy_weight(pfx + "fc2.bias", cpfx + "fc2.bias")
        copy_weight(pfx + "final_layer_norm.weight", cpfx + "final_layer_norm.weight")
        copy_weight(pfx + "final_layer_norm.bias", cpfx + "final_layer_norm.bias")

    # Decoder Layers
    for i in range(config.model.n_dec_layers):
        pfx = f"model.decoder.layers.{i}."
        cpfx = f"decoder_layers.{i}."
        # Self-Attention
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            copy_weight(pfx + f"self_attn.{proj}.weight", cpfx + f"self_attn.{proj}.weight")
            copy_weight(pfx + f"self_attn.{proj}.bias", cpfx + f"self_attn.{proj}.bias")
        copy_weight(pfx + "self_attn_layer_norm.weight", cpfx + "self_attn_layer_norm.weight")
        copy_weight(pfx + "self_attn_layer_norm.bias", cpfx + "self_attn_layer_norm.bias")
        # Cross-Attention
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            copy_weight(pfx + f"encoder_attn.{proj}.weight", cpfx + f"encoder_attn.{proj}.weight")
            copy_weight(pfx + f"encoder_attn.{proj}.bias", cpfx + f"encoder_attn.{proj}.bias")
        copy_weight(pfx + "encoder_attn_layer_norm.weight", cpfx + "encoder_attn_layer_norm.weight")
        copy_weight(pfx + "encoder_attn_layer_norm.bias", cpfx + "encoder_attn_layer_norm.bias")
        # FFN
        copy_weight(pfx + "fc1.weight", cpfx + "fc1.weight")
        copy_weight(pfx + "fc1.bias", cpfx + "fc1.bias")
        copy_weight(pfx + "fc2.weight", cpfx + "fc2.weight")
        copy_weight(pfx + "fc2.bias", cpfx + "fc2.bias")
        copy_weight(pfx + "final_layer_norm.weight", cpfx + "final_layer_norm.weight")
        copy_weight(pfx + "final_layer_norm.bias", cpfx + "final_layer_norm.bias")

    # 5. Save everything to hf_export folder
    export_dir = project_root / "hf_export"
    os.makedirs(export_dir, exist_ok=True)
    
    hf_model.load_state_dict(state_dict)
    hf_model.save_pretrained(export_dir)
    hf_tokenizer.save_pretrained(export_dir)
    
    # Fix tokenizer_config.json class name
    config_path = export_dir / "tokenizer_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        t_config = json.load(f)
    t_config["tokenizer_class"] = "BartTokenizer"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(t_config, f, indent=2, ensure_ascii=False)

    print(f"✅ Full export completed to {export_dir}")

if __name__ == "__main__":
    export_to_bart()
