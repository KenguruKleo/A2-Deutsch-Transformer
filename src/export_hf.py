import torch
import os
from pathlib import Path
from transformers import BartConfig, BartForConditionalGeneration, PreTrainedTokenizerFast
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

    # 2. Create BART config matching our architecture
    bart_config = BartConfig(
        vocab_size=config.model.vocab_size,
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
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        is_encoder_decoder=True,
        tie_word_embeddings=True
    )

    # 3. Create HF BART model
    hf_model = BartForConditionalGeneration(bart_config)
    
    # 4. Map weights from custom to HF
    # Custom: shared.weight -> HF: model.shared.weight
    # Custom: encoder_layers -> HF: model.encoder.layers
    # Custom: decoder_layers -> HF: model.decoder.layers
    
    state_dict = hf_model.state_dict()
    custom_sd = custom_model.state_dict()
    
    def copy_weight(hf_key, custom_key):
        if custom_key in custom_sd:
            state_dict[hf_key].copy_(custom_sd[custom_key])
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
        
        # Self-Attention
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            copy_weight(pfx + f"self_attn.{proj}.weight", cpfx + f"self_attn.{proj}.weight")
            copy_weight(pfx + f"self_attn.{proj}.bias", cpfx + f"self_attn.{proj}.bias")
            
        copy_weight(pfx + "self_attn_layer_norm.weight", cpfx + "self_attn_layer_norm.weight")
        copy_weight(pfx + "self_attn_layer_norm.bias", cpfx + "self_attn_layer_norm.bias")
        
        # FFN (BART uses fc1/fc2 naming we already used)
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
        
        # Cross-Attention (Encoder-Attention)
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
    
    hf_model.save_pretrained(export_dir)
    print(f"✅ Model weights exported to {export_dir}")
    
    # 6. Copy Tokenizer
    tok_json = project_root / "src/tokenizer/tokenizer.json"
    if tok_json.exists():
        from transformers import BartTokenizerFast
        hf_tokenizer = BartTokenizerFast(
            tokenizer_file=str(tok_json),
            bos_token="<BOS>",
            eos_token="<EOS>",
            pad_token="<PAD>",
            unk_token="<UNK>",
            model_max_length=config.model.max_seq_len
        )
        hf_tokenizer.save_pretrained(export_dir)
        
        # Force BartTokenizer class name in config for professional looks
        import json
        config_path = export_dir / "tokenizer_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            t_config = json.load(f)
        t_config["tokenizer_class"] = "BartTokenizer"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(t_config, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Tokenizer exported to {export_dir} (with BartTokenizer class)")

if __name__ == "__main__":
    export_to_bart()
