import torch
import argparse
from src.model.model import GrammarTransformer
from src.tokenizer.tokenizer import Tokenizer
from src.config import load_config, get_device, get_project_root

def generate_response(text: str, model, tokenizer, device, max_len=64):
    model.eval()
    
    # 1. Encode source text
    # We use BOS token to mark the start of the source in the encoder if needed, 
    # but usually for Seq2Seq simple tokens + EOS is enough.
    src_ids = tokenizer.encode(text, add_bos=False, add_eos=True, max_len=max_len)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    
    # 2. Generate using model's internal autoregressive loop
    with torch.no_grad():
        generated_ids = model.generate(
            src_tensor, 
            bos_id=tokenizer.bos_id, 
            eos_id=tokenizer.eos_id, 
            max_len=max_len
        )
    
    # 3. Decode result
    # generated_ids is [1, T], we remove BOS (first token)
    result = tokenizer.decode(generated_ids[0].tolist()[1:], skip_special=True)
    return result.strip()

def main():
    parser = argparse.ArgumentParser(description="A2 Deutsch Grammar Tutor (v2.0 Encoder-Decoder)")
    parser.add_argument("--text", type=str, required=True, help="German sentence to check")
    parser.add_argument("--model", type=str, default="model_final.pth", help="Path to model checkpoint")
    args = parser.parse_args()

    config = load_config()
    device = get_device("auto")
    project_root = get_project_root()
    
    # Load tokenizer
    tokenizer = Tokenizer(project_root / "src/tokenizer/tokenizer.json")
    
    # Init Model (v2.0 Architecture)
    model = GrammarTransformer(
        vocab_size=config.model.vocab_size,
        max_seq_len=config.model.max_seq_len,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_enc_layers=config.model.n_enc_layers,
        n_dec_layers=config.model.n_dec_layers,
        d_ff=config.model.d_ff,
    ).to(device)

    # Load Weights
    checkpoint_path = project_root / args.model
    if not checkpoint_path.exists():
        print(f"❌ Model not found at {checkpoint_path}. Please run training first.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Loaded v2.0 model from {args.model}")

    # Generate
    response = generate_response(args.text, model, tokenizer, device, config.model.max_seq_len)
    
    print(f"\nInput:  {args.text}")
    print(f"Output: {response}\n")

if __name__ == "__main__":
    main()
