import torch
import torch.nn.functional as F
import yaml
import argparse
from src.model.model import TransformerModel
from src.tokenizer.tokenizer import Tokenizer
from src.config import load_config, get_device

def generate(text, model_path="model_final.pth", config_path="config.yaml"):
    # 1. Load Config
    config = load_config(config_path)

    device = get_device(getattr(config.training, "device", None))
    
    # 2. Load Tokenizer & Model
    tokenizer = Tokenizer("src/tokenizer/vocab.json")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = TransformerModel(
        vocab_size=config.model.vocab_size,
        max_seq_len=config.model.max_seq_len,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Preparation
    # We add BOS to help the model start
    input_ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    max_new_tokens = 50
    temperature = config.generation.temperature
    top_k = config.generation.top_k

    # 4. Auto-regressive generation
    for _ in range(max_new_tokens):
        # We take only the last max_seq_len tokens
        idx_cond = input_tensor[:, -config.model.max_seq_len:]
        
        # Get predictions
        with torch.no_grad():
            logits = model(idx_cond) # [batch=1, seq_len, vocab_size]
        
        # Take logits for the last token only
        logits = logits[:, -1, :] / temperature
        
        # Optional: Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from distribution
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        input_tensor = torch.cat((input_tensor, next_token), dim=1)
        
        # Stop if EOS reached
        if next_token.item() == tokenizer.eos_id:
            break

    # 5. Decode
    # We only decode what was generated AFTER the initial input
    # But usually, it's easier to decode the whole thing and strip the input prefix
    full_ids = input_tensor[0].tolist()
    # Remove BOS (index 0)
    if full_ids[0] == tokenizer.bos_id:
        full_ids = full_ids[1:]
        
    result = tokenizer.decode(full_ids)
    print("\n" + "="*50)
    print(f"Input:  {text}")
    print("-" * 50)
    # The result contains the input text as well
    print(f"Output:\n{result}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Sentence to check")
    parser.add_argument("--model", type=str, default="model_final.pth", help="Path to checkpoint")
    args = parser.parse_args()
    
    generate(args.text, args.model)
