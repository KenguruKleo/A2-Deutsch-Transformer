"""
generate.py — CLI inference for A2 Deutsch Grammar Tutor v2.1 (Standard HF BART).

Uses BartForConditionalGeneration.generate() — the standard HF generation pipeline.
No custom autoregressive loop, no manual BOS/EOS handling.

Usage:
    python -m src.generate --text "Ich habe nach Berlin gefahren."
    python -m src.generate --text "Wo du wohnst?" --model model_final
"""

import argparse
from transformers import BartForConditionalGeneration
from src.tokenizer.tokenizer import Tokenizer
from src.config import load_config, get_device, get_project_root


def generate_response(text: str, model, tokenizer, device, max_len=64) -> str:
    """
    Generate a grammar correction response using HF's model.generate().

    Pipeline:
      text → tokenizer.encode() → [token IDs + EOS]
           → Encoder → memory ∈ ℝ^{1×T_src×d}
           → Decoder (autoregressive, greedy) → output IDs
           → tokenizer.decode() → response text

    Args:
        text: Input German sentence.
        model: BartForConditionalGeneration in eval mode.
        tokenizer: Our Tokenizer wrapper.
        device: torch device string.
        max_len: Maximum output length.

    Returns:
        Generated response string.
    """
    model.eval()

    # Encode source: <BOS> + [tokens] + <EOS>
    import torch
    src_ids = tokenizer.encode(text, add_bos=True, add_eos=True, max_len=max_len)
    input_ids = torch.tensor([src_ids], dtype=torch.long, device=device)
    attention_mask = (input_ids != tokenizer.pad_id).long()

    # Generate using HF's built-in method (handles decoder_start_token, causal mask, etc.)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_len,
            num_beams=1,        # Greedy search (matches training)
            do_sample=False,
        )

    # Decode: skip special tokens (<BOS>, <EOS>, <PAD>)
    result = tokenizer.decode(output_ids[0].tolist(), skip_special=True)
    return result.strip()


def main():
    parser = argparse.ArgumentParser(description="A2 Deutsch Grammar Tutor v2.1 (HF BART)")
    parser.add_argument("--text", type=str, required=True, help="German sentence to check")
    parser.add_argument("--model", type=str, default="model_final", help="Path to model directory")
    args = parser.parse_args()

    config = load_config()
    device = get_device("auto")
    project_root = get_project_root()

    # Load tokenizer
    tokenizer = Tokenizer(project_root / "src/tokenizer/tokenizer.json")

    # Load model (HF format directory)
    model_dir = project_root / args.model
    if not model_dir.exists():
        print(f"❌ Model not found at {model_dir}. Please run training first.")
        return

    model = BartForConditionalGeneration.from_pretrained(str(model_dir))
    model = model.to(device)
    model.eval()
    print(f"✅ Loaded HF BART model from {model_dir}")

    # Generate
    response = generate_response(args.text, model, tokenizer, device, config.model.max_seq_len)

    print(f"\nInput:  {args.text}")
    print(f"Output: {response}\n")


if __name__ == "__main__":
    main()
