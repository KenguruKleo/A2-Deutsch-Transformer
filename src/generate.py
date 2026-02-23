import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.inference import load_model, generate_response


def generate(text: str, model_path=None, config_path=None) -> None:
    config = load_config(config_path)
    model, tokenizer, device = load_model(model_path, config)

    result = generate_response(text, model, tokenizer, config, device, max_new_tokens=50)

    print("\n" + "=" * 50)
    print(f"Input:  {text}")
    print("-" * 50)
    print(f"Output:\n{result}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Sentence to check")
    parser.add_argument("--model", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    generate(args.text, args.model)
