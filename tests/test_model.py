import torch
import sys
from pathlib import Path

# Add src to path to import the model
sys.path.append(str(Path(__file__).parent.parent))

from src.model.model import TransformerModel

def test_model_dimensions():
    print("🚀 Test 1: Checking output dimensions...")
    
    # Parameters as in our config
    v, t, d, h, l, f = 4000, 64, 128, 4, 4, 512
    batch_size = 2
    
    model = TransformerModel(v, t, d, h, l, f)
    
    # Fake input data (2 sentences of 64 tokens each)
    dummy_input = torch.randint(0, v, (batch_size, t))
    
    # Forward pass
    logits = model(dummy_input)
    
    print(f"Input: {dummy_input.shape}")
    print(f"Output (Logits): {logits.shape}")
    
    expected_shape = (batch_size, t, v)
    assert logits.shape == expected_shape, f"Error! Expected {expected_shape}, got {logits.shape}"
    print("✅ Dimension test passed!")

def test_device_compatibility():
    print("\n🚀 Test 2: Checking device compatibility...")
    
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    
    print(f"Using device: {device}")
    
    v, t, d, h, l, f = 4000, 64, 128, 4, 4, 512
    model = TransformerModel(v, t, d, h, l, f).to(device)
    
    dummy_input = torch.randint(0, v, (4, t)).to(device)
    
    try:
        logits = model(dummy_input)
        print(f"✅ Model successfully runs on {device}!")
    except Exception as e:
        print(f"❌ Error on device {device}: {e}")

if __name__ == "__main__":
    try:
        test_model_dimensions()
        test_device_compatibility()
        print("\n🎉 All model tests passed successfully!")
    except Exception as e:
        print(f"\n💀 Tests failed: {e}")
