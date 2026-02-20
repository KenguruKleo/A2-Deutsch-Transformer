import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import json
from pathlib import Path
from tqdm import tqdm
from src.model.model import TransformerModel
from src.tokenizer.tokenizer import Tokenizer
from src.config import load_config, get_device

class GrammarDataset(Dataset):
    """Dataset for training the geometry model on grammar corrections."""
    def __init__(self, data_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        
        # Combine input and output for Causal LM training
        # Target: "input <EOS> output <EOS>"
        # We train the model to predict the next token.
        full_text = f"{item['input']} {item['output']}"
        ids = self.tokenizer.encode(full_text, add_bos=True, add_eos=True, max_len=self.max_len)
        ids = self.tokenizer.pad_sequence(ids, max_len=self.max_len)
        
        # Returns tensor of shape: [max_len]
        return torch.tensor(ids, dtype=torch.long)

def train():
    # 1. Load Config
    config = load_config()

    # 2. Set Device (auto: cuda → mps → cpu)
    device = get_device(getattr(config.training, "device", None))
    print(f"🚀 Training on device: {device}")

    # 3. Init Tokenizer & Model
    tokenizer = Tokenizer("src/tokenizer/vocab.json")
    
    model = TransformerModel(
        vocab_size=config.model.vocab_size,
        max_seq_len=config.model.max_seq_len,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff
    ).to(device)

    # 4. Prepare Data
    train_ds = GrammarDataset(config.data.train_path, tokenizer, config.model.max_seq_len)
    validation_ds = GrammarDataset(config.data.val_path, tokenizer, config.model.max_seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_ds, batch_size=config.training.batch_size)

    # 5. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.training.learning_rate))
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # 6. Training Loop
    epochs = config.training.epochs
    
    print(f"📊 Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            # batch: [batch_size, seq_len]
            batch = batch.to(device)
            
            # For Causal LM: inputs are [0...T-1], targets are [1...T]
            inputs = batch[:, :-1]  # shape: [batch_size, seq_len - 1]
            targets = batch[:, 1:]   # shape: [batch_size, seq_len - 1]
            
            # Forward pass
            logits = model(inputs) # shape: [batch_size, seq_len - 1, vocab_size]
            
            # Flatten for loss calculation:
            # Logits: [batch_size * (seq_len - 1), vocab_size]
            # Targets: [batch_size * (seq_len - 1)]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in validation_loader:
                # batch: [batch_size, seq_len]
                batch = batch.to(device)

                inputs = batch[:, :-1] # shape: [batch_size, seq_len - 1]
                targets = batch[:, 1:]  # shape: [batch_size, seq_len - 1]

                logits = model(inputs) # shape: [batch_size, seq_len - 1, vocab_size]

                # Flatten for loss calculation
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(validation_loader)
        print(f"✨ Epoch {epoch+1} finished. Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # 7. Save Model
    save_path = Path("model_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': config.model.vocab_size
    }, save_path)
    print(f"📦 Model saved to {save_path}")

if __name__ == "__main__":
    train()
