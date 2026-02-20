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

    # 6. Training Loop with early stopping
    epochs = config.training.epochs
    patience = getattr(config.training, "early_stopping_patience", 0) or 0
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    print(f"📊 Starting training for up to {epochs} epochs" + (f" (early stop if no val improvement for {patience} epochs)" if patience else "") + "...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch = batch.to(device)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(validation_loader)
        print(f"✨ Epoch {epoch+1} finished. Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience and epochs_without_improvement >= patience:
            print(f"⏹ Early stopping: no val improvement for {patience} epochs (best Val Loss: {best_val_loss:.4f})")
            break

    # 7. Save best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    save_path = Path("model_final.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": config.model.vocab_size,
    }, save_path)
    print(f"📦 Best model saved to {save_path}")

if __name__ == "__main__":
    train()
