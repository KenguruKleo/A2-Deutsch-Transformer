import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from tqdm import tqdm
from src.model.model import GrammarTransformer
from src.tokenizer.tokenizer import Tokenizer
from src.config import load_config, get_device, get_project_root

class Seq2SeqDataset(Dataset):
    """
    Dataset for v2.0 Encoder-Decoder (Seq2Seq).
    Returns (src_ids, tgt_ids, label_ids).
    """
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
        
        # 1. Source (Encoder Input): Input sentence with EOS
        src_ids = self.tokenizer.encode(item['input'], add_bos=False, add_eos=True, max_len=self.max_len)
        src_ids = self.tokenizer.pad_sequence(src_ids, max_len=self.max_len)
        
        # 2. Target Input (Decoder Input): <BOS> + Output
        # We don't add EOS here because it's for generation
        tgt_text = item['output']
        tgt_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(tgt_text, add_bos=False, add_eos=False, max_len=self.max_len-1)
        tgt_ids = self.tokenizer.pad_sequence(tgt_ids, max_len=self.max_len)
        
        # 3. Label (Expected Output): Output + <EOS>
        label_ids = self.tokenizer.encode(tgt_text, add_bos=False, add_eos=True, max_len=self.max_len)
        label_ids = self.tokenizer.pad_sequence(label_ids, max_len=self.max_len)
        
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long)
        )

def create_masks(src_ids, tgt_ids, pad_id, device):
    """
    Creates masks for Encoder and Decoder.
    """
    # 1. Source Mask: [batch, 1, 1, src_len]
    src_mask = (src_ids != pad_id).unsqueeze(1).unsqueeze(2)
    
    # 2. Target Mask: [batch, 1, tgt_len, tgt_len]
    batch_size, tgt_len = tgt_ids.shape
    causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) # [1, 1, tgt_len, tgt_len]
    
    padding_mask = (tgt_ids != pad_id).unsqueeze(1).unsqueeze(2)
    tgt_mask = causal_mask & padding_mask
    
    return src_mask, tgt_mask

import argparse

def train():
    # 0. Parse CLI Arguments
    parser = argparse.ArgumentParser(description="Train A2 Deutsch Grammar Transformer")
    parser.add_argument("--epochs", type=int, help="Override number of epochs from config")
    parser.add_argument("--continue", dest="continue_train", action="store_true", help="Continue training from model_final.pth")
    args = parser.parse_args()

    # 1. Load Config
    config = load_config()

    # 2. Set Device
    device = get_device(config.training.device)
    print(f"🚀 Training v2.0 on device: {device}")

    # 3. Init Tokenizer & Model
    project_root = get_project_root()
    tokenizer = Tokenizer(project_root / "src/tokenizer/tokenizer.json")
    
    model = GrammarTransformer(
        vocab_size=config.model.vocab_size,
        max_seq_len=config.model.max_seq_len,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_enc_layers=config.model.n_enc_layers,
        n_dec_layers=config.model.n_dec_layers,
        d_ff=config.model.d_ff,
    ).to(device)

    # 4. Load existing weights if requested
    if args.continue_train:
        checkpoint_path = project_root / "model_final.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"🔄 Resuming training from {checkpoint_path}")
        else:
            print(f"⚠️  {checkpoint_path} not found. Starting from scratch.")

    # 5. Prepare Data
    train_ds = Seq2SeqDataset(config.data.train_path, tokenizer, config.model.max_seq_len)
    validation_ds = Seq2SeqDataset(config.data.val_path, tokenizer, config.model.max_seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_ds, batch_size=config.training.batch_size)

    # 6. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.training.learning_rate))
    
    # Decision tokens weighting for classification part
    decision_w = config.training.decision_token_weight
    decision_token_ids: set[int] = set()
    for tok_str in ["✅", "❌"]:
        tid = tokenizer.token_to_id.get(tok_str)
        if tid is not None:
            decision_token_ids.add(tid)
            
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, reduction="none")

    # 7. Training Loop with early stopping
    epochs = args.epochs if args.epochs is not None else config.training.epochs
    patience = config.training.early_stopping_patience
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    print(f"📊 Starting training for up to {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for src_ids, tgt_ids, labels in pbar:
            src_ids, tgt_ids, labels = src_ids.to(device), tgt_ids.to(device), labels.to(device)
            
            # Create Masks
            src_mask, tgt_mask = create_masks(src_ids, tgt_ids, tokenizer.pad_id, device)
            
            # Forward: Encoder (src_ids) -> Decoder (tgt_ids)
            logits = model(src_ids, tgt_ids, src_mask, tgt_mask)
            
            # Loss Calculation [B, T, V] -> reshape for [B*T, V]
            per_token_loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            if decision_w > 1.0 and decision_token_ids:
                flat_targets = labels.reshape(-1)
                weights = torch.ones_like(per_token_loss)
                for tid in decision_token_ids:
                    weights[flat_targets == tid] = decision_w
                per_token_loss = per_token_loss * weights
                
            loss = per_token_loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src_ids, tgt_ids, labels in validation_loader:
                src_ids, tgt_ids, labels = src_ids.to(device), tgt_ids.to(device), labels.to(device)
                src_mask, tgt_mask = create_masks(src_ids, tgt_ids, tokenizer.pad_id, device)
                logits = model(src_ids, tgt_ids, src_mask, tgt_mask)
                per_token_loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                val_loss += per_token_loss.mean().item()

        avg_val_loss = val_loss / len(validation_loader)
        print(f"✨ Epoch {epoch+1} finished. Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience and epochs_without_improvement >= patience:
            print(f"⏹ Early stopping at epoch {epoch+1}")
            break

    # 8. Save best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    save_path = project_root / "model_final.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": config.model.vocab_size,
        "arch": "encoder_decoder"  # Marker for v2.0
    }, save_path)
    print(f"📦 Best v2.0 model saved to {save_path}")

if __name__ == "__main__":
    train()
