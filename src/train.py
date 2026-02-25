"""
train.py — Training loop for A2 Deutsch Grammar Tutor v2.1 (Standard HF BART).

Uses BartForConditionalGeneration directly. The model IS the HF model,
so saving/loading uses save_pretrained/from_pretrained (no weight mapping).

Data flow per batch:
  1. src_ids  → Encoder (with attention_mask for PAD)
  2. tgt_ids  → Decoder (with decoder_attention_mask + causal mask auto-applied)
  3. labels   → CrossEntropyLoss(ignore_index=pad_id)
  4. Logits shape: [B, T, V] where V = vocab_size
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from transformers import BartForConditionalGeneration

from src.model.model import create_model
from src.tokenizer.tokenizer import Tokenizer
from src.config import load_config, get_device, get_project_root


class Seq2SeqDataset(Dataset):
    """
    Dataset for Encoder-Decoder (Seq2Seq) training.

    Each example produces:
      - src_ids:          [tok₁, tok₂, ..., <EOS>, <PAD>, ...]     → Encoder input
      - attention_mask:   [1, 1, ..., 1, 0, ...]                   → Encoder mask (1=real, 0=pad)
      - decoder_input_ids: [<BOS>, tok₁, tok₂, ..., <PAD>, ...]   → Decoder input (teacher forcing)
      - labels:           [tok₁, tok₂, ..., <EOS>, -100, ...]     → Target (shifted, -100=ignore)
    """
    def __init__(self, data_path, tokenizer, max_len, pad_id):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = pad_id
        self.examples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]

        # ── Encoder Input: [tokens] + <EOS> + <PAD>... ──
        # The encoder sees the input sentence, terminated by EOS.
        src_ids = self.tokenizer.encode(item['input'], add_bos=False, add_eos=True, max_len=self.max_len)
        attention_mask = [1] * len(src_ids) + [0] * (self.max_len - len(src_ids))
        src_ids = self.tokenizer.pad_sequence(src_ids, max_len=self.max_len)

        # ── Decoder Input: <BOS> + [output tokens] + <PAD>... ──
        # Teacher forcing: decoder sees <BOS> followed by the target tokens.
        tgt_text = item['output']
        tgt_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(
            tgt_text, add_bos=False, add_eos=False, max_len=self.max_len - 1
        )
        decoder_attention_mask = [1] * len(tgt_ids) + [0] * (self.max_len - len(tgt_ids))
        tgt_ids = self.tokenizer.pad_sequence(tgt_ids, max_len=self.max_len)

        # ── Labels: [output tokens] + <EOS> + [-100]... ──
        # -100 tells CrossEntropyLoss to ignore padding positions.
        label_ids = self.tokenizer.encode(tgt_text, add_bos=False, add_eos=True, max_len=self.max_len)
        # Pad with -100 (HF standard ignore index) instead of pad_id
        label_pad_len = self.max_len - len(label_ids)
        label_ids = label_ids + [-100] * label_pad_len

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
            torch.tensor(decoder_attention_mask, dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long),
        )


def train():
    # ── 0. Parse CLI Arguments ──
    parser = argparse.ArgumentParser(description="Train A2 Deutsch Grammar Transformer (HF BART)")
    parser.add_argument("--epochs", type=int, help="Override number of epochs from config")
    parser.add_argument("--continue", dest="continue_train", action="store_true",
                        help="Continue training from saved model")
    args = parser.parse_args()

    # ── 1. Load Config ──
    config = load_config()

    # ── 2. Set Device ──
    device = get_device(config.training.device)
    print(f"🚀 Training v2.1 (HF BART) on device: {device}")

    # ── 3. Init Tokenizer ──
    project_root = get_project_root()
    tokenizer = Tokenizer(project_root / "src/tokenizer/tokenizer.json")

    # ── 4. Init or Load Model ──
    save_dir = project_root / "model_final"

    if args.continue_train and save_dir.exists():
        # Resume from HF-format checkpoint
        model = BartForConditionalGeneration.from_pretrained(str(save_dir))
        model = model.to(device)
        print(f"🔄 Resuming training from {save_dir}")
    else:
        # Create fresh model from config
        # create_model() syncs vocab_size and special token IDs from tokenizer
        model = create_model(config, tokenizer)
        model = model.to(device)
        if args.continue_train:
            print(f"⚠️  {save_dir} not found. Starting from scratch.")

    # ── 5. Prepare Data ──
    train_ds = Seq2SeqDataset(config.data.train_path, tokenizer, config.model.max_seq_len, tokenizer.pad_id)
    validation_ds = Seq2SeqDataset(config.data.val_path, tokenizer, config.model.max_seq_len, tokenizer.pad_id)

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_ds, batch_size=config.training.batch_size)

    # ── 6. Optimizer & Loss ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.training.learning_rate))

    # Decision tokens weighting for classification part (✅/❌)
    decision_w = config.training.decision_token_weight
    use_custom_loss = decision_w > 1.0
    decision_token_ids: set[int] = set()
    if use_custom_loss:
        for tok_str in ["✅", "❌"]:
            tid = tokenizer.token_to_id.get(tok_str)
            if tid is not None:
                decision_token_ids.add(tid)

    # When using custom loss, we compute it ourselves with per-token weighting.
    # When not, we let HF compute it via model(labels=...).loss.
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="none") if use_custom_loss else None

    # ── 7. Training Loop with Early Stopping ──
    epochs = args.epochs if args.epochs is not None else config.training.epochs
    patience = config.training.early_stopping_patience
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    print(f"📊 Starting training for up to {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for src_ids, attn_mask, tgt_ids, dec_attn_mask, labels in pbar:
            # Move to device
            src_ids = src_ids.to(device)            # [B, T] — encoder input tokens
            attn_mask = attn_mask.to(device)        # [B, T] — encoder attention mask
            tgt_ids = tgt_ids.to(device)            # [B, T] — decoder input tokens
            dec_attn_mask = dec_attn_mask.to(device) # [B, T] — decoder attention mask
            labels = labels.to(device)              # [B, T] — target labels (-100 = ignore)

            # ── Forward Pass ──
            # HF BART forward:
            #   input_ids → Encoder → memory ∈ ℝ^{B×T×d}
            #   decoder_input_ids → Decoder(memory) → logits ∈ ℝ^{B×T×V}
            outputs = model(
                input_ids=src_ids,
                attention_mask=attn_mask,
                decoder_input_ids=tgt_ids,
                decoder_attention_mask=dec_attn_mask,
                labels=labels if not use_custom_loss else None,
            )

            if use_custom_loss:
                # Custom per-token weighted loss for decision tokens (✅/❌)
                # logits: [B, T, V] → reshape to [B*T, V]
                logits = outputs.logits
                per_token_loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

                flat_targets = labels.reshape(-1)
                weights = torch.ones_like(per_token_loss)
                for tid in decision_token_ids:
                    weights[flat_targets == tid] = decision_w
                loss = (per_token_loss * weights).mean()
            else:
                # HF-computed CrossEntropyLoss (ignore_index=pad automatically)
                loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        # ── Validation ──
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src_ids, attn_mask, tgt_ids, dec_attn_mask, labels in validation_loader:
                src_ids = src_ids.to(device)
                attn_mask = attn_mask.to(device)
                tgt_ids = tgt_ids.to(device)
                dec_attn_mask = dec_attn_mask.to(device)
                labels = labels.to(device)

                outputs = model(
                    input_ids=src_ids,
                    attention_mask=attn_mask,
                    decoder_input_ids=tgt_ids,
                    decoder_attention_mask=dec_attn_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(validation_loader)
        print(f"✨ Epoch {epoch+1} finished. Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # ── Early Stopping ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            # Save best model in HF format — directly loadable by from_pretrained()
            model.save_pretrained(str(save_dir))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience and epochs_without_improvement >= patience:
            print(f"⏹ Early stopping at epoch {epoch+1}")
            break

    # ── 8. Final Report ──
    print(f"📦 Best model (epoch {best_epoch}) saved to {save_dir}/")
    print(f"   Load with: BartForConditionalGeneration.from_pretrained('{save_dir}')")


if __name__ == "__main__":
    train()
