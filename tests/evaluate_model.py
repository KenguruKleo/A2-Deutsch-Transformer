"""
Evaluation script for A2 Deutsch Grammar Tutor v2.1 (HF BART).
Evaluates Detection & Correction accuracy with beautiful formatting and high speed.
"""

import json
import sys
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
from transformers import BartForConditionalGeneration

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import load_config, get_device
from src.tokenizer.tokenizer import Tokenizer

class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_ids = self.tokenizer.encode(item['input'], add_bos=True, add_eos=True, max_len=self.max_len)
        src_ids = self.tokenizer.pad_sequence(src_ids, max_len=self.max_len)
        return torch.tensor(src_ids, dtype=torch.long), idx

def normalize(text):
    if text is None: return ""
    return text.strip().rstrip(".").lower().strip()

def parse_output(response):
    detected_correct = "✅ Correct." in response and "❌" not in response
    detected_incorrect = "❌ Incorrect" in response
    correction = None
    if "✅ Correct:" in response:
        for line in response.split("\n"):
            if "✅ Correct:" in line:
                correction = line.split("✅ Correct:")[1].strip()
                break
    return detected_correct, detected_incorrect, correction

def evaluate(model_path="model_final", batch_size=64, verbose=False):
    config = load_config()
    device = get_device("auto")
    tokenizer = Tokenizer(project_root / "src/tokenizer/tokenizer.json")
    
    model_dir = project_root / model_path
    if not model_dir.exists() and not (model_dir / "config.json").exists():
        print(f"❌ Model missing at {model_dir}")
        return

    model = BartForConditionalGeneration.from_pretrained(str(model_dir))
    model = model.to(device)
    model.eval()

    with open(project_root / "tests/test_data.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    dataset = TestDataset(test_data, tokenizer, config.model.max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"\n{'='*80}")
    print(f"🧪 A2 Deutsch Grammar Tutor (HF BART) — Batch Evaluation (BS={batch_size})")
    print(f"{'='*80}")
    
    results = [None] * len(test_data)
    
    with torch.no_grad():
        for batch_src, indices in loader:
            batch_src = batch_src.to(device)
            attention_mask = (batch_src != tokenizer.pad_id).long()
            
            # Batch generate
            generated_ids = model.generate(
                input_ids=batch_src,
                attention_mask=attention_mask,
                max_length=config.model.max_seq_len,
                num_beams=1,
                do_sample=False
            )
            
            for i, idx in enumerate(indices):
                response = tokenizer.decode(generated_ids[i].tolist(), skip_special=True)
                test_item = test_data[idx]
                
                det_c, det_inc, corr = parse_output(response)
                
                # Logic
                expected = test_item["expected_type"]
                det_ok = (expected == "correct" and det_c) or (expected == "incorrect" and det_inc)
                
                corr_ok = False
                if expected == "incorrect" and test_item.get("expected_correction") and corr:
                    if normalize(corr) == normalize(test_item["expected_correction"]):
                        corr_ok = True
                elif expected == "correct" and det_ok:
                    corr_ok = True

                results[idx] = {
                    **test_item,
                    "det_ok": det_ok,
                    "corr_ok": corr_ok,
                    "output": response,
                    "model_corr": corr
                }
                
                # Instant feedback (Compact style)
                mark = "✅" if det_ok else "❌"
                c_mark = "✓" if corr_ok else "f" if expected == "incorrect" else " "
                sys.stdout.write(f"{mark}{c_mark} ")
                sys.stdout.flush()
                if (idx + 1) % 10 == 0: print()

    # --- Summary ---
    print(f"\n\n{'='*80}")
    print(f"📊 FINAL STATISTICS")
    print(f"{'='*80}")
    
    topic_stats = defaultdict(lambda: {"total": 0, "det": 0, "corr": 0})
    total_det = 0
    total_corr = 0

    for r in results:
        t = r["topic"]
        topic_stats[t]["total"] += 1
        if r["det_ok"]: 
            topic_stats[t]["det"] += 1
            total_det += 1
        if r["corr_ok"]: 
            topic_stats[t]["corr"] += 1
            total_corr += 1

    det_acc = total_det / len(test_data) * 100
    corr_acc = total_corr / len(test_data) * 100

    print(f"  {'Topic':<25} {'Total':>5} {'Det.✅':>7} {'Det.%':>7} {'Corr.✅':>7}")
    print(f"  {'-'*25} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")

    for t in sorted(topic_stats.keys()):
        s = topic_stats[t]
        d_p = s["det"]/s["total"]*100
        emoji = "🟢" if d_p > 95 else "🟡" if d_p > 80 else "🔴"
        print(f"  {emoji} {t:<22} {s['total']:>5} {s['det']:>7} {d_p:>6.1f}% {s['corr']:>7}")

    print(f"{'='*80}")
    print(f"⭐ OVERALL DETECTION:  {total_det}/{len(test_data)} ({det_acc:.1f}%)")
    print(f"⭐ OVERALL CORRECTION: {total_corr}/{len(test_data)} ({corr_acc:.1f}%)")
    print(f"{'='*80}\n")

    # --- Failed Examples ---
    failed = [r for r in results if not r["det_ok"] or not r["corr_ok"]]
    if failed:
        print(f"🔴 FAILED EXAMPLES ({len(failed)}):")
        print(f"{'-'*80}")
        for r in failed:
            print(f"  #{r['id']:3d} [{r['topic']}] Input: {r['input']}")
            print(f"       Expected: {r['expected_type']} | Expected Correction: {r.get('expected_correction', 'None')}")
            print(f"       Model:    {'✅ Correct.' if r['det_ok'] and r['expected_type']=='correct' else '❌ Incorrect.'}")
            print(f"       Output:   {r['output'].replace('\\n', ' | ')}")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", type=str, default="model_final")
    parser.add_argument("--verbose", action="store_true", help="Show detailed failure summary")
    args = parser.parse_args()
    evaluate(model_path=args.model, batch_size=args.batch_size, verbose=args.verbose)
