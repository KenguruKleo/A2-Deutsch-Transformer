"""
Evaluation script for A2 Deutsch Grammar Tutor model.

Runs 100 hand-crafted test examples through the model and evaluates:
1. Detection accuracy: Does the model correctly identify correct/incorrect sentences?
2. Correction accuracy: When incorrect, does the model suggest the right correction?
3. Per-topic breakdown: How well does each grammar topic perform?

Usage:
    python tests/evaluate_model.py
    python tests/evaluate_model.py --model model_final.pth
    python tests/evaluate_model.py --verbose
"""

import json
import torch
import torch.nn.functional as F
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model.model import TransformerModel
from src.tokenizer.tokenizer import Tokenizer
from src.config import load_config


def generate_response(text, model, tokenizer, config, device):
    """Generate model response for a given input text."""
    input_ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    max_new_tokens = 64
    temperature = config.generation.temperature
    top_k = config.generation.top_k

    for _ in range(max_new_tokens):
        idx_cond = input_tensor[:, -config.model.max_seq_len:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_tensor = torch.cat((input_tensor, next_token), dim=1)
        
        if next_token.item() == tokenizer.eos_id:
            break

    full_ids = input_tensor[0].tolist()
    if full_ids[0] == tokenizer.bos_id:
        full_ids = full_ids[1:]
        
    return tokenizer.decode(full_ids)


def parse_model_output(full_output, input_text):
    """Parse the model output to extract detection and correction."""
    # Remove the input text prefix from the output
    response = full_output.replace(input_text, "", 1).strip()
    
    detected_correct = "✅ Correct." in response and "❌" not in response
    detected_incorrect = "❌ Incorrect" in response
    
    # Extract correction if present
    correction = None
    if "✅ Correct:" in response:
        for line in response.split("\n"):
            if "✅ Correct:" in line:
                correction = line.split("✅ Correct:")[1].strip()
                break
    
    return {
        "detected_correct": detected_correct,
        "detected_incorrect": detected_incorrect,
        "correction": correction,
        "full_response": response
    }


def normalize(text):
    """Normalize text for comparison."""
    return text.strip().rstrip(".").lower().strip()


def evaluate(model_path="model_final.pth", verbose=False):
    """Run full evaluation on the test dataset."""
    # Load config
    config = load_config(str(project_root / "config.yaml"))
    
    device = config.training.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    
    # Load model
    tokenizer = Tokenizer(str(project_root / "src/tokenizer/vocab.json"))
    checkpoint = torch.load(str(project_root / model_path), map_location=device)
    
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
    
    # Load test data
    test_path = Path(__file__).parent / "test_data.json"
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"🧪 A2 Deutsch Grammar Tutor — Model Evaluation")
    print(f"{'='*70}")
    print(f"📦 Model: {model_path}")
    print(f"🔢 Test examples: {len(test_data)}")
    print(f"🌡️  Temperature: {config.generation.temperature}")
    print(f"{'='*70}\n")
    
    # Track results
    results = []
    topic_stats = defaultdict(lambda: {"total": 0, "detection_correct": 0, "correction_correct": 0})
    level_stats = defaultdict(lambda: {"total": 0, "detection_correct": 0, "correction_correct": 0})
    
    total_detection_correct = 0
    total_correction_correct = 0
    total_correct_sentences = 0
    total_incorrect_sentences = 0
    correct_detected_as_correct = 0
    incorrect_detected_as_incorrect = 0
    
    for i, test in enumerate(test_data):
        test_id = test["id"]
        input_text = test["input"]
        expected_type = test["expected_type"]
        expected_correction = test.get("expected_correction")
        topic = test["topic"]
        level = test["level"]
        
        # Generate response
        full_output = generate_response(input_text, model, tokenizer, config, device)
        parsed = parse_model_output(full_output, input_text)
        
        # Check detection accuracy
        detection_ok = False
        if expected_type == "correct" and parsed["detected_correct"]:
            detection_ok = True
            correct_detected_as_correct += 1
        elif expected_type == "incorrect" and parsed["detected_incorrect"]:
            detection_ok = True
            incorrect_detected_as_incorrect += 1
        
        if expected_type == "correct":
            total_correct_sentences += 1
        else:
            total_incorrect_sentences += 1
        
        if detection_ok:
            total_detection_correct += 1
        
        # Check correction accuracy (only for incorrect sentences)
        correction_ok = False
        if expected_type == "incorrect" and expected_correction and parsed["correction"]:
            if normalize(parsed["correction"]) == normalize(expected_correction):
                correction_ok = True
                total_correction_correct += 1
        elif expected_type == "correct" and detection_ok:
            # Correct sentence correctly detected — counts as correction match too
            correction_ok = True
            total_correction_correct += 1
        
        # Update topic stats
        topic_stats[topic]["total"] += 1
        if detection_ok:
            topic_stats[topic]["detection_correct"] += 1
        if correction_ok:
            topic_stats[topic]["correction_correct"] += 1
        
        # Update level stats
        level_stats[level]["total"] += 1
        if detection_ok:
            level_stats[level]["detection_correct"] += 1
        if correction_ok:
            level_stats[level]["correction_correct"] += 1
        
        # Store result
        result = {
            "id": test_id,
            "input": input_text,
            "expected_type": expected_type,
            "expected_correction": expected_correction,
            "topic": topic,
            "level": level,
            "detection_ok": detection_ok,
            "correction_ok": correction_ok,
            "model_output": parsed["full_response"],
            "model_correction": parsed["correction"]
        }
        results.append(result)
        
        # Print progress
        status = "✅" if detection_ok else "❌"
        corr_status = "✅" if correction_ok else ("❌" if expected_type == "incorrect" else "—")
        
        if verbose or not detection_ok:
            print(f"  #{test_id:3d} {status} [{topic:20s}] {input_text}")
            if not detection_ok or verbose:
                print(f"       Expected: {expected_type} | Got: {'correct' if parsed['detected_correct'] else 'incorrect' if parsed['detected_incorrect'] else 'unknown'}")
                if expected_correction:
                    print(f"       Expected correction: {expected_correction}")
                    print(f"       Model correction:    {parsed['correction']}")
                if verbose:
                    response_short = parsed['full_response'][:100].replace('\n', ' | ')
                    print(f"       Response: {response_short}")
                print()
        else:
            print(f"  #{test_id:3d} {status} {corr_status} [{topic:20s}] {input_text}")
    
    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"📊 RESULTS SUMMARY")
    print(f"{'='*70}")
    
    det_acc = total_detection_correct / len(test_data) * 100
    corr_acc = total_correction_correct / len(test_data) * 100
    
    print(f"\n🎯 Overall Detection Accuracy: {total_detection_correct}/{len(test_data)} ({det_acc:.1f}%)")
    print(f"🎯 Overall Correction Accuracy: {total_correction_correct}/{len(test_data)} ({corr_acc:.1f}%)")
    
    if total_correct_sentences > 0:
        tp_rate = correct_detected_as_correct / total_correct_sentences * 100
        print(f"\n   ✅ Correct → Detected as Correct: {correct_detected_as_correct}/{total_correct_sentences} ({tp_rate:.1f}%)")
    if total_incorrect_sentences > 0:
        tn_rate = incorrect_detected_as_incorrect / total_incorrect_sentences * 100
        print(f"   ❌ Incorrect → Detected as Incorrect: {incorrect_detected_as_incorrect}/{total_incorrect_sentences} ({tn_rate:.1f}%)")
    
    # Per-topic breakdown
    print(f"\n{'─'*70}")
    print(f"📋 Per-Topic Breakdown:")
    print(f"{'─'*70}")
    print(f"  {'Topic':<22} {'Total':>5} {'Det.✅':>6} {'Det.%':>6} {'Corr.✅':>7} {'Corr.%':>7}")
    print(f"  {'─'*22} {'─'*5} {'─'*6} {'─'*6} {'─'*7} {'─'*7}")
    
    for topic in sorted(topic_stats.keys()):
        s = topic_stats[topic]
        d_pct = s["detection_correct"] / s["total"] * 100 if s["total"] > 0 else 0
        c_pct = s["correction_correct"] / s["total"] * 100 if s["total"] > 0 else 0
        emoji = "🟢" if d_pct >= 90 else "🟡" if d_pct >= 75 else "🔴"
        print(f"  {emoji} {topic:<20} {s['total']:>5} {s['detection_correct']:>6} {d_pct:>5.1f}% {s['correction_correct']:>7} {c_pct:>6.1f}%")
    
    # Per-level breakdown
    print(f"\n{'─'*70}")
    print(f"📋 Per-Level Breakdown:")
    print(f"{'─'*70}")
    for level in sorted(level_stats.keys()):
        s = level_stats[level]
        d_pct = s["detection_correct"] / s["total"] * 100 if s["total"] > 0 else 0
        c_pct = s["correction_correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {level}: Detection {s['detection_correct']}/{s['total']} ({d_pct:.1f}%) | Correction {s['correction_correct']}/{s['total']} ({c_pct:.1f}%)")
    
    # Failed examples list
    failed = [r for r in results if not r["detection_ok"]]
    if failed:
        print(f"\n{'─'*70}")
        print(f"🔴 Failed Detection ({len(failed)} examples):")
        print(f"{'─'*70}")
        for r in failed:
            print(f"  #{r['id']:3d} [{r['topic']}] Input: {r['input']}")
            print(f"       Expected: {r['expected_type']} | Model: {r['model_output'][:80]}")
            if r['expected_correction']:
                print(f"       Expected: {r['expected_correction']} | Got: {r['model_correction']}")
            print()
    
    # Save detailed results to JSON
    results_path = Path(__file__).parent / "eval_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total": len(test_data),
                "detection_accuracy": round(det_acc, 1),
                "correction_accuracy": round(corr_acc, 1),
                "correct_detected_as_correct": correct_detected_as_correct,
                "incorrect_detected_as_incorrect": incorrect_detected_as_incorrect,
                "total_correct_sentences": total_correct_sentences,
                "total_incorrect_sentences": total_incorrect_sentences
            },
            "topic_stats": {k: dict(v) for k, v in topic_stats.items()},
            "level_stats": {k: dict(v) for k, v in level_stats.items()},
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_path}")
    print(f"{'='*70}\n")
    
    return det_acc, corr_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate A2 Deutsch Grammar model")
    parser.add_argument("--model", type=str, default="model_final.pth", help="Path to model checkpoint")
    parser.add_argument("--verbose", action="store_true", help="Show all outputs, not just failures")
    args = parser.parse_args()
    
    evaluate(model_path=args.model, verbose=args.verbose)
