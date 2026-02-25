# Project Structure

> Last updated: v2.2

```text
A2-Deutsch-Transformer/
├── src/
│   ├── model/
│   │   └── model.py                # Core Transformer (BartForConditionalGeneration)
│   ├── tokenizer/
│   │   ├── train_tokenizer.py      # Trains BPE tokenizer (HF tokenizers library)
│   │   ├── tokenizer.py            # BPE tokenizer wrapper
│   │   └── tokenizer.json          # Cached trained tokenizer
│   ├── data/
│   │   ├── generator.py            # Orchestrates synthetic data generation
│   │   └── generators/             # Specialised topic generators
│   │       ├── base.py             # BaseGenerator + shared helpers
│   │       ├── cases.py            # Akkusativ, Dativ, Genitiv, Präpositionen…
│   │       ├── syntax.py           # Inversion, Nebensätze, Separable verbs…
│   │       └── verbs.py            # Präsens, Perfekt, Modal, Reflexive…
│   ├── config.py                   # Loads & validates config.yaml
│   ├── train.py                    # Training loop (device auto-detection)
│   ├── inference.py                # Shared model loading and generation logic
│   ├── generate.py                 # CLI inference script
│   └── export_hf.py                # Exports model as native BART to hf_export/
├── hf_export/                      # Bundle uploaded to HF Hub
│   ├── model.safetensors           # Weights (FP32)
│   ├── config.json                 # BartConfig
│   ├── generation_config.json      # Beam-search / generation settings
│   ├── tokenizer.json              # PreTrainedTokenizerFast definition
│   ├── tokenizer_config.json       # Tokenizer metadata
│   └── README.md                   # HF model card
├── hf_space/                       # Bundle deployed to HF Spaces
│   ├── app.py                      # Gradio interface
│   └── requirements.txt            # Space-specific dependencies
├── scripts/
│   ├── eval_tokenizer.py           # Measures tokenizer quality metrics
│   ├── export_hf_precommit.sh      # Pre-commit hook: auto-export before commit
│   ├── upload_to_hf.py             # Upload hf_export/ to HF Hub
│   ├── upload_space_to_hf.py       # Upload hf_space/ to HF Spaces
│   └── restart_space.py            # Force-restart the HF Space via API
├── tests/
│   ├── test_model.py               # Architecture and device tests (pytest)
│   ├── evaluate_model.py           # Full evaluation on 248 test examples
│   ├── test_data.json              # Hand-crafted test sentences per topic
│   └── eval_results.json           # Latest evaluation output (auto-generated)
├── .github/
│   └── workflows/
│       ├── push-to-hf.yml          # CI: push hf_export/ to HF Hub on main
│       └── push-space-to-hf.yml    # CI: push hf_space/ to HF Spaces on main
├── docs/                           # Architecture & grammar documentation
│   ├── architecture_v2.md          # Current architecture (v2.x, BART-based)
│   ├── architecture_v1.md          # ⚠️ Archive — v1.0 Decoder-only architecture
│   ├── tokenizer_metrics.md        # BPE tokenizer quality metrics & how to run
│   ├── topics_examples.md          # Grammar topics with model output examples
│   ├── test_categories_a1_a2.md    # Test coverage map by error category
│   ├── roadmap_v2.md               # ✅ Completed — v1.0→v2.0 migration plan
│   └── project_structure.md        # This file
├── data/                           # Generated JSONL datasets (train/val)
├── data_raw/                       # Raw PDF textbooks
├── model_final/                    # Saved model checkpoint (after training)
├── config.yaml                     # Model & training hyperparameters
├── requirements.txt
└── README.md
```

---

## Key Entry Points

| What you want to do | File |
|---|---|
| **Generate training data** | `src/data/generator.py` |
| **Train the model** | `src/train.py` |
| **Run inference (CLI)** | `src/generate.py` |
| **Export to HF Hub format** | `src/export_hf.py` |
| **Train / re-train tokenizer** | `src/tokenizer/train_tokenizer.py` |
| **Run unit tests** | `tests/test_model.py` |
| **Run full evaluation** | `tests/evaluate_model.py` |
| **Measure tokenizer quality** | `scripts/eval_tokenizer.py` |
| **Upload model to HF Hub** | `scripts/upload_to_hf.py` |
| **Upload Gradio Space** | `scripts/upload_space_to_hf.py` |

---

## Generated / Ignored Directories

| Directory | Purpose | In `.gitignore`? |
|---|---|---|
| `data/` | Generated `.jsonl` datasets | Partial (large files) |
| `model_final/` | Checkpoint `.pth` saved after training | Yes (weights) |
| `.venv/` | Python virtual environment | Yes |
| `src/tokenizer/tokenizer.json` | Cached tokenizer | No (small, committed) |
| `hf_export/model.safetensors` | Exported weights | Committed (24 MB) |
