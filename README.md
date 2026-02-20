# 🇩🇪 A2 Deutsch Grammar Tutor

A compact Transformer model for checking and explaining errors in German sentences at the A2 level. 
Trained to correct grammar and provide simple explanations **in Ukrainian**.

🤗 **Model on Hugging Face:** [kengurukleo/deutsch_a2_transformer](https://huggingface.co/kengurukleo/deutsch_a2_transformer)  
✨ **Live Demo (Space):** [kengurukleo/deutsch-a2-tutor](https://huggingface.co/spaces/kengurukleo/deutsch-a2-tutor)

## Features

| Function | Example |
|---|---|
| ✅ Validity Check | `Ich bin nach Hause gegangen.` → ✅ |
| ❌ Error Correction | `Dann ich gehe...` → `Dann gehe ich...` |
| 📝 Explanations | Detailed grammar feedback provided in **Ukrainian**. |

### ✅ Covered A1-A2 Topics (100% Complete)

| Topic | Level | What the model does / Explanation | Status |
|:---|:---:|:---|:---:|
| **Präsens** | A1 | Verb conjugation (e.g., *ich esse, du isst*). | ✅ |
| **W-Fragen** | A1 | Question word order (*Wo wohnst du?*). | ✅ |
| **Akkusativ** | A1 | Masculine article change (*der -> den*). | ✅ |
| **Negation** | A1 | Usage of *nicht* vs. *kein* (noun vs. adj/verb). | ✅ |
| **Modalverben** | A1/A2 | Conjugation and putting the main verb at the end. | ✅ |
| **Possessivpron.**| A1/A2 | Agreement of *mein, dein, sein...* in Nominativ. | ✅ |
| **Fixed Prepos.** | A1/A2 | Prepositions with fixed cases (*mit + Dat*, *für + Akk*). | ✅ |
| **Imperativ** | A1/A2 | Command forms for *du*, *ihr*, and *Sie*. | ✅ |
| **Perfekt** | A2 | Choosing *haben* vs. *sein* and *Partizip II* forms. | ✅ |
| **Inversion** | A2 | Verb-second rule after adverbs (*Heute gehe ich...*). | ✅ |
| **Separable Verbs**| A2 | Prefix position in present tense (*Ich kaufe ein*). | ✅ |
| **Dativ** | A2 | Articles after Dativ-governing verbs (*helfen, danken*). | ✅ |
| **Wechselpräp.** | A2 | Two-way prepositions (Wohin? -> Akk / Wo? -> Dat). | ✅ |
| **Nebensätze** | A2 | Verb-last order in clauses with *weil, dass, wenn*. | ✅ |
| **Adjektivdekl.** | A2 | Basic endings after *ein* in Nominativ. | ✅ |
| **Reflexive Verben**| A2 | Correct reflexive pronouns (*mich, dich, sich...*). | ✅ |
| **Präteritum** | A2 | Past tense forms for *sein* and *haben* (*war, hatte*). | ✅ |
| **Komparation** | A2 | Adjective comparison (*gut - besser*, not *mehr gut*). | ✅ |
| **Nominativ** | A1 | Article as subject (*Der/Die/Das* + noun). | ✅ |
| **Genitiv** | A2 | Genitive with prepositions (*während des*, *wegen der*, *trotz des*). | ✅ |

For a complete list of examples and model explanations for each topic, see:  
👉 **[Grammar Topics & Examples](docs/topics_examples.md)**

## Architecture

```
Transformer Decoder Only
├── V = 4,000 tokens (words + forms + explanations)
├── T = 64  (max sequence length)
├── d_model = 128
├── L = 4 Layers
├── H = 4 Attention Heads
├── Weight tying = ON (shared weights between Embeddings and LM Head)
└── Precision = FP16 → Model size ≈ 2.5 MB

Detailed mathematical description of all matrix transformations can be found in [docs/architecture.md](docs/architecture.md).
```

## Project Structure

```text
A2-Deutsch-Transformer/
├── src/
│   ├── model/
│   │   ├── model.py                # Core Transformer architecture
│   │   ├── configuration_custom.py # HF Config wrapper
│   │   └── modeling_custom.py      # HF Model wrapper (custom code)
│   ├── tokenizer/
│   │   ├── build_vocab.py          # PDF analysis and vocab creation
│   │   └── tokenizer.py            # Word-level tokenizer
│   ├── data/
│   │   ├── generator.py            # Main synthetic data generator
│   │   └── generators/             # Specialized topic generators
│   ├── train.py                    # Training loop (MPS optimized)
│   ├── generate.py                 # CLI inference script
│   └── export_hf.py                # Hugging Face export script
├── hf_export/                      # Bundle for HF Hub (weights + code)
├── hf_space/                       # Bundle for HF Spaces (Gradio app)
├── tests/
│   ├── test_model.py              # Architecture and device tests
│   ├── evaluate_model.py          # Full evaluation on 248 test examples
│   └── test_data.json              # Hand-crafted test sentences per topic
├── data/                           # Generated JSONL datasets
├── data_raw/                       # Raw PDF textbooks
├── docs/                           # Architecture and grammar docs
├── config.yaml                     # Model & training hyperparameters
├── requirements.txt
└── README.md
```

## How It Works

### 1. `build_vocab.py`
Creates the "brain" of the tokenizer. It analyzes the `Begegnungen_A2.pdf` textbook, extracts the most frequent German words, adds conjugation tables, and includes words for explanations. The result is a `vocab.json` file with 4000 unique tokens.

### 2. `generator.py`
Generates thousands of training examples. It knows grammar rules, takes a correct sentence and intentionally "breaks" it (e.g., changes word order or auxiliary verb), adding an explanation of why it is an error.

### 3. Training
The model is trained locally on **Apple Silicon (M1/M2/M3)** using `torch.device("mps")`. Due to its small size (2.5 MB), training takes only a few minutes.

## Installation & Setup

Follow these steps to initialize the project and set up the environment:

```bash
# 1. Clone the repository (if not already done)
git clone https://github.com/KenguruKleo/A2-Deutsch-Transformer.git
cd A2-Deutsch-Transformer

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate     # On Windows

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Testing

To verify the model architecture and device compatibility, run the following command:

```bash
# Run model unit tests
python tests/test_model.py
```

These tests check:
- Output dimensions (`[batch, seq_len, vocab_size]`).
- Successful execution on **MPS** (Apple Silicon) or **CPU**.

## Quick Start

Once the environment is set up and activated:

```bash
# 1. Build vocabulary (if changing word lists)
python src/tokenizer/build_vocab.py

# 2. Generate training data
python src/data/generator.py

# 3. Run training
python src/train.py

# 4. Test the model
python src/generate.py --text "Ich habe nach Berlin gefahren."

# 5. Export to Hugging Face format
python src/export_hf.py
```

## Evaluation

The project includes an evaluation script that runs the trained model on **248 hand-crafted test examples** and measures detection accuracy (correct vs incorrect) and correction accuracy (whether the suggested fix matches the expected one).  
**Note:** The model is never trained on this test set; the examples are held out to measure generalization.

### How to run

```bash
# Activate the environment first
source .venv/bin/activate   # macOS/Linux

# Evaluate the default model (model_final.pth)
python tests/evaluate_model.py

# Verbose: print each test case and model response
python tests/evaluate_model.py --verbose

# Use a specific checkpoint
python tests/evaluate_model.py --model path/to/checkpoint.pth
```

### Results (example run)

| Metric | Value |
|--------|--------|
| Test examples | 248 |
| **Detection accuracy** | **248/248 (100.0%)** |
| **Correction accuracy** | **242/248 (97.6%)** |
| Correct → detected as correct | 138/138 (100.0%) |
| Incorrect → detected as incorrect | 110/110 (100.0%) |

**Per level:**  
- A1: Detection 99/99 (100.0%) | Correction 98/99 (99.0%)  
- A2: Detection 149/149 (100.0%) | Correction 144/149 (96.6%)

**Per-topic breakdown:**

| Topic | Total | Det.✅ | Det.% | Corr.✅ | Corr.% |
|-------|-------|--------|-------|--------|--------|
| adjective_endings | 8 | 8 | 100.0% | 8 | 100.0% |
| akkusativ | 19 | 19 | 100.0% | 19 | 100.0% |
| dativ | 12 | 12 | 100.0% | 12 | 100.0% |
| fixed_prepositions | 17 | 17 | 100.0% | 17 | 100.0% |
| genitiv | 4 | 4 | 100.0% | 2 | 50.0% |
| haben_sein | 17 | 17 | 100.0% | 16 | 94.1% |
| imperativ | 5 | 5 | 100.0% | 5 | 100.0% |
| inversion | 16 | 16 | 100.0% | 15 | 93.8% |
| komparation | 6 | 6 | 100.0% | 6 | 100.0% |
| modal | 15 | 15 | 100.0% | 15 | 100.0% |
| nebensatz_dass_wenn | 11 | 11 | 100.0% | 11 | 100.0% |
| nebensatz_weil | 6 | 6 | 100.0% | 6 | 100.0% |
| negation | 12 | 12 | 100.0% | 12 | 100.0% |
| nominativ | 6 | 6 | 100.0% | 6 | 100.0% |
| partizip | 7 | 7 | 100.0% | 6 | 85.7% |
| perfekt_aux | 20 | 20 | 100.0% | 20 | 100.0% |
| possessive | 8 | 8 | 100.0% | 8 | 100.0% |
| praesens | 13 | 13 | 100.0% | 13 | 100.0% |
| praeteritum | 8 | 8 | 100.0% | 8 | 100.0% |
| questions | 8 | 8 | 100.0% | 8 | 100.0% |
| reflexive | 5 | 5 | 100.0% | 5 | 100.0% |
| separable | 8 | 8 | 100.0% | 7 | 87.5% |
| strong_verbs | 6 | 6 | 100.0% | 6 | 100.0% |
| wechselpraep | 11 | 11 | 100.0% | 11 | 100.0% |

Detailed results are written to `tests/eval_results.json`. At the end of each run, the script prints **Failed Detection** cases (if any): sentences where the model’s correct/incorrect decision did not match the expected one.

## Hugging Face Hub

This model is hosted on the [Hugging Face Hub](https://huggingface.co/kengurukleo/deutsch_a2_transformer). Below are instructions for users and developers.

### 📥 For Users (Loading the Model)
You can load and use this model directly in your Python code using the `transformers` library. Note that `trust_remote_code=True` is required because the model uses a custom architecture and tokenizer.

```python
from transformers import AutoModelForCausalLM

# Load model and use custom code from the Hub
model = AutoModelForCausalLM.from_pretrained(
    "kengurukleo/deutsch_a2_transformer", 
    trust_remote_code=True
)
```

### 🛠 For Developers (Export and Publish)
If you want to re-export the model or publish your own version:

1. **Export to Safetensors**:
   Run the export script to create a compatible bundle (weights, config, and source code):
   ```bash
   python src/export_hf.py
   ```
   This creates a `hf_export/` directory with `model.safetensors`, `config.json`, and the necessary `.py` files.

2. **Publish to Hub**:
   Use the Hugging Face CLI to upload the export bundle:
   ```bash
   huggingface-cli upload kengurukleo/deutsch_a2_transformer ./hf_export .
   ```
   Or push to `main` and let the GitHub Action do it (see below).

### Pre-commit: auto-export HF model on every commit
To regenerate `hf_export/` automatically before each commit (so the bundle always matches `model_final.pth`):

```bash
pip install pre-commit
pre-commit install
```

After that, every `git commit` will run `python src/export_hf.py` and stage `hf_export/`. If `model_final.pth` is missing, the hook skips without blocking the commit.

### GitHub Action: push model to Hugging Face
The workflow [`.github/workflows/push-to-hf.yml`](.github/workflows/push-to-hf.yml) pushes `hf_export/` to [kengurukleo/deutsch_a2_transformer](https://huggingface.co/kengurukleo/deutsch_a2_transformer) when you push to `main` (if relevant files changed) or when you run it manually (**Actions → Push model to Hugging Face → Run workflow**).

**Required:** Add a repository secret with a Hugging Face write token:
- **Settings → Secrets and variables → Actions → New repository secret**
- Name: `HF_TOKEN` or `HG_TOKEN`
- Value: your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (with write access)

The workflow expects `model_final.pth` and `config.yaml` to be in the repo (e.g. committed after training).

## Data Format (JSONL)

```json
{
  "input": "Heute ich gehe ins Kino.",
  "output": "❌ Incorrect.\n✅ Correct: Heute gehe ich ins Kino.\n📝 Explanation: The verb must be in the second position."
}
```

## License

MIT
