---
language: de
license: mit
tags:
- grammar
- german
- transformer
- education
- pytorch
datasets:
- synthetic
---

# Deutsch A2 Grammar Transformer (Ukrainian Explanations)

This is a compact, custom-built Transformer Decoder model designed to identify, correct, and explain German grammar errors at the A1-A2 level. It is specifically tailored for **Ukrainian-speaking learners**, providing detailed grammar feedback and explanations in **Ukrainian**.

## 🚀 Model Capabilities
The model is trained to handle a wide range of grammar topics:
- **Verbs:** Conjugation, Perfekt (auxiliary choice), Modal verbs (word order), Separable and Reflexive verbs.
- **Syntax:** Inversion, W-Questions, Subordinate clauses (using *weil, dass, wenn*).
- **Cases:** Akkusativ vs. Dativ, Wechselpräpositionen, Adjective endings, and Possessive pronouns.
- **Negation:** Proper usage of *nicht* vs. *kein*.

## 🛠 Architecture
- **Type:** Transformer Decoder (GPT-style)
- **Parameters:** ~5M (Model size: 2.5 MB)
- **Layers:** 4
- **Attention Heads:** 4
- **Embedding Dim:** 128
- **Tokenizer:** Custom Word-level (4,000 tokens)

## 📖 How to Use
Since this model uses a custom architecture, you must enable `trust_remote_code=True`.

```python
from transformers import AutoModelForCausalLM

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    "kengurukleo/deutsch_a2_transformer", 
    trust_remote_code=True
)

# This model requires its custom tokenizer.py and vocab.json included in the repo.
```

## ⚠️ Limitations
- **Vocabulary:** Uses a fixed word-level vocabulary of 4,000 tokens. Words outside this list are treated as `<UNK>`.
- **Explaining Language:** Explanations for errors are currently provided in **Ukrainian**, as it was designed for Ukrainian-speaking learners of German.
- **Synthetic Data:** The model was trained on a large synthetic dataset and may not generalize perfectly to complex literary German.

## 👨‍💻 Author
Created by [KenguruKleo](https://github.com/KenguruKleo).
