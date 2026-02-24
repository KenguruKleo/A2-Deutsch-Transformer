# Tokenizer Quality Metrics

This document explains how to measure BPE tokenizer quality and how to interpret the results.

Measurement script: [`scripts/eval_tokenizer.py`](../scripts/eval_tokenizer.py)

---

## How to Run

```bash
# Activate the environment
source .venv/bin/activate

# Run all metrics
python scripts/eval_tokenizer.py

# With a different max_seq_len (if changed in config.yaml)
python scripts/eval_tokenizer.py --max-seq-len 128

# Compare two tokenizers
python scripts/eval_tokenizer.py --tokenizer path/to/other/tokenizer.json
```

**Prerequisites:** training data must exist (`data/train.jsonl`, `data/val.jsonl`) and the tokenizer must be trained (`src/tokenizer/tokenizer.json`). If not — run `python src/data/generator.py` first.

---

## Metrics

### 1. Fertility (tokens / word)

**What it measures:** average number of tokens produced per word.

```
"Ich gehe nach Hause"  → 4 words, 4 tokens  → fertility = 1.0
"Donaudampfschiff"     → 1 word,  8 tokens  → fertility = 8.0
```

**Formula:**
```
fertility = total_tokens / total_words   (across all sentences)
```

**Interpretation:**

| Value     | Assessment |
|-----------|------------|
| 1.0 – 1.3 | 🟢 Excellent — most words are single tokens |
| 1.3 – 1.7 | 🟢 Normal for BPE |
| 1.7 – 2.5 | 🟡 Acceptable — consider increasing `vocab_size` |
| > 2.5     | 🔴 Vocabulary too small |

**Current result:** `1.462` → 🟢 Normal

**Relationship to `vocab_size`:** larger vocabulary → lower fertility, because more words become whole tokens:

```
vocab=4000  → fertility ≈ 1.8–2.2
vocab=8000  → fertility ≈ 1.4–1.6   ← current
vocab=16000 → fertility ≈ 1.2–1.4
vocab=32000 → fertility ≈ 1.1–1.3   (like LLaMA)
```

---

### 2. UNK Rate

**What it measures:** percentage of `<UNK>` (unknown) tokens in the data.

**Formula:**
```
unk_rate = (number of <UNK> tokens) / (total tokens) × 100%
```

**Interpretation:**

| Value   | Assessment |
|---------|------------|
| 0.0%    | 🟢 Perfect |
| < 0.1%  | 🟡 Acceptable |
| > 0.1%  | 🔴 Vocabulary does not cover the data |

**Current result:** `0.0000%` → 🟢 Perfect

**Note:** Byte-level BPE **guarantees** 0% UNK rate — any character is decomposed into bytes (256 possible values). This metric is therefore a sanity check rather than a diagnostic tool.

With the old word-level tokenizer (v1.0), UNK rate could be significant — any word outside the fixed vocabulary produced `<UNK>`.

---

### 3. Continuation Token Rate

**What it measures:** fraction of tokens that are **continuations** of a word (subwords in the middle).

In Byte-level BPE, the start of a word is marked with `Ġ` (a space prefix). Tokens without `Ġ` are continuations:

```
"gegangen" → ["Ġgeg", "angen"]
               ↑        ↑
             start    continuation
```

**Formula:**
```
continuation_rate = (tokens without Ġ prefix) / (all tokens) × 100%
```

**Interpretation:**

| Value  | Assessment |
|--------|------------|
| < 25%  | 🟢 Excellent — most words are whole tokens |
| 25–40% | 🟢 Normal |
| 40–55% | 🟡 Moderate — consider increasing vocabulary |
| > 55%  | 🔴 Vocabulary too small |

**Current result:** `33.9%` → 🟢 Normal

---

### 4. Sequence Lengths

**What it measures:** distribution of tokenized sequence lengths relative to `max_seq_len` from `config.yaml`.

**Key value:** percentage of sequences that **exceed** `max_seq_len`.

**Why it matters:** sequences longer than `max_seq_len` are truncated during training. The model never sees the end of the response, which degrades learning quality.

**Interpretation:**

| Truncation % | Assessment |
|---|---|
| 0%   | 🟢 `max_seq_len` has sufficient headroom |
| < 5% | 🟡 Small fraction is being truncated |
| > 5% | 🔴 Increase `max_seq_len` in `config.yaml` |

**Current results** (max_seq_len=64):

```
Average:  12.2 tokens
Minimum:  1 token
Maximum:  48 tokens
> 64:     0 / 10000 (0.0%)
```

→ 🟢 `max_seq_len=64` has comfortable headroom (maximum is only 48).

---

## When to Re-run

- After changing `vocab_size` in `train_tokenizer.py`
- After adding new grammar topics or B1 data (word distribution changes)
- If the model trains poorly — the tokenizer may not cover important words
- Before increasing `max_seq_len` — to verify it is actually needed

---

## Choosing vocab_size

General rule: `vocab_size` depends on **data volume** and **language complexity**.

```
< 10k sentences   → vocab 4000–8000
10k–100k          → vocab 8000–16000    ← current
100k–1M           → vocab 16000–32000
> 1M              → vocab 32000+
```

To compare two sizes, train a second tokenizer with a different `VOCAB_SIZE` in `train_tokenizer.py`, save it under a different filename, then compare:

```bash
python scripts/eval_tokenizer.py --tokenizer src/tokenizer/tokenizer_16k.json
```

The primary criterion is **fertility**. If increasing the vocabulary reduces fertility by less than 0.1 — the gain does not justify the larger embedding matrix `[vocab_size, d_model]`.
