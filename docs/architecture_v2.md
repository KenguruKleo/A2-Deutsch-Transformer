# Математична архітектура моделі v2.1 (HF BART)

Цей документ описує архітектуру Encoder-Decoder Transformer (v2.1), яка тепер базується на стандартній реалізації **Hugging Face `BartForConditionalGeneration`**. Це усуває потребу у власному коді моделі (`model.py`) та кастомному циклі генерації, дозволяючи нативну інтеграцію зі всією екосистемою HF (включаючи пряме розгортання в HF Spaces без сторонніх скриптів).

---

## Чому перейшли на HF BART?

Попередня версія v2.0 використовувала власноруч написаний клас `GrammarTransformer`. Під час спроби експорту ваг у формат HF BART виникли нездоланні розбіжності (так званий "архітектурний дрейф"):
1. **Positional Encodings:** Ми використовували фіксовані синусоїдальні PE `sin(pos/10000)`, тоді як HF BART жорстко вимагає `LearnedPositionalEmbedding` (з обов'язковим зсувом `offset=2` для паддинга).
2. **LayerNorms:** HF BART має додаткову нормалізацію `layernorm_embedding` одразу після суми токенних та позиційних ембедінгів.
3. **Embedding Scaling:** HF BART автоматично домножує ембедінги на `sqrt(d_model)`.
4. **Tokenization Mismatch:** Токенізатор BART (`BartTokenizerFast`) самовільно додавав власні спец-токени `<s>`, `</s>`, змінюючи розмірність матриці ембедінгів (`8000` → `8003`), що призводило до краху при завантаженні ваг.

У v2.1 ми переписали скрипти тренування (`train.py`), щоб вони працювали **безпосередньо** з чистим `BartForConditionalGeneration`. Таким чином, чекпоінти зберігаються одразу у стандартному HF форматі, і жоден конвертер більше не потрібен.

---

## Параметри моделі (через `config.yaml` -> `BartConfig`)

| Параметр | Значення | Опис |
|---|---|---|
| `V` | 8003 | Розмір словника (8000 BPE + 3 від HF BART) |
| `d` | 256 | Розмірність моделі (d_model) |
| `h` | 4 | Кількість голів уваги (heads) |
| `d_k` | 64 | Розмір однієї голови (`d / h = 256 / 4`) |
| `d_ff` | 512 | Внутрішній розмір FFN |
| `N` | 3 | Кількість блоків encoder і decoder |
| `S` | до 64 | Довжина вхідного речення |
| `T` | до 64 | Довжина вихідної відповіді |

---

## Загальний потік даних (Data Flow)

```
encoder_input ids [S]                decoder_input ids [T]
        ↓                                     ↓
   Embedding E                           Embedding E        ← спільна матриця
        ↓                                     ↓
   Learned Pos Emb P                    Learned Pos Emb P
        ↓                                     ↓
   LayerNorm (embed)                     LayerNorm (embed)
        ↓                                     ↓
   EncoderBlock 1                        DecoderBlock 1 ←─────┐
        ↓                                     ↓              │
   EncoderBlock 2                        DecoderBlock 2 ←─────┤
        ↓                                     ↓              │
   EncoderBlock 3                        DecoderBlock 3 ←─────┤
        ↓                                     ↓              │
   M = memory [S, d]  ─────────────────────────────────────────┘
                          (M подається у cross-attention кожного блоку)
                                             ↓
                                        LM Head (E^T)
                                             ↓
                                      Logits [T, V]
```

---

## 1. Вхід (Embeddings)

```python
# 1. Token Embeddings: спільна матриця для encoder і decoder
X = Embed(ids, E) → [B, Seq_len, 256]

# (HF BART scale_embedding відключено в BartConfig)

# 2. Positional Embeddings: матриця P, що навчається (на відміну від v2.0)
P_matrix = LearnedPositionalEmbedding(max_position_embeddings=64, d_model=256)
X_pos = P_matrix(positions) → [B, Seq_len, 256]

X = X + X_pos

# 3. Embedding LayerNorm (додатковий шар у HF BART)
X = LayerNorm(X) → [B, Seq_len, 256]
```

---

## 2. Encoder Block (N=3) та Decoder Block (N=3)

Кожен **Encoder Block** складається з:
1. `Self-Attention` (без маски, бачить усе) -> [B, S, 256]
2. `LayerNorm` + `Residual`
3. `FFN` (`W1` -> `GELU` -> `W2`) -> [B, S, 256]
4. `LayerNorm` + `Residual`

Кожен **Decoder Block** складається з:
1. `Masked Self-Attention` (каузальна маска, нижній трикутник) -> [B, T, 256]
2. `LayerNorm` + `Residual`
3. `Cross-Attention` (Query від Decoder, Key/Value від Encoder Memory `M`) -> [B, T, 256]
4. `LayerNorm` + `Residual`
5. `FFN` -> [B, T, 256]
6. `LayerNorm` + `Residual`

---

## 3. Тренування

Завдяки переходу на HF `BartForConditionalGeneration`, ми викликаємо лише один метод `forward()` замість ручного створення масок:

```python
outputs = model(
    input_ids=src_ids,                     # Вхід у Encoder
    attention_mask=encoder_mask,           # [1] для слів, [0] для <PAD>
    decoder_input_ids=tgt_ids,             # Teacher forcing (<BOS> + слова)
    decoder_attention_mask=decoder_mask,
    labels=labels                          # Цілі (зсунуті: слова + <EOS>)
)

# Якщо decision_weight == 1.0, HF BART сам рахує:
loss = outputs.loss
logits = outputs.logits
```

**Особливість:** Label для `<PAD>` маркується як `-100`, що є стандартом у PyTorch `CrossEntropyLoss` для ігнорування індексів.

---

## 4. Генерація

Для генерації достатньо одного рядка (авторегресивний цикл реалізований на C++ під капотом Transformers):

```python
output_ids = model.generate(
    input_ids=src_ids,
    attention_mask=encoder_mask,
    max_length=64,
    num_beams=1,      # Greedy search
    do_sample=False
)
```

## Висновок

Міграція на `BartForConditionalGeneration` (v2.1) спростила наш код більше ніж на 300+ рядків, зробила експорт тривіальним викликом `model.save_pretrained()`, і забезпечила 100% гарантію відповідності інференсу локально та в Hugging Face Spaces.
