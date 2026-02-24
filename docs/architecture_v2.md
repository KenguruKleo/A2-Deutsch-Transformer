# Математична архітектура моделі v2.0

Цей документ описує архітектуру Encoder-Decoder Transformer (v2.0) — шлях даних крізь модель, використовуючи матричні розмірності та математичні операції.

Порівняння з v1.0 (Decoder-only): [architecture.md](architecture.md)

## Позначення
- `[R, C]` — матриця розмірністю R рядків та C стовпчиків.
- `*` — матричне множення.
- `+` — елементне додавання.
- `^T` — транспонування матриці.
- `S` — довжина вхідної послідовності (encoder input).
- `T` — довжина вихідної послідовності (decoder input/output).

---

## Параметри моделі

| Параметр | Значення | Опис |
|---|---|---|
| `V` | 8000 | Розмір словника (vocab size) |
| `d` | 256 | Розмірність моделі (d_model) |
| `h` | 4 | Кількість голів уваги (heads) |
| `d_k` | 64 | Розмір однієї голови (`d / h = 256 / 4`) |
| `d_ff` | 512 | Внутрішній розмір FFN |
| `N` | 3 | Кількість блоків encoder і decoder |
| `S` | до 64 | Довжина вхідного речення |
| `T` | до 64 | Довжина вихідної відповіді |

Конкретний приклад для цього документу:
```
encoder_input: "Heute ich gehe ins Kino ."   → S = 6 токенів
decoder_input: "<BOS> ❌ Incorrect . ✅ ..." → T = 10 токенів
```

---

## Загальна схема

```
encoder_input ids [S]                decoder_input ids [T]
        ↓                                     ↓
   Embedding E                           Embedding E        ← спільна матриця
        ↓                                     ↓
  Positional Enc                        Positional Enc
        ↓                                     ↓
  EncoderBlock 1                        DecoderBlock 1 ←─────┐
        ↓                                     ↓              │
  EncoderBlock 2                        DecoderBlock 2 ←─────┤
        ↓                                     ↓              │
  EncoderBlock 3                        DecoderBlock 3 ←─────┤
        ↓                                     ↓              │
  M = memory [S, d]  ─────────────────────────────────────────┘
                          (cross-attention у кожному блоці)
                                             ↓
                                        LM Head (E^T)
                                             ↓
                                      Logits [T, V]
```

---

## 1. Токенізатор (BPE)

На відміну від v1.0 (word-level, фіксований словник 4000 слів), v2.0 використовує **Byte-level BPE** токенізатор натренований на даних проекту.

- Словник: 8000 токенів (підслова, не цілі слова)
- Спеціальні токени: `<PAD>=0`, `<BOS>=1`, `<EOS>=2`, `<UNK>=3`
- Будь-яке слово розкладається на підчастини — ніколи не дає `<UNK>`:
  ```
  "gegangen"  → ["▁geg", "angen"]
  "Donaudampfschifffahrt" → ["▁Don", "au", "dam", "pf", "schiff", "fahrt"]
  ```

---

## 2. Вхід (Input & Embeddings)

Операція однакова для encoder і decoder — спільна матриця ваг `E`.

### Encoder
1. **Вхідні токени:** `encoder_ids [S] = [6]`
2. **Token Embedding Matrix (`E`):** `[V, d] = [8000, 256]`
   - `X_enc = Embed(encoder_ids, E) → [S, d] = [6, 256]`
3. **Positional Encoding (`PE`):** фіксована матриця `[max_len, d]`
   - `PE[pos, 2i]   = sin( pos / 10000^(2i/d) )`
   - `PE[pos, 2i+1] = cos( pos / 10000^(2i/d) )`
   - `X_enc = X_enc + PE[:S, :] → [6, 256]`

### Decoder
1. **Вхідні токени:** `decoder_ids [T] = [10]`
2. - `X_dec = Embed(decoder_ids, E) → [T, d] = [10, 256]`
3. - `X_dec = X_dec + PE[:T, :] → [10, 256]`

> **Чому Sinusoidal, а не Learnable (як у v1.0)?**
> Encoder і decoder мають різні довжини послідовностей (S ≠ T).
> Sinusoidal PE — фіксована функція, працює на будь-якій довжині
> навіть якщо під час тренування модель не бачила такої довжини.

---

## 3. Encoder Block (Повторюється N=3 рази)

Приймає `X_enc [S, d] = [6, 256]`, повертає `X_enc [6, 256]`.

**Відмінність від v1.0:** Self-Attention без causal mask — кожен токен бачить усіх сусідів одночасно в обох напрямках.

### А. Multi-Head Self-Attention (bidirectional)

#### Проекції Q, K, V
Три незалежні матриці ваг `Wq_enc, Wk_enc, Wv_enc`, кожна `[d, d] = [256, 256]`:
```
Q_enc = LayerNorm(X_enc) * Wq_enc → [6, 256]
K_enc = LayerNorm(X_enc) * Wk_enc → [6, 256]
V_enc = LayerNorm(X_enc) * Wv_enc → [6, 256]
```

**Q** (Query) — "що шукаємо", **K** (Key) — "що пропонуємо", **V** (Value) — "що повертаємо".

> **Важливо:** Wq_enc, Wk_enc, Wv_enc — окремі матриці для encoder.
> Decoder має власні набори матриць (Wq_self, Wq_cross тощо) — детальніше в розділі 4.

#### Розбиття на h=4 голови
```
Q_enc → [h, S, d_k] = [4, 6, 64]
K_enc → [4, 6, 64]
V_enc → [4, 6, 64]
```

#### Scaled Dot-Product Attention (без маски)
```
Scores_enc = (Q_enc * K_enc^T) / sqrt(d_k)

Q_enc:      [4, 6, 64]
K_enc^T:    [4, 64, 6]   ← транспонуємо останні два виміри
Scores_enc: [4, 6, 6]    ← кожен токен до кожного

/ sqrt(64) = / 8         ← масштабування щоб градієнти не зникали
```

> **Чому ділимо на sqrt(d_k)?**
> При великих d_k dot product дає великі числа → softmax насичується →
> градієнти майже нулі → модель не вчиться. Ділення нормалізує масштаб.

**Без маски** — всі 6 токенів бачать один одного:
```
Weights_enc = Softmax(Scores_enc) → [4, 6, 6]   ← сума кожного рядка = 1
```

```
Context_enc = Weights_enc * V_enc → [4, 6, 6] * [4, 6, 64] = [4, 6, 64]
```

#### Конкатенація та проекція
```
Context_enc → [S, h*d_k] = [6, 256]      ← склеюємо голови
X_attn_enc  = Context_enc * Wo_enc → [6, 256]    (Wo_enc: [256, 256])
```

### Б. Residual + LayerNorm
```
X_enc = X_enc + X_attn_enc → [6, 256]
```

### В. Feed-Forward Network (FFN)
Матриці ваг `W1_enc [256, 512]` та `W2_enc [512, 256]`:
```
X_hidden = GELU( LayerNorm(X_enc) * W1_enc ) → [6, 512]
X_ffn    = X_hidden * W2_enc                 → [6, 256]
X_enc    = X_enc + X_ffn                     → [6, 256]   ← residual
```

### Г. Stacking (×3)
```
X_enc_1 = EncoderBlock_1(X_enc)     → [6, 256]
X_enc_2 = EncoderBlock_2(X_enc_1)   → [6, 256]
M       = EncoderBlock_3(X_enc_2)   → [6, 256]   ← memory
```

**`M` (memory)** — контекстуалізоване представлення кожного вхідного токена
з урахуванням усього речення. Передається в кожен Decoder Block.

---

## 4. Decoder Block (Повторюється N=3 рази)

Приймає `X_dec [T, d] = [10, 256]` та `M [S, d] = [6, 256]`.
Повертає `X_dec [10, 256]`.

**Кожен блок має три підшари** замість двох як у v1.0.

### А. Masked Multi-Head Self-Attention

Ідентично encoder self-attention, але з causal mask.
Матриці ваг: `Wq_self, Wk_self, Wv_self, Wo_self` — власні для decoder.

```
Q_self = LayerNorm(X_dec) * Wq_self → [10, 256]   ← від decoder
K_self = LayerNorm(X_dec) * Wk_self → [10, 256]   ← від decoder
V_self = LayerNorm(X_dec) * Wv_self → [10, 256]   ← від decoder

Q_self → [4, 10, 64]
K_self → [4, 10, 64]
V_self → [4, 10, 64]

Scores_self = (Q_self * K_self^T) / sqrt(64) → [4, 10, 10]
```

**Causal mask** — верхній трикутник заповнюємо `-inf`:
```
позиція 0 бачить: [<BOS>]
позиція 1 бачить: [<BOS>][❌]
позиція 2 бачить: [<BOS>][❌][Incorrect]
...
```

```
Weights_self = Softmax(Scores_self) → [4, 10, 10]
X_dec = X_dec + (Weights_self * V_self → [10, 256]) * Wo_self
```

### Б. Cross-Attention ← НОВИЙ ПІДШАР

**Це серце encoder-decoder.** Query від decoder, Key і Value від encoder memory `M`.
Матриці ваг: `Wq_cross, Wk_cross, Wv_cross, Wo_cross` — окремі від self-attention.

```
Q_cross = LayerNorm(X_dec) * Wq_cross → [10, 256]   ← від DECODER
K_cross = M * Wk_cross                → [6,  256]   ← від ENCODER (memory)
V_cross = M * Wv_cross                → [6,  256]   ← від ENCODER (memory)
```

Розбиваємо на голови:
```
Q_cross → [4, 10, 64]
K_cross → [4,  6, 64]
V_cross → [4,  6, 64]

Scores_cross = (Q_cross * K_cross^T) / sqrt(64)

Q_cross:        [4, 10, 64]
K_cross^T:      [4, 64,  6]
Scores_cross:   [4, 10,  6]   ← кожен з T=10 decoder токенів
                               дивиться на всі S=6 encoder токенів
```

**Без маски** — decoder може дивитись на весь вхід одразу:
```
Weights_cross = Softmax(Scores_cross) → [4, 10, 6]
Context_cross = Weights_cross * V_cross → [4, 10, 6] * [4, 6, 64] = [4, 10, 64]
Context_cross → [10, 256]                 ← склеюємо голови
X_dec = X_dec + Context_cross * Wo_cross → [10, 256]   ← residual
```

> **Інтуїція cross-attention:**
> На кожному кроці генерації decoder "запитує" encoder:
> "яка частина вхідного речення важлива зараз?"
> `Weights_cross [10, 6]` — це буквально матриця уваги decoder→encoder.
> Після тренування можна візуалізувати які вхідні токени були важливі
> для генерації кожного вихідного токена.

### В. Feed-Forward Network (FFN)
Ідентично encoder FFN, матриці ваг `W1_dec [256, 512]` та `W2_dec [512, 256]`:
```
X_hidden = GELU( LayerNorm(X_dec) * W1_dec ) → [10, 512]
X_ffn    = X_hidden * W2_dec                 → [10, 256]
X_dec    = X_dec + X_ffn                     → [10, 256]   ← residual
```

### Г. Stacking (×3)
```
X_dec_1 = DecoderBlock_1(X_dec, M)    → [10, 256]
X_dec_2 = DecoderBlock_2(X_dec_1, M)  → [10, 256]
X_dec_3 = DecoderBlock_3(X_dec_2, M)  → [10, 256]
```

`M` передається **незмінним** в кожен блок — encoder рахується один раз.

---

## 5. Вихід (LM Head)

Weight tying — використовуємо транспоновану матрицю Embedding `E`:
```
Logits = LayerNorm(X_dec_3) * E^T → [10, 256] * [256, 8000] = [10, 8000]
```

На кожній з T=10 позицій маємо 8000 оцінок — по одній на кожен токен словника.

---

## 6. Тренування (Teacher Forcing)

### Три тензори на один приклад
```
encoder_input:  tokenize("Heute ich gehe ins Kino.")
                → ids [S]

decoder_input:  tokenize("<BOS> ❌ Incorrect . ✅ Correct : Heute gehe ich ins Kino .")
                → ids [T]

decoder_target: tokenize("❌ Incorrect . ✅ Correct : Heute gehe ich ins Kino . <EOS>")
                → ids [T]   ← зсунуто на 1 вправо відносно decoder_input
```

### Зсув на 1 — чому
```
decoder_input:  [<BOS>][  ❌  ][Inc][.][✅][Correct]...[  .  ]
                  ↓       ↓     ↓   ↓   ↓    ↓            ↓
decoder_target: [  ❌  ][Inc][.][✅][Correct]...[  .  ][<EOS>]

позиція 0: бачу <BOS>       → маю передбачити ❌
позиція 1: бачу ❌          → маю передбачити Incorrect
позиція 2: бачу Incorrect   → маю передбачити .
...
```

Модель отримує правильні попередні токени (а не свої передбачення) — це дозволяє рахувати loss для всіх T позицій **паралельно** за один forward pass.

### Loss
```
M      = Encoder(encoder_input)
Logits = Decoder(decoder_input, M) → [T, 8000]

Loss = CrossEntropy(Logits, decoder_target)
     = mean( -log( softmax(Logits)[i, decoder_target[i]] ) for i in 0..T )
```

---

## 7. Інференс (Авторегресія)

Encoder рахується **один раз**, decoder — авторегресивно:

```python
M = encoder(encoder_input)      # один раз → [S, d]

y = [bos_id]
for _ in range(max_new_tokens):
    logits = decoder(y, M)      # M не змінюється
    next_id = sample(logits[-1])  # беремо останню позицію
    y.append(next_id)
    if next_id == eos_id:
        break

output = tokenizer.decode(y[1:])  # прибираємо <BOS>
```

---

## 8. Всі матриці ваг моделі

| Компонент | Матриці | Розмір | Кількість |
|---|---|---|---|
| Embedding | `E` | [8000, 256] | ×1 |
| Encoder ×3 | `Wq_enc, Wk_enc, Wv_enc, Wo_enc` | [256, 256] | ×4×3 = 12 |
| Encoder ×3 | `W1_enc, W2_enc` | [256,512]+[512,256] | ×2×3 = 6 |
| Decoder ×3 | `Wq_self, Wk_self, Wv_self, Wo_self` | [256, 256] | ×4×3 = 12 |
| Decoder ×3 | `Wq_cross, Wk_cross, Wv_cross, Wo_cross` | [256, 256] | ×4×3 = 12 |
| Decoder ×3 | `W1_dec, W2_dec` | [256,512]+[512,256] | ×2×3 = 6 |
| LM Head | `E^T` (weight tying) | [256, 8000] | ×0 (не нова) |

Всього ~50 матриць, ~6M параметрів, ~12 MB у float32.

### Порівняння з v1.0

| | v1.0 (Decoder-only) | v2.0 (Encoder-Decoder) |
|---|---|---|
| Архітектура | Decoder-only | Encoder + Decoder |
| Токенізатор | Word-level, 4000 слів | BPE, 8000 підслів |
| d_model | 128 | 256 |
| Шари | 4 decoder | 3 encoder + 3 decoder |
| Heads | 4 | 4 |
| Self-attention | causal mask | enc: bidirectional / dec: causal |
| Cross-attention | немає | є (Q від dec, K/V від enc) |
| Positional Enc | Learnable | Sinusoidal (фіксована) |
| Параметри | ~2.5M | ~6M |
| Розмір | ~2.5 MB | ~12 MB |
| Тензорів на приклад | 1 (input+output разом) | 3 (enc_input, dec_input, dec_target) |
