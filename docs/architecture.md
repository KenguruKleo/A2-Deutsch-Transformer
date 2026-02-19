# Математична архітектура моделі

Цей документ описує шлях даних крізь модель Transformer, використовуючи матричні розмірності та математичні операції.

## Позначення
- `[R, C]` — матриця розмірністю R рядків та C стовпчиків.
- `*` — матричне множення.
- `+` — елементне додавання.
- `^T` — транспонування матриці.

---

## 1. Вхід (Input & Embeddings)

1. **Вхідні Tokens:** Список ID слів `ids` розміром `[64]`.
2. **Token Embedding Matrix (`E`):** Таблиця ваг розміром `[4000, 128]`.
   - `X_token = Embed(ids, E) -> [64, 128]`
3. **Positional Encoding Matrix (`P`):** Фіксована матриця позицій розміром `[64, 128]`.
   - `X_inp = X_token + P -> [64, 128]`

---

## 2. Transformer Block (Повторюється 4 рази)

Кожен блок приймає `X_inp [64, 128]` та видає `X_out [64, 128]`.

### А. Self-Attention (Механізм уваги)

Ми використовуємо **Multi-Head Attention**, але для розуміння спочатку розглянемо спрощений варіант з однією "головою".

#### 1. Single-Head Attention (Спрощена теорія)
Тут ми розглядаємо весь простір ембедінгу (128 вимірів) як одне ціле.

- **Query, Key, Value:**
  - `Q = X_inp * W_Q -> [64, 128]`
  - `K = X_inp * W_K -> [64, 128]`
  - `V = X_inp * W_V -> [64, 128]`
- **Attention Scores:**
  - `Scores = (Q * K^T) / sqrt(128) -> [64, 64]`
  - `Weights = Softmax(Scores) -> [64, 64]`
- **Context:**
  - `X_attn = Weights * V -> [64, 128]`

#### 2. Multi-Head Attention (Реальна реалізація)
Цей механізм дозволяє моделі фокусуватися на різних типах зв'язків одночасно через 4 незалежні "голови".

- **Splitting (Розділення):**
  - Вхідна матриця `X_inp [64, 128]` розділяється поперек виміру ембедінгу на 4 голови.
  - Кожна голова працює з під-матрицею розміром `[64, 32]`, де `32 = 128 / 4`.
- **Процес для кожної голови (Head_i):**
  - `q_i, k_i, v_i = [64, 32]`
  - `scores_i = (q_i * k_i^T) / sqrt(32) -> [64, 64]`
  - `weights_i = Softmax(scores_i) -> [64, 64]`
  - `out_i = weights_i * v_i -> [64, 32]`
- **Concatenation & Projection:**
  - Всі 4 виходи з'єднуються назад: `Context = [out_1, out_2, out_3, out_4] -> [64, 128]`.
  - **Output Projection:** `X_attn = Context * W_O -> [64, 128]`.
    *(Матриця `W_O` має розмір `[128, 128]`)*.

### Б. Residual + LayerNorm (1)
- `X_res1 = X_inp + X_attn -> [64, 128]`
- `X_norm1 = LayerNorm(X_res1) -> [64, 128]`

### В. Feed-Forward Network (FFN)
Використовуються дві матриці ваг: `W1 [128, 512]` та `W2 [512, 128]`.

1. **Expansion:**
   - `X_hidden = X_norm1 * W1 -> [64, 512]`
2. **Activation:**
   - `X_act = GELU(X_hidden) -> [64, 512]`
3. **Compression:**
   - `X_ffn = X_act * W2 -> [64, 128]`

### Г. Residual + LayerNorm (2)
- `X_res2 = X_norm1 + X_ffn -> [64, 128]`
- `X_final_block = LayerNorm(X_res2) -> [64, 128]`

---

## 3. Stacking (Послідовне поєднання)

Модель складається з 4-х Transformer Blocks, де вихід попереднього є входом для наступного:
1. `X_block1 = Transformer Block 1 (X_inp)`
2. `X_block2 = Transformer Block 2 (X_block1)`
3. `X_block3 = Transformer Block 3 (X_block2)`
4. `X_block4 = Transformer Block 4 (X_block3)`

В результаті отримуємо фінальну матрицю ознак `X_final_features` розміром `[64, 128]`.

---

## 4. Вихід (LM Head)

Для отримання ймовірностей слів ми використовуємо ту саму матрицю Embeddings (`E`), але транспоновану.

- **Weight Tying:** Використовуємо `E^T [128, 4000]`.
- **Final Projection:**
  - `Logits = X_final_features * E^T -> [64, 4000]`

Для кожної з 64 позицій ми маємо 4000 оцінок. Слово з найвищою оцінкою стає результатом генерації.
