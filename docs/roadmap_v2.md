# Roadmap: v2.0 — Encoder-Decoder + SentencePiece

## Поточний стан (v1.0)

| Компонент | Реалізація |
|---|---|
| Архітектура | Decoder-only (GPT-стиль) |
| Токенізатор | Word-level, фіксований словник 4000 слів |
| Рівень | A2 |
| Точність | 99% на синтетичних тестах, низька на реальних реченнях |

**Головна проблема v1.0:** модель вивчила таблицю шаблонів, а не граматику. Реальні речення від користувачів часто падають в `<UNK>` або не розпізнаються.

---

## Крок 1 — SentencePiece токенізатор

### Чому

Word-level токенізатор не може обробити:
- Слова поза словником → `<UNK>`
- Опечатки користувача (`gehts` замість `gehst`)
- Нові слова B1 рівня без перебудови словника
- Складні німецькі композити (`Donaudampfschifffahrt`)

### Що таке SentencePiece + BPE

BPE (Byte Pair Encoding) — алгоритм що будує словник підслів з даних:
1. Починає з окремих символів
2. Ітеративно зливає найчастіші пари символів в один токен
3. Зупиняється коли досягнуто розмір словника

SentencePiece — бібліотека Google що реалізує BPE/Unigram без залежності від пробілів як роздільників (підходить для будь-якої мови).

```
Word-level:  "gegangen" → [<UNK>]  (якщо не в словнику)
SentencePiece: "gegangen" → ["▁geg", "angen"]  (завжди розбере)
```

### План реалізації

- [ ] `src/tokenizer/train_tokenizer.py` — тренування SentencePiece на `data/train.jsonl`
- [ ] `src/tokenizer/tokenizer.py` — замінити word-level на SP wrapper
- [ ] Розмір словника: 8000-16000 токенів
- [ ] Зберегти API сумісним (`encode`, `decode`, `pad_sequence`)

---

## Крок 2 — Encoder-Decoder архітектура

### Чому

Поточний decoder-only обробляє вхід і вихід як одну послідовність:
```
[вхід][вихід] → авторегресія
```

Для задачі виправлення граматики природніша seq2seq структура:
```
Encoder: читає вхід ПОВНІСТЮ (бідирекціонально)
Decoder: генерує вихід, звертаючись до encoder через cross-attention
```

Encoder бачить одразу весь контекст речення — це критично для розуміння помилок де важливо і що до, і що після слова.

### Архітектура (як в "Attention is All You Need", 2017)

```
Encoder Stack (N layers):
  └── Multi-Head Self-Attention (bidirectional, без causal mask)
  └── Add & Norm
  └── Feed-Forward Network
  └── Add & Norm

Decoder Stack (N layers):
  └── Masked Multi-Head Self-Attention (causal, як в v1.0)
  └── Add & Norm
  └── Cross-Attention (queries від decoder, keys/values від encoder output)
  └── Add & Norm
  └── Feed-Forward Network
  └── Add & Norm

Shared Embedding + LM Head (weight tying)
```

### Новий компонент: Cross-Attention

Це ключова різниця від v1.0. У cross-attention:
- **Query** = поточний стан decoder (що ми генеруємо)
- **Key, Value** = output encoder (закодований вхід)

Тобто на кожному кроці генерації decoder "дивиться" на весь вхід і вирішує яка частина речення важлива зараз.

### Гіперпараметри (орієнтовні)

| Параметр | v1.0 | v2.0 |
|---|---|---|
| Архітектура | Decoder-only | Encoder-Decoder |
| Токенізатор | Word-level 4k | SentencePiece 8-16k |
| d_model | 128 | 256 |
| Layers (enc/dec) | 4 | 4 / 4 |
| Heads | 4 | 8 |
| d_ff | 512 | 1024 |
| Розмір моделі | ~2.5 MB | ~15-20 MB |

### План реалізації

- [ ] `src/model/encoder.py` — EncoderBlock + EncoderStack
- [ ] `src/model/decoder.py` — DecoderBlock (з cross-attention) + DecoderStack
- [ ] `src/model/model.py` — EncoderDecoderModel (замінює TransformerModel)
- [ ] `src/train.py` — оновити під seq2seq (encoder_input / decoder_input / labels)
- [ ] `src/inference.py` — оновити генерацію (encoder forward один раз, decoder авторегресивно)

---

## Крок 3 — Дані для B1

- [ ] Розширити генератори на B1 теми (Konjunktiv II, Passiv, Relativsätze...)
- [ ] Додати реальні приклади помилок від людей (не тільки синтетика)
- [ ] Розглянути аугментацію: один вхід → кілька варіантів помилок

---

## Порядок виконання

```
next branch
│
├── 1. train_tokenizer.py + новий tokenizer.py
├── 2. оновити generator.py → генерувати з новим токенізатором
├── 3. EncoderBlock + DecoderBlock (з cross-attention)
├── 4. EncoderDecoderModel
├── 5. оновити train.py
├── 6. оновити inference.py + generate.py
├── 7. тести для нової архітектури
└── 8. merge → main, tag v2.0
```
