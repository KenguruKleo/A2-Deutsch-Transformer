# 🇩🇪 A2 Deutsch Grammar Tutor

Компактна Transformer-модель для перевірки та пояснення помилок у німецьких реченнях рівня A2.

## Що вміє модель

| Функція | Приклад |
|---|---|
| ✅ Визначає правильність | `Ich bin nach Hause gegangen.` → ✅ |
| ❌ Виправляє помилки | `Ich habe gegangen` → `Ich bin gegangen` |
| 📝 Пояснює | «gehen» вживається з «sein» у Perfekt |

### Покриті теми A2

- **Perfekt** — haben/sein + Partizip II
- **Модальні дієслова** — können, müssen, dürfen, wollen, sollen, mögen
- **Відокремлювані дієслова** — aufstehen, einkaufen…
- **Порядок слів** — інверсія, weil + дієслово наприкінці
- **Dativ / Akkusativ** — mit, zu, bei, für, ohne…
- **Заперечення** — kein vs. nicht, позиція
- **Займенники** — присвійні та особові
- **Порівняння** — größer als, am besten

## Архітектура

```
Transformer Decoder (causal LM)
├── V = 2 000 токенів (word-level + часті форми)
├── T = 64  (контекст)
├── d_model = 128
├── L = 4 блоки
├── H = 4 голови  (d_head = 32)
├── d_ff = 512
├── Weight tying = ON
└── FP16 → модель ≤ 3 MB
```

## Структура проєкту

```
A2-Deutsch-Transformer/
├── config.yaml          # гіперпараметри
├── model.py             # Transformer decoder
├── tokenizer.py         # word-level токенізатор (V=2000)
├── data_gen.py          # генератор синтетичних A2 помилок
├── train.py             # тренувальний цикл (AMP / FP16)
├── generate.py          # інференс з temperature sampling
├── data/
│   ├── train.jsonl      # тренувальні дані
│   └── val.jsonl        # валідаційні дані
├── checkpoints/         # збережені ваги
└── requirements.txt
```

## Швидкий старт

```bash
# 1. Клонувати
git clone https://github.com/KenguruKleo/A2-Deutsch-Transformer.git
cd A2-Deutsch-Transformer

# 2. Створити середовище
python3 -m venv .venv
source .venv/bin/activate

# 3. Встановити залежності
pip install -r requirements.txt

# 4. Згенерувати дані
python data_gen.py

# 5. Тренувати
python train.py

# 6. Спробувати
python generate.py --prompt "Ich habe gegangen nach Hause."
```

## Формат даних (JSONL)

```json
{
  "instruction": "You are a German A2 tutor. Check the sentence. If it is wrong, correct it and explain simply.",
  "input": "Ich habe gegangen nach Hause.",
  "output": "❌ Incorrect.\n✅ Correct: Ich bin nach Hause gegangen.\n📝 Explanation: «gehen» вживає «sein» у Perfekt: ich bin gegangen."
}
```

## Навчання

- **Де:** Google Cloud — 1× NVIDIA T4 (16 GB VRAM)
- **Бюджет:** ~$20
- **Precision:** mixed precision (FP16 / AMP)
- **Дані:** 20k–50k синтетичних + 1k–3k ручних прикладів

## Ліцензія

MIT
