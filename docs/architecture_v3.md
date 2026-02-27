# Архітектура v3.0 — LLM-backed Data Pipeline

Цей документ описує архітектуру **версії 3.0**, зосереджуючись на новому підході до генерації тренувальних даних.

- Архітектура моделі (Encoder-Decoder Transformer) та математичні деталі: [architecture_v2.md](architecture_v2.md)
- Порівняння v1.0 та v2.0: [architecture_v2.md → Порівняння](architecture_v2.md#порівняння-з-v10)

---

## Проблема з v2.0

Попередня версія використовувала **шаблонний генератор** (`src/data/generators/`), який будував речення з фіксованого словника (~20 іменників, ~10 прислівників). Модель не вивчала граматику — вона запам'ятовувала поверхневі патерни:

```
"Heute" на початку → завжди ✅ Correct
"ich gehe" → завжди ✅ Correct
```

Результат: модель відповідала `✅ Correct.` навіть на явно помилкові `Wo wohnst Ich?`.

---

## Рішення v3.0 — LLM-backed генерація

Генератор замінено на систему, що викликає **локальну або хмарну LLM** для створення різноманітних, реалістичних прикладів.

```
┌──────────────────────────────────────────────────────┐
│                  generate_data.py (CLI)               │
│  --provider ollama | openai   --count N   --append   │
└────────────────────┬─────────────────────────────────┘
                     │ будує
          ┌──────────▼──────────┐
          │   OllamaGenerator   │
          │   (orcherstrator)   │
          └──────────┬──────────┘
                     │ uses
          ┌──────────▼──────────┐
          │  BaseLLMProvider    │  ← ABC
          └──────┬──────┬───────┘
                 │      │
        ┌────────▼─┐  ┌─▼──────────┐
        │ Ollama   │  │  OpenAI    │
        │ Provider │  │  Provider  │
        └──────────┘  └────────────┘
                     │ produces
          ┌──────────▼──────────┐
          │   data/train.jsonl  │  ← append-mode, safe to interrupt
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  HF datasets        │
          │  train_test_split() │  ← автоматичний 90/10 сплiт при навчанні
          └─────────────────────┘
```

---

## Компоненти

### `src/data/llm_provider.py` — Abstract Base

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str | None: ...
```

Будь-який новий бекенд (Anthropic, Mistral, LM Studio тощо) потребує лише реалізацію цього одного методу.

### `src/data/providers/ollama.py` — OllamaProvider

Викликає локальний Ollama через REST API (`/api/generate`, `stream=False`).

> **Важливо:** `num_predict` не обмежується. Reasoning-моделі (Qwen3, DeepSeek-R1) використовують сотні токенів для "думок" перед відповіддю. Ліміт відрізає думки → порожня відповідь.

### `src/data/providers/openai.py` — OpenAIProvider

Використовує офіційний `openai` SDK. Підтримує будь-який `base_url`:

| Сервіс | base_url |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| LM Studio | `http://localhost:1234/v1` |
| Ollama OpenAI layer | `http://localhost:11434/v1` |

API ключ завжди читається з env-змінної (не з коду чи конфігу).

### `src/data/llm_generator.py` — LLMGenerator (OllamaGenerator)

Оркеструє генерацію:
- Вибирає тему випадково (`_pick_topic()`)
- Вибирає режим: `incorrect` (60%) або `correct` (40%)
- Будує промпт (`build_prompt(topic, mode)`)
- Викликає `provider.complete(prompt)`
- Парсить JSON відповідь, валідує emoji-маркери (`❌`/`✅`, `📝`)
- При невалідній відповіді — retry (до `max_retries` разів)
- Записує в JSONL файл з `f.flush()` після кожного прикладу (safe interrupt)

### `src/data/llm_prompts.py` — Теми та промпти

**20 тем** у 4 категоріях:

| Категорія | Теми |
|---|---|
| Дієслова | `verb_conjugation`, `haben_sein`, `modal_verbs`, `separable_verbs`, `reflexive_verbs`, `perfekt`, `imperativ` |
| Синтаксис | `word_order_inversion`, `questions`, `subordinate_weil`, `subordinate_dass_wenn`, `negation` |
| Відмінки | `nominativ`, `akkusativ`, `dativ`, `genitiv`, `adjective_endings`, `prepositions` |
| Артиклі | `definite_articles`, `indefinite_articles`, `possessive_pronouns` |

---

## Формат даних (незмінний)

```jsonl
{"input": "Er lerne Deutsch.", "output": "❌ Incorrect.\n✅ Correct: Er lernt Deutsch.\n📝 Пояснення: У третій особі однини дієслово отримує закінчення -t."}
{"input": "Sie wohnt in Berlin.", "output": "✅ Correct.\n📝 Пояснення: Правильна форма 'wohnt' для третьої особи однини."}
```

Ключова зміна: `✅ Correct` тепер **також містить пояснення** (`📝`) — модель вчиться не тільки виявляти помилки, а й пояснювати правильні конструкції.

---

## Тренувальний пайплайн

```
data/train.jsonl
      ↓
datasets.load_dataset("json", ...)
      ↓
.train_test_split(test_size=0.1, seed=42, shuffle=True)
      ↓                              ↓
train split (90%)             val split (10%)
      ↓                              ↓
Seq2SeqDataset              Seq2SeqDataset
      ↓
train.py (existing Encoder-Decoder training loop)
```

Сплiт відбувається **під час завантаження** в `train.py`, а не при генерації. Це дозволяє:
- Мати один файл `data/train.jsonl`
- Довільно змінювати `val_split` у `config.yaml` без регенерації даних
- Безпечно перервати і продовжити генерацію (`--append`)

---

## Конфігурація (`config.yaml`)

```yaml
data:
  train_path: "data/train.jsonl"
  val_split: 0.1              # автоматичний сплiт при навчанні

ollama:
  host: "http://localhost:11434"
  model: "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M"
  ratio_incorrect: 0.6
  timeout_seconds: 90
  max_retries: 3

openai:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"
  temperature: 0.7
  max_tokens: 512
```

---

## Швидкий старт

```bash
# Генерація з Ollama (default)
python -m src.data.generate_data --count 5000

# Продовжити перерваний сеанс
python -m src.data.generate_data --count 2000 --append

# Згенерувати через OpenAI
OPENAI_API_KEY=sk-... python -m src.data.generate_data --count 5000 --provider openai

# Валідація датасету
python -m src.data.validate_dataset --input data/train.jsonl

# Перенавчання токенізатора (обов'язково перед train.py після нової генерації)
python src/tokenizer/train_tokenizer.py

# Навчання
python src/train.py
```
