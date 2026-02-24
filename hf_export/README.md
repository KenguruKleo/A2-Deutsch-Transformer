---
language:
- de
- uk
license: mit
pipeline_tag: text-generation
tags:
- grammar
- german
- deutsch
- transformer
- education
- pytorch
- gpt2
widget:
- text: "Ich habe den Auto."
  example_title: "Akkusativ correction"
- text: "Heute я gehe додому."
  example_title: "Mixed language / Mixed grammar"
- text: "Dann я bin gegangen."
  example_title: "Inversion / Perfekt"
model-index:
- name: Deutsch A2 Grammar Transformer
  results:
  - task:
      type: text-generation
      name: Grammar Correction
    metrics:
    - type: accuracy
      value: 99.6
      name: Detection Accuracy
---

# Deutsch A2 Grammar Transformer (Ukrainian Explanations)

This is a compact, custom-built Transformer Decoder model designed to identify, correct, and explain German grammar errors at the A1-A2 level. It is specifically tailored for **Ukrainian-speaking learners**, providing detailed grammar feedback and explanations in **Ukrainian**.

✨ **Live Demo (Spaces):** [kengurukleo/deutsch-a2-tutor](https://huggingface.co/spaces/kengurukleo/deutsch-a2-tutor)

## 🚀 Model Capabilities
The model covers over 18 essential grammar topics for A1 and A2 levels.

## 📚 Covered Topics & Examples

### 🟢 Level A1: Basic Grammar
| Topic | Level | Positive (✅) | Negative (❌) | Model Output |
|:---|:---:|:---|:---|:---|
| **Präsens** | A1 | Ich esse Brot. | Ich isst Brot. | ✅ Correct: Ich esse Brot. <br> 📝 Пояснення: У теперішньому часі (Präsens) для підмета 'ich' дієслово має закінчення '-e', тому правильно 'esse', а не 'isst'. |
| **W-Fragen** | A1 | Wo wohnst du? | Wo du wohnst? | ✅ Correct: Wo wohnst du? <br> 📝 Пояснення: У запитаннях після питального слова 'Wo' дієслово 'wohnst' має стояти на другому місці, перед підметом 'du'. |
| **Akkusativ** | A1 | Ich habe den Tisch. | Ich habe der Tisch. | ✅ Correct: Ich habe den Tisch. <br> 📝 Пояснення: Дієслово 'habe' вимагає Akkusativ. Для чоловічого роду артикль 'der' змінюється на 'den'. |
| **Negation** | A1 | Ich habe kein Auto. | Ich habe nicht Auto. | ✅ Correct: Ich habe kein Auto. <br> 📝 Пояснення: Для заперечення іменників (без означеного артикля) використовується 'kein', а не 'nicht'. |
| **Imperativ** | A1 | Komm! | Du kommst! | ✅ Correct: Komm! <br> 📝 Пояснення: У наказовому способі (Imperativ) для 'du' закінчення '-st' та займенник 'du' відкидаються. |

### 🟡 Level A1/A2: Intermediate Topics
| Topic | Level | Positive (✅) | Negative (❌) | Model Output |
|:---|:---:|:---|:---|:---|
| **Modalverben** | A1/A2 | Ich kann Deutsch sprechen.| Ich kann sprechen Deutsch. | ✅ Correct: Ich kann Deutsch sprechen. <br> 📝 Пояснення: У реченнях з модальним дієсловом ('kann') основне дієслово ('sprechen') має стояти в самому кінці речення в інфінітиві. |
| **Possessivpron.**| A1/A2 | Das ist mein Bruder.| Das ist meine Bruder. | ✅ Correct: Das ist mein Bruder. <br> 📝 Пояснення: Для m-роду ('Bruder') присвійний займенник 'mein' не повинен мати закінчення '-e' у початковій формі (Nominativ). |
| **Fixed Prepos.** | A1/A2 | Mit dem Bus. | Mit den Bus. | ✅ Correct: Mit dem Bus. <br> 📝 Пояснення: Прийменник 'mit' завжди вимагає Dativ. Тому артикль має бути 'dem'. |

### 🔵 Level A2: Advanced Grammar
| Topic | Level | Positive (✅) | Negative (❌) | Model Output |
|:---|:---:|:---|:---|:---|
| **Perfekt** | A2 | Ich bin gegangen. | Ich habe gegangen. | ✅ Correct: Ich bin gegangen. <br> 📝 Пояснення: Дієслово 'gehen' означає рух, тому використовуємо 'bin', а не 'habe'. |
| **Inversion** | A2 | Heute gehe ich. | Heute ich gehe. | ✅ Correct: Heute gehe ich. <br> 📝 Пояснення: Коли речення починається з 'Heute', дієслово 'gehe' має стояти на другому місці, перед підметом 'ich'. |
| **Separable Verbs**| A2 | Ich stehe auf. | Ich aufstehe. | ✅ Correct: Ich stehe на 7 Uhr auf. <br> 📝 Пояснення: Дієслово 'aufstehen' є відокремлюваним. У теперішньому часі приставка 'auf' має стояти в самому кінці речення. |
| **Dativ** | A2 | Ich helfe dem Kind. | Ich helфе das Kind. | ✅ Correct: Ich helfe dem Kind. <br> 📝 Пояснення: Дієслово 'helfen' завжди вимагає Dativ. Тому артикль для n-роду має бути 'dem'. |
| **Wechselpräp.** | A2 | In das Kino. | In dem Kino (to). | ✅ Correct: In das Kino. <br> 📝 Пояснення: Прийменник 'in' у значенні 'Куди? (центр)' вимагає Akkusativ. Для n-роду це 'das'. |
| **Nebensätze** | A2 | ...weil er krank ist. | ...weil er ist krank. | ✅ Correct: ...weil er krank ist. <br> 📝 Пояснення: У підрядному реченні зі сполучником 'weil' дієслово 'ist' має стояти в самому кінці речення. |
| **Reflexive V.** | A2 | Ich freue mich. | Ich freue dich. | ✅ Correct: Ich freue mich. <br> 📝 Пояснення: Дієслово 'freuen sich' вимагає зворотного займенника 'mich' для підмета 'ich'. |
| **Adjektivdekl.** | A2 | Ein guter Mann. | Ein gut Mann. | ✅ Correct: Ein guter Mann. <br> 📝 Пояснення: Після неозначеного артикля 'ein' у Nominativ прикметник 'gut' для m-роду отримує закінчення '-er'. |
| **Komparation** | A2 | Das ist besser. | Das ist mehr gut. | ✅ Correct: Das ist besser. <br> 📝 Пояснення: У німецькій мові ступені порівняння утворюються за допомогою суфіксів (або зміни кореня), а не словом 'mehr'. |
| **Präteritum** | A2 | Ich war zu Hause. | Ich waren zu Hause. | ✅ Correct: Ich war zu Hause. <br> 📝 Пояснення: У минулому часі (Präteritum) дієслово 'sein' для 'ich' має форму 'war'. |

## 🛠 Architecture
- **Type:** Transformer Decoder (GPT-2 style)
- **Parameters:** ~5.5M
- **Layers:** 4
- **Attention Heads:** 4
- **Embedding Dim:** 128
- **Tokenizer:** Byte-level BPE (8,000 tokens)

## 📖 How to Use
The model is fully compatible with the standard `transformers` library. No custom code is required.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_id = "kengurukleo/deutsch_a2_transformer"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Test generation
text = "Ich habe den Auto."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 👨‍💻 Author
Created by [KenguruKleo](https://github.com/KenguruKleo).
