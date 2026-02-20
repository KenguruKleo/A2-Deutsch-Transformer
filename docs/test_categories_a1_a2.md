# Покриття тестів за категоріями типових помилок A1–A2

Тести в `tests/test_data.json` розбиті за темами. Нижче — відповідність **найпоширеніших помилок** (за категоріями) і наших тестів.

---

## 1. Порядок слів (Wortstellung)

| Тип помилки | Що перевіряємо | Приклад тестів (id) |
|-------------|----------------|----------------------|
| Дієслово не на 2-му місці | Inversion: після обставини дієслово друге | 42–47: „Heute spiele ich” ✅, „Heute ich spiele” ❌ |
| W-питання: дієслово перед підметом | Questions | 51–54, 153–156: „Wo wohnst du?” ✅, „Wo du wohnst?” ❌ |
| Рамкова конструкція: інфінітив в кінці | Modal: „muss/ will/ kann + … + Infinitiv” | 26–27, 59, **230–233**: „Ich kann Deutsch sprechen” ✅, „Ich kann sprechen Deutsch” ❌; „Ich muss heute nach Hause gehen” ✅, „Ich muss heute gehen nach Hause” ❌ |
| Дієслово в кінці в підрядному | Nebensatz (weil, dass, wenn) | 48–50, 55–59, 151–152, 157–160: „weil ich Hunger habe” ✅, „weil ich habe Hunger” ❌ |

---

## 2. Артиклі та відмінки (Kasus)

| Тип помилки | Що перевіряємо | Приклад тестів (id) |
|-------------|----------------|----------------------|
| Плутанина Dativ / Akkusativ | Після дієслова: helfen, danken → Dativ; haben, sehen → Akkusativ | 66–69, 70–74, 101–105, 165–168, 206–207, 208–209 |
| mit + Dativ, für + Akkusativ | Fixed prepositions | 91–94, 213–214, 216–217, **244–246**: „mit dem Freund” ✅, „mit den Freund” ❌; „für den Freund” ✅, „für dem Freund” ❌ |
| Рід іменника (der/die/das) | Nominativ, Akkusativ, правильний артикль за родом | 218–223 (Nominativ), 66–69, 206–207 (Akk: „das Auto”, не „die/der Auto”) |
| Пропуск артикля | Після haben/brauchen злічний іменник з артиклем | **228–229**: „Ich habe Auto” ❌ → „Ich habe ein Auto” ✅ |
| Wechselpräpositionen (in/auf) | Akk = куди?, Dat = де? | 75–78, 210–212, 173–174: „in das Kino” (куди) ✅, „in dem Kino” (де) ✅ |

---

## 3. Дієслова (Verben)

| Тип помилки | Що перевіряємо | Приклад тестів (id) |
|-------------|----------------|----------------------|
| Відмінювання в Präsens | Правильні закінчення -e/-st/-t/-en | 1–6, 96, 106–110, 107 |
| Сильні дієслова: зміна голосної | schlafen → du schläfst, er schläft; fahren → du fährst | **234–239**: „Du schläfst” ✅, „Du schlafst” ❌; „Er schläft” ✅, „Er schlaft” ❌; „Du fährst” ✅, „Du fahrst” ❌ |
| haben/sein у Präsens | bin/hast/ist тощо | 7–14, 111–114, 193, 202 |
| Perfekt: haben vs sein | Рух/зміна стану → sein | 15–21, 98, 119, **240–243**: „Ich bin gefahren” ✅, „Ich habe gefahren” ❌; „Sie ist gekommen” ✅, „Sie hat gekommen” ❌; „Ich bin geblieben” ✅, „Ich habe geblieben” ❌ |
| Partizip II в кінці | Рамка: „haben/sein + … + Partizip II” | 22–25, 121–123 |
| Модальні: форма та інфінітив в кінці | kann/muss/will + Infinitiv в кінці | 26–30, 124–128, 230–233 |
| Відокремлювані дієслова | Приставка в кінці в Präsens | 31–34, 129–132 |
| Рефлексиви | mich/dich/sich після дієслова | 35–36, 133–134 |
| Imperativ | Geh! / Macht! / Gehen Sie! без du/ihr | 40–41, 140–141 |
| Präteritum sein/haben | war/hatte, waren/hatten | 37–39, 136–139 |

---

## 4. Прийменники та частинки

| Тип помилки | Що перевіряємо | Приклад тестів (id) |
|-------------|----------------|----------------------|
| nicht vs kein | kein + іменник без артикля; nicht для прикметника/дієслова | 60–65, 161–164, **247–248**: „Ich habe kein Auto” ✅, „Ich habe nicht Auto” ❌; „Ich habe kein Geld” ✅, „Ich habe nicht Geld” ❌; „Das ist nicht gut” ✅, „Das ist kein gut” ❌ |
| „На роботі” = bei/auf der Arbeit | Не „in der Arbeit” у значенні „на роботі” | **244–246**: „Ich bin bei der Arbeit” ✅, „Ich bin in der Arbeit” ❌; „Er ist auf der Arbeit” ✅ |
| Фіксовані прийменники (Dativ/Akkusativ) | mit, nach, von, bei + Dat; für, ohne + Akk | 91–95, 187–190, 213–217 |

---

## Нові тести (id 230–248)

Додано для повного покриття категорій:

- **230–233**: Wortstellung — рамка з модальним („will Deutsch lernen”, „muss nach Hause gehen”).
- **234–239**: Strong verbs — schlafen (schläfst/schläft), fahren (fährst).
- **240–243**: Perfekt — kommen/bleiben + sein („Sie ist gekommen”, „Ich bin geblieben”).
- **244–246**: Прийменник „на роботі” — bei der Arbeit / auf der Arbeit, не „in der Arbeit”.
- **247–248**: Negation — „Ich habe kein Geld” ✅, „Ich habe nicht Geld” ❌.

Усього тестів: **248** (було 229).
