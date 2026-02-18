import json
import random
from pathlib import Path

class A2DataGenerator:
    """Генератор синтетичних даних для навчання A2 Grammar Tutor.
    
    Створює пари: 
    Неправильне речення (input) -> Виправлення + Пояснення (output)
    """
    
    def __init__(self):
        self.instruction = "You are a German A2 tutor. Check the sentence. If it is wrong, correct it and explain simply."
        
    def generate_perfekt_errors(self):
        """Помилки у минулому часі (Perfekt): плутанина haben/sein."""
        templates = [
            {
                "verb": "gehen",
                "correct_aux": "bin",
                "wrong_aux": "habe",
                "partizip": "gegangen",
                "context": "nach Hause",
                "explanation": "Дієслова руху (як 'gehen') використовують 'sein' у Perfekt."
            },
            {
                "verb": "essen",
                "correct_aux": "habe",
                "wrong_aux": "bin",
                "partizip": "gegessen",
                "context": "Pizza",
                "explanation": "Більшість дієслів (зокрема 'essen') використовують 'haben' у Perfekt."
            },
            {
                "verb": "fahren",
                "correct_aux": "sind",
                "wrong_aux": "haben",
                "partizip": "gefahren",
                "context": "nach Berlin",
                "explanation": "Дієслово 'fahren' позначає рух, тому потребує 'sein' (wir sind)."
            }
        ]
        
        data = []
        for t in templates:
            # Правильний варіант
            correct = f"Ich {t['correct_aux']} {t['context']} {t['partizip']}."
            if "sind" in t['correct_aux']: correct = f"Wir {t['correct_aux']} {t['context']} {t['partizip']}."
            
            # Неправильний варіант
            wrong = f"Ich {t['wrong_aux']} {t['context']} {t['partizip']}."
            if "sind" in t['correct_aux']: wrong = f"Wir {t['wrong_aux']} {t['context']} {t['partizip']}."
            
            data.append({
                "instruction": self.instruction,
                "input": wrong,
                "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: {t['explanation']}"
            })
        return data

    def generate_word_order_errors(self):
        """Помилки порядку слів (Inversion)."""
        data = []
        # Шаблон: Прислівник часу + підмет + дієслово (має бути дієслово на 2 місці)
        templates = [
            ("Heute", "ich gehe", "gehe ich", "ins Kino"),
            ("Dann", "wir spielen", "spielen wir", "Fußball"),
            ("Jetzt", "du trinkst", "trinkst du", "Kaffee")
        ]
        
        for adv, wrong_order, correct_order, rest in templates:
            wrong = f"{adv} {wrong_order} {rest}."
            correct = f"{adv} {correct_order} {rest}."
            data.append({
                "instruction": self.instruction,
                "input": wrong,
                "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: У німецькому реченні дієслово має стояти на другому місці (після '{adv}')."
            })
        return data

    def generate_all(self, count_per_type=100):
        """Збирає всі типи помилок разом."""
        all_data = []
        all_data.extend(self.generate_perfekt_errors())
        all_data.extend(self.generate_word_order_errors())
        
        # shuffle
        random.shuffle(all_data)
        return all_data

    def save_jsonl(self, data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"✅ Saved {len(data)} examples to {filename}")

if __name__ == "__main__":
    gen = A2DataGenerator()
    train_data = gen.generate_all()
    
    # Створюємо папку data якщо нема
    Path("data").mkdir(exist_ok=True)
    gen.save_jsonl(train_data, "data/train.jsonl")
