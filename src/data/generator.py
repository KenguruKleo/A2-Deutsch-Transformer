import json
import random
from pathlib import Path
from src.config import load_config

class A2SmartGenerator:
    """Розумний генератор для створення великої кількості A2 прикладів."""
    
    def __init__(self):
        self.instruction = "You are a German A2 tutor. Check the sentence. If it is wrong, correct it and explain simply."
        
        # Словники для комбінування
        self.subjects = {
            "ich": {"bin": "bin", "habe": "habe", "ending": "e"},
            "du": {"bin": "bist", "habe": "hast", "ending": "st"},
            "er": {"bin": "ist", "habe": "hat", "ending": "t"},
            "sie": {"bin": "ist", "habe": "hat", "ending": "t"},
            "wir": {"bin": "sind", "habe": "haben", "ending": "en"},
            "ihr": {"bin": "seid", "habe": "habt", "ending": "t"},
        }
        
        self.time_adv = ["Heute", "Morgen", "Dann", "Jetzt", "Am Montag", "Nach der Arbeit", "Am Abend"]
        self.places = ["nach Hause", "nach Berlin", "ins Kino", "in die Schule", "zum Arzt", "nach München", "in den Park", "ins Restaurant"]
        self.foods = ["Pizza", "Brot", "Eis", "Kaffee", "Apfel", "Kuchen", "Suppe", "Bier", "Wasser", "Salat"]
        
    def get_verb_form(self, verb_stem, sub_key):
        """Returns the correct verb form based on the stem and subject."""
        # Special case for 'essen'
        if verb_stem == "ess":
            forms = {"ich": "esse", "du": "isst", "er": "isst", "sie": "isst", "wir": "essen", "ihr": "esst"}
            return forms.get(sub_key, "essen")
            
        ending = self.subjects[sub_key]["ending"]
        # Simplified logic for regular verbs
        if verb_stem.endswith('t') and ending in ['st', 't']:
            return verb_stem + 'e' + ending
        return verb_stem + ending

    def generate_perfekt(self, count=1000):
        """Генерує помилки haben/sein у Perfekt."""
        verbs_sein = [
            ("gehen", "gegangen"), ("fahren", "gefahren"), 
            ("kommen", "gekommen"), ("laufen", "gelaufen")
        ]
        verbs_haben = [
            ("essen", "gegessen"), ("trinken", "getrunken"), 
            ("machen", "gemacht"), ("kaufen", "gekauft")
        ]
        
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            sub = sub_key.capitalize()
            
            # Вибираємо тип дієслова (sein чи haben)
            is_movement = random.random() > 0.5
            verb_data = random.choice(verbs_sein if is_movement else verbs_haben)
            
            # Правильні допоміжні
            c_sein = self.subjects[sub_key]["bin"]
            c_haben = self.subjects[sub_key]["habe"]
            
            if is_movement:
                correct = f"{sub} {c_sein} {random.choice(self.places)} {verb_data[1]}."
                wrong = f"{sub} {c_haben} {random.choice(self.places)} {verb_data[1]}."
                expl = f"Дієслово '{verb_data[0]}' означає рух, тому в Perfekt використовуємо '{c_sein}' (від 'sein'), а не '{c_haben}'."
            else:
                correct = f"{sub} {c_haben} {random.choice(self.foods)} {verb_data[1]}."
                wrong = f"{sub} {c_sein} {random.choice(self.foods)} {verb_data[1]}."
                expl = f"Дієслово '{verb_data[0]}' потребує допоміжного '{c_haben}' (від 'haben') у минулому часі."

            data.append({
                "input": wrong,
                "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: {expl}"
            })
        return data

    def generate_wrong_partizip(self, count=1000):
        """Generates errors where Infinitiv is used instead of Partizip II."""
        verbs = [
            ("gehen", "gegangen", "bin"), ("fahren", "gefahren", "bin"),
            ("essen", "gegessen", "habe"), ("trinken", "getrunken", "habe"),
            ("machen", "gemacht", "habe"), ("kaufen", "gekauft", "habe"),
            ("sehen", "gesehen", "habe"), ("lesen", "gelesen", "habe")
        ]
        
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            sub = sub_key.capitalize()
            verb_inf, verb_p2, aux_type = random.choice(verbs)
            
            aux = self.subjects[sub_key][aux_type]
            obj = random.choice(self.places if aux_type == "bin" else self.foods)
            
            correct = f"{sub} {aux} {obj} {verb_p2}."
            wrong = f"{sub} {aux} {obj} {verb_inf}."
            
            expl = f"У минулому часі (Perfekt) основне дієслово має бути у формі Partizip II ('{verb_p2}'), а не в інфінітиві ('{verb_inf}')."
            
            data.append({
                "input": wrong,
                "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: {expl}"
            })
        return data

    def generate_inversion(self, count=1000):
        """Генерує помилки порядку слів (Inversion)."""
        simple_verbs = [
            ("spiel", "Fußball"), ("lern", "Deutsch"), 
            ("koch", "Suppe"), ("les", "ein Buch")
        ]
        
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            adv = random.choice(self.time_adv)
            verb_stem, obj = random.choice(simple_verbs)
            
            v_form = self.get_verb_form(verb_stem, sub_key)
            
            # Правильно: Adv + Verb + Subj + Obj
            correct = f"{adv} {v_form} {sub_key} {obj}."
            # Неправильно: Adv + Subj + Verb + Obj
            wrong = f"{adv} {sub_key} {v_form} {obj}."
            
            data.append({
                "input": wrong,
                "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: Коли речення починається з '{adv}', дієслово '{v_form}' має стояти на другому місці, перед підметом '{sub_key}'."
            })
        return data

    def generate_praesens_conjugation(self, count=1000):
        """Generates errors in present tense (Präsens) verb endings."""
        verbs = [
            ("spiel", "Fußball"), ("lern", "Deutsch"), 
            ("koch", "Suppe"), ("les", "ein Buch"),
            ("trink", "Kaffee"), ("kauf", "Brot"),
            ("ess", "Apfel"), ("trink", "Wasser")
        ]
        
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            sub = sub_key.capitalize()
            
            # Pick a verb
            verb_stem, obj = random.choice(verbs)
            
            # Correct form
            correct_verb = self.get_verb_form(verb_stem, sub_key)
            
            # Wrong form: intentionally pick a different person's ending
            wrong_sub_key = random.choice([k for k in self.subjects.keys() if k != sub_key])
            wrong_verb = self.get_verb_form(verb_stem, wrong_sub_key)
            
            correct = f"{sub} {correct_verb} {obj}."
            wrong = f"{sub} {wrong_verb} {obj}."
            
            expl = f"У теперішньому часі (Präsens) для підмета '{sub_key}' дієслово має закінчення '-{self.subjects[sub_key]['ending']}', тому правильно '{correct_verb}', а не '{wrong_verb}'."
            
            data.append({
                "input": wrong,
                "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: {expl}"
            })
        return data

    def generate_all(self):
        """Generates a large dataset."""
        dataset = []
        dataset.extend(self.generate_perfekt(3000))
        dataset.extend(self.generate_inversion(3000))
        dataset.extend(self.generate_wrong_partizip(3000))
        dataset.extend(self.generate_praesens_conjugation(3000))
        random.shuffle(dataset)
        return dataset

    def save(self, data, path="data/train.jsonl"):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"🚀 Сгенеровано {len(data)} прикладів у {path}")

if __name__ == "__main__":
    config = load_config()
    generator = A2SmartGenerator()
    data = generator.generate_all()
    # Split into training and validation (90/10)
    split = int(len(data) * 0.9)
    generator.save(data[:split], config.data.train_path)
    generator.save(data[split:], config.data.val_path)
