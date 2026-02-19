import random
from .base import BaseGenerator

class VerbGenerator(BaseGenerator):
    """Generates examples for Verb topics: Conjugation, Perfekt, Präteritum, Modal Verbs."""
    
    def generate_praesens(self, count=1000):
        """A1: Standard present tense conjugation errors."""
        verbs = [
            ("spiel", "Fußball"), ("lern", "Deutsch"), 
            ("koch", "Suppe"), ("trink", "Kaffee"), 
            ("kauf", "Brot"), ("ess", "Apfel")
        ]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            verb_stem, obj = random.choice(verbs)
            
            correct_v = self.get_verb_form(verb_stem, sub_key)
            wrong_sub = random.choice([k for k in self.subjects.keys() if k != sub_key])
            wrong_v = self.get_verb_form(verb_stem, wrong_sub)
            
            data.append({
                "input": f"{sub_key.capitalize()} {wrong_v} {obj}.",
                "output": f"❌ Incorrect.\n✅ Correct: {sub_key.capitalize()} {correct_v} {obj}.\n📝 Пояснення: У теперішньому часі (Präsens) для підмета '{sub_key}' дієслово має закінчення '-{self.subjects[sub_key]['ending']}', тому правильно '{correct_v}', а не '{wrong_v}'."
            })
        return data

    def generate_perfekt_aux(self, count=1000):
        """A2: Haben vs Sein errors in Perfekt."""
        verbs_sein = [("gehen", "gegangen"), ("fahren", "gefahren"), ("kommen", "gekommen")]
        verbs_haben = [("essen", "gegessen"), ("machen", "gemacht"), ("kaufen", "gekauft")]
        
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            is_movement = random.random() > 0.5
            verb_inf, verb_p2 = random.choice(verbs_sein if is_movement else verbs_haben)
            
            c_aux = self.subjects[sub_key]["bin" if is_movement else "habe"]
            w_aux = self.subjects[sub_key]["habe" if is_movement else "bin"]
            
            item = random.choice(self.nouns["place" if is_movement else "food"])[0]
            
            expl = f"Дієслово '{verb_inf}' {'означає рух' if is_movement else 'потребує допоміжного haben'}, тому використовуємо '{c_aux}', а не '{w_aux}'."
            
            data.append({
                "input": f"{sub_key.capitalize()} {w_aux} {item} {verb_p2}.",
                "output": f"❌ Incorrect.\n✅ Correct: {sub_key.capitalize()} {c_aux} {item} {verb_p2}.\n📝 Пояснення: {expl}"
            })
        return data

    def generate_partizip_forms(self, count=1000):
        """A2: Wrong Partizip II form (using Infinitiv instead)."""
        verbs = [("essen", "gegessen", "habe"), ("gehen", "gegangen", "bin"), ("sehen", "gesehen", "habe")]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            inf, p2, aux_type = random.choice(verbs)
            aux = self.subjects[sub_key][aux_type]
            obj = random.choice(self.nouns["food" if aux_type == "habe" else "place"])[0]
            
            data.append({
                "input": f"{sub_key.capitalize()} {aux} {obj} {inf}.",
                "output": f"❌ Incorrect.\n✅ Correct: {sub_key.capitalize()} {aux} {obj} {p2}.\n📝 Пояснення: У минулому часі (Perfekt) основне дієслово має бути у формі Partizip II ('{p2}'), а не в інфінітиві ('{inf}')."
            })
        return data

    def generate_modal_verbs(self, count=1000):
        """A1/A2: Modal verbs (können, müssen, wollen) conjugation and position."""
        modals = {
            "können": {"ich": "kann", "du": "kannst", "er": "kann", "sie": "kann", "wir": "können", "ihr": "könnt"},
            "müssen": {"ich": "muss", "du": "musst", "er": "muss", "sie": "muss", "wir": "müssen", "ihr": "müsst"},
            "wollen": {"ich": "will", "du": "willst", "er": "will", "sie": "will", "wir": "wollen", "ihr": "wollt"}
        }
        main_verbs = [("Deutsch sprechen", "sprechen"), ("nach Hause gehen", "gehen"), ("Suppe kochen", "kochen")]
        
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            m_inf = random.choice(list(modals.keys()))
            m_form = modals[m_inf][sub_key]
            phrase, v_inf = random.choice(main_verbs)
            
            # Error type 1: Wrong conjugation of modal
            wrong_sub = random.choice([k for k in self.subjects.keys() if k != sub_key])
            wrong_m = modals[m_inf][wrong_sub]
            
            data.append({
                "input": f"{sub_key.capitalize()} {wrong_m} {phrase}.",
                "output": f"❌ Incorrect.\n✅ Correct: {sub_key.capitalize()} {m_form} {phrase}.\n📝 Пояснення: Модальне дієслово '{m_inf}' для підмета '{sub_key}' має форму '{m_form}'."
            })
            
            # Error type 2: Main verb not at the end
            # "Ich kann sprechen Deutsch" instead of "Ich kann Deutsch sprechen"
            if " " in phrase:
                parts = phrase.split()
                wrong_phrase = f"{parts[1]} {parts[0]}" # "sprechen Deutsch"
                data.append({
                    "input": f"{sub_key.capitalize()} {m_form} {wrong_phrase}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {sub_key.capitalize()} {m_form} {phrase}.\n📝 Пояснення: У реченнях з модальним дієсловом ('{m_form}') основне дієслово ('{v_inf}') має стояти в самому кінці речення в інфінітиві."
                })
        return data

    def generate_separable_verbs(self, count=1000):
        """A2: Separable verbs (aufstehen, einkaufen) - prefix position in Präsens."""
        verbs = [
            ("aufstehen", "steh", "auf", "um 7 Uhr"),
            ("einkaufen", "kauf", "ein", "im Supermarkt"),
            ("anrufen", "ruf", "an", "meine Mutter")
        ]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            inf, stem, prefix, extra = random.choice(verbs)
            v_form = self.get_verb_form(stem, sub_key)
            
            # Correct: Ich stehe um 7 Uhr auf.
            # Wrong: Ich aufstehe um 7 Uhr.
            correct = f"{sub_key.capitalize()} {v_form} {extra} {prefix}."
            wrong = f"{sub_key.capitalize()} {prefix}{v_form} {extra}."
            
            data.append({
                "input": wrong,
                "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: Дієслово '{inf}' є відокремлюваним. У теперішньому часі приставка '{prefix}' має стояти в самому кінці речення."
            })
        return data

    def generate_reflexive_verbs(self, count=1000):
        """A2: Reflexive verbs (freuen sich, waschen sich)."""
        verbs = [
            ("freuen", "freue", "на відпустку", "freuen sich"),
            ("waschen", "wasche", "обличчя", "waschen sich"),
            ("ausruhen", "ruhe", "після роботи", "ausruhen sich")
        ]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            inf, stem, extra, full_inf = random.choice(verbs)
            v_form = self.get_verb_form(inf[:-2] if inf.endswith("en") else inf, sub_key)
            c_refl = self.reflexive_pronouns[sub_key]
            
            # Error: wrong reflexive pronoun
            wrong_sub = random.choice([k for k in self.subjects.keys() if k != sub_key])
            w_refl = self.reflexive_pronouns[wrong_sub]
            
            data.append({
                "input": f"{sub_key.capitalize()} {v_form} {w_refl} {extra}.",
                "output": f"❌ Incorrect.\n✅ Correct: {sub_key.capitalize()} {v_form} {c_refl} {extra}.\n📝 Пояснення: Дієслово '{full_inf}' вимагає зворотного займенника '{c_refl}' для підмета '{sub_key}'."
            })
        return data

    def generate_praeteritum_essentials(self, count=1000):
        """A2: Präteritum of sein (war) and haben (hatte)."""
        scenarios = [
            ("war", "sein", "вчора вдома"),
            ("hatte", "haben", "багато роботи")
        ]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            aux_type, inf, extra = random.choice(scenarios)
            c_form = self.subjects[sub_key][aux_type]
            
            # Error: wrong conjugation or confusing with Perfekt
            wrong_sub = random.choice([k for k in self.subjects.keys() if k != sub_key])
            w_form = self.subjects[wrong_sub][aux_type]
            
            data.append({
                "input": f"{sub_key.capitalize()} {w_form} {extra}.",
                "output": f"❌ Incorrect.\n✅ Correct: {sub_key.capitalize()} {c_form} {extra}.\n📝 Пояснення: У минулому часі (Präteritum) дієслово '{inf}' для '{sub_key}' має форму '{c_form}'."
            })
        return data

    def generate_imperativ(self, count=1000):
        """A1/A2: Imperativ (du, ihr, Sie forms)."""
        verbs = [
            ("gehen", "Geh", "Geht", "Gehen Sie"),
            ("machen", "Mach", "Macht", "Machen Sie"),
            ("kommen", "Komm", "Kommt", "Kommen Sie")
        ]
        data = []
        for _ in range(count):
            inf, du, ihr, sie = random.choice(verbs)
            
            # Error type 1: "Du gehst!" instead of "Geh!"
            data.append({
                "input": f"Du {inf[:-2]}st!",
                "output": f"❌ Incorrect.\n✅ Correct: {du}!\n📝 Пояснення: У наказовому способі (Imperativ) для 'du' закінчення '-st' та займенник 'du' відкидаються."
            })
            
            # Error type 2: "Ihr gehen!" instead of "Geht!"
            data.append({
                "input": f"Ihr {inf}!",
                "output": f"❌ Incorrect.\n✅ Correct: {ihr}!\n📝 Пояснення: У наказовому способі (Imperativ) для 'ihr' дієслово має закінчення '-t' (як у теперішньому часі), але без займенника 'ihr'."
            })
        return data
