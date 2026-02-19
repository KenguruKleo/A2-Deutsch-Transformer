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
