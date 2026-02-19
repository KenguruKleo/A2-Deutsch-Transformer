import random
from .base import BaseGenerator

class CaseGenerator(BaseGenerator):
    """Generates examples for Cases (Akkusativ, Dativ) and Articles."""

    def generate_akkusativ_masculine(self, count=1000):
        """A1: Akkusativ masculine (der -> den)."""
        verbs = ["suche", "sehe", "kaufe", "brauche"]
        nouns_m = ["Apfel", "Schlüssel", "Computer", "Tisch", "Hund"]
        
        data = []
        for _ in range(count):
            verb = random.choice(verbs)
            noun = random.choice(nouns_m)
            
            data.append({
                "input": f"Ich {verb} der {noun}.",
                "output": f"❌ Incorrect.\n✅ Correct: Ich {verb} den {noun}.\n📝 Пояснення: Дієслово '{verb}' вимагає Akkusativ. Для чоловічого роду артикль 'der' змінюється на 'den'."
            })
        return data

    def generate_dativ(self, count=1000):
        """A2: Dativ case (der/das -> dem, die -> der)."""
        verbs_dat = [("helfe", "helfen"), ("antworte", "antworten"), ("danke", "danken")]
        nouns = [("Bruder", "m"), ("Kind", "n"), ("Mann", "m"), ("Frau", "f")]
        
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            verb, v_inf = random.choice(verbs_dat)
            v_form = self.get_verb_form(verb[:-1], sub_key) # Simple stem extraction
            noun, gender = random.choice(nouns)
            
            c_art = "dem" if gender in ["m", "n"] else "der"
            w_art = "den" if gender == "m" else ("die" if gender == "f" else "das")
            
            data.append({
                "input": f"{sub_key.capitalize()} {v_form} {w_art} {noun}.",
                "output": f"❌ Incorrect.\n✅ Correct: {sub_key.capitalize()} {v_form} {c_art} {noun}.\n📝 Пояснення: Дієслово '{v_inf}' завжди вимагає Dativ. Тому артикль для {gender}-роду має бути '{c_art}'."
            })
        return data
