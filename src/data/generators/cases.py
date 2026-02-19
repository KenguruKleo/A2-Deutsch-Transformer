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

    def generate_prepositions_akk_dat(self, count=1000):
        """A2: Wechselpräpositionen (Wohin? + Akk / Wo? + Dat)."""
        scenarios = [
            ("gehe", "in", "Kino", "n", "Akkusativ", "das", "dem", "Куди? (двигун)"),
            ("bin", "in", "Kino", "n", "Dativ", "dem", "das", "Де? (статика)"),
            ("lege", "на", "Tisch", "m", "Akkusativ", "den", "dem", "Куди?"),
            ("liegt", "на", "Tisch", "m", "Dativ", "dem", "den", "Де?")
        ]
        # Adjust 'на' to 'auf' for German output
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            v, prep_name, noun, gender, case, c_art, w_art, logic = random.choice(scenarios)
            prep = "auf" if prep_name == "на" else "in"
            
            data.append({
                "input": f"{sub_key.capitalize()} {v} {prep} {w_art} {noun}.",
                "output": f"❌ Incorrect.\n✅ Correct: {sub_key.capitalize()} {v} {prep} {c_art} {noun}.\n📝 Пояснення: Прийменник '{prep}' у значенні '{logic}' вимагає {case}. Для {gender}-роду це '{c_art}'."
            })
        return data

    def generate_adjective_endings(self, count=1000):
        """A2: Basic adjective endings after 'ein' (mixed declension)."""
        adjectives = [("gut", "er", "m"), ("neu", "es", "n"), ("schön", "e", "f")]
        nouns = {"m": "Mann", "n": "Auto", "f": "Frau"}
        
        data = []
        for _ in range(count):
            adj, ending, gender = random.choice(adjectives)
            noun = nouns[gender]
            
            data.append({
                "input": f"Das ist ein {adj} {noun}.",
                "output": f"❌ Incorrect.\n✅ Correct: Das ist ein {adj}{ending} {noun}.\n📝 Пояснення: Після неозначеного артикля 'ein' у Nominativ прикметник '{adj}' для {gender}-роду отримує закінчення '-{ending}'."
            })
        return data

    def generate_possessive_pronouns(self, count=1000):
        """A2: Possessive pronouns (mein, dein, sein, ihr) - correct agreement."""
        # Focus on "This is my/your/his X" (Nominative)
        nouns = [("Bruder", "m"), ("Kind", "n"), ("Schwester", "f")]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            pos_base = self.possessives.get(sub_key, "mein")
            noun, gender = random.choice(nouns)
            
            c_pos = pos_base if gender in ["m", "n"] else pos_base + "e"
            
            # Error: forgetting the 'e' for feminine nouns
            if gender == "f":
                wrong = f"Das ist {pos_base} {noun}."
                data.append({
                    "input": wrong,
                    "output": f"❌ Incorrect.\n✅ Correct: Das ist {c_pos} {noun}.\n📝 Пояснення: Присвійний займенник '{pos_base}' для жіночого роду '{noun}' повинен мати закінчення '-e'."
                })
            else:
                # Error: adding an unnecessary 'e' for masculine/neuter
                wrong = f"Das ist {pos_base}e {noun}."
                data.append({
                    "input": wrong,
                    "output": f"❌ Incorrect.\n✅ Correct: Das ist {c_pos} {noun}.\n📝 Пояснення: Для {gender}-роду ('{noun}') присвійний займенник '{pos_base}' не повинен мати закінчення '-e' у початковій формі (Nominativ)."
                })
        return data
