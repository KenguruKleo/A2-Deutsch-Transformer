import random
from .base import BaseGenerator

class CaseGenerator(BaseGenerator):
    """Generates examples for Cases (Akkusativ, Dativ) and Articles."""

    def generate_akkusativ_masculine(self, count=1000):
        """A1: Akkusativ for all genders (der->den, die->die, das->das) vs Dativ confusion."""
        verbs = ["suche", "sehe", "kaufe", "brauche", "habe"]
        nouns = [
            # (noun, gender, akk_article, wrong_dativ_article)
            ("Apfel", "m", "den", "dem"),
            ("Schlüssel", "m", "den", "dem"),
            ("Computer", "m", "den", "dem"),
            ("Hund", "m", "den", "dem"),
            ("Auto", "n", "das", "dem"),
            ("Buch", "n", "das", "dem"),
            ("Handy", "n", "das", "dem"),
            ("Katze", "f", "die", "der"),
            ("Tasche", "f", "die", "der"),
            ("Schwester", "f", "die", "der"),
        ]
        gender_names = {"m": "чоловічого", "n": "середнього", "f": "жіночого"}
        data = []
        for _ in range(count):
            verb = random.choice(verbs)
            noun, gender, c_art, w_art = random.choice(nouns)
            if random.random() > 0.5:
                data.append({
                    "input": f"Ich {verb} {w_art} {noun}.",
                    "output": f"❌ Incorrect.\n✅ Correct: Ich {verb} {c_art} {noun}.\n📝 Пояснення: Дієслово '{verb}' вимагає Akkusativ. Для {gender_names[gender]} роду артикль у Akkusativ — '{c_art}', а не '{w_art}'."
                })
            else:
                data.append({"input": f"Ich {verb} {c_art} {noun}.", "output": "✅ Correct."})
        return data

    def generate_dativ(self, count=1000):
        """A2: Dativ case."""
        verbs_dat = [("helfe", "helfen"), ("antworte", "antworten"), ("danke", "danken")]
        nouns = [("Bruder", "m"), ("Kind", "n"), ("Mann", "m"), ("Frau", "f")]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            dn = self.get_display_name(sub_key)
            verb, v_inf = random.choice(verbs_dat)
            v_form = self.get_verb_form(verb[:-1], sub_key)
            noun, gender = random.choice(nouns)
            c_art = "dem" if gender in ["m", "n"] else "der"
            
            if random.random() > 0.5:
                w_art = "den" if gender == "m" else ("die" if gender == "f" else "das")
                data.append({
                    "input": f"{dn} {v_form} {w_art} {noun}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {dn} {v_form} {c_art} {noun}.\n📝 Пояснення: Дієслово '{v_inf}' завжди вимагає Dativ. Тому артикль для {gender}-роду має бути '{c_art}'."
                })
            else:
                data.append({"input": f"{dn} {v_form} {c_art} {noun}.", "output": "✅ Correct."})
        return data

    def generate_prepositions_akk_dat(self, count=1000):
        """A2: Wechselpräpositionen."""
        scenarios = [
            ("gehe", "in", "Kino", "n", "Akkusativ", "das", "dem", "Куди? (двигун)"),
            ("bin", "in", "Kino", "n", "Dativ", "dem", "das", "Де? (статика)"),
            ("lege", "auf", "Tisch", "m", "Akkusativ", "den", "dem", "Куди?"),
            ("liegt", "auf", "Tisch", "m", "Dativ", "dem", "den", "Де?")
        ]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            dn = self.get_display_name(sub_key)
            v, prep, noun, gender, case, c_art, w_art, logic = random.choice(scenarios)
            if random.random() > 0.5:
                data.append({
                    "input": f"{dn} {v} {prep} {w_art} {noun}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {dn} {v} {prep} {c_art} {noun}.\n📝 Пояснення: Прийменник '{prep}' у значенні '{logic}' вимагає {case}. Для {gender}-роду це '{c_art}'."
                })
            else:
                data.append({"input": f"{dn} {v} {prep} {c_art} {noun}.", "output": "✅ Correct."})
        return data

    def generate_adjective_endings(self, count=1000):
        """A2: Adjective endings."""
        adjectives = [("gut", "er", "m"), ("neu", "es", "n"), ("schön", "e", "f")]
        nouns = {"m": "Mann", "n": "Auto", "f": "Frau"}
        data = []
        for _ in range(count):
            adj, ending, gender = random.choice(adjectives)
            noun = nouns[gender]
            correct = f"Das ist ein {adj}{ending} {noun}."
            if random.random() > 0.5:
                data.append({
                    "input": f"Das ist ein {adj} {noun}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: Після неозначеного артикля 'ein' у Nominativ прикметник '{adj}' для {gender}-роду отримує закінчення '-{ending}'."
                })
            else:
                data.append({"input": correct, "output": "✅ Correct."})
        return data

    def generate_possessive_pronouns(self, count=1000):
        """A2: Possessive pronouns."""
        nouns = [("Bruder", "m"), ("Kind", "n"), ("Schwester", "f")]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            pos_base = self.possessives.get(sub_key, "mein")
            noun, gender = random.choice(nouns)
            c_pos = pos_base if gender in ["m", "n"] else pos_base + "e"
            
            if random.random() > 0.5:
                if gender == "f":
                    wrong = f"Das ist {pos_base} {noun}."
                    msg = f"Присвійний займенник '{pos_base}' для жіночого роду '{noun}' повинен мати закінчення '-e'."
                else:
                    wrong = f"Das ist {pos_base}e {noun}."
                    msg = f"Для {gender}-роду ('{noun}') присвійний займенник '{pos_base}' не повинен мати закінчення '-e' у початковій формі (Nominativ)."
                data.append({
                    "input": wrong,
                    "output": f"❌ Incorrect.\n✅ Correct: Das ist {c_pos} {noun}.\n📝 Пояснення: {msg}"
                })
            else:
                data.append({"input": f"Das ist {c_pos} {noun}.", "output": "✅ Correct."})
        return data

    def generate_komparation(self, count=1000):
        """A2: Comparison."""
        adjectives = [("gut", "besser"), ("viel", "mehr"), ("schnell", "schneller")]
        data = []
        for _ in range(count):
            adj, comp = random.choice(adjectives)
            if random.random() > 0.5:
                data.append({
                    "input": f"Das ist mehr {adj}.",
                    "output": f"❌ Incorrect.\n✅ Correct: Das ist {comp}.\n📝 Пояснення: У німецькій мові ступені порівняння утворюються за допомогою суфіксів (або зміни кореня), а не словом 'mehr'."
                })
            else:
                data.append({"input": f"Das ist {comp}.", "output": "✅ Correct."})
        return data

    def generate_fixed_prepositions(self, count=1000):
        """A1/A2: Fixed prepositions."""
        preps_dat = [("mit", "dem", "den"), ("nach", "dem", "das")]
        preps_akk = [("für", "den", "dem"), ("ohne", "den", "der")]
        nouns = [("Freund", "m"), ("Auto", "n")]
        data = []
        for _ in range(count):
            is_dat = random.random() > 0.5
            prep, c_art, w_art = random.choice(preps_dat if is_dat else preps_akk)
            noun = random.choice(nouns)[0]
            case = "Dativ" if is_dat else "Akkusativ"
            if random.random() > 0.5:
                data.append({
                    "input": f"Ich gehe {prep} {w_art} {noun}.",
                    "output": f"❌ Incorrect.\n✅ Correct: Ich gehe {prep} {c_art} {noun}.\n📝 Пояснення: Прийменник '{prep}' завжди вимагає {case}. Тому артикль має бути '{c_art}'."
                })
            else:
                data.append({"input": f"Ich gehe {prep} {c_art} {noun}.", "output": "✅ Correct."})
        return data
