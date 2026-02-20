import random
from .base import BaseGenerator

class CaseGenerator(BaseGenerator):
    """Generates examples for all four German cases: Nominativ, Genitiv, Dativ, Akkusativ."""

    def generate_nominativ(self, count=1000):
        """A1: Nominativ — article as subject (Der/Die/Das + Noun + verb). Wrong gender or other case."""
        # (noun, gender, nom_article, list of wrong articles), verb in 3rd sg
        nouns = [
            ("Mann", "m", "Der", ["Die", "Das", "Den", "Dem"]),
            ("Vater", "m", "Der", ["Die", "Das", "Den", "Dem"]),
            ("Hund", "m", "Der", ["Die", "Das", "Den", "Dem"]),
            ("Frau", "f", "Die", ["Der", "Das", "Den", "Dem"]),
            ("Mutter", "f", "Die", ["Der", "Das", "Den", "Dem"]),
            ("Katze", "f", "Die", ["Der", "Das", "Den", "Dem"]),
            ("Kind", "n", "Das", ["Der", "Die", "Den", "Dem"]),
            ("Auto", "n", "Das", ["Der", "Die", "Den", "Dem"]),
            ("Buch", "n", "Das", ["Der", "Die", "Den", "Dem"]),
        ]
        # 3rd person singular verb forms (er/sie/es)
        verb_phrases = [
            ("kommt", "kommen"),
            ("geht", "gehen"),
            ("spielt", "spielen"),
            ("schläft", "schlafen"),
            ("arbeitet", "arbeiten"),
            ("liest", "lesen"),
        ]
        gender_names = {"m": "чоловічого", "n": "середнього", "f": "жіночого"}
        data = []
        for _ in range(count):
            noun, gender, c_art, wrong_articles = random.choice(nouns)
            v_form, v_inf = random.choice(verb_phrases)
            w_art = random.choice(wrong_articles)
            if random.random() > 0.5:
                data.append({
                    "input": f"{w_art} {noun} {v_form}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {c_art} {noun} {v_form}.\n📝 Пояснення: У Nominativ (підмет) для {gender_names[gender]} роду артикль — '{c_art}', а не '{w_art}'."
                })
            else:
                data.append({"input": f"{c_art} {noun} {v_form}.", "output": "✅ Correct."})
        return data

    def generate_akkusativ_masculine(self, count=1000):
        """A1: Akkusativ for all genders (der->den, die->die, das->das). Wrong: Dativ or wrong gender."""
        verbs = ["suche", "sehe", "kaufe", "brauche", "habe"]
        # (noun, gender, akk_article, list of wrong articles: Dativ + wrong gender)
        nouns = [
            ("Apfel", "m", "den", ["dem", "die", "das"]),
            ("Schlüssel", "m", "den", ["dem", "die", "das"]),
            ("Computer", "m", "den", ["dem", "die", "das"]),
            ("Hund", "m", "den", ["dem", "die", "das"]),
            ("Auto", "n", "das", ["dem", "der", "die"]),
            ("Buch", "n", "das", ["dem", "der", "die"]),
            ("Handy", "n", "das", ["dem", "der", "die"]),
            ("Katze", "f", "die", ["der", "den", "das"]),
            ("Tasche", "f", "die", ["der", "den", "das"]),
            ("Schwester", "f", "die", ["der", "den", "das"]),
        ]
        gender_names = {"m": "чоловічого", "n": "середнього", "f": "жіночого"}
        data = []
        for _ in range(count):
            verb = random.choice(verbs)
            noun, gender, c_art, wrong_articles = random.choice(nouns)
            w_art = random.choice(wrong_articles)
            if random.random() > 0.5:
                data.append({
                    "input": f"Ich {verb} {w_art} {noun}.",
                    "output": f"❌ Incorrect.\n✅ Correct: Ich {verb} {c_art} {noun}.\n📝 Пояснення: Дієслово '{verb}' вимагає Akkusativ. Для {gender_names[gender]} роду артикль у Akkusativ — '{c_art}', а не '{w_art}'."
                })
            else:
                data.append({"input": f"Ich {verb} {c_art} {noun}.", "output": "✅ Correct."})
        return data

    def generate_dativ(self, count=1000):
        """A2: Dativ for all genders. Correct: dem (m/n), der (f). Wrong: Akkusativ or wrong gender."""
        verbs_dat = [("helfe", "helfen"), ("antworte", "antworten"), ("danke", "danken")]
        # (noun, gender, dativ_article, list of wrong articles)
        nouns = [
            ("Bruder", "m", "dem", ["den", "die", "das", "der"]),
            ("Mann", "m", "dem", ["den", "die", "das", "der"]),
            ("Vater", "m", "dem", ["den", "die", "das", "der"]),
            ("Freund", "m", "dem", ["den", "die", "das", "der"]),
            ("Kind", "n", "dem", ["das", "den", "die", "der"]),
            ("Auto", "n", "dem", ["das", "den", "die", "der"]),
            ("Frau", "f", "der", ["die", "den", "das", "dem"]),
            ("Mutter", "f", "der", ["die", "den", "das", "dem"]),
            ("Schwester", "f", "der", ["die", "den", "das", "dem"]),
        ]
        gender_names = {"m": "чоловічого", "n": "середнього", "f": "жіночого"}
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            dn = self.get_display_name(sub_key)
            verb, v_inf = random.choice(verbs_dat)
            v_form = self.get_verb_form(verb[:-1], sub_key)
            noun, gender, c_art, wrong_articles = random.choice(nouns)
            w_art = random.choice(wrong_articles)
            if random.random() > 0.5:
                data.append({
                    "input": f"{dn} {v_form} {w_art} {noun}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {dn} {v_form} {c_art} {noun}.\n📝 Пояснення: Дієслово '{v_inf}' завжди вимагає Dativ. Для {gender_names[gender]} роду артикль у Dativ — '{c_art}', а не '{w_art}'."
                })
            else:
                data.append({"input": f"{dn} {v_form} {c_art} {noun}.", "output": "✅ Correct."})
        return data

    def generate_genitiv(self, count=500):
        """A2: Genitiv — limited set with fixed prepositions (während, wegen, trotz). Correct: des (m/n), der (f)."""
        # (prep, noun_in_genitiv, correct_article, wrong_articles, short_explanation)
        scenarios = [
            ("während", "Tages", "des", ["dem", "den", "die", "das"], "прийменник 'während' вимагає Genitiv"),
            ("während", "Abends", "des", ["dem", "den", "die", "das"], "прийменник 'während' вимагає Genitiv"),
            ("wegen", "Arbeit", "der", ["die", "den", "dem", "das"], "прийменник 'wegen' вимагає Genitiv"),
            ("wegen", "Zeit", "der", ["die", "den", "dem", "das"], "прийменник 'wegen' вимагає Genitiv"),
            ("trotz", "Regens", "des", ["dem", "den", "die", "das"], "прийменник 'trotz' вимагає Genitiv"),
            ("trotz", "Problems", "des", ["dem", "den", "die", "das"], "прийменник 'trotz' вимагає Genitiv"),
        ]
        data = []
        for _ in range(count):
            prep, noun_gen, c_art, wrong_articles, expl = random.choice(scenarios)
            w_art = random.choice(wrong_articles)
            if random.random() > 0.5:
                data.append({
                    "input": f"Ich bin {prep} {w_art} {noun_gen} müde.",
                    "output": f"❌ Incorrect.\n✅ Correct: Ich bin {prep} {c_art} {noun_gen} müde.\n📝 Пояснення: {expl}. У Genitiv для цього іменника потрібен артикль '{c_art}', а не '{w_art}'."
                })
            else:
                data.append({"input": f"Ich bin {prep} {c_art} {noun_gen} müde.", "output": "✅ Correct."})
        return data

    def generate_prepositions_akk_dat(self, count=1000):
        """A2: Wechselpräpositionen — in/auf with Akkusativ (direction) vs Dativ (location). All genders."""
        # (verb, prep, noun, gender, case, c_art, list of wrong articles, logic)
        scenarios = [
            ("gehe", "in", "Kino", "n", "Akkusativ", "das", ["dem", "der", "die"], "Куди? (двигун)"),
            ("bin", "in", "Kino", "n", "Dativ", "dem", ["das", "den", "der", "die"], "Де? (статика)"),
            ("lege", "auf", "Tisch", "m", "Akkusativ", "den", ["dem", "die", "das"], "Куди?"),
            ("liegt", "auf", "Tisch", "m", "Dativ", "dem", ["den", "die", "das", "der"], "Де?"),
            ("gehe", "in", "Küche", "f", "Akkusativ", "die", ["der", "dem", "den", "das"], "Куди?"),
            ("bin", "in", "Küche", "f", "Dativ", "der", ["die", "dem", "den", "das"], "Де?"),
            ("gehe", "in", "Park", "m", "Akkusativ", "den", ["dem", "die", "das"], "Куди?"),
            ("bin", "in", "Park", "m", "Dativ", "dem", ["den", "die", "das"], "Де?"),
            ("stelle", "auf", "Bank", "f", "Akkusativ", "die", ["der", "dem", "den"], "Куди?"),
            ("liegt", "auf", "Bank", "f", "Dativ", "der", ["die", "dem", "den"], "Де?"),
        ]
        gender_names = {"m": "чоловічого", "n": "середнього", "f": "жіночого"}
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            dn = self.get_display_name(sub_key)
            v, prep, noun, gender, case, c_art, wrong_list, logic = random.choice(scenarios)
            w_art = random.choice(wrong_list)
            if random.random() > 0.5:
                data.append({
                    "input": f"{dn} {v} {prep} {w_art} {noun}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {dn} {v} {prep} {c_art} {noun}.\n📝 Пояснення: Прийменник '{prep}' у значенні '{logic}' вимагає {case}. Для {gender_names[gender]} роду це '{c_art}'."
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
        """A1/A2: Fixed prepositions — Dativ (mit, nach, von, bei) and Akkusativ (für, ohne, gegen). All genders."""
        # Dativ: (prep, (noun, gender, c_art, wrong_articles))
        preps_dat = [
            ("mit", [("Freund", "m", "dem", ["den", "die", "das", "der"]), ("Frau", "f", "der", ["die", "den", "das", "dem"]), ("Kind", "n", "dem", ["das", "den", "die", "der"]), ("Bus", "m", "dem", ["den", "die", "das"])]),
            ("nach", [("Arzt", "m", "dem", ["den", "die", "das"]), ("Arbeit", "f", "der", ["die", "den", "dem"]), ("Konzert", "n", "dem", ["das", "den", "die"])]),
            ("von", [("Vater", "m", "dem", ["den", "die", "das"]), ("Mutter", "f", "der", ["die", "den", "dem"]), ("Bahnhof", "m", "dem", ["den", "die", "das"])]),
            ("bei", [("Freund", "m", "dem", ["den", "die", "das"]), ("Tante", "f", "der", ["die", "den", "dem"]), ("Onkel", "m", "dem", ["den", "die", "das"])]),
        ]
        # Akkusativ: (prep, (noun, gender, c_art, wrong_articles))
        preps_akk = [
            ("für", [("Mann", "m", "den", ["dem", "der", "die", "das"]), ("Frau", "f", "die", ["der", "dem", "den", "das"]), ("Kind", "n", "das", ["dem", "der", "die"])]),
            ("ohne", [("Hund", "m", "den", ["dem", "der", "die", "das"]), ("Tasche", "f", "die", ["der", "dem", "den", "das"]), ("Auto", "n", "das", ["dem", "der", "die"])]),
            ("gegen", [("Tisch", "m", "den", ["dem", "der", "die"]), ("Wand", "f", "die", ["der", "dem", "den"]), ("Fenster", "n", "das", ["dem", "der", "die"])]),
        ]
        data = []
        for _ in range(count):
            is_dat = random.random() > 0.5
            prep, noun_list = random.choice(preps_dat if is_dat else preps_akk)
            noun, gender, c_art, wrong_articles = random.choice(noun_list)
            w_art = random.choice(wrong_articles)
            case = "Dativ" if is_dat else "Akkusativ"
            if random.random() > 0.5:
                data.append({
                    "input": f"Ich gehe {prep} {w_art} {noun}.",
                    "output": f"❌ Incorrect.\n✅ Correct: Ich gehe {prep} {c_art} {noun}.\n📝 Пояснення: Прийменник '{prep}' завжди вимагає {case}. Тому артикль має бути '{c_art}'."
                })
            else:
                data.append({"input": f"Ich gehe {prep} {c_art} {noun}.", "output": "✅ Correct."})
        return data
