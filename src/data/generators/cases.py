import random
from .base import BaseGenerator

class CaseGenerator(BaseGenerator):
    """Generates examples for all four German cases: Nominativ, Genitiv, Dativ, Akkusativ."""

    def generate_nominativ(self, count=1000):
        """A1: Nominativ — article as subject — all nouns from shared pool."""
        verb_phrases = [
            ("kommt", "kommen"), ("geht", "gehen"), ("spielt", "spielen"),
            ("schläft", "schlafen"), ("arbeitet", "arbeiten"), ("liest", "lesen"),
        ]
        data = []
        for _ in range(count):
            noun, gender = random.choice(self.nouns_with_gender)
            c_art = self.articles["nom"][gender].capitalize()
            wrong_articles = [a.capitalize() for a in self.all_def_articles if a.capitalize() != c_art]
            w_art = random.choice(wrong_articles)
            v_form, v_inf = random.choice(verb_phrases)
            if random.random() > 0.5:
                data.append({
                    "input": f"{w_art} {noun} {v_form}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {c_art} {noun} {v_form}.\n📝 Пояснення: У Nominativ (підмет) для {self.gender_names[gender]} роду артикль — '{c_art}', а не '{w_art}'."
                })
            else:
                data.append({"input": f"{c_art} {noun} {v_form}.", "output": "✅ Correct."})
        return data

    def generate_akkusativ_masculine(self, count=1000):
        """A1: Akkusativ for all genders — all subjects × verbs × nouns. Gender-contrastive: den Hund ✅ vs den Auto ❌."""
        verb_stems = [("such", "suchen"), ("seh", "sehen"), ("kauf", "kaufen"), ("brauch", "brauchen"), ("hab", "haben")]
        # Split nouns by gender for contrastive pairs
        masc_nouns = [(n, g) for n, g in self.nouns_with_gender if g == "m"]
        neut_nouns = [(n, g) for n, g in self.nouns_with_gender if g == "n"]
        fem_nouns  = [(n, g) for n, g in self.nouns_with_gender if g == "f"]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            dn = self.get_display_name(sub_key)
            stem, v_inf = random.choice(verb_stems)
            v_form = self.get_verb_form(stem, sub_key)
            r = random.random()
            if r < 0.35:
                # Correct masc: "den Hund" ✅
                noun, gender = random.choice(masc_nouns)
                data.append({"input": f"{dn} {v_form} den {noun}.", "output": "✅ Correct."})
            elif r < 0.55:
                # Wrong: "den" + neuter/feminine noun ❌ (most frequent learner error)
                noun, gender = random.choice(neut_nouns + fem_nouns)
                c_art = self.articles["akk"][gender]
                data.append({
                    "input": f"{dn} {v_form} den {noun}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {dn} {v_form} {c_art} {noun}.\n📝 Пояснення: Іменник '{noun}' — {self.gender_names[gender]} роду. У Akkusativ артикль — '{c_art}', а не 'den'."
                })
            elif r < 0.75:
                # Correct neut/fem: "das Auto" ✅ / "die Katze" ✅
                noun, gender = random.choice(neut_nouns + fem_nouns)
                c_art = self.articles["akk"][gender]
                data.append({"input": f"{dn} {v_form} {c_art} {noun}.", "output": "✅ Correct."})
            else:
                # Other wrong articles (dem, der for masc, etc.)
                noun, gender = random.choice(self.nouns_with_gender)
                c_art = self.articles["akk"][gender]
                wrong_articles = [a for a in self.all_def_articles if a != c_art]
                w_art = random.choice(wrong_articles)
                data.append({
                    "input": f"{dn} {v_form} {w_art} {noun}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {dn} {v_form} {c_art} {noun}.\n📝 Пояснення: Дієслово '{v_inf}' вимагає Akkusativ. Для {self.gender_names[gender]} роду артикль у Akkusativ — '{c_art}', а не '{w_art}'."
                })
        return data

    def generate_article_required_akkusativ(self, count=500):
        """A1: Countable noun needs article after haben/brauchen — all subjects × nouns."""
        verb_stems = [("hab", "haben"), ("brauch", "brauchen"), ("kauf", "kaufen"), ("seh", "sehen")]
        indef_art = {"m": "einen", "n": "ein", "f": "eine"}
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            dn = self.get_display_name(sub_key)
            stem, v_inf = random.choice(verb_stems)
            v_form = self.get_verb_form(stem, sub_key)
            noun, gender = random.choice(self.nouns_with_gender)
            use_definite = random.random() > 0.5
            art = self.articles["akk"][gender] if use_definite else indef_art[gender]
            if random.random() > 0.5:
                data.append({
                    "input": f"{dn} {v_form} {noun}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {dn} {v_form} {art} {noun}.\n📝 Пояснення: Злічний іменник '{noun}' потребує артикля (наприклад '{art}')."
                })
            else:
                data.append({"input": f"{dn} {v_form} {art} {noun}.", "output": "✅ Correct."})
        return data

    def generate_dativ(self, count=1000):
        """A2: Dativ for all genders — all subjects × verbs × nouns from shared pool."""
        verb_stems = [("helf", "helfen"), ("antwort", "antworten"), ("dank", "danken")]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            dn = self.get_display_name(sub_key)
            stem, v_inf = random.choice(verb_stems)
            v_form = self.get_verb_form(stem, sub_key)
            noun, gender = random.choice(self.nouns_with_gender)
            c_art = self.articles["dat"][gender]
            wrong_articles = [a for a in self.all_def_articles if a != c_art]
            w_art = random.choice(wrong_articles)
            if random.random() > 0.5:
                data.append({
                    "input": f"{dn} {v_form} {w_art} {noun}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {dn} {v_form} {c_art} {noun}.\n📝 Пояснення: Дієслово '{v_inf}' завжди вимагає Dativ. Для {self.gender_names[gender]} роду артикль у Dativ — '{c_art}', а не '{w_art}'."
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
        """A2: Wechselpräpositionen — in/auf with Akkusativ (direction) vs Dativ (location). All genders. Subject+verb match."""
        # (subject_display, verb, prep, noun, gender, case, c_art, wrong_list, logic)
        scenarios = [
            ("Ich", "gehe", "in", "Kino", "n", "Akkusativ", "das", ["dem", "der", "die"], "Куди? (двигун)"),
            ("Ich", "bin", "in", "Kino", "n", "Dativ", "dem", ["das", "den", "der", "die"], "Де? (статика)"),
            ("Ich", "lege", "auf", "Tisch", "m", "Akkusativ", "den", ["dem", "die", "das"], "Куди?"),
            ("Er", "legt", "auf", "Tisch", "m", "Akkusativ", "den", ["dem", "die", "das"], "Куди?"),
            ("Er", "liegt", "auf", "Tisch", "m", "Dativ", "dem", ["den", "die", "das", "der"], "Де?"),
            ("Ich", "gehe", "in", "Küche", "f", "Akkusativ", "die", ["der", "dem", "den", "das"], "Куди?"),
            ("Ich", "bin", "in", "Küche", "f", "Dativ", "der", ["die", "dem", "den", "das"], "Де?"),
            ("Ich", "gehe", "in", "Park", "m", "Akkusativ", "den", ["dem", "die", "das"], "Куди?"),
            ("Ich", "bin", "in", "Park", "m", "Dativ", "dem", ["den", "die", "das"], "Де?"),
            ("Ich", "stelle", "auf", "Bank", "f", "Akkusativ", "die", ["der", "dem", "den"], "Куди?"),
            ("Sie", "stellt", "auf", "Bank", "f", "Akkusativ", "die", ["der", "dem", "den"], "Куди?"),
            ("Sie", "liegt", "auf", "Bank", "f", "Dativ", "der", ["die", "dem", "den"], "Де?"),
        ]
        gender_names = {"m": "чоловічого", "n": "середнього", "f": "жіночого"}
        data = []
        for _ in range(count):
            dn, v, prep, noun, gender, case, c_art, wrong_list, logic = random.choice(scenarios)
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
        """A2: Adjective endings after 'ein'/'eine' in Nominativ. Includes 'eine schöne Frau' (not 'eine schön Frau')."""
        # (adj, ending_m, ending_n, ending_f, noun_m, noun_n, noun_f)
        variants = [
            ("gut", "er", "es", "e", "Mann", "Buch", "Frau"),
            ("gut", "er", "es", "e", "Tisch", "Auto", "Tasche"),
            ("neu", "er", "es", "e", "Mann", "Buch", "Frau"),
            ("neu", "er", "es", "e", "Tisch", "Auto", "Tasche"),
            ("schön", "er", "es", "e", "Mann", "Buch", "Frau"),
        ]
        # feminine with "eine": (adj, ending_f, noun_f)
        eine_variants = [
            ("schön", "e", "Frau"),
            ("gut", "e", "Frau"),
            ("neu", "e", "Tasche"),
        ]
        data = []
        for _ in range(count):
            if random.random() < 0.2:
                adj, end_f, noun = random.choice(eine_variants)
                correct = f"Das ist eine {adj}{end_f} {noun}."
                if random.random() > 0.5:
                    data.append({
                        "input": f"Das ist eine {adj} {noun}.",
                        "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: Після артикля 'eine' у Nominativ прикметник '{adj}' отримує закінчення '-{end_f}' (eine schöne Frau)."
                    })
                else:
                    data.append({"input": correct, "output": "✅ Correct."})
                continue
            adj, end_m, end_n, end_f, noun_m, noun_n, noun_f = random.choice(variants)
            gender = random.choice(["m", "n", "f"])
            if gender == "m":
                ending, noun = end_m, noun_m
            elif gender == "n":
                ending, noun = end_n, noun_n
            else:
                ending, noun = end_f, noun_f
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
        """A2: Comparison — correct comparatives (größer, kleiner, besser, schneller, älter) vs wrong 'mehr + adj'."""
        # (positive, comparative) — irregular and regular
        adjectives = [
            ("gut", "besser"), ("viel", "mehr"), ("schnell", "schneller"),
            ("groß", "größer"), ("klein", "kleiner"), ("alt", "älter"),
            ("warm", "wärmer"), ("kalt", "kälter"), ("jung", "jünger"),
        ]
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
            ("für", [("Mann", "m", "den", ["dem", "der", "die", "das"]), ("Freund", "m", "den", ["dem", "der", "die", "das"]), ("Frau", "f", "die", ["der", "dem", "den", "das"]), ("Kind", "n", "das", ["dem", "der", "die"])]),
            ("ohne", [("Hund", "m", "den", ["dem", "der", "die", "das"]), ("Tasche", "f", "die", ["der", "dem", "den", "das"]), ("Auto", "n", "das", ["dem", "der", "die"])]),
            ("gegen", [("Tisch", "m", "den", ["dem", "der", "die"]), ("Wand", "f", "die", ["der", "dem", "den"]), ("Fenster", "n", "das", ["dem", "der", "die"])]),
        ]
        # Fixed phrases: (correct, wrong, explanation) — e.g. "at work" = bei der Arbeit, not in der Arbeit
        fixed_phrases = [
            ("Ich bin bei der Arbeit.", "Ich bin in der Arbeit.", "Для значення «на роботі» (at work) використовується прийменник 'bei', а не 'in'. Правильно: bei der Arbeit."),
            ("Er ist auf der Arbeit.", "Er ist in der Arbeit.", "Для «на роботі» можна сказати 'auf der Arbeit' або 'bei der Arbeit'; 'in der Arbeit' тут не вживається."),
        ]
        # Correct-only to reduce false positives (model marking "Ich gehe mit dem Freund" as wrong)
        correct_only = ["Ich gehe mit dem Freund.", "Ich gehe mit der Frau.", "Er geht mit dem Freund."]
        data = []
        for _ in range(count):
            if random.random() < 0.08:
                data.append({"input": random.choice(correct_only), "output": "✅ Correct."})
                continue
            if random.random() < 0.12:
                correct, wrong, expl = random.choice(fixed_phrases)
                if random.random() > 0.5:
                    data.append({"input": wrong, "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: {expl}"})
                else:
                    data.append({"input": correct, "output": "✅ Correct."})
                continue
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
