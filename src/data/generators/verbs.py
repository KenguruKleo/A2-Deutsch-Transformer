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
            dn = self.get_display_name(sub_key)
            verb_stem, obj = random.choice(verbs)
            correct_v = self.get_verb_form(verb_stem, sub_key)
            
            if random.random() > 0.5:
                wrong_sub = random.choice([k for k in self.subjects.keys() if k != sub_key])
                wrong_v = self.get_verb_form(verb_stem, wrong_sub)
                data.append({
                    "input": f"{dn} {wrong_v} {obj}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {dn} {correct_v} {obj}.\n📝 Пояснення: У теперішньому часі (Präsens) для підмета '{dn}' дієслово має закінчення '-{self.subjects[sub_key]['ending']}', тому правильно '{correct_v}', а не '{wrong_v}'."
                })
            else:
                data.append({"input": f"{dn} {correct_v} {obj}.", "output": "✅ Correct."})
        return data

    def generate_haben_sein_praesens(self, count=1000):
        """A1: Irregular verbs haben/sein as main verbs in present tense."""
        haben_forms = {"ich": "habe", "du": "hast", "er": "hat", "sie": "hat", "wir": "haben", "ihr": "habt", "sie_plural": "haben"}
        sein_forms = {"ich": "bin", "du": "bist", "er": "ist", "sie": "ist", "wir": "sind", "ihr": "seid", "sie_plural": "sind"}
        display_names = {k: self.get_display_name(k) for k in haben_forms}
        
        haben_objects = [
            # With indefinite article
            ("ein Auto", "n"), ("einen Hund", "m"), ("eine Katze", "f"),
            ("ein Buch", "n"), ("einen Bruder", "m"), ("eine Schwester", "f"),
            # With definite article (Akkusativ)
            ("das Auto", "n"), ("den Hund", "m"), ("die Katze", "f"),
            ("das Buch", "n"), ("den Schlüssel", "m"), ("die Tasche", "f"),
            # Without article
            ("Hunger", None), ("Zeit", None), ("Durst", None), ("Geld", None)
        ]
        sein_complements = [
            "müde", "krank", "zu Hause", "in Berlin", "glücklich", "traurig",
            "Lehrer", "Student", "Arzt", "hier", "dort", "fertig", "groß"
        ]
        
        data = []
        for _ in range(count):
            sub_key = random.choice(list(haben_forms.keys()))
            dn = display_names[sub_key]
            
            if random.random() > 0.5:
                # haben as main verb
                correct_v = haben_forms[sub_key]
                obj, _ = random.choice(haben_objects)
                
                if random.random() > 0.5:
                    wrong_sub = random.choice([k for k in haben_forms.keys() if k != sub_key])
                    wrong_v = haben_forms[wrong_sub]
                    data.append({
                        "input": f"{dn} {wrong_v} {obj}.",
                        "output": f"❌ Incorrect.\n✅ Correct: {dn} {correct_v} {obj}.\n📝 Пояснення: Дієслово 'haben' для підмета '{dn}' має форму '{correct_v}', а не '{wrong_v}'."
                    })
                else:
                    data.append({"input": f"{dn} {correct_v} {obj}.", "output": "✅ Correct."})
            else:
                # sein as main verb
                correct_v = sein_forms[sub_key]
                complement = random.choice(sein_complements)
                
                if random.random() > 0.5:
                    wrong_sub = random.choice([k for k in sein_forms.keys() if k != sub_key])
                    wrong_v = sein_forms[wrong_sub]
                    data.append({
                        "input": f"{dn} {wrong_v} {complement}.",
                        "output": f"❌ Incorrect.\n✅ Correct: {dn} {correct_v} {complement}.\n📝 Пояснення: Дієслово 'sein' для підмета '{dn}' має форму '{correct_v}', а не '{wrong_v}'."
                    })
                else:
                    data.append({"input": f"{dn} {correct_v} {complement}.", "output": "✅ Correct."})

        return data

    def generate_perfekt_aux(self, count=1000):
        """A2: Haben vs Sein errors in Perfekt."""
        verbs_sein = [("gehen", "gegangen"), ("fahren", "gefahren"), ("kommen", "gekommen")]
        verbs_haben = [("essen", "gegessen"), ("machen", "gemacht"), ("kaufen", "gekauft")]
        
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            dn = self.get_display_name(sub_key)
            is_movement = random.random() > 0.5
            verb_inf, verb_p2 = random.choice(verbs_sein if is_movement else verbs_haben)
            c_aux = self.subjects[sub_key]["bin" if is_movement else "habe"]
            item = random.choice(self.nouns["place" if is_movement else "food"])[0]
            
            if random.random() > 0.5:
                w_aux = self.subjects[sub_key]["habe" if is_movement else "bin"]
                expl = f"Дієслово '{verb_inf}' {'означає рух' if is_movement else 'потребує допоміжного haben'}, тому використовуємо '{c_aux}', а не '{w_aux}'."
                data.append({
                    "input": f"{dn} {w_aux} {item} {verb_p2}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {dn} {c_aux} {item} {verb_p2}.\n📝 Пояснення: {expl}"
                })
            else:
                data.append({"input": f"{dn} {c_aux} {item} {verb_p2}.", "output": "✅ Correct."})
        return data

    def generate_partizip_forms(self, count=1000):
        """A2: Wrong Partizip II form."""
        verbs = [("essen", "gegessen", "habe"), ("gehen", "gegangen", "bin"), ("sehen", "gesehen", "habe")]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            dn = self.get_display_name(sub_key)
            inf, p2, aux_type = random.choice(verbs)
            aux = self.subjects[sub_key][aux_type]
            obj = random.choice(self.nouns["food" if aux_type == "habe" else "place"])[0]
            
            if random.random() > 0.5:
                data.append({
                    "input": f"{dn} {aux} {obj} {inf}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {dn} {aux} {obj} {p2}.\n📝 Пояснення: У минулому часі (Perfekt) основне дієслово має бути у формі Partizip II ('{p2}'), а не в інфінітиві ('{inf}')."
                })
            else:
                data.append({"input": f"{dn} {aux} {obj} {p2}.", "output": "✅ Correct."})
        return data

    def generate_modal_verbs(self, count=1000):
        """A1/A2: Modal verbs."""
        modals = {
            "können": {"ich": "kann", "du": "kannst", "er": "kann", "sie": "kann", "wir": "können", "ihr": "könnt", "sie_plural": "können"},
            "müssen": {"ich": "muss", "du": "musst", "er": "muss", "sie": "muss", "wir": "müssen", "ihr": "müsst", "sie_plural": "müssen"},
            "wollen": {"ich": "will", "du": "willst", "er": "will", "sie": "will", "wir": "wollen", "ihr": "wollt", "sie_plural": "wollen"}
        }
        main_verbs = [("Deutsch sprechen", "sprechen"), ("nach Hause gehen", "gehen"), ("Suppe kochen", "kochen")]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            dn = self.get_display_name(sub_key)
            m_inf = random.choice(list(modals.keys()))
            m_form = modals[m_inf][sub_key]
            phrase, v_inf = random.choice(main_verbs)
            
            rand = random.random()
            if rand > 0.7:
                # Error: Wrong conjugation
                wrong_sub = random.choice([k for k in self.subjects.keys() if k != sub_key])
                wrong_m = modals[m_inf][wrong_sub]
                data.append({
                    "input": f"{dn} {wrong_m} {phrase}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {dn} {m_form} {phrase}.\n📝 Пояснення: Модальне дієслово '{m_inf}' для підмета '{dn}' має форму '{m_form}'."
                })
            elif rand > 0.4 and " " in phrase:
                # Error: Position
                parts = phrase.split()
                wrong_phrase = f"{parts[1]} {parts[0]}"
                data.append({
                    "input": f"{dn} {m_form} {wrong_phrase}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {dn} {m_form} {phrase}.\n📝 Пояснення: У реченнях з модальним дієсловом ('{m_form}') основне дієслово ('{v_inf}') має стояти в самому кінці речення в інфінітиві."
                })
            else:
                data.append({"input": f"{dn} {m_form} {phrase}.", "output": "✅ Correct."})
        return data

    def generate_separable_verbs(self, count=1000):
        """A2: Separable verbs."""
        verbs = [
            ("aufstehen", "steh", "auf", "um 7 Uhr"),
            ("einkaufen", "kauf", "ein", "im Supermarkt"),
            ("anrufen", "ruf", "an", "meine Mutter")
        ]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            dn = self.get_display_name(sub_key)
            inf, stem, prefix, extra = random.choice(verbs)
            v_form = self.get_verb_form(stem, sub_key)
            
            correct = f"{dn} {v_form} {extra} {prefix}."
            if random.random() > 0.5:
                wrong = f"{dn} {prefix}{v_form} {extra}."
                data.append({
                    "input": wrong,
                    "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: Дієслово '{inf}' є відокремлюваним. У теперішньому часі приставка '{prefix}' має стояти в самому кінці речення."
                })
            else:
                data.append({"input": correct, "output": "✅ Correct."})
        return data

    def generate_reflexive_verbs(self, count=1000):
        """A2: Reflexive verbs."""
        verbs = [
            ("freuen", "freue", "auf die Ferien", "sich freuen"),
            ("waschen", "wasche", "das Gesicht", "sich waschen"),
            ("ausruhen", "ruhe", "nach der Arbeit", "sich ausruhen")
        ]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            dn = self.get_display_name(sub_key)
            inf, stem, extra, full_inf = random.choice(verbs)
            v_form = self.get_verb_form(inf[:-2] if inf.endswith("en") else inf, sub_key)
            c_refl = self.reflexive_pronouns[sub_key]
            
            if random.random() > 0.5:
                wrong_sub = random.choice([k for k in self.subjects.keys() if k != sub_key])
                w_refl = self.reflexive_pronouns[wrong_sub]
                data.append({
                    "input": f"{dn} {v_form} {w_refl} {extra}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {dn} {v_form} {c_refl} {extra}.\n📝 Пояснення: Дієслово '{full_inf}' вимагає зворотного займенника '{c_refl}' для підмета '{dn}'."
                })
            else:
                data.append({"input": f"{dn} {v_form} {c_refl} {extra}.", "output": "✅ Correct."})
        return data

    def generate_praeteritum_essentials(self, count=1000):
        """A2: Präteritum."""
        scenarios = [("war", "sein", "gestern zu Hause"), ("hatte", "haben", "viel Arbeit"), ("war", "sein", "sehr müde"), ("hatte", "haben", "Hunger")]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            dn = self.get_display_name(sub_key)
            aux_type, inf, extra = random.choice(scenarios)
            c_form = self.subjects[sub_key][aux_type]
            
            if random.random() > 0.5:
                wrong_sub = random.choice([k for k in self.subjects.keys() if k != sub_key])
                w_form = self.subjects[wrong_sub][aux_type]
                data.append({
                    "input": f"{dn} {w_form} {extra}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {dn} {c_form} {extra}.\n📝 Пояснення: У минулому часі (Präteritum) дієслово '{inf}' для '{dn}' має форму '{c_form}'."
                })
            else:
                data.append({"input": f"{dn} {c_form} {extra}.", "output": "✅ Correct."})
        return data

    def generate_imperativ(self, count=1000):
        """A1/A2: Imperativ."""
        verbs = [("gehen", "Geh", "Geht", "Gehen Sie"), ("machen", "Mach", "Macht", "Machen Sie")]
        data = []
        for _ in range(count):
            inf, du, ihr, sie = random.choice(verbs)
            rand = random.random()
            if rand > 0.7:
                data.append({
                    "input": f"Du {inf[:-2]}st!",
                    "output": f"❌ Incorrect.\n✅ Correct: {du}!\n📝 Пояснення: У наказовому способі (Imperativ) для 'du' закінчення '-st' та займенник 'du' відкидаються."
                })
            elif rand > 0.4:
                data.append({
                    "input": f"Ihr {inf}!",
                    "output": f"❌ Incorrect.\n✅ Correct: {ihr}!\n📝 Пояснення: У наказовому способі (Imperativ) для 'ihr' дієслово має закінчення '-t', але без займенника 'ihr'."
                })
            else:
                data.append({"input": f"{du}!", "output": "✅ Correct."})
        return data
