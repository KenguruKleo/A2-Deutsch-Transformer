import random
from .base import BaseGenerator

class SyntaxGenerator(BaseGenerator):
    """Generates examples for Sentence Structure: Inversion, Subordinate clauses."""

    def generate_inversion(self, count=1000):
        """A2: Word order after adverbs."""
        regular_verbs = [("spiel", "Fußball"), ("lern", "Deutsch"), ("koch", "Suppe")]
        # Irregular verbs: sein and haben with their own forms
        irregular_verbs = {
            "sein": {"ich": "bin", "du": "bist", "er": "ist", "sie": "ist", "wir": "sind", "ihr": "seid", "sie_plural": "sind"},
            "haben": {"ich": "habe", "du": "hast", "er": "hat", "sie": "hat", "wir": "haben", "ihr": "habt", "sie_plural": "haben"}
        }
        sein_complements = ["zu Hause", "müde", "krank", "in Berlin", "hier"]
        haben_complements = ["Hunger", "Zeit", "Durst", "viel Arbeit"]
        
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            dn = self.get_display_name(sub_key)
            adv = random.choice(self.time_adv)
            
            if random.random() > 0.5:
                # Use irregular verb (sein/haben)
                if random.random() > 0.5:
                    v_form = irregular_verbs["sein"][sub_key]
                    obj = random.choice(sein_complements)
                else:
                    v_form = irregular_verbs["haben"][sub_key]
                    obj = random.choice(haben_complements)
            else:
                # Use regular verb
                v_stem, obj = random.choice(regular_verbs)
                v_form = self.get_verb_form(v_stem, sub_key)
            
            correct = f"{adv} {v_form} {dn.lower()} {obj}."
            wrong = f"{adv} {dn.lower()} {v_form} {obj}."
            
            # Mix positive and negative
            if random.random() > 0.5:
                data.append({
                    "input": wrong,
                    "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: Коли речення починається з '{adv}', дієслово '{v_form}' має стояти на другому місці, перед підметом '{dn.lower()}'."
                })
            else:
                data.append({"input": correct, "output": "✅ Correct."})
        return data

    def generate_nebensatz_weil(self, count=1000):
        """A2: Subordinate clause word order (Verb at the end). Includes modal+infinitive: 'weil ich Deutsch sprechen will'."""
        # Main clause starters (variety so model sees "Ich lerne, weil..." not only "Ich esse, weil...")
        main_clauses = ["Ich esse", "Ich lerne", "Ich bleibe", "Ich bin müde", "Ich komme", "Ich trinke"]
        # Simple: (subject, verb, object) -> weil subj obj verb
        reasons_simple = [
            ("ich", "habe", "Hunger"),
            ("es", "ist", "kalt"),
            ("du", "hast", "Zeit"),
            ("er", "hat", "Durst"),
            ("sie", "ist", "krank"),
        ]
        # Modal + infinitive: (subject, modal, infinitive_phrase) -> weil subj inf phrase modal
        reasons_modal = [
            ("ich", "will", "Deutsch sprechen"),
            ("ich", "muss", "nach Hause gehen"),
            ("ich", "kann", "gut kochen"),
            ("du", "willst", "Kaffee trinken"),
            ("er", "muss", "arbeiten"),
            ("sie", "will", "Deutsch lernen"),
        ]
        data = []
        for _ in range(count):
            main = random.choice(main_clauses)
            if random.random() < 0.4:
                # Modal + infinitive in weil-clause (so "Ich lerne, weil ich Deutsch sprechen will." appears)
                sub_key, modal, inf_phrase = random.choice(reasons_modal)
                correct = f"{main}, weil {sub_key} {inf_phrase} {modal}."
                wrong = f"{main}, weil {sub_key} {modal} {inf_phrase}."
                verb_at_end = modal
            else:
                # Simple weil-clause
                sub_key, aux, obj = random.choice(reasons_simple)
                correct = f"{main}, weil {sub_key} {obj} {aux}."
                wrong = f"{main}, weil {sub_key} {aux} {obj}."
                verb_at_end = aux
            if random.random() > 0.5:
                data.append({
                    "input": wrong,
                    "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: У підрядному реченні зі сполучником 'weil' дієслово '{verb_at_end}' має стояти в самому кінці речення."
                })
            else:
                data.append({"input": correct, "output": "✅ Correct."})
        return data

    def generate_questions(self, count=1000):
        """A1: W-Questions word order. Includes Wo wohnt er?, Was macht er? etc."""
        questions = [
            ("Wo", "wohn", "du", ""),
            ("Wo", "wohn", "er", ""),
            ("Was", "mach", "er", "heute"),
            ("Wann", "komm", "wir", ""),
            ("Wann", "komm", "du", ""),
        ]
        data = []
        for _ in range(count):
            w_word, stem, sub_key, extra = random.choice(questions)
            v_form = self.get_verb_form(stem, sub_key)
            
            correct = f"{w_word} {v_form} {sub_key}{' ' + extra if extra else ''}?"
            wrong = f"{w_word} {sub_key} {v_form}{' ' + extra if extra else ''}?"
            
            if random.random() > 0.5:
                data.append({
                    "input": wrong,
                    "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: У запитаннях після питального слова '{w_word}' дієслово '{v_form}' має стояти на другому місці, перед підметом '{sub_key}'."
                })
            else:
                data.append({"input": correct, "output": "✅ Correct."})
        # Fixed correct W-questions (irregular/fixed phrase) so model learns "Wie heißt du?" etc. as correct
        for q in ["Wie heißt du?", "Wie geht es dir?"]:
            for _ in range(max(1, count // 50)):
                data.append({"input": q, "output": "✅ Correct."})
        return data

    def generate_nebensatz_dass_wenn(self, count=1000):
        """A2: Subordinate clauses mit dass, wenn (Verb at the end). Includes 'Ich denke, dass...' and 'Ich glaube, dass wir Deutsch lernen' as correct."""
        conjunctions = ["dass", "wenn"]
        scenarios = [
            ("ich", "habe", "Zeit", "Ich komme,"),
            ("du", "hast", "Zeit", "Ich denke,"),
            ("er", "ist", "krank", "Ich glaube,"),
            ("wir", "lernen", "Deutsch", "Ich glaube,"),
            ("wir", "lernen", "Deutsch", "Es ist gut,"),
        ]
        data = []
        for _ in range(count):
            sub, verb, obj, main = random.choice(scenarios)
            conj = random.choice(conjunctions)
            
            if random.random() > 0.5:
                data.append({
                    "input": f"{main} {conj} {sub} {verb} {obj}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {main} {conj} {sub} {obj} {verb}.\n📝 Пояснення: У підрядному реченні зі сполучником '{conj}' дієслово '{verb}' має стояти в самому кінці речення."
                })
            else:
                data.append({"input": f"{main} {conj} {sub} {obj} {verb}.", "output": "✅ Correct."})
        return data

    def generate_negation(self, count=1000):
        """A1: Negation with 'nicht' vs 'kein'. Covers kein/keine/keinen (Akk) after haben."""
        # (noun, gender) -> after "Ich habe" use: keinen (m), kein (n), keine (f)
        nouns = [
            ("Hunger", "m"), ("Durst", "m"), ("Appetit", "m"),
            ("Auto", "n"), ("Geld", "n"), ("Brot", "n"), ("Buch", "n"),
            ("Zeit", "f"), ("Luft", "f"), ("Milch", "f"), ("Arbeit", "f"),
        ]
        adjectives = [("gut", "Das ist"), ("kalt", "Es ist"), ("teuer", "Das ist"), ("schnell", "Er ist")]
        
        data = []
        for _ in range(count):
            if random.random() > 0.5:
                # Noun negation (should be kein/keine/keinen in Akkusativ)
                noun, gender = random.choice(nouns)
                c_neg = "keinen" if gender == "m" else ("keine" if gender == "f" else "kein")
                if random.random() > 0.5:
                    data.append({
                        "input": f"Ich habe nicht {noun}.",
                        "output": f"❌ Incorrect.\n✅ Correct: Ich habe {c_neg} {noun}.\n📝 Пояснення: Для заперечення іменників (без означеного артикля) використовується '{c_neg}', а не 'nicht'."
                    })
                else:
                    data.append({"input": f"Ich habe {c_neg} {noun}.", "output": "✅ Correct."})
            else:
                # Adjective negation (should be nicht)
                adj, prefix = random.choice(adjectives)
                if random.random() > 0.5:
                    data.append({
                        "input": f"{prefix} kein {adj}.",
                        "output": f"❌ Incorrect.\n✅ Correct: {prefix} nicht {adj}.\n📝 Пояснення: Для заперечення прикметників або обставин використовується 'nicht', а не 'kein'."
                    })
                else:
                    data.append({"input": f"{prefix} nicht {adj}.", "output": "✅ Correct."})
        return data
