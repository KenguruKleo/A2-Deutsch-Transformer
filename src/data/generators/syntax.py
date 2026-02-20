import random
from .base import BaseGenerator

class SyntaxGenerator(BaseGenerator):
    """Generates examples for Sentence Structure: Inversion, Subordinate clauses."""

    def generate_inversion(self, count=1000):
        """A2: Word order after adverbs."""
        verbs = [("spiel", "Fußball"), ("lern", "Deutsch"), ("koch", "Suppe")]
        data = []
        for _ in range(count):
            sub_key = random.choice(list(self.subjects.keys()))
            adv = random.choice(self.time_adv)
            v_stem, obj = random.choice(verbs)
            v_form = self.get_verb_form(v_stem, sub_key)
            
            correct = f"{adv} {v_form} {sub_key} {obj}."
            wrong = f"{adv} {sub_key} {v_form} {obj}."
            
            # Mix positive and negative
            if random.random() > 0.4:
                data.append({
                    "input": wrong,
                    "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: Коли речення починається з '{adv}', дієслово '{v_form}' має стояти на другому місці, перед підметом '{sub_key}'."
                })
            else:
                data.append({"input": correct, "output": "✅ Correct."})
        return data

    def generate_nebensatz_weil(self, count=1000):
        """A2: Subordinate clause word order (Verb at the end)."""
        reasons = [
            ("ich", "habe", "Hunger"), 
            ("es", "ist", "kalt"), 
            ("du", "hast", "Zeit")
        ]
        
        data = []
        for _ in range(count):
            sub_key, aux, obj = random.choice(reasons)
            correct = f"Ich esse, weil {sub_key} {obj} {aux}."
            wrong = f"Ich esse, weil {sub_key} {aux} {obj}."
            
            if random.random() > 0.4:
                data.append({
                    "input": wrong,
                    "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: У підрядному реченні зі сполучником 'weil' дієслово '{aux}' має стояти в самому кінці речення."
                })
            else:
                data.append({"input": correct, "output": "✅ Correct."})
        return data

    def generate_questions(self, count=1000):
        """A1: W-Questions word order."""
        questions = [
            ("Wo", "wohn", "du", ""), 
            ("Was", "mach", "er", "heute"), 
            ("Wann", "komm", "wir", "")
        ]
        data = []
        for _ in range(count):
            w_word, stem, sub_key, extra = random.choice(questions)
            v_form = self.get_verb_form(stem, sub_key)
            
            correct = f"{w_word} {v_form} {sub_key}{' ' + extra if extra else ''}?"
            wrong = f"{w_word} {sub_key} {v_form}{' ' + extra if extra else ''}?"
            
            if random.random() > 0.4:
                data.append({
                    "input": wrong,
                    "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: У запитаннях після питального слова '{w_word}' дієслово '{v_form}' має стояти на другому місці, перед підметом '{sub_key}'."
                })
            else:
                data.append({"input": correct, "output": "✅ Correct."})
        return data

    def generate_nebensatz_dass_wenn(self, count=1000):
        """A2: Subordinate clauses mit dass, wenn (Verb at the end)."""
        conjunctions = ["dass", "wenn"]
        scenarios = [
            ("ich", "habe", "Zeit", "Ich komme,"),
            ("er", "ist", "krank", "Ich glaube,"),
            ("wir", "lernen", "Deutsch", "Es ist gut,")
        ]
        data = []
        for _ in range(count):
            sub, verb, obj, main = random.choice(scenarios)
            conj = random.choice(conjunctions)
            
            if random.random() > 0.4:
                # Error: Verb not in the end
                data.append({
                    "input": f"{main} {conj} {sub} {verb} {obj}.",
                    "output": f"❌ Incorrect.\n✅ Correct: {main} {conj} {sub} {obj} {verb}.\n📝 Пояснення: У підрядному реченні зі сполучником '{conj}' дієслово '{verb}' має стояти в самому кінці речення."
                })
            else:
                # Correct: Verb at the end
                data.append({"input": f"{main} {conj} {sub} {obj} {verb}.", "output": "✅ Correct."})
        return data

    def generate_negation(self, count=1000):
        """A1: Negation with 'nicht' vs 'kein'."""
        nouns = [("Hunger", "m"), ("Auto", "n"), ("Zeit", "f")]
        adjectives = [("gut", "Das ist"), ("kalt", "Es ist")]
        
        data = []
        for _ in range(count):
            if random.random() > 0.5:
                # Noun negation (should be kein)
                noun, gender = random.choice(nouns)
                c_neg = "kein" if gender != "f" else "keine"
                if random.random() > 0.4:
                    data.append({
                        "input": f"Ich habe nicht {noun}.",
                        "output": f"❌ Incorrect.\n✅ Correct: Ich habe {c_neg} {noun}.\n📝 Пояснення: Для заперечення іменників (без означеного артикля) використовується '{c_neg}', а не 'nicht'."
                    })
                else:
                    data.append({"input": f"Ich habe {c_neg} {noun}.", "output": "✅ Correct."})
            else:
                # Adjective negation (should be nicht)
                adj, prefix = random.choice(adjectives)
                if random.random() > 0.4:
                    data.append({
                        "input": f"{prefix} kein {adj}.",
                        "output": f"❌ Incorrect.\n✅ Correct: {prefix} nicht {adj}.\n📝 Пояснення: Для заперечення прикметників або обставин використовується 'nicht', а не 'kein'."
                    })
                else:
                    data.append({"input": f"{prefix} nicht {adj}.", "output": "✅ Correct."})
        return data
