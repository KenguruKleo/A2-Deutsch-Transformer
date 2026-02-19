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
            
            data.append({
                "input": f"{adv} {sub_key} {v_form} {obj}.",
                "output": f"❌ Incorrect.\n✅ Correct: {adv} {v_form} {sub_key} {obj}.\n📝 Пояснення: Коли речення починається з '{adv}', дієслово '{v_form}' має стояти на другому місці, перед підметом '{sub_key}'."
            })
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
            # Incorrect: weil ich habe Hunger.
            # Correct: weil ich Hunger habe.
            data.append({
                "input": f"Ich esse, weil {sub_key} {aux} {obj}.",
                "output": f"❌ Incorrect.\n✅ Correct: Ich esse, weil {sub_key} {obj} {aux}.\n📝 Пояснення: У підрядному реченні зі сполучником 'weil' дієслово '{aux}' має стояти в самому кінці."
            })
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
            
            # Correct: Wo wohnst du?
            # Wrong: Wo du wohnst?
            correct = f"{w_word} {v_form} {sub_key}{' ' + extra if extra else ''}?"
            wrong = f"{w_word} {sub_key} {v_form}{' ' + extra if extra else ''}?"
            
            data.append({
                "input": wrong,
                "output": f"❌ Incorrect.\n✅ Correct: {correct}\n📝 Пояснення: У запитаннях після питального слова '{w_word}' дієслово '{v_form}' має стояти на другому місці, перед підметом '{sub_key}'."
            })
        return data
