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
