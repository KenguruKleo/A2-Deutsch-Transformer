import random
from pathlib import Path

class BaseGenerator:
    """Base class with shared vocabulary and helper methods for all generators."""
    
    def __init__(self):
        # Subjects with conjugation data
        self.subjects = {
            "ich": {"bin": "bin", "habe": "habe", "war": "war", "hatte": "hatte", "ending": "e"},
            "du": {"bin": "bist", "habe": "hast", "war": "warst", "hatte": "hattest", "ending": "st"},
            "er": {"bin": "ist", "habe": "hat", "war": "war", "hatte": "hatte", "ending": "t"},
            "sie": {"bin": "ist", "habe": "hat", "war": "war", "hatte": "hatte", "ending": "t"},
            "wir": {"bin": "sind", "habe": "haben", "war": "waren", "hatte": "hatten", "ending": "en"},
            "ihr": {"bin": "seid", "habe": "habt", "war": "wart", "hatte": "hattet", "ending": "t"},
            "sie_plural": {"bin": "sind", "habe": "haben", "war": "waren", "hatte": "hatten", "ending": "en", "display": "Sie"},
        }

        # Reflexive pronouns mapping
        self.reflexive_pronouns = {
            "ich": "mich",
            "du": "dich",
            "er": "sich",
            "sie": "sich",
            "es": "sich",
            "wir": "uns",
            "ihr": "euch",
            "sie_plural": "sich"
        }

        # Possessive pronouns (Nominative forms)
        self.possessives = {
            "ich": "mein",
            "du": "dein",
            "er": "sein",
            "sie": "ihr",
            "wir": "unser",
            "ihr": "euer",
            "sie_plural": "ihr"
        }

        # Nouns by category and gender (for future cases)
        self.nouns = {
            "food": [
                ("Pizza", "f"), ("Brot", "n"), ("Eis", "n"), ("Kaffee", "m"), 
                ("Apfel", "m"), ("Kuchen", "m"), ("Suppe", "f"), ("Bier", "n")
            ],
            "place": [
                ("nach Hause", ""), ("nach Berlin", ""), ("ins Kino", ""), 
                ("in die Schule", ""), ("zum Arzt", ""), ("nach München", "")
            ],
            "objects": [
                ("ein Buch", "n"), ("einen Brief", "m"), ("eine E-Mail", "f")
            ]
        }

        self.time_adv = ["Heute", "Morgen", "Dann", "Jetzt", "Am Montag", "Nach der Arbeit"]

    def get_display_name(self, sub_key):
        """Returns the display name for a subject key (handles sie_plural -> Sie)."""
        display = self.subjects[sub_key].get("display")
        return display if display else sub_key.capitalize()

    def get_verb_form(self, verb_stem, sub_key):
        """Returns the correct verb form based on the stem and subject."""
        # Special case for 'essen'
        if verb_stem == "ess":
            forms = {"ich": "esse", "du": "isst", "er": "isst", "sie": "isst", "wir": "essen", "ihr": "esst", "sie_plural": "essen"}
            return forms.get(sub_key, "essen")
            
        ending = self.subjects[sub_key]["ending"]
        if verb_stem.endswith('t') and ending in ['st', 't']:
            return verb_stem + 'e' + ending
        return verb_stem + ending
