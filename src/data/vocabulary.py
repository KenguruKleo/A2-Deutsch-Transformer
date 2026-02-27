"""
vocabulary.py — A1/A2 German vocabulary bank for prompt seeding (~1200 words).

Used by llm_generator.py to inject random words into each prompt,
ensuring diverse training examples even when the grammar topic repeats.

Sources: Goethe A1/A2 word list, Netzwerk A1/A2, Begegnungen A2.
"""

from __future__ import annotations

# ── Nouns (Substantive) ───────────────────────────────────────────────────────

NOUNS: list[tuple[str, str]] = [
    # People & family
    ("Mann", "der"), ("Frau", "die"), ("Kind", "das"), ("Mutter", "die"),
    ("Vater", "der"), ("Bruder", "der"), ("Schwester", "die"), ("Freund", "der"),
    ("Freundin", "die"), ("Sohn", "der"), ("Tochter", "die"), ("Oma", "die"),
    ("Opa", "der"), ("Onkel", "der"), ("Tante", "die"), ("Nichte", "die"),
    ("Neffe", "der"), ("Junge", "der"), ("Mädchen", "das"), ("Herr", "der"),
    ("Dame", "die"), ("Mensch", "der"), ("Person", "die"), ("Gast", "der"),
    # Professions
    ("Lehrer", "der"), ("Lehrerin", "die"), ("Arzt", "der"), ("Ärztin", "die"),
    ("Student", "der"), ("Studentin", "die"), ("Kollege", "der"), ("Kollegin", "die"),
    ("Chef", "der"), ("Chefin", "die"), ("Nachbar", "der"), ("Nachbarin", "die"),
    ("Verkäufer", "der"), ("Kellner", "der"), ("Fahrer", "der"), ("Polizist", "der"),
    ("Ingenieur", "der"), ("Informatiker", "der"), ("Anwalt", "der"), ("Richter", "der"),
    ("Journalist", "der"), ("Fotograf", "der"), ("Koch", "der"), ("Bäcker", "der"),
    ("Elektriker", "der"), ("Klempner", "der"), ("Architekt", "der"), ("Pilot", "der"),
    ("Krankenschwester", "die"), ("Mechaniker", "der"), ("Buchhalter", "der"),
    # Home & rooms
    ("Haus", "das"), ("Wohnung", "die"), ("Zimmer", "das"), ("Küche", "die"),
    ("Schlafzimmer", "das"), ("Wohnzimmer", "das"), ("Esszimmer", "das"),
    ("Bad", "das"), ("Flur", "der"), ("Keller", "der"), ("Dach", "das"),
    ("Balkon", "der"), ("Garage", "die"), ("Garten", "der"), ("Treppe", "die"),
    # Furniture & home objects
    ("Bett", "das"), ("Tisch", "der"), ("Stuhl", "der"), ("Sofa", "das"),
    ("Schrank", "der"), ("Regal", "das"), ("Lampe", "die"), ("Spiegel", "der"),
    ("Fenster", "das"), ("Tür", "die"), ("Wand", "die"), ("Boden", "der"),
    ("Fernseher", "der"), ("Kühlschrank", "der"), ("Herd", "der"), ("Mikrowelle", "die"),
    ("Waschmaschine", "die"), ("Staubsauger", "der"), ("Bügeleisen", "das"),
    ("Tasse", "die"), ("Teller", "der"), ("Glas", "das"), ("Topf", "der"),
    ("Löffel", "der"), ("Gabel", "die"), ("Messer", "das"), ("Blume", "die"),
    ("Pflanze", "die"), ("Bild", "das"), ("Foto", "das"),
    # Technology
    ("Computer", "der"), ("Laptop", "der"), ("Handy", "das"), ("Telefon", "das"),
    ("Radio", "das"), ("Fernseher", "der"), ("App", "die"), ("Programm", "das"),
    ("Datei", "die"), ("Ordner", "der"), ("Bildschirm", "der"), ("Tastatur", "die"),
    ("Drucker", "der"), ("Kamera", "die"), ("Akku", "der"), ("Kabel", "das"),
    ("WLAN", "das"), ("Internet", "das"), ("Website", "die"), ("Passwort", "das"),
    # Personal items
    ("Tasche", "die"), ("Schlüssel", "der"), ("Uhr", "die"), ("Brille", "die"),
    ("Brief", "der"), ("Paket", "das"), ("Ausweis", "der"),
    # Food & drink
    ("Brot", "das"), ("Brötchen", "das"), ("Wasser", "das"), ("Kaffee", "der"),
    ("Tee", "der"), ("Milch", "die"), ("Saft", "der"), ("Bier", "das"),
    ("Wein", "der"), ("Apfel", "der"), ("Birne", "die"), ("Banane", "die"),
    ("Orange", "die"), ("Erdbeere", "die"), ("Traube", "die"), ("Zitrone", "die"),
    ("Kuchen", "der"), ("Torte", "die"), ("Keks", "der"), ("Schokolade", "die"),
    ("Eis", "das"), ("Suppe", "die"), ("Salat", "der"), ("Fleisch", "das"),
    ("Fisch", "der"), ("Hähnchen", "das"), ("Wurst", "die"), ("Schinken", "der"),
    ("Käse", "der"), ("Ei", "das"), ("Butter", "die"), ("Öl", "das"),
    ("Zucker", "der"), ("Salz", "das"), ("Pfeffer", "der"), ("Mehl", "das"),
    ("Reis", "der"), ("Nudeln", "die"), ("Kartoffel", "die"), ("Zwiebel", "die"),
    ("Tomate", "die"), ("Paprika", "die"), ("Gurke", "die"), ("Gemüse", "das"),
    ("Obst", "das"), ("Frühstück", "das"), ("Mittagessen", "das"), ("Abendessen", "das"),
    # City & places
    ("Stadt", "die"), ("Dorf", "das"), ("Straße", "die"), ("Platz", "der"),
    ("Brücke", "die"), ("Bahnhof", "der"), ("Flughafen", "der"),
    ("Kino", "das"), ("Theater", "das"), ("Museum", "das"), ("Bibliothek", "die"),
    ("Supermarkt", "der"), ("Markt", "der"), ("Kaufhaus", "das"), ("Laden", "der"),
    ("Bank", "die"), ("Post", "die"), ("Rathaus", "das"), ("Kirche", "die"),
    ("Park", "der"), ("Zoo", "der"), ("Schwimmbad", "das"), ("Stadion", "das"),
    ("Restaurant", "das"), ("Café", "das"), ("Hotel", "das"), ("Bäckerei", "die"),
    ("Metzgerei", "die"), ("Friseur", "der"), ("Tankstelle", "die"), ("Werkstatt", "die"),
    ("Apotheke", "die"), ("Krankenhaus", "das"), ("Schule", "die"), ("Universität", "die"),
    ("Büro", "das"), ("Fabrik", "die"),
    # Transport
    ("Auto", "das"), ("Bus", "der"), ("Zug", "der"), ("Straßenbahn", "die"),
    ("U-Bahn", "die"), ("Flugzeug", "das"), ("Fahrrad", "das"), ("Motorrad", "das"),
    ("Taxi", "das"), ("Schiff", "das"), ("Ticket", "das"), ("Fahrkarte", "die"),
    ("Haltestelle", "die"), ("Stau", "der"), ("Parkplatz", "der"),
    # Nature & weather
    ("Sonne", "die"), ("Mond", "der"), ("Stern", "der"), ("Regen", "der"),
    ("Schnee", "der"), ("Wind", "der"), ("Sturm", "der"), ("Wolke", "die"),
    ("Hitze", "die"), ("Kälte", "die"), ("Nebel", "der"),
    ("Baum", "der"), ("Wald", "der"), ("Wiese", "die"), ("Feld", "das"),
    ("Berg", "der"), ("Tal", "das"), ("Fluss", "der"), ("See", "der"),
    ("Meer", "das"), ("Strand", "der"), ("Sand", "der"), ("Hügel", "der"),
    ("Luft", "die"), ("Erde", "die"), ("Feuer", "das"), ("Stein", "der"),
    ("Gras", "das"), ("Blatt", "das"), ("Ast", "der"), ("Wurzel", "die"),
    # Animals
    ("Hund", "der"), ("Katze", "die"), ("Vogel", "der"), ("Pferd", "das"),
    ("Kuh", "die"), ("Schwein", "das"), ("Schaf", "das"), ("Maus", "die"),
    ("Hase", "der"), ("Bär", "der"), ("Fuchs", "der"),
    # Time & calendar
    ("Tag", "der"), ("Nacht", "die"), ("Woche", "die"), ("Monat", "der"),
    ("Jahr", "das"), ("Stunde", "die"), ("Minute", "die"), ("Sekunde", "die"),
    ("Morgen", "der"), ("Mittag", "der"), ("Abend", "der"), ("Wochenende", "das"),
    ("Montag", "der"), ("Dienstag", "der"), ("Mittwoch", "der"), ("Donnerstag", "der"),
    ("Freitag", "der"), ("Samstag", "der"), ("Sonntag", "der"),
    ("Januar", "der"), ("Februar", "der"), ("März", "der"), ("April", "der"),
    ("Mai", "der"), ("Juni", "der"), ("Juli", "der"), ("August", "der"),
    ("September", "der"), ("Oktober", "der"), ("November", "der"), ("Dezember", "der"),
    ("Frühling", "der"), ("Sommer", "der"), ("Herbst", "der"), ("Winter", "der"),
    ("Urlaub", "der"), ("Ferien", "die"), ("Feiertag", "der"), ("Geburtstag", "der"),
    ("Weihnachten", "das"), ("Ostern", "das"), ("Silvester", "das"),
    # Work & education
    ("Arbeit", "die"), ("Job", "der"), ("Beruf", "der"), ("Termin", "der"),
    ("Besprechung", "die"), ("Projekt", "das"), ("Aufgabe", "die"), ("Prüfung", "die"),
    ("Note", "die"), ("Kurs", "der"), ("Unterricht", "der"), ("Hausaufgabe", "die"),
    ("Pause", "die"), ("Stift", "der"), ("Heft", "das"), ("Tafel", "die"),
    ("Klasse", "die"), ("Schüler", "der"), ("Professor", "der"),
    ("Bleistift", "der"), ("Kugelschreiber", "der"), ("Radiergummi", "der"),
    ("Lineal", "das"), ("Schere", "die"), ("Papier", "das"), ("Kalender", "der"),
    ("Notizbuch", "das"), ("Buch", "das"), ("Zeitung", "die"),
    # Hobbies & leisure
    ("Sport", "der"), ("Fußball", "der"), ("Tennis", "das"),
    ("Musik", "die"), ("Gitarre", "die"), ("Klavier", "das"), ("Lied", "das"),
    ("Konzert", "das"), ("Film", "der"), ("Serie", "die"), ("Party", "die"),
    ("Reise", "die"), ("Ausflug", "der"), ("Spiel", "das"),
    ("Roman", "der"), ("Gedicht", "das"), ("Hobby", "das"),
    # Body & health
    ("Kopf", "der"), ("Gesicht", "das"), ("Auge", "das"), ("Ohr", "das"),
    ("Nase", "die"), ("Mund", "der"), ("Zahn", "der"), ("Hals", "der"),
    ("Schulter", "die"), ("Arm", "der"), ("Hand", "die"), ("Finger", "der"),
    ("Bauch", "der"), ("Rücken", "der"), ("Bein", "das"), ("Knie", "das"),
    ("Fuß", "der"), ("Herz", "das"), ("Lunge", "die"),
    ("Schmerz", "der"), ("Erkältung", "die"), ("Fieber", "das"), ("Allergie", "die"),
    ("Medikament", "das"), ("Tablette", "die"), ("Rezept", "das"),
    ("Schmerzmittel", "das"), ("Pflaster", "das"), ("Salbe", "die"),
    # Shopping & money
    ("Geld", "das"), ("Euro", "der"), ("Preis", "der"), ("Rechnung", "die"),
    ("Quittung", "die"), ("Kasse", "die"), ("Angebot", "das"), ("Rabatt", "der"),
    ("Größe", "die"), ("Farbe", "die"),
    # Clothes
    ("Kleid", "das"), ("Bluse", "die"), ("Hemd", "das"), ("Hose", "die"),
    ("Rock", "der"), ("Jacke", "die"), ("Mantel", "der"), ("Pullover", "der"),
    ("T-Shirt", "das"), ("Schuhe", "die"), ("Stiefel", "die"), ("Socken", "die"),
    ("Mütze", "die"), ("Handschuhe", "die"), ("Gürtel", "der"),
    # Communication & abstract
    ("Gespräch", "das"), ("Frage", "die"), ("Antwort", "die"), ("Nachricht", "die"),
    ("E-Mail", "die"), ("Anruf", "der"), ("Adresse", "die"), ("Name", "der"),
    ("Nummer", "die"), ("Formular", "das"),
    ("Grund", "der"), ("Meinung", "die"), ("Idee", "die"), ("Plan", "der"),
    ("Problem", "das"), ("Lösung", "die"), ("Fehler", "der"), ("Ergebnis", "das"),
    ("Erfahrung", "die"), ("Gewohnheit", "die"), ("Regel", "die"), ("Beispiel", "das"),
    ("Unterschied", "der"), ("Möglichkeit", "die"), ("Entscheidung", "die"),
    ("Interesse", "das"), ("Wunsch", "der"), ("Traum", "der"),
    ("Angst", "die"), ("Hoffnung", "die"), ("Freude", "die"), ("Liebe", "die"),
    ("Freundschaft", "die"), ("Vertrauen", "das"), ("Respekt", "der"),
    # Quantities
    ("Stück", "das"), ("Kilo", "das"), ("Liter", "der"), ("Meter", "der"),
    ("Kilometer", "der"), ("Prozent", "das"), ("Hälfte", "die"), ("Viertel", "das"),
]

# ── Verbs (Verben) ────────────────────────────────────────────────────────────

VERBS: list[str] = [
    # Common regular
    "kaufen", "kochen", "lernen", "arbeiten", "spielen", "wohnen",
    "fragen", "antworten", "machen", "brauchen", "suchen", "hören",
    "zeigen", "warten", "bestellen", "bezahlen", "buchen", "öffnen",
    "schließen", "beginnen", "enden", "schreiben", "lesen", "zeichnen",
    "tanzen", "singen", "kosten", "bedeuten", "erklären", "erzählen",
    "vergessen", "glauben", "meinen", "sagen", "informieren",
    "reservieren", "mieten", "verkaufen", "zahlen", "wechseln",
    "verdienen", "sparen", "ausgeben", "benutzen", "verwenden",
    "folgen", "führen", "bringen", "schicken", "empfangen",
    "besuchen", "kennen", "wissen", "studieren", "unterrichten",
    "üben", "wiederholen", "feiern", "lachen", "weinen",
    "frühstücken", "rauchen", "atmen", "schwimmen",
    "rennen", "gehen", "fahren", "fliegen", "reisen", "wandern",
    "steigen", "klettern", "fallen", "heizen", "putzen",
    "waschen", "backen", "pflanzen", "gießen", "füttern",
    "spazieren", "joggen", "trainieren", "fotografieren", "basteln", "sammeln",
    # Cognitive & communication
    "hoffen", "wünschen", "träumen", "planen", "entscheiden", "wählen",
    "vergleichen", "prüfen", "testen", "kontrollieren", "untersuchen",
    "messen", "zählen", "rechnen", "erkennen", "bemerken", "verstehen",
    "beschreiben", "berichten", "diskutieren", "streiten", "einigen",
    "zustimmen", "ablehnen", "bitten", "danken", "entschuldigen",
    "begrüßen", "versprechen", "warnen", "empfehlen", "erlauben", "verbieten",
    # Cooking & household
    "braten", "schneiden", "hacken", "mischen", "rühren",
    "salzen", "würzen", "servieren", "decken", "abräumen", "spülen",
    "trocknen", "bügeln", "staubsaugen", "wischen", "schrubben", "reparieren",
    # Digital
    "herunterladen", "hochladen", "speichern", "löschen", "kopieren",
    "drucken", "scannen", "posten", "teilen", "kommentieren", "googeln",
    "bestätigen", "stornieren",
    # Change & progress
    "gewinnen", "verlieren", "aufhören", "beenden", "aufgeben",
    "verbessern", "zunehmen", "abnehmen", "wachsen", "bauen", "zerstören",
    # Weather & nature
    "regnen", "schneien", "scheinen", "wehen", "blitzen", "donnern",
    # Strong / irregular
    "fahren", "schlafen", "sehen", "geben", "nehmen", "kommen",
    "stehen", "liegen", "tragen", "treffen", "heißen",
    "denken", "rennen", "sprechen", "essen", "trinken", "helfen",
    "gefallen", "laufen", "sitzen",
    # Separable
    "aufmachen", "zumachen", "aufstehen", "anrufen", "einkaufen", "ausgehen",
    "ankommen", "mitbringen", "aufräumen", "abholen", "zurückkommen",
    "aufhören", "anfangen", "fernsehen", "vorstellen", "anziehen", "ausziehen",
    "einsteigen", "aussteigen", "umsteigen", "abfahren", "anhalten",
    "aufpassen", "zuhören", "nachfragen", "durchlesen", "ausfüllen",
    "teilnehmen", "weitermachen", "aufschreiben", "nachschlagen",
    "einladen", "absagen", "vorbereiten", "zurückgeben", "weggehen",
    "mitnehmen", "weitergehen", "aufessen", "austrinken", "kennenlernen",
    # Modal
    "können", "müssen", "wollen", "sollen", "dürfen", "möchten",
    # Reflexive
    "sich freuen", "sich setzen", "sich waschen", "sich fühlen",
    "sich vorstellen", "sich erinnern", "sich beeilen", "sich befinden",
    "sich treffen", "sich verabschieden", "sich entscheiden", "sich interessieren",
    "sich beschweren", "sich erholen", "sich ausruhen", "sich anmelden",
    # Auxiliary
    "haben", "sein", "werden",
]

# ── Adjectives (Adjektive) ────────────────────────────────────────────────────

ADJECTIVES: list[str] = [
    # Size & age
    "groß", "klein", "lang", "kurz", "breit", "eng", "hoch", "niedrig",
    "alt", "jung", "neu", "modern", "antik", "frisch",
    # Quality
    "gut", "schlecht", "schön", "hässlich", "toll", "prima", "super",
    "wunderbar", "schrecklich", "furchtbar", "ausgezeichnet", "perfekt",
    "richtig", "falsch", "wichtig", "unwichtig", "nützlich", "unnötig",
    # Size/weight/speed
    "leicht", "schwer", "schnell", "langsam", "dünn", "dick",
    # Temperature & weather
    "kalt", "warm", "heiß", "kühl", "nass", "trocken", "sonnig",
    "regnerisch", "windig", "bewölkt", "frostig", "schwül",
    # Light & colour
    "hell", "dunkel", "bunt", "rot", "blau", "grün", "gelb",
    "schwarz", "weiß", "grau", "orange", "lila", "rosa", "braun",
    # Price & quantity
    "teuer", "billig", "günstig", "kostenlos", "voll", "leer",
    "viel", "wenig", "genug", "ausreichend", "knapp",
    # Emotions & personality
    "froh", "traurig", "glücklich", "unglücklich", "zufrieden", "unzufrieden",
    "aufgeregt", "nervös", "ruhig", "laut", "leise", "müde", "wach",
    "hungrig", "satt", "durstig", "krank", "gesund", "fit",
    "freundlich", "unfreundlich", "nett", "höflich", "unhöflich", "lustig",
    "langweilig", "interessant", "spannend", "aufregend", "seltsam", "komisch",
    "fleißig", "faul", "klug", "dumm", "mutig", "ängstlich",
    # State/condition
    "offen", "geschlossen", "kaputt", "ganz", "sauber", "schmutzig",
    "ordentlich", "unordentlich", "einfach", "schwierig", "möglich", "unmöglich",
    "gefährlich", "sicher", "pünktlich", "unpünktlich", "bereit", "fertig",
    # Spatial
    "nah", "weit", "gerade", "schief", "oben", "unten",
    # Texture & shape
    "rund", "eckig", "flach", "tief", "rau", "glatt",
    "hart", "weich", "fest", "locker", "dicht",
    # Taste & smell
    "süß", "sauer", "bitter", "salzig", "scharf", "mild", "lecker", "eklig",
    # Social
    "beliebt", "bekannt", "berühmt", "öffentlich", "privat",
    "sozial", "kulturell", "international",
    # Medical
    "schwach", "stark", "verletzt", "erholt", "erschöpft",
    # Nature
    "natürlich", "künstlich", "organisch", "ökologisch",
]

# ── Adverbs & time/place expressions ─────────────────────────────────────────

ADVERBS: list[str] = [
    # Time
    "heute", "morgen", "übermorgen", "gestern", "vorgestern",
    "jetzt", "gerade", "gleich", "sofort", "bald", "später", "früher",
    "immer", "nie", "oft", "manchmal", "selten", "meistens",
    "normalerweise", "gewöhnlich", "täglich", "wöchentlich", "monatlich",
    "schon", "noch", "bereits", "endlich", "früh", "spät",
    "zuerst", "dann", "danach", "anschließend", "zuletzt", "schließlich",
    "am Morgen", "am Mittag", "am Abend", "in der Nacht",
    "am Wochenende", "in der Woche", "nächste Woche", "letzte Woche",
    "nächsten Monat", "nächstes Jahr", "letztes Jahr",
    "jeden Tag", "jede Woche", "zweimal pro Woche", "einmal im Monat",
    "vor einer Stunde", "in zwei Stunden", "seit gestern",
    # Place
    "hier", "dort", "da", "überall", "nirgends", "irgendwo",
    "draußen", "drinnen", "oben", "unten", "vorne", "hinten",
    "links", "rechts", "geradeaus", "nebenan", "gegenüber",
    "nach Hause", "zu Hause", "in der Schule", "im Büro", "auf der Arbeit",
    # Degree
    "sehr", "ziemlich", "etwas", "ein bisschen", "kaum", "gar nicht",
    "zu", "genug", "fast", "ungefähr", "genauso",
    "besonders", "wirklich", "absolut", "total", "völlig",
    # Modal / attitude
    "gern", "lieber", "am liebsten", "leider", "natürlich", "sicher",
    "vielleicht", "wahrscheinlich", "bestimmt", "eigentlich", "doch",
    "auch", "nur", "noch", "nicht",
    "zusammen", "allein", "gemeinsam",
    # Connectors
    "trotzdem", "deshalb", "deswegen", "daher", "außerdem", "dennoch", "sonst",
    "erstens", "zweitens", "einerseits", "andererseits",
    "im Vergleich dazu", "im Gegensatz dazu", "genauso wie",
    # Stance
    "glücklicherweise", "zum Glück", "bedauerlicherweise",
    "ehrlich gesagt", "grundsätzlich", "im Prinzip", "in der Regel",
    "meiner Meinung nach", "laut", "angeblich",
]

# ── Full flat list for random sampling ───────────────────────────────────────

_ALL_NOUNS_FLAT: list[str] = [f"{article} {noun}" for noun, article in NOUNS]
ALL_WORDS: list[str] = _ALL_NOUNS_FLAT + VERBS + ADJECTIVES + ADVERBS


def pick_words(n: int = 2) -> list[str]:
    """
    Pick *n* diverse vocabulary words for prompt seeding.

    Always includes at least one noun and one verb/adjective/adverb
    for natural sentence variety.
    """
    import random  # noqa: PLC0415

    result: list[str] = [random.choice(_ALL_NOUNS_FLAT)]
    other_pool = VERBS + ADJECTIVES + ADVERBS
    for _ in range(n - 1):
        result.append(random.choice(other_pool))
    random.shuffle(result)
    return result
