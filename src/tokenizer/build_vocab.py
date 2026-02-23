"""
build_vocab.py — Створює vocab.json зі словником ~2000 токенів для A2 German Grammar Tutor.

Запуск:
    python build_vocab.py          # → створить vocab.json
    python build_vocab.py --stats  # → покаже статистику по категоріях

Словник організований за категоріями:
    1. Спеціальні токени (<PAD>, <BOS>, <EOS>, <UNK>)
    2. Пунктуація та маркери
    3. Артиклі та вказівні слова
    4. Займенники (особові, присвійні, зворотні)
    5. Прийменники
    6. Сполучники та прислівники
    7. Модальні дієслова з усіма формами
    8. Основні дієслова з формами (Präsens, Perfekt, Partizip II)
    9. Відокремлювані дієслова (Separable verbs)
    10. Іменники (A2 теми: сім'я, їжа, побут, місто, робота…)
    11. Прикметники
    12. Числівники
    13. Службові/частотні слова
    14. Слова для відповідей тьютора (Correct, Incorrect, Explanation…)
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import load_config
from src.tokenizer.tokenizer import Tokenizer

def build_vocab() -> dict[str, int]:
    """Збирає все в один великий словник: token → id."""
    tokens: list[str] = []

    # ─── 1. Спеціальні токени (4) ─────────────────────────
    # PAD — заповнення коротких речень до однакової довжини
    # BOS — "beginning of sequence" — початок тексту
    # EOS — "end of sequence" — кінець тексту
    # UNK — "unknown" — невідоме слово
    special = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    tokens.extend(special)

    # ─── 1b. Токени з train/val (пріоритет — щоб не обрізати при trim до TARGET_V)
    # Усі слова з даних мають потрапити в словник, інакше модель виводить <UNK>.
    data_words = _extract_tokens_from_data()
    tokens.extend(data_words)

    # ─── 2. Пунктуація та маркери (25) ────────────────────
    punctuation = [
        ".", ",", "!", "?", ":", ";", "-", "(", ")", '"', "'",
        "...",
        # Маркери відповіді тьютора
        "✅", "❌", "📝",
        # Нові рядки і пробіли (як окремі токени)
        "\n",
        # Emoji-like маркери, які модель бачитиме у відповідях
        "Correct", "Incorrect", "Correct:", "Incorrect.",
        "Explanation", "Explanation:",
        # Cпеціальні слова для формату відповіді
        "correct:", "incorrect.",
    ]
    tokens.extend(punctuation)

    # ─── 3. Артиклі та вказівні/неозначені (30) ──────────
    articles = [
        # Означені артиклі (Nominativ, Akkusativ, Dativ)
        "der", "die", "das", "den", "dem", "des",
        # Неозначені артиклі
        "ein", "eine", "einen", "einem", "einer",
        # Заперечні артиклі
        "kein", "keine", "keinen", "keinem", "keiner", "keines",
        # Вказівні
        "dieser", "diese", "dieses", "diesen", "diesem",
        # jeder
        "jeder", "jede", "jedes", "jeden", "jedem",
    ]
    tokens.extend(articles)

    # ─── 4. Займенники (55) ──────────────────────────────
    pronouns = [
        # Особові (Nominativ)
        "ich", "du", "er", "sie", "es", "wir", "ihr",
        # Особові (Akkusativ)
        "mich", "dich", "ihn", "uns", "euch",
        # Особові (Dativ)
        "mir", "dir", "ihm", "ihr",  # "ihr" вже є, додамо "ihnen"
        "ihnen",
        # Присвійні (Nominativ — основні форми)
        "mein", "meine", "meinen", "meinem", "meiner",
        "dein", "deine", "deinen", "deinem", "deiner",
        "sein", "seine", "seinen", "seinem", "seiner",
        "ihre", "ihren", "ihrem", "ihrer",  # ihr/ihre
        "unser", "unsere", "unseren", "unserem",
        "euer", "eure", "euren", "eurem",
        # Зворотні
        "sich",
        # Відносні / питальні
        "wer", "was", "wen", "wem",
        "man",
    ]
    tokens.extend(pronouns)

    # ─── 5. Прийменники (35) ─────────────────────────────
    prepositions = [
        # + Dativ
        "mit", "nach", "bei", "von", "zu", "aus", "seit", "gegenüber",
        # + Akkusativ
        "für", "ohne", "gegen", "durch", "um", "bis",
        # Wechselpräpositionen (+ Dat або + Akk)
        "in", "an", "auf", "über", "unter", "vor", "hinter",
        "neben", "zwischen",
        # Скорочені форми
        "im", "am", "zum", "zur", "ins", "ans", "vom",
        # Додаткові
        "ab", "außer",
    ]
    tokens.extend(prepositions)

    # ─── 6. Сполучники та прислівники (75) ────────────────
    conjunctions_adverbs = [
        # Координаційні сполучники
        "und", "oder", "aber", "denn", "sondern",
        # Підрядні сполучники (порядок слів!)
        "weil", "dass", "wenn", "ob", "als", "obwohl",
        "damit", "bevor", "nachdem", "während",
        # Прислівники часу
        "heute", "morgen", "gestern", "jetzt", "dann", "danach",
        "immer", "nie", "oft", "manchmal", "selten",
        "schon", "noch", "bald", "gerade", "früher", "später",
        "montags", "dienstags", "mittwochs", "donnerstags",
        "freitags", "samstags", "sonntags",
        # Прислівники місця
        "hier", "dort", "da", "oben", "unten", "links", "rechts",
        "draußen", "drinnen", "überall",
        # Прислівники способу
        "sehr", "gern", "gerne", "lieber", "viel", "wenig",
        "schnell", "langsam", "zusammen", "allein",
        "ungefähr", "etwa", "fast", "genug",
        # Питальні слова
        "wie", "wo", "wann", "warum", "wohin", "woher",
    ]
    tokens.extend(conjunctions_adverbs)

    # ─── 7. Модальні дієслова з усіма формами (36) ───────
    # Кожне модальне дієслово має 6 форм Präsens + Präteritum
    modal_verbs = [
        # können
        "können", "kann", "kannst", "könnt", "konnte", "konnten",
        # müssen
        "müssen", "muss", "musst", "müsst", "musste", "mussten",
        # dürfen
        "dürfen", "darf", "darfst", "dürft", "durfte", "durften",
        # wollen
        "wollen", "will", "willst", "wollt", "wollte", "wollten",
        # sollen
        "sollen", "soll", "sollst", "sollt", "sollte", "sollten",
        # mögen / möchten
        "mögen", "mag", "magst", "möchten", "möchte", "möchtest",
    ]
    tokens.extend(modal_verbs)

    # ─── 8. Основні дієслова з формами (350) ─────────────
    # Формат: інфінітив, ich, du, er/sie/es, wir/sie, Partizip II
    # Допоміжні дієслова
    core_verbs = [
        # sein (to be) — найважливіше!
        "bin", "bist", "ist", "sind", "seid",
        "war", "warst", "waren", "wart",
        "gewesen",
        # Інфінітив "sein" додамо окремо
        "sein",
        # haben (to have)
        "haben", "habe", "hast", "hat", "habt",
        "hatte", "hattest", "hatten", "hattet",
        "gehabt",
        # werden (to become / auxiliary)
        "werden", "werde", "wirst", "wird", "werdet",
        "wurde", "geworden",
    ]
    tokens.extend(core_verbs)

    # A2 дієслова (інфінітив + ключові форми + Partizip II)
    a2_verbs = [
        # Рух / переміщення (з sein!)
        "gehen", "gehe", "gehst", "geht", "gegangen",
        "kommen", "komme", "kommst", "kommt", "gekommen",
        "fahren", "fahre", "fährst", "fährt", "gefahren",
        "laufen", "laufe", "läufst", "läuft", "gelaufen",
        "fliegen", "fliege", "fliegst", "fliegt", "geflogen",
        "schwimmen", "schwimme", "schwimmst", "schwimmt", "geschwommen",
        "bleiben", "bleibe", "bleibst", "bleibt", "geblieben",
        "reisen", "reise", "reist",  "gereist",

        # Повсякденне життя
        "machen", "mache", "machst", "macht", "gemacht",
        "arbeiten", "arbeite", "arbeitest", "arbeitet", "gearbeitet",
        "lernen", "lerne", "lernst", "lernt", "gelernt",
        "spielen", "spiele", "spielst", "spielt", "gespielt",
        "kaufen", "kaufe", "kaufst", "kauft", "gekauft",
        "kochen", "koche", "kochst", "kocht", "gekocht",
        "essen", "esse", "isst", "gegessen",
        "trinken", "trinke", "trinkst", "trinkt", "getrunken",
        "schlafen", "schlafe", "schläfst", "schläft", "geschlafen",
        "wohnen", "wohne", "wohnst", "wohnt", "gewohnt",
        "leben", "lebe", "lebst", "lebt", "gelebt",

        # Комунікація
        "sprechen", "spreche", "sprichst", "spricht", "gesprochen",
        "sagen", "sage", "sagst", "sagt", "gesagt",
        "fragen", "frage", "fragst", "fragt", "gefragt",
        "antworten", "antworte", "antwortest", "antwortet", "geantwortet",
        "erzählen", "erzähle", "erzählst", "erzählt",
        "schreiben", "schreibe", "schreibst", "schreibt", "geschrieben",
        "lesen", "lese", "liest", "gelesen",
        "hören", "höre", "hörst", "hört", "gehört",
        "verstehen", "verstehe", "verstehst", "versteht", "verstanden",
        "helfen", "helfe", "hilfst", "hilft", "geholfen",
        "rufen", "rufe", "rufst", "ruft", "gerufen",

        # Інші часті дієслова
        "sehen", "sehe", "siehst", "sieht", "gesehen",
        "geben", "gebe", "gibst", "gibt", "gegeben",
        "nehmen", "nehme", "nimmst", "nimmt", "genommen",
        "finden", "finde", "findest", "findet", "gefunden",
        "wissen", "weiß", "weißt", "gewusst",
        "denken", "denke", "denkst", "denkt", "gedacht",
        "glauben", "glaube", "glaubst", "glaubt", "geglaubt",
        "brauchen", "brauche", "brauchst", "braucht", "gebraucht",
        "bringen", "bringe", "bringst", "bringt", "gebracht",
        "legen", "lege", "legst", "legt", "gelegt",
        "stellen", "stelle", "stellst", "stellt", "gestellt",
        "setzen", "setze", "setzt", "gesetzt",
        "liegen", "liege", "liegst", "liegt", "gelegen",
        "stehen", "stehe", "stehst", "steht", "gestanden",
        "sitzen", "sitze", "sitzt", "gesessen",
        "tragen", "trage", "trägst", "trägt", "getragen",
        "waschen", "wasche", "wäschst", "wäscht", "gewaschen",
        "putzen", "putze", "putzt", "geputzt",
        "öffnen", "öffne", "öffnest", "öffnet", "geöffnet",
        "schließen", "schließe", "schließt", "geschlossen",
        "beginnen", "beginne", "beginnst", "beginnt", "begonnen",
        "besuchen", "besuche", "besuchst", "besucht",
        "bezahlen", "bezahle", "bezahlst", "bezahlt",
        "vergessen", "vergesse", "vergisst", "vergessen",
        "bekommen", "bekomme", "bekommst", "bekommt", "bekommen",
        "treffen", "treffe", "triffst", "trifft", "getroffen",
        "kennen", "kenne", "kennst", "kennt", "gekannt",
        "mögen", "gefallen", "gefällt",
        "freuen", "freue", "freust", "freut", "gefreut",
        "hoffen", "hoffe", "hoffst", "hofft", "gehofft",
        "wünschen", "wünsche", "wünschst", "wünscht", "gewünscht",
        "dauern", "dauert",
        "kosten", "kostet",
        "gehören", "gehört",
        "passen", "passt",
        "fehlen", "fehlt",
        "stimmen", "stimmt",
        "ändern", "ändere", "änderst", "ändert", "geändert",
        "zeigen", "zeige", "zeigst", "zeigt", "gezeigt",
    ]
    tokens.extend(a2_verbs)

    # ─── 9. Відокремлювані дієслова (Separable verbs) (80) ─
    # Важливо для A2: позиція приставки змінюється!
    # "Ich stehe um 7 Uhr auf." vs "Ich muss aufstehen."
    separable_verbs = [
        # aufstehen
        "aufstehen", "aufgestanden",
        "auf",
        # einkaufen
        "einkaufen", "eingekauft",
        "ein",
        # anfangen
        "anfangen", "angefangen",
        "an",  # вже є як прийменник, не дублюємо
        "fangen", "fängt",
        # aufräumen
        "aufräumen", "aufgeräumt",
        "räume", "räumst", "räumt",
        # anrufen
        "anrufen", "angerufen",
        # mitkommen
        "mitkommen", "mitgekommen",
        # mitbringen
        "mitbringen", "mitgebracht",
        # ausgehen
        "ausgehen", "ausgegangen",
        # fernsehen
        "fernsehen", "ferngesehen",
        "fern",
        # zumachen / aufmachen
        "zumachen", "zugemacht",
        "aufmachen", "aufgemacht",
        # ankommen
        "ankommen", "angekommen",
        # abfahren
        "abfahren", "abgefahren",
        # umsteigen
        "umsteigen", "umgestiegen",
        "steige", "steigst", "steigt",
        # einladen
        "einladen", "eingeladen",
        "lade", "lädst", "lädt",
        # zurückkommen
        "zurückkommen", "zurückgekommen",
        "zurück",
        # vorstellen
        "vorstellen", "vorgestellt",
        "vor",
        # aufhören
        "aufhören", "aufgehört",
        # weitergehen
        "weitergehen",
        "weiter",
        # vorbereiten
        "vorbereiten", "vorbereitet",
        # teilnehmen
        "teilnehmen", "teilgenommen",
        "teil",
        # Загальні приставки (як окремі токени)
        "ab", "mit", "um", "zu",
    ]
    tokens.extend(separable_verbs)

    # ─── 10. Іменники по темах A2 (450) ──────────────────

    # Сім'я та люди
    nouns_family = [
        "Vater", "Mutter", "Eltern", "Kind", "Kinder",
        "Sohn", "Tochter", "Bruder", "Schwester",
        "Großvater", "Großmutter", "Opa", "Oma",
        "Onkel", "Tante", "Cousin", "Cousine",
        "Mann", "Frau", "Freund", "Freundin",
        "Nachbar", "Nachbarin", "Mensch", "Menschen",
        "Leute", "Person", "Junge", "Mädchen",
        "Baby", "Kollege", "Kollegin", "Chef", "Chefin",
    ]
    tokens.extend(nouns_family)

    # Їжа та напої
    nouns_food = [
        "Essen", "Brot", "Brötchen", "Butter", "Käse",
        "Wurst", "Fleisch", "Fisch", "Ei", "Eier",
        "Reis", "Nudeln", "Kartoffel", "Kartoffeln",
        "Suppe", "Salat", "Gemüse", "Obst",
        "Apfel", "Banane", "Orange", "Tomate", "Tomaten",
        "Kuchen", "Schokolade", "Eis",
        "Wasser", "Milch", "Kaffee", "Tee", "Saft",
        "Bier", "Wein", "Getränk",
        "Frühstück", "Mittagessen", "Abendessen",
        "Mahlzeit", "Zucker", "Salz",
    ]
    tokens.extend(nouns_food)

    # Дім та побут
    nouns_home = [
        "Haus", "Hause", "Wohnung", "Zimmer", "Küche",
        "Bad", "Badezimmer", "Schlafzimmer", "Wohnzimmer",
        "Garten", "Balkon", "Tür", "Fenster", "Treppe",
        "Möbel", "Tisch", "Stuhl", "Bett", "Schrank",
        "Sofa", "Lampe", "Spiegel", "Regal",
        "Fernseher", "Computer", "Handy", "Telefon",
    ]
    tokens.extend(nouns_home)

    # Місто та транспорт
    nouns_city = [
        "Stadt", "Straße", "Platz", "Park", "Bahnhof",
        "Haltestelle", "Flughafen", "Hotel",
        "Restaurant", "Café", "Laden", "Geschäft",
        "Supermarkt", "Markt", "Apotheke", "Bank",
        "Post", "Kirche", "Museum", "Kino", "Theater",
        "Bibliothek", "Krankenhaus", "Arzt", "Ärztin",
        "Polizei", "Schule", "Universität",
        "Bus", "Zug", "U-Bahn", "Straßenbahn", "Fahrrad",
        "Auto", "Taxi", "Flugzeug", "Schiff",
        "Ticket", "Fahrkarte", "Weg", "Brücke",
    ]
    tokens.extend(nouns_city)

    # Робота та навчання
    nouns_work = [
        "Arbeit", "Beruf", "Job", "Büro", "Firma",
        "Lehrer", "Lehrerin", "Schüler", "Schülerin",
        "Student", "Studentin", "Kurs", "Unterricht",
        "Prüfung", "Aufgabe", "Hausaufgabe", "Übung",
        "Buch", "Heft", "Stift", "Tafel",
        "Sprache", "Deutsch", "Englisch", "Wort", "Satz",
        "Text", "Brief", "E-Mail", "Nachricht",
        "Zeitung", "Zeitschrift", "Seite",
    ]
    tokens.extend(nouns_work)

    # Час та дати
    nouns_time = [
        "Zeit", "Uhr", "Stunde", "Minute", "Sekunde",
        "Tag", "Tage", "Woche", "Monat", "Jahr", "Jahre",
        "Morgen", "Mittag", "Abend", "Nacht",
        "Montag", "Dienstag", "Mittwoch", "Donnerstag",
        "Freitag", "Samstag", "Sonntag", "Wochenende",
        "Januar", "Februar", "März", "April", "Mai", "Juni",
        "Juli", "August", "September", "Oktober", "November", "Dezember",
        "Frühling", "Sommer", "Herbst", "Winter",
        "Geburtstag", "Feiertag", "Urlaub", "Ferien",
        "Termin", "Datum",
    ]
    tokens.extend(nouns_time)

    # Тіло та здоров'я
    nouns_health = [
        "Kopf", "Auge", "Augen", "Ohr", "Ohren",
        "Nase", "Mund", "Zahn", "Zähne",
        "Hand", "Hände", "Arm", "Bein", "Fuß", "Füße",
        "Rücken", "Bauch", "Herz",
        "Gesundheit", "Krankheit", "Schmerzen",
        "Medikament", "Rezept", "Fieber",
    ]
    tokens.extend(nouns_health)

    # Одяг
    nouns_clothes = [
        "Kleidung", "Hemd", "Hose", "Rock", "Kleid",
        "Jacke", "Mantel", "Pullover", "T-Shirt",
        "Schuh", "Schuhe", "Socke", "Socken",
        "Mütze", "Tasche", "Koffer",
    ]
    tokens.extend(nouns_clothes)

    # Природа та погода
    nouns_nature = [
        "Wetter", "Sonne", "Regen", "Schnee", "Wind",
        "Wolke", "Himmel", "Temperatur", "Grad",
        "Berg", "See", "Meer", "Fluss", "Wald",
        "Baum", "Blume", "Tier", "Hund", "Katze", "Vogel",
    ]
    tokens.extend(nouns_nature)

    # Абстрактні та різні
    nouns_abstract = [
        "Name", "Adresse", "Nummer", "Alter",
        "Problem", "Frage", "Antwort", "Idee",
        "Hilfe", "Beispiel", "Grund", "Meinung",
        "Erfahrung", "Möglichkeit", "Unterschied",
        "Anfang", "Ende", "Ziel", "Plan",
        "Geld", "Preis", "Euro", "Cent",
        "Spaß", "Freude", "Angst", "Glück",
        "Musik", "Sport", "Spiel", "Film",
        "Foto", "Bild", "Farbe", "Größe", "Form",
        "Richtung", "Seite", "Teil", "Stück",
        "Land", "Länder", "Deutschland", "Österreich", "Schweiz",
        "Information", "Programm", "Gruppe", "Klasse",
        "Familie", "Hochzeit", "Party", "Fest",
    ]
    tokens.extend(nouns_abstract)

    # ─── 11. Прикметники (160) ───────────────────────────
    adjectives = [
        # Основні
        "gut", "schlecht", "schön", "hässlich",
        "groß", "klein", "lang", "kurz",
        "alt", "neu", "jung",
        "hoch", "tief", "breit", "schmal",
        "schwer", "leicht",
        "schnell", "langsam",  # вже є як прислівники, але ОК для adj
        "warm", "kalt", "heiß", "kühl",
        "hell", "dunkel",
        "laut", "leise",
        "nah", "weit", "nächst",
        "richtig", "falsch",
        "wichtig", "möglich", "nötig",
        "einfach", "schwierig", "kompliziert",
        "billig", "teuer", "günstig",
        "frei", "fertig", "bereit",
        "offen", "geschlossen",
        "voll", "leer",
        "sauber", "schmutzig",
        "gesund", "krank", "müde",
        "hungrig", "durstig", "satt",
        "glücklich", "traurig", "zufrieden",
        "freundlich", "nett", "lustig",
        "interessant", "langweilig",
        "bekannt", "berühmt",
        "typisch", "normal", "besonder",
        "verschieden", "gleich", "ähnlich",
        "letzt", "erst", "zweit", "dritt",
        "ganz", "halb",
        "andere", "anderer", "anderes",
        # Компаративи та суперлативи (часті)
        "besser", "beste", "besten",
        "größer", "größte", "größten",
        "mehr", "meisten",
        "lieber", "liebsten",
        "höher", "höchste",
        "länger", "längste",
        "älter", "älteste",
        "schöner", "schönste", "schönsten",
        "kleiner", "kleinste",
        # als, am — для порівнянь
        "als",
        # Partizip як прикметник
        "interessiert", "verheiratet", "geschieden",
    ]
    tokens.extend(adjectives)

    # ─── 12. Числівники (35) ─────────────────────────────
    numerals = [
        "null", "eins", "zwei", "drei", "vier", "fünf",
        "sechs", "sieben", "acht", "neun", "zehn",
        "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn",
        "zwanzig", "dreißig", "vierzig", "fünfzig",
        "sechzig", "siebzig", "achtzig", "neunzig",
        "hundert", "tausend", "Million",
        "erste", "zweite", "dritte", "vierte", "fünfte",
        # Цифри як токени (для годин, дат)
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "11", "12", "15", "20", "30",
    ]
    tokens.extend(numerals)

    # ─── 13. Службові слова та частки (90) ────────────────
    function_words = [
        # Заперечення
        "nicht", "nichts", "niemand", "niemals",
        # Невизначені
        "etwas", "alles", "alle", "viele", "einige", "manche",
        "jemand", "niemand",
        "beide", "mehrere",
        # Артикельні + кількісні
        "mehr", "weniger", "genug",
        "jeder", "jede", "jedes",
        # Модальні частки
        "doch", "mal", "ja", "nein", "denn",
        "eben", "halt", "wohl", "etwa", "eigentlich",
        # Прислівникові
        "auch", "nur", "schon", "noch", "wieder",
        "zuerst", "zunächst", "endlich", "plötzlich",
        "natürlich", "leider", "hoffentlich",
        "vielleicht", "bestimmt", "sicher", "wahrscheinlich",
        "besonders", "wirklich", "ziemlich",
        "trotzdem", "deshalb", "deswegen", "darum",
        "außerdem", "übrigens",
        # Прийменникові сполуки
        "dafür", "dagegen", "dazu", "davon", "damit",
        "darüber", "darunter",
        # Допоміжні
        "es", "so", "zu", "am",
        "hin", "her",
        # Напрямки
        "nach", "Hause",  # nach Hause — дуже часте
    ]
    tokens.extend(function_words)

    # ─── 14. Слова для пояснень тьютора (50) ──────────────
    tutor_words_ua = [
        "\n",
        # Українські слова для пояснень
        "Пояснення", "Пояснення:",
        "Правильно", "Неправильно",
        "Речення", "правильне", "неправильне",
        "Помилка", "помилка",
        "Тут", "тут", "треба", "потрібно",
        "використовувати", "вживати", "вживається", "використовуємо",
        "замість", "бо", "від", "не", "в", "а", "з",
        "дієслово", "артикль", "іменник", "прийменник", "Дієслово",
        "форма", "форму", "формі", "означає", "рух", "тому",
        "Dativ", "Akkusativ", "Nominativ",
        "Perfekt", "Präsens", "Präteritum",
        "Partizip", "II",
        "допоміжне", "основне", "допоміжного",
        "узгоджуватися",
        "після", "перед", "наприкінці",
        "минулому", "теперішньому", "часі",
        "порядок", "слів", "позиція",
        "відокремлювана", "приставка",
        "mit", "sein", "haben",  # дублі OK — їх відфільтруємо
        "вимагає", "керує", "потребує",
        "У", "реченні", "Коли", "починається", "має", "стояти", "на", "другому", "місці", "перед", "підметом", "підмета",
        "інфінітиві", "бути", "для", "закінчення", "правильно",
        # Закінчення дієслів
        "e", "st", "t", "en",
        "Correct:", "Incorrect:",
    ]
    tokens.extend(tutor_words_ua)

    # ─── 15. Додаткові часті A2 слова ─────────────────────
    extra_words = [
        # Слова, що часто зустрічаються в A2 текстах
        "bitte", "danke", "Entschuldigung", "tut",
        "Herr", "Frau", "Doktor",
        "Europa", "Welt",
        "Hobby", "Reise", "Ausflug", "Wanderung",
        "Ergebnis", "Erfolg",
        "Regel", "Regeln",
        "Recht", "Pflicht",
        "Kultur", "Tradition",
        "Umwelt",
        "Zukunft", "Vergangenheit",
        "Gesicht", "Stimme",
        "Geschenk", "Überraschung",
        "Haushalt", "Miete", "Strom",
        "Bewerbung", "Lebenslauf",
        "Pass", "Ausweis", "Visum",
        "Vertrag", "Formular",
        "Ordnung", "Sicherheit",
        "Verkehr", "Unfall",
        "Erklärung",
        "hätte", "wäre", "könnte",
        "würde", "würden",
    ]
    tokens.extend(extra_words)

    # ─── 16. Додаткові дієслівні форми (wir/ihr/sie) (120) ─
    # Багато дієслів вище мали лише ich/du/er форми.
    # Додаємо wir/sie/ihr форми + Präteritum для частих дієслів.
    extra_verb_forms = [
        # gehen Prät.
        "ging", "gingen", "gingst",
        # kommen Prät.
        "kam", "kamen", "kamst",
        # fahren Prät.
        "fuhr", "fuhren",
        # sprechen Prät.
        "sprach", "sprachen",
        # sehen Prät.
        "sah", "sahen",
        # geben Prät.
        "gab", "gaben",
        # nehmen Prät.
        "nahm", "nahmen",
        # schreiben Prät.
        "schrieb", "schrieben",
        # lesen Prät.
        "las", "lasen",
        # finden Prät.
        "fand", "fanden",
        # bringen Prät.
        "brachte", "brachten",
        # denken Prät.
        "dachte", "dachten",
        # essen Prät.
        "aß", "aßen",
        # trinken Prät.
        "trank", "tranken",
        # schlafen Prät.
        "schlief", "schliefen",
        # treffen Prät.
        "traf", "trafen",
        # bleiben Prät.
        "blieb", "blieben",
        # rufen Prät.
        "rief", "riefen",
        # stehen Prät.
        "stand", "standen",
        # sitzen Prät.
        "saß", "saßen",
        # liegen Prät.
        "lag", "lagen",
        # laufen Prät.
        "lief", "liefen",
        # fliegen Prät.
        "flog", "flogen",
        # schwimmen Prät.
        "schwamm", "schwammen",
        # tragen Prät.
        "trug", "trugen",
        # helfen Prät.
        "half", "halfen",
        # beginnen Prät.
        "begann", "begannen",
        # vergessen Prät.
        "vergaß", "vergaßen",
        # fallen
        "fallen", "falle", "fällst", "fällt", "gefallen",
        "fiel", "fielen",
        # wachsen
        "wachsen", "wächst", "gewachsen",
        # rennen
        "rennen", "renne", "rennst", "rennt", "gerannt",
        # sterben
        "sterben", "stirbt", "gestorben",
        # passieren
        "passieren", "passiert",
        # erklären
        "erklären", "erkläre", "erklärst", "erklärt",
        # versuchen
        "versuchen", "versuche", "versuchst", "versucht",
        # entscheiden
        "entscheiden", "entscheide", "entscheidet", "entschieden",
        # erlauben
        "erlauben", "erlaube", "erlaubt",
        # empfehlen
        "empfehlen", "empfehle", "empfiehlt", "empfohlen",
        # bestellen
        "bestellen", "bestelle", "bestellst", "bestellt",
        # übersetzen
        "übersetzen", "übersetze", "übersetzt",
        # wiederholen
        "wiederholen", "wiederhole", "wiederholt",
        # reparieren
        "reparieren", "repariere", "repariert",
        # studieren
        "studieren", "studiere", "studierst", "studiert",
        # telefonieren
        "telefonieren", "telefoniere", "telefoniert",
        # funktionieren
        "funktionieren", "funktioniert",
        # reservieren
        "reservieren", "reserviert",
        # informieren
        "informieren", "informiert",
        # interessieren
        "interessieren", "interessiere", "interessiert",
        # probieren
        "probieren", "probiere", "probiert",
    ]
    tokens.extend(extra_verb_forms)

    # ─── 17. Множини іменників та додаткові іменники (200) ─
    extra_nouns = [
        # Множини (Plural) — часті
        "Väter", "Mütter", "Söhne", "Töchter", "Brüder", "Schwestern",
        "Männer", "Frauen", "Freunde", "Freundinnen",
        "Häuser", "Wohnungen", "Zimmer", "Tische", "Stühle",
        "Bücher", "Briefe", "Sätze", "Wörter", "Texte",
        "Autos", "Busse", "Züge", "Fahrräder",
        "Städte", "Straßen", "Plätze", "Parks",
        "Bilder", "Filme", "Spiele", "Lieder",
        "Tiere", "Hunde", "Katzen", "Vögel",
        "Blumen", "Bäume",
        "Probleme", "Fragen", "Antworten", "Ideen",
        "Äpfel", "Bananen", "Orangen",
        "Hemden", "Hosen", "Kleider", "Jacken", "Mäntel",
        "Geschäfte", "Restaurants", "Hotels",
        "Kurse", "Prüfungen", "Aufgaben", "Übungen",
        "Termine", "Pläne", "Ziele",
        "Preise", "Regeln",
        # Додаткові іменники (A2 теми)
        "Ankunft", "Abfahrt", "Abflug", "Anschluss",
        "Eingang", "Ausgang", "Notausgang",
        "Erdgeschoss", "Stock", "Etage",
        "Schlüssel", "Rechnung", "Quittung",
        "Nachbar", "Vermieter", "Mieterin",
        "Rathaus", "Amt", "Behörde",
        "Kindergarten", "Spielplatz",
        "Einladung", "Geburtstag",
        "Abendessen", "Mittagessen",
        "Gabel", "Messer", "Löffel", "Teller", "Glas", "Tasse",
        "Dose", "Flasche", "Packung", "Stück",
        "Seife", "Handtuch", "Zahnbürste",
        "Bettwäsche", "Decke", "Kissen",
        "Drucker", "Tastatur", "Bildschirm",
        "Waschmaschine", "Kühlschrank", "Herd", "Ofen",
        "Papier", "Schere", "Kleber",
        "Landkarte", "Stadtplan", "Fahrplan",
        "Fahrkarte", "Monatskarte",
        "Zeugnis", "Diplom", "Zertifikat",
        "Gehalt", "Lohn", "Rente",
        "Steuer", "Versicherung",
        "Woche", "Wochen",
        "Urlaub", "Reise",
        "Geburtstag", "Hochzeit",
        "Ruhe", "Lärm", "Stress",
        "Praktikum", "Ausbildung",
    ]
    tokens.extend(extra_nouns)

    # ─── 18. Declined adjective endings + more adj forms (80) ─
    extra_adjectives = [
        # Часті форми з відмінковими закінченнями
        "guter", "gutes", "guten", "gutem", "gute",
        "neuer", "neues", "neuen", "neuem", "neue",
        "alter", "altes", "alten", "altem", "alte",
        "großer", "großes", "großen", "großem", "große",
        "kleiner", "kleines", "kleinen", "kleinem", "kleine",
        "schöner", "schönes", "schönen", "schönem",
        "langer", "langes", "langen", "langem", "lange",
        # Ще прикметники
        "praktisch", "gemütlich", "bequem",
        "pünktlich", "ordentlich", "höflich",
        "gefährlich", "ungefährlich",
        "möglich", "unmöglich",
        "notwendig", "dringend",
        "angenehm", "unangenehm",
        "zufrieden", "unzufrieden",
        "bekannt", "unbekannt",
        "verheiratet", "ledig",
        "arbeitslos", "berufstätig",
        "spannend", "aufregend",
        "ruhig", "nervös",
        "stolz", "böse",
        "froh", "frisch",
        "trocken", "nass", "feucht",
        "dick", "dünn",
        "eng", "locker",
        "weich", "hart",
        "süß", "sauer", "bitter", "scharf",
        "lecker",
        "kostenlos", "gratis",
    ]
    tokens.extend(extra_adjectives)

    # ─── 19. Більше українських слів для тьютора (100) ────
    extra_tutor_ua = [
        # Пояснення граматики українською
        "речення", "слово", "слова",
        "означений", "неозначений",
        "однина", "множина",
        "чоловічий", "жіночий", "середній", "рід",
        "відмінок", "називний", "знахідний", "давальний",
        "підмет", "присудок", "додаток",
        "головне", "підрядне",
        "сполучник", "частка",
        "правильний", "неправильний",
        "минулий", "теперішній", "час",
        "допоміжний",
        "дієслова", "руху", "стану",
        "закінчення", "основа", "корінь",
        "префікс", "суфікс",
        "наголос", "вимова",
        "значення", "переклад",
        "приклад", "правило",
        "виняток", "винятки",
        "заперечення", "ствердження",
        "запитання", "відповідь",
        "порівняння", "ступінь",
        "вищий", "найвищий",
        "прямий", "непрямий",
        "ввічлива", "неформальна",
        "формальна", "звертання",
        "послідовність", "структура",
        "стоїть", "стояти",
        "наприкінці", "на", "початку",
        "другу", "другій", "третю",
        "другому", "місці",
        "інфінітив", "дієвідміна",
        "відмінюється", "змінюється",
        "модальне", "смислове",
        "знаходиться", "переміщується",
        "вказує", "означає",
        "потребує", "вимагає",
        "правильна", "неправильна",
        "Відповідь", "Виправлення",
        # Шаблони відповідей
        "вживати", "з",
        "у", "цьому", "випадку",
        "тому", "що",
        "має", "бути",
        "стояти", "кінці",
    ]
    tokens.extend(extra_tutor_ua)

    # ─── 20. Складні / композитні слова та решта (150) ────
    compound_and_misc = [
        # Складні слова, часті в A2
        "Hauptbahnhof", "Einkaufszentrum",
        "Arbeitgeber", "Arbeitnehmer",
        "Muttersprache", "Fremdsprache",
        "Sprachkurs", "Deutschkurs",
        "Arbeitsplatz", "Parkplatz",
        "Kinderzimmer", "Esszimmer", "Arbeitszimmer",
        "Einfamilienhaus", "Mehrfamilienhaus",
        "Briefkasten", "Mülleimer",
        "Geburtsort", "Geburtsdatum",
        "Familienstand", "Staatsangehörigkeit",
        "Aufenthaltserlaubnis",
        # Колір
        "rot", "blau", "grün", "gelb", "schwarz", "weiß",
        "braun", "grau", "rosa", "orange", "lila",
        # Напрямки / сторони
        "Norden", "Süden", "Westen", "Osten",
        "nördlich", "südlich", "westlich", "östlich",
        # Матеріали та речовини
        "Holz", "Metall", "Glas", "Plastik", "Stoff",
        # Прийоми їжі / кухня
        "backen", "braten", "grillen", "schneiden",
        "gebacken", "gebraten", "gegrillt", "geschnitten",
        # Ще дієслова (часті)
        "aufpassen", "aufgepasst",
        "nachdenken", "nachgedacht",
        "umziehen", "umgezogen",
        "spazieren", "spaziert",
        "wandern", "gewandert",
        "tanzen", "getanzt",
        "singen", "gesungen",
        "lächeln", "gelächelt",
        "weinen", "geweint",
        "lachen", "gelacht",
        "träumen", "geträumt",
        "fühlen", "gefühlt",
        "merken", "gemerkt",
        "bemerken", "bemerkt",
        "vermissen", "vermisst",
        "stören", "gestört",
        "nutzen", "genutzt",
        "sammeln", "gesammelt",
        "teilen", "geteilt",
        "warten", "gewartet",
        "suchen", "gesucht",
        "verlieren", "verloren",
        "gewinnen", "gewonnen",
        "ziehen", "gezogen",
        "drücken", "gedrückt",
        "schieben", "geschoben",
        "hängen", "gehängt",
        "schenken", "geschenkt",
        "schicken", "geschickt",
        "liefern", "geliefert",
        "buchen", "gebucht",
        "anmelden", "angemeldet",
        "abmelden", "abgemeldet",
        "unterschreiben", "unterschrieben",
        "ausfüllen", "ausgefüllt",
        # Ще прислівники / частки
        "normalerweise", "meistens", "wenigstens",
        "mindestens", "höchstens",
        "irgendwo", "irgendwann", "irgendwie",
        "nirgendwo", "nirgends",
        "sofort", "gleich",
        "inzwischen", "zwischendurch",
        "hinterher", "vorher",
        "obendrein", "insgesamt",
        "allerdings", "jedoch",
        "nämlich", "beispielsweise",
        "ungefähr",
        # Ще прийменникові фрази
        "ums", "fürs", "aufs", "übers",
        # Грам. терміни (для тьютора, нім. мовою)
        "Verb", "Nomen", "Adjektiv", "Adverb",
        "Artikel", "Pronomen", "Präposition",
        "Konjunktion", "Subjekt", "Objekt",
        "Singular", "Plural",
        "maskulin", "feminin", "neutral",
        "Endung", "Stamm", "Vorsilbe",
        "Nebensatz", "Hauptsatz",
        "Infinitiv", "Imperativ",
        "Genitiv",
    ]
    tokens.extend(compound_and_misc)

    # ─── 21. Географія та країни (для прикладів) (50) ─────
    geography = [
        # Міста Німеччини
        "Berlin", "München", "Hamburg", "Köln", "Frankfurt",
        "Stuttgart", "Düsseldorf", "Dresden", "Leipzig", "Hannover",
        "Bremen", "Nürnberg", "Bonn", "Heidelberg", "Freiburg",
        # Інші країни/міста
        "Wien", "Zürich", "Bern", "Salzburg",
        "Paris", "London", "Rom", "Madrid", "Moskau",
        "Türkei", "Spanien", "Italien", "Frankreich", "Polen",
        "Russland", "Ukraine", "Griechenland", "Kroatien",
        "Asien", "Afrika", "Amerika",
        # Географічні поняття
        "Rhein", "Donau", "Alpen", "Nordsee", "Ostsee",
        "Insel", "Inseln", "Gebirge", "Küste",
        "Stadtrand", "Stadtzentrum", "Altstadt",
    ]
    tokens.extend(geography)

    # ─── 22. Часті імена для вправ (40) ───────────────────
    names = [
        "Anna", "Maria", "Peter", "Thomas", "Michael",
        "Hans", "Klaus", "Karl", "Stefan", "Martin",
        "Julia", "Sabine", "Monika", "Petra", "Andrea",
        "Ali", "Fatima", "Mohammed", "Olga", "Sergei",
        "Max", "Felix", "Laura", "Sophie", "Emma",
        "Leon", "Lena", "Tim", "Lisa", "David",
        "Müller", "Schmidt", "Fischer", "Weber", "Meyer",
        "Herr", "Frau",  # already exist but dedup handles it
    ]
    tokens.extend(names)

    # ─── 23. Рефлексивні дієслова (30) ────────────────────
    reflexive_verbs = [
        "waschen",  # sich waschen
        "anziehen", "angezogen", "ausziehen", "ausgezogen",
        "hinsetzen", "hingesetzt",
        "hinlegen", "hingelegt",
        "beeilen", "beeilt",
        "beschweren", "beschwert",
        "erinnern", "erinnert",
        "entschuldigen", "entschuldigt",
        "unterhalten", "unterhalten",
        "verabreden", "verabredet",
        "verspäten", "verspätet",
        "konzentrieren", "konzentriert",
        "gewöhnen", "gewöhnt",
        "befinden", "befindet",
        "beschäftigen", "beschäftigt",
    ]
    tokens.extend(reflexive_verbs)

    # ─── 24. Ще дієслова, яких не вистачає (60) ──────────
    more_verbs = [
        "wechseln", "gewechselt",
        "verdienen", "verdient",
        "ausgeben", "ausgegeben",
        "sparen", "gespart",
        "leihen", "geliehen",
        "berichten", "berichtet",
        "beschreiben", "beschrieben",
        "diskutieren", "diskutiert",
        "planen", "geplant",
        "organisieren", "organisiert",
        "kontrollieren", "kontrolliert",
        "korrigieren", "korrigiert",
        "überprüfen", "überprüft",
        "vergleichen", "verglichen",
        "verbessern", "verbessert",
        "verschenken", "verschenkt",
        "versprechen", "versprochen",
        "vorhaben", "vorgehabt",
        "zuhören", "zugehört",
        "aufwachen", "aufgewacht",
        "einschlafen", "eingeschlafen",
        "duschen", "geduscht",
        "frühstücken", "gefrühstückt",
        "unterrichten", "unterrichtet",
        "übersetzen", # already in vocab
        "ausdrücken", "ausgedrückt",
        "heiraten", "geheiratet",
        "trennen", "getrennt",
        "streiten", "gestritten",
    ]
    tokens.extend(more_verbs)

    # ─── 25. Ще іменники та множини (100) ─────────────────
    more_nouns = [
        # Освіта
        "Abitur", "Studium", "Semester", "Vorlesung",
        "Professor", "Professorin", "Dozent", "Dozentin",
        "Note", "Noten", "Fehler",
        # Побут
        "Müll", "Mülleimer", "Staubsauger", "Bügeleisen",
        "Geschirr", "Geschirrspüler", "Spülmaschine",
        "Dusche", "Toilette", "Waschbecken",
        # Фінанси
        "Konto", "Sparkonto", "Überweisung", "Kredit",
        "Schulden", "Bargeld", "Kreditkarte", "Bankkarte",
        # Медіа
        "Radio", "Sendung", "Nachrichten", "Werbung",
        "Kanal", "Serie", "Folge",
        # Подорожі
        "Gepäck", "Handgepäck", "Koffer",  # Koffer already exists
        "Reservierung", "Unterkunft", "Pension",
        "Jugendherberge", "Campingplatz",
        "Sehenswürdigkeit", "Rundfahrt", "Stadtführung",
        # Адміністративні
        "Anmeldung", "Abmeldung", "Bescheinigung",
        "Führerschein", "Personalausweis",
        "Meldebescheinigung", "Antrag",
        # Ще
        "Projekt", "Ergebnis",  # Ergebnis already exists
        "Beitrag", "Bericht", "Vortrag",
        "Eingang", "Ausgang",  # may exist
        "Empfang", "Rezeption",
        "Trinkgeld", "Bedienung",
        "Speisekarte", "Vorspeise", "Hauptgericht", "Nachspeise",
        "Getränkekarte", "Rechnung",
        "Vegetarier", "Allergiker",
        "Portion", "Scheibe", "Dose",  # Dose may exist
        "Kanne", "Becher",
        "Spieler", "Mannschaft", "Verein",
        "Stadion", "Halle", "Schwimmbad",
        "Training", "Wettkampf", "Meisterschaft",
        "Nachricht", "Anruf", "Anrufbeantworter",
        "Absender", "Empfänger", "Betreff",
        "Anhang", "Datei", "Dokument",
        "Drucker",  # may exist
        "Bildung", "Wissen", "Kenntnis",
    ]
    tokens.extend(more_nouns)

    # ─── 26. Ще прикметники та прислівники (50) ───────────
    more_adj_adv = [
        "direkt", "indirekt",
        "automatisch", "elektrisch",
        "international", "national",
        "privat", "öffentlich",
        "modern", "klassisch",
        "original", "aktuell",
        "regelmäßig", "unregelmäßig",
        "persönlich", "beruflich",
        "schriftlich", "mündlich",
        "täglich", "wöchentlich", "monatlich", "jährlich",
        "doppelt", "einzeln",
        "gemeinsam", "getrennt",
        "selbstständig", "abhängig",
        "anwesend", "abwesend",
        "einverstanden",
        "neugierig", "ängstlich", "wütend",
        "ernst", "komisch",
        "verrückt", "verantwortlich",
        "tatsächlich", "offensichtlich",
        "hauptsächlich", "grundsätzlich",
        "ausgezeichnet", "hervorragend",
        "anständig", "vernünftig",
        "dankbar", "ehrlich",
        "geduldig", "ungeduldig",
        "aufmerksam",
        "irgendein", "irgendeine",
    ]
    tokens.extend(more_adj_adv)

    # ─── 27. Слова з підручника Begegnungen A2 (PDF) ─────
    # Автоматично додаємо реальні A2 слова, яких ще немає в словнику
    pdf_words = _extract_pdf_words()
    tokens.extend(pdf_words)

    # (Токени з даних вже додано на початку — 1b — щоб не обрізати при trim.)

    # ─── Дедуплікація та побудова словника ────────────────
    seen: set[str] = set()
    unique_tokens: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)

    # Target size: V from config
    config = load_config()
    TARGET_V = config.model.vocab_size
    
    while len(unique_tokens) < TARGET_V:
        reserved = f"<RESERVED_{len(unique_tokens)}>"
        unique_tokens.append(reserved)

    # If more — trim (keep first TARGET_V)
    if len(unique_tokens) > TARGET_V:
        unique_tokens = unique_tokens[:TARGET_V]

    vocab = {token: idx for idx, token in enumerate(unique_tokens)}
    return vocab


def _extract_tokens_from_data() -> list[str]:
    """Витягує всі унікальні токени з файлів згенерованих даних."""
    config = load_config()
    data_paths = [config.data.train_path, config.data.val_path]
    
    unique_tokens = set()
    # Тимчасовий токенізатор (без завантаження vobab.json, нам потрібен лише метод _tokenize)
    # Ми не можемо створити екземпляр Tokenizer, бо він потребує vocab.json
    # Використовуємо логіку прямо з Tokenizer
    import re
    pattern = (
        r"Correct:|Incorrect\.|Explanation:|Пояснення:"
        r"|\.\.\."
        r"|[✅❌📝]"
        r"|\n"
        r"|[A-Za-zÄäÖöÜüß\u0400-\u04FF]+"
        r"|[.,!?;:\"'\-()]"
    )

    for path_str in data_paths:
        p = Path(path_str)
        if not p.exists():
            continue
        
        print(f"  🔍 Extracting tokens from {p.name}...")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Токенізуємо input та output
                    for text in [entry.get("input", ""), entry.get("output", "")]:
                        tokens = re.findall(pattern, text)
                        for t in tokens:
                            if t:
                                unique_tokens.add(t)
                except:
                    continue
    
    return list(unique_tokens)


def _extract_pdf_words() -> list[str]:
    """Витягує реальні A2 слова з підручника Begegnungen A2 (PDF).

    Фільтрує:
    - URL-и та коди (wordwall, learningapps, https, www…)
    - Занадто короткі слова (< 3 символи)
    - Граматичні скорочення (Akk, Dat, Pl…)

    Returns:
        Список слів з PDF, відсортований за частотністю.
    """
    import re
    from collections import Counter

    pdf_path = Path(__file__).parent.parent.parent / "data_raw" / "Begegnungen_А2.pdf"
    if not pdf_path.exists():
        print("  ⚠️  PDF not found — skipping PDF word extraction")
        return []

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("  ⚠️  PyMuPDF not installed — skipping PDF extraction")
        return []

    doc = fitz.open(str(pdf_path))
    all_text = ""
    for page in doc:
        all_text += str(page.get_text()) + " "
    doc.close()

    # Знаходимо всі слова (2+ символи, з умлаутами)
    words = re.findall(r"[A-Za-zÄäÖöÜüß]{3,}", all_text)
    freq = Counter(words)

    # Стоп-слова: URL-фрагменти, коди, скорочення
    junk = {
        "https", "http", "www", "net", "org", "com",
        "wordwall", "resource", "learningapps",
        "QR", "USB", "Akk", "Dat", "THEMA", "WORTSCHATZ",
        "GRAMMATIK", "VERTIEFUNGSTEIL", "Investor", "Selma",
        "Nico", "Pauli", "Codes", "overgeordneten",
        "übergeordneten",
    }

    result = []
    for word, count in freq.most_common():
        if count < 2:
            break
        if word in junk:
            continue
        result.append(word)

    return result


def print_stats(vocab: dict[str, int]) -> None:
    """Виводить статистику по словнику."""
    print(f"{'='*50}")
    print(f"  Vocab size: {len(vocab)} tokens")
    print(f"{'='*50}")
    # Покажемо перші та останні токени
    items = list(vocab.items())
    print(f"\n  First 10: {items[:10]}")
    print(f"  Last  10: {items[-10:]}")
    print()


if __name__ == "__main__":
    vocab = build_vocab()

    if "--stats" in sys.argv:
        print_stats(vocab)
    else:
        # Зберігаємо у файл (поруч зі скриптом)
        save_path = Path(__file__).parent / "vocab.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"✅ vocab.json saved — {len(vocab)} tokens")
