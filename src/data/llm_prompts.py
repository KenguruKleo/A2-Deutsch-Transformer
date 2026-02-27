"""
llm_prompts.py — Prompt templates for LLM-based training data generation.

Supports two modes:
  - "incorrect": generates a sentence with a grammar error + correction + Ukrainian explanation
  - "correct":   generates a grammatically correct sentence + Ukrainian explanation

Topic is chosen randomly by OllamaGenerator before calling these prompts.
"""

# ── Topic Taxonomy ──────────────────────────────────────────────────────────
# Randomly picked per request. Each maps to a human-readable description
# that is passed verbatim into the prompt.

TOPICS: dict[str, str] = {
    # Verbs
    "verb_conjugation":      "verb conjugation (present tense: ich, du, er/sie/es, wir, ihr, sie)",
    "haben_sein":            "correct use of haben vs. sein as auxiliary verbs",
    "modal_verbs":           "modal verbs (können, müssen, wollen, sollen, dürfen, möchten)",
    "separable_verbs":       "separable verbs (aufmachen, anrufen, aufstehen, etc.)",
    "reflexive_verbs":       "reflexive verbs (sich waschen, sich freuen, sich setzen, etc.)",
    "perfekt":               "Perfekt tense (haben/sein + Partizip II)",
    "imperativ":             "Imperativ (commands: du, ihr, Sie forms)",
    # Syntax
    "word_order_inversion":  "word order inversion (verb-second rule after adverbs of time/place)",
    "questions":             "question formation (W-Fragen and Ja/Nein-Fragen)",
    "subordinate_weil":      "subordinate clauses with 'weil' (verb goes to the end)",
    "subordinate_dass_wenn": "subordinate clauses with 'dass' or 'wenn'",
    "negation":              "negation with 'nicht' or 'kein'",
    # Cases
    "nominativ":             "Nominativ case (subject of the sentence)",
    "akkusativ":             "Akkusativ case (direct object, articles: den/einen for masculine)",
    "dativ":                 "Dativ case (indirect object, mit/bei/nach/von/zu/aus + Dativ)",
    "genitiv":               "Genitiv case (possession: des/der)",
    "adjective_endings":     "adjective endings after definite/indefinite articles",
    "prepositions":          "prepositions with fixed cases (auf/an/in + Akk or Dat for direction/location)",
    # Articles & Pronouns
    "definite_articles":     "definite articles (der/die/das in Nominativ, Akkusativ, Dativ)",
    "indefinite_articles":   "indefinite articles (ein/eine/einen/einem)",
    "possessive_pronouns":   "possessive pronouns (mein/dein/sein/ihr/unser/euer/ihr + correct endings)",
}

# ── Prompt Templates ─────────────────────────────────────────────────────────

_BASE_RULES = """\
You are creating training data for a German grammar checker. \
Respond with valid JSON only. No thinking. No explanation. No markdown. No code blocks. \
Just the raw JSON object. \
The JSON must use actual UTF-8 characters, not unicode escapes. \
Use ❌ ✅ 📝 directly."""

PROMPT_INCORRECT = (
    _BASE_RULES
    + """

Generate 1 example of an INCORRECT German A1-A2 sentence about: {topic}

Required JSON format (copy this structure exactly):
{{"input": "INCORRECT_SENTENCE", "output": "❌ Incorrect.\\n✅ Correct: CORRECTED_SENTENCE\\n📝 Пояснення: EXPLANATION_IN_UKRAINIAN"}}

Rules:
- The sentence must contain exactly ONE grammatical error related to the topic.
- The correction must fix only that error and nothing else.
- The explanation must be in Ukrainian, 1-2 sentences max.
- Use simple A1-A2 vocabulary.
{vocab_hint}
Example:
{{"input": "Heute ich gehe ins Kino.", "output": "❌ Incorrect.\\n✅ Correct: Heute gehe ich ins Kino.\\n📝 Пояснення: Після обставини часу дієслово стоїть на другому місці (інверсія)."}}

Generate a new example about {topic}. Raw JSON only:"""
)

PROMPT_CORRECT = (
    _BASE_RULES
    + """

Generate 1 example of a CORRECT German A1-A2 sentence about: {topic}

Required JSON format (copy this structure exactly):
{{"input": "CORRECT_SENTENCE", "output": "✅ Correct.\\n📝 Пояснення: EXPLANATION_IN_UKRAINIAN"}}

Rules:
- The sentence must be grammatically correct.
- The explanation must describe the grammar rule demonstrated, in Ukrainian, 1-2 sentences max.
- Use simple A1-A2 vocabulary.
- Do NOT make errors — the sentence must be fully correct.
{vocab_hint}
Example:
{{"input": "Heute gehe ich ins Kino.", "output": "✅ Correct.\\n📝 Пояснення: Після обставини часу 'Heute' дієслово 'gehe' стоїть на другому місці (правило інверсії)."}}

Generate a new example about {topic}. Raw JSON only:"""
)


def build_prompt(
    topic_key: str,
    mode: str,
    vocabulary_hint: list[str] | None = None,
) -> str:
    """
    Build a prompt string for the given topic and mode.

    Args:
        topic_key:       One of the keys in TOPICS dict.
        mode:            Either "incorrect" or "correct".
        vocabulary_hint: Optional list of German words to include in the prompt.
                         When provided, the LLM is asked to use them in the sentence,
                         increasing diversity across repeated topic calls.

    Returns:
        Formatted prompt string ready to send to an LLM provider.
    """
    topic_description = TOPICS[topic_key]
    if vocabulary_hint:
        words_str = ", ".join(f'"{w}"' for w in vocabulary_hint)
        vocab_line = f"- Try to use the following German words in the sentence: {words_str}\n"
    else:
        vocab_line = ""

    if mode == "incorrect":
        return PROMPT_INCORRECT.format(topic=topic_description, vocab_hint=vocab_line)
    elif mode == "correct":
        return PROMPT_CORRECT.format(topic=topic_description, vocab_hint=vocab_line)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'incorrect' or 'correct'.")
