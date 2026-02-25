"""
app.py — Gradio interface for A2 Deutsch Grammar Tutor (HF Space).

Loads a standard BartForConditionalGeneration from a HF model repo.
No custom model code needed — pure HF transformers inference.
"""

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── 1. Load Model & Tokenizer (standard HF pipeline) ──
MODEL_ID = "KenguruKleo/deutsch-a2-tutor-model"

print(f"📥 Loading model from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model.eval()
print("✅ Model loaded successfully")


def check_grammar(text: str) -> str:
    """
    Check grammar of a German sentence.

    Pipeline:
      text → tokenizer() → input_ids [1, T]
           → Encoder → memory [1, T, 256]
           → Decoder (greedy) → output_ids [1, T_out]
           → decode → result string
    """
    if not text.strip():
        return "Будь ласка, введіть німецьке речення."

    # Tokenize (HF handles special tokens automatically)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=64,
            num_beams=1,
            do_sample=False,
        )

    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return result.strip()


# ── 2. Premium UI ──
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="Deutsch A2 Grammar Tutor", theme=theme) as demo:
    gr.Markdown("""
    # 🇩🇪 Deutsch A2 Grammar Tutor (v2.1)
    ### Identify, correct, and learn from your German mistakes!

    *Powered by a standard HF BART Encoder-Decoder Transformer.*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Your sentence",
                placeholder="e.g., Ich habe nach Berlin gefahren.",
                lines=3,
            )
            check_btn = gr.Button("Check Grammar", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Markdown(label="Result and Explanation")

    examples = [
        ["Ich habe den Auto."],
        ["Wo du wohnst?"],
        ["Ich habe nach Berlin gefahren."],
    ]

    gr.Examples(examples=examples, inputs=input_text)

    check_btn.click(fn=check_grammar, inputs=input_text, outputs=output_text)

    gr.Markdown("""
    ---
    ### 📚 How it works
    1. Enter a German sentence (A1-A2 level).
    2. The model will analyze the structure.
    3. If there's an error, you'll get a **Correction** and a detailed **Explanation in Ukrainian**.
    """)

if __name__ == "__main__":
    demo.launch()
