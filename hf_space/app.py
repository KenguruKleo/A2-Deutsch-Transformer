import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer from the Hugging Face Hub (v2.0 BART Edition)
repo_id = "kengurukleo/deutsch_a2_transformer" 

try:
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

def check_grammar(text: str) -> str:
    if not text.strip():
        return "Будь ласка, введіть німецьке речення."
    
    if model is None:
        return "Модель завантажується або недоступна. Спробуйте пізніше."

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)

    output_ids = model.generate(
        **inputs,
        max_length=64,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# Gradio interface (Rich UI)
with gr.Blocks() as demo:
    gr.Markdown("# 🇩🇪 A2 Deutsch Grammar Tutor v2.0")
    gr.Markdown(
        "Powered by a custom **BART (Encoder-Decoder)** Transformer. "
        "Enter a German sentence (A1–A2 level) and get instant feedback with explanations in Ukrainian."
    )

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Your sentence",
                placeholder="e.g. Ich habe nach Berlin gefahren...",
                lines=2,
            )
            check_btn = gr.Button("Check Grammar", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Result and Explanation", lines=6, interactive=False)

    gr.Examples(
        examples=[
            ["Ich habe nach Berlin gefahren."],
            ["Heute gehe Ich ins Kino."],
            ["Ich kann sprechen Deutsch."],
            ["Wo du wohnst?"],
            ["Weil er є krank."],
            ["Er lerne Deutsch."],
            ["Ich habe den Auto."],
            ["Mit den Bus."],
        ],
        inputs=input_text,
    )

    check_btn.click(fn=check_grammar, inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
