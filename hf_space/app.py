import gradio as gr
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

# Load model and tokenizer from the Hugging Face Hub
repo_id = "kengurukleo/deutsch_a2_transformer"
model    = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
tokenizer = PreTrainedTokenizerFast.from_pretrained(repo_id, trust_remote_code=True)

def check_grammar(text: str) -> str:
    if not text.strip():
        return "Будь ласка, введіть німецьке речення."

    bos_id = tokenizer.convert_tokens_to_ids("<BOS>")
    eos_id = tokenizer.convert_tokens_to_ids("<EOS>")

    ids = [bos_id] + tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([ids]).long()

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=64)

    # Decode and strip the input prefix
    full_ids = output_ids[0].tolist()
    # Remove BOS and everything up to the input length
    response_ids = full_ids[len(ids):]
    # Remove EOS if present
    if eos_id in response_ids:
        response_ids = response_ids[:response_ids.index(eos_id)]

    return tokenizer.decode(response_ids).strip()


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🇩🇪 A2 Deutsch Grammar Tutor")
    gr.Markdown(
        "Enter a German sentence (A1–A2 level) and the model will check it for errors "
        "and provide an explanation in Ukrainian."
    )

    with gr.Row():
        input_text = gr.Textbox(
            label="Your sentence",
            placeholder="e.g. Ich habe nach Berlin gefahren...",
            lines=2,
        )

    with gr.Row():
        check_btn = gr.Button("Check", variant="primary")

    output_text = gr.Textbox(label="Result and explanation", lines=5)

    gr.Examples(
        examples=[
            ["Ich habe nach Berlin gefahren."],
            ["Heute gehe Ich ins Kino."],
            ["Ich kann sprechen Deutsch."],
            ["Wo du wohnst?"],
            ["Ich habe kein Auto."],
            ["Ich bin nach Hause gegangen."],
            ["Ich habe gegangen."],
            ["Er sind Lehrer."],
            ["Ich freue dich."],
            ["Ich aufstehe um 7 Uhr."],
            ["Weil er ist krank."],
            ["Das ist mehr gut."],
            ["Ich habe den Auto."],
            ["Mit den Bus."],
        ],
        inputs=input_text,
    )

    check_btn.click(fn=check_grammar, inputs=input_text, outputs=output_text)

demo.launch(ssr_mode=False)
