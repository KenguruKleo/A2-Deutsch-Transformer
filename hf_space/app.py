import gradio as gr
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

# Load model and tokenizer from the Hugging Face Hub
repo_id = "kengurukleo/deutsch_a2_transformer"
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
tokenizer = PreTrainedTokenizerFast.from_pretrained(repo_id, trust_remote_code=True)

bos_id = tokenizer.convert_tokens_to_ids("<BOS>")
eos_id = tokenizer.convert_tokens_to_ids("<EOS>")


def check_grammar(text: str) -> str:
    if not text.strip():
        return "Будь ласка, введіть німецьке речення."

    prompt_ids = [bos_id] + tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids]).long()

    output_ids = model.generate(
        input_ids,
        max_new_tokens=64,
        eos_token_id=eos_id,
    )

    # Decode only the generated part (after the prompt)
    generated = output_ids[0].tolist()[len(prompt_ids):]
    if eos_id in generated:
        generated = generated[:generated.index(eos_id)]

    return tokenizer.decode(generated).strip()


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
