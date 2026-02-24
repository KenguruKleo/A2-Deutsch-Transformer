import gradio as gr
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

# Load model and tokenizer from the Hugging Face Hub
repo_id = "kengurukleo/deutsch_a2_transformer"
model = AutoModelForCausalLM.from_pretrained(repo_id)
# Using PreTrainedTokenizerFast directly as it's a native GPT-2 tokenizer now
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_id)

bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id


def check_grammar(text: str) -> str:
    if not text.strip():
        return "Будь ласка, введіть німецьке речення."

    prompt_ids = [bos_id] + tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids]).long()

    output_ids = model.generate(
        input_ids,
        max_length=64,
        do_sample=True,
        temperature=0.3,
        top_k=50,
        eos_token_id=eos_id,
        pad_token_id=eos_id, # Standard practice for GPT-2 padding
    )

    # Decode only the generated part (after the prompt)
    generated = output_ids[0].tolist()[len(prompt_ids):]
    if eos_id in generated:
        generated = generated[:generated.index(eos_id)]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


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
