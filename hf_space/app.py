import gradio as gr
import torch
from transformers import AutoModelForCausalLM

# Load the model directly from the Hugging Face Hub
repo_id = "kengurukleo/deutsch_a2_transformer"
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)

def check_grammar(text):
    if not text.strip():
        return "Будь ласка, введіть німецьке речення."
    
    # Check if tokenizer was initialized by the model wrapper
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return "Помилка: Токенізатор не завантажений. Перевірте конфігурацію моделі."
        
    # Prepare input
    input_ids = torch.tensor([tokenizer.encode(text, add_eos=False)]).long()
    
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=64)
    
    # Decode and extract only the correction part
    full_text = tokenizer.decode(output_ids[0].tolist())
    response = full_text.replace(text, "").strip()
    
    return response

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🇩🇪 A2 Deutsch Grammar Tutor")
    gr.Markdown("Введіть німецьке речення рівнів A1-A2, і модель перевірить його на помилки та надасть пояснення українською мовою.")
    
    with gr.Row():
        input_text = gr.Textbox(
            label="Ваше речення", 
            placeholder="Наприклад: Ich habe nach Berlin gefahren...",
            lines=2
        )
    
    with gr.Row():
        check_btn = gr.Button("Перевірити", variant="primary")
    
    output_text = gr.Textbox(label="Результат та пояснення", lines=5)
    
    gr.Examples(
        examples=[
            ["Ich habe nach Berlin gefahren."],
            ["Heute gehe Ich ins Kino."],
            ["Ich kann sprechen Deutsch."],
            ["Wo du wohnst?"],
            ["Ich habe kein Auto."],
            ["Ich bin nach Hause gegangen."],
            ["Ich war zu Hause."],
            ["Ich habe gegangen."],
            ["Er ist Lehrer."],
            ["Er sind Lehrer."],
            ["In das Kino."],
            ["Das ist mein Bruder."],
            ["Das ist meine Bruder."],
            ["Ich freue mich."],
            ["Heute gehe ich ins Kino."],
            ["Ich freue dich."],
            ["Ich bin einen Apfel gegessen."],
            ["Mit den Bus."],
            ["Ich war zu Hause gestern."],
            ["In dem Kino gehen wir."],
            ["Ich aufstehe um 7 Uhr."],
            ["Weil er ist krank."],
            ["Das ist mehr gut."],
            ["Ich habe der Tisch."],
        ],
        inputs=input_text,
    )

    check_btn.click(fn=check_grammar, inputs=input_text, outputs=output_text)

# Launch with SSR mode disabled for better stability in Spaces
demo.launch(ssr_mode=False)
