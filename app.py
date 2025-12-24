

import gradio as gr
from utils.model_loader import load_models
from utils.predict import predict

vectorizer, model = load_models()

def classify_email(text):
    if not text.strip():
        return {"__not_spam__": 0.5}  
    
    result = predict(text, vectorizer, model)
    
    if result == "Spam":
        return {"Spam": 1.0}  
    else:
        return {"Not Spam": 1.0}  


with gr.Blocks(theme="soft", css="footer {display: none !important}") as demo:
    gr.Markdown(
        """
        # 🚨 Spam Email Classifier
        Classify emails as **Spam** or **Not Spam** using TF-IDF + SVM
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            input_text = gr.Textbox(
                lines=10,
                placeholder="Paste the full email content here...",
                label="Email Text",
                info="Include subject and body for better accuracy"
            )
        with gr.Column(scale=1, min_width=200):
            output_label = gr.Label(
                label="Prediction",
                num_top_classes=1
            )
    
    with gr.Row():
        submit_btn = gr.Button("Classify", variant="primary", size="lg")
        clear_btn = gr.ClearButton([input_text, output_label], value="Clear")
    
    submit_btn.click(
        fn=classify_email,
        inputs=input_text,
        outputs=output_label
    )
    
    gr.Markdown("### Examples (click to load)")
    examples = gr.Examples(
        examples=[
            ["Win a free iPhone! Click here now!!! Limited time offer."],
            ["Earn money from home with this simple trick. Start today."],
            ["Hey, are we still meeting for lunch tomorrow?"],
            ["Meeting rescheduled to 3 PM. See you then!"],
        ],
        inputs=input_text,
        outputs=output_label,
        fn=classify_email,
        cache_examples=False  
                )


if __name__ == "__main__":
    demo.launch()