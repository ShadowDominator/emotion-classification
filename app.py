import gradio as gr
from transformers import pipeline


classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
 
def fn_emotion(text):
    results = classifier(text, padding='max_length', max_length=512)
    return {label['label']: [label['score']] for label in results[0]}



with gr.Blocks(title="Emotion",css="footer {visibility: hidden}") as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Sentence Emotion")
            with gr.Row():           
                with gr.Column():
                    inputs = gr.TextArea(label="sentence",value=" I am so excited to go on vacation!",interactive=True)
                    btn = gr.Button(value="RUN")
                with gr.Column():
                    output = gr.Label(label="output")
                btn.click(fn=fn_emotion,inputs=[inputs],outputs=[output])
demo.launch()