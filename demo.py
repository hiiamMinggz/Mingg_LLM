import gradio as gr
from synthesis import Response_Generator 

my_synthesis = Response_Generator()  
def medical_response(message, history):
    # if '?' not in message:
    #     message += ' ?'
    response = my_synthesis.llm_response(message)
    for i in range(len(response)):
        yield response[: i+1]

gr.ChatInterface(
    medical_response,
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="Ask me !", container=False, scale=7),
    title="Medical Bot",
    description="Ask Medical Bot any question",
    theme="soft",
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch()