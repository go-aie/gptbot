import os

import gradio as gr
import requests


URL = 'http://127.0.0.1:8080'
CORPUS_ID = os.getenv('CORPUS_ID', '')


def handle_error(resp):
    if resp.status_code != 200:
        raise gr.Error(resp.json()['error']['message'])


def create(text):
    resp = requests.post(URL+'/upsert', json=dict(documents=[dict(text=text,metadata={'corpus_id':CORPUS_ID})]))
    handle_error(resp)
    return 'Created successfully!'


def split_text(text):
    resp = requests.post(URL+'/debug/split', json=dict(doc=dict(text=text,metadata={'corpus_id':CORPUS_ID})))
    handle_error(resp)
    return resp.json()


def split_file(file):
    with open(file.name) as f:
        return split_text(f.read())


def upload(file):
    with open(file.name) as f:
        resp = requests.post(URL+'/upload', data=dict(corpus_id=CORPUS_ID), files=dict(file=f))
        handle_error(resp)
        return 'Uploaded successfully!'


def clear():
    resp = requests.post(URL+'/delete', json=dict(document_ids=[]))
    handle_error(resp)
    return 'Cleared successfully!'


def chat(question, history):
    turns = [dict(question=h[0], answer=h[1]) for h in history]
    resp = requests.post(URL+'/chat', json=dict(question=question, in_debug=True, corpus_id=CORPUS_ID, history=turns))
    handle_error(resp)
    json = resp.json()
    return json['answer'], json['debug'].get('backend_prompt', '')


with gr.Blocks() as demo:
    with gr.Tab('Chat Bot'):
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label='Input')
        with gr.Accordion('Debug', open=False):
            prompt = gr.Textbox(label='Prompt')

        def user(msg, history):
            question = msg
            return '', history + [[question, None]]

        def bot(history):
            question = history[-1][0]
            answer, prompt = chat(question, history[:-1])
            history[-1][1] = answer
            return history, prompt

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot], [chatbot, prompt]
        )

    with gr.Tab('Knowledge Base'):
        status = gr.Textbox(label='Status Bar')
        btn = gr.Button(value="Clear All")
        btn.click(clear, inputs=None, outputs=[status])

        with gr.Tab('Document Text'):
            text = gr.Textbox(label='Document Text', lines=8)
            btn = gr.Button(value="Create")
            btn.click(create, inputs=[text], outputs=[status])

            with gr.Accordion('Debug', open=False):
                btn = gr.Button(value="Split")
                json = gr.JSON()
                btn.click(split_text, inputs=[text], outputs=[json])

        with gr.Tab('Document File'):
            file = gr.File(label='Document File')
            btn = gr.Button(value="Upload")
            btn.click(upload, inputs=[file], outputs=[status])

            with gr.Accordion('Debug', open=False):
                btn = gr.Button(value="Split")
                json = gr.JSON()
                btn.click(split_file, inputs=[file], outputs=[json])


if __name__ == '__main__':
    demo.launch(share=True)
