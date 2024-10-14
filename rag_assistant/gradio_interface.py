# Deprecating ...

from typing import Optional
import os

import gradio as gr
from RagAssistant import RagAssistant

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

assistant = RagAssistant(api_key=OPENAI_API_KEY, llm="openai")
head = """<link rel="icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAAXNSR0IArs4c6QAABWFJREFUWEe1V2tsFFUU/s7M7KO77Ra2LRisUArasLjbLqVdaRfQkACCNISgUQJCiCHGfyaYEIMSfIQY/K3GEAUkqNHwUJ6JEoht7ZZuu92uaCktILUJpSy0233PzNU7SClNu9MWvX8ms+ec7/vOuefMvUsYxypxu2dYgHRLS8utcbhjrttdIMJiaG+p69HzJz0Hl8dTKMLYwRiIMeW11ov1hzLFlFZUbSQS93EfNZGaGwz6ujP56wpwOCoeM2abrhOREUBCluXKNv+vbaOBOssXOSVJagRgZgxJOZaaFQr5bj6SAB7sqvS+KoA+J4KJqawucLHWOxpoWaW3joiqOLmqsm3BptqDehXWrcB9gPlub6lkwGcEeAC8xxgjgmDhdgY1xp9E9A7AGmRZfj3U7AvqkWsx43G67+NaWLVeFMVvAIhjxCmKorwcbKr/fry4ExJQ5vFeI9CsTOCMsauBxtri/1yAy+WyCmZbhIgyimaMsd7ueHZPj1/bFr01gQqUG9yerBhAUmZQJrfIcQv8/rQe+YR7oKzS20ZET+tsQSjQWOscD7muAN7p5wOBXAuZp0GQ8g4fPrgjMhCpUWQZ4XAYf3Xf+8Y8XlgIu90OUZJgs9mOv7Jh00dGSehTTOgtLy4eICI2liBtC862tlrzJOtWEK0gYDZAFkbMAgYbANPwaWGMIRpPIJVKI3zrpjZH9vzpMBoNsGaZ+SgO5+LESRAGiFEMYDEGXCWiM32pyJcrSkuj1NDaUSgahJ8JeCpT2RRFwZ2BCG6F7yIWT2iuT0yza88bvWHtackyo8A+BVNtORDFsSZ1iKU9lZKXUdOlK98C9NJo5IqiIhKNItw/gP7IIFT1XiVzrBZMy5uKntbvtIxnuF7EzdthRKL3Gl8QCLk52bDn2pCTbYUoCKPmxsC+oabfO/t5qXlpeZaxRBLRWFwDG4zFwX/nyyBJGmC+PRdmkwn1F85i3ewTmu3otTVYtGQ5Eskk+sL9muC0LGs2LjDbmoUciwXZlixkmU1adbStYrhLvlBH/HLXdXMilYaqqg8pNRoMyM2xYootR8uaB3FBZ06dgDfvDJ51GTT/88E06u+swvKVq4Z8eAJ3ByLoj0SRSj88kYIgwGw0oGROUYIaQx117V1/VimqAk5oNhm1veSE/H14U8XjcXy1fz8W5Aewrpr35oN1pC6JQNiNjZu3wGw2Dxm4YC4gMhhDLJFAIpnS3kVBREnxrHpq/qOrRlXZsUznAgf5LRTCoQMHcPt2X6ZeRX5BATZt3oJ5DsfIiRgZx5iqrtVmpulS5xsA7QWYdroNX52dV3Dyhx/RFmzNSDzS6Corwws1NSgqmj2aEN6t2xc65nw6NLQNbV3TRUFdCyInAWZFUbs+2L3r3e4bNx6u9YRkADNnFiV37t71PhgVCwL/FqAtLUePP+N0aheVjGfB1g1L/YudxgXi6FOkK0VRgV+CqZYvvr6wYCznjAKSDav3GiVhuy5TBgdZVT82VJ58a1ICButWlVlNQjOf5smJYCwWVcqtS0+3TEoAD0o1rD5nkITnJiNAVtg5g+fEskyxupm5K6sXgoSGDNewsfAVQsrT7PP5H0kADy6tqN4jCMKOiVRBYezDYGPtTr0Y3Qr8CyCWVniPCgKt0QPkdsbYsUBj7XoAip7/eAXA4XAYXfMK9lU8KW1kY9wLiTHm75APtF8b3Ob/P65kPJven55fb7MIu0xGmv9gOhhLy2iLJtnuqUtPHtHLerh93BUYCdp1emXJJ8eS5xlAb9ZkLSlcferyRIjv+05aAAco8yz28UM94Kvl/5YmtR5JgKuy+m0wsODFuj2TYv8n6G+wcRzTJW9piwAAAABJRU5ErkJggg==" type="image/png">"""


def info_fn(msg):
    gr.Info(msg)


def error_fn(msg):
    raise gr.Error(msg)


def warning_fn(msg):
    gr.Warning(msg)


# Gradio Interface
def setup_gradio_interface():
    # First
    with gr.Blocks(css=assistant.css) as assistant.retrieval_table:
        with gr.Row():
            assistant.retrieval_table = gr.Dataframe(
                interactive=False,
                wrap=True,
                line_breaks=True,
                elem_id="document-table-container",
                # height="800",
            )

    # Second
    with gr.Blocks(css=assistant.css) as assistant.chat:
        with gr.Row():
            chat_id = gr.State(None)
            assistant.chatbot = gr.Chatbot(
                type="messages",
                label="Chat History",
                layout="panel",
                elem_id="chat-dialogue-container",
                container=True,
                show_copy_button=True,
                height="70vh",
            )
        with gr.Row():
            assistant.input_box = gr.Textbox(label="Type here:", scale=6)
            assistant.send_button = gr.Button("Send", scale=1)
            assistant.clear_button = gr.Button("Clear", scale=1)

        with gr.Accordion("See Details", open=False):
            assistant.additional_inputs = [
                gr.Textbox(
                    label="API Key",
                    value=assistant.api_key or "Enter your API Key",
                    visible=assistant._show_api_key,
                ),
                gr.Dropdown(
                    assistant.get_allowed_models(),
                    value=assistant.model_name,
                    label="Model",
                    info="Select openai model",
                    visible=assistant._show_provider_model,
                ),
                gr.Textbox(
                    label="System Context",
                    value=assistant.system_context,
                    visible=assistant._show_system_context,
                ),
                gr.Checkbox(label="Fixed Retrieval", value=True, interactive=True),
                gr.Slider(
                    label="Top K",
                    value=10,
                    minimum=0,
                    maximum=100,
                    step=5,
                    interactive=True,
                ),
                gr.Slider(
                    label="Temperature",
                    value=1,
                    minimum=0.0,
                    maximum=1,
                    step=0.01,
                    interactive=True,
                ),
                gr.Slider(
                    label="Max new tokens",
                    value=256,
                    minimum=256,
                    maximum=4096,
                    step=64,
                    interactive=True,
                ),
                gr.Slider(
                    label="Conversation size",
                    value=12,
                    minimum=2,
                    maximum=36,
                    step=2,
                    interactive=True,
                ),
                gr.Slider(
                    label="Session Cache size",
                    value=12,
                    minimum=2,
                    maximum=36,
                    step=2,
                    interactive=True,
                ),
            ]

        submit_inputs = [chat_id, assistant.input_box]
        submit_inputs.extend(assistant.additional_inputs)
        # Function to handle sending messages
        assistant.send_button.click(
            assistant.chatbot_function,
            inputs=submit_inputs,
            outputs=[
                chat_id,
                assistant.input_box,
                assistant.retrieval_table,
                assistant.chatbot,
            ],
        )

        # Function to handle clearing the chat
        assistant.clear_button.click(
            assistant.clear_chat,
            inputs=[chat_id],
            outputs=[
                chat_id,
                assistant.input_box,
                assistant.retrieval_table,
                assistant.chatbot,
            ],
        )

    # Third
    with gr.Blocks(css=assistant.css) as assistant.document_table:
        with gr.Row():
            assistant.file = gr.File(
                label="Upload JSON File", value=assistant._init_file_path
            )
            assistant.vectorizer = gr.Dropdown(
                choices=assistant.available_vectorizers.keys(),
                value=assistant.available_vectorizers.keys()[0],
                label="Select vectorizer",
            )
            assistant.load_button = gr.Button("load")
        with gr.Row():
            if assistant._init_file_path:
                df = assistant._load_and_filter_json(assistant._init_file_path)
            assistant.data_frame = gr.Dataframe(
                interactive=True,
                wrap=True,
                line_breaks=True,
                elem_id="document-table-container",
                # height="700",
                # value=df,
            )
        with gr.Row():
            assistant.save_button = gr.Button("save")

        assistant.vectorizer.change(
            assistant.change_vectorizer,
            inputs=[assistant.vectorizer],
            outputs=assistant.data_frame,
        )
        assistant.load_button.click(
            assistant.load_json_from_file_info,
            inputs=[assistant.file],
            outputs=assistant.data_frame,
        )
        assistant.save_button.click(assistant.save_df, inputs=[assistant.data_frame])

    # Fourth
    with gr.Blocks(
        css=assistant.css, title="Swarmauri Rag Agent", head=head
    ) as assistant.app:
        with gr.Tab("chat", visible=True):
            assistant.chat.render()
        with gr.Tab("retrieval", visible=assistant._show_documents_tab):
            assistant.retrieval_table.render()
        with gr.Tab("documents", visible=assistant._show_documents_tab):
            assistant.document_table.render()


def launch(
    assistant,
    share: bool = False,
    show_api_key: bool = False,
    show_provider_model: bool = False,
    show_system_context: bool = False,
    show_documents_tab: bool = False,
    server_name: Optional[str] = None,
    documents_file_path: Optional[str] = None,
):
    # Create interfaces
    assistant._show_api_key = show_api_key
    assistant._show_provider_model = show_provider_model
    assistant._show_system_context = show_system_context
    assistant._show_documents_tab = show_documents_tab
    assistant._init_file_path = documents_file_path
    setup_gradio_interface()

    # Deploy interfaces
    kwargs = {}
    kwargs.update({"share": share})
    if server_name:
        kwargs.update({"server_name": server_name})
