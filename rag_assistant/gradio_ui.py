import logging

import gradio as gr

from RagAssistant import RagAssistant


class Gradio_UI:
    def __init__(self, api_key: str, llm: str, model_name: str = None):
        # params
        self.api_key = api_key
        self.llm = llm
        self.model_name = model_name

        # Rag Assistant
        self.assistant = RagAssistant(api_key=api_key, llm=llm)
        self.allowed_models = self.assistant.get_allowed_models()

        # toggle values
        self._show_api_key = True
        self._show_documents_tab = True
        self._show_provider_llm = True
        self._show_provider_model = True
        self._show_system_context = True

        # gradio components
        self.send_button = None
        self.clear_button = None
        self.load_button = None
        self.input_box = None
        self.file = None
        self.vectorizer = None

        # ui variables
        self.chat_id = None
        self.chatbot = None
        self._init_file_path = None
        self._show_documents_tab = False
        self.data_frame = None
        self.save_button = None
        self.save_df = None

        # static
        self.head = """<link rel="icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAAXNSR0IArs4c6QAABWFJREFUWEe1V2tsFFUU/s7M7KO77Ra2LRisUArasLjbLqVdaRfQkACCNISgUQJCiCHGfyaYEIMSfIQY/K3GEAUkqNHwUJ6JEoht7ZZuu92uaCktILUJpSy0233PzNU7SClNu9MWvX8ms+ec7/vOuefMvUsYxypxu2dYgHRLS8utcbhjrttdIMJiaG+p69HzJz0Hl8dTKMLYwRiIMeW11ov1hzLFlFZUbSQS93EfNZGaGwz6ujP56wpwOCoeM2abrhOREUBCluXKNv+vbaOBOssXOSVJagRgZgxJOZaaFQr5bj6SAB7sqvS+KoA+J4KJqawucLHWOxpoWaW3joiqOLmqsm3BptqDehXWrcB9gPlub6lkwGcEeAC8xxgjgmDhdgY1xp9E9A7AGmRZfj3U7AvqkWsx43G67+NaWLVeFMVvAIhjxCmKorwcbKr/fry4ExJQ5vFeI9CsTOCMsauBxtri/1yAy+WyCmZbhIgyimaMsd7ueHZPj1/bFr01gQqUG9yerBhAUmZQJrfIcQv8/rQe+YR7oKzS20ZET+tsQSjQWOscD7muAN7p5wOBXAuZp0GQ8g4fPrgjMhCpUWQZ4XAYf3Xf+8Y8XlgIu90OUZJgs9mOv7Jh00dGSehTTOgtLy4eICI2liBtC862tlrzJOtWEK0gYDZAFkbMAgYbANPwaWGMIRpPIJVKI3zrpjZH9vzpMBoNsGaZ+SgO5+LESRAGiFEMYDEGXCWiM32pyJcrSkuj1NDaUSgahJ8JeCpT2RRFwZ2BCG6F7yIWT2iuT0yza88bvWHtackyo8A+BVNtORDFsSZ1iKU9lZKXUdOlK98C9NJo5IqiIhKNItw/gP7IIFT1XiVzrBZMy5uKntbvtIxnuF7EzdthRKL3Gl8QCLk52bDn2pCTbYUoCKPmxsC+oabfO/t5qXlpeZaxRBLRWFwDG4zFwX/nyyBJGmC+PRdmkwn1F85i3ewTmu3otTVYtGQ5Eskk+sL9muC0LGs2LjDbmoUciwXZlixkmU1adbStYrhLvlBH/HLXdXMilYaqqg8pNRoMyM2xYootR8uaB3FBZ06dgDfvDJ51GTT/88E06u+swvKVq4Z8eAJ3ByLoj0SRSj88kYIgwGw0oGROUYIaQx117V1/VimqAk5oNhm1veSE/H14U8XjcXy1fz8W5Aewrpr35oN1pC6JQNiNjZu3wGw2Dxm4YC4gMhhDLJFAIpnS3kVBREnxrHpq/qOrRlXZsUznAgf5LRTCoQMHcPt2X6ZeRX5BATZt3oJ5DsfIiRgZx5iqrtVmpulS5xsA7QWYdroNX52dV3Dyhx/RFmzNSDzS6Corwws1NSgqmj2aEN6t2xc65nw6NLQNbV3TRUFdCyInAWZFUbs+2L3r3e4bNx6u9YRkADNnFiV37t71PhgVCwL/FqAtLUePP+N0aheVjGfB1g1L/YudxgXi6FOkK0VRgV+CqZYvvr6wYCznjAKSDav3GiVhuy5TBgdZVT82VJ58a1ICButWlVlNQjOf5smJYCwWVcqtS0+3TEoAD0o1rD5nkITnJiNAVtg5g+fEskyxupm5K6sXgoSGDNewsfAVQsrT7PP5H0kADy6tqN4jCMKOiVRBYezDYGPtTr0Y3Qr8CyCWVniPCgKt0QPkdsbYsUBj7XoAip7/eAXA4XAYXfMK9lU8KW1kY9wLiTHm75APtF8b3Ob/P65kPJven55fb7MIu0xGmv9gOhhLy2iLJtnuqUtPHtHLerh93BUYCdp1emXJJ8eS5xlAb9ZkLSlcferyRIjv+05aAAco8yz28UM94Kvl/5YmtR5JgKuy+m0wsODFuj2TYv8n6G+wcRzTJW9piwAAAABJRU5ErkJggg==" type="image/png">"""
        self.css = """
#chat-dialogue-container {
    min-height: 54vh !important;
}

#document-table-container {
    min-height: 80vh !important;
}

footer {
    display: none !important;
}
"""

        # Columns
        self.retrieval_table = None
        self.chat = None
        self.additional_inputs = None
        self.llm_variables = None
        self.settings = None
        self.document_table = None

        # App
        self.app = self._layout()

    def _retrieval_table(self):
        with gr.Blocks(css=self.css) as self.retrieval_table:
            with gr.Row():
                self.retrieval_table = gr.Dataframe(
                    interactive=False,
                    wrap=True,
                    line_breaks=True,
                    elem_id="document-table-container",
                    # height="800",
                )

    def _chat(self):
        """Defines the chat UI."""
        with gr.Blocks(css=self.css) as self.chat:
            with gr.Row():
                self.chat_id = gr.State([])  # Initialize chat history

                self.chatbot = gr.Chatbot(
                    label="Chat History",
                    elem_id="chat-dialogue-container",
                    show_copy_button=True,
                    height="70vh",
                )

            with gr.Row():
                self.input_box = gr.Textbox(label="Type here:", scale=6)
                self.send_button = gr.Button("Send", scale=1)
                self.clear_button = gr.Button("Clear", scale=1)

            # Send button click event
            self.send_button.click(
                fn=self._reply_to_chat,
                inputs=[self.input_box, self.chat_id],
                outputs=[self.chatbot, self.chat_id, self.input_box],
            )

            # Clear button click event
            self.clear_button.click(
                fn=lambda: ([], []),  # Reset both chatbot and chat history
                outputs=[self.chatbot, self.chat_id],
            )

    def _reply_to_chat(self, message, chat_history):
        """Processes the user message and adds it to the chat."""
        llm_kwargs = {
            "temperature": self.settings["Temperature"].value,
            "max_tokens": self.settings["Max tokens"].value,
        }

        # Get the response from the assistant
        response_content = self.assistant.agent.exec(
            input_data=message,
            top_k=self.settings["Top K elements"].value,
            llm_kwargs=llm_kwargs,
        )

        # Initialize chat history if it's None
        chat_history = chat_history or []

        # Add user and assistant messages as tuples
        chat_history.append((message, response_content))

        # Log the updated chat history for debugging purposes
        logging.info(f"Chat history: {chat_history}")

        return chat_history, chat_history, ""  # Update both chatbot and state

    def _additional_inputs(self):
        with gr.Accordion("Credentials", open=False):
            self.additional_inputs = {
                "API Key": gr.Textbox(
                    label="API Key",
                    value=self.api_key or "Enter your API Key",
                    visible=self._show_api_key,
                ),
                "Fixed Retrieval": gr.Checkbox(
                    label="Fixed Retrieval",
                    value=True,
                    interactive=True,
                ),
            }

    def _llm_variables(self):
        with gr.Accordion("LLM Variables", open=False):
            self.llm_variables = {
                "LLM": gr.Dropdown(
                    value=self.assistant.get_llm_name(),
                    choices=list(self.assistant.available_llms.keys()),
                    label="LLM",
                    info="Select the language model",
                    interactive=True,
                    visible=self._show_provider_llm,
                ),
                "Model": gr.Dropdown(
                    value=self.assistant.agent.llm.name,
                    choices=self.assistant.agent.llm.allowed_models,
                    label="Model",
                    info="Select openai model",
                    interactive=True,
                    visible=self._show_provider_model,
                ),
                "System Context": gr.Textbox(
                    label="System Context",
                    value=self.assistant.system_context.content,
                    visible=self._show_system_context,
                ),
            }
        self._llm_event_handlers()

    def _settings(self):
        with gr.Accordion("Settings", open=False):
            self.settings = {
                "Top K elements": gr.Slider(
                    label="Top K",
                    value=10,
                    minimum=0,
                    maximum=100,
                    step=5,
                    interactive=True,
                ),
                "Temperature": gr.Slider(
                    label="Temperature",
                    value=1,
                    minimum=0.0,
                    maximum=100.0,
                    step=0.1,
                    interactive=True,
                ),
                "Max tokens": gr.Slider(
                    label="Max tokens",
                    value=256,
                    minimum=256,
                    maximum=4096,
                    step=64,
                    interactive=True,
                ),
                "Conversation size": gr.Slider(
                    label="Conversation size",
                    value=12,
                    minimum=2,
                    maximum=36,
                    step=2,
                    interactive=True,
                ),
                "Session Cache size": gr.Slider(
                    label="Session Cache size",
                    value=12,
                    minimum=2,
                    maximum=36,
                    step=2,
                    interactive=True,
                ),
            }

        def _settings_event_handlers(self):
            self.settings["Conversation size"].change(
                fn=self.assistant.set_conversation_size,
                inputs=[self.settings["Conversation size"]],
                outputs=[],
            )
            self.settings["Session Cache size"].change(
                fn=self.assistant.set_session_cache_size,
                inputs=[self.settings["Session Cache size"]],
                outputs=[],
            )

    def _change_llm(self, llm):
        """Sets the selected LLM and updates the available models."""
        self.assistant.set_llm(llm)
        self.allowed_models = self.assistant.get_allowed_models()

        # Update the LLM dropdown and models dynamically
        return gr.update(choices=self.allowed_models, value=self.allowed_models[0])

    def _llm_event_handlers(self):
        self.llm_variables["LLM"].change(
            fn=self._change_llm,
            inputs=[self.llm_variables["LLM"]],
            outputs=[self.llm_variables["Model"]],
        )

        self.llm_variables["Model"].change(
            fn=self.assistant.set_model,
            inputs=[self.llm_variables["Model"]],
            outputs=[],
        )

    def _document_table(self):
        with gr.Blocks(css=self.css) as self.document_table:
            with gr.Row():
                self.file = gr.File(
                    label="Upload JSON File", value=self._init_file_path
                )
            self.vectorizer = gr.Dropdown(
                choices=self.assistant.available_vectorizers.keys(),
                value=list(self.assistant.available_vectorizers.keys())[0],
                label="Select vectorizer",
            )
            self.load_button = gr.Button("load")

            # Place event handlers inside the Blocks context
            self.vectorizer.change(
                self.assistant.set_vectorizer,
                inputs=[self.vectorizer],
                outputs=[],
            )
            self.load_button.click(
                self.assistant.load_json_from_file_info,
                inputs=[self.file],
                outputs=[],
            )

        with gr.Row():
            if self._init_file_path:
                df = self.assistant._load_and_filter_json(self._init_file_path)
            self.data_frame = gr.Dataframe(
                interactive=True,
                wrap=True,
                line_breaks=True,
                elem_id="document-table-container",
                # height="700",
                # value=df,
            )

        with gr.Row():
            self.save_button = gr.Button("save")
            # Ensure the save button's event is within the context
            self.save_button.click(self.save_df, inputs=[self.data_frame])

    def _layout(self):
        """Defines the overall layout of the application."""
        with gr.Blocks(css=self.css, title="RAG Assistant") as self.app:
            with gr.Tabs():
                with gr.Tab("Chat"):
                    self._chat()

                    with gr.Row():
                        self._llm_variables()
                        self._settings()
                        self._additional_inputs()

                with gr.Tab("Documents"):
                    self._document_table()

                with gr.Tab("Retrieval Table"):
                    self._retrieval_table()
                    # self.retrieval_table.render()

            # with gr.Row():
            #     self.file = gr.File(
            #         label="Upload Document", file_types=[".pdf", ".txt"]
            #     )
            #     self.load_button = gr.Button("Load Document")
            #     self.save_button = gr.Button("Save Settings")

            # Button event handlers
            # self.send_button.click(
            #     self._on_send_message, inputs=[self.input_box], outputs=[self.chatbot]
            # )
            # self.clear_button.click(self._on_clear_chat, outputs=[self.chatbot])
            # self.load_button.click(
            #     self._on_load_document,
            #     inputs=[self.file],
            #     outputs=[self.document_table],
            # )
            self.save_button.click(self._on_save_settings)

    def _on_load_document(self, file):
        """Loads a document and updates the document table."""
        if file:
            doc_name = file.name
            self.document_table.add_row([doc_name, "Loaded", gr.update.datetime_now()])

    def _on_save_settings(self):
        """Handles saving current settings."""
        # Here you can implement any logic to save settings (e.g., API key, model)
        print("Settings saved!")

    def launch(self):
        """Launches the Gradio application."""
        self._layout()
        self.app.launch()
