import logging

import gradio as gr

from RagAssistant import RagAssistant


class Gradio_UI:
    def __init__(
        self,
        api_key: str,
        llm: str,
        model_name: str = None,
        # Gradio UI settings
        title="Rag Assistant",
        share_url=False,
        show_api_key=True,
        show_documents_tab=True,
        show_provider_llm=True,
        show_provider_model=True,
        show_system_context=True,
    ):
        # params
        self.api_key = api_key
        self.llm = llm
        self.model_name = model_name
        self.title = title
        self.share_url = share_url

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

        # Tabs
        self.chat = None
        self.retrieval_table = None
        self.document_table = None

        # Tabs components
        self.additional_inputs = None
        self.llm_variables = None
        self.settings = None

        # App
        self.app = self._layout()

    # ------------------------------------------------------------ TABS AND THEIR COMPONENTS  ------------------------------------------------

    def _chat(self):
        """Chat Tab (*Main component*). Defines the chat UI"""
        with gr.Blocks(css=self.assistant.css) as self.chat:
            with gr.Row():
                self.chat_id = gr.State([])  # Initialize chat history

                self.chatbot = gr.Chatbot(
                    label="Chat History",
                    type="messages",
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

    def _additional_inputs(self):
        """Chat Tab; Additional settings component"""
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
        """Chat Tab; LLM vars component"""
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
        """Chat Tab; Settings component"""
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

    def _document_table(self):
        with gr.Blocks(css=self.assistant.css) as self.document_table:
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
        print(self.assistant.agent)

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
            self.save_button.click(
                self._on_load_document,
                inputs=[self.file],
                outputs=[self.data_frame],
            )

    def _retrieval_table(self):
        """retrieve table tab"""
        with gr.Blocks(css=self.assistant.css) as self.retrieval_table:
            with gr.Row():
                self.retrieval_table = gr.Dataframe(
                    interactive=False,
                    wrap=True,
                    line_breaks=True,
                    elem_id="document-table-container",
                    # height="800",
                )

    # -------------------------------------------------- HANDLERS ------------------------------------------------------

    def _reply_to_chat(self, message, chat_history):
        """Chat handler
        Processes the user message and adds it to the chat."""
        llm_kwargs = {
            "temperature": self.settings["Temperature"].value,
            "max_tokens": self.settings["Max tokens"].value,
        }

        # Get the response from the assistant
        self.assistant.agent.exec(
            input_data=message,
            top_k=self.settings["Top K elements"].value,
            llm_kwargs=llm_kwargs,
        )

        # Initialize chat history if it's None
        chat_history = self.assistant.conversation.session_to_dict()

        # Log the updated chat history for debugging purposes
        logging.info(f"Chat history: {chat_history}")
        print(f"Files: {self.assistant.agent.last_retrieved}")

        return chat_history, chat_history, ""  # Update both chatbot and state

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

    def _on_load_document(self, file):
        """Loads a document and updates the document table."""
        if file:
            doc_name = file.name
            self.assistant.load_json_from_file_info(file)
            from datetime import datetime

            return [[doc_name], ["Loaded"], [datetime.now().isoformat()]]

    def _document_event_handler(self):
        self.load_button.click(
            # self.assistant.load_json_from_file_info,
            self._on_load_document,
            inputs=[self.file],
            outputs=[self.data_frame],
        )

    def _on_save_settings(self):
        """Handles saving current settings."""
        # Here you can implement any logic to save settings (e.g., API key, model)

        print("Settings saved!")

    # ----------------------------------- LAYOUT (arrangement of the interface) ---------------------------

    def _layout(self):
        """Defines the overall layout of the application."""
        with gr.Blocks(css=self.assistant.css, title=self.title) as self.app:
            with gr.Tabs():
                with gr.Tab("Chat"):
                    self._chat()

                    with gr.Row():
                        self._llm_variables()
                        self._settings()
                        self._additional_inputs()

                with gr.Tab("Documents"):
                    self._document_table()
                    self._document_event_handler()

                with gr.Tab("Retrieval Table"):
                    self._retrieval_table()

            self.save_button.click(self._on_save_settings)

    # ------------------------------------ LAUNCH (main method to launch interface) ----------------------------------

    def launch(self):
        """Launches the Gradio application."""
        self._layout()
        self.app.launch(share=self.share_url)
