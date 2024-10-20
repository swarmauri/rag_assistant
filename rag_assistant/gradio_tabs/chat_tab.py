import gradio as gr


class ChatTab:
    def __init__(
        self,
        assistant,
        api_key: str,
        show_api_key=True,
        show_provider_llm=True,
        show_provider_model=True,
        show_system_context=True,
    ):
        self.assistant = assistant
        self.api_key = api_key

        self._show_system_context = show_system_context
        self._show_api_key = show_api_key
        self._show_provider_llm = show_provider_llm
        self._show_provider_model = show_provider_model

        # Gradio UI settings
        self.chat = None

        # gradio components
        self.chat_id = None
        self.chatbot = None
        self.input_box = None
        self.send_button = None
        self.clear_button = None

    def chat_tab(self):
        with gr.Blocks(css=self.assistant.css) as self.chat:
            self._chat_component()

    # -------------------------------------------------- COMPONENTS ------------------------------------------------------
    def _chat_component(self):
        """Modified Chat Tab to handle user-specific state."""
        with gr.Row():
            # Each session state is user-specific
            self.chat_id = gr.State([])

            self.chatbot = gr.Chatbot(
                label="Chat History",
                type="messages",
                elem_id="chat-dialogue-container",
                show_copy_button=True,
                height="70vh",
                scale=2,
            )
            with gr.Column(scale=1):
                self._additional_inputs_component()
                self._llm_variables_component()
                self._settings_component()

        with gr.Row():
            self.input_box = gr.Textbox(label="Type here:", scale=6)
            self.send_button = gr.Button("Send", scale=1)
            self.clear_button = gr.Button("Clear", scale=1)

        # Modify send button to use session-based state
        self.send_button.click(
            fn=self._reply_to_chat_handler,
            inputs=[self.input_box, self.chat_id],
            outputs=[self.chatbot, self.chat_id, self.input_box],
        )

        # Clear button now resets the user's session state
        self.clear_button.click(
            fn=lambda: ([], []),
            outputs=[self.chatbot, self.chat_id],
        )

    def _additional_inputs_component(self):
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

    def _llm_variables_component(self):
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

    def _settings_component(self):
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

    # -------------------------------------------------- HANDLERS ------------------------------------------------------

    def _change_llm(self, llm):
        """Sets the selected LLM and updates the available models."""
        self.assistant.set_llm(llm)
        self.allowed_models = self.assistant.get_allowed_models()

        # Update the LLM dropdown and models dynamically
        return gr.update(choices=self.allowed_models, value=self.allowed_models[0])

    def _reply_to_chat_handler(self, message, chat_history):
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

        conversation_dict = self.assistant.conversation.session_to_dict()

        return (
            conversation_dict,
            conversation_dict,
            "",
        )

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
