import logging
import pandas as pd
import json
from typing import Dict, List

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
        # vector store params
        vector_store_db_path: str = "prompt_responses.db",
        vector_store_collection_name: str = "RAG Prompt Responses",
        vector_store_vector_size: int = 1024,
        vector_store_vectorizer: str = None,
        vector_store_api_key: str = None,
        vector_store_url: str = None,
    ):
        # params
        self.api_key = api_key
        self.llm = llm
        self.model_name = model_name
        self.title = title
        self.share_url = share_url

        # Rag Assistant
        self.assistant = RagAssistant(
            api_key=api_key,
            llm=llm,
            # rag params
            vector_store_db_path=vector_store_db_path,
            vector_store_api_key=vector_store_api_key,
            vector_store_url=vector_store_url,
            vector_store_collection_name=vector_store_collection_name,
            vector_store_vector_size=vector_store_vector_size,
            vector_store_vectorizer=vector_store_vectorizer,
        )
        self.allowed_models = self.assistant.get_allowed_models()

        # toggle values
        self._show_api_key = show_api_key
        self._show_documents_tab = show_documents_tab
        self._show_provider_llm = show_provider_llm
        self._show_provider_model = show_provider_model
        self._show_system_context = show_system_context
        self._show_documents_tab = show_documents_tab

        # gradio components
        self.send_button = None
        self.clear_button = None
        self.load_button = None
        self.input_box = None
        self.file = None
        self.vector_store = None

        # ui variables
        self.user_sessions: Dict[str, List] = {}  # Store chat states per user
        self.documents = []
        self.chat_id = None
        self.chatbot = None
        self._init_file_path = None
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
        """Modified Chat Tab to handle user-specific state."""
        with gr.Blocks(css=self.assistant.css) as self.chat:
            with gr.Row():
                # Each session state is user-specific
                self.chat_id = gr.State([])

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

            # Modify send button to use session-based state
            self.send_button.click(
                fn=self._reply_to_chat,
                inputs=[self.input_box, self.chat_id],
                outputs=[self.chatbot, self.chat_id, self.input_box],
            )

            # Clear button now resets the user's session state
            self.clear_button.click(
                fn=lambda: ([], []),
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
                    label="Upload JSON/PDF/txt File",
                    value=self._init_file_path,
                    file_count="multiple",
                    file_types=[".json", ".pdf", ".txt"],
                )
            self.vector_store = gr.Dropdown(
                choices=self.assistant.available_vector_stores.keys(),
                value=list(self.assistant.available_vector_stores.keys())[0],
                label="Select vector_store",
            )
            self.load_button = gr.Button("load")
            # Place event handlers inside the Blocks context
            self.vector_store.change(
                self.assistant.set_vector_store,
                inputs=[self.vector_store],
                outputs=[],
            )

        with gr.Row():
            if self._init_file_path:
                df = self.assistant._load_and_filter_json(self._init_file_path)
            self.data_frame = gr.Dataframe(
                headers=["Document", "Status", "Timestamp"],
                col_count=(3, "fixed"),
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
                outputs=[self.file, self.data_frame],
            )

    def _doc_edits(self):
        """Build the UI to load, edit, and save different file types."""
        with gr.Blocks(css=self.assistant.css) as self.retrieval_interface:
            with gr.Row():
                self.file_upload = gr.File(
                    label="Upload File (CSV, JSON, TXT)",
                )

            with gr.Row():
                self.content_display = gr.Textbox(
                    label="File Content",
                    lines=20,
                    interactive=True,
                    visible=False,
                )

            with gr.Row():
                self.save_button = gr.Button("Update & Upload")

            # Event handlers for file upload and save button
            self.file_upload.change(
                fn=self._on_file_upload,
                inputs=[self.file_upload],
                outputs=[
                    self.content_display,
                ],
            )

            self.save_button.click(
                fn=self._on_update_and_upload,
                inputs=[
                    self.content_display,
                ],
                outputs=[self.file_upload, self.content_display],
            )

    def _on_file_upload(self, file):
        """Handle file upload and show appropriate editor."""
        content = ""

        with open(file, "r") as f:
            content = f.read()

        if isinstance(content, str):  # TXT case
            self.documents = content
            return (gr.update(value=content, visible=True),)

        else:
            return "Unsupported file type."

    def _on_update_and_upload(self, content):
        """Handle saving edits based on the file type."""
        self.assistant.add_to_vector_store(content)
        gr.Info("Successfully updated and added to store")
        return None, ""

    # -------------------------------------------------- HANDLERS ------------------------------------------------------

    def _get_user_session(self, user_id: str) -> List:
        """Retrieve the chat history for a specific user."""
        return self.user_sessions.setdefault(user_id, [])

    def _update_user_session(self, user_id: str, message: str):
        """Update chat history for the given user."""
        self.user_sessions[user_id].append(message)

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

        conversation_dict = self.assistant.conversation.session_to_dict()

        return (
            conversation_dict,
            conversation_dict,
            "",
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

    def _on_load_document(self, files):
        """Loads a document and updates the document table."""
        for file in files:
            file_type = file.name.split(".")[-1]
            doc_name = file.name
            if file_type == "json":
                self.assistant.load_json_from_file_info(file)
            elif file_type == "pdf":
                self.assistant.load_pdf_from_file_info(file)
            from datetime import datetime

            # Create a new row as a list (expected format by gr.Dataframe)
            new_row = [
                doc_name,
                "Loaded",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ]
            self.documents.append(new_row)

            # Safely extract the current data or initialize an empty list
            current_data = self.data_frame.value

            # Ensure the data is a list of lists
            current_data["data"] = self.documents

            self.data_frame.value = current_data

            # Clear the file input and return the updated dataframe
            return None, gr.update(value=current_data)

        # If no file is provided, return the current state
        return None, self.documents

    def _document_event_handler(self):
        """Set up the load button click event."""
        self.load_button.click(
            fn=self._on_load_document,
            inputs=[self.file],  # Input: Uploaded file
            outputs=[
                self.file,
                self.data_frame,
            ],  # Output: Clear file input, update dataframe
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

                with gr.Tab("Documents", visible=self._show_documents_tab):
                    self._document_table()
                    self._document_event_handler()

                with gr.Tab("Document Edits", visible=self._show_documents_tab):
                    self._doc_edits()

            self.save_button.click(self._on_save_settings)

    # ------------------------------------ LAUNCH (main method to launch interface) ----------------------------------

    def launch(self):
        """Launches the Gradio application."""
        self._layout()
        self.app.launch(share=self.share_url)
