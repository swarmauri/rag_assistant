import logging
import pandas as pd
import json
from typing import Dict, List

import gradio as gr

from RagAssistant import RagAssistant

from gradio_tabs.chat_tab import ChatTab
from gradio_tabs.document_tab import DocumentTab
from gradio_tabs.document_edits_tab import DocumentEditsTab


class Gradio_UI:
    def __init__(
        self,
        api_key: str,
        llm: str,
        model_name: str = None,
        _init_file_path: str = None,
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
        self.load_button = None
        self.input_box = None
        self.file = None
        self.vector_store = None

        # ui variables
        self.user_sessions: Dict[str, List] = {}  # Store chat states per user
        self.documents = []
        self._init_file_path = None
        self.data_frame = None
        self.save_button = None
        self.save_df = None

        # Tabs
        self.chat_tab = ChatTab(
            assistant=self.assistant,
            api_key=self.api_key,
            show_system_context=self._show_system_context,
            show_api_key=self._show_api_key,
        ).chat_tab

        self.document_tab = DocumentTab(
            assistant=self.assistant,
            show_documents_tab=self._show_documents_tab,
            _init_file_path=_init_file_path,
        ).document_tab

        self.document_edit_tab = DocumentEditsTab(
            assistant=self.assistant
        ).document_edit_tab

        # App
        self.app = self._layout()

    # ------------------------------------------------------------ TABS AND THEIR COMPONENTS  ------------------------------------------------

    # -------------------------------------------------- HANDLERS ------------------------------------------------------

    # ----------------------------------- LAYOUT (arrangement of the interface) ---------------------------

    def _layout(self):
        """Defines the overall layout of the application."""
        with gr.Blocks(css=self.assistant.css, title=self.title) as self.app:
            with gr.Tabs():
                with gr.Tab("Chat"):
                    self.chat_tab()

                with gr.Tab("Documents", visible=self._show_documents_tab):
                    self.document_tab()

                with gr.Tab("Document Edits", visible=self._show_documents_tab):
                    self.document_edit_tab()

    # ------------------------------------ LAUNCH (main method to launch interface) ----------------------------------

    def launch(self):
        """Launches the Gradio application."""
        self._layout()
        self.app.launch(share=self.share_url)
