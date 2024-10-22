import logging
import pandas as pd
import json
from typing import Dict, List

import gradio as gr

from RagAssistant import RagAssistant

from gradio_tabs.chat_tab import ChatTab
from gradio_tabs.document_tab import DocumentTab
from gradio_tabs.document_edits_tab import DocumentEditsTab


class GradioUI:
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
        show_document_edit_tab=True,
        show_provider_llm=True,
        show_provider_model=True,
        show_system_context=True,
        # vector store params
        vector_store_vector_size: int = 1024,
        vector_store_vectorizer: str = None,
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
            # vector store params
            vector_store_vector_size=vector_store_vector_size,
            vector_store_vectorizer=vector_store_vectorizer,
        )
        self.allowed_models = self.assistant.get_allowed_models()

        # toggle values
        self._show_documents_tab = show_documents_tab
        self._show_document_edit_tab = show_documents_tab

        # Tabs
        self.chat_tab = ChatTab(
            show_api_key=show_api_key,
            show_system_context=show_system_context,
            show_provider_llm=show_provider_llm,
            show_provider_model=show_provider_model,
            assistant=self.assistant,
            api_key=api_key,
        ).chat_tab

        self.document_tab = DocumentTab(
            assistant=self.assistant,
            _init_file_path=_init_file_path,
        ).document_tab

        self.document_edit_tab = DocumentEditsTab(
            assistant=self.assistant
        ).document_edit_tab

        # App
        self.app = self._layout()

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
