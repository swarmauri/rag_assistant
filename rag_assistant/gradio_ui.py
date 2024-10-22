from typing import Dict
import gradio as gr

from rag_assistant.RagAssistant import RagAssistant
from gradio_tabs.chat_tab import ChatTab
from gradio_tabs.document_tab import DocumentTab
from gradio_tabs.document_edits_tab import DocumentEditsTab


class GradioUI:
    def __init__(
        self,
        # Gradio UI settings
        config: Dict,
        # RAG Assistant params
        api_key: str,
        llm: str = "openai",
        model_name: str = None,
        db_path: str = "conversations.db",
        vector_store_vector_size: int = 1024,
        vector_store_vectorizer: str = None,
    ):
        # params
        self.api_key = api_key
        self.llm = llm
        self.model_name = model_name
        self.system_context = config.get(
            "system_context", "You are a helpful assistant."
        )

        # Gradio UI settings
        self.title = config.get("title", "RAG Assistant")
        self.share_url = config.get("share_url", False)
        self._init_file_path = config.get("init_file_path", None)
        self._show_chat_tab = config.get("show_chat_tab", True)
        self._show_documents_tab = config.get("show_documents_tab", True)
        self._show_document_edit_tab = config.get("show_document_edit_tab", True)
        self.chat_tab_config = config.get("chat_tab_config", {})
        self.document_tab_config = config.get("document_tab_config", {})

        # Rag Assistant
        self.assistant_kwargs = {
            "api_key": api_key,
            "llm": llm,
            "model_name": model_name,
            "system_context": self.system_context,
            "db_path": db_path,
            "vector_store_vector_size": vector_store_vector_size,
            "vector_store_vectorizer": vector_store_vectorizer,
        }
        self.assistant = RagAssistant(**self.assistant_kwargs)
        self.allowed_models = self.assistant.get_allowed_models()

        # Tabs
        self.chat_tab = ChatTab(
            assistant=self.assistant, api_key=api_key, config=self.chat_tab_config
        ).chat_tab

        self.document_tab = DocumentTab(
            assistant=self.assistant,
            config=self.document_tab_config,
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
                with gr.Tab("Chat", visible=self._show_chat_tab):
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
