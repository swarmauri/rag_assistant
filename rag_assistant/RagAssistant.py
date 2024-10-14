from datetime import datetime
import json
import logging
import uuid

import pandas as pd

from swarmauri.agents.concrete.RagAgent import RagAgent
from swarmauri.conversations.concrete.SessionCacheConversation import (
    SessionCacheConversation,
)
from swarmauri.documents.concrete.Document import Document
from swarmauri.llms.concrete.AnthropicModel import AnthropicModel
from swarmauri.llms.concrete.GeminiProModel import GeminiProModel
from swarmauri.llms.concrete.GroqModel import GroqModel
from swarmauri.llms.concrete.MistralModel import MistralModel
from swarmauri.llms.concrete.OpenAIModel import OpenAIModel
from swarmauri.utils.load_documents_from_json import load_documents_from_json_file
from swarmauri.utils.sql_log import sql_log
from swarmauri.vector_stores.concrete.Doc2VecVectorStore import Doc2VecVectorStore
from swarmauri.vector_stores.concrete.MlmVectorStore import MlmVectorStore
from swarmauri.vector_stores.concrete.TfidfVectorStore import TfidfVectorStore

head = """<link rel="icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAAXNSR0IArs4c6QAABWFJREFUWEe1V2tsFFUU/s7M7KO77Ra2LRisUArasLjbLqVdaRfQkACCNISgUQJCiCHGfyaYEIMSfIQY/K3GEAUkqNHwUJ6JEoht7ZZuu92uaCktILUJpSy0233PzNU7SClNu9MWvX8ms+ec7/vOuefMvUsYxypxu2dYgHRLS8utcbhjrttdIMJiaG+p69HzJz0Hl8dTKMLYwRiIMeW11ov1hzLFlFZUbSQS93EfNZGaGwz6ujP56wpwOCoeM2abrhOREUBCluXKNv+vbaOBOssXOSVJagRgZgxJOZaaFQr5bj6SAB7sqvS+KoA+J4KJqawucLHWOxpoWaW3joiqOLmqsm3BptqDehXWrcB9gPlub6lkwGcEeAC8xxgjgmDhdgY1xp9E9A7AGmRZfj3U7AvqkWsx43G67+NaWLVeFMVvAIhjxCmKorwcbKr/fry4ExJQ5vFeI9CsTOCMsauBxtri/1yAy+WyCmZbhIgyimaMsd7ueHZPj1/bFr01gQqUG9yerBhAUmZQJrfIcQv8/rQe+YR7oKzS20ZET+tsQSjQWOscD7muAN7p5wOBXAuZp0GQ8g4fPrgjMhCpUWQZ4XAYf3Xf+8Y8XlgIu90OUZJgs9mOv7Jh00dGSehTTOgtLy4eICI2liBtC862tlrzJOtWEK0gYDZAFkbMAgYbANPwaWGMIRpPIJVKI3zrpjZH9vzpMBoNsGaZ+SgO5+LESRAGiFEMYDEGXCWiM32pyJcrSkuj1NDaUSgahJ8JeCpT2RRFwZ2BCG6F7yIWT2iuT0yza88bvWHtackyo8A+BVNtORDFsSZ1iKU9lZKXUdOlK98C9NJo5IqiIhKNItw/gP7IIFT1XiVzrBZMy5uKntbvtIxnuF7EzdthRKL3Gl8QCLk52bDn2pCTbYUoCKPmxsC+oabfO/t5qXlpeZaxRBLRWFwDG4zFwX/nyyBJGmC+PRdmkwn1F85i3ewTmu3otTVYtGQ5Eskk+sL9muC0LGs2LjDbmoUciwXZlixkmU1adbStYrhLvlBH/HLXdXMilYaqqg8pNRoMyM2xYootR8uaB3FBZ06dgDfvDJ51GTT/88E06u+swvKVq4Z8eAJ3ByLoj0SRSj88kYIgwGw0oGROUYIaQx117V1/VimqAk5oNhm1veSE/H14U8XjcXy1fz8W5Aewrpr35oN1pC6JQNiNjZu3wGw2Dxm4YC4gMhhDLJFAIpnS3kVBREnxrHpq/qOrRlXZsUznAgf5LRTCoQMHcPt2X6ZeRX5BATZt3oJ5DsfIiRgZx5iqrtVmpulS5xsA7QWYdroNX52dV3Dyhx/RFmzNSDzS6Corwws1NSgqmj2aEN6t2xc65nw6NLQNbV3TRUFdCyInAWZFUbs+2L3r3e4bNx6u9YRkADNnFiV37t71PhgVCwL/FqAtLUePP+N0aheVjGfB1g1L/YudxgXi6FOkK0VRgV+CqZYvvr6wYCznjAKSDav3GiVhuy5TBgdZVT82VJ58a1ICButWlVlNQjOf5smJYCwWVcqtS0+3TEoAD0o1rD5nkITnJiNAVtg5g+fEskyxupm5K6sXgoSGDNewsfAVQsrT7PP5H0kADy6tqN4jCMKOiVRBYezDYGPtTr0Y3Qr8CyCWVniPCgKt0QPkdsbYsUBj7XoAip7/eAXA4XAYXfMK9lU8KW1kY9wLiTHm75APtF8b3Ob/P65kPJven55fb7MIu0xGmv9gOhhLy2iLJtnuqUtPHtHLerh93BUYCdp1emXJJ8eS5xlAb9ZkLSlcferyRIjv+05aAAco8yz28UM94Kvl/5YmtR5JgKuy+m0wsODFuj2TYv8n6G+wcRzTJW9piwAAAABJRU5ErkJggg==" type="image/png">"""


class RagAssistant:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        system_context: str = "You are a helpful assistant.",
        db_path: str = "prompt_responses.db",
    ):
        logging.info("Initializing... this will take a moment.")
        self.system_context = system_context
        self.api_key = api_key
        self.db_path = db_path
        self.conversation = SessionCacheConversation(
            max_size=2, system_message_content=self.system_context
        )

        self.model = None
        self.chat_idx = {}
        self.retrieval_table = []
        self.document_table = []
        self.long_term_memory_df = None
        self.last_recall_df = None
        self.agent = self.initialize_agent()
        self.model_name = model_name
        self.set_model(model_name)
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
        self.favicon_path = "./favicon-32x32.png"
        self._show_api_key = False
        self._show_provider_model = False
        self._show_system_context = False
        self._show_documents_tab = False
        self._init_file_path = None

    def initialize_agent(self):
        VS = Doc2VecVectorStore()
        agent = RagAgent(
            name="Rag",
            system_context=self.system_context,
            model=self.model,
            conversation=self.conversation,
            vector_store=VS,
        )
        return agent

    def set_model(self, provider_model_choice: str):
        if provider_model_choice in self.allowed_models:
            provider, model_name = provider_model_choice.split("_")
            if provider == "groq":
                self.model = GroqModel(api_key=self.api_key, model_name=model_name)

            if provider == "mistral":
                self.model = MistralModel(api_key=self.api_key, model_name=model_name)

            if provider == "openai":
                self.model = OpenAIModel(api_key=self.api_key, model_name=model_name)

            if provider == "google":
                self.model = GeminiProModel(api_key=self.api_key, model_name=model_name)

            if provider == "anthropic":
                self.model = AnthropicModel(api_key=self.api_key, model_name=model_name)

            self.agent.model = self.model

            self.model_name = provider_model_choice

        else:
            raise ValueError(
                f"Model name '{model_name}' is not supported. Choose from {self.allowed_models}"
            )

    def change_vectorizer(self, vectorizer: str):
        if vectorizer == "Doc2Vec":
            self.agent.vector_store = Doc2VecVectorStore()
        if vectorizer == "MLM":
            self.agent.vector_store = MlmVectorStore()
        else:
            self.agent.vector_store = TfidfVectorStore()

    def load_json_from_file_info(self, file_info):
        self._load_and_filter_json(file_info.name)

    def _load_and_filter_json(self, filename):
        # Load JSON file using json library
        try:
            documents = load_documents_from_json_file(filename)
            self.agent.vector_store.documents = []
            self.agent.vector_store.add_documents(documents)
            self.long_term_memory_df = self.preprocess_documents(documents)
            return self.long_term_memory_df
        except json.JSONDecodeError:
            # error_fn("Invalid JSON file. Please check the file and try again.")
            return "Invalid JSON file. Please check the file and try again."

    def preprocess_documents(self, documents):
        try:
            docs = [d.to_dict() for d in documents]
            for d in docs:
                metadata = d["metadata"]
                for each in metadata:
                    d[each] = metadata[each]
                del d["metadata"]
                del d["type"]
                del d["embedding"]
            df = pd.DataFrame.from_dict(docs)
            return df
        except Exception as e:
            # error_fn("preprocess_documents failed: {e}")
            logging.info(f"preprocess_documents: {e}")

    def save_df(self, df):
        documents = self.dataframe_to_documents(df)
        self.agent.vector_store.documents = []
        self.agent.vector_store.add_documents(documents)
        self.long_term_memory_df = self.preprocess_documents(documents)
        return self.long_term_memory_df

    def dataframe_to_documents(self, df):
        documents = []
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            id = row_dict.pop("id", "")
            content = row_dict.pop("content", "")
            metadata = row_dict  # remaining data becomes metadata

            # Convert the row to dictionary and create a DocumentBase instance

            document = Document(id=id, content=content, metadata=metadata)
            documents.append(document)
        return documents

    def clear_chat(self, chat_id):
        # We could clear, but if we create a new instance, we get a new conversation id
        try:
            del self.chat_idx[chat_id]
        except KeyError:
            pass
        return chat_id, "", [], []

    async def chatbot_function(
        self,
        chat_id,
        message,
        api_key: str = None,
        model_name: str = None,
        system_context: str = None,
        fixed_retrieval: bool = True,
        top_k: int = 5,
        temperature: int = 1,
        max_tokens: int = 256,
        conversation_size: int = 2,
        session_cache_size: int = 2,
    ):
        try:
            if not chat_id:
                chat_id = str(uuid.uuid4())
            start_datetime = datetime.now()

            # Set additional parameters
            self.api_key = api_key
            self.set_model(model_name)

            # Set Conversation Size
            self.agent.system_context = system_context

            if chat_id not in self.chat_idx:
                self.chat_idx[chat_id] = SessionCacheConversation(
                    max_size=conversation_size,
                    system_message_content=system_context,
                    session_cache_max_size=session_cache_size,
                )

            self.agent.conversation = self.chat_idx[chat_id]

            # Predict
            try:
                response = self.agent.exec(
                    message,
                    top_k=top_k,
                    fixed=fixed_retrieval,
                    model_kwargs={"temperature": temperature, "max_tokens": max_tokens},
                )
            except Exception as e:
                logging.info(f"chatbot_function agent error: {e}")

            # Update Retrieval Document Table
            self.last_recall_df = self.preprocess_documents(self.agent.last_retrieved)

            # Get History

            history = [
                each["content"] for each in self.agent.conversation.session_to_dict()
            ]
            history = [(history[i], history[i + 1]) for i in range(0, len(history), 2)]

            # SQL Log
            end_datetime = datetime.now()
            sql_log(
                self.agent.conversation.id,
                model_name,
                message,
                response,
                start_datetime,
                end_datetime,
            )

            return chat_id, "", self.last_recall_df, history
        except Exception as e:
            # error_fn(f"chatbot_function error: {e}")
            self.agent.conversation._history.pop(0)
            logging.info(f"chatbot_function error: {e}")
            return chat_id, "", [], history
