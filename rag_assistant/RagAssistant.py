from datetime import datetime
import uuid
import sqlite3
import gradio as gr
import pandas as pd
import json
from typing import Any, Optional, Union, Dict
from swarmauri.standard.models.concrete.OpenAIModel import OpenAIModel
from swarmauri.standard.models.concrete.GroqModel import GroqModel
from swarmauri.standard.models.concrete.MistralModel import MistralModel
from swarmauri.standard.models.concrete.GeminiProModel import GeminiProModel
from swarmauri.standard.models.concrete.AnthropicModel import AnthropicModel
from swarmauri.standard.vector_stores.concrete.TFIDFVectorStore import TFIDFVectorStore
from swarmauri.standard.vector_stores.concrete.Doc2VecVectorStore import Doc2VecVectorStore
from swarmauri.standard.vector_stores.concrete.MLMVectorStore import MLMVectorStore
from swarmauri.standard.conversations.concrete.SessionCacheConversation import SessionCacheConversation
from swarmauri.standard.agents.concrete.RagAgent import RagAgent
from swarmauri.core.messages import IMessage
from swarmauri.core.models.IModel import IModel
from swarmauri.standard.conversations.base.SystemContextBase import SystemContextBase
from swarmauri.standard.agents.base.VectorStoreAgentBase import VectorStoreAgentBase
from swarmauri.standard.vector_stores.base.VectorDocumentStoreRetrieveBase import VectorDocumentStoreRetrieveBase
from swarmauri.standard.documents.concrete.Document import Document
from swarmauri.standard.documents.concrete.EmbeddedDocument import EmbeddedDocument
from swarmauri.standard.messages.concrete import (HumanMessage, 
                                                  SystemMessage,
                                                  AgentMessage)
from swarmauri.standard.utils.load_documents_from_json import load_documents_from_json_file
head="""<link rel="icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAAXNSR0IArs4c6QAABWFJREFUWEe1V2tsFFUU/s7M7KO77Ra2LRisUArasLjbLqVdaRfQkACCNISgUQJCiCHGfyaYEIMSfIQY/K3GEAUkqNHwUJ6JEoht7ZZuu92uaCktILUJpSy0233PzNU7SClNu9MWvX8ms+ec7/vOuefMvUsYxypxu2dYgHRLS8utcbhjrttdIMJiaG+p69HzJz0Hl8dTKMLYwRiIMeW11ov1hzLFlFZUbSQS93EfNZGaGwz6ujP56wpwOCoeM2abrhOREUBCluXKNv+vbaOBOssXOSVJagRgZgxJOZaaFQr5bj6SAB7sqvS+KoA+J4KJqawucLHWOxpoWaW3joiqOLmqsm3BptqDehXWrcB9gPlub6lkwGcEeAC8xxgjgmDhdgY1xp9E9A7AGmRZfj3U7AvqkWsx43G67+NaWLVeFMVvAIhjxCmKorwcbKr/fry4ExJQ5vFeI9CsTOCMsauBxtri/1yAy+WyCmZbhIgyimaMsd7ueHZPj1/bFr01gQqUG9yerBhAUmZQJrfIcQv8/rQe+YR7oKzS20ZET+tsQSjQWOscD7muAN7p5wOBXAuZp0GQ8g4fPrgjMhCpUWQZ4XAYf3Xf+8Y8XlgIu90OUZJgs9mOv7Jh00dGSehTTOgtLy4eICI2liBtC862tlrzJOtWEK0gYDZAFkbMAgYbANPwaWGMIRpPIJVKI3zrpjZH9vzpMBoNsGaZ+SgO5+LESRAGiFEMYDEGXCWiM32pyJcrSkuj1NDaUSgahJ8JeCpT2RRFwZ2BCG6F7yIWT2iuT0yza88bvWHtackyo8A+BVNtORDFsSZ1iKU9lZKXUdOlK98C9NJo5IqiIhKNItw/gP7IIFT1XiVzrBZMy5uKntbvtIxnuF7EzdthRKL3Gl8QCLk52bDn2pCTbYUoCKPmxsC+oabfO/t5qXlpeZaxRBLRWFwDG4zFwX/nyyBJGmC+PRdmkwn1F85i3ewTmu3otTVYtGQ5Eskk+sL9muC0LGs2LjDbmoUciwXZlixkmU1adbStYrhLvlBH/HLXdXMilYaqqg8pNRoMyM2xYootR8uaB3FBZ06dgDfvDJ51GTT/88E06u+swvKVq4Z8eAJ3ByLoj0SRSj88kYIgwGw0oGROUYIaQx117V1/VimqAk5oNhm1veSE/H14U8XjcXy1fz8W5Aewrpr35oN1pC6JQNiNjZu3wGw2Dxm4YC4gMhhDLJFAIpnS3kVBREnxrHpq/qOrRlXZsUznAgf5LRTCoQMHcPt2X6ZeRX5BATZt3oJ5DsfIiRgZx5iqrtVmpulS5xsA7QWYdroNX52dV3Dyhx/RFmzNSDzS6Corwws1NSgqmj2aEN6t2xc65nw6NLQNbV3TRUFdCyInAWZFUbs+2L3r3e4bNx6u9YRkADNnFiV37t71PhgVCwL/FqAtLUePP+N0aheVjGfB1g1L/YudxgXi6FOkK0VRgV+CqZYvvr6wYCznjAKSDav3GiVhuy5TBgdZVT82VJ58a1ICButWlVlNQjOf5smJYCwWVcqtS0+3TEoAD0o1rD5nkITnJiNAVtg5g+fEskyxupm5K6sXgoSGDNewsfAVQsrT7PP5H0kADy6tqN4jCMKOiVRBYezDYGPtTr0Y3Qr8CyCWVniPCgKt0QPkdsbYsUBj7XoAip7/eAXA4XAYXfMK9lU8KW1kY9wLiTHm75APtF8b3Ob/P65kPJven55fb7MIu0xGmv9gOhhLy2iLJtnuqUtPHtHLerh93BUYCdp1emXJJ8eS5xlAb9ZkLSlcferyRIjv+05aAAco8yz28UM94Kvl/5YmtR5JgKuy+m0wsODFuj2TYv8n6G+wcRzTJW9piwAAAABJRU5ErkJggg==" type="image/png">"""

def info_fn(msg):
    gr.Info(msg)

def error_fn(msg):
    raise gr.Error(msg)

def warning_fn(msg):
    gr.Warning(msg)

class RagAssistant:
    def __init__(self, 
                 api_key: str = "", 
                 system_context: str = "You are a helpful assistant.",
                 model_name = "openai_gpt-4o",
                 db_path='prompt_responses.db'):
        print('Initializing... this will take a moment.')
        self.system_context = system_context
        self.api_key = api_key
        self.db_path = db_path
        self.conversation = SessionCacheConversation(max_size=2, system_message_content=self.system_context)
        self.allowed_models =sorted(['groq_llama3-8b-8192',
                            'groq_llama3-70b-8192',
                            'groq_mixtral-8x7b-32768',
                            'groq_gemma-7b-it',
                            'mistral_mistral-medium-latest',
                            'mistral_mistral-small-latest',
                            'mistral_open-mixtral-8x22b',
                            'mistral_open-mixtral-8x7b',
                            'mistral_open-mistral-7b',
                            'mistral_mistral-large-latest',
                            'openai_gpt-3.5-turbo',
                            'openai_gpt-3.5-turbo-16k',
                            'openai_gpt-3.5-turbo-16k-0613',
                            'openai_gpt-4-0125-preview',
                            'openai_gpt-4o',
                            'google_gemini-1.5-pro-latest',
                            'anthropic_claude-2.0',
                            'anthropic_claude-instant-1.2',
                            'anthropic_claude-2.1',
                            'anthropic_claude-3-opus-20240229',
                            'anthropic_claude-3-sonnet-20240229',
                            'anthropic_claude-3-haiku-20240307'])
            
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
        agent = RagAgent(name="Rag", 
                         system_context=self.system_context, 
                         model=self.model, 
                         conversation=self.conversation, 
                         vector_store=VS)
        return agent

    def set_model(self, provider_model_choice: str):
        if provider_model_choice in self.allowed_models:
            provider, model_name = provider_model_choice.split('_')
            if provider == 'groq':
                self.model = GroqModel(api_key=self.api_key, model_name=model_name)
                
            if provider == 'mistral':
                self.model = MistralModel(api_key=self.api_key, model_name=model_name)
                
            if provider == 'openai':
                self.model = OpenAIModel(api_key=self.api_key, model_name=model_name)

            if provider == 'google':
                self.model = GeminiProModel(api_key=self.api_key, model_name=model_name)

            if provider == 'anthropic':
                self.model = AnthropicModel(api_key=self.api_key, model_name=model_name)
            
            self.agent.model = self.model

            self.model_name = provider_model_choice
                
        else:
            raise ValueError(f"Model name '{model_name}' is not supported. Choose from {self.allowed_models}")

    def change_vectorizer(self, vectorizer: str):
        if vectorizer == 'Doc2Vec':
            self.agent.vector_store = Doc2VecVectorStore()
        if vectorizer == 'MLM':
            self.agent.vector_store = MLMVectorStore()
        else:
            self.agent.vector_store = TFIDFVectorStore()
    
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
            error_fn("Invalid JSON file. Please check the file and try again.")
            return "Invalid JSON file. Please check the file and try again."

    def preprocess_documents(self, documents):
        try:
            docs = [d.to_dict() for d in documents]
            for d in docs:
                metadata = d['metadata']
                for each in metadata:
                    d[each] = metadata[each]
                del d['metadata']
                del d['type']
                del d['embedding']
            df = pd.DataFrame.from_dict(docs)
            return df
        except Exception as e:
            error_fn("preprocess_documents failed: {e}")
            print(f"preprocess_documents: {e}")

    
    def sql_log(self, conversation_id, model_name, prompt, response, start_datetime, end_datetime):
        try:
            duration = (end_datetime - start_datetime).total_seconds()
            start_datetime = start_datetime.isoformat()
            end_datetime = end_datetime.isoformat()
            conversation_id = str(conversation_id)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS conversations
                            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                            conversation_id TEXT, 
                            model_name TEXT, 
                            prompt TEXT, 
                            response TEXT, 
                            start_datetime TEXT, 
                            end_datetime TEXT,
                            duration NUMERIC)''')
            cursor.execute('''INSERT INTO conversations (
                            conversation_id, 
                            model_name, 
                            prompt, 
                            response, 
                            start_datetime, 
                            end_datetime,
                            duration) VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                           (conversation_id, 
                            model_name, 
                            prompt, 
                            response, 
                            start_datetime, 
                            end_datetime, 
                            duration))
            conn.commit()
            conn.close()
        except:
            raise
    
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
            
            document = EmbeddedDocument(id=id, content=content, metadata=metadata)
            documents.append(document)
        return documents
        
    def clear_chat(self, chat_id):
        # We could clear, but if we create a new instance, we get a new conversation id
        del self.chat_idx[chat_id]
        return chat_id, "", [], []    
        
    async def chatbot_function(self, 
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
                         session_cache_size: int = 2):
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
                self.chat_idx[chat_id] = SessionCacheConversation(max_size=conversation_size, 
                                                     system_message_content=system_context, 
                                                     session_cache_max_size=session_cache_size)


            self.agent.conversation = self.chat_idx[chat_id]
            
            # Predict
            try:
                response = self.agent.exec(message, 
                                           top_k=top_k, 
                                           fixed=fixed_retrieval,
                                           model_kwargs={'temperature': temperature, 'max_tokens': max_tokens})
            except Exception as e:
                print(f"chatbot_function agent error: {e}")
                

            # Update Retrieval Document Table
            self.last_recall_df = self.preprocess_documents(self.agent.last_retrieved)
            

            # Get History

            history = [each['content'] for each in self.agent.conversation.session_to_dict()]
            history = [(history[i], history[i+1]) for i in range(0, len(history), 2)]

            # SQL Log
            end_datetime = datetime.now()
            self.sql_log(self.agent.conversation.id, model_name, message, response, start_datetime, end_datetime)
            
            return chat_id, "", self.last_recall_df, history
        except Exception as e:
            error_fn(f"chatbot_function error: {e}")
            self.agent.conversation._history.pop(0)
            print(f"chatbot_function error: {e}")
            return chat_id, "", [], history
    

        
    def setup_gradio_interface(self):
        with gr.Blocks(css = self.css) as self.retrieval_table:
            with gr.Row():
                self.retrieval_table = gr.Dataframe(interactive=False, wrap=True, line_breaks=True, elem_id="document-table-container", height="800")

        
        with gr.Blocks(css = self.css) as self.chat:
            with gr.Row():
                chat_id = gr.State(None)
                self.chatbot = gr.Chatbot(label="Chat History", 
                                           layout="panel", 
                                           elem_id="chat-dialogue-container", 
                                           container=True, 
                                           show_copy_button=True,
                                           height="70vh")
            with gr.Row():
                self.input_box = gr.Textbox(label="Type here:", scale=6)
                self.send_button = gr.Button("Send", scale=1)
                self.clear_button = gr.Button("Clear", scale=1)
                
            with gr.Accordion("See Details", open=False):
                self.additional_inputs = [
                    gr.Textbox(label="API Key", value=self.api_key or "Enter your API Key", visible=self._show_api_key),
                    gr.Dropdown(self.allowed_models, 
                                value=self.model_name, 
                                label="Model",
                                info="Select openai model",
                                visible=self._show_provider_model),
                    gr.Textbox(label="System Context", value = self.system_context, visible=self._show_system_context),
                    gr.Checkbox(label="Fixed Retrieval", value=True, interactive=True),
                    gr.Slider(label="Top K", value=10, minimum=0, maximum=100, step=5, interactive=True),
                    gr.Slider(label="Temperature", value=1, minimum=0.0, maximum=1, step=0.01, interactive=True),
                    gr.Slider(label="Max new tokens", value=256, minimum=256, maximum=4096, step=64, interactive=True),
                    gr.Slider(label="Conversation size", value=12, minimum=2, maximum=36, step=2, interactive=True),
                    gr.Slider(label="Session Cache size", value=12, minimum=2, maximum=36, step=2, interactive=True)
                ]
    
    
            submit_inputs = [chat_id, self.input_box]
            submit_inputs.extend(self.additional_inputs)
            # Function to handle sending messages
            self.send_button.click(
                self.chatbot_function, 
                inputs=submit_inputs, 
                outputs=[chat_id, self.input_box, self.retrieval_table, self.chatbot]
            )
        
            # Function to handle clearing the chat
            self.clear_button.click(
                self.clear_chat, 
                inputs=[chat_id], 
                outputs=[chat_id, self.input_box, self.retrieval_table, self.chatbot]
            )

        
        with gr.Blocks(css = self.css) as self.document_table:
            with gr.Row():
                self.file = gr.File(label="Upload JSON File", value=self._init_file_path)
                self.vectorizer = gr.Dropdown(choices=["Doc2Vec", "TFIDF", "MLM"], value="Doc2Vec", label="Select vectorizer")
                self.load_button = gr.Button("load")
            with gr.Row():
                if self._init_file_path:
                    df = self._load_and_filter_json(self._init_file_path)
                self.data_frame = gr.Dataframe(interactive=True, 
                    wrap=True, 
                    line_breaks=True, 
                    elem_id="document-table-container", 
                    height="700", 
                    value=df)
            with gr.Row():
                self.save_button = gr.Button("save")
                
            self.vectorizer.change(self.change_vectorizer, inputs=[self.vectorizer], outputs=self.data_frame)
            self.load_button.click(self.load_json_from_file_info, inputs=[self.file], outputs=self.data_frame)
            self.save_button.click(self.save_df, inputs=[self.data_frame])

        with gr.Blocks(css = self.css, title="Swarmauri Rag Agent", head=head) as self.app:
            

            with gr.Tab("chat", visible=True):
                self.chat.render()
            with gr.Tab("retrieval", visible=self._show_documents_tab):
                self.retrieval_table.render()
            with gr.Tab("documents", visible=self._show_documents_tab):
                self.document_table.render()
    
    def launch(self, 
        share: bool = False, 
        show_api_key: bool = False,
        show_provider_model: bool = False,
        show_system_context: bool = False,
        show_documents_tab: bool = False,
        server_name: Optional[str] = None,
        documents_file_path: Optional[str] = None,
        ):


        # Create interfaces
        self._show_api_key = show_api_key
        self._show_provider_model = show_provider_model
        self._show_system_context = show_system_context
        self._show_documents_tab = show_documents_tab
        self._init_file_path = documents_file_path
        self.setup_gradio_interface()


        # Deploy interfaces
        kwargs = {}
        kwargs.update({'share': share})
        if server_name:
            kwargs.update({'server_name': server_name})
        #kwargs.update({'favicon_path': self.favicon_path})

        self.app.launch(**kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Swarmauri Developer Assistant Command Line Tool")
    parser.add_argument('-api_key', '--api_key', type=str, help='Your api key', required=True)
    parser.add_argument('-show_api_key', '--show_api_key', 
        type=bool, help='Toggle displaying api key on app', default=False, required=False)

    parser.add_argument('-provider_model', '--provider_model', type=str, help='Your provider model', required=False)
    parser.add_argument('-show_provider_model', '--show_provider_model', 
        type=bool, help='Toggle displaying Provider Model on app', default=False, required=False)

    parser.add_argument('-system_context', '--system_context', type=str, help='Assistants System Context', required=False)
    parser.add_argument('-show_system_context', '--show_system_context', 
        type=bool, help='Toggle displaying System Context on app', default=False, required=False)

    parser.add_argument('-documents_file_path', '--documents_file_path', type=str, help='Filepath of Documents JSON', required=False)
    parser.add_argument('-show_documents_tab', '--show_documents_tab', 
        type=bool, help='Toggle displaying Document Tabs on app', default=False, required=False)

    parser.add_argument('-db_path', '--db_path', type=str, help='path to sqlite3 db', required=False)

    parser.add_argument('-share', '--share', type=bool, help='Deploy a live app on gradio', default=False, required=False)
    parser.add_argument('-server_name', '--server_name', type=str, help='Server name', required=False)
    #parser.add_argument('-favicon_path', '--favicon_path', type=str, help='Path of application favicon', required=False)

    args = parser.parse_args()


    api_key = args.api_key
    

    # Create Assistant
    assistant = RagAssistant(api_key=api_key)

    # If params then modify Assistant's config related to model
    if args.provider_model:
        assistant.set_model(args.provider_model) 

    # If params then modify Assistant's config related to agent
    if args.system_context:
        assistant.system_context = args.system_context


    # If params then modify Assistant's config related to logging
    if args.db_path:
        assistant.db_path = args.db_path

    # If params then modify Assistant's config
    launch_kwargs = {}
    if args.share:
        launch_kwargs.update({'share': args.share})
    if args.server_name:
        launch_kwargs.update({'server_name': args.server_name})

    if args.show_api_key:
        launch_kwargs.update({'show_api_key': args.show_api_key})
    if args.show_provider_model:
        launch_kwargs.update({'show_provider_model': args.show_provider_model})
    if args.show_system_context:
        launch_kwargs.update({'show_system_context': args.show_system_context})

    if args.documents_file_path:
        launch_kwargs.update({'documents_file_path': args.documents_file_path})

    if args.show_documents_tab == True:
        launch_kwargs.update({'show_documents_tab': args.show_documents_tab})


    #if args.favicon_path:
        #launch_kwargs.update({'favicon_path': args.favicon_path})
    #else:
        #launch_kwargs.update({'favicon_path': "favicon-32x32.png"})


    assistant.initialize_agent()
    assistant.launch(**launch_kwargs)
        
if __name__ == "__main__":
    main()
