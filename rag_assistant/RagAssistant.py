from datetime import datetime
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
        self.retrieval_table = []
        self.document_table = []
        self.long_term_memory_df = None
        self.last_recall_df = None
        self.agent = self.initialize_agent()
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
        self.setup_gradio_interface()
        
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
                
        else:
            raise ValueError(f"Model name '{model_name}' is not supported. Choose from {self.allowed_models}")

    def change_vectorizer(self, vectorizer: str):
        if vectorizer == 'Doc2Vec':
            self.agent.vector_store = Doc2VecVectorStore()
        if vectorizer == 'MLM':
            self.agent.vector_store = MLMVectorStore()
        else:
            self.agent.vector_store = TFIDFVectorStore()
            
    
    def load_and_filter_json(self, file_info):
        # Load JSON file using json library
        try:
            documents = load_documents_from_json_file(file_info.name)
            self.agent.vector_store.documents = []
            self.agent.vector_store.add_documents(documents)
            self.long_term_memory_df = self.preprocess_documents(documents)
            return self.long_term_memory_df
        except json.JSONDecodeError:
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
        
    def clear_chat(self):
        # We could clear, but if we create a new instance, we get a new conversation id
        max_size = self.agent.conversation.max_size
        session_cache_size = self.agent.conversation.session_cache_max_size
        self.conversation = SessionCacheConversation(max_size=max_size, 
                                                     system_message_content="", 
                                                     session_cache_max_size=session_cache_size)
        self.agent.conversation = self.conversation
        return "", [], []    
        
    async def chatbot_function(self, 
                         message, 
                         history, 
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
            start_datetime = datetime.now()
            if self.agent.vector_store.document_count() == 0:
                return "", [], [(message, "⚠️ Add Documents First")]
            else:

                
                
                # Set additional parameters
                self.api_key = api_key
                self.set_model(model_name)
                #print(self.model, self.model.model_name, self.api_key)

                # Set Conversation Size
                self.agent.system_context = system_context
                self.agent.conversation.max_size = conversation_size
                self.agent.conversation.session_cache_max_size = session_cache_size
                
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
                
                return "", self.last_recall_df, history
        except Exception as e:
            gr.Error(f"{e}")
            self.agent.conversation._history.pop(0)
            print(f"chatbot_function error: {e}")
            return "", [], history
    

        
    def setup_gradio_interface(self):
        with gr.Blocks(css = self.css) as self.retrieval_table:
            with gr.Row():
                self.retrieval_table = gr.Dataframe(interactive=False, wrap=True, line_breaks=True, elem_id="document-table-container", height="800")

        
        with gr.Blocks(css = self.css) as self.chat:
            with gr.Row():
                self.chat_history = gr.Chatbot(label="Chat History", 
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
                    gr.Textbox(label="API Key", value=self.api_key or "Enter your API Key"),
                    gr.Dropdown(self.allowed_models, 
                                value="openai_gpt-3.5-turbo", 
                                label="Model",
                                info="Select openai model"),
                    gr.Textbox(label="System Context", value=self.system_context or "You are a helpful assistant"),
                    gr.Checkbox(label="Fixed Retrieval", value=True, interactive=True),
                    gr.Slider(label="Top K", value=10, minimum=0, maximum=100, step=5, interactive=True),
                    gr.Slider(label="Temperature", value=1, minimum=0.0, maximum=1, step=0.01, interactive=True),
                    gr.Slider(label="Max new tokens", value=256, minimum=256, maximum=4096, step=64, interactive=True),
                    gr.Slider(label="Conversation size", value=12, minimum=2, maximum=36, step=2, interactive=True),
                    gr.Slider(label="Session Cache size", value=12, minimum=2, maximum=36, step=2, interactive=True)
                ]
    
    
            submit_inputs = [self.input_box, self.chat_history]
            submit_inputs.extend(self.additional_inputs)
            # Function to handle sending messages
            self.send_button.click(
                self.chatbot_function, 
                inputs=submit_inputs, 
                outputs=[self.input_box, self.retrieval_table, self.chat_history]
            )
        
            # Function to handle clearing the chat
            self.clear_button.click(
                self.clear_chat, 
                inputs=[], 
                outputs=[self.input_box, self.retrieval_table, self.chat_history]
            )

        
        with gr.Blocks(css = self.css) as self.document_table:
            with gr.Row():
                self.file = gr.File(label="Upload JSON File")
                self.vectorizer = gr.Dropdown(choices=["Doc2Vec", "TFIDF", "MLM"], value="Doc2Vec", label="Select vectorizer")
                self.load_button = gr.Button("load")
            with gr.Row():
                self.data_frame = gr.Dataframe(interactive=True, wrap=True, line_breaks=True, elem_id="document-table-container", height="700")
            with gr.Row():
                self.save_button = gr.Button("save")
                
            self.vectorizer.change(self.change_vectorizer, inputs=[self.vectorizer], outputs=self.data_frame)
            self.load_button.click(self.load_and_filter_json, inputs=[self.file], outputs=self.data_frame)
            self.save_button.click(self.save_df, inputs=[self.data_frame])

        with gr.Blocks(css = self.css, title="Swarmauri Rag Agent") as self.app:
            gr.TabbedInterface(interface_list=[self.chat, self.retrieval_table, self.document_table], 
                                      tab_names=["chat", "retrieval", "documents"])

    
    def launch(self, 
        share: bool = False, 
        server_name: Optional[str] = None
        #favicon_path: Optional[str] = None
        ):

        kwargs = {}
        kwargs.update({'share': share})
        if server_name:
            kwargs.update({'server_name': server_name})

        kwargs.update({'favicon_path': self.favicon_path})

        self.app.launch(**kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Swarmauri Developer Assistant Command Line Tool")
    parser.add_argument('-api_key', '--api_key', type=str, help='Your api key', required=True)
    parser.add_argument('-provider_model', '--provider_model', type=str, help='Your provider model', required=False)
    parser.add_argument('-system_context', '--system_context', type=str, help='Assistants System Context', required=False)
    parser.add_argument('-db_path', '--db_path', type=str, help='path to sqlite3 db', required=False)
    parser.add_argument('-share', '--share', type=bool, help='Deploy a live app on gradio', required=False)
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
    #if args.favicon_path:
        #launch_kwargs.update({'favicon_path': args.favicon_path})
    #else:
        #launch_kwargs.update({'favicon_path': "favicon-32x32.png"})


    assistant.initialize_agent()
    assistant.launch(**launch_kwargs)
        
if __name__ == "__main__":
    main()
