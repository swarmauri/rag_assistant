import os

import gradio as gr
from typing import Optional
import logging

from swarmauri.documents.concrete.Document import Document
from RagAssistant import RagAssistant


head = """<link rel="icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAAXNSR0IArs4c6QAABWFJREFUWEe1V2tsFFUU/s7M7KO77Ra2LRisUArasLjbLqVdaRfQkACCNISgUQJCiCHGfyaYEIMSfIQY/K3GEAUkqNHwUJ6JEoht7ZZuu92uaCktILUJpSy0233PzNU7SClNu9MWvX8ms+ec7/vOuefMvUsYxypxu2dYgHRLS8utcbhjrttdIMJiaG+p69HzJz0Hl8dTKMLYwRiIMeW11ov1hzLFlFZUbSQS93EfNZGaGwz6ujP56wpwOCoeM2abrhOREUBCluXKNv+vbaOBOssXOSVJagRgZgxJOZaaFQr5bj6SAB7sqvS+KoA+J4KJqawucLHWOxpoWaW3joiqOLmqsm3BptqDehXWrcB9gPlub6lkwGcEeAC8xxgjgmDhdgY1xp9E9A7AGmRZfj3U7AvqkWsx43G67+NaWLVeFMVvAIhjxCmKorwcbKr/fry4ExJQ5vFeI9CsTOCMsauBxtri/1yAy+WyCmZbhIgyimaMsd7ueHZPj1/bFr01gQqUG9yerBhAUmZQJrfIcQv8/rQe+YR7oKzS20ZET+tsQSjQWOscD7muAN7p5wOBXAuZp0GQ8g4fPrgjMhCpUWQZ4XAYf3Xf+8Y8XlgIu90OUZJgs9mOv7Jh00dGSehTTOgtLy4eICI2liBtC862tlrzJOtWEK0gYDZAFkbMAgYbANPwaWGMIRpPIJVKI3zrpjZH9vzpMBoNsGaZ+SgO5+LESRAGiFEMYDEGXCWiM32pyJcrSkuj1NDaUSgahJ8JeCpT2RRFwZ2BCG6F7yIWT2iuT0yza88bvWHtackyo8A+BVNtORDFsSZ1iKU9lZKXUdOlK98C9NJo5IqiIhKNItw/gP7IIFT1XiVzrBZMy5uKntbvtIxnuF7EzdthRKL3Gl8QCLk52bDn2pCTbYUoCKPmxsC+oabfO/t5qXlpeZaxRBLRWFwDG4zFwX/nyyBJGmC+PRdmkwn1F85i3ewTmu3otTVYtGQ5Eskk+sL9muC0LGs2LjDbmoUciwXZlixkmU1adbStYrhLvlBH/HLXdXMilYaqqg8pNRoMyM2xYootR8uaB3FBZ06dgDfvDJ51GTT/88E06u+swvKVq4Z8eAJ3ByLoj0SRSj88kYIgwGw0oGROUYIaQx117V1/VimqAk5oNhm1veSE/H14U8XjcXy1fz8W5Aewrpr35oN1pC6JQNiNjZu3wGw2Dxm4YC4gMhhDLJFAIpnS3kVBREnxrHpq/qOrRlXZsUznAgf5LRTCoQMHcPt2X6ZeRX5BATZt3oJ5DsfIiRgZx5iqrtVmpulS5xsA7QWYdroNX52dV3Dyhx/RFmzNSDzS6Corwws1NSgqmj2aEN6t2xc65nw6NLQNbV3TRUFdCyInAWZFUbs+2L3r3e4bNx6u9YRkADNnFiV37t71PhgVCwL/FqAtLUePP+N0aheVjGfB1g1L/YudxgXi6FOkK0VRgV+CqZYvvr6wYCznjAKSDav3GiVhuy5TBgdZVT82VJ58a1ICButWlVlNQjOf5smJYCwWVcqtS0+3TEoAD0o1rD5nkITnJiNAVtg5g+fEskyxupm5K6sXgoSGDNewsfAVQsrT7PP5H0kADy6tqN4jCMKOiVRBYezDYGPtTr0Y3Qr8CyCWVniPCgKt0QPkdsbYsUBj7XoAip7/eAXA4XAYXfMK9lU8KW1kY9wLiTHm75APtF8b3Ob/P65kPJven55fb7MIu0xGmv9gOhhLy2iLJtnuqUtPHtHLerh93BUYCdp1emXJJ8eS5xlAb9ZkLSlcferyRIjv+05aAAco8yz28UM94Kvl/5YmtR5JgKuy+m0wsODFuj2TYv8n6G+wcRzTJW9piwAAAABJRU5ErkJggg==" type="image/png">"""


# Assuming the RagAssistant class has been defined/imported as provided.
class RagAssistantUI:
    def __init__(self, api_key: str):
        """Initialize the RagAssistant instance."""
        self.assistant = RagAssistant(api_key=api_key, llm="openai")
        self.chat_history = []

    # === Reusable Gradio Components === #
    def build_chat_interface(self) -> gr.Chatbot:
        """Creates a Gradio chatbot component."""
        return gr.Chatbot(
            label="Chat with RagAssistant", elem_id="chat-dialogue-container"
        )

    def build_text_input(self) -> gr.Textbox:
        """Creates a Gradio input textbox for user queries."""
        return gr.Textbox(placeholder="Type your message...", label="Your Message")

    def build_document_uploader(self) -> gr.File:
        """Creates a file uploader for adding documents."""
        return gr.File(label="Upload Documents", file_types=[".json"])

    def build_status_display(self) -> gr.Label:
        """Creates a status label to display system updates."""
        return gr.Label(label="Status", value="Welcome! Ready to chat.")

    def build_buttons(self) -> tuple:
        """Creates submit and clear buttons."""
        submit_button = gr.Button(value="Submit", variant="primary")
        clear_button = gr.Button(value="Clear Chat", variant="secondary")
        return submit_button, clear_button

    # === Functional Methods === #
    def handle_message(self, message: str) -> tuple:
        """Handle user messages and return the chat response."""
        response = self.assistant.agent.exec(input_data=message)
        self.chat_history.append((message, response))
        return self.chat_history, ""

    def upload_documents(self, files: Optional[list]) -> str:
        """Upload documents to the assistant's vector store."""
        if not files:  # Handle None or empty list
            return "No files selected for upload."

        for file in files:
            document = Document(content=file)
            self.assistant.vectorstore.add_document(document=document)
        return f"{len(files)} documents uploaded successfully."

    def clear_chat(self) -> list:
        """Clear the chat history."""
        self.chat_history = []
        return self.chat_history

    # === Gradio Interface Layout === #
    def build_interface(self):
        """Build the complete Gradio interface using reusable components."""
        with gr.Blocks(css=self.assistant.css) as interface:
            with gr.Row():
                gr.HTML(head)
                gr.Markdown("# RagAssistant Chatbot")
            with gr.Row():
                chatbox = self.build_chat_interface()
                status_display = self.build_status_display()

            with gr.Row():
                text_input = self.build_text_input()
                submit_button, clear_button = self.build_buttons()

            with gr.Row():
                document_uploader = self.build_document_uploader()

            # === Event Handling === #
            submit_button.click(
                fn=self.handle_message,
                inputs=[text_input],
                outputs=[chatbox, text_input],
            )

            clear_button.click(fn=self.clear_chat, outputs=[chatbox])

            # document_uploader.upload(fn=self.upload_documents, outputs=[status_display])
            # === Event Handling Update === #
            document_uploader.upload(
                fn=self.upload_documents,
                inputs=[document_uploader],
                outputs=[status_display],
            )
        return interface


# === Running the Interface === #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    api_key = os.getenv("OPENAI_API_KEY")  # Replace with your actual API key
    ui = RagAssistantUI(api_key=api_key)
    interface = ui.build_interface()
    interface.launch()
