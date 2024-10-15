import os

from gradio_ui import Gradio_UI


api_key = os.getenv("OPENAI_API_KEY")

# app
app = Gradio_UI(api_key=api_key, llm="openai")
app.launch()
