import gradio as gr


class DocumentEditsTab:
    def __init__(self, assistant):
        self.assistant = assistant
        self.documents = []

        self.file_upload = None
        self.content_display = None
        self.save_button = None

    def document_edit_tab(self):
        """Build the UI to load, edit, and save different file types."""
        with gr.Blocks(css=self.assistant.css) as self.retrieval_interface:
            self._doc_upload()
            self._doc_editor()
            self._save_button()
            self._doc_event_handler()

    # -------------------------------------------------- COMPONENTS ------------------------------------------------------

    def _doc_upload(self):
        with gr.Row():
            self.file_upload = gr.File(
                label="Upload File (CSV, JSON, TXT)",
            )

    def _doc_editor(self):
        with gr.Row():
            self.content_display = gr.Textbox(
                label="Edit Document",
                placeholder="Edit the document",
                type="text",
                lines=20,
                interactive=True,
                visible=False,
            )

    def _save_button(self):
        with gr.Row():
            self.save_button = gr.Button("Update & Upload")

    # -------------------------------------------------- HANDLERS ------------------------------------------------------
    def _doc_event_handler(self):
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
            outputs=[self.content_display],
        )

    def _on_file_upload(self, file):
        """Handle file upload and show appropriate editor."""
        content = ""
        self.file = file
        print("File: ", file)
        with open(file, "r") as f:
            content = f.read()

        if isinstance(content, str):  # TXT case
            self.documents.append(content)
            return gr.update(value=content, visible=True)

        else:
            return "Unsupported file type."

    def _on_update_and_upload(self, content):
        """Handle saving edits based on the file type."""
        self.assistant.add_to_vector_store(content)
        gr.Info("Successfully updated and added to store")
        return gr.update(value="", visible=False)
