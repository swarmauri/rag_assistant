# Swarmauri Rag Assistant

## Overview

The Swarmauri Rag Assistant is your go-to tool for managing different configurations and settings for your application. With a simple command-line interface, you can customize and deploy the Assistant using various arguments according to your needs.

## Installation

Getting started with Swarmauri’s RAG Assistant is straightforward. You can install it via pip by running the following command:

```bash
pip install rag_assistant==0.1.19 --user
```

## Usage

Below is a comprehensive explanation of each argument you can provide to the Assistant and how to use them effectively:

### Required Arguments

**--api_key**

- **Description**: Your API key.
- **Type**: `str`
- **Required**: Yes
- **Usage**:
    ```bash
    rag_assistant --api_key your_api_key
    ```

### Optional Arguments

**--show_api_key**

- **Description**: Toggle displaying API key on the app.
- **Type**: `bool`
- **Default**: `False`
- **Usage**:
    ```bash
    rag_assistant --api_key your_api_key --show_api_key True
    ```

**--provider_model**

- **Description**: Your provider model.
- **Type**: `str`
- **Usage**:
    ```bash
    rag_assistant --api_key your_api_key --provider_model your_provider_model
    ```

**--show_provider_model**

- **Description**: Toggle displaying the provider model on the app.
- **Type**: `bool`
- **Default**: `False`
- **Usage**:
    ```bash
    rag_assistant --api_key your_api_key --show_provider_model True
    ```

**--system_context**

- **Description**: Assistant’s system context.
- **Type**: `str`
- **Usage**:
    ```bash
    rag_assistant --api_key your_api_key --system_context "Your system context"
    ```

**--show_system_context**

- **Description**: Toggle displaying the system context on the app.
- **Type**: `bool`
- **Default**: `False`
- **Usage**:
    ```bash
    rag_assistant --api_key your_api_key --show_system_context True
    ```

**--documents_file_path**

- **Description**: Filepath of documents JSON.
- **Type**: `str`
- **Usage**:
    ```bash
    rag_assistant --api_key your_api_key --documents_file_path "path/to/your/documents.json"
    ```

**--show_documents_tab**

- **Description**: Toggle displaying document tabs on the app.
- **Type**: `bool`
- **Default**: `False`
- **Usage**:
    ```bash
    rag_assistant --api_key your_api_key --show_documents_tab True
    ```

**--db_path**

- **Description**: Path to SQLite3 database.
- **Type**: `str`
- **Usage**:
    ```bash
    rag_assistant --api_key your_api_key --db_path "path/to/your/db.sqlite3"
    ```

**--share**

- **Description**: Deploy a live app on Gradio.
- **Type**: `bool`
- **Default**: `False`
- **Usage**:
    ```bash
    rag_assistant --api_key your_api_key --share True
    ```

**--server_name**

- **Description**: Server name.
- **Type**: `str`
- **Usage**:
    ```bash
    rag_assistant --api_key your_api_key --server_name "your_server_name"
    ```

## Example Usage

Here is an example command that utilizes multiple arguments to launch the Assistant:

```bash
rag_assistant --api_key your_api_key --show_api_key True \
--provider_model your_provider_model --show_provider_model True \
--system_context "Development Environment" --show_system_context True \
--documents_file_path "path/to/your/documents.json" --show_documents_tab True \
--db_path "path/to/your/db.sqlite3" --share True --server_name "your_server_name"
```
