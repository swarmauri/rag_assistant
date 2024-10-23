__version__ = "0.1.23"
__short_desc__ = """The Swarmauri Rag Assistant is part of the Swarmauri framework."""
__long_desc__ = f"""# Swarmauri Rag Assistant

## Overview
The Swarmauri Rag Assistant

## Installation
```bash
pip install rag_assistant=={__version__} --user
```

## Execution
```bash
rag_assistant --api_key your_api_key --show_api_key True \
--provider_model your_provider_model --show_provider_model True \
--system_context "Development Environment" --show_system_context True \
--documents_file_path "path/to/your/documents.json" --show_documents_tab True \
--db_path "path/to/your/db.sqlite3" --share True --server_name "your_server_name"

```
"""
