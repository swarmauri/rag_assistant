# Swarmauri Rag Assistant

## Overview

The Swarmauri Rag Assistant is your go-to tool for managing different configurations and settings for your application. With a simple command-line interface, you can customize and deploy the Assistant using various arguments according to your needs.

## Installation

Getting started with Swarmauri’s RAG Assistant is straightforward. You can install it via pip by running the following command:

```bash
pip install rag_assistant==0.2.0 --user
```

## Usage

Below is a comprehensive explanation of each argument you can provide to the Assistant and how to use them effectively:

### Required Arguments

#### Generating config files

- For json config files

```sh
rag_assistant generate -o config.json
```

- For yaml config files

```sh
rag_assistant generate -o config.yaml
```

#### Launching app

- To launch the app:

```sh
rag_assistant launch --api_key $OPENAI_API_KEY --provider-llm openai
```

- *N/B: to see the currently supported LLMs do: `rag_assistant launch --help` under `--provider-llm`*

#### For more help on `rag_assistant`:

```sh
rag_assistant --help
```

##### Example results

```sh
Swarmauri Developer Assistant

positional arguments:
  {generate,launch}
    generate         Generate a configuration file
    launch           Launch the Gradio UI application

options:
  -h, --help         show this help message and exit
```

#### For more help on `rag_assistant generate`

```sh
rag_assistant generate --help
```
 
##### Example results

```sh
options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file path
```

#### For more help on `rag_assistant launch`

```sh
rag_assistant launch --help
```

##### Example result

```sh
options:
  -h, --help            show this help message and exit
  -api_key API_KEY, --api_key API_KEY
                        Your API key
  -provider_llm PROVIDER_LLM, --provider_llm PROVIDER_LLM
                        Your provider LLM: openai | groq | mistral | gemini | anthropic
  -provider_model PROVIDER_MODEL, --provider_model PROVIDER_MODEL
                        Your provider model
  -config_file CONFIG_FILE, --config_file CONFIG_FILE
                        Path to config file
```
