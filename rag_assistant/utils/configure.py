from rag_assistant.RagAssistant import RagAssistant
import argparse
from typing import Dict, Any
import os
import yaml
import json


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swarmauri Developer Assistant Command Line Tool"
    )
    parser.add_argument(
        "-config_file",
        "--config_file",
        type=str,
        help="Path to config file",
        default=None,
    )
    parser.add_argument(
        "-api_key", "--api_key", type=str, help="Your API key", required=True
    )
    parser.add_argument(
        "-show_api_key",
        "--show_api_key",
        type=bool,
        help="Toggle displaying API key on app",
        default=True,
    )
    parser.add_argument(
        "-provider_llm",
        "--provider_llm",
        type=str,
        required=True,
        help=f"Your provider LLM: {' | '.join(RagAssistant.available_llms.keys())}",
    )
    parser.add_argument(
        "-provider_model", "--provider_model", type=str, help="Your provider model"
    )
    parser.add_argument(
        "-show_provider_model",
        "--show_provider_model",
        type=bool,
        help="Toggle displaying Provider Model on app",
        default=True,
    )
    parser.add_argument(
        "-system_context",
        "--system_context",
        type=str,
        help="Assistant's System Context",
    )
    parser.add_argument(
        "-show_system_context",
        "--show_system_context",
        type=bool,
        help="Toggle displaying System Context on app",
        default=True,
    )
    parser.add_argument(
        "-documents_file_path",
        "--documents_file_path",
        type=str,
        help="Filepath of Documents JSON",
    )
    parser.add_argument(
        "-show_documents_tab",
        "--show_documents_tab",
        type=bool,
        help="Toggle displaying Document Tabs on app",
        default=True,
    )
    parser.add_argument("-db_path", "--db_path", type=str, help="Path to sqlite3 db")
    parser.add_argument(
        "-share",
        "--share",
        type=bool,
        help="Deploy a live app on gradio",
        default=False,
    )
    parser.add_argument("-server_name", "--server_name", type=str, help="Server name")
    return parser.parse_args()


def read_config_file(config_file: None) -> Dict[str, Any]:
    if config_file is not None:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found")

        with open(config_file, "r") as file:
            if config_file.endswith(".yaml") or config_file.endswith(".yml"):
                config = yaml.safe_load(file)
            elif config_file.endswith(".json"):
                config = json.load(file)
            else:
                raise ValueError(
                    "Unsupported config file format. Use .yaml, .yml, or .json"
                )
        return config

    config_path_yaml = os.path.join(os.getcwd(), "config.yaml")
    config_path_json = os.path.join(os.getcwd(), "config.json")

    config = {}

    if os.path.exists(config_path_yaml):
        with open(config_path_yaml, "r") as file:
            config = yaml.safe_load(file)
    elif os.path.exists(config_path_json):
        with open(config_path_json, "r") as file:
            config = json.load(file)

    return config


def build_launch_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    launch_kwargs = {
        "llm": args.provider_llm,
        "model_name": args.provider_model,
        "api_key": args.api_key,
        "share_url": args.share,
        "title": args.server_name,
        "show_api_key": args.show_api_key,
        "show_provider_model": args.show_provider_model,
        "show_system_context": args.show_system_context,
        "_init_file_path": args.documents_file_path,
        "show_documents_tab": args.show_documents_tab,
    }
    # Remove keys with None values
    return {k: v for k, v in launch_kwargs.items() if v is not None}
