from rag_assistant.RagAssistant import RagAssistant
from rag_assistant.utils.config_generator import (
    generate_config_json,
    generate_config_yaml,
)
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
        "generate", nargs="?", help="Generate a configuration file", default=None
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path for generated configuration",
        default=None,
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
        "-provider_llm",
        "--provider_llm",
        type=str,
        required=True,
        help=f"Your provider LLM: {' | '.join(RagAssistant.available_llms.keys())}",
    )
    parser.add_argument(
        "-provider_model", "--provider_model", type=str, help="Your provider model"
    )
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
