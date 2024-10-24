import argparse
from rag_assistant.gradio_ui import GradioUI
from rag_assistant.utils.configure import (
    read_config_file,
    generate_config_json,
    generate_config_yaml,
)
from rag_assistant.RagAssistant import RagAssistant
import sys


def generate_config(output_path: str) -> None:
    """Generates a config file based on the output path's extension."""
    if output_path.endswith(".json"):
        generate_config_json(output_path)
    elif output_path.endswith((".yaml", ".yml")):
        generate_config_yaml(output_path)
    else:
        raise ValueError("Output file must have a .json or .yaml/.yml extension")

    print(f"Configuration file generated at {output_path}")


def launch_app(args) -> None:
    """Launches the Gradio UI application."""
    try:
        config_kwargs = read_config_file(args.config_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    app = GradioUI(
        llm=args.provider_llm,
        api_key=args.api_key,
        config=config_kwargs,
    )
    app.launch()


def main() -> None:
    parser = argparse.ArgumentParser(description="Swarmauri Developer Assistant")

    # Create subparsers for `generate` and `launch` commands
    subparsers = parser.add_subparsers(dest="command", required=True)

    # `generate` subcommand
    generate_parser = subparsers.add_parser(
        "generate", help="Generate a configuration file"
    )
    generate_parser.add_argument(
        "-o", "--output", type=str, default="config.json", help="Output file path"
    )

    # `launch` subcommand
    launch_parser = subparsers.add_parser(
        "launch", help="Launch the Gradio UI application"
    )
    launch_parser.add_argument(
        "-api_key", "--api_key", type=str, required=True, help="Your API key"
    )
    launch_parser.add_argument(
        "-provider_llm",
        "--provider_llm",
        type=str,
        required=True,
        help=f"Your provider LLM: {' | '.join(RagAssistant.available_llms.keys())}",
    )
    launch_parser.add_argument(
        "-provider_model", "--provider_model", type=str, help="Your provider model"
    )
    launch_parser.add_argument(
        "-config_file", "--config_file", type=str, help="Path to config file"
    )

    args = parser.parse_args()

    # Handle subcommand logic
    if args.command == "generate":
        generate_config(args.output)
    elif args.command == "launch":
        launch_app(args)


if __name__ == "__main__":
    main()
