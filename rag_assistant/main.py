from gradio_ui import GradioUI
from rag_assistant.utils.configure import (
    parse_arguments,
    read_config_file,
    generate_config_json,
    generate_config_yaml,
)


def main() -> None:
    args = parse_arguments()

    if args.generate:
        output_path = args.output
        if not output_path:
            print("saving in config.json")
            output_path = "config.json"

        if output_path.endswith(".json"):
            generate_config_json(output_path)
        elif output_path.endswith(".yaml") or output_path.endswith(".yml"):
            generate_config_yaml(output_path)
        else:
            raise ValueError("Output file must have a .json or .yaml extension")

        print(f"Configuration file generated at {output_path}")
        return

    llm = args.provider_llm
    api_key = args.api_key

    config_file_launch_kwargs = read_config_file(args.config_file)

    app = GradioUI(llm=llm, api_key=api_key, config=config_file_launch_kwargs)

    # cli_launch_kwargs = build_launch_kwargs(args)
    # app = None
    # if len(config_file_launch_kwargs) > 0:
    # else:
    # app = GradioUI(**cli_launch_kwargs)

    # Create Gradio UI
    app.launch()


if __name__ == "__main__":
    main()
