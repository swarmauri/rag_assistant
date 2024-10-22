from gradio_ui import GradioUI
from rag_assistant.utils.configure import (
    parse_arguments,
    build_launch_kwargs,
    read_config_file,
)


def main() -> None:
    args = parse_arguments()

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
