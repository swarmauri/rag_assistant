from RagAssistant import RagAssistant
from gradio_interface import launch


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Swarmauri Developer Assistant Command Line Tool"
    )
    parser.add_argument(
        "-api_key", "--api_key", type=str, help="Your api key", required=True
    )
    parser.add_argument(
        "-show_api_key",
        "--show_api_key",
        type=bool,
        help="Toggle displaying api key on app",
        default=False,
        required=False,
    )

    parser.add_argument(
        "-provider_model",
        "--provider_model",
        type=str,
        help="Your provider model",
        required=False,
    )
    parser.add_argument(
        "-show_provider_model",
        "--show_provider_model",
        type=bool,
        help="Toggle displaying Provider Model on app",
        default=False,
        required=False,
    )

    parser.add_argument(
        "-system_context",
        "--system_context",
        type=str,
        help="Assistants System Context",
        required=False,
    )
    parser.add_argument(
        "-show_system_context",
        "--show_system_context",
        type=bool,
        help="Toggle displaying System Context on app",
        default=False,
        required=False,
    )

    parser.add_argument(
        "-documents_file_path",
        "--documents_file_path",
        type=str,
        help="Filepath of Documents JSON",
        required=False,
    )
    parser.add_argument(
        "-show_documents_tab",
        "--show_documents_tab",
        type=bool,
        help="Toggle displaying Document Tabs on app",
        default=False,
        required=False,
    )

    parser.add_argument(
        "-db_path", "--db_path", type=str, help="path to sqlite3 db", required=False
    )

    parser.add_argument(
        "-share",
        "--share",
        type=bool,
        help="Deploy a live app on gradio",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-server_name", "--server_name", type=str, help="Server name", required=False
    )
    # parser.add_argument('-favicon_path', '--favicon_path', type=str, help='Path of application favicon', required=False)

    args = parser.parse_args()

    api_key = args.api_key

    # Create Assistant
    assistant = RagAssistant(api_key=api_key, llm="openai")

    # If params then modify Assistant's config related to model
    if args.provider_model:
        assistant.set_model(args.provider_model)

    # If params then modify Assistant's config related to agent
    if args.system_context:
        assistant.system_context = args.system_context

    # If params then modify Assistant's config related to logging
    if args.db_path:
        assistant.db_path = args.db_path

    # If params then modify Assistant's config
    launch_kwargs = {}
    if args.share:
        launch_kwargs.update({"share": args.share})
    if args.server_name:
        launch_kwargs.update({"server_name": args.server_name})

    if args.show_api_key:
        launch_kwargs.update({"show_api_key": args.show_api_key})
    if args.show_provider_model:
        launch_kwargs.update({"show_provider_model": args.show_provider_model})
    if args.show_system_context:
        launch_kwargs.update({"show_system_context": args.show_system_context})

    if args.documents_file_path:
        launch_kwargs.update({"documents_file_path": args.documents_file_path})

    if args.show_documents_tab == True:
        launch_kwargs.update({"show_documents_tab": args.show_documents_tab})

    # launch_kwargs["llm"] = "openai"

    # if args.favicon_path:
    # launch_kwargs.update({'favicon_path': args.favicon_path})
    # else:
    # launch_kwargs.update({'favicon_path': "favicon-32x32.png"})

    assistant.initialize_agent()
    launch(assistant=assistant, **launch_kwargs)


if __name__ == "__main__":
    main()
