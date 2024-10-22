import json
import yaml
import os

PATH = "./rag_assistant/templates/config_template.json"


def generate_config_json(output_path):
    with open(PATH, "r") as template_file:
        config_data = json.load(template_file)

    with open(output_path, "w") as output_file:
        json.dump(config_data, output_file, indent=4)


def generate_config_yaml(output_path):
    with open(PATH, "r") as template_file:
        config_data = json.load(template_file)

    with open(output_path, "w") as output_file:
        yaml.dump(config_data, output_file, default_flow_style=False)


# Example usage:
# generate_config_json('/path/to/output/config.json')
# generate_config_yaml('/path/to/output/config.yaml')
