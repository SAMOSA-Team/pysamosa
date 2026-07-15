import os
import subprocess
import yaml
import toml


def export_and_clean_env(env_name, output_file):
    """
    Exports a specified conda environment to a YAML file and cleans the dependencies.

    :param env_name: The name of the conda environment to export.
    :type env_name: str
    :param output_file: The file path where the cleaned environment YAML will be saved.
    :type output_file: str
    :return: None
    :rtype: None
    """

    result = subprocess.run(
        ["mamba", "env", "export", "-n", env_name],
        capture_output=True,
        text=True,
        check=True,
    )

    # Parse the YAML output
    env = yaml.safe_load(result.stdout)

    # Remove the 'prefix' key from the environment dictionary
    if "prefix" in env:
        del env["prefix"]

    # Clean dependencies
    clean_deps = []
    for dep in env["dependencies"]:
        if isinstance(dep, str) and not dep.startswith("_"):
            clean_deps.append(dep.split("=")[0].split("<")[0].split(">")[0])
        elif isinstance(dep, dict) and "pip" in dep:
            clean_pip = [
                p.split("==")[0].split("<=")[0].split(">=")[0]
                for p in dep["pip"]
                if not p.startswith("-e")
            ]
            clean_deps.append({"pip": clean_pip})

    env["dependencies"] = clean_deps

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the cleaned environment to the output file
    with open(output_file, "w") as file:
        yaml.dump(env, file, default_flow_style=False)


def update_pyproject_toml(yaml_file, toml_file):
    """
    Updates the dependencies in pyproject.toml based on a given YAML file,
    while preserving other content in the TOML file.

    :param yaml_file: The file path of the YAML file containing environment dependencies.
    :type yaml_file: str
    :param toml_file: The file path of the pyproject.toml file to update.
    :type toml_file: str
    :return: None
    :rtype: None
    """

    # Read YAML file
    with open(yaml_file, "r") as file:
        yaml_data = yaml.safe_load(file)

    # Read existing pyproject.toml
    with open(toml_file, "r") as file:
        toml_data = toml.load(file)

    # Ensure 'project' section exists
    if "project" not in toml_data:
        toml_data["project"] = {}

    # Ensure to map YAML `dependencies` correctly
    if "dependencies" in yaml_data:
        if not isinstance(toml_data["project"], dict):
            toml_data["project"] = {}
        if "dependencies" not in toml_data["project"]:
            toml_data["project"]["dependencies"] = []
        else:
            if not isinstance(toml_data["project"]["dependencies"], list):
                toml_data["project"]["dependencies"] = []

        for dep in yaml_data["dependencies"]:
            if isinstance(dep, str):
                toml_data["project"]["dependencies"].append(dep)
            elif isinstance(dep, dict) and "pip" in dep:
                for pip_dep in dep["pip"]:
                    toml_data["project"]["dependencies"].append(f"pip:{pip_dep}")
    print(toml_data)
    with open(toml_file, "w") as file:
        toml.dump(toml_data, file)


if __name__ == "__main__":

    env_name = "pysamosa"
    yaml_name = "envs/pysamosa.yaml"

    export_and_clean_env(env_name, yaml_name)
    update_pyproject_toml(yaml_name, "pyproject.toml")
