import os
import subprocess
import yaml
import toml


def generate_yaml_file(env_name, output_file='environment.yml'):
    """
    Generate a YAML file from a Mamba environment.

    :param env_name: The name of the Mamba environment to export.
    :type env_name: str
    :param output_file: The name of the output YAML file, defaults to 'environment.yml'
    :type output_file: str, optional
    :return: True if the YAML file was successfully generated, False otherwise.
    :rtype: bool
    """
    # Run mamba env export command
    result = subprocess.run(['mamba', 'env', 'export', '-n', '-p', env_name],
                            capture_output=True, text=True, check=True)

    # Parse the YAML output
    env_yaml = yaml.safe_load(result.stdout)

    # Remove the prefix key if it exists
    env_yaml.pop('prefix', None)

    # Write the modified YAML to a file
    with open(output_file, 'w') as file:
        yaml.dump(env_yaml, file, default_flow_style=False)

    return True


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
    result = subprocess.run(['mamba', 'env', 'export', '-n', '-p', env_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error exporting environment: {result.stderr}")
        return

    env = yaml.safe_load(result.stdout)

    # Remove the 'prefix' key from the environment dictionary
    if 'prefix' in env:
        del env['prefix']

    # Clean dependencies
    clean_deps = []
    for dep in env['dependencies']:
        if isinstance(dep, str) and not dep.startswith('_'):
            clean_deps.append(dep.split('=')[0].split('<')[0].split('>')[0])
        elif isinstance(dep, dict) and 'pip' in dep:
            clean_pip = [p.split('==')[0].split('<=')[0].split('>=')[0] for p in dep['pip'] if not p.startswith('-e')]
            clean_deps.append({'pip': clean_pip})

    env['dependencies'] = clean_deps

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the cleaned environment to the output file
    with open(output_file, 'w') as file:
        yaml.dump(env, file, default_flow_style=False)


def update_pyproject_toml(yaml_file, toml_file):
    """
    Updates the dependencies in pyproject.toml based on a given YAML file.

    :param yaml_file: The file path of the YAML file containing environment dependencies.
    :type yaml_file: str
    :param toml_file: The file path of the pyproject.toml file to update.
    :type toml_file: str
    :return: None
    :rtype: None
    """
    with open(yaml_file, 'r') as file:
        yaml_data = yaml.safe_load(file)

    with open(toml_file, 'r') as file:
        toml_data = toml.load(file)

    toml_data['project']['dependencies'] = yaml_data['dependencies']

    # Write the updated data back to the toml file
    with open(toml_file, 'w') as file:
        toml.dump(toml_data, file)


if __name__ == "__main__":
    env_name = "envs/samosa_phase1.yaml"  # Replace with your environment name
    export_and_clean_env(env_name, env_name)
    update_pyproject_toml(env_name, 'pyproject.toml')
