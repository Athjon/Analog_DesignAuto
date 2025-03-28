import argparse
import random
import os
import yaml


def represent_list(dumper, data):
    """
    Custom representer for lists to force flow style for range type values
    """
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


def sample_ideal_specs(ideal_specs_config_path, sample_num, output_folder):
    """
    Sample ideal specs from configuration file and save to output files.
    For range type objectives, directly convert bounds to value in flow style.

    Args:
        ideal_specs_config_path: File path of the ideal specs config file
        sample_num: Number of samples to generate
        output_folder: Output folder
    Returns:
        Sampled ideal specs files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder don't exist, generate: {output_folder}")

    with open(ideal_specs_config_path, 'r') as file:
        ideal_specs_config = yaml.safe_load(file)

    # Register the custom representer only for range type values
    yaml.add_representer(list, represent_list)

    for i in range(sample_num):
        sample_data = {}
        for category in ideal_specs_config:
            for spec, detail in ideal_specs_config[category].items():
                # Copy the 'objective' and 'reward_type' directly
                sample_detail = {
                    'objective': detail['objective'],
                    'reward_type': detail['reward_type']
                }

                # Special handling for range type objectives
                if detail['objective'] == 'range':
                    # Directly use bounds as value for range type
                    if 'bounds' not in detail:
                        raise ValueError(f"Missing bounds field for range type objective: {spec}")
                    sample_detail['value'] = detail['bounds']
                else:
                    # Generate a random value within the bounds for other types
                    lower_bound, upper_bound = detail['bounds']
                    sample_detail['value'] = random.uniform(lower_bound, upper_bound)

                sample_data[spec] = sample_detail

        # Save the sampled data to a yaml file
        sample_file_name = f"{i + 1}.yaml"
        sample_file_path = os.path.join(output_folder, sample_file_name)
        with open(sample_file_path, 'w') as file:
            yaml.dump(sample_data, file, sort_keys=False, allow_unicode=True)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate random sample YAML files based on a given template.")
    parser.add_argument("yaml_path", type=str, help="Path to the input YAML file.")
    parser.add_argument("num_samples", type=int, help="Number of random sample files to generate.")
    parser.add_argument("output_folder", type=str, help="Folder to save the generated sample files.")
    args = parser.parse_args()

    # Generate random sample YAML files
    sample_ideal_specs(args.yaml_path, args.num_samples, args.output_folder)


if __name__ == "__main__":
    main()

# Test Code
# python sample_ideal_specs.py
# /Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/config_3/generalize_specs.yaml 100
# /Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/sampled_specs
