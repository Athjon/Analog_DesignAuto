import os
import itertools
import yaml
import numpy as np


def generate_parameter_combinations(ideal_specs_config, sample_counts):
    parameter_samples = {}
    for category in ideal_specs_config:
        for spec, detail in ideal_specs_config[category].items():
            sample_num = sample_counts[spec]
            values = np.linspace(detail['bounds'][0], detail['bounds'][1], sample_num)
            parameter_samples[spec] = values

    # Generate all possible combinations of parameters
    keys, values = zip(*parameter_samples.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return combinations


def generate_random_samples_and_save(ideal_specs_config, sample_counts, output_folder, sample_num):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder doesn't exist, generating: {output_folder}")

    for i in range(1, sample_num + 1):
        sample_data = {}
        for category in ideal_specs_config:
            for spec, detail in ideal_specs_config[category].items():
                value = np.random.uniform(detail['bounds'][0], detail['bounds'][1])
                sample_data[spec] = {
                    'objective': detail['objective'],
                    'constrain_type': detail['constrain_type'],
                    'value': float(value)
                }

        # Save the sampled data to a yaml file
        sample_file_name = f"{i}.yaml"
        sample_file_path = os.path.join(output_folder, sample_file_name)
        with open(sample_file_path, 'w') as file:
            yaml.dump(sample_data, file)

    print(f"Generated {sample_num} YAML files in {output_folder}")


def collect_and_process_user_input():
    yaml_path = input("Please enter the path to the input YAML file: ").strip()
    output_folder = input("Please enter the folder to save the generated sample files: ").strip()
    sample_mode = input("Please choose the sampling mode - 'uniform' or 'random': ").strip()
    sample_counts = {}
    sample_num = 0  # Used only in random sampling mode

    with open(yaml_path, 'r') as file:
        ideal_specs_config = yaml.safe_load(file)

    if sample_mode == 'random':
        sample_num = int(input("Enter the number of random sample files to generate: "))
    else:
        for category in ideal_specs_config:
            for spec, detail in ideal_specs_config[category].items():
                print(f"{spec}: min={detail['bounds'][0]}, max={detail['bounds'][1]}")
                sample_count = int(input(f"Enter the number of samples for {spec}: "))
                sample_counts[spec] = sample_count

    if sample_mode == 'uniform':
        combinations = generate_parameter_combinations(ideal_specs_config, sample_counts)
        print(f"Total possible combinations: {len(combinations)}")
        proceed = input("Do you want to generate the YAML files? (yes/no): ").strip().lower()
        if proceed == 'yes':
            sample_ideal_specs(ideal_specs_config, combinations, output_folder)
    elif sample_mode == 'random':
        generate_random_samples_and_save(ideal_specs_config, sample_counts, output_folder, sample_num)

    else:
        print("Operation cancelled.")


def sample_ideal_specs(ideal_specs_config, combinations, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder doesn't exist, generating: {output_folder}")

    for i, combination in enumerate(combinations, start=1):
        sample_data = {}
        for spec, value in combination.items():
            for category in ideal_specs_config:
                if spec in ideal_specs_config[category]:
                    detail = ideal_specs_config[category][spec]
                    sample_data[spec] = {
                        'objective': detail['objective'],
                        'constrain_type': detail['constrain_type'],
                        'value': float(value)
                    }
                    break

        # Save the sampled data to a yaml file
        sample_file_name = f"{i}.yaml"
        sample_file_path = os.path.join(output_folder, sample_file_name)
        with open(sample_file_path, 'w') as file:
            yaml.dump(sample_data, file)

    print(f"Generated {len(combinations)} YAML files in {output_folder}")


if __name__ == "__main__":
    collect_and_process_user_input()
