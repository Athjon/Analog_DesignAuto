import yaml
import os
import random


def generalize_config(generalize, path):
    if generalize:
        specs_path_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        select_specs_path = random.choice(specs_path_list)
        specs_config_path = os.path.join(path, select_specs_path)
        # print(f"Debug!!! Selecting specs config: {specs_config_path}")
    else:
        specs_config_path = path

    with open(specs_config_path, 'r') as file:
        ideal_specs = yaml.safe_load(file)

    return ideal_specs
