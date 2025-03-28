import os
import pandas as pd
import yaml
import importlib
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def load_config(config_path):
    with open(os.path.join(config_path, "norm_specs.yaml"), 'r') as file:
        norm_specs = yaml.safe_load(file)
    with open(os.path.join(config_path, "norm_specs_cal.yaml"), 'r') as file:
        ideal_specs = yaml.safe_load(file)
    return norm_specs, ideal_specs


def import_reward_function(reward_func_name):
    try:
        reward_module = importlib.import_module('cal_reward')
        cal_reward_func = getattr(reward_module, reward_func_name)
        return cal_reward_func
    except (ImportError, AttributeError) as e:
        print(f"Error importing reward function '{reward_func_name}': {e}")
        raise


def update_reward(row, cal_rew_func, ideal_specs_dict, norm_specs_dict):
    try:
        sim_result = {k: v for k, v in row.items() if k in ideal_specs_dict}
        new_reward = cal_rew_func(ideal_specs_dict, sim_result, norm_specs_dict)
        return new_reward
    except Exception as e:
        print(f"Error occurred in cal_reward: {str(e)}. Skipping and continuing.")
        return None


def process_excel_file(file_path, cal_rew_func, ideal_specs_dict, norm_specs_dict):
    df = pd.read_excel(file_path)

    update_reward_partial = partial(update_reward, cal_rew_func=cal_rew_func,
                                    ideal_specs_dict=ideal_specs_dict,
                                    norm_specs_dict=norm_specs_dict)

    with Pool() as pool:
        df['New_Reward'] = list(tqdm(pool.imap(update_reward_partial, df.to_dict('records')),
                                     total=len(df), desc=f"Processing {file_path}"))

    df.to_excel(file_path.replace('.xlsx', '_updated.xlsx'), index=False)
    print(f"Updated file saved: {file_path.replace('.xlsx', '_updated.xlsx')}")


def main():
    excel_path = input("Enter the path of the Excel files to be updated: ")
    config_path = input("Enter the path of the config folder: ")
    reward_func_name = input("Enter the reward function name: ")

    norm_specs, ideal_specs = load_config(config_path)
    cal_rew_func = import_reward_function(reward_func_name)

    process_excel_file(excel_path, cal_rew_func, ideal_specs, norm_specs)


if __name__ == "__main__":
    main()