import os
import pickle
import csv
from multiprocessing import Pool
import yaml
import importlib
from functools import partial
from tqdm import tqdm
import numpy as np
import traceback
import logging


def setup_logging(debug_mode):
    """
    Setup logging configuration based on debug mode

    Args:
        debug_mode: Boolean indicating if debug mode is enabled
    """
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types.
    Handle both numpy arrays and native Python types.

    Args:
        obj: Any Python object that might contain numpy values

    Returns:
        Converted object with numpy types replaced by native Python types
    """
    if obj is None:
        return None

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]

    if isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)

    if isinstance(obj, (int, float, str, bool)):
        return obj

    return obj


def parse_parameters(file_path):
    """Parse parameters from SCS file"""
    parameters_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('parameters'):
                parameters_line = line.strip().split(' ', 1)[1]
                parameters_pairs = parameters_line.split()
                for pair in parameters_pairs:
                    key, value = pair.split('=')
                    parameters_dict[key] = value
                break
    return parameters_dict


def process_folder(folder_path, cal_rew_func=None, ideal_specs_dict=None, norm_specs_dict=None, debug=False):
    """
    Process a single folder containing simulation results with detailed debug information

    Args:
        folder_path: Path to the folder
        cal_rew_func: Optional reward calculation function
        ideal_specs_dict: Optional dictionary of ideal specifications
        norm_specs_dict: Optional dictionary of normalization specifications
        debug: Boolean indicating if debug mode is enabled
    """
    try:
        if debug:
            logging.debug(f"\nProcessing folder: {folder_path}")

        parameters_dict = {}
        scs_files = [f for f in os.listdir(folder_path) if f.endswith('.scs')]
        scs_files.sort()

        if scs_files:
            scs_path = os.path.join(folder_path, scs_files[0])
            parameters_dict = parse_parameters(scs_path)
            if debug:
                logging.debug(f"Parsed parameters: {parameters_dict}")

        folder_name = os.path.basename(folder_path)
        tag_parts = folder_name.split('_', 2)
        tag_name = tag_parts[2] if len(tag_parts) > 2 else None

        for folder_file in os.listdir(folder_path):
            if folder_file.endswith(".pkl"):
                pkl_path = os.path.join(folder_path, folder_file)
                if debug:
                    logging.debug(f"Processing pickle file: {pkl_path}")

                with open(pkl_path, "rb") as f:
                    step_data = pickle.load(f)
                    if debug:
                        logging.debug(f"Original pickle data:\n{step_data}")

                try:
                    sim_result = step_data['sim_result']
                    if debug:
                        logging.debug(f"Original sim_result:\n{sim_result}")
                        logging.debug(f"Original sim_result type: {type(sim_result)}")

                    if cal_rew_func:
                        sim_result_converted = convert_numpy_types(sim_result)
                        if debug:
                            logging.debug(f"Converted sim_result:\n{sim_result_converted}")
                            logging.debug("Calculating new reward...")

                        rew = cal_rew_func(ideal_specs_dict, sim_result_converted, norm_specs_dict)
                        if debug:
                            logging.debug(f"Calculated reward: {rew}")
                    else:
                        if debug:
                            logging.debug("Getting reward from step_data...")
                            logging.debug(f"Original reward: {step_data.get('reward')}")

                        rew = convert_numpy_types(step_data.get('reward'))
                        sim_result = convert_numpy_types(sim_result)

                        if debug:
                            logging.debug(f"Converted reward: {rew}")

                    return folder_name, tag_name, rew, sim_result, parameters_dict

                except Exception as e:
                    if debug:
                        logging.error("Error in reward processing:")
                        logging.error(traceback.format_exc())
                    else:
                        logging.error(f"Error occurred in processing reward: {str(e)}. Skipping and continuing.")
                    return None
    except Exception as e:
        if debug:
            logging.error("Error in folder processing:")
            logging.error(traceback.format_exc())
        else:
            logging.error(f"Error processing folder {folder_path}: {str(e)}")
        return None


def process_folders(folder_paths, process_func, debug=False):
    """
    Process all folders either in parallel or sequentially based on debug mode

    Args:
        folder_paths: List of folder paths to process
        process_func: Function to process each folder
        debug: Boolean indicating if debug mode is enabled
    """
    if debug:
        # Sequential processing in debug mode
        results = []
        for folder_path in tqdm(folder_paths, desc="Processing folders"):
            result = process_func(folder_path, debug=True)
            if result is not None:
                results.append(result)
    else:
        # Parallel processing in normal mode
        with Pool() as pool:
            results = list(tqdm(pool.imap(process_func, folder_paths),
                                total=len(folder_paths), desc="Processing folders"))
    return results


def main():
    """Main function with debug mode option"""
    run_test_path = input("Enter the path of the run_test folder: ")
    output_csv_path = input("Enter the path of the output csv file: ")
    debug_mode = input("Enable debug mode? (y/n): ").lower() == 'y'

    # Setup logging
    setup_logging(debug_mode)

    folder_paths = [os.path.join(run_test_path, folder) for folder in os.listdir(run_test_path)
                    if os.path.isdir(os.path.join(run_test_path, folder))]

    if debug_mode:
        logging.debug(f"Found {len(folder_paths)} folders to process")

    update_reward = input("Update reward? (y/n): ")
    if update_reward.lower() == 'y':
        config_path = input("Enter the path of the config folder: ")
        reward_func = input("Enter the reward function name: ")

        with open(os.path.join(config_path, "norm_specs.yaml"), 'r') as file:
            norm_specs = yaml.safe_load(file)
        with open(os.path.join(config_path, "norm_specs_cal.yaml"), 'r') as file:
            ideal_specs = yaml.safe_load(file)

        try:
            reward_module = importlib.import_module('cal_reward')
            cal_reward_func = getattr(reward_module, reward_func)
        except (ImportError, AttributeError) as e:
            logging.error(f"Error importing reward function '{reward_func}': {e}")
            raise

        process_folder_partial = partial(process_folder, cal_rew_func=cal_reward_func,
                                         ideal_specs_dict=ideal_specs, norm_specs_dict=norm_specs)
    else:
        process_folder_partial = process_folder

    results = process_folders(folder_paths, process_folder_partial, debug_mode)

    # Filter out None results
    results = [r for r in results if r is not None]

    if debug_mode:
        logging.debug(f"Successfully processed {len(results)} folders")

    print("Writing results to CSV...")
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Folder Name", "Tag", "Reward", "Specs", "Parameters"])
        writer.writerows([result for result in results if result])

    print(f"Processing complete. Results saved to {output_csv_path}")


if __name__ == "__main__":
    main()