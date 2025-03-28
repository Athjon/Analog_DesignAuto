import os
import pickle
import pandas as pd
import ast
from multiprocessing import Pool
import yaml
import importlib
from functools import partial
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


def save_dataframe_to_parquet(df, output_path, chunk_size=10000):
    table = pa.Table.from_pandas(df)

    # Open a ParquetWriter
    with pq.ParquetWriter(output_path, table.schema) as writer:
        # Write the data in chunks
        for i in tqdm(range(0, len(df), chunk_size), desc=f"Saving to {output_path}"):
            chunk = table.slice(i, chunk_size)
            writer.write_table(chunk)

def flatten_nested_dict(d):
    flattened = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flattened[f"{key}_{subkey}"] = subvalue
        else:
            flattened[key] = value
    return flattened


def is_valid_entry(value):
    return value not in [0.0, 100.0]


def parse_parameters(file_path):
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


def process_folder(folder_path, cal_rew_func=None, ideal_specs_dict=None, norm_specs_dict=None):
    parameters_dict = {}
    scs_files = [f for f in os.listdir(folder_path) if f.endswith('.scs')]
    scs_files.sort()
    if scs_files:
        scs_path = os.path.join(folder_path, scs_files[0])
        parameters_dict = parse_parameters(scs_path)

    folder_name = os.path.basename(folder_path)
    tag_parts = folder_name.split('_', 2)
    tag_name = tag_parts[2] if len(tag_parts) > 2 else None

    for folder_file in os.listdir(folder_path):
        if folder_file.endswith(".pkl"):
            with open(os.path.join(folder_path, folder_file), "rb") as f:
                step_data = pickle.load(f)
            try:
                sim_result = step_data['sim_result']
                if cal_rew_func:
                    rew = cal_rew_func(ideal_specs_dict, sim_result, norm_specs_dict)
                else:
                    rew = step_data.get('reward')
                rew = float(rew) if rew is not None else None
            except Exception as e:
                print(f"Error occurred in processing reward: {str(e)}. Skipping and continuing.")
                rew = None
                sim_result = None
            return folder_name, tag_name, rew, sim_result, parameters_dict
    return None


def expand_data(data):
    def safe_eval(x):
        if isinstance(x, dict):
            return x
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            print(f"Warning: Unable to parse: {x}")
            return {}

    # Expand Specs
    data['Valid_Specs'] = data['Specs'].apply(safe_eval)
    specs_df = data['Valid_Specs'].apply(flatten_nested_dict).apply(pd.Series)

    # Expand Parameters
    def safe_eval_parameters(x):
        if isinstance(x, dict):
            return x
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            print(f"Warning: Unable to parse Parameters: {x}")
            return {}

    params_df = data['Parameters'].apply(safe_eval_parameters).apply(pd.Series)

    # Combine expanded data
    expanded_data = pd.concat([data.drop(columns=['Specs', 'Parameters']), specs_df, params_df], axis=1)

    return expanded_data


def count_valid_specs(specs_dict):
    return sum(is_valid_entry(v) for v in flatten_nested_dict(specs_dict).values())


def plot_label_distribution(labels, group_size=1000, output_path='valid_spec_count_distribution.png'):
    label_values = list(labels.values())
    max_label = max(label_values) if label_values else 0
    colors = LinearSegmentedColormap.from_list("gradient", ["#FFFFFF", "#1C4E87"], N=max_label + 1)

    num_groups = (len(label_values) + group_size - 1) // group_size
    results = []

    for i in range(num_groups):
        start = i * group_size
        end = min(start + group_size, len(label_values))
        current_group = label_values[start:end]
        counts = {label: 0 for label in range(max_label + 1)}
        for label in current_group:
            counts[label] = counts.get(label, 0) + 1
        proportions = {key: value / len(current_group) * 100 for key, value in counts.items()}
        results.append(proportions)

    fig, ax = plt.subplots(figsize=(15, 10))
    base = np.zeros(num_groups)
    for label in range(max_label + 1):
        heights = [result.get(label, 0) for result in results]
        ax.bar(np.arange(num_groups) * group_size, heights, bottom=base, width=group_size, label=f'Count {label}',
               color=colors(label))
        base = np.add(base, heights)

    ax.set_xlabel('Directory Number Range')
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Distribution of Valid Spec Counts per Group of Directories')
    ax.set_xticks(np.arange(0, group_size * num_groups, group_size))
    ax.set_xticklabels([f"{x}-{x + group_size - 1}" for x in range(0, len(label_values), group_size)], rotation=45)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Valid spec count distribution plot saved as '{output_path}'")


def plot_data(data, column_name, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data.index, data[column_name], alpha=0.6, s=1, color='blue', edgecolors='none')
    ax.set_title(f'Scatter Plot of {column_name} over Time')
    ax.set_xlabel('Time Index')
    ax.set_ylabel(column_name)
    ax.set_xticks([0, len(data) - 1])
    ax.set_xticklabels([data['Timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S'),
                        data['Timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')])
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def select_columns(columns):
    while True:
        print("\nAvailable columns:")
        for i, col in enumerate(columns):
            print(f"{i}: {col}")

        start_index = int(input("Enter the starting column index: "))
        end_index = int(input("Enter the ending column index: "))

        selected_columns = columns[start_index:end_index + 1]

        print("\nYou have selected the following columns:")
        for i, col in enumerate(selected_columns, start=1):
            print(f"{i}. {col}")

        confirm = input("\nDo you confirm this selection? (yes/no): ").lower()
        if confirm == 'yes' or confirm == 'y':
            return selected_columns
        else:
            print("Selection cancelled. Please try again.")


def filter_by_tag(data):
    if 'Tag' not in data.columns:
        use_all = input("Tag column not found. Do you want to proceed with all data? (yes/no): ").lower()
        return data if use_all == 'yes' or use_all == 'y' else None

    default_tags = ['ff', 'ss', 'fs', 'tt']
    use_default = input(f"Do you want to use default tags {default_tags}? (yes/no): ").lower()

    if use_default == 'yes' or use_default == 'y':
        tags = default_tags
    else:
        tags = input("Enter tags to filter (comma-separated): ").split(',')
        tags = [tag.strip() for tag in tags]

    return data[data['Tag'].isin(tags)]


def sort_data_by_path(data):
    print("Sorting data based on file paths...")
    sorted_data = data.sort_values('Folder Name')
    return sorted_data.reset_index(drop=True)


def main():
    run_test_path = input("Enter the path of the run_test folder: ")
    output_parquet_prefix = input("Enter the prefix for output Parquet files: ")

    folder_paths = [os.path.join(run_test_path, folder) for folder in os.listdir(run_test_path)
                    if os.path.isdir(os.path.join(run_test_path, folder))]

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
            print(f"Error importing reward function '{reward_func}': {e}")
            raise

        process_folder_partial = partial(process_folder, cal_rew_func=cal_reward_func,
                                         ideal_specs_dict=ideal_specs, norm_specs_dict=norm_specs)
    else:
        process_folder_partial = process_folder

    print("Processing folders...")
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_folder_partial, folder_paths),
                            total=len(folder_paths), desc="Processing folders"))

    print("Creating DataFrame...")
    df = pd.DataFrame(results, columns=["Folder Name", "Tag", "Reward", "Specs", "Parameters"])
    df = df.dropna()  # Remove any None results

    print("Expanding data...")
    expanded_df = expand_data(df)

    print("Debug: Columns in expanded_df:")
    print(expanded_df.columns)

    print("Debug: First few rows of expanded_df:")
    print(expanded_df.head())

    print("Calculating valid spec count...")
    expanded_df['ValidSpecCount'] = expanded_df['Valid_Specs'].apply(count_valid_specs)

    valid_df = expanded_df[expanded_df['ValidSpecCount'] > 0]

    print("Saving to Parquet files...")
    save_dataframe_to_parquet(expanded_df, f"{output_parquet_prefix}_all_data.parquet")
    save_dataframe_to_parquet(valid_df, f"{output_parquet_prefix}_valid_data.parquet")

    print("Plotting valid spec count distribution...")
    labels = dict(zip(expanded_df['Folder Name'], expanded_df['ValidSpecCount']))
    plot_label_distribution(labels, output_path=f"{output_parquet_prefix}_valid_spec_count_distribution.png")

    print(
        f"Processing complete. Results saved to {output_parquet_prefix}_all_data.parquet and {output_parquet_prefix}_valid_data.parquet")
    print(f"Valid spec count distribution plot saved as '{output_parquet_prefix}_valid_spec_count_distribution.png'")

    # Visualization part for valid data
    print("Starting scatter plot generation for valid data...")
    valid_data = pq.read_table(f"{output_parquet_prefix}_valid_data.parquet").to_pandas()

    # Filter data by Tag
    filtered_data = filter_by_tag(valid_data)
    if filtered_data is None:
        print("Data filtering cancelled. Exiting.")
        return

    # Extract the timestamp from 'Folder Name' and convert it to datetime
    print("Extracting timestamps from 'Folder Name'...")
    filtered_data['Timestamp'] = pd.to_datetime(filtered_data['Folder Name'].str.extract(r'(\d{14})')[0],
                                                format='%Y%m%d%H%M%S')

    # Sort the data based on the full file path
    sorted_data = sort_data_by_path(filtered_data)

    # Get all column names except 'Folder Name', 'Timestamp', 'Tag', 'Specs', 'Parameters', and 'ValidSpecCount'
    all_columns = [col for col in sorted_data.columns if
                   col not in ['Folder Name', 'Timestamp', 'Tag', 'Specs', 'Parameters', 'ValidSpecCount']]

    # Let the user select columns
    column_name_list = select_columns(all_columns)

    # Create a directory to store the plots
    plot_dir = os.path.join(os.path.dirname(f"{output_parquet_prefix}_valid_data.parquet"), 'scatter_plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Plot the data and save each plot as a separate image file
    for column_name in tqdm(column_name_list, desc="Generating plots"):
        plot_file_name = f'{column_name}_scatter_plot.png'
        plot_file_path = os.path.join(plot_dir, plot_file_name)
        plot_data(sorted_data, column_name, plot_file_path)

    print(f"All plots have been generated and saved in the '{plot_dir}' directory.")

    # Summary of saved files
    print("\nSummary of saved files:")
    print(f"1. All data Parquet file: {output_parquet_prefix}_all_data.parquet")
    print(f"2. Valid data Parquet file: {output_parquet_prefix}_valid_data.parquet")
    print(f"3. ValidSpecCount distribution plot: {output_parquet_prefix}_valid_spec_count_distribution.png")
    print(f"4. Scatter plots: {plot_dir}/")
    print("\nAll processing and visualization tasks are complete.")


if __name__ == "__main__":
    main()