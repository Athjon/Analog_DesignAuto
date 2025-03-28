import pandas as pd
import os
from assign_param2netlist import update_netlist
import yaml


def csv2netlist(csv_file_path, base_dir_path, sim_config, netlist_template_dir_path):
    """
    Convert the CSV file to the netlist file
    :param csv_file_path:  CSV file path
    :param base_dir_path:  base directory path
    :param sim_config:  simulation configuration
    :param netlist_template_dir_path:  netlist template directory path
    :return: None
    """
    df = pd.read_csv(csv_file_path)

    with open(sim_config, 'r') as f:
        sim_config = yaml.safe_load(f)

    #  Iterate through each row in the dataframe
    for index, row in df.iterrows():
        # Get the folder name
        folder_name = row['Folder Name']
        work_dir = os.path.join(base_dir_path, folder_name)
        os.makedirs(work_dir, exist_ok=True)
        print(f"Working directory created for {folder_name}.")

        # Create param_dict from the row data (columns from 24 to end)
        param_dict = row.loc['C1':].to_dict()
        print(f"Parameter dictionary created is {param_dict}.")

        # Update the netlist file
        update_netlist(work_dir, sim_config, param_dict, netlist_template_dir_path)

        print(f"Netlist file updated for {folder_name}.")


if __name__ == "__main__":
    csv_file_path = "/Users/hanwu/Downloads/Log_N65/Lab/b8cb5/Select_Point_List.csv"
    base_dir_path = "/Users/hanwu/Downloads/Log_N65/Lab/b8cb5/Select_Points"
    sim_config = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/config/config_Lab_N65/simulation.yaml"
    unassigned_netlist_dir_path = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_template/netlist_template_Lab_N65"
    csv2netlist(csv_file_path, base_dir_path, sim_config, unassigned_netlist_dir_path)
