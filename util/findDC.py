# from extract_device_param_value import parse_device_values, parse_device_param, find_device_param_value
from util.extract_device_param_value import parse_device_values, parse_device_param, find_device_param_value
import os
import numpy as np
from typing import List, Dict, Union

def findDCValue(filepath):
    # Extract the properties using the previously defined function

    instance_name = "V0"
    property_name = "pwr"
    default_value = 100.0

    try:
        value_dict = parse_device_values(filepath)
        param_dict = parse_device_param(filepath)
        value = find_device_param_value(instance_name, property_name, value_dict, param_dict)
        value = abs(value)

        # Check if the value is a valid result
        if isinstance(value, str):
            raise ValueError("Error in finding device parameter value")

    except Exception as e:
        print(f"Warning: {e}. Setting value to default ({default_value}).")
        value = default_value

    return {property_name: value}


# Test Code
# file_path = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/DC.raw/dcOpInfo.info.encode"
# print(findDCValue(file_path))

def extract_dcOP_data(file_path, search_keyword):
    """
    Search the first line in a file containing a specific keyword and extract float data.

    Args:
    - file_path: Path to the file to be processed.
    - search_keyword: Keyword to search for in the file.

    Returns:
    - A float extracted from the line containing the keyword, or None if not found.
    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if search_keyword in line:
                    # Split the line by space and extract the third element as data
                    parts = line.split()
                    if len(parts) >= 3:  # Ensure there are at least 3 parts
                        extracted_data = float(parts[2])  # Convert to float
                    break  # Stop searching after finding the first match
        if extracted_data is None:
            raise ValueError("Keyword not found or file does not contain valid data format.")
    except FileNotFoundError:
        raise FileNotFoundError("File does not exist.")
    except ValueError as e:
        raise ValueError(f"Error while processing the file: {e}")

    return abs(extracted_data)


def findIQ(filepath):
    search_keyword = "V2:p"
    try:
        value = extract_dcOP_data(filepath, search_keyword)
    except Exception as e:
        print(f"Warning: {e}. Setting IQ to default (1.0).")
        value = 100.0

    if value > 0.00002:
        print(f"Power is too large, return to max value")
        value = 100.0
    return {"IQ": value}


# Test Code
# path = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/DC.raw/dcOp.dc.encode"
# print(findIQ(path))


def findIQ_KNL(filepath):
    search_keyword = "V2:p"
    try:
        value = extract_dcOP_data(filepath, search_keyword)
    except Exception as e:
        print(f"Warning: {e}. Setting IQ to default (100.0).")
        value = 100.0

    return {"IQ": value}


def findIQ_Jianping(filepath):
    search_keyword = "V2:p"
    try:
        value = extract_dcOP_data(filepath, search_keyword)
        value = value
    except Exception as e:
        print(f"Warning: {e}. return to max value.")
        value = 100.0
    if value <= 0.0:
        print(f"Power is lower than zero, impossible, return to max value")
        value = 100.0
    return {"IQ": value}

def findIQ_Lab(filepath):
    search_keyword = "V1:p"
    try:
        value = extract_dcOP_data(filepath, search_keyword)
        value = value - 0.0005
    except Exception as e:
        print(f"Warning: {e}. return to max value.")
        value = 100.0
    if value <= 0.0:
        print(f"Power is lower than zero, impossible, return to max value")
        value = 100.0
    return {"IQ": value}


def findIQ_Cai(filepath):
    power_keyword = "V2:p"
    bias_keyword = "I3:sink"
    try:
        power_value = extract_dcOP_data(filepath, power_keyword)
        bias_value = extract_dcOP_data(filepath, bias_keyword)
        power_value = power_value - 0.001 - bias_value
    except Exception as e:
        print(f"Warning: {e}. return to max value.")
        power_value = 100.0
    if power_value <= 0.0:
        print(f"Power is lower than zero, impossible, return to max value")
        power_value = 100.0
    return {"IQ": power_value}


def findIQ_Haoqiang(filepath):
    power_keyword = "V0:p"
    bias_keyword = "I2:sink"
    try:
        power_value = extract_dcOP_data(filepath, power_keyword)
        # print(f"Debug!!! power_value: {power_value}")
        bias_value = extract_dcOP_data(filepath, bias_keyword)
        # print(f"Debug!!! bias_value: {bias_value}")
        power_value = power_value - bias_value
    except Exception as e:
        print(f"Warning: {e}. return to max value.")
        power_value = 100.0
    if power_value <= 0.0:
        print(f"Power is lower than zero, impossible, return to max value")
        power_value = 100.0
    return {"IQ": power_value}

def findIQ_AXS(filepath):
    power_keyword = "V2:p"
    try:
        power_value = extract_dcOP_data(filepath, power_keyword)
        # print(f"Debug!!! power_value: {power_value}")
        # print(f"Debug!!! bias_value: {bias_value}")
        power_value = power_value - 0.000001
    except Exception as e:
        print(f"Warning: {e}. return to max value.")
        power_value = 100.0
    if power_value <= 0.0:
        print(f"Power is lower than zero, impossible, return to max value")
        power_value = 100.0
    return {"IQ": power_value}

# if __name__ == "__main__":
#     # Test code
#     file_path = "/Users/hanwu/Downloads/Netlist_AXS/DC.raw/dcOp.dc"
#     print(findIQ_AXS(file_path))

def findVOS_AXS_MC(filepath):
    power_keyword = "net1"
    try:
        power_value = extract_dcOP_data(filepath, power_keyword)
        # print(f"Debug!!! power_value: {power_value}")
        # print(f"Debug!!! bias_value: {bias_value}")
        vos = power_value - 1.2
    except Exception as e:
        print(f"Warning: {e}. return to max value.")
        vos = 100.0
    return vos

# if __name__ == "__main__":
#     # Test code
#     file_path = "/Users/hanwu/Downloads/Netlist_AXS/Mismatch.raw/mc1_separate/1101/dcOp.dc"
#     print(findIQ_AXS_MC(file_path))

def findMismatch_AXS(root_dir: str) -> Dict[str, float]:
    """
    Scan all dcOp.dc files in the directory and calculate sigma value

    Args:
        root_dir (str): Root directory path

    Returns:
        Dict[str, float]: Returns a dictionary with sigma value
    """
    try:
        # Store all found dcOp.dc file paths
        dc_files = []

        # Traverse directory
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename == "dcOp.dc":
                    full_path = os.path.join(dirpath, filename)
                    dc_files.append(full_path)

        # Store all calculation results
        results = []

        # Process each file with findIQ_AXS_MC function and collect results
        for dc_file in dc_files:
            try:
                result = findVOS_AXS_MC(dc_file)  # Assumes this function exists
                results.append(result)
            except Exception as e:
                print(f"Error processing file {dc_file}: {str(e)}")
                continue

        if not results:
            print("No valid calculation results found")
            return {'sigma': 100.0}

        # Calculate statistics
        results_array = np.array(results)
        sigma = np.std(results_array)

        if sigma < 0:
            sigma = 100.0

        # Plot distribution
        # plot_distribution(results, sigma)

        return {'sigma': sigma}

    except Exception as e:
        print(f"Error in findMismatch_AXS: {str(e)}")
        return {'sigma': 100.0}


def plot_distribution(results: List[float], sigma: float):
    """
    Plot distribution of results

    Args:
        results (List[float]): List of calculation results
        sigma (float): Sigma value
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))

        # Plot histogram
        plt.hist(results, bins=30, density=True, alpha=0.7, color='b')

        # Add normal distribution curve
        x = np.linspace(min(results), max(results), 100)
        mean = np.mean(results)
        gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))
        plt.plot(x, gaussian, 'r-', lw=2, label='Normal Distribution')

        plt.title('Distribution of Results')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.grid(True)
        plt.legend()

        plt.show()
    except Exception as e:
        print(f"Error in plot_distribution: {str(e)}")

# if __name__ == "__main__":
#
#     root_directory = "/Users/hanwu/Downloads/Netlist_AXS/Mismatch.raw/mc1_separate"  # Replace with actual directory path
#     result_dict = findMismatch_AXS(root_directory)
#     print(result_dict)

def findMismatch_AXS_Simple(root_dir: str) -> Dict[str, float]:

        return {'sigma': 0.01}