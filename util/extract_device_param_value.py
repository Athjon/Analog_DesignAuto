import re
import csv


def process_paragraph(paragraph):
    """
    Processes a paragraph of device data and converts it into a dictionary.

    Args:
    paragraph (str): A string containing the device data in a specific format.

    Returns:
    dict: A dictionary with the device key and its corresponding values and type.
    """
    # Split the paragraph into lines for easier processing
    lines = paragraph.split('\n')

    # Extracting the key (like 'V0', 'V3', etc.) and device type (like 'vsource')
    key, device_type = re.findall(r'"([^"]+)"', lines[0])[0], re.findall(r'"([^"]+)"', lines[0])[1]
    device_dict = {"Device_Type": device_type}

    # Processing the content inside the parentheses to get the device values
    values = lines[1:-2]  # Exclude the first and last two lines which are not values
    values_list = []
    for value in values:
        # Convert 'nan' to string, otherwise keep the original type (float or int)
        if value.strip().lower() == 'nan':
            values_list.append('nan')
        else:
            try:
                # Convert to float or int as appropriate
                num = float(value)
                if num.is_integer():
                    num = int(num)
                values_list.append(num)
            except ValueError:
                values_list.append(value.strip())

    # Building the final dictionary for the device
    paragraph_dict = {key: device_dict}
    paragraph_dict[key]["values"] = values_list

    return paragraph_dict


def parse_device_values(filepath):
    """
    Parses a file containing device data and extracts the values.

    Args:
    filepath (str): Path to the file containing device data.

    Returns:
    list: A list of dictionaries, each representing a device's data.
    """
    with open(filepath, "r") as file:
        file_content = file.read()

    # Find the section after 'VALUE' in the file content
    value_section_match = re.search(r'VALUE(.+)', file_content, re.DOTALL)
    if not value_section_match:
        return "No 'VALUE' section found in the file."

    value_section = value_section_match.group(1)

    # Regular expression pattern to match the device data format
    pattern = r'\"[^\"]+\"\s+\"[^\"]+\"\s+\([\s\S]+?\)\s+PROP\(\s+\"[^\"]+\"\s+\"[^\"]+\"'
    device_raw_info = re.findall(pattern, value_section)

    # Process each device's raw information and return the processed data
    processed_device_dict = [process_paragraph(paragraph) for paragraph in device_raw_info]

    return processed_device_dict


# Test Code
# filepath = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/DC.raw/dcOpInfo.info.encode"
# value_section = parse_device_values(filepath)
# print(value_section)

# Output
# [{'I10': {'Device_Type': 'isource', 'values': [5e-05,
# -0.0825866, -4.12933e-06]}}, {'V0': {'Device_Type': 'vsource', 'values': [1.8, -0.000446445, -0.000803601]}},
# {'V3': {'Device_Type': 'vsource', 'values': [0.9, 0, 0]}}, {'V4': {'Device_Type': 'vsource', 'values': [0.9, 0, 0]}}]


def parse_device_param(filepath):
    """
    Parses device parameters from a file and returns a nested dictionary.

    Args:
    filepath (str): Path to the file containing device parameters.

    Returns:
    dict: A dictionary with device types as keys and another dictionary of parameters as values.
    """

    with open(filepath, "r") as file:
        file_content = file.read()

    # Find positions of 'TYPE' and 'VALUE' in the file content
    type_pos = file_content.find("TYPE")
    type_next_line_pos = file_content.find("\n", type_pos) + 1
    value_pos = file_content.find("VALUE")

    # Extract the text between 'TYPE' and 'VALUE'
    extracted_text = file_content[type_next_line_pos:value_pos].strip()

    # Split the extracted text into lines
    lines = extracted_text.split('\n')

    # Define regular expression patterns
    key_pattern = re.compile(r'"(.+?)"\s+STRUCT\(')
    prop_pattern = re.compile(r'"(.+?)"\s+(.+)\s+PROP\(')

    # Initialize result dictionary
    result = {}

    # Iterate over each line
    for i in range(len(lines)):
        # Extract outer key (device type)
        key_match = key_pattern.match(lines[i])
        if key_match:
            key_name = key_match.group(1)
            result[key_name] = {}

        # Extract inner properties
        prop_match = prop_pattern.match(lines[i])
        if prop_match:
            prop_key = prop_match.group(1)
            value_type = prop_match.group(2)

            # Safely extract value unit and description to avoid IndexError
            value_unit = lines[i + 1].split('"')[3] if i + 1 < len(lines) and len(lines[i + 1].split('"')) > 3 else None
            description = lines[i + 2].split('"')[3] if i + 2 < len(lines) and len(
                lines[i + 2].split('"')) > 3 else None

            # Add the property to the result dictionary
            result[key_name][prop_key] = {
                "Value_Type": value_type,
                "Value_Unit": value_unit,
                "Description": description
            }

    return result


def dict_to_csv(dict_data, csv_file_path):
    """
    Converts a nested dictionary to a CSV file.

    Args:
    dict_data (dict): The nested dictionary to convert.
    csv_file_path (str): The file path where the CSV will be saved.

    The CSV file will have columns for outer keys, inner keys, and values of the inner dictionary.
    """

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)

        # Write the header
        csv_writer.writerow(["Device Type", "Param Name", "Value Type", "Value Unit", "Description"])

        # Write the rows
        for outer_key, inner_dict in dict_data.items():
            for inner_key, properties in inner_dict.items():
                csv_writer.writerow([
                    outer_key,
                    inner_key,
                    properties.get("Value_Type", ""),
                    properties.get("Value_Unit", ""),
                    properties.get("Description", "")
                ])


# Test Code
# filepath = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/DC.raw/dcOpInfo.info.encode"
# value_param = parse_device_param(filepath)
# print(value_param)
#
# csv_path = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/DC.raw/dcOpInfo.info.encode.csv"
# dict_to_csv(value_param, csv_path)

# Output
# {'isource': {'i': {'Value_Type': 'FLOAT DOUBLE', 'Value_Unit': 'A', 'Description': 'Current through the
# source'}, 'v': {'Value_Type': 'FLOAT DOUBLE', 'Value_Unit': 'V', 'Description': 'Voltage across the source'},
# 'pwr': {'Value_Type': 'FLOAT DOUBLE', 'Value_Unit': 'W', 'Description': 'Power dissipation'}}, 'vsource': {'v': {
# 'Value_Type': 'FLOAT DOUBLE', 'Value_Unit': 'V', 'Description': 'Voltage across the source'}, 'i': {'Value_Type':
# 'FLOAT DOUBLE', 'Value_Unit': 'A', 'Description': 'Current through the source'}, 'pwr': {'Value_Type': 'FLOAT
# DOUBLE', 'Value_Unit': 'W', 'Description': 'Power dissipation'}}}

def find_device_param_value(device_name, param_name, value_dict, param_dict):
    """
    Find the value of a specified parameter for a given device.

    :param device_name: Name of the device (e.g., 'I10', 'V0').
    :param param_name: Name of the parameter (e.g., 'i', 'v', 'pwr').
    :param value_dict: Dictionary containing device values.
    :param param_dict: Dictionary containing parameter specifications.
    :return: The value corresponding to the specified parameter for the given device.
    """
    # Find the device in the value dictionary
    device_info = next((item for item in value_dict if device_name in item), None)
    if device_info is None:
        return "Device not found"

    # Extract the device type
    device_type = device_info[device_name]['Device_Type']

    # Check if the device type is in the param dictionary
    if device_type not in param_dict:
        return "Device type not found in parameters"

    # Find the index of the parameter in the param dictionary
    if param_name not in param_dict[device_type]:
        return "Parameter not found for device type"

    param_index = list(param_dict[device_type].keys()).index(param_name)

    # Retrieve the value from the values list
    return device_info[device_name]['values'][param_index]


# Test Code
# device_name = "V0"
# param_name = "pwr"
# filepath = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/DC.raw/dcOpInfo.info.encode"
# value_dict = parse_device_values(filepath)
# print(value_dict)
# param_dict = parse_device_param(filepath)
# print(param_dict)
# print(find_device_param_value(device_name, param_name, value_dict, param_dict))

# Output
# -0.000803601


def extract_group_params(device_type_str, param_name_str, value_dict_list, param_list_dict):
    extract_values = []
    for values_dict in value_dict_list:
        for device_id, device_info in values_dict.items():
            if device_info['Device_Type'] == device_type_str:
                param_index = list(param_list_dict[device_type_str].keys()).index(param_name_str)
                value = device_info['values'][param_index]
                extract_values.append(value)

    return extract_values


# Test Code
# filepath = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/DC.raw/dcOpInfo.info.encode"
# value_dict = parse_device_values(filepath)
# param_dict = parse_device_param(filepath)
# device_type = "bsim4"
# param_name = "region"
# print(extract_group_params(device_type, param_name, value_dict, param_dict))

# Output
# [2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1]

# 0 cut-off
# 1 triode
# 2 sat
# 3 subth
# 4 breakdown

def extract_group_params_w_name(device_type_str, param_name_str, value_dict_list, param_list_dict):
    extract_values = {}
    for values_dict in value_dict_list:
        for device_id, device_info in values_dict.items():
            if device_info['Device_Type'] == device_type_str:
                param_index = list(param_list_dict[device_type_str].keys()).index(param_name_str)
                value = device_info['values'][param_index]
                extract_values[device_id] = value

    return extract_values


# Test Code
# filepath = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/netlist_assign_test/DC.raw/dcOpInfo.info.encode"
# value_dict = parse_device_values(filepath)
# param_dict = parse_device_param(filepath)
# device_type = "bsim4"
# param_name = "region"
# print(extract_group_params_w_name(device_type, param_name, value_dict, param_dict))

# Output
# {'I9.M17': 2, 'I9.M18': 2, 'I9.M22': 2, 'I9.M23': 2, 'I9.M24': 1, 'I9.M16': 1, 'I9.M36': 1, 'I9.M19': 2, 'I9.M21':
# 2, 'I9.M20': 2, 'I9.M11': 1, 'I9.M12': 1, 'I9.M13': 1, 'I9.M25': 2, 'I9.M35': 2, 'I9.M14': 1}


def extract_operation_region(dc_result_file_path):
    device_type_str = "bsim4"
    param_name_str = "region"
    value_dict_list = parse_device_values(dc_result_file_path)
    param_list_dict = parse_device_param(dc_result_file_path)
    operation_region_list = extract_group_params(device_type_str, param_name_str, value_dict_list, param_list_dict)

    return operation_region_list


def extract_operation_region_w_name(dc_result_file_path):
    device_type_str = "bsim4"
    param_name_str = "region"
    value_dict_list = parse_device_values(dc_result_file_path)
    param_list_dict = parse_device_param(dc_result_file_path)
    operation_region_list = extract_group_params_w_name(device_type_str, param_name_str, value_dict_list,
                                                        param_list_dict)

    return operation_region_list

# Test Code
# filepath = "/Users/hanwu/Downloads/Netlist_AXS/DC.raw/dcOpInfo.info"
# print(extract_operation_region_w_name(filepath))

# Output
# {'I9.M17': 2, 'I9.M18': 2, 'I9.M22': 2, 'I9.M23': 2, 'I9.M24': 1, 'I9.M16': 1, 'I9.M36': 1, 'I9.M19': 2, 'I9.M21': 2,
# 'I9.M20': 2, 'I9.M11': 1, 'I9.M12': 1, 'I9.M13': 1, 'I9.M25': 2, 'I9.M35': 2, 'I9.M14': 1}
