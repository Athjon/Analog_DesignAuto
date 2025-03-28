from collections import OrderedDict
import yaml
from numpy import array
from typing import Dict, Tuple, Union


# Helper function: Split a value into its numeric and unit parts.
def split_value_unit(value: Union[str, int]) -> Tuple[float, str]:
    if isinstance(value, int):
        return float(value), ''  # Convert int to float and return with an empty unit
    if isinstance(value, float):
        return value, ''
    else:
        for i, char in enumerate(value):
            if not char.isdigit() and char != '.':
                return float(value[:i]), value[i:]  # Split the string into numeric and unit parts
        return float(value), ''


# Helper function: Combine a numeric value with a unit into a string.
def combine_value_unit(value: float, unit: str) -> str:
    return f"{value}{unit}"


# Helper function: Adjust a value within a given range.
def adjust_value_within_range(value: str, step: str, range_limits: list, action: int, is_integer: bool = False) -> str:
    value_num, unit = split_value_unit(value)
    step_num, _ = split_value_unit(step)
    min_limit, max_limit = split_value_unit(range_limits[0])[0], split_value_unit(range_limits[1])[0]

    if action == 0:  # Decrease
        value_num = max(min_limit, value_num - step_num)
    elif action == 2:  # Increase
        value_num = min(max_limit, value_num + step_num)
    elif action == 1:  # Keep the same
        pass

    value_num = round(value_num, 2)

    if is_integer:  # Convert to integer if needed
        value_num = int(round(value_num))

    return combine_value_unit(value_num, unit)


# Main function: Update parameters based on the action_idx_dict and cur_param_dict
def update_parameters(action_idx_dict, cur_param_dict, yaml_config):

    updated_params = OrderedDict()

    with open(yaml_config, 'r') as file:
        yaml_config = yaml.safe_load(file)

    for key, actions in action_idx_dict.items():
        if key.startswith('M'):
            for i, action in enumerate(actions):
                param_keys = [f"w_{key}_per_finger", f"l_{key}", f"nf_{key}"]  # Construct parameter keys
                if i < len(param_keys):
                    param_key = param_keys[i]
                    current_value = cur_param_dict[param_key]
                    step_info = yaml_config[key]['params'][i]['value']['step']
                    range_info = yaml_config[key]['params'][i]['value']['range']
                    is_integer = param_key.startswith("nf_")  # Check if the parameter is of type nf_X
                    updated_value = adjust_value_within_range(current_value, step_info, range_info, action, is_integer)
                    updated_params[param_key] = updated_value
        else:
            current_value = cur_param_dict[key]
            param_index = None
            for index, value in enumerate(yaml_config['other_variable']['params']):
                if value['variable_name'] == key:
                    param_index = index
                    break
                else:
                    continue
            step_info = yaml_config['other_variable']['params'][param_index]['value']['step']
            range_info = yaml_config['other_variable']['params'][param_index]['value']['range']
            updated_value = adjust_value_within_range(current_value, step_info, range_info, actions[0])
            updated_params[key] = updated_value

    return updated_params


# Test Code

# action_idx = OrderedDict([('IBP', array([2])), ('M1', array([2, 2, 0])), ('M4', array([0, 1, 0])), ('M5', array([0,
# 0, 2])), ('M6', array([0, 0, 2])), ('M9', array([1, 1, 2])), ('M10', array([1, 1, 2])), ('M12', array([2, 0, 1])),
# ('M11', array([2, 0, 1])), ('M13', array([2, 0, 1])), ('M2', array([2, 0, 1])), ('CM', array([0])), ('M0',
# array([2, 0, 0])), ('R0', array([0])), ('R1', array([1]))])

# cur_param_dict = {'IBP': '1u', 'R0': '12.0k', 'R1': '24.0k', 'CM': '80.0p', 'w_M13_per_finger': '2.0u',
# 'l_M13': '2.0u', 'nf_M13': 1, 'w_M12_per_finger': '2.0u', 'l_M12': '2.0u', 'nf_M12': 4, 'w_M11_per_finger': '2.0u',
# 'l_M11': '2.0u', 'nf_M11': 1, 'w_M2_per_finger': '2.0u', 'l_M2': '2.0u', 'nf_M2': 4, 'w_M10_per_finger': '2.0u',
# 'l_M10': '0.5u', 'nf_M10': 2, 'w_M9_per_finger': '2.0u', 'l_M9': '0.5u', 'nf_M9': 2, 'w_M6_per_finger': '2.0u',
# 'l_M6': '2.0u', 'nf_M6': 1, 'w_M5_per_finger': '2.0u', 'l_M5': '2.0u', 'nf_M5': 1, 'w_M4_per_finger': '1.0u',
# 'l_M4': '1.0u', 'nf_M4': 1, 'w_M1_per_finger': '1.0u', 'l_M1': '1.0u', 'nf_M1': 3, 'w_M0_per_finger': '300.0u',
# 'l_M0': '0.5u', 'nf_M0': 1}

# yaml_config = "../config/param_range.yaml"
# updated_params = update_parameters(action_idx, cur_param_dict, yaml_config)
# print(updated_params)

# action_idx = OrderedDict([('CM', array([2]))])
# cur_param_dict = {'CM': '20.0p'}
# yaml_config = "../config/param_range.yaml"
# updated_params = update_parameters(action_idx, cur_param_dict, yaml_config)
# print(updated_params)

# Output

# OrderedDict([('IBP', '2.0u'), ('w_M1_per_finger', '1.0u'), ('l_M1', '1.0u'), ('nf_M1', '2'), ('w_M4_per_finger',
# '1.0u'), ('l_M4', '1.0u'), ('nf_M4', '1'), ('w_M5_per_finger', '2.0u'), ('l_M5', '2.5u'), ('nf_M5', '2'),
# ('w_M6_per_finger', '2.0u'), ('l_M6', '2.5u'), ('nf_M6', '2'), ('w_M9_per_finger', '2.0u'), ('l_M9', '1.0u'),
# ('nf_M9', '3'), ('w_M10_per_finger', '2.0u'), ('l_M10', '1.0u'), ('nf_M10', '3'), ('w_M12_per_finger', '2.0u'),
# ('l_M12', '1.5u'), ('nf_M12', '5'), ('w_M11_per_finger', '2.0u'), ('l_M11', '1.5u'), ('nf_M11', '2'),
# ('w_M13_per_finger', '2.0u'), ('l_M13', '1.5u'), ('nf_M13', '2'), ('w_M2_per_finger', '2.0u'), ('l_M2', '1.5u'),
# ('nf_M2', '5'), ('CM', '5.0p'), ('w_M0_per_finger', '250.0u'), ('l_M0', '1.0u'), ('nf_M0', '2'), ('R0', '1.2k'),
# ('R1', '1.4k')])

