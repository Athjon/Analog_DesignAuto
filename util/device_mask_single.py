import numpy
from collections import OrderedDict
import copy
from numpy import array
import yaml

import copy
from collections import OrderedDict
import numpy

def masked_action_dict_mapping(device_mask_dict, step_action_dict):
    """
    Map the device mask dictionary to the step action dictionary.
    Supports both dictionary and NumPy array inputs for step_action_dict.
    
    Args:
        device_mask_dict: device mask dictionary
        step_action_dict: step action dictionary or NumPy array
        
    Returns:
        mapped_step_action_dict: mapped step action dictionary
    """
    # Handle case where input is a NumPy array
    if isinstance(step_action_dict, numpy.ndarray):
        # Convert the array to a dictionary with "single_agent" key
        step_action_dict = {"single_agent": step_action_dict}
    
    # Copy to avoid modifying original
    device_mask_dict_processed = copy.deepcopy(device_mask_dict)
    
    # Remove 'Other_Constrain' dict from device_mask_dict
    if device_mask_dict_processed and 'Other_Constrain' in device_mask_dict_processed:
        del device_mask_dict_processed['Other_Constrain']
    
    flattened_action_dict = {}
    for key, value in step_action_dict.items():
        if key in flattened_action_dict:
            raise ValueError(f'Duplicate key {key} found in device mask dictionary')
        flattened_action_dict[key] = value
    
    # Process masking rules
    if device_mask_dict_processed:
        for key in device_mask_dict_processed:
            if key in flattened_action_dict:
                new_values = device_mask_dict_processed[key]
                for new_key in new_values:
                    if new_key.endswith('_Match'):
                        match_action = flattened_action_dict[key][:2]
                        new_value = numpy.append(match_action, flattened_action_dict[new_key])
                    else:
                        new_value = flattened_action_dict[key]
                    flattened_action_dict[new_key] = new_value
    
    # Clean up keys
    mapped_action_dict = OrderedDict()
    for key, value in flattened_action_dict.items():
        if hasattr(key, 'removesuffix'):  # Python 3.9+
            new_key = key.removesuffix('_Match')
        else:
            # For Python < 3.9
            new_key = key[:-6] if key.endswith('_Match') else key
        mapped_action_dict[new_key] = value
    
    mapped_step_action_dict = copy.deepcopy(mapped_action_dict)
    return mapped_step_action_dict

# def masked_action_dict_mapping(device_mask_dict, step_action_dict):
#     """
#     Map the device mask dictionary to the step action dictionary.
#     :param device_mask_dict: device mask dictionary
#     :param step_action_dict: step action dictionary
#     :return: mapped_step_action_dict: mapped step action dictionary
#     """

#     device_mask_dict_processed = copy.deepcopy(device_mask_dict)

#     # Remove 'Other_Constrain' dict from device_mask_dict
#     if 'Other_Constrain' in device_mask_dict_processed:
#         del device_mask_dict_processed['Other_Constrain']

#     flattened_action_dict = {}
#     for key,value in step_action_dict.items():
#         if key in flattened_action_dict:
#             raise ValueError(f'Duplicate key {key} found in device mask dictionary')
#         flattened_action_dict[key] = value
#     # print(flattened_action_dict)

#     for key in device_mask_dict_processed:
#         if key in flattened_action_dict:
#             new_values = device_mask_dict_processed[key]
#             for new_key in new_values:
#                 if new_key.endswith('_Match'):
#                     match_action = flattened_action_dict[key][:2]
#                     new_value = numpy.append(match_action, flattened_action_dict[new_key])
#                 else:
#                     new_value = flattened_action_dict[key]
#                 flattened_action_dict[new_key] = new_value
#     # print(flattened_action_dict)

#     # Clean up keys
#     mapped_action_dict = OrderedDict()
#     for key, value in flattened_action_dict.items():
#         new_key = key.removesuffix('_Match')  # This works in Python 3.9+
#         mapped_action_dict[new_key] = value

#     mapped_step_action_dict = copy.deepcopy(mapped_action_dict)

    # for key in device_mask_dict:
    #     for agent_key in step_action_dict:
    #         if key in step_action_dict[agent_key]:
    #             new_values = device_mask_dict[key]
    #             for new_key in new_values:
    #                 if new_key.endswith('_Match'):
    #                     match_action = step_action_dict[agent_key][key][:2]
    #                     new_value = numpy.append(match_action, step_action_dict[agent_key][new_key])
    #                 else:
    #                     new_value = step_action_dict[agent_key][key]
    #                 step_action_dict[agent_key][new_key] = new_value

    # mapped_action_dict = OrderedDict()
    # for agent, agent_dict in step_action_dict.items():
    #     new_agent_dict = OrderedDict()
    #     for key, value in agent_dict.items():
    #         if key.endswith('_Match'):
    #             new_key = key.replace('_Match', '')
    #         else:
    #             new_key = key

    #         new_agent_dict[new_key] = value
    #     mapped_action_dict[agent] = new_agent_dict
    # mapped_step_action_dict = copy.deepcopy(mapped_action_dict)
    # return mapped_step_action_dict


# Test Code

# device_mask_yaml_path = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/config/config_Haoqiang/device_mask.yaml"
# with open(device_mask_yaml_path, 'r') as file:
#     device_mask_dict = yaml.safe_load(file)
# print
# action_space = {'Agent_1': OrderedDict([('M0', array([0.54674484, 0.25847102, 0.27725503]))]), 'Agent_2':
# OrderedDict([('C2', array([0.57240988])), ('C3', array([0.83542615])), ('M11_Match', array([0.72357407])), ('M14',
# array([0.14192225, 0.69342182, 0.05768928])), ('M16', array([0.98959885, 0.58093919, 0.11830176])), ('M18_Match',
# array([0.47449955])), ('M19', array([0.93148546, 0.23587164, 0.8388987 ])), ('M20_Match', array([0.4656418])),
# ('M6_Match', array([0.47810074])), ('ibias', array([0.51908655]))]), 'Agent_3': OrderedDict([('C0',
# array([0.37611206])), ('M1', array([0.89895646, 0.98283317, 0.13459567])), ('M10', array([0.38748166, 0.53243692,
# 0.97795154])), ('M12', array([0.9580772 , 0.28345907, 0.93970978])), ('M2_Match', array([0.96475009])),
# ('M3_Match', array([0.03660552])), ('M4_Match', array([0.25983479])), ('M5', array([0.78713286, 0.81911009,
# 0.05963406])), ('M7_Match', array([0.71436175])), ('M8', array([0.17287862, 0.47499465, 0.56030395])), ('M9',
# array([0.33406592, 0.13666506, 0.27619794])), ('R0', array([0.49225847]))]), 'Agent_4': OrderedDict([('C1',
# array([0.00486522])), ('C4', array([0.34878068])), ('C8', array([0.81525437])), ('M13_Match', array([0.30697816])),
# ('M21', array([0.49171918, 0.31032125, 0.14518133])), ('M22_Match', array([0.22680378])), ('M23',
# array([0.26137663, 0.89399361, 0.13530297])), ('M24', array([0.46692201, 0.2067227 , 0.04495567])), ('M25',
# array([0.9102547 , 0.65796408, 0.4847875 ])), ('M27', array([0.31410649, 0.72579488, 0.46150037])), ('M28',
# array([0.78166016, 0.6206851 , 0.79229841])), ('M29_Match', array([0.02651546])), ('M30', array([0.00263441,
# 0.10953637, 0.57922666])), ('M31', array([0.86286497, 0.23171877, 0.95366433])), ('M33', array([0.7680436 ,
# 0.5741837 , 0.92171029])), ('M34_Match', array([0.70309832])), ('M35_Match', array([0.03227022])), ('M36_Match',
# array([0.1089344])), ('M37', array([0.83543717, 0.35578124, 0.05358355]))]), 'Agent_5': OrderedDict([('M40',
# array([0.59285472, 0.50061883, 0.71590392])), ('M41', array([0.34331583, 0.46231484, 0.6527113 ])), ('M42_Match',
# array([0.33592356])), ('M43_Match', array([0.77138011])), ('M44', array([0.77233861, 0.99607863, 0.84046217])),
# ('M45_Match', array([0.33824189])), ('M46', array([0.24335651, 0.86807357, 0.34040986])), ('M47_Match',
# array([0.37747644]))])}

# print(masked_action_dict_mapping(device_mask_dict, action_space))

# Output

# OrderedDict([('M0', array([0.54674484, 0.25847102, 0.27725503])), ('C2', array([0.57240988])), ('C3',
# array([0.83542615])), ('M11', array([0.38748166, 0.53243692, 0.72357407])), ('M14', array([0.14192225, 0.69342182,
# 0.05768928])), ('M16', array([0.98959885, 0.58093919, 0.11830176])), ('M18', array([0.38748166, 0.53243692,
# 0.47449955])), ('M19', array([0.93148546, 0.23587164, 0.8388987 ])), ('M20', array([0.38748166, 0.53243692,
# 0.4656418 ])), ('M6', array([0.89895646, 0.98283317, 0.47810074])), ('ibias', array([0.51908655])), ('C0',
# array([0.37611206])), ('M1', array([0.89895646, 0.98283317, 0.13459567])), ('M10', array([0.38748166, 0.53243692,
# 0.97795154])), ('M12', array([0.9580772 , 0.28345907, 0.93970978])), ('M2', array([0.38748166, 0.53243692,
# 0.96475009])), ('M3', array([0.33406592, 0.13666506, 0.03660552])), ('M4', array([0.38748166, 0.53243692,
# 0.25983479])), ('M5', array([0.78713286, 0.81911009, 0.05963406])), ('M7', array([0.38748166, 0.53243692,
# 0.71436175])), ('M8', array([0.17287862, 0.47499465, 0.56030395])), ('M9', array([0.33406592, 0.13666506,
# 0.27619794])), ('R0', array([0.49225847])), ('C1', array([0.00486522])), ('C4', array([0.34878068])), ('C8',
# array([0.81525437])), ('M13', array([0.7680436 , 0.5741837 , 0.30697816])), ('M21', array([0.49171918, 0.31032125,
# 0.14518133])), ('M22', array([0.49171918, 0.31032125, 0.22680378])), ('M23', array([0.26137663, 0.89399361,
# 0.13530297])), ('M24', array([0.46692201, 0.2067227 , 0.04495567])), ('M25', array([0.9102547 , 0.65796408,
# 0.4847875 ])), ('M27', array([0.31410649, 0.72579488, 0.46150037])), ('M28', array([0.78166016, 0.6206851 ,
# 0.79229841])), ('M29', array([0.00263441, 0.10953637, 0.02651546])), ('M30', array([0.00263441, 0.10953637,
# 0.57922666])), ('M31', array([0.86286497, 0.23171877, 0.95366433])), ('M33', array([0.7680436 , 0.5741837 ,
# 0.92171029])), ('M34', array([0.7680436 , 0.5741837 , 0.70309832])), ('M35', array([0.7680436 , 0.5741837 ,
# 0.03227022])), ('M36', array([0.38748166, 0.53243692, 0.1089344 ])), ('M37', array([0.83543717, 0.35578124,
# 0.05358355])), ('M40', array([0.59285472, 0.50061883, 0.71590392])), ('M41', array([0.34331583, 0.46231484,
# 0.6527113 ])), ('M42', array([0.34331583, 0.46231484, 0.33592356])), ('M43', array([0.77233861, 0.99607863,
# 0.77138011])), ('M44', array([0.77233861, 0.99607863, 0.84046217])), ('M45', array([0.77233861, 0.99607863,
# 0.33824189])), ('M46', array([0.24335651, 0.86807357, 0.34040986])), ('M47', array([0.24335651, 0.86807357,
# 0.37747644])), ('M17', array([0.98959885, 0.58093919, 0.11830176])), ('M15', array([0.14192225, 0.69342182,
# 0.05768928])), ('M38', array([0.83543717, 0.35578124, 0.05358355])), ('M32', array([0.86286497, 0.23171877,
# 0.95366433])), ('M26', array([0.31410649, 0.72579488, 0.46150037]))])

