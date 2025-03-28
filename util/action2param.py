from collections import OrderedDict
from numpy import array
import yaml
import copy


def action2param(device_mask_flag, device_mask_dict, flatten_action, param_rang_dict):
    """
    Generate param with given continuous action.
    :param device_mask_flag:
    :param device_mask_dict:
    :param flatten_action:
    :param param_rang_dict:
    :return: param_dict
    """

    # print(flatten_action_space)

    # Extend action space via mask config
    masked_action_space = copy.deepcopy(flatten_action)

    # if device_mask_flag and device_mask_dict is not None:
    #     for master_device in flatten_action:
    #         if master_device in device_mask_dict:
    #             slave_devices = device_mask_dict[master_device]
    #             for slave_device in slave_devices:
    #                 param_value = flatten_action[master_device]
    #                 masked_action_space[slave_device] = param_value

    # print(masked_action_space)

    extend_action_space = {}
    for key, action in masked_action_space.items():
        if key.startswith('M'):
            for index, value in enumerate(action):
                key_name = None
                if index == 0:
                    key_name = f"w_{key}_per_finger"
                if index == 1:
                    key_name = f"l_{key}"
                if index == 2:
                    key_name = f"nf_{key}"
                extend_action_space[key_name] = value
        else:
            try:
                value = action[0]
            except (TypeError, KeyError, IndexError):
                value = action
            extend_action_space[key] = float(value)
    # print(extend_action_space)

    param_dict = OrderedDict()
    for key, value in param_rang_dict.items():
        for param in value['params']:
            variable_name = param['variable_name']
            value_range = param['value']['range']
            step = param['value']['step']
            action_value = extend_action_space[variable_name]
            # Check if the variable name starts with 'nf_' to determine if it should be integer
            is_integer = variable_name.startswith('nf_')
            param_value = parse_action2param(value_range, step, action_value, is_integer)
            # print(variable_name, value_range, step, action_value, is_integer)
            # print(param_value)
            param_dict[variable_name] = param_value

    # Apply Other_Constrain
    # print(f"Other_Constrain is {device_mask_dict}")
    if device_mask_flag and device_mask_dict is not None and 'Other_Constrain' in device_mask_dict:
        other_constrain = device_mask_dict['Other_Constrain']
        for constraint in other_constrain:
            for master_param, slave_param in constraint.items():
                if master_param in param_dict and slave_param in param_dict:
                    # print(f"slave_param is {slave_param} with value {param_dict[slave_param]} and master_param is {master_param} with value {param_dict[master_param]}")
                    param_dict[slave_param] = param_dict[master_param]
                    # print(f"Update, slave_param is {slave_param} with value {param_dict[slave_param]} and master_param is {master_param} with value {param_dict[master_param]}")
                else:
                    print(f"Warning: {master_param} or {slave_param} not found in param_dict")

    return param_dict


def parse_action2param(value_range, step, action_value, is_intege):
    """
    Parse action to param.
    :param value_range:
    :param step:
    :param action_value:
    :param is_intege:
    :return: param_value
    """
    min_value, max_value = value_range

    if isinstance(min_value, str):
        magnitude = ''.join(filter(str.isalpha, min_value))
        min_value, max_value, step = [float(x.replace(magnitude, '')) for x in [min_value, max_value, step]]
    else:
        magnitude = ''
        min_value, max_value, step = float(min_value), float(max_value), float(step)

    if min_value == max_value:
        nearest_value = min_value
    else:
        original_value = min_value + (max_value - min_value) * action_value
        nearest_value = min_value + round((original_value - min_value) / step) * step

    # Check within the range
    if nearest_value <= min_value:
        nearest_value = min_value
    if nearest_value >= max_value:
        nearest_value = max_value

    nearest_value = round(nearest_value, 2)

    if is_intege:
        nearest_value = str(int(nearest_value))
    else:
        nearest_value = f"{nearest_value}{magnitude}"

    return nearest_value


# Test Code

# device_mask_flag = True
# device_mask = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/config/config_Haoqiang/device_mask_bk.yaml"
# with open(device_mask, 'r') as file:
#     device_mask_dict = yaml.safe_load(file)

# action_space = OrderedDict([('M0', array([0.54674484, 0.25847102, 0.27725503])), ('C2', array([0.57240988])),
# ('C3', array([0.83542615])), ('M11', array([0.38748166, 0.53243692, 0.72357407])), ('M14', array([0.14192225,
# 0.69342182, 0.05768928])), ('M16', array([0.98959885, 0.58093919, 0.11830176])), ('M18', array([0.38748166,
# 0.53243692, 0.89])), ('M19', array([0.93148546, 0.23587164, 0.8388987 ])), ('M20', array([0.38748166, 0.53243692,
# 0.4656418 ])), ('M6', array([0.89895646, 0.98283317, 0.4781])), ('ibias', array([0.51908655])), ('C0',
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

# param_range = "/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/config/config_Haoqiang/param_range.yaml"
# with open(param_range, 'r') as file:
#     param_rang_dict = yaml.safe_load(file)
# print(action2param(device_mask_flag, device_mask_dict, action_space, param_rang_dict))

# Output
#
# OrderedDict([('ibias', '1.5u'), ('C0', '3.8p'), ('C1', '1.0p'), ('C2', '9.0p'), ('C3', '4.2p'), ('C4', '6.0p'),
# ('C8', '12.0f'), ('R0', '7.6k'), ('w_M40_per_finger', '0.64u'), ('l_M40', '60.0n'), ('nf_M40', '4'),
# ('w_M0_per_finger', '10.0u'), ('l_M0', '60.0n'), ('nf_M0', '23'), ('w_M47_per_finger', '0.6u'), ('l_M47', '1.8u'),
# ('nf_M47', '3'), ('w_M46_per_finger', '0.6u'), ('l_M46', '1.8u'), ('nf_M46', '2'), ('w_M42_per_finger', '0.7u'),
# ('l_M42', '1.2u'), ('nf_M42', '4'), ('w_M41_per_finger', '0.7u'), ('l_M41', '1.2u'), ('nf_M41', '7'),
# ('w_M27_per_finger', '0.7u'), ('l_M27', '1.6u'), ('nf_M27', '3'), ('w_M26_per_finger', '0.7u'), ('l_M26', '1.6u'),
# ('nf_M26', '3'), ('w_M25_per_finger', '0.95u'), ('l_M25', '0.8u'), ('nf_M25', '2'), ('w_M24_per_finger', '0.55u'),
# ('l_M24', '0.9u'), ('nf_M24', '1'), ('w_M23_per_finger', '0.36u'), ('l_M23', '450.0n'), ('nf_M23', '2'),
# ('w_M19_per_finger', '0.95u'), ('l_M19', '0.28u'), ('nf_M19', '4'), ('w_M22_per_finger', '0.7u'), ('l_M22',
# '1.0u'), ('nf_M22', '3'), ('w_M15_per_finger', '0.6u'), ('l_M15', '1.5u'), ('nf_M15', '1'), ('w_M14_per_finger',
# '0.6u'), ('l_M14', '1.5u'), ('nf_M14', '1'), ('w_M6_per_finger', '1.8u'), ('l_M6', '1.95u'), ('nf_M6', '5'),
# ('w_M21_per_finger', '0.7u'), ('l_M21', '1.0u'), ('nf_M21', '2'), ('w_M1_per_finger', '1.8u'), ('l_M1', '1.95u'),
# ('nf_M1', '10'), ('w_M36_per_finger', '0.6u'), ('l_M36', '1.3u'), ('nf_M36', '2'), ('w_M35_per_finger', '0.9u'),
# ('l_M35', '1.4u'), ('nf_M35', '1'), ('w_M34_per_finger', '0.9u'), ('l_M34', '1.4u'), ('nf_M34', '7'),
# ('w_M13_per_finger', '0.9u'), ('l_M13', '1.4u'), ('nf_M13', '4'), ('w_M33_per_finger', '0.9u'), ('l_M33', '1.4u'),
# ('nf_M33', '9'), ('w_M32_per_finger', '0.9u'), ('l_M32', '1.1u'), ('nf_M32', '5'), ('w_M31_per_finger', '0.9u'),
# ('l_M31', '1.1u'), ('nf_M31', '5'), ('w_M20_per_finger', '0.6u'), ('l_M20', '1.3u'), ('nf_M20', '5'),
# ('w_M18_per_finger', '0.6u'), ('l_M18', '1.3u'), ('nf_M18', '5'), ('w_M11_per_finger', '0.6u'), ('l_M11', '1.3u'),
# ('nf_M11', '8'), ('w_M10_per_finger', '0.6u'), ('l_M10', '1.3u'), ('nf_M10', '10'), ('w_M8_per_finger', '0.5u'),
# ('l_M8', '0.62u'), ('nf_M8', '3'), ('w_M7_per_finger', '0.6u'), ('l_M7', '1.3u'), ('nf_M7', '7'),
# ('w_M12_per_finger', '1.0u'), ('l_M12', '0.76u'), ('nf_M12', '5'), ('w_M4_per_finger', '0.6u'), ('l_M4', '1.3u'),
# ('nf_M4', '3'), ('w_M2_per_finger', '0.6u'), ('l_M2', '1.3u'), ('nf_M2', '10'), ('w_M38_per_finger', '0.9u'),
# ('l_M38', '1.0u'), ('nf_M38', '1'), ('w_M37_per_finger', '0.9u'), ('l_M37', '1.0u'), ('nf_M37', '1'),
# ('w_M9_per_finger', '0.6u'), ('l_M9', '0.6u'), ('nf_M9', '2'), ('w_M5_per_finger', '0.9u'), ('l_M5', '0.86u'),
# ('nf_M5', '1'), ('w_M3_per_finger', '0.6u'), ('l_M3', '0.6u'), ('nf_M3', '1'), ('w_M43_per_finger', '0.9u'),
# ('l_M43', '2.0u'), ('nf_M43', '8'), ('w_M44_per_finger', '0.9u'), ('l_M44', '2.0u'), ('nf_M44', '9'),
# ('w_M30_per_finger', '0.2u'), ('l_M30', '0.25u'), ('nf_M30', '3'), ('w_M29_per_finger', '0.2u'), ('l_M29',
# '0.25u'), ('nf_M29', '1'), ('w_M28_per_finger', '0.42u'), ('l_M28', '0.34u'), ('nf_M28', '4'), ('w_M17_per_finger',
# '1.0u'), ('l_M17', '1.3u'), ('nf_M17', '1'), ('w_M16_per_finger', '1.0u'), ('l_M16', '1.3u'), ('nf_M16', '1'),
# ('w_M45_per_finger', '0.9u'), ('l_M45', '2.0u'), ('nf_M45', '4')])
