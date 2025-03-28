import yaml
import random
from collections import OrderedDict


def check_device_mask_consistency(init_param, device_mask_dict):
    inconsistencies = []
    for master_device, slave_devices in device_mask_dict.items():

        if master_device == 'Other_Constrain':
            for constrain in slave_devices:
                for master_param, slave_param in constrain.items():
                    if master_param in init_param and slave_param in init_param:
                        if init_param[master_param] != init_param[slave_param]:
                            inconsistencies.append(f"{master_param} ({init_param[master_param]}) != {slave_param} ({init_param[slave_param]})")
                    else:
                        inconsistencies.append(f"Parameter {master_param} or {slave_param} not found in init_param")

        else:
            for slave_device in slave_devices:
                is_match = slave_device.endswith('_Match')
                slave_device = slave_device.replace('_Match', '')

                l_master, w_master, nf_master = f'l_{master_device}', f'w_{master_device}_per_finger', f'nf_{master_device}'
                l_slave, w_slave, nf_slave = f'l_{slave_device}', f'w_{slave_device}_per_finger', f'nf_{slave_device}'

                if l_master in init_param and l_slave in init_param:
                    if init_param[l_master] != init_param[l_slave]:
                        inconsistencies.append(f"{l_master} ({init_param[l_master]}) != {l_slave} ({init_param[l_slave]})")
                if w_master in init_param and w_slave in init_param:
                    if init_param[w_master] != init_param[w_slave]:
                        inconsistencies.append(f"{w_master} ({init_param[w_master]}) != {w_slave} ({init_param[w_slave]})")

                if not is_match:
                    if nf_master in init_param and nf_slave in init_param:
                        if init_param[nf_master] != init_param[nf_slave]:
                            inconsistencies.append(
                                f"{nf_master} ({init_param[nf_master]}) != {nf_slave} ({init_param[nf_slave]})")

    return inconsistencies


def gen_init_param(init_method_flag, init_param_path, action_mask_flag, device_mask_dict, param_space):
    # Load device mask configuration if action_mask_flag is True

    device_mask = {}

    if action_mask_flag:
        device_mask = device_mask_dict

    init_param = OrderedDict()

    # Determine initialization method if init_method_flag is 'mixed'
    if init_method_flag == 'mixed':
        rand_num = random.random()
        if rand_num <= 0.7:
            print(f"Initialing!!!init method: mixed -> file")
            init_method_flag = 'file'
        else:
            print(f"Initialing!!!init method: mixed -> random")
            init_method_flag = 'random'

    # Load initial parameters from a file if init_method_flag is 'file'
    if init_method_flag == 'file':
        with open(init_param_path, 'r') as f:
            param_sets = yaml.safe_load(f)
            init_param = random.choice(param_sets)

        if device_mask_dict:
            inconsistencies = check_device_mask_consistency(init_param, device_mask_dict)
            if inconsistencies:
                print(f"Device mask inconsistency found: {inconsistencies}")
                raise ValueError("Device mask inconsistency found")

        print(f"Initialing!!!init method: file, init param: {init_param}")

    # Initialize parameters to the middle value of their range if init_method_flag is 'half'
    if init_method_flag == 'half':
        for param, value_list in param_space.items():
            n = len(value_list)
            middle_index = n // 2 - 1 if n % 2 == 0 else n // 2
            init_param[param] = value_list[middle_index]

        if device_mask_dict:
            inconsistencies = check_device_mask_consistency(init_param, device_mask_dict)
            if inconsistencies:
                print(f"Device mask inconsistency found: {inconsistencies}")
                raise ValueError("Device mask inconsistency found")

        print(f"Initialing!!!init method: half, init param: {init_param}")

    # Initialize parameters randomly within their range if init_method_flag is 'random'
    if init_method_flag == 'random':
        for param, value_list in param_space.items():
            n = len(value_list)
            random_index = random.randint(0, n - 1)
            init_param[param] = value_list[random_index]

        # Apply device mask to ensure matching parameter values across related devices
        if action_mask_flag and device_mask_dict:
            for master_device, slave_devices in device_mask_dict.items():
                if master_device == 'Other_Constrain':
                    for constraint in slave_devices:
                        for key, value in constraint.items():
                            if key in init_param and value in init_param:
                                init_param[key] = init_param[value]
                            else:
                                print(f"Error: {key} or {value} not found in init_param")
                                raise ValueError(f"Missing parameter: {key} or {value}")
                else:
                    for slave_device in slave_devices:
                        is_match = slave_device.endswith('_Match')
                        slave_device = slave_device.replace('_Match', '')

                        l_master = f'l_{master_device}'
                        w_master = f'w_{master_device}_per_finger'
                        nf_master = f'nf_{master_device}'

                        l_slave = f'l_{slave_device}'
                        w_slave = f'w_{slave_device}_per_finger'
                        nf_slave = f'nf_{slave_device}'

                        if l_master in init_param and l_slave in init_param:
                            init_param[l_slave] = init_param[l_master]
                        if w_master in init_param and w_slave in init_param:
                            init_param[w_slave] = init_param[w_master]

                        if not is_match:
                            if nf_master in init_param and nf_slave in init_param:
                                init_param[nf_slave] = init_param[nf_master]

        print(f"Initialing!!!init method: random, init param: {init_param}")

    return init_param


# Test Code

# init_method_flag = 'random'
# init_param_path = '/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/config/config_Haoqiang/init_param.yaml'
# action_mask_flag = True
# device_mask_path = '/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/config/config_Haoqiang/device_mask.yaml'
# param_space_path = '/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/config/config_Haoqiang/param_range.yaml'
# with open(device_mask_path, 'r') as f:
#     device_mask_dict = yaml.safe_load(f)
# with open(param_space_path, 'r') as f:
#     param_range = yaml.safe_load(f)
# from gen_param_space import gen_param_space
# param_space = gen_param_space(param_range)
# gen_init_param(init_method_flag, init_param_path, action_mask_flag, device_mask_dict, param_space)

# Test Output

# Initialing!!!init method: random, init param: OrderedDict([('ibias', '1.5u'), ('C0', '5.8p'), ('C1', '36.0p'),
# ('C2', '5.0p'), ('C3', '3.2p'), ('C4', '7.0p'), ('C8', '1.0f'), ('R0', '3.6k'), ('w_M40_per_finger', '0.92u'),
# ('l_M40', '60.0n'), ('nf_M40', '4'), ('w_M0_per_finger', '10.0u'), ('l_M0', '60.0n'), ('nf_M0', '65'),
# ('w_M47_per_finger', '0.9u'), ('l_M47', '0.5u'), ('nf_M47', '3'), ('w_M46_per_finger', '0.9u'), ('l_M46', '0.5u'),
# ('nf_M46', '2'), ('w_M42_per_finger', '0.8u'), ('l_M42', '1.1u'), ('nf_M42', '10'), ('w_M41_per_finger', '0.8u'),
# ('l_M41', '1.1u'), ('nf_M41', '5'), ('w_M27_per_finger', '0.8u'), ('l_M27', '1.2u'), ('nf_M27', '5'),
# ('w_M26_per_finger', '0.8u'), ('l_M26', '1.2u'), ('nf_M26', '5'), ('w_M25_per_finger', '0.9u'), ('l_M25', '0.5u'),
# ('nf_M25', '3'), ('w_M24_per_finger', '0.6u'), ('l_M24', '1.3u'), ('nf_M24', '2'), ('w_M23_per_finger', '0.42u'),
# ('l_M23', '110.0n'), ('nf_M23', '3'), ('w_M19_per_finger', '0.75u'), ('l_M19', '0.83u'), ('nf_M19', '3'),
# ('w_M22_per_finger', '1.0u'), ('l_M22', '0.8u'), ('nf_M22', '10'), ('w_M15_per_finger', '0.8u'), ('l_M15', '1.0u'),
# ('nf_M15', '5'), ('w_M14_per_finger', '0.8u'), ('l_M14', '1.0u'), ('nf_M14', '5'), ('w_M6_per_finger', '0.7u'),
# ('l_M6', '0.7u'), ('nf_M6', '8'), ('w_M21_per_finger', '1.0u'), ('l_M21', '0.8u'), ('nf_M21', '4'),
# ('w_M1_per_finger', '0.7u'), ('l_M1', '0.7u'), ('nf_M1', '4'), ('w_M36_per_finger', '0.7u'), ('l_M36', '0.9u'),
# ('nf_M36', '7'), ('w_M35_per_finger', '0.7u'), ('l_M35', '1.2u'), ('nf_M35', '4'), ('w_M34_per_finger', '0.7u'),
# ('l_M34', '1.2u'), ('nf_M34', '1'), ('w_M13_per_finger', '0.7u'), ('l_M13', '1.2u'), ('nf_M13', '10'),
# ('w_M33_per_finger', '0.7u'), ('l_M33', '1.2u'), ('nf_M33', '7'), ('w_M32_per_finger', '1.0u'), ('l_M32', '1.8u'),
# ('nf_M32', '5'), ('w_M31_per_finger', '1.0u'), ('l_M31', '1.8u'), ('nf_M31', '5'), ('w_M20_per_finger', '0.7u'),
# ('l_M20', '0.9u'), ('nf_M20', '4'), ('w_M18_per_finger', '0.7u'), ('l_M18', '0.9u'), ('nf_M18', '8'),
# ('w_M11_per_finger', '0.7u'), ('l_M11', '0.9u'), ('nf_M11', '3'), ('w_M10_per_finger', '0.7u'), ('l_M10', '0.9u'),
# ('nf_M10', '3'), ('w_M8_per_finger', '0.9u'), ('l_M8', '0.72u'), ('nf_M8', '3'), ('w_M7_per_finger', '0.7u'),
# ('l_M7', '0.9u'), ('nf_M7', '2'), ('w_M12_per_finger', '0.5u'), ('l_M12', '0.62u'), ('nf_M12', '2'),
# ('w_M4_per_finger', '0.7u'), ('l_M4', '0.9u'), ('nf_M4', '1'), ('w_M2_per_finger', '0.7u'), ('l_M2', '0.9u'),
# ('nf_M2', '4'), ('w_M38_per_finger', '0.5u'), ('l_M38', '2.0u'), ('nf_M38', '2'), ('w_M37_per_finger', '0.5u'),
# ('l_M37', '2.0u'), ('nf_M37', '2'), ('w_M9_per_finger', '0.6u'), ('l_M9', '1.9u'), ('nf_M9', '1'),
# ('w_M5_per_finger', '0.9u'), ('l_M5', '0.42u'), ('nf_M5', '1'), ('w_M3_per_finger', '0.6u'), ('l_M3', '1.9u'),
# ('nf_M3', '4'), ('w_M43_per_finger', '0.7u'), ('l_M43', '0.5u'), ('nf_M43', '2'), ('w_M44_per_finger', '0.7u'),
# ('l_M44', '0.5u'), ('nf_M44', '2'), ('w_M30_per_finger', '0.5u'), ('l_M30', '0.6u'), ('nf_M30', '4'),
# ('w_M29_per_finger', '0.5u'), ('l_M29', '0.6u'), ('nf_M29', '4'), ('w_M28_per_finger', '0.18u'), ('l_M28',
# '0.36u'), ('nf_M28', '3'), ('w_M17_per_finger', '0.5u'), ('l_M17', '0.5u'), ('nf_M17', '4'), ('w_M16_per_finger',
# '0.5u'), ('l_M16', '0.5u'), ('nf_M16', '4'), ('w_M45_per_finger', '0.7u'), ('l_M45', '0.5u'), ('nf_M45', '10')])
