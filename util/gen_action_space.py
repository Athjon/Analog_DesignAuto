import copy

import gymnasium
import yaml
import numpy as np

def gen_action_space(agent_assign_yaml_path):
    """
    Generate action space for the environment.
    Parameters:
        agent_assign.yaml: the file that defines the action space including agent name and corresponding device name.
    Returns:
        Dict w/ Several MultiDiscrete: the action space for the multi-agent environment.
    """

    # Define operation:
    # 0: -1 index;
    # 1: Keep the same;
    # 2: +1 index;
    operation_number = 3

    with open(agent_assign_yaml_path, 'r') as file:
        agent_assign = yaml.safe_load(file)

    action_space_dict = {}
    # for group_name, element in agent_assign.items():
    #     action_space_dict[group_name] = gymnasium.spaces.MultiDiscrete([operation_number] * len(element))

    # For different type device, their action space is different. e.g., MOSFET -> MultiDiscrete([3]*3)
    for group_name, device_list in agent_assign.items():
        space_dict = {}
        for device in device_list:
            if device.startswith('M'):
                space = gymnasium.spaces.MultiDiscrete([operation_number] * 3)
            else:
                space = gymnasium.spaces.MultiDiscrete([operation_number])
            space_dict[device] = space
        action_space_dict[group_name] = gymnasium.spaces.Dict(space_dict)
    return gymnasium.spaces.Dict(action_space_dict)


# Test Code
# agent_assign_yaml_path = "../config_3/agent_assign.yaml"
# action_space_dict = gen_action_space(agent_assign_yaml_path)
# print(action_space_dict)
#
# all_action = {}
# for group_name, action_space in action_space_dict.items():
#     action = action_space_dict[group_name].sample()
#     all_action[group_name] = action
# print(all_action)
#
# all_action_flatten = OrderedDict()
# for action in all_action.values():
#     all_action_flatten.update(action)
# print(all_action_flatten)

# Output Dict('Agent_1': Dict('M13': MultiDiscrete([3 3 3]), 'M14': MultiDiscrete([3 3 3]), 'M16': MultiDiscrete([3 3
# 3]), 'M23': MultiDiscrete([3 3 3]), 'M24': MultiDiscrete([3 3 3]), 'M25': MultiDiscrete([3 3 3]),
# 'M35': MultiDiscrete([3 3 3]), 'M36': MultiDiscrete([3 3 3])), 'Agent_2': Dict('M17': MultiDiscrete([3 3 3]),
# 'M18': MultiDiscrete([3 3 3])), 'Agent_3': Dict('M11': MultiDiscrete([3 3 3]), 'M12': MultiDiscrete([3 3 3]),
# 'M19': MultiDiscrete([3 3 3]), 'M20': MultiDiscrete([3 3 3]), 'M21': MultiDiscrete([3 3 3]), 'M22': MultiDiscrete([
# 3 3 3])), 'Agent_4': Dict('IB': MultiDiscrete([3])))
#
# {'Agent_1': OrderedDict([('M13', array([1, 2, 2])), ('M14', array([2, 1, 1])), ('M16', array([1, 2, 1])), ('M23',
# array([2, 0, 2])), ('M24', array([2, 0, 1])), ('M25', array([2, 2, 2])), ('M35', array([0, 0, 2])), ('M36',
# array([0, 1, 2]))]), 'Agent_2': OrderedDict([('M17', array([0, 0, 1])), ('M18', array([1, 0, 1]))]), 'Agent_3':
# OrderedDict([('M11', array([1, 0, 0])), ('M12', array([0, 2, 0])), ('M19', array([0, 2, 0])), ('M20', array([2, 0,
# 1])), ('M21', array([2, 2, 0])), ('M22', array([1, 2, 2]))]), 'Agent_4': OrderedDict([('IB', array([1]))])}
#
# OrderedDict([('M13', array([1, 2, 2])), ('M14', array([2, 1, 1])), ('M16', array([1, 2, 1])), ('M23', array([2, 0,
# 2])), ('M24', array([2, 0, 1])), ('M25', array([2, 2, 2])), ('M35', array([0, 0, 2])), ('M36', array([0, 1, 2])),
# ('M17', array([0, 0, 1])), ('M18', array([1, 0, 1])), ('M11', array([1, 0, 0])), ('M12', array([0, 2, 0])), ('M19',
# array([0, 2, 0])), ('M20', array([2, 0, 1])), ('M21', array([2, 2, 0])), ('M22', array([1, 2, 2])), ('IB',
# array([1]))])


def gen_masked_action_space(device_mask_dict, agent_assign_dict):
    """
    Generate action space for the environment.
    Parameters:
        device_mask_dict: dict to the YAML file that contains devices to be masked.
        agent_assign_dict: dict to the YAML file that defines the action space including agent name and corresponding
        device name.
    Returns:
        gymnasium.spaces.Dict: The action space for the multi-agent environment as a dictionary of MultiDiscrete spaces.
    """

    # Define the number of operations for action space
    # 0: -1 index;
    # 1: Keep the same;
    # 2: +1 index;

    if device_mask_dict:
        device_list = []
        for device_group in agent_assign_dict.values():
            device_list.extend(device_group)
        # print(f"Device list is {device_list}")

        for values in device_mask_dict.values():
            for value in values:
                device_name = value.replace('_Match', '')  # Remove _Match suffix if exists
                if device_name not in device_list:
                    raise ValueError(f"Device {device_name} not found in device list")

        for key, values in device_mask_dict.items():
            for value in values:
                if value.endswith('_Match'):
                    # If the value ends with '_Match', replace the corresponding device in the list with the value
                    device_list = [device if device != value.replace('_Match', '') else value for device in device_list]
                else:
                    # If the value does not end with '_Match', remove the corresponding device from the list
                    device_list = [device for device in device_list if device != value]
        # print(f"Device list after mask is {device_list}")

        action_space = {}
        for group_name, group_device_list in agent_assign_dict.items():
            space_dict = {}
            for device in device_list:
                if device in group_device_list:
                    if device.startswith('M') and not device.endswith('_Match'):
                        # If the device starts with 'M' and does not end with '_Match', create a MultiDiscrete space
                        # with 3 actions
                        space_dict[device] = gymnasium.spaces.MultiDiscrete([3, 3, 3])
                    else:
                        # Otherwise, create a MultiDiscrete space with 1 action
                        space_dict[device] = gymnasium.spaces.MultiDiscrete([3])
                if device.replace('_Match', '') in group_device_list and device.endswith('_Match'):
                    # If the device ends with '_Match', create a MultiDiscrete space with 1 action and
                    # add '_Match' suffix
                    space_dict[device] = gymnasium.spaces.MultiDiscrete([3])
            action_space[group_name] = gymnasium.spaces.Dict(space_dict)

    else:
        action_space = {}
        for group_name, device_list in agent_assign_dict.items():
            space_dict = {}
            for device in device_list:
                if device.startswith('M'):
                    space = gymnasium.spaces.MultiDiscrete([3, 3, 3])
                else:
                    space = gymnasium.spaces.MultiDiscrete([3])
                space_dict[device] = space
            action_space[group_name] = gymnasium.spaces.Dict(space_dict)

    # Return a Dict space containing all the action spaces
    return gymnasium.spaces.Dict(action_space)


# Test Code
# agent_assign_yaml_path = "../config/agent_assign.yaml"
# with open(agent_assign_yaml_path, 'r') as file:
#     agent_assign_dict = yaml.safe_load(file)
# device_mask_yaml_path = "../config/device_mask_test.yaml"
# with open(device_mask_yaml_path, 'r') as file:
#     device_mask_dict = yaml.safe_load(file)
# action_space_dict = gen_masked_action_space(True, device_mask_dict, agent_assign_dict)
# print(action_space_dict)

# Output

# Device list is ['IB1', 'IB2', 'IB3', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'CC1', 'IB4', 'IB5', 'M7', 'M8', 'M9',
# 'M10', 'M12', 'RS1', 'RS2', 'RS3', 'IB6', 'IB7', 'M13', 'M14', 'M15', 'M16', 'M18', 'RC2', 'RFF1', 'RFF2', 'CC2',
# 'CFF1', 'M0', 'RF1', 'RF2']

# Device list after mask is ['IB1', 'IB2', 'IB3', 'M1', 'M3', 'M5', 'CC1', 'IB4', 'IB5', 'M7', 'M9', 'M12', 'RS1',
# 'RS2', 'RS3', 'IB6', 'IB7', 'M13', 'M15', 'M18_Match', 'RC2', 'RFF1', 'RFF2', 'CC2', 'CFF1', 'M0', 'RF1', 'RF2']

# Dict('Agent_1': Dict('CC1': MultiDiscrete([3]), 'IB1': MultiDiscrete([3]), 'IB2': MultiDiscrete([3]),
# 'IB3': MultiDiscrete([3]), 'M1': MultiDiscrete([3 3 3]), 'M3': MultiDiscrete([3 3 3]), 'M5': MultiDiscrete([3 3
# 3])), 'Agent_2': Dict('IB4': MultiDiscrete([3]), 'IB5': MultiDiscrete([3]), 'M12': MultiDiscrete([3 3 3]),
# 'M7': MultiDiscrete([3 3 3]), 'M9': MultiDiscrete([3 3 3]), 'RS1': MultiDiscrete([3]), 'RS2': MultiDiscrete([3]),
# 'RS3': MultiDiscrete([3])), 'Agent_3': Dict('CC2': MultiDiscrete([3]), 'CFF1': MultiDiscrete([3]),
# 'IB6': MultiDiscrete([3]), 'IB7': MultiDiscrete([3]), 'M13': MultiDiscrete([3 3 3]), 'M15': MultiDiscrete([3 3 3]),
# 'M18_Match': MultiDiscrete([3]), 'RC2': MultiDiscrete([3]), 'RFF1': MultiDiscrete([3]), 'RFF2': MultiDiscrete([
# 3])), 'Agent_4': Dict('M0': MultiDiscrete([3 3 3]), 'RF1': MultiDiscrete([3]), 'RF2': MultiDiscrete([3])))


def gen_masked_continuous_action_space(device_mask_dict, agent_assign_dict):
    """
    Generate action space for the environment.
    Parameters:
        device_mask_dict: dict to the YAML file that contains devices to be masked.
        agent_assign_dict: dict to the YAML file that defines the action space including agent name and corresponding
        device name.
    Returns:
        gymnasium.spaces.Dict: The action space for the multi-agent environment as a dictionary of MultiDiscrete spaces.
    """

    triple_act_space = gymnasium.spaces.Box(low=0, high=1, shape=(3,), dtype=float)
    single_act_space = gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=float)

    # Early return if device_mask_dict is None or empty
    if not device_mask_dict:
        # Generate standard action space without masking
        continuous_action_space_dict = {}
        for group_name, device_list in agent_assign_dict.items():
            space_dict = {}
            for device in device_list:
                if device.startswith('M'):
                    space = triple_act_space
                else:
                    space = single_act_space
                space_dict[device] = space
            continuous_action_space_dict[group_name] = gymnasium.spaces.Dict(space_dict)
            continuous_action_space_dict_single = continuous_action_space_dict["single_agent"]
        return continuous_action_space_dict_single

    # Process device mask dictionary
    device_mask_dict_processed = copy.deepcopy(device_mask_dict)

    # Remove 'Other_Constrain' dict from device_mask_dict if exists
    if 'Other_Constrain' in device_mask_dict_processed:
        del device_mask_dict_processed['Other_Constrain']

    # Get list of all devices
    device_list = []
    for device_group in agent_assign_dict.values():
        device_list.extend(device_group)

    # Validate devices in mask dictionary
    for values in device_mask_dict_processed.values():
        for value in values:
            device_name = value.replace('_Match', '')  # Remove _Match suffix if exists
            if device_name not in device_list:
                raise ValueError(f"Device {device_name} not found in device list")

    # Process device masking
    for key, values in device_mask_dict_processed.items():
        for value in values:
            if value.endswith('_Match'):
                # If the value ends with '_Match', replace the corresponding device in the list with the value
                device_list = [device if device != value.replace('_Match', '') else value for device in device_list]
            else:
                # If the value does not end with '_Match', remove the corresponding device from the list
                device_list = [device for device in device_list if device != value]

    # Generate action space with masking
    continuous_action_space_dict = {}
    for group_name, group_device_list in agent_assign_dict.items():
        space_dict = {}
        for device in device_list:
            if device in group_device_list:
                if device.startswith('M') and not device.endswith('_Match'):
                    space_dict[device] = triple_act_space
                else:
                    space_dict[device] = single_act_space
            if device.replace('_Match', '') in group_device_list and device.endswith('_Match'):
                space_dict[device] = single_act_space
        continuous_action_space_dict[group_name] = gymnasium.spaces.Dict(space_dict)
        
    print(f"continuous_action_space_dict is {continuous_action_space_dict}")
    continuous_action_space_dict_single = continuous_action_space_dict["single_agent"]
    

    return continuous_action_space_dict_single




def gen_single_continuous_action_space(device_mask_dict, device_list):
    """
    单智能体版本动作空间生成函数
    参数：
        device_list: 设备名称列表 (如 ['M0', 'ibias', 'M11',...])
        device_mask_dict: 掩码配置字典
    返回：
        gymnasium.spaces.Box: 单智能体连续动作空间
    """
    # 定义基础动作空间类型
    triple_act_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=float)
    single_act_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float)

    # 处理设备掩码
    processed_devices = _process_mask(device_list, device_mask_dict)

    # 生成动作空间字典
    space_dict = {}
    for device in processed_devices:
        if device.startswith('M') and not device.endswith('_Match'):
            space_dict[device] = triple_act_space
        else:
            space_dict[device] = single_act_space

    # 转换为单智能体Box空间
    total_dims = sum(
        3 if dev.startswith('M') and not dev.endswith('_Match') else 1 
        for dev in space_dict.keys()
    )
    
    return gymnasium.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(total_dims,),
        dtype=float
    )

def _process_mask(original_devices, mask_dict):
    """处理设备掩码逻辑"""
    processed = copy.deepcopy(original_devices)
    
    # 移除特殊约束部分
    mask_dict = {k: v for k, v in (mask_dict or {}).items() 
                if k != 'Other_Constrain'}

    # 应用掩码规则
    for mask_type, devices in mask_dict.items():
        for device in devices:
            if device.endswith('_Match'):
                # 替换匹配设备
                target = device.replace('_Match', '')
                processed = [d if d != target else device for d in processed]
            else:
                # 移除被屏蔽设备
                processed = [d for d in processed if d != device]

    return processed



# Test Code

# agent_assign_yaml_path = "/home/jianghaoning/ICCAD/AnalogDesignAuto_MultiAgent/custom_env/config/config_single_agent/agent_assign.yaml"

# with open('/home/jianghaoning/ICCAD/AnalogDesignAuto_MultiAgent/custom_env/config/config_single_agent/agent_assign.yaml') as f:
#     agent_assign_dict = yaml.safe_load(f)['single_agent']
# device_mask_yaml_path = "/home/jianghaoning/ICCAD/AnalogDesignAuto_MultiAgent/custom_env/config/config_single_agent/device_mask.yaml"
# with open(device_mask_yaml_path, 'r') as file:
#     device_mask_dict = yaml.safe_load(file)
# # action_space_dict = gen_single_continuous_action_space(device_mask_dict, agent_assign_dict)
# # # 加载配置
# # with open('agent_assign.yaml') as f:
# #     device_list = yaml.safe_load(f)['single_agent']

# # with open('device_mask.yaml') as f:
# #     device_mask = yaml.safe_load(f)

# # 生成动作空间
# action_space = gen_single_continuous_action_space(agent_assign_dict, device_mask_dict)

# print(action_space)
# 输出示例: Box(0.0, 1.0, (127,), float32) 

# print(action_space_dict)
# print(action_space_dict.sample())

# Output

# Dict('Agent_1': Dict('CC1': Box(0.0, 1.0, (1,), float64), 'IB1': Box(0.0, 1.0, (1,), float64), 'IB2': Box(0.0, 1.0,
# (1,), float64), 'IB3': Box(0.0, 1.0, (1,), float64), 'M1': Box(0.0, 1.0, (3,), float64), 'M3': Box(0.0, 1.0, (3,),
# float64), 'M5': Box(0.0, 1.0, (3,), float64)),
# 'Agent_2': Dict('IB4': Box(0.0, 1.0, (1,), float64), 'IB5': Box(0.0,
# 1.0, (1,), float64), 'M12': Box(0.0, 1.0, (3,), float64), 'M7': Box(0.0, 1.0, (3,), float64), 'M9': Box(0.0, 1.0,
# (3,), float64), 'RS1': Box(0.0, 1.0, (1,), float64), 'RS2': Box(0.0, 1.0, (1,), float64), 'RS3': Box(0.0, 1.0, (1,
# ), float64)),
# 'Agent_3': Dict('CC2': Box(0.0, 1.0, (1,), float64), 'CFF1': Box(0.0, 1.0, (1,), float64),
# 'IB6': Box(0.0, 1.0, (1,), float64), 'IB7': Box(0.0, 1.0, (1,), float64), 'M13': Box(0.0, 1.0, (3,), float64),
# 'M15': Box(0.0, 1.0, (3,), float64), 'M18_Match': Box(0.0, 1.0, (1,), float64), 'RC2': Box(0.0, 1.0, (1,),
# float64), 'RFF1': Box(0.0, 1.0, (1,), float64), 'RFF2': Box(0.0, 1.0, (1,), float64)),
# 'Agent_4': Dict('M0': Box(
# 0.0, 1.0, (3,), float64), 'RF1': Box(0.0, 1.0, (1,), float64), 'RF2': Box(0.0, 1.0, (1,), float64)))

# OrderedDict([('Agent_1', OrderedDict([('CC1', array([0.493357])), ('IB1', array([0.53245834])), ('IB2',
# array([0.87585361])), ('IB3', array([0.2147892])), ('M1', array([0.34548798, 0.19526191, 0.31041265])), ('M3',
# array([0.64578267, 0.48986625, 0.52779528])), ('M5', array([0.50632692, 0.97714935, 0.46945506]))])), ('Agent_2',
# OrderedDict([('IB4', array([0.0950411])), ('IB5', array([0.158443])), ('M12', array([0.8665787 , 0.30845822,
# 0.29615566])), ('M7', array([0.58860131, 0.55341384, 0.70000145])), ('M9', array([0.88636789, 0.46847989,
# 0.34320644])), ('RS1', array([0.96523337])), ('RS2', array([0.45065067])), ('RS3', array([0.32390361]))])),
# ('Agent_3', OrderedDict([('CC2', array([0.81840415])), ('CFF1', array([0.89620556])), ('IB6', array([0.72583433])),
# ('IB7', array([0.46248085])), ('M13', array([0.93667   , 0.80311077, 0.68963442])), ('M15', array([0.07773608,
# 0.13162735, 0.67831186])), ('M18_Match', array([0.16219355])), ('RC2', array([0.17780553])), ('RFF1',
# array([0.87879238])), ('RFF2', array([0.92046156]))])), ('Agent_4', OrderedDict([('M0', array([0.74038348,
# 0.45659135, 0.75843968])), ('RF1', array([0.52462717])), ('RF2', array([0.36224194]))]))])
