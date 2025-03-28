import gymnasium
import numpy as np
import yaml
import logging

from util.util_func import unit_conversion
# from util_func import unit_conversion

# Test Code
# result_config_file = "../config_3/result.yaml"
# param_range_config_file = "../config_3/param_range.yaml"
# agent_assign_yaml_file = "../config_3/agent_assign.yaml"
# observation_space = gen_obs_space_extend(result_config_file, param_range_config_file, agent_assign_yaml_file)
# print(f"observation_space: {observation_space}")
# observation_space_flat = flatten_obs_space(observation_space)
# print(f"observation_space_flat: {observation_space_flat}")


def gen_obs_space_w_type(sim_config_file, param_range_config_file, agent_assign_yaml_path):
    """
    Generate observation space for the custom environment.
    :param agent_assign_yaml_path: path of the agent assign yaml file
    :param sim_config_file: path of the result config file
    :param param_range_config_file: path of the parameter range config file
    :return: obs_space: gymnasium.spaces.Dict, observation space for the custom environment
    """

    with open(agent_assign_yaml_path, 'r') as file:
        agent_assign = yaml.safe_load(file)

    # Import YAML file
    with open(sim_config_file, 'r') as file:
        sim_config = yaml.safe_load(file)

    result_config = {}
    for item in sim_config:
        sim_name = item['simulation_name']
        sim_item = item['simulation_item']
        result_config[sim_name] = sim_item

    with open(param_range_config_file, 'r') as file:
        param_range_config = yaml.safe_load(file)

    # Create spaces for ideal_specs and cur_specs
    ideal_specs_spaces = {key: gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                          for key in sum(result_config.values(), [])}
    cur_specs_spaces = {key: gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                        for key in sum(result_config.values(), [])}

    # Create spaces for cur_param
    cur_param_spaces = {}
    for component, data in param_range_config.items():
        for param in data['params']:
            variable_name = param['variable_name']
            range_min, range_max = param['value']['range']
            cur_param_spaces[variable_name] = gymnasium.spaces.Box(low=unit_conversion(range_min),
                                                                   high=unit_conversion(range_max), shape=(1,),
                                                                   dtype=np.float32)

    # 1 for pch, 2 for nch, 3 for vsource, 4 for isource
    device_type_space = {}
    for component, data in param_range_config.items():
        if component == 'other_variable':
            for param in data['params']:
                variable_name = param['variable_name']
                device_type_space[variable_name] = gymnasium.spaces.Discrete(4)
        else:
            variable_name = component
            device_type_space[variable_name] = gymnasium.spaces.Discrete(4)

    # Combine three dicts into one gymnasium.spaces.Dict
    obs_space_single = gymnasium.spaces.Dict({
        'ideal_specs': gymnasium.spaces.Dict(ideal_specs_spaces),
        'cur_specs': gymnasium.spaces.Dict(cur_specs_spaces),
        'cur_param': gymnasium.spaces.Dict(cur_param_spaces),
        'device_type': gymnasium.spaces.Dict(device_type_space)
    })

    obs_space = gymnasium.spaces.Dict()
    for group_name in agent_assign.keys():
        obs_space[group_name] = obs_space_single

    return obs_space


def flatten_obs_space_w_type(obs_space: gymnasium.spaces.Dict):
    """
    :param obs_space: Complex obs dict
    :return: obs_space_flat: Flatten obs dict, only remain first level keys. For sub-dict, sort the element and convert
    to tuple. Rearrange the dict as 'cur_specs', 'ideal_specs' and 'cur_param'.
    """
    flattened_space = {}

    for agent, agent_space in obs_space.items():
        # Sort and prepare the data for cur_specs, ideal_specs, and cur_param
        sorted_cur_specs = {key: agent_space['cur_specs'][key] for key in sorted(agent_space['cur_specs'])}
        sorted_ideal_specs = {key: agent_space['ideal_specs'][key] for key in sorted(agent_space['ideal_specs'])}
        sorted_cur_param = {key: agent_space['cur_param'][key] for key in sorted(agent_space['cur_param'])}
        sorted_device_type = {key: agent_space['device_type'][key] for key in sorted(agent_space['device_type'])}

        # Print the sorted dicts for debugging
        # print(f"Agent: {agent}")
        # print("Sorted cur_specs:", sorted_cur_specs)
        # print("Sorted ideal_specs:", sorted_ideal_specs)
        # print("Sorted cur_param:", sorted_cur_param)

        # Combine all boxes from sorted dicts
        combined_boxes = []
        combined_boxes.extend(sorted_cur_specs.values())
        combined_boxes.extend(sorted_ideal_specs.values())
        combined_boxes.extend(sorted_cur_param.values())
        combined_boxes.extend(sorted_device_type.values())

        # Convert the combined list to a Tuple space and assign to the agent
        flattened_space[agent] = gymnasium.spaces.Tuple(combined_boxes)

    return gymnasium.spaces.Dict(flattened_space)


# Test Code
# result_config_file = "../config_3/simulation.yaml"
# param_range_config_file = "../config_3/param_range.yaml"
# agent_assign_yaml_file = "../config_3/agent_assign.yaml"
# observation_space = gen_obs_space_w_type(result_config_file, param_range_config_file, agent_assign_yaml_file)
# print(f"observation_space: {observation_space}")
# observation_space_flat = flatten_obs_space_w_type(observation_space)
# print(f"observation_space_flat: {observation_space_flat}")


def gen_obs_space_w_region(sim_config_dict, param_range_config_dict, agent_assign_dict):
    """
    Generate observation space for the custom environment.
    :param agent_assign_dict: dict of the agent assign yaml file
    :param sim_config_dict: dict of the result config file
    :param param_range_config_dict: dict of the parameter range config file
    :return: obs_space: gymnasium.spaces.Dict, observation space for the custom environment
    """

    result_config = {}

    for item in sim_config_dict:
        sim_name = item['simulation_name']
        sim_item = item['simulation_item']
        # Rename sim_item with sim_name
        modified_sim_item = [f"{sim_name}_" + sim_item_name for sim_item_name in sim_item]
        result_config[sim_name] = modified_sim_item
    print(f"sim_config_dict is {sim_config_dict}")
    # Create spaces for ideal_specs and cur_specs
    ideal_specs_spaces = {key: gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                          for key in sum(result_config.values(), [])}
    cur_specs_spaces = {key: gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                        for key in sum(result_config.values(), [])}

    # Create spaces for cur_param
    cur_param_spaces = {}
    for component, data in param_range_config_dict.items():
        for param in data['params']:
            variable_name = param['variable_name']
            range_min, range_max = param['value']['range']
            cur_param_spaces[variable_name] = gymnasium.spaces.Box(low=unit_conversion(range_min),
                                                                   high=unit_conversion(range_max), shape=(1,),
                                                                   dtype=np.float32)

    # Create spaces for transistor region
    transistor_region_space = {}
    for component, data in param_range_config_dict.items():
        if component == 'other_variable':
            pass
        else:
            variable_name = component
            # 0 cut-off, 1 triode, 2 saturation, 3 sub-th, 4 breakdown
            transistor_region_space[variable_name] = gymnasium.spaces.Discrete(5)

    # Combine three dicts into one gymnasium.spaces.Dict
    # obs_space_single = gymnasium.spaces.Dict({
    #     'ideal_specs': gymnasium.spaces.Dict(ideal_specs_spaces),
    #     'cur_specs': gymnasium.spaces.Dict(cur_specs_spaces),
    #     'cur_param': gymnasium.spaces.Dict(cur_param_spaces),
    #     'transistor_region': gymnasium.spaces.Dict(transistor_region_space)
    # })
    all_spaces = {}
    all_spaces.update(ideal_specs_spaces)
    all_spaces.update(cur_specs_spaces)
    all_spaces.update(cur_param_spaces)
    all_spaces.update(transistor_region_space)

    # obs_space = gymnasium.spaces.Dict()
   
    # for group_name in agent_assign_dict.keys():
    #     obs_space[group_name] = obs_space_single

    # obs_space_single = obs_space["single_agent"]
    obs_space_single = gymnasium.spaces.Dict(all_spaces)
    # print(f"obs_space_single is {obs_space_single}\n\n")
    print(isinstance(obs_space_single, gymnasium.spaces.Dict))
    
    # print(f"Debug in gen_obs_space,\n obs_space: {obs_space_single}")

    return obs_space_single


def flatten_obs_space_w_region(obs_space: gymnasium.spaces.Dict):
    """
    :param obs_space: Complex obs dict
    :return: obs_space_flat: Flatten obs dict, only remain first level keys. For sub-dict, sort the element and convert
    to tuple. Rearrange the dict as 'cur_specs', 'ideal_specs' and 'cur_param'.
    """
    # if len(obs_space.spaces) == 1 and 'single_agent' in obs_space.spaces:
    #     agent = 'single_agent'
    #     agent_space = obs_space[agent]
        
    #     # 提取并合并所有子空间
    #     combined_boxes = []
    #     for subspace in agent_space.values():
    #         if isinstance(subspace, gymnasium.spaces.Dict):
    #             # 对Dict类型继续展开
    #             combined_boxes.extend(sorted(subspace.spaces.values(), key=lambda x: x.name))
    #         else:
    #             combined_boxes.append(subspace)
                
    #     return gymnasium.spaces.Tuple(combined_boxes)
    
    flattened_space = {}

    for agent, agent_space in obs_space.items():
        # Sort and prepare the data for cur_specs, ideal_specs, and cur_param
        sorted_cur_specs = {key: agent_space['cur_specs'][key] for key in sorted(agent_space['cur_specs'])}
        sorted_ideal_specs = {key: agent_space['ideal_specs'][key] for key in sorted(agent_space['ideal_specs'])}
        sorted_cur_param = {key: agent_space['cur_param'][key] for key in sorted(agent_space['cur_param'])}
        sorted_transistor_region = {key: agent_space['transistor_region'][key] for key in
                                    sorted(agent_space['transistor_region'])}

        logging.debug(f"The size of sorted_cur_specs is {len(sorted_cur_specs)}")
        logging.debug(f"The size of sorted_ideal_specs is {len(sorted_ideal_specs)}")
        logging.debug(f"The size of sorted_cur_param is {len(sorted_cur_param)}")
        logging.debug(f"The size of sorted_transistor_region is {len(sorted_transistor_region)}")

        # Combine all boxes from sorted dicts
        combined_boxes = []
        combined_boxes.extend(sorted_cur_specs.values())
        combined_boxes.extend(sorted_ideal_specs.values())
        combined_boxes.extend(sorted_cur_param.values())
        combined_boxes.extend(sorted_transistor_region.values())

        logging.debug(f"The size of combined_boxes is {len(combined_boxes)}")

        # Convert the combined list to a Tuple space and assign to the agent
        flattened_space[agent] = gymnasium.spaces.Tuple(combined_boxes)
        
        flattened_space_single = flattened_space["single_agent"]
        
        # print the type of flattened_space_single
        

    return flattened_space_single


def gen_obs_space(sim_config_dict, param_range_config_dict, agent_assign_dict):
    """
    Generate observation space for the custom environment.
    :param agent_assign_dict: dict of the agent assign yaml file
    :param sim_config_dict: dict of the result config file
    :param param_range_config_dict: dict of the parameter range config file
    :return: obs_space: gymnasium.spaces.Dict, observation space for the custom environment
    """

    result_config = {}

    for item in sim_config_dict:
        sim_name = item['simulation_name']
        sim_item = item['simulation_item']
        # Rename sim_item with sim_name
        modified_sim_item = [f"{sim_name}_" + sim_item_name for sim_item_name in sim_item]
        result_config[sim_name] = modified_sim_item

    # Create spaces for ideal_specs and cur_specs
    ideal_specs_spaces = {key: gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                          for key in sum(result_config.values(), [])}
    cur_specs_spaces = {key: gymnasium.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                        for key in sum(result_config.values(), [])}

    # Create spaces for cur_param
    cur_param_spaces = {}
    for component, data in param_range_config_dict.items():
        for param in data['params']:
            variable_name = param['variable_name']
            range_min, range_max = param['value']['range']
            cur_param_spaces[variable_name] = gymnasium.spaces.Box(low=unit_conversion(range_min),
                                                                   high=unit_conversion(range_max), shape=(1,),
                                                                   dtype=np.float32)

    # Combine three dicts into one gymnasium.spaces.Dict
    # obs_space_single = gymnasium.spaces.Dict({
    #     'ideal_specs': gymnasium.spaces.Dict(ideal_specs_spaces),
    #     'cur_specs': gymnasium.spaces.Dict(cur_specs_spaces),
    #     'cur_param': gymnasium.spaces.Dict(cur_param_spaces)
    # })
    all_spaces = {}
    all_spaces.update(ideal_specs_spaces)
    all_spaces.update(cur_specs_spaces)
    all_spaces.update(cur_param_spaces)
    obs_space_single = gymnasium.spaces.Dict(all_spaces)    
    # obs_space = gymnasium.spaces.Dict()
   
    # for group_name in agent_assign_dict.keys():
    #     obs_space[group_name] = obs_space_single
    
    # obs_space_single = obs_space["single_agent"]
    # print(isinstance(obs_space_single, gymnasium.spaces.Dict)) 
    return obs_space_single


def flatten_obs_space(obs_space: gymnasium.spaces.Dict):
    """
    :param obs_space: Complex obs dict
    :return: obs_space_flat: Flatten obs dict, only remain first level keys. For sub-dict, sort the element and convert
    to tuple. Rearrange the dict as 'cur_specs', 'ideal_specs' and 'cur_param'.
    """
    flattened_space = {}

    for agent, agent_space in obs_space.items():
        # Sort and prepare the data for cur_specs, ideal_specs, and cur_param
        sorted_cur_specs = {key: agent_space['cur_specs'][key] for key in sorted(agent_space['cur_specs'])}
        sorted_ideal_specs = {key: agent_space['ideal_specs'][key] for key in sorted(agent_space['ideal_specs'])}
        sorted_cur_param = {key: agent_space['cur_param'][key] for key in sorted(agent_space['cur_param'])}

        logging.debug(f"The size of sorted_cur_specs is {len(sorted_cur_specs)}")
        logging.debug(f"The size of sorted_ideal_specs is {len(sorted_ideal_specs)}")
        logging.debug(f"The size of sorted_cur_param is {len(sorted_cur_param)}")

        # Print the sorted dicts for debugging
        # print(f"Agent: {agent}")
        # print("Sorted cur_specs:", sorted_cur_specs)
        # print("Sorted ideal_specs:", sorted_ideal_specs)
        # print("Sorted cur_param:", sorted_cur_param)

        # Combine all boxes from sorted dicts
        combined_boxes = []
        combined_boxes.extend(sorted_cur_specs.values())
        combined_boxes.extend(sorted_ideal_specs.values())
        combined_boxes.extend(sorted_cur_param.values())

        logging.debug(f"The size of combined_boxes is {len(combined_boxes)}")

        # Convert the combined list to a Tuple space and assign to the agent
        flattened_space[agent] = gymnasium.spaces.Tuple(combined_boxes)
        flattened_space_single = flattened_space["single_agent"]
        

    return flattened_space_single

# if __name__ == "__main__":
    
#     param_range_config_file = "/home/jianghaoning/ICCAD/AnalogDesignAuto_MultiAgent/custom_env/config/config_Haoqiang_Regroup_single/param_range.yaml"
#     agent_assign_yaml_file = "/home/jianghaoning/ICCAD/AnalogDesignAuto_MultiAgent/custom_env/config/config_Haoqiang_Regroup_single/agent_assign.yaml"

#     obs_generator = gen_obs_space_w_region if self.region_extract else gen_obs_space
#         flatten_obs = flatten_obs_space_w_region if self.region_extract else flatten_obs_space
#         raw_obs_space = obs_generator(
#             sim_config_dict,
#             param_range_dict,
#             agent_assign_dict
#         )

#         # 扁平化处理
#         flatten_obs = flatten_obs_space_w_region if self.region_extract else flatten_obs_space
#         if len(agent_assign_dict) == 1:  # 单智能体
#             observation_space = flatten_obs(raw_obs_space)['single_agent']
#         else:  # 多智能体
#             observation_space = flatten_obs(raw_obs_space)