import gymnasium
import numpy as np
import yaml
import logging

from util.util_func import unit_conversion
def gen_obs_space_w_region(sim_config_dict, param_range_config_dict, agent_assign_dict):
    """
    单智能体版本观测空间生成（包含晶体管工作区）
    """
    result_config = {}

    for item in sim_config_dict:
        sim_name = item['simulation_name']
        sim_item = item['simulation_item']
        modified_sim_item = [f"{sim_name}_" + name for name in sim_item]
        result_config[sim_name] = modified_sim_item

    # 生成specs空间
    all_specs = sum(result_config.values(), [])
    specs_space = gymnasium.spaces.Dict({
        'ideal_specs': gymnasium.spaces.Box(-1, 1, (len(all_specs),), dtype=np.float32),
        'cur_specs': gymnasium.spaces.Box(-1, 1, (len(all_specs),), dtype=np.float32)
    })

    # 生成参数空间
    param_dims = []
    for comp in param_range_config_dict.values():
        param_dims.extend([p['variable_name'] for p in comp['params']])
    param_space = gymnasium.spaces.Box(
        low=-1, high=1, 
        shape=(len(param_dims),), 
        dtype=np.float32
    )

    # 晶体管工作区空间
    device_regions = [
        comp for comp in param_range_config_dict 
        if comp != 'other_variable'
    ]
    region_space = gymnasium.spaces.Box(
        0, 4, 
        shape=(len(device_regions),), 
        dtype=np.int32
    )

    return gymnasium.spaces.Dict({
        'specs': specs_space,
        'params': param_space,
        'regions': region_space
    })
def flatten_obs_space_w_region(obs_space):
    """展平包含工作区的观测空间"""
    return gymnasium.spaces.Box(
        low=np.concatenate([
            obs_space['specs']['ideal_specs'].low,
            obs_space['specs']['cur_specs'].low,
            obs_space['params'].low,
            obs_space['regions'].low
        ]),
        high=np.concatenate([
            obs_space['specs']['ideal_specs'].high,
            obs_space['specs']['cur_specs'].high,
            obs_space['params'].high,
            obs_space['regions'].high
        ]),
        dtype=np.float32
    )

def gen_obs_space(sim_config_dict, param_range_config_dict, agent_assign_dict):
    """单智能体基础观测空间生成"""
    result_config = {}
    
    for item in sim_config_dict:
        sim_name = item['simulation_name']
        result_config[sim_name] = [
            f"{sim_name}_{name}" 
            for name in item['simulation_item']
        ]

    # 合并所有specs
    all_specs = sum(result_config.values(), [])
    specs_space = gymnasium.spaces.Dict({
        'ideal_specs': gymnasium.spaces.Box(-1, 1, (len(all_specs),), dtype=np.float32),
        'cur_specs': gymnasium.spaces.Box(-1, 1, (len(all_specs),), dtype=np.float32)
    })

    # 参数空间
    param_dims = [
        p['variable_name'] 
        for comp in param_range_config_dict.values() 
        for p in comp['params']
    ]
    param_space = gymnasium.spaces.Box(
        low=-1, high=1,
        shape=(len(param_dims),),
        dtype=np.float32
    )

    return gymnasium.spaces.Dict({
        'specs': specs_space,
        'params': param_space
    })
    
def flatten_obs_space(obs_space):
    """展平基础观测空间"""
    return gymnasium.spaces.Box(
        low=np.concatenate([
            obs_space['specs']['ideal_specs'].low,
            obs_space['specs']['cur_specs'].low,
            obs_space['params'].low
        ]),
        high=np.concatenate([
            obs_space['specs']['ideal_specs'].high,
            obs_space['specs']['cur_specs'].high,
            obs_space['params'].high
        ]),
        dtype=np.float32
    )