import os
import yaml
import argparse
import logging
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Any

from spectre_environment_wrapper import SpectreEnvironmentWrapper
from nsga2_optimizer import NSGA2CircuitOptimizer
from parallel_evaluation import ParallelEvaluator
from result_tracker import OptimizationTracker


def setup_logging(log_level: str = "INFO"):
    """配置日志系统"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"无效的日志级别: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def plot_optimization_history(save_dir: str, optimization_path: list):
    """绘制优化历史曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(optimization_path, '-o')
    plt.xlabel('Generation')
    plt.ylabel('Best Reward')
    plt.title('Optimization Progress')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'optimization_history.png'))
    plt.close()


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        "config_folder_name": "config_Haoqiang_Regroup_single",
        "run_folder_name": "run_nsga2_spectre",
        "netlist_folder_name": "netlist_template_Haoqiang",
        "specs_folder_name": "sampled_specs_Haoqiang",
        "device_mask_flag": True,
        "reward_func": "cal_reward_Haoqiang",
        "log_level": "INFO",
        "n_init": 50,               # 初始种群大小
        "n_generations": 100,       # 总代数
        "population_size": 40,      # 每代种群大小
        "save_freq": 5,             # 结果保存频率
        "cpu_usage": 30,            # 并行计算的CPU核心数
        "gpu_usage": 0,             # GPU使用（未使用）
        "max_step": 1,              # 环境每次评估的最大步数
        "generalize": True,         # 是否泛化规范
        "sim_output": False,        # 是否输出仿真详情
        "init_method": "random",    # 初始化方法
        "corner_sim": False,        # 是否进行工艺角仿真
        "dc_check": False,          # 是否检查DC工作点
        "region_extract": False,    # 是否提取工作区域
        "dynamic_queue": False,     # 是否使用动态队列
        "continue_steps_enable": False  # 是否在获得正奖励后继续步数
    }


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用NSGA-II优化Spectre电路设计")
    parser.add_argument('--config', type=str, help="配置文件路径")
    parser.add_argument('--train_config', type=str, 
                        default="/home/anwenbo/custom_env/train_config/Haoqiang_eex04_ReGroup_single_ga.yaml",
                        help="train_config目录中的训练配置文件路径")
    args = parser.parse_args()

    # 加载配置
    config = get_default_config()
    
    # 如有指定，首先尝试从train_config加载
    if args.train_config:
        try:
            with open(args.train_config, 'r') as f:
                train_config = yaml.safe_load(f)
                logging.info(f"从{args.train_config}加载训练配置")
                # 用train_config更新配置
                config.update(train_config)
        except Exception as e:
            logging.warning(f"无法加载train_config文件: {e}")
    
    # 如果提供了显式配置文件，用它覆盖
    if args.config is not None:
        with open(args.config, 'r') as f:
            config_override = yaml.safe_load(f)
            logging.info(f"从{args.config}加载覆盖配置")
            config.update(config_override)

    # 设置日志
    setup_logging(config["log_level"])
    logging.info("开始使用NSGA-II为电路设计进行优化...")
    logging.info(f"配置: {config}")

    # 创建目录
    current_path = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(current_path, config["run_folder_name"], f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    try:
        # 从加载的配置创建环境配置
        env_config = {
            "generalize": config.get("generalize", True),
            "max_step": config.get("max_step", 1),
            "netlist_folder_name": config.get("netlist_folder_name", "netlist_template_Haoqiang"),
            "specs_folder_name": config.get("specs_folder_name", "sampled_specs_Haoqiang"),
            "config_folder_name": config.get("config_folder_name", "config_Haoqiang_Regroup_single"),
            "run_folder_name": run_dir,
            "sim_output": config.get("sim_output", False),
            "init_method": config.get("init_method", "random"),
            "corner_sim": config.get("corner_sim", False),
            "dc_check": config.get("dc_check", False),
            "region_extract": config.get("region_extract", False),
            "dynamic_queue": config.get("dynamic_queue", False),
            "log_level": config.get("log_level", "INFO"),
            "reward_func": config.get("reward_func", "cal_reward_Haoqiang"),
            "continue_steps_enable": config.get("continue_steps_enable", False)
        }
        
        # 初始化Spectre环境包装器
        evaluator = SpectreEnvironmentWrapper(
            config_folder=os.path.join(current_path, 'config', config["config_folder_name"]),
            run_folder=run_dir,
            reward_func=config["reward_func"],
            device_mask_flag=config["device_mask_flag"],
            env_config=env_config
        )

        # 获取参数空间信息
        input_dim = evaluator.get_param_dims()
        bounds = evaluator.get_bounds()
        param_names = evaluator._get_param_names()
        
        # 初始化结果跟踪器
        tracker = OptimizationTracker(
            save_dir=run_dir,
            param_names=param_names,
            live_update=True,
            update_interval=1
        )
        
        # 初始化并行评估器
        n_workers = int(config.get("cpu_usage", 1))
        parallel_evaluator = ParallelEvaluator(
            eval_func=evaluator.evaluate_single,
            n_workers=n_workers
        )
        
        logging.info(f"使用 {n_workers} 个CPU核心进行并行评估")

        # 初始化NSGA-II优化器
        optimizer = NSGA2CircuitOptimizer(
            eval_func=evaluator.evaluate_single,  # 单样本评估函数
            parallel_eval_func=parallel_evaluator.evaluate_batch,  # 并行批量评估函数
            bounds=bounds,
            input_dim=input_dim,
            save_dir=run_dir,
            pop_size=config["population_size"],
            result_tracker=tracker
        )
        
        logging.info(f"NSGA-II优化器初始化完成，输入维度: {input_dim}")
        logging.info(f"开始优化: n_init={config['n_init']}, n_generations={config['n_generations']}, pop_size={config['population_size']}")
        
        # 运行优化
        result = optimizer.optimize(
            n_generations=config["n_generations"],
            n_init=config["n_init"],
            save_freq=config["save_freq"]
        )

        # 保存最终结果
        np.savez(
            os.path.join(run_dir, 'final_results.npz'),
            best_params=result.best_params,
            best_value=result.best_value,
            optimization_path=result.optimization_path,
            batch_avg_values=result.batch_avg_values
        )

        # 绘制优化历史
        plot_optimization_history(run_dir, result.optimization_path)
        
        # 以可读格式保存最佳参数
        evaluator.save_best_parameters(
            result.best_params, 
            result.best_value,
            os.path.join(run_dir, 'best_parameters')
        )
        
        # 创建总结文件
        with open(os.path.join(run_dir, 'optimization_summary.txt'), 'w') as f:
            f.write(f"NSGA-II 优化总结\n")
            f.write(f"=======================\n\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总代数: {config['n_generations']}\n")
            f.write(f"初始样本数: {config['n_init']}\n")
            f.write(f"种群大小: {config['population_size']}\n\n")
            
            f.write(f"最佳奖励值: {result.best_value:.4f}\n")
            f.write(f"参数维度: {input_dim}\n\n")
            
            f.write(f"优化路径:\n")
            for i, reward in enumerate(result.optimization_path):
                f.write(f"代数 {i}: {reward:.4f}\n")

        # 打印最终结果
        logging.info("优化成功完成!")
        logging.info(f"最佳奖励值: {result.best_value:.4f}")
        logging.info(f"结果保存在: {run_dir}")
        
        # 输出最佳参数
        logging.info("找到的最佳参数:")
        for i, (name, value) in enumerate(zip(param_names, result.best_params)):
            if i < len(result.best_params):
                logging.info(f"  {name}: {value:.6f}")

    except Exception as e:
        logging.error(f"优化过程中出错: {str(e)}")
        logging.exception("详细错误信息:")
        raise


if __name__ == "__main__":
    main()