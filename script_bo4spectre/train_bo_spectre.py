import os
import yaml
import argparse
import logging
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Any

from spectre_environment_wrapper import SpectreEnvironmentWrapper
from bayesian_optimizer import BayesianOptimizer
from parallel_evaluation import ParallelEvaluator
from result_tracker import OptimizationTracker


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def plot_optimization_history(save_dir: str, optimization_path: list):
    """Plot optimization history"""
    plt.figure(figsize=(10, 6))
    plt.plot(optimization_path, '-o')
    plt.xlabel('Iteration')
    plt.ylabel('Best Reward')
    plt.title('Optimization Progress')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'optimization_history.png'))
    plt.close()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        "config_folder_name": "config_Haoqiang_Regroup_single",
        "run_folder_name": "run_bo_spectre",
        "netlist_folder_name": "netlist_template_Haoqiang",
        "specs_folder_name": "sampled_specs_Haoqiang",
        "device_mask_flag": True,
        "reward_func": "cal_reward_Haoqiang",
        "log_level": "INFO",
        "n_init": 50,
        "n_iter": 200,
        "batch_size": 200,
        "surrogate_epochs": 100,
        "exploration_weight": 0.3,
        "n_restarts": 10,
        "save_freq": 5,
        "cpu_usage": 30,
        "gpu_usage": 0,
        "max_step": 1,
        "generalize": True,
        "sim_output": False,
        "init_method": "random",
        "corner_sim": False,
        "dc_check": False,
        "region_extract": False,
        "dynamic_queue": False,
        "continue_steps_enable": False
    }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Bayesian Optimization for Circuit Design with Spectre")
    parser.add_argument('--config', type=str, help="Path to configuration file")
    parser.add_argument('--train_config', type=str, 
                        default="/home/jianghaoning/ICCAD/AnalogDesignAuto_MultiAgent/custom_env/train_config/Haoqiang_eex04_ReGroup_single.yaml",
                        help="Path to training configuration file in train_config directory")
    args = parser.parse_args()

    # Load configuration
    config = get_default_config()
    
    # Try to load from train_config directory first if specified
    if args.train_config:
        try:
            with open(args.train_config, 'r') as f:
                train_config = yaml.safe_load(f)
                logging.info(f"Loaded training config from {args.train_config}")
                # Update config with values from train_config
                config.update(train_config)
        except Exception as e:
            logging.warning(f"Could not load train_config file: {e}")
    
    # Override with explicit config file if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            config_override = yaml.safe_load(f)
            logging.info(f"Loaded override config from {args.config}")
            config.update(config_override)

    # Setup logging
    setup_logging(config["log_level"])
    logging.info("Starting Bayesian Optimization for circuit design with Spectre...")
    logging.info(f"Configuration: {config}")

    # Create directories
    current_path = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(current_path, config["run_folder_name"], f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    try:
        # Create environment config from loaded configuration
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
        
        # Initialize Spectre environment wrapper
        evaluator = SpectreEnvironmentWrapper(
            config_folder=os.path.join(current_path, 'config', config["config_folder_name"]),
            run_folder=run_dir,
            reward_func=config["reward_func"],
            device_mask_flag=config["device_mask_flag"],
            env_config=env_config
        )

        # Get parameter space information
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

        # Initialize Bayesian optimizer with parallel evaluator
        optimizer = BayesianOptimizer(
            eval_func=evaluator.evaluate_single,  # 单样本评估函数
            parallel_eval_func=parallel_evaluator.evaluate_batch,  # 并行批量评估函数
            bounds=bounds,
            input_dim=input_dim,
            save_dir=run_dir,
            exploration_weight=config["exploration_weight"],
            n_restarts=config["n_restarts"],
            result_tracker=tracker  # 添加结果跟踪器
        )
        
        logging.info(f"Bayesian optimizer initialized with {input_dim} dimensions")
        logging.info(f"Starting optimization with n_init={config['n_init']}, n_iter={config['n_iter']}")
        
        # Run optimization
        result = optimizer.optimize(
            n_init=config["n_init"],
            n_iter=config["n_iter"],
            batch_size=config["batch_size"],
            surrogate_epochs=config["surrogate_epochs"],
            save_freq=config["save_freq"]
        )

        # Save final results
        np.savez(
            os.path.join(run_dir, 'final_results.npz'),
            best_params=result.best_params,
            best_value=result.best_value,
            optimization_path=result.optimization_path
        )

        # Plot optimization history
        plot_optimization_history(run_dir, result.optimization_path)
        
        # Save best parameters in readable format
        evaluator.save_best_parameters(
            result.best_params, 
            result.best_value,
            os.path.join(run_dir, 'best_parameters')
        )
        
        # Create summary file
        with open(os.path.join(run_dir, 'optimization_summary.txt'), 'w') as f:
            f.write(f"Bayesian Optimization Summary\n")
            f.write(f"==============================\n\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total iterations: {config['n_iter']}\n")
            f.write(f"Initial random samples: {config['n_init']}\n\n")
            
            f.write(f"Best reward achieved: {result.best_value:.4f}\n")
            f.write(f"Parameter dimensions: {input_dim}\n\n")
            
            f.write(f"Optimization path:\n")
            for i, reward in enumerate(result.optimization_path):
                f.write(f"Iter {i}: {reward:.4f}\n")

        # Print final results
        logging.info("Optimization completed successfully!")
        logging.info(f"Best reward achieved: {result.best_value:.4f}")
        logging.info(f"Results saved in: {run_dir}")
        
        # Output best parameters
        logging.info("Best parameters found:")
        param_names = evaluator._get_param_names()
        for i, (name, value) in enumerate(zip(param_names, result.best_params)):
            if i < len(result.best_params):
                logging.info(f"  {name}: {value:.6f}")

    except Exception as e:
        logging.error(f"Error during optimization: {str(e)}")
        logging.exception("Detailed traceback:")
        raise


if __name__ == "__main__":
    main()

