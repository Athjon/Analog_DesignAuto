from dataclasses import dataclass
import numpy as np
import logging
from typing import List, Optional, Callable, Union, Tuple, Any
import os
from datetime import datetime
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt  # 用于绘图

@dataclass
class OptimizationResult:
    """优化结果数据类"""
    best_params: np.ndarray
    best_value: float
    all_params: List[np.ndarray]
    all_values: List[float]
    optimization_path: List[float]
    batch_avg_values: List[float]  # 新增：每批次的平均reward

class BayesianOptimizer:
    """
    使用神经网络代理模型的贝叶斯优化器

    属性:
        eval_func: 电路评估函数
        parallel_eval_func: 并行批量评估函数
        bounds: 参数边界
        surrogate: 神经网络代理模型
        exploration_weight: 采集函数中的探索权重
        n_restarts: 采集函数优化的随机重启次数
        result_tracker: 结果跟踪和可视化工具
    """

    def __init__(self,
                 eval_func: Callable,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 input_dim: int,
                 save_dir: str,
                 parallel_eval_func: Optional[Callable] = None,
                 exploration_weight: float = 0.1,
                 n_restarts: int = 10,
                 result_tracker: Any = None):
        """
        初始化贝叶斯优化器
        
        Args:
            eval_func: 电路评估函数（单个样本）
            bounds: 参数下限和上限的元组
            input_dim: 输入参数维度
            save_dir: 保存模型和结果的目录
            parallel_eval_func: 并行批量评估函数
            exploration_weight: 探索项的权重
            n_restarts: 采集函数优化的随机重启次数
            result_tracker: 结果跟踪和可视化工具
        """
        self.eval_func = eval_func
        self.parallel_eval_func = parallel_eval_func
        self.bounds = bounds
        self.input_dim = input_dim
        self.save_dir = save_dir
        self.exploration_weight = exploration_weight
        self.n_restarts = n_restarts
        self.result_tracker = result_tracker

        # 创建带时间戳的保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f"bo_run_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)

        # 初始化代理模型
        from surrogate_model import SurrogateModel
        self.surrogate = SurrogateModel(input_dim=input_dim)

        # 初始化优化历史
        self.X_history = []
        self.y_history = []
        self.best_value = -np.inf
        self.best_params = None
        
        # 新增：初始化批次平均reward列表
        self.batch_avg_values = []
        
        # 记录开始时间
        self.start_time = time.time()
        logging.info(f"贝叶斯优化器初始化完成，输入维度: {input_dim}")

    def optimize(self,
                 n_init: int = 10,
                 n_iter: int = 100,
                 batch_size: int = 32,
                 surrogate_epochs: int = 100,
                 save_freq: int = 10) -> OptimizationResult:
        """
        运行贝叶斯优化
        
        Args:
            n_init: 初始随机点数量
            n_iter: 优化迭代次数
            batch_size: 代理模型训练的批量大小
            surrogate_epochs: 代理模型训练的轮数
            save_freq: 保存结果的频率
            
        Returns:
            result: 优化结果
        """
        # 初始随机采样
        logging.info("开始初始随机采样...")
        
        if self.parallel_eval_func:
            # 使用并行评估
            logging.info(f"使用并行评估进行初始采样 (n={n_init})...")
            
            # 生成均匀分布的随机样本
            X_init = np.random.uniform(
                self.bounds[0],
                self.bounds[1],
                size=(n_init, self.input_dim)
            )
            
            # 并行评估样本
            y_init = self.parallel_eval_func(X_init)
            
            # 新增：计算并存储初始批次的平均reward
            batch_avg = np.mean(y_init)
            self.batch_avg_values.append(batch_avg)
            logging.info(f"初始批次的平均reward: {batch_avg:.4f}")
        else:
            # 顺序评估
            logging.info(f"使用顺序评估进行初始采样 (n={n_init})...")
            X_init = np.random.uniform(
                self.bounds[0],
                self.bounds[1],
                size=(n_init, self.input_dim)
            )
            y_init = np.array([self.eval_func(x) for x in X_init])
            
            # 新增：计算并存储初始批次的平均reward
            batch_avg = np.mean(y_init)
            self.batch_avg_values.append(batch_avg)
            logging.info(f"初始批次的平均reward: {batch_avg:.4f}")

        # 更新历史
        self.X_history.extend(X_init)
        self.y_history.extend(y_init)

        # 更新最佳观测
        best_idx = np.argmax(y_init)
        self.best_value = y_init[best_idx]
        self.best_params = X_init[best_idx]
        
        # 新增：绘制初始批次的平均reward图表
        self._plot_batch_average_rewards(0)
        
        # 如果有结果跟踪器，初始化它
        if self.result_tracker:
            for i, (params, value) in enumerate(zip(X_init, y_init)):
                # 将初始样本视为第0次迭代
                self.result_tracker.update(0, params, value, X_init, y_init)

        # 主优化循环
        logging.info("开始主优化循环...")
        optimization_path = [self.best_value]
        
        for i in range(n_iter):
            # 迭代开始时间
            iter_start_time = time.time()
            logging.info(f"开始第 {i+1}/{n_iter} 次迭代...")
        
            # 训练代理模型
            X_train = np.array(self.X_history)
            y_train = np.array(self.y_history)

            # 创建每次迭代的检查点目录
            model_save_dir = os.path.join(self.save_dir, f"surrogate_iter_{i}")
            
            logging.info(f"训练代理模型，迭代 {i + 1}/{n_iter}... \n当前训练数据形状: {X_train.shape} ")
            self.surrogate.fit(
                X_train,
                y_train,
                save_dir=model_save_dir,
                batch_size=batch_size,
                epochs=surrogate_epochs
            )

            # 生成多个候选点
            next_points = self._propose_multiple_points(self.n_restarts)
            
            # 并行评估候选点
            if self.parallel_eval_func:
                next_y_values = self.parallel_eval_func(next_points)
            else:
                # 顺序评估
                next_y_values = np.array([self.eval_func(x) for x in next_points])
            
            # 新增：计算并存储当前批次的平均reward
            batch_avg = np.mean(next_y_values)
            self.batch_avg_values.append(batch_avg)
            logging.info(f"批次 {i+1} 的平均reward: {batch_avg:.4f}")
                
            # 找到最佳点
            best_idx = np.argmax(next_y_values)
            next_x = next_points[best_idx]
            next_y = next_y_values[best_idx]

            # 更新历史
            self.X_history.append(next_x)
            self.y_history.append(next_y)

            # 更新最佳观测
            if next_y > self.best_value:
                self.best_value = next_y
                self.best_params = next_x
                logging.info(f"发现新的最佳值: {self.best_value:.4f}")

            # 更新优化路径
            optimization_path.append(self.best_value)
            
            # 更新结果跟踪器
            if self.result_tracker:
                self.result_tracker.update(i+1, next_x, next_y, next_points, next_y_values)
            
            # 在每次仿真（reward计算后）绘制并保存优化历史图
            plt.figure(figsize=(10,6))
            plt.plot(optimization_path, '-o')
            plt.xlabel('Round')
            plt.ylabel('Best Reward')
            plt.title('Optimization Progress')
            plt.grid(True)
            plot_path = os.path.join(self.save_dir, f'optimization_history_iter_{i+1}.png')
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"已保存优化历史图: {plot_path}")
            
            # 新增：绘制批次平均reward图表
            self._plot_batch_average_rewards(i+1)
            
            # 定期保存结果
            if (i + 1) % save_freq == 0:
                self._save_results(i + 1)
                
            # 记录迭代时间
            iter_time = time.time() - iter_start_time
            total_time = time.time() - self.start_time
            logging.info(f"迭代 {i + 1}/{n_iter}: 最佳值 = {self.best_value:.4f}, 平均值 = {batch_avg:.4f}, "
                         f"迭代耗时 = {iter_time:.2f}秒, 总耗时 = {total_time:.2f}秒")

        # 最终保存
        self._save_results('final')
        
        # 生成最终报告
        if self.result_tracker:
            self.result_tracker.final_report()

        return OptimizationResult(
            best_params=self.best_params,
            best_value=self.best_value,
            all_params=self.X_history,
            all_values=self.y_history,
            optimization_path=optimization_path,
            batch_avg_values=self.batch_avg_values  # 新增：返回批次平均值列表
        )
    
    # 新增：绘制批次平均reward的函数
    def _plot_batch_average_rewards(self, iteration: int):
        """
        绘制每个批次的平均reward
        
        Args:
            iteration: 当前迭代次数
        """
        plt.figure(figsize=(10,6))
        
        # 绘制批次平均reward
        plt.plot(range(len(self.batch_avg_values)), self.batch_avg_values, '-o', color='green', label='Batch Average')
        
        # 如果有3个或以上的点，计算移动平均线
        if len(self.batch_avg_values) >= 3:
            window_size = min(3, len(self.batch_avg_values))
            moving_avg = np.convolve(self.batch_avg_values, np.ones(window_size)/window_size, mode='valid')
            # 绘制移动平均线
            x_vals = range(window_size-1, len(self.batch_avg_values))
            plt.plot(x_vals, moving_avg, '--', color='red', label=f'{window_size}-point Moving Avg')
        
        plt.xlabel('Batch Number')
        plt.ylabel('Average Reward')
        plt.title('Batch Average Rewards')
        plt.grid(True)
        plt.legend()
        
        # 添加数值标签
        for i, avg in enumerate(self.batch_avg_values):
            plt.annotate(f'{avg:.3f}', (i, avg), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        # 保存图片
        plot_path = os.path.join(self.save_dir, f'batch_average_rewards_iter_{iteration}.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"已保存批次平均reward图: {plot_path}")

    def _acquisition_function(self, x: np.ndarray) -> float:
        """
        期望改进采集函数
        
        Args:
            x: 要评估的参数
            
        Returns:
            acquisition_value: 采集函数值
        """
        x = x.reshape(1, -1)
        mean, std = self.surrogate.predict(x)

        # 计算改进量
        improvement = mean - self.best_value
        # 添加探索奖励
        exploration_bonus = self.exploration_weight * std

        return -(improvement + exploration_bonus)  # 最小化负期望改进

    def _propose_next_point(self) -> np.ndarray:
        """
        通过优化采集函数提出下一个评估点
        
        Returns:
            next_point: 下一次评估的参数
        """
        best_x = None
        best_acquisition_value = np.inf

        # 多个随机起点
        for _ in range(self.n_restarts):
            x0 = np.random.uniform(self.bounds[0], self.bounds[1])

            # 优化采集函数
            result = minimize(
                self._acquisition_function,
                x0,
                bounds=list(zip(self.bounds[0], self.bounds[1])),
                method='L-BFGS-B'
            )

            if result.fun < best_acquisition_value:
                best_acquisition_value = result.fun
                best_x = result.x

        return best_x
        
    def _propose_multiple_points(self, n_points: int) -> np.ndarray:
        """
        提出多个候选评估点
        
        Args:
            n_points: 要生成的点数量
            
        Returns:
            points: 候选点数组
        """
        points = []
        acquisition_values = []

        # 从多个起点优化采集函数
        for _ in range(n_points):
            x0 = np.random.uniform(self.bounds[0], self.bounds[1])

            # 优化采集函数
            result = minimize(
                self._acquisition_function,
                x0,
                bounds=list(zip(self.bounds[0], self.bounds[1])),
                method='L-BFGS-B'
            )

            points.append(result.x)
            acquisition_values.append(result.fun)

        return np.array(points)

    def _save_results(self, iteration: Union[int, str]):
        """保存优化结果"""
        results = {
            'X_history': self.X_history,
            'y_history': self.y_history,
            'best_params': self.best_params,
            'best_value': self.best_value,
            'batch_avg_values': self.batch_avg_values  # 新增：保存批次平均值
        }
        save_path = os.path.join(self.save_dir, f'results_iter_{iteration}.npz')
        np.savez(save_path, **results)
        logging.info(f"已保存结果: {save_path}")