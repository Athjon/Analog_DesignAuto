import os
import numpy as np
import logging
from typing import List, Optional, Callable, Dict, Any, Tuple
import time
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass

# PyMOO imports - 使用更兼容的导入方式
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator

# 避免直接导入 MultiObjectiveDisplay 类
# 改用 pymoo 推荐的 Display 机制
try:
    from pymoo.util.display import Display
except ImportError:
    # 如果较新版本的pymoo也没有这个导入，创建一个简单的Display类替代
    class Display:
        def _do(self, problem, evaluator, algorithm):
            # 只打印基本信息
            return algorithm.n_gen, \
                   evaluator.n_eval, \
                   algorithm.pop.get("F").min(), \
                   algorithm.pop.get("F").mean()


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    best_params: np.ndarray
    best_value: float
    all_params: List[np.ndarray]
    all_values: List[float]
    optimization_path: List[float]
    batch_avg_values: List[float]  # 每代的平均reward


class CircuitDesignProblem(Problem):
    """电路设计优化问题定义"""
    
    def __init__(self, eval_func, n_var, xl, xu):
        """
        初始化电路设计问题
        
        Args:
            eval_func: 电路评估函数
            n_var: 设计变量数量
            xl: 参数下限
            xu: 参数上限
        """
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=xl, xu=xu)
        self.eval_func = eval_func
    
    def _evaluate(self, x, out, *args, **kwargs):
        """评估电路参数"""
        n = x.shape[0]
        f = np.zeros((n, 1))
        
        for i in range(n):
            # 电路评估返回需要最大化的值，而 pymoo 是最小化问题，所以加负号
            f[i, 0] = -self.eval_func(x[i])
            
        out["F"] = f


# 自定义简易显示类，兼容各版本的 pymoo
class SimpleDisplay(Display):
    def _do(self, problem, evaluator, algorithm):
        """打印每代信息"""
        gen = algorithm.n_gen
        n_eval = evaluator.n_eval if hasattr(evaluator, 'n_eval') else 0
        
        # 获取当前种群的目标值
        F = algorithm.pop.get("F")
        f_min = F.min()
        f_mean = F.mean()
        
        # 优化的目标是最小化负的奖励，转换回原始奖励值
        best_reward = -f_min
        avg_reward = -f_mean
        
        # 格式化信息字符串
        msg = f"Generation: {gen:>4d} | Evaluations: {n_eval:>6d} | Best Reward: {best_reward:>10.4f} | Avg Reward: {avg_reward:>10.4f}"
        logging.info(msg)
        
        return gen, n_eval, f_min, f_mean


class NSGA2CircuitOptimizer:
    """
    使用 NSGA-II 算法的电路优化器
    
    重要属性:
        eval_func: 电路评估函数
        parallel_eval_func: 并行批量评估函数
        bounds: 参数边界
        input_dim: 输入维度
        pop_size: 种群大小
        save_dir: 保存结果的目录
        X_history: 历史参数记录
        y_history: 历史评估结果
        best_value: 最佳评估值
        best_params: 最佳参数组合
    """
    
    def __init__(
        self,
        eval_func: Callable,
        bounds: Tuple[np.ndarray, np.ndarray],
        input_dim: int,
        save_dir: str,
        parallel_eval_func: Optional[Callable] = None,
        pop_size: int = 40,
        result_tracker: Any = None
    ):
        """
        初始化 NSGA-II 电路优化器
        
        Args:
            eval_func: 电路评估函数
            bounds: 参数下限和上限元组
            input_dim: 输入参数维度
            save_dir: 保存模型和结果的目录
            parallel_eval_func: 并行批量评估函数
            pop_size: 种群大小
            result_tracker: 结果跟踪器
        """
        self.eval_func = eval_func
        self.parallel_eval_func = parallel_eval_func
        self.bounds = bounds
        self.input_dim = input_dim
        self.pop_size = pop_size
        self.result_tracker = result_tracker
        
        # 创建带时间戳的保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f"nsga2_run_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化优化历史
        self.X_history = []
        self.y_history = []
        self.best_value = -np.inf
        self.best_params = None
        
        # 批次平均reward列表
        self.batch_avg_values = []
        
        # 记录开始时间
        self.start_time = time.time()
        logging.info(f"NSGA-II 优化器初始化完成，输入维度: {input_dim}, 种群大小: {pop_size}")
    
    def evaluate_circuit_batch(self, X):
        """
        评估一批电路设计并更新历史记录
        
        Args:
            X: 参数矩阵 (种群大小 x 参数维度)
            
        Returns:
            y: 评估结果数组
        """
        if self.parallel_eval_func:
            # 使用并行评估
            logging.info(f"使用并行评估评估 {X.shape[0]} 个设计...")
            y = self.parallel_eval_func(X)
        else:
            # 顺序评估
            logging.info(f"顺序评估 {X.shape[0]} 个设计...")
            y = np.array([self.eval_func(x) for x in X])
        
        # 更新历史记录
        self.X_history.extend(X)
        self.y_history.extend(y)
        
        # 更新最佳观测
        if len(y) > 0:
            best_idx = np.argmax(y)
            if y[best_idx] > self.best_value:
                self.best_value = y[best_idx]
                self.best_params = X[best_idx].copy()
                logging.info(f"发现新的最佳值: {self.best_value:.4f}")
        
        # 计算并存储批次平均值
        batch_avg = np.mean(y)
        self.batch_avg_values.append(batch_avg)
        logging.info(f"批次的平均reward: {batch_avg:.4f}")
        
        return y
    
    def optimize(
        self,
        n_generations: int = 100,
        n_init: int = 40,
        save_freq: int = 10,
        surrogate_epochs: int = None,  # 保持接口兼容，但不使用
    ) -> OptimizationResult:
        """
        运行 NSGA-II 优化
        
        Args:
            n_generations: 迭代代数
            n_init: 初始种群大小（如果未指定，使用pop_size）
            save_freq: 保存结果的频率
            surrogate_epochs: 保持与贝叶斯优化接口兼容的参数（未使用）
            
        Returns:
            result: 优化结果
        """
        if n_init is None or n_init <= 0:
            n_init = self.pop_size
        
        logging.info(f"开始 NSGA-II 优化: {n_generations} 代，种群大小 {self.pop_size}...")
        
        # 定义问题
        problem = CircuitDesignProblem(
            eval_func=self.eval_func,
            n_var=self.input_dim,
            xl=self.bounds[0],
            xu=self.bounds[1]
        )
        
        # 定义算法
        algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=LHS(),  # 使用拉丁超立方采样，与贝叶斯优化的初始化相似
            crossover=SBX(prob=0.9, eta=15),  # 模拟二进制交叉
            mutation=PM(eta=20),  # 多项式变异
            # 锦标赛选择的大小设置与电路优化任务相适应
            selection=TournamentSelection(func_comp=lambda a, b: a <= b),
            eliminate_duplicates=True
        )

        # 设置显示类，兼容各版本的pymoo
        algorithm.display = SimpleDisplay()
        
        # 初始采样点评估（与贝叶斯优化类似的初始化）
        if n_init > 0:
            logging.info(f"生成 {n_init} 个初始样本...")
            
            # 生成初始种群
            initial_samples = LHS().do(problem, n_init).get("X")
            
            # 评估初始种群
            initial_values = self.evaluate_circuit_batch(initial_samples)
                
            # 生成可视化
            self._plot_batch_average_rewards(0)
            
            # 如果有结果跟踪器，更新它
            if self.result_tracker:
                for i, (params, value) in enumerate(zip(initial_samples, initial_values)):
                    self.result_tracker.update(0, params, value, initial_samples, initial_values)
        
        try:
            # 运行优化
            result = minimize(
                problem,
                algorithm,
                ('n_gen', n_generations),
                seed=42,
                verbose=True,
                save_history=True,
                callback=lambda algo: self._algorithm_callback(algo)
            )
            
            # 处理结果
            self._process_final_results(result)
            
            # 保存最终结果
            self._save_results('final')
            
            # 构建优化路径（历史最佳值序列）
            optimization_path = []
            best_so_far = -np.inf
            
            # 收集每次评估后的最佳值
            for reward in self.y_history:
                if reward > best_so_far:
                    best_so_far = reward
                optimization_path.append(best_so_far)
            
            # 返回优化结果
            return OptimizationResult(
                best_params=self.best_params,
                best_value=self.best_value,
                all_params=self.X_history,
                all_values=self.y_history,
                optimization_path=optimization_path,
                batch_avg_values=self.batch_avg_values
            )
            
        except Exception as e:
            logging.error(f"优化过程中出错: {str(e)}")
            logging.exception("详细错误信息:")
            
            # 发生错误时，尝试返回已有的最佳结果
            return OptimizationResult(
                best_params=self.best_params if self.best_params is not None else np.zeros(self.input_dim),
                best_value=self.best_value if self.best_value > -np.inf else 0.0,
                all_params=self.X_history,
                all_values=self.y_history,
                optimization_path=optimization_path,
                batch_avg_values=self.batch_avg_values
            )
    
    def _algorithm_callback(self, algorithm):
        """
        算法迭代回调函数
        
        Args:
            algorithm: NSGA-II 算法实例
        """
        # 获取当前种群
        pop = algorithm.pop
        
        # 获取当前种群的评估结果 (F是负值，需要取反)
        gen_values = -pop.get("F").flatten()
        
        # 评估并更新历史
        X = pop.get("X")
        
        # 更新最佳观测
        best_idx = np.argmax(gen_values)
        if gen_values[best_idx] > self.best_value:
            self.best_value = gen_values[best_idx]
            self.best_params = X[best_idx].copy()
            logging.info(f"第 {algorithm.n_gen} 代发现新的最佳值: {self.best_value:.4f}")
        
        # 计算并存储批次平均值
        batch_avg = np.mean(gen_values)
        self.batch_avg_values.append(batch_avg)
        logging.info(f"第 {algorithm.n_gen} 代的平均reward: {batch_avg:.4f}")
        
        # 保存历史
        for i, x in enumerate(X):
            if x.tolist() not in [arr.tolist() for arr in self.X_history]:
                self.X_history.append(x.copy())
                self.y_history.append(gen_values[i])
        
        # 添加批次平均 reward 图表
        self._plot_batch_average_rewards(algorithm.n_gen)
        
        # 更新结果跟踪器
        if self.result_tracker:
            self.result_tracker.update(
                algorithm.n_gen, 
                self.best_params, 
                self.best_value, 
                X, 
                gen_values
            )
        
        # 定期保存结果
        if algorithm.n_gen % 10 == 0:
            self._save_results(algorithm.n_gen)
            
        # 记录迭代时间
        curr_time = time.time()
        total_time = curr_time - self.start_time
        logging.info(f"第 {algorithm.n_gen} 代完成，总耗时 = {total_time:.2f}秒")
        
        return True  # 继续优化
    
    def _process_final_results(self, result):
        """
        处理最终优化结果
        
        Args:
            result: PyMOO 优化结果
        """
        # 获取最终种群
        final_pop = result.pop
        
        # 转换评估结果 (F是负值，需要取反)
        final_values = -final_pop.get("F").flatten()
        
        # 更新最佳观测
        best_idx = np.argmax(final_values)
        if final_values[best_idx] > self.best_value:
            self.best_value = final_values[best_idx]
            self.best_params = final_pop.get("X")[best_idx].copy()
            
        logging.info(f"优化完成！最佳值: {self.best_value:.4f}")
    
    def _save_results(self, iteration):
        """
        保存优化结果
        
        Args:
            iteration: 当前迭代次数
        """
        # 构建优化路径（历史最佳值序列）
        # 每次进行新的评估时，记录最佳reward
        optimization_path = []
        best_so_far = -np.inf
        eval_counts = []
        eval_count = 0
        
        # 收集每次评估后的最佳值
        for i, reward in enumerate(self.y_history):
            eval_count = i + 1
            if reward > best_so_far:
                best_so_far = reward
            optimization_path.append(best_so_far)
            eval_counts.append(eval_count)
        
        results = {
            'X_history': np.array(self.X_history),
            'y_history': np.array(self.y_history),
            'best_params': self.best_params,
            'best_value': self.best_value,
            'batch_avg_values': np.array(self.batch_avg_values),
            'optimization_path': np.array(optimization_path),
            'eval_counts': np.array(eval_counts)
        }
        
        save_path = os.path.join(self.save_dir, f'results_iter_{iteration}.npz')
        np.savez(save_path, **results)
        logging.info(f"结果已保存到: {save_path}")
        
        # 绘制历史优化结果（与BO相似）
        # 横轴：仿真评估次数，纵轴：历史最佳reward
        plt.figure(figsize=(10, 6))
        if len(optimization_path) > 0:
            plt.plot(eval_counts, optimization_path, '-o', color='blue')
            plt.xlabel('Simulation Count')
            plt.ylabel('Best Reward So Far')
            plt.title('Optimization History (Best Reward vs Simulation Count)')
            plt.grid(True)
            
            # 添加一些关键点的标注
            for i in range(0, len(eval_counts), max(1, len(eval_counts)//10)):
                plt.annotate(f"{optimization_path[i]:.3f}", 
                             (eval_counts[i], optimization_path[i]), 
                             textcoords="offset points",
                             xytext=(0, 10), 
                             ha='center')
        
        plot_path = os.path.join(self.save_dir, f'optimization_history_iter_{iteration}.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"优化历史图已保存到: {plot_path}")
        
        # 绘制世代平均Reward（与BO的批次平均对应）
        plt.figure(figsize=(10, 6))
        if len(self.batch_avg_values) > 0:
            generations = range(len(self.batch_avg_values))
            plt.plot(generations, self.batch_avg_values, '-o', color='green', label='Generation Average')
            
            # 同时绘制世代最佳值
            plt.plot(generations, [self.best_value] * len(self.batch_avg_values), 
                    '--', color='red', label='Global Best')
            
            plt.xlabel('Generation')
            plt.ylabel('Reward')
            plt.title('Generation Average Rewards')
            plt.grid(True)
            plt.legend()
            
            # 添加数值标签
            for i, avg in enumerate(self.batch_avg_values):
                plt.annotate(f'{avg:.3f}', (i, avg), textcoords="offset points", 
                            xytext=(0, 10), ha='center')
        
        gen_avg_path = os.path.join(self.save_dir, f'generation_avg_rewards_iter_{iteration}.png')
        plt.savefig(gen_avg_path)
        plt.close()
        logging.info(f"世代平均reward图已保存到: {gen_avg_path}")
    
    def _plot_batch_average_rewards(self, iteration):
        """
        绘制批次平均reward图表和历史优化结果图
        
        Args:
            iteration: 当前迭代次数
        """
        # ==================== 1. 绘制世代平均Reward图 ====================
        if not self.batch_avg_values:
            return
            
        plt.figure(figsize=(10, 6))
        
        # 绘制批次平均reward
        plt.plot(range(len(self.batch_avg_values)), 
                 self.batch_avg_values, 
                 '-o', color='green', label='Generation Average')
        
        # 如果有3个或以上的点，计算移动平均线
        if len(self.batch_avg_values) >= 3:
            window_size = min(3, len(self.batch_avg_values))
            moving_avg = np.convolve(self.batch_avg_values, 
                                     np.ones(window_size)/window_size, 
                                     mode='valid')
            # 绘制移动平均线
            x_vals = range(window_size-1, len(self.batch_avg_values))
            plt.plot(x_vals, moving_avg, '--', color='red', 
                     label=f'{window_size}-point Moving Avg')
        
        # 绘制当前全局最佳值
        if self.best_value > -np.inf:
            plt.axhline(y=self.best_value, color='blue', linestyle='--', 
                      label=f'Global Best: {self.best_value:.3f}')
        
        plt.xlabel('Generation')
        plt.ylabel('Average Reward')
        plt.title('Generation Average Rewards')
        plt.grid(True)
        plt.legend()
        
        # 添加数值标签
        for i, avg in enumerate(self.batch_avg_values):
            plt.annotate(f'{avg:.3f}', (i, avg), textcoords="offset points", 
                         xytext=(0, 10), ha='center')
        
        # 保存图片
        plot_path = os.path.join(self.save_dir, f'generation_avg_rewards_gen_{iteration}.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"世代平均reward图已保存到: {plot_path}")
        
        # ==================== 2. 绘制历史优化结果图 ====================
        # 构建优化路径（历史最佳值序列）
        if not self.y_history:
            return
            
        optimization_path = []
        best_so_far = -np.inf
        
        # 收集每次评估后的最佳值
        for reward in self.y_history:
            if reward > best_so_far:
                best_so_far = reward
            optimization_path.append(best_so_far)
        
        eval_counts = list(range(1, len(self.y_history) + 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(eval_counts, optimization_path, '-o', color='blue')
        plt.xlabel('Simulation Count')
        plt.ylabel('Best Reward So Far')
        plt.title('Optimization History (Best Reward vs Simulation Count)')
        plt.grid(True)
        
        # 添加一些关键点的标注
        step_size = max(1, len(eval_counts)//10)
        for i in range(0, len(eval_counts), step_size):
            plt.annotate(f"{optimization_path[i]:.3f}", 
                        (eval_counts[i], optimization_path[i]), 
                        textcoords="offset points",
                        xytext=(0, 10), 
                        ha='center')
        
        # 标记最终最佳值
        if len(optimization_path) > 0:
            plt.annotate(f"Best: {optimization_path[-1]:.4f}", 
                        (eval_counts[-1], optimization_path[-1]), 
                        textcoords="offset points",
                        xytext=(30, 0), 
                        arrowprops=dict(arrowstyle="->"))
        
        history_path = os.path.join(self.save_dir, f'optimization_history_sim_{iteration}.png')
        plt.savefig(history_path)
        plt.close()
        logging.info(f"优化历史图已保存到: {history_path}")