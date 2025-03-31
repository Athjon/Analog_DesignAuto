import numpy as np
import logging
import time
from typing import Callable, List
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

class ParallelEvaluator:
    """
    并行电路设计评估器
    使用多进程对多个电路参数设计进行并行评估
    """
    
    def __init__(self, eval_func: Callable, n_workers: int = None):
        """
        初始化并行评估器
        
        Args:
            eval_func: 评估单个参数集的函数
            n_workers: 并行工作进程数量（默认为CPU核心数）
        """
        self.eval_func = eval_func
        # 确保至少有一个工作进程，最多使用所有CPU核心减1
        self.n_workers = n_workers if n_workers is not None else max(1, cpu_count() - 1)
        
        logging.info(f"并行评估器初始化完成，工作进程数: {self.n_workers}")
    
    def _worker_init(self):
        """工作进程初始化函数"""
        # 设置随机种子，确保各进程随机性独立
        np.random.seed(int(time.time() * 1000) % 100000 + np.random.randint(10000))
        logging.debug("工作进程初始化完成")
    
    def evaluate_batch(self, param_batch: np.ndarray) -> np.ndarray:
        """
        并行评估一批参数设计
        
        Args:
            param_batch: 要评估的参数设计批次
            
        Returns:
            results: 评估结果数组
        """
        n_samples = len(param_batch)
        logging.info(f"开始并行评估 {n_samples} 个设计...")
        
        start_time = time.time()
        
        # 单工作进程情况 - 使用顺序评估
        if self.n_workers == 1 or n_samples == 1:
            logging.info("使用单进程顺序评估")
            results = []
            for i, params in enumerate(tqdm(param_batch, desc="评估进度")):
                try:
                    result = self.eval_func(params)
                    results.append(result)
                except Exception as e:
                    logging.error(f"评估第 {i} 个设计时出错: {str(e)}")
                    # 评估失败时返回极小值
                    results.append(-np.inf)
            
            end_time = time.time()
            logging.info(f"批量评估完成，耗时: {end_time - start_time:.2f}秒")
            return np.array(results)
        
        # 多进程情况
        try:
            logging.info(f"使用 {self.n_workers} 个并行进程进行评估")
            
            # 准备参数列表
            param_list = [p for p in param_batch]
            
            # 使用进程池并行评估
            with Pool(processes=self.n_workers, initializer=self._worker_init) as pool:
                # 带进度条的并行映射
                results = list(tqdm(
                    pool.imap(self._safe_eval, param_list),
                    total=len(param_list),
                    desc="并行评估进度"
                ))
            
            # 处理结果，确保所有评估都返回有效值
            processed_results = []
            for i, res in enumerate(results):
                if res is None or np.isnan(res) or np.isinf(res):
                    logging.warning(f"第 {i} 个设计评估返回无效结果，替换为默认值")
                    processed_results.append(-np.inf)
                else:
                    processed_results.append(res)
            
            end_time = time.time()
            logging.info(f"并行批量评估完成，耗时: {end_time - start_time:.2f}秒，平均每个设计: {(end_time - start_time) / n_samples:.2f}秒")
            
            return np.array(processed_results)
        
        except Exception as e:
            logging.error(f"并行评估过程中发生错误: {str(e)}")
            logging.exception("详细错误信息:")
            
            # 发生错误时回退到顺序评估
            logging.warning("回退到顺序评估")
            results = []
            for i, params in enumerate(tqdm(param_batch, desc="评估进度(回退)")):
                try:
                    result = self.eval_func(params)
                    results.append(result)
                except Exception as e:
                    logging.error(f"顺序评估第 {i} 个设计时出错: {str(e)}")
                    results.append(-np.inf)
            
            end_time = time.time()
            logging.info(f"回退评估完成，耗时: {end_time - start_time:.2f}秒")
            return np.array(results)
    
    def _safe_eval(self, params):
        """
        安全的评估函数包装器，捕获异常
        
        Args:
            params: 要评估的参数
            
        Returns:
            result: 评估结果或出错时的默认值
        """
        try:
            return self.eval_func(params)
        except Exception as e:
            logging.error(f"评估过程中出错: {str(e)}")
            return -np.inf  # 返回极小值表示评估失败