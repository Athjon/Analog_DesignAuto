import numpy as np
import multiprocessing
from typing import List, Callable, Any
import time
import logging
import os
import tqdm

class ParallelEvaluator:
    """
    Class implementing parallel circuit evaluation for Bayesian optimization
    
    Uses Python's multiprocessing library to evaluate multiple circuit configurations in parallel
    """
    
    def __init__(self, eval_func: Callable, n_workers: int = None):
        """
        Initialize parallel evaluator
        
        Args:
            eval_func: Evaluation function that takes parameter array and returns reward value
            n_workers: Number of parallel worker processes, defaults to 90% of CPU cores
        """
        if n_workers is None:
            # Default to 90% of available CPU cores
            n_workers = max(1, int(multiprocessing.cpu_count() * 0.9))
            
        self.n_workers = n_workers
        self.eval_func = eval_func
        logging.info(f"Initializing parallel evaluator with {self.n_workers} worker processes")
            
    def evaluate_batch(self, params_batch: np.ndarray) -> np.ndarray:
        """
        Evaluate a batch of parameters in parallel
        
        Args:
            params_batch: Batch of parameter arrays with shape [batch_size, param_dims]
            
        Returns:
            rewards: Array of evaluation rewards
        """
        start_time = time.time()
        n_samples = len(params_batch)
        
        if n_samples == 0:
            return np.array([])
            
        if n_samples == 1:
            # For single sample, evaluate directly
            return np.array([self.eval_func(params_batch[0])])
            
        # Use process pool for parallel evaluation
        with multiprocessing.Pool(processes=min(self.n_workers, n_samples)) as pool:
            # Use tqdm for progress bar
            rewards = list(tqdm.tqdm(
                pool.imap(self.eval_func, params_batch),
                total=n_samples,
                desc="Parallel circuit evaluation",
                unit="samples"
            ))
            
        elapsed = time.time() - start_time
        logging.info(f"Completed parallel evaluation of {n_samples} samples in {elapsed:.2f} seconds, average {elapsed/n_samples:.2f} seconds per sample")
        
        return np.array(rewards)
        
    def evaluate_initial_samples(self, n_samples: int, param_dims: int, 
                                bounds: tuple) -> tuple:
        """
        Evaluate initial random samples in parallel
        
        Args:
            n_samples: Number of initial samples
            param_dims: Parameter dimensionality
            bounds: Parameter boundaries as (lower_bounds, upper_bounds)
            
        Returns:
            tuple: (X_samples, y_values) - parameter samples and corresponding rewards
        """
        logging.info(f"Generating and evaluating {n_samples} initial random samples")
        
        # Generate uniformly distributed random samples in parameter space
        X_samples = np.random.uniform(
            bounds[0], bounds[1], size=(n_samples, param_dims)
        )
        
        # Evaluate samples in parallel
        y_values = self.evaluate_batch(X_samples)
        
        return X_samples, y_values