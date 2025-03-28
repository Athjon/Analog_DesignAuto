import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
import yaml
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

class OptimizationTracker:
    """
    Bayesian optimization result tracking and visualization tool
    
    Tracks optimization progress in real-time, generates charts, and saves detailed results
    """
    
    def __init__(self, save_dir: str, param_names: List[str] = None, 
                 live_update: bool = True, update_interval: int = 1):
        """
        Initialize results tracker
        
        Args:
            save_dir: Directory to save results
            param_names: List of parameter names
            live_update: Whether to update charts in real-time
            update_interval: Update interval (in iterations)
        """
        self.save_dir = save_dir
        self.param_names = param_names
        self.live_update = live_update
        self.update_interval = update_interval
        
        # Create results directories
        self.plots_dir = os.path.join(save_dir, "plots")
        self.data_dir = os.path.join(save_dir, "data")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize results storage
        self.optimization_path = []
        self.best_rewards = []
        self.all_rewards = []
        self.best_params = None
        self.best_reward = -np.inf
        self.iteration = 0
        self.start_time = time.time()
        
        # Initialize charts if live updating
        if self.live_update:
            plt.ion()  # Enable interactive mode
            self.fig_reward, self.ax_reward = plt.subplots(figsize=(10, 6))
            self.fig_reward.suptitle('Bayesian Optimization Progress', fontsize=16)
            self.ax_reward.set_xlabel('Iteration')
            self.ax_reward.set_ylabel('Best Reward')
            self.reward_line, = self.ax_reward.plot([], [], 'b-o', label='Best Reward')
            self.ax_reward.legend()
            self.ax_reward.grid(True)
            
            # Create HTML dashboard
            self.create_html_dashboard()
            
    def update(self, iteration: int, params: np.ndarray, reward: float, 
               all_params: Optional[List[np.ndarray]] = None, 
               all_rewards: Optional[List[float]] = None):
        """
        Update optimization results
        
        Args:
            iteration: Current iteration number
            params: Current parameters
            reward: Current reward
            all_params: All parameters evaluated in this iteration
            all_rewards: All rewards from this iteration
        """
        self.iteration = iteration
        
        # Use proper list extension
        if all_rewards is not None:
            # Convert to list if numpy array
            if isinstance(all_rewards, np.ndarray):
                all_rewards = all_rewards.tolist()
            self.all_rewards.extend(all_rewards)
        else:
            self.all_rewards.append(reward)
        
        # Check if this is the best value
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_params = params.copy() if isinstance(params, np.ndarray) else params
            
        # Save optimization path
        self.best_rewards.append(self.best_reward)
        self.optimization_path.append({
            'iteration': iteration,
            'reward': float(reward),
            'best_reward': float(self.best_reward),
            'elapsed_time': time.time() - self.start_time
        })
        
        # Save current iteration data
        self.save_iteration_data(iteration, params, reward, all_params, all_rewards)
        
        # Update charts
        if self.live_update and (iteration % self.update_interval == 0):
            self.update_plots()
            
        # Always update summary file
        self.save_summary()
        
        # Save parameter importance plot every 5 iterations
        if all_params is not None and all_rewards is not None and iteration % 5 == 0:
            self.plot_parameter_importance(all_params, all_rewards)
        
    def update_plots(self):
        """Update all chart displays"""
        # Update reward curve
        x = list(range(len(self.best_rewards)))
        self.reward_line.set_data(x, self.best_rewards)
        self.ax_reward.relim()
        self.ax_reward.autoscale_view()
        
        # Set appropriate Y-axis range
        if len(self.best_rewards) > 1:
            y_min = min(self.best_rewards)
            y_max = max(self.best_rewards)
            margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
            self.ax_reward.set_ylim([y_min - margin, y_max + margin])
            
        # Refresh chart
        self.fig_reward.canvas.draw_idle()
        self.fig_reward.canvas.flush_events()
        
        # Save charts
        self.save_plots()
        
        # Update HTML dashboard
        self.update_html_dashboard()
        
    def save_plots(self):
        """Save all charts to files"""
        # Save reward curve
        plt.figure(self.fig_reward.number)
        plt.savefig(os.path.join(self.plots_dir, 'optimization_progress.png'), dpi=300)
        plt.savefig(os.path.join(self.plots_dir, 'optimization_progress.svg'), format='svg')
        
    def save_iteration_data(self, iteration: int, params: np.ndarray, reward: float,
                           all_params: Optional[List[np.ndarray]] = None,
                           all_rewards: Optional[List[float]] = None):
        """
        Save detailed data for each iteration
        
        Args:
            iteration: Current iteration number
            params: Parameters from this iteration
            reward: Reward value from this iteration
            all_params: All parameters evaluated in this iteration
            all_rewards: All rewards from this iteration
        """
        # Convert NumPy arrays to lists for JSON serialization
        params_list = params.tolist() if isinstance(params, np.ndarray) else params
        
        iter_data = {
            'iteration': iteration,
            'params': params_list,
            'reward': float(reward),
            'best_reward': float(self.best_reward),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_time': time.time() - self.start_time
        }
        
        # Add named parameters if parameter names are available
        if self.param_names and len(self.param_names) == len(params):
            iter_data['named_params'] = {name: float(value) 
                                        for name, value in zip(self.param_names, params)}
        
        # Save to JSON file
        with open(os.path.join(self.data_dir, f'iteration_{iteration:04d}.json'), 'w') as f:
            json.dump(iter_data, f, indent=2)
            
        # If we have all evaluations for this iteration, save as CSV
        if all_params is not None and all_rewards is not None:
            try:
                # Convert list of numpy arrays to 2D array
                if isinstance(all_params, list):
                    if isinstance(all_params[0], np.ndarray):
                        params_array = np.vstack(all_params)
                    else:
                        params_array = np.array(all_params)
                else:
                    params_array = all_params
                
                # Convert rewards to array
                if isinstance(all_rewards, list):
                    rewards_array = np.array(all_rewards)
                else:
                    rewards_array = all_rewards
                    
                # Reshape rewards for concatenation
                if len(rewards_array.shape) == 1:
                    rewards_array = rewards_array.reshape(-1, 1)
                
                # Create dataframe
                data = np.hstack([params_array, rewards_array])
                if len(self.param_names) != params_array.shape[1]:
                    # 使用数据实际的列数生成通用列名
                    columns = [f'param_{i}' for i in range(params_array.shape[1])]
                else:
                    columns = self.param_names.copy()
                columns += ['reward']

                df = pd.DataFrame(data, columns=columns)
                df.to_csv(os.path.join(self.data_dir, f'evaluations_{iteration:04d}.csv'), index=False)
            except Exception as e:
                logging.error(f"Error saving evaluations CSV: {e}")
                logging.exception("Evaluation save error details:")
            
    def save_summary(self):
        """Save optimization summary information"""
        summary = {
            'iterations_completed': self.iteration,
            'best_reward': float(self.best_reward),
            'elapsed_time': time.time() - self.start_time,
            'start_time': datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S'),
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # Add best parameters
        if self.best_params is not None:
            # Convert NumPy arrays to lists for serialization
            params_list = self.best_params.tolist() if isinstance(self.best_params, np.ndarray) else self.best_params
            summary['best_params'] = params_list
            
            if self.param_names and len(self.param_names) == len(self.best_params):
                summary['best_named_params'] = {name: float(value) 
                                               for name, value in zip(self.param_names, self.best_params)}
        
        # Save as YAML and JSON
        with open(os.path.join(self.save_dir, 'optimization_summary.yaml'), 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
            
        with open(os.path.join(self.save_dir, 'optimization_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
    def plot_parameter_importance(self, all_params: List[np.ndarray], all_rewards: List[float]):
        """
        Analyze parameter importance and plot charts
        
        Args:
            all_params: List of parameter arrays or 2D array of parameters
            all_rewards: List of reward values
        """
        # Ensure we have enough data
        if isinstance(all_params, list):
            if len(all_params) < 10:
                return
        elif isinstance(all_params, np.ndarray):
            if all_params.shape[0] < 10:
                return
            
        if isinstance(all_rewards, list):
            if len(all_rewards) < 10:
                return
        elif isinstance(all_rewards, np.ndarray):
            if all_rewards.shape[0] < 10:
                return
            
        # Convert parameters and rewards to arrays
        # Ensure all_params is properly converted to a 2D array
        if isinstance(all_params, list):
            if isinstance(all_params[0], np.ndarray):
                try:
                    X = np.vstack(all_params)  # Stack arrays in sequence vertically
                except ValueError:
                    # If arrays have different shapes, try a different approach
                    X = np.array([p.flatten() if isinstance(p, np.ndarray) else p for p in all_params])
            else:
                X = np.array(all_params)
        else:
            X = all_params
            
        # Convert rewards to array
        if isinstance(all_rewards, list):
            y = np.array(all_rewards)
        else:
            y = all_rewards
            
        # Ensure y is 1D
        if len(y.shape) > 1:
            y = y.flatten()
            
        # Use correlation analysis
        correlations = []
        param_count = X.shape[1]
        
        for i in range(param_count):
            try:
                # Calculate correlation coefficient
                correlation = np.corrcoef(X[:, i], y)[0, 1]
                # Handle NaN values
                if np.isnan(correlation):
                    correlation = 0.0
                correlations.append((i, abs(correlation)))
            except Exception as e:
                logging.warning(f"Error calculating correlation for parameter {i}: {e}")
                correlations.append((i, 0.0))
            
        # Sort by correlation magnitude
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Plot parameter importance
        plt.figure(figsize=(12, 8))
        indices = [c[0] for c in correlations]
        importance = [c[1] for c in correlations]
        
        # Get labels for parameters
        if self.param_names and len(indices) > 0:
            # Make sure we don't access beyond the available parameter names
            labels = []
            for i in indices:
                if i < len(self.param_names):
                    labels.append(self.param_names[i])
                else:
                    labels.append(f'param_{i}')
        else:
            labels = [f'param_{i}' for i in indices]
        
        # Only show top 20 parameters
        max_display = min(20, len(indices))
        if max_display > 0:
            indices = indices[:max_display]
            importance = importance[:max_display]
            labels = labels[:max_display]
            
            plt.barh(range(len(indices)), importance, align='center')
            plt.yticks(range(len(indices)), labels)
            plt.xlabel('Parameter Importance (Absolute Correlation)')
            plt.title('Parameter Importance Analysis')
            plt.tight_layout()
            
            # Save chart
            plt.savefig(os.path.join(self.plots_dir, f'parameter_importance_iter_{self.iteration}.png'), dpi=300)
            plt.savefig(os.path.join(self.plots_dir, f'parameter_importance_iter_{self.iteration}.svg'), format='svg')
            plt.close()
        
    def create_html_dashboard(self):
        """Create HTML dashboard page"""
        html_path = os.path.join(self.save_dir, 'dashboard.html')
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bayesian Optimization Dashboard</title>
            <meta http-equiv="refresh" content="10">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; }
                .plots { display: flex; flex-wrap: wrap; justify-content: center; }
                .plot-container { margin: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                .metrics { display: flex; flex-wrap: wrap; margin-top: 30px; }
                .metric-card { 
                    flex: 1; min-width: 200px; margin: 10px; padding: 20px; 
                    background-color: #f8f9fa; border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    text-align: center;
                }
                .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
                .metric-title { font-size: 14px; color: #666; }
                .timestamp { text-align: center; margin-top: 20px; color: #666; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Bayesian Optimization Progress Dashboard</h1>
                    <p>Real-time monitoring of optimization progress and results</p>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-title">Current Iteration</div>
                        <div class="metric-value" id="current-iteration">0</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Best Reward</div>
                        <div class="metric-value" id="best-reward">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Running Time</div>
                        <div class="metric-value" id="elapsed-time">-</div>
                    </div>
                </div>
                
                <div class="plots">
                    <div class="plot-container">
                        <h3>Optimization Progress</h3>
                        <img src="plots/optimization_progress.png" width="600" id="progress-plot">
                    </div>
                    <div class="plot-container">
                        <h3>Parameter Importance Analysis</h3>
                        <img src="" width="600" id="importance-plot">
                    </div>
                </div>
                
                <div class="timestamp" id="update-time"></div>
            </div>
            
            <script>
                // Dynamically update image src to avoid caching
                function updateImages() {
                    const timestamp = new Date().getTime();
                    document.getElementById('progress-plot').src = 'plots/optimization_progress.png?' + timestamp;
                    
                    // Get latest parameter importance plot
                    fetch('optimization_summary.json?' + timestamp)
                        .then(response => response.json())
                        .then(data => {
                            const iteration = data.iterations_completed;
                            const importancePlot = document.getElementById('importance-plot');
                            importancePlot.src = `plots/parameter_importance_iter_${iteration}.png?` + timestamp;
                            
                            // Update metrics
                            document.getElementById('current-iteration').textContent = iteration;
                            document.getElementById('best-reward').textContent = data.best_reward.toFixed(4);
                            
                            const elapsedSeconds = data.elapsed_time;
                            const hours = Math.floor(elapsedSeconds / 3600);
                            const minutes = Math.floor((elapsedSeconds % 3600) / 60);
                            const seconds = Math.floor(elapsedSeconds % 60);
                            document.getElementById('elapsed-time').textContent = 
                                `${hours}h ${minutes}m ${seconds}s`;
                                
                            document.getElementById('update-time').textContent = 
                                `Last updated: ${new Date().toLocaleString()}`;
                        })
                        .catch(error => console.error('Error fetching data:', error));
                }
                
                // Initial load
                updateImages();
                
                // Update every 10 seconds
                setInterval(updateImages, 10000);
            </script>
        </body>
        </html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logging.info(f"Created HTML dashboard: {html_path}")
        logging.info(f"Open this file in a browser to view real-time optimization progress")
        
    def update_html_dashboard(self):
        """Update HTML dashboard data"""
        # HTML dashboard auto-refreshes by loading JSON data periodically
        # Just ensure summary JSON is updated
        self.save_summary()
        
    def final_report(self):
        """Generate final optimization report"""
        # Create detailed final report
        report_path = os.path.join(self.save_dir, 'final_report.md')
        
        # Calculate statistics
        if len(self.all_rewards) > 0:
            mean_reward = np.mean(self.all_rewards)
            median_reward = np.median(self.all_rewards)
            std_reward = np.std(self.all_rewards)
        else:
            mean_reward = median_reward = std_reward = 0.0
            
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        # Create report content
        report = f"""# Bayesian Optimization Final Report

## Optimization Summary

- **Start Time**: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}
- **End Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Runtime**: {hours} hours {minutes} minutes {seconds} seconds
- **Total Iterations**: {self.iteration}
- **Best Reward**: {self.best_reward:.6f}
- **Average Reward**: {mean_reward:.6f}
- **Reward Std Dev**: {std_reward:.6f}

## Best Parameters

```
"""
        
        # Add best parameters
        if self.best_params is not None and self.param_names:
            for name, value in zip(self.param_names, self.best_params):
                report += f"{name}: {value:.6f}\n"
        elif self.best_params is not None:
            for i, value in enumerate(self.best_params):
                report += f"param_{i}: {value:.6f}\n"
                
        report += """```

## Optimization Progress

![Optimization Progress](plots/optimization_progress.png)

## Parameter Importance Analysis

![Parameter Importance](plots/parameter_importance_iter_{}.png)

## Notes

- See the `data` directory for detailed evaluation data for each iteration
- All charts can be found in the `plots` directory
- Complete optimization results are saved in the `optimization_summary.yaml` file

""".format(self.iteration)

        with open(report_path, 'w') as f:
            f.write(report)
            
        logging.info(f"Generated final report: {report_path}")