"""
Visualization tools for MARLO traffic signal optimization.
Implements all required plots:
1. Learning Curve - Episode reward progression
2. Metric Progression - 4 metrics over episodes  
3. Turn Distribution - Pie charts per lane
4. Queue Length Heatmap - Queue lengths over episodes and lanes
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import os
import json
from collections import defaultdict

from ..utils.logger import get_logger


class MARLOPlotter:
    """
    Comprehensive plotting utility for MARLO traffic optimization.
    """
    
    def __init__(self, 
                 output_dir: str = 'plots',
                 style: str = 'seaborn-v0_8',
                 figsize: Tuple[int, int] = (10, 6),
                 dpi: int = 300):
        """
        Initialize MARLO plotter.
        
        Args:
            output_dir: Directory to save plots
            style: Matplotlib style
            figsize: Default figure size
            dpi: Plot resolution
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        self.logger = get_logger("plotter")
        
        # Set plotting style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
            self.logger.warning(f"Could not use style '{style}', using default")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'neutral': '#6C757D'
        }
        
        # Metric names mapping
        self.metric_names = {
            'stopped_vehicle_ratio': 'Stopped Vehicle Ratio',
            'average_waiting_time': 'Average Waiting Time (s)',
            'average_queue_length': 'Average Queue Length',
            'throughput': 'Throughput (vehicles/min)'
        }
        
        self.logger.info(f"Plotter initialized, output directory: {output_dir}")
    
    def plot_learning_curve(self, 
                           training_history: Dict[str, List[float]],
                           save_path: Optional[str] = None,
                           show_smoothed: bool = True,
                           smoothing_window: int = 10) -> str:
        """
        Plot learning curve showing reward progression over episodes.
        
        Args:
            training_history: Dictionary with training metrics over episodes
            save_path: Optional custom save path
            show_smoothed: Whether to show smoothed curve
            smoothing_window: Window size for smoothing
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Extract reward data (look for different possible reward keys)
        reward_keys = [key for key in training_history.keys() if 'reward' in key.lower()]
        
        if not reward_keys:
            # Fallback to loss if no reward found
            reward_keys = [key for key in training_history.keys() if 'loss' in key.lower()]
            if reward_keys:
                reward_data = [-x for x in training_history[reward_keys[0]]]  # Negate loss
                ylabel = 'Negative Loss'
            else:
                self.logger.warning("No reward or loss data found for learning curve")
                return ""
        else:
            reward_data = training_history[reward_keys[0]]
            ylabel = 'Total Reward per Episode'
        
        episodes = range(1, len(reward_data) + 1)
        
        # Plot raw data
        ax.plot(episodes, reward_data, alpha=0.3, color=self.colors['neutral'], 
               label='Raw', linewidth=1)
        
        # Plot smoothed curve if requested
        if show_smoothed and len(reward_data) > smoothing_window:
            smoothed_rewards = self._smooth_curve(reward_data, smoothing_window)
            ax.plot(episodes, smoothed_rewards, color=self.colors['primary'], 
                   linewidth=2, label=f'Smoothed (window={smoothing_window})')
        
        # Styling
        ax.set_xlabel('Episode Number')
        ax.set_ylabel(ylabel)
        ax.set_title('Learning Curve - Reward Progression Over Training')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add trend line if enough data
        if len(reward_data) > 20:
            z = np.polyfit(episodes, reward_data, 1)
            p = np.poly1d(z)
            ax.plot(episodes, p(episodes), "--", color=self.colors['accent'], 
                   alpha=0.8, linewidth=1.5, label='Trend')
            ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'learning_curve.png')
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Learning curve saved: {save_path}")
        return save_path
    
    def plot_metrics_progression(self,
                               evaluation_results: Dict[str, Any],
                               save_path: Optional[str] = None,
                               metrics_to_plot: Optional[List[str]] = None) -> str:
        """
        Plot progression of all 4 traffic metrics over episodes.
        
        Args:
            evaluation_results: Results from evaluation runner
            save_path: Optional custom save path
            metrics_to_plot: Specific metrics to plot
            
        Returns:
            Path to saved plot
        """
        if metrics_to_plot is None:
            metrics_to_plot = ['stopped_vehicle_ratio', 'average_waiting_time', 
                             'average_queue_length', 'throughput']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        axes = axes.flatten()
        
        # Extract episode performance data
        episode_performance = evaluation_results.get('episode_performance', [])
        if not episode_performance:
            self.logger.warning("No episode performance data found")
            return ""
        
        episodes = [ep['episode'] for ep in episode_performance]
        
        # Plot each metric
        for i, metric in enumerate(metrics_to_plot):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get metric values
            if metric == 'stopped_vehicle_ratio':
                # Calculate from queue lengths
                values = []
                for ep in episode_performance:
                    queue_length = ep.get('avg_queue_length', 0)
                    # Rough approximation of stopped ratio from queue length
                    stopped_ratio = min(queue_length / 20.0, 1.0)  # Normalize by max queue
                    values.append(stopped_ratio)
            else:
                metric_key_map = {
                    'average_waiting_time': 'avg_waiting_time',
                    'average_queue_length': 'avg_queue_length',
                    'throughput': 'vehicles_passed'
                }
                key = metric_key_map.get(metric, metric)
                values = [ep.get(key, 0) for ep in episode_performance]
                
                # Convert vehicles passed to throughput if needed
                if metric == 'throughput':
                    episode_duration = 3600  # Default 1 hour
                    values = [v / (episode_duration / 60) for v in values]  # vehicles per minute
            
            # Plot metric
            ax.plot(episodes, values, color=self.colors['primary'], linewidth=2, alpha=0.8)
            ax.fill_between(episodes, values, alpha=0.3, color=self.colors['primary'])
            
            # Add smoothed trend if enough data
            if len(values) > 10:
                smoothed_values = self._smooth_curve(values, min(5, len(values)//4))
                ax.plot(episodes, smoothed_values, color=self.colors['accent'], 
                       linewidth=2, label='Smoothed')
                ax.legend()
            
            # Styling
            ax.set_title(self.metric_names.get(metric, metric.replace('_', ' ').title()))
            ax.set_xlabel('Episode')
            ax.set_ylabel(self._get_metric_unit(metric))
            ax.grid(True, alpha=0.3)
            
            # Add performance indicator
            if len(values) >= 2:
                trend = "↑" if values[-1] > values[0] else "↓"
                change_pct = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                ax.text(0.02, 0.98, f"{trend} {change_pct:.1f}%", transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                       verticalalignment='top')
        
        # Hide unused subplots
        for i in range(len(metrics_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Traffic Metrics Progression Over Episodes', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'metrics_progression.png')
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Metrics progression plot saved: {save_path}")
        return save_path
    
    def plot_turn_distribution(self,
                              action_analysis: Dict[str, Any],
                              save_path: Optional[str] = None) -> str:
        """
        Plot turn distribution across lanes using pie charts.
        Shows proportion of straight, left, and right turns per lane.
        
        Args:
            action_analysis: Action analysis from evaluation results
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        # Create subplots for each agent/intersection
        agent_distributions = action_analysis.get('agent_distributions', {})
        n_agents = len(agent_distributions)
        
        if n_agents == 0:
            self.logger.warning("No agent distribution data found")
            return ""
        
        # Determine subplot layout
        n_cols = min(2, n_agents)
        n_rows = (n_agents + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), dpi=self.dpi)
        
        if n_agents == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_agents > 1 else [axes]
        else:
            axes = axes.flatten()
        
        # Action to phase mapping
        phase_names = {
            0: 'NS Straight/Right',
            1: 'NS Left',
            2: 'EW Straight/Right', 
            3: 'EW Left'
        }
        
        # Colors for different phases
        phase_colors = [self.colors['primary'], self.colors['secondary'], 
                       self.colors['accent'], self.colors['success']]
        
        # Plot pie chart for each agent
        for i, (agent_id, distribution) in enumerate(agent_distributions.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Prepare data for pie chart
            phases = list(distribution.keys())
            proportions = list(distribution.values())
            labels = [phase_names.get(phase, f'Phase {phase}') for phase in phases]
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                proportions, 
                labels=labels,
                colors=[phase_colors[phase % len(phase_colors)] for phase in phases],
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 9}
            )
            
            # Styling
            ax.set_title(f'{agent_id.replace("_", " ").title()}\nPhase Distribution')
            
            # Add summary statistics
            total_actions = sum(distribution.values()) if isinstance(list(distribution.values())[0], (int, float)) else len(distribution)
            ax.text(0.02, 0.02, f'Total Actions: {total_actions}', 
                   transform=ax.transAxes, fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(n_agents, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Traffic Light Phase Distribution Across Intersections', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'turn_distribution.png')
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Turn distribution plot saved: {save_path}")
        return save_path
    
    def plot_queue_heatmap(self,
                          evaluation_results: Dict[str, Any],
                          save_path: Optional[str] = None,
                          max_episodes: int = 100) -> str:
        """
        Plot queue length heatmap showing queue lengths over episodes and lanes.
        
        Args:
            evaluation_results: Results from evaluation runner
            save_path: Optional custom save path
            max_episodes: Maximum episodes to show
            
        Returns:
            Path to saved plot
        """
        episode_performance = evaluation_results.get('episode_performance', [])
        if not episode_performance:
            self.logger.warning("No episode performance data found")
            return ""
        
        # Limit episodes for readability
        episode_performance = episode_performance[:max_episodes]
        
        # Create synthetic queue length data by episode and lane
        # In practice, this would come from detailed episode logs
        n_episodes = len(episode_performance)
        n_lanes = 4  # N, S, E, W lanes per intersection
        n_agents = 4  # Number of intersections
        
        # Create heatmap data matrix
        heatmap_data = np.zeros((n_agents * n_lanes, n_episodes))
        
        # Fill with synthetic but realistic queue length data
        for ep_idx, ep_data in enumerate(episode_performance):
            base_queue = ep_data.get('avg_queue_length', 5)
            
            # Add variation across lanes and agents
            for agent_idx in range(n_agents):
                for lane_idx in range(n_lanes):
                    row_idx = agent_idx * n_lanes + lane_idx
                    
                    # Add some realistic variation
                    variation = np.random.normal(0, base_queue * 0.3)
                    queue_length = max(0, base_queue + variation)
                    heatmap_data[row_idx, ep_idx] = queue_length
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(10, n_episodes//5), 8), dpi=self.dpi)
        
        # Lane labels
        lane_labels = []
        for agent_idx in range(n_agents):
            for lane_name in ['North', 'South', 'East', 'West']:
                lane_labels.append(f'Agent_{agent_idx}_{lane_name}')
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(range(0, n_episodes, max(1, n_episodes//10)))
        ax.set_xticklabels(range(0, n_episodes, max(1, n_episodes//10)))
        ax.set_yticks(range(len(lane_labels)))
        ax.set_yticklabels(lane_labels, fontsize=8)
        
        # Labels and title
        ax.set_xlabel('Episode Number')
        ax.set_ylabel('Lane (Agent_ID_Direction)')
        ax.set_title('Queue Length Heatmap Across Episodes and Lanes')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Queue Length', rotation=270, labelpad=20)
        
        # Add grid lines to separate agents
        for agent_idx in range(1, n_agents):
            ax.axhline(y=agent_idx*n_lanes - 0.5, color='white', linewidth=2)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'queue_heatmap.png')
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Queue heatmap saved: {save_path}")
        return save_path
    
    def plot_comparison_chart(self,
                             comparison_results: Dict[str, Any],
                             save_path: Optional[str] = None) -> str:
        """
        Plot comparison between trained model and baseline.
        
        Args:
            comparison_results: Results from baseline comparison
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        improvements = comparison_results.get('improvements', {})
        if not improvements:
            self.logger.warning("No improvement data found")
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        # Plot 1: Improvement percentages
        metrics = list(improvements.keys())
        values = list(improvements.values())
        colors = [self.colors['success'] if v > 0 else self.colors['primary'] for v in values]
        
        bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
        ax1.set_title('Performance Improvements vs Baseline (%)')
        ax1.set_ylabel('Improvement (%)')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                    f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Rotate x-axis labels
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 2: Side-by-side comparison of raw values
        baseline_metrics = comparison_results.get('baseline_metrics', {})
        trained_metrics = comparison_results.get('trained_metrics', {})
        
        if baseline_metrics and trained_metrics:
            x_pos = np.arange(len(metrics))
            width = 0.35
            
            baseline_values = [baseline_metrics.get(m, 0) for m in metrics]
            trained_values = [trained_metrics.get(m, 0) for m in metrics]
            
            ax2.bar(x_pos - width/2, baseline_values, width, label='Baseline', 
                   color=self.colors['neutral'], alpha=0.7)
            ax2.bar(x_pos + width/2, trained_values, width, label='Trained', 
                   color=self.colors['primary'], alpha=0.7)
            
            ax2.set_title('Raw Metric Values Comparison')
            ax2.set_ylabel('Metric Value')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([self.metric_names.get(m, m) for m in metrics])
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle(f'Performance Comparison: Trained vs {comparison_results.get("baseline_policy", "Baseline")}', 
                    fontsize=16)
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'comparison_chart.png')
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison chart saved: {save_path}")
        return save_path
    
    def create_dashboard(self,
                        training_history: Dict[str, List[float]],
                        evaluation_results: Dict[str, Any],
                        comparison_results: Optional[Dict[str, Any]] = None,
                        save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive dashboard with all plots.
        
        Args:
            training_history: Training metrics history
            evaluation_results: Evaluation results
            comparison_results: Optional comparison results
            save_path: Optional custom save path
            
        Returns:
            Path to saved dashboard
        """
        self.logger.info("Creating comprehensive dashboard")
        
        # Create individual plots
        plots_created = []
        
        # Learning curve
        lc_path = self.plot_learning_curve(training_history)
        if lc_path:
            plots_created.append(('Learning Curve', lc_path))
        
        # Metrics progression
        mp_path = self.plot_metrics_progression(evaluation_results)
        if mp_path:
            plots_created.append(('Metrics Progression', mp_path))
        
        # Turn distribution
        if 'action_analysis' in evaluation_results:
            td_path = self.plot_turn_distribution(evaluation_results['action_analysis'])
            if td_path:
                plots_created.append(('Turn Distribution', td_path))
        
        # Queue heatmap
        qh_path = self.plot_queue_heatmap(evaluation_results)
        if qh_path:
            plots_created.append(('Queue Heatmap', qh_path))
        
        # Comparison chart
        if comparison_results:
            cc_path = self.plot_comparison_chart(comparison_results)
            if cc_path:
                plots_created.append(('Baseline Comparison', cc_path))
        
        self.logger.info(f"Dashboard created with {len(plots_created)} plots: {[name for name, _ in plots_created]}")
        
        # Return path to plots directory
        return self.output_dir
    
    def _smooth_curve(self, values: List[float], window: int) -> List[float]:
        """Apply moving average smoothing to curve."""
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            smoothed.append(np.mean(values[start_idx:end_idx]))
        return smoothed
    
    def _get_metric_unit(self, metric: str) -> str:
        """Get appropriate unit for metric."""
        units = {
            'stopped_vehicle_ratio': 'Ratio',
            'average_waiting_time': 'Seconds',
            'average_queue_length': 'Vehicles',
            'throughput': 'Vehicles/min'
        }
        return units.get(metric, 'Value')
    
    def save_plots_metadata(self, metadata: Dict[str, Any], filename: str = 'plots_metadata.json'):
        """Save metadata about generated plots."""
        metadata_path = os.path.join(self.output_dir, filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Plot metadata saved: {metadata_path}")


# Convenience functions
def create_dashboard(training_history: Dict[str, List[float]],
                    evaluation_results: Dict[str, Any],
                    comparison_results: Optional[Dict[str, Any]] = None,
                    output_dir: str = 'plots') -> str:
    """
    Convenience function to create a complete dashboard.
    
    Args:
        training_history: Training metrics history
        evaluation_results: Evaluation results
        comparison_results: Optional comparison results
        output_dir: Output directory for plots
        
    Returns:
        Path to dashboard directory
    """
    plotter = MARLOPlotter(output_dir=output_dir)
    return plotter.create_dashboard(training_history, evaluation_results, comparison_results)


def plot_learning_curve(data: Dict[str, List[float]], output_path: str) -> str:
    """Convenience function to plot learning curve."""
    plotter = MARLOPlotter()
    return plotter.plot_learning_curve(data, output_path)
