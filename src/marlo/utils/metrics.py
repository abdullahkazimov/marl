"""
Traffic metrics for evaluation and reward calculation.
Implements all required metrics from the problem specification:
- Primary: Average Stopped Vehicle Ratio
- Supporting: Average Waiting Time, Average Queue Length, Throughput
"""

import numpy as np
from typing import List, Dict, Union, Tuple


def stopped_vehicle_ratio(queue_lengths: Union[List[float], np.ndarray], 
                         lane_capacities: Union[List[float], np.ndarray] = None) -> float:
    """
    Calculate the primary metric: Average Stopped Vehicle Ratio.
    This is the ratio of vehicles waiting at red lights to total vehicles 
    that might occupy each given lane.
    
    Args:
        queue_lengths: List or array of queue lengths per lane
        lane_capacities: Maximum capacity per lane (default: 20)
        
    Returns:
        Stopped vehicle ratio (lower is better)
    """
    queue_lengths = np.asarray(queue_lengths)
    if lane_capacities is None:
        lane_capacities = np.full_like(queue_lengths, 20.0)  # Default capacity
    else:
        lane_capacities = np.asarray(lane_capacities)
    
    # Avoid division by zero
    lane_capacities = np.maximum(lane_capacities, 1.0)
    
    # Calculate ratio per lane and return average
    ratios = queue_lengths / lane_capacities
    return float(np.mean(ratios))


def average_waiting_time(waiting_times: Union[List[float], np.ndarray]) -> float:
    """
    Calculate average waiting time in seconds.
    The average duration that vehicles spend idle at red lights.
    
    Args:
        waiting_times: List or array of waiting times for vehicles
        
    Returns:
        Average waiting time in seconds (lower is better)
    """
    waiting_times = np.asarray(waiting_times)
    if len(waiting_times) == 0:
        return 0.0
    return float(np.mean(waiting_times))


def average_queue_length(queue_lengths: Union[List[float], np.ndarray]) -> float:
    """
    Calculate average queue length per lane.
    The mean number of vehicles lined up in each lane during red phase.
    
    Args:
        queue_lengths: List or array of queue lengths per lane
        
    Returns:
        Average queue length (lower is better)
    """
    queue_lengths = np.asarray(queue_lengths)
    if len(queue_lengths) == 0:
        return 0.0
    return float(np.mean(queue_lengths))


def throughput(passed_vehicles: int, episode_duration: float = 3600.0) -> float:
    """
    Calculate throughput: vehicles per minute.
    Total number of vehicles that successfully pass through intersection.
    
    Args:
        passed_vehicles: Total number of vehicles that passed
        episode_duration: Duration of episode in seconds (default: 1 hour)
        
    Returns:
        Throughput in vehicles per minute (higher is better)
    """
    minutes = episode_duration / 60.0
    if minutes == 0:
        return 0.0
    return float(passed_vehicles / minutes)


def calculate_reward(queue_lengths: Union[List[float], np.ndarray], 
                    alpha: float = 1.0, 
                    lane_capacities: Union[List[float], np.ndarray] = None) -> float:
    """
    Calculate the reward function as specified in the problem:
    R = -alpha * stopped_vehicle_ratio
    
    Args:
        queue_lengths: Current queue lengths per lane
        alpha: Weight parameter for stopped vehicle ratio
        lane_capacities: Maximum capacity per lane
        
    Returns:
        Reward value (higher is better)
    """
    svr = stopped_vehicle_ratio(queue_lengths, lane_capacities)
    return -alpha * svr


class MetricsTracker:
    """
    Tracks and aggregates metrics over multiple episodes.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.stopped_ratios = []
        self.waiting_times = []
        self.queue_lengths = []
        self.throughputs = []
        self.rewards = []
    
    def add_episode_metrics(self, 
                           queue_lengths: List[float],
                           waiting_times: List[float],
                           passed_vehicles: int,
                           episode_duration: float = 3600.0,
                           lane_capacities: List[float] = None):
        """
        Add metrics from a single episode.
        
        Args:
            queue_lengths: Queue lengths throughout the episode
            waiting_times: Waiting times for vehicles
            passed_vehicles: Number of vehicles that passed
            episode_duration: Duration of episode in seconds
            lane_capacities: Lane capacities
        """
        # Aggregate episode metrics
        avg_queue = average_queue_length(queue_lengths)
        avg_wait = average_waiting_time(waiting_times)
        stopped_ratio = stopped_vehicle_ratio(queue_lengths, lane_capacities)
        throughput_val = throughput(passed_vehicles, episode_duration)
        reward = calculate_reward(queue_lengths, lane_capacities=lane_capacities)
        
        # Store metrics
        self.stopped_ratios.append(stopped_ratio)
        self.waiting_times.append(avg_wait)
        self.queue_lengths.append(avg_queue)
        self.throughputs.append(throughput_val)
        self.rewards.append(reward)
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics across all episodes.
        
        Returns:
            Dictionary with mean and std for each metric
        """
        summary = {}
        
        metrics = {
            'stopped_vehicle_ratio': self.stopped_ratios,
            'average_waiting_time': self.waiting_times,
            'average_queue_length': self.queue_lengths,
            'throughput': self.throughputs,
            'reward': self.rewards
        }
        
        for name, values in metrics.items():
            if values:
                summary[f'{name}_mean'] = float(np.mean(values))
                summary[f'{name}_std'] = float(np.std(values))
                summary[f'{name}_min'] = float(np.min(values))
                summary[f'{name}_max'] = float(np.max(values))
            else:
                summary[f'{name}_mean'] = 0.0
                summary[f'{name}_std'] = 0.0
                summary[f'{name}_min'] = 0.0
                summary[f'{name}_max'] = 0.0
        
        return summary
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get metrics from the latest episode."""
        if not self.stopped_ratios:
            return {}
        
        return {
            'stopped_vehicle_ratio': self.stopped_ratios[-1],
            'average_waiting_time': self.waiting_times[-1],
            'average_queue_length': self.queue_lengths[-1],
            'throughput': self.throughputs[-1],
            'reward': self.rewards[-1]
        }
