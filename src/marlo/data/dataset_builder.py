"""
Dataset builder for offline reinforcement learning.
Generates synthetic and semi-synthetic traffic datasets using MockTrafficEnv.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

from ..envs.mock_env import MockTrafficEnv
from ..utils.logger import get_logger
from ..utils.metrics import MetricsTracker


class DatasetBuilder:
    """
    Builds datasets for offline RL training from environment rollouts.
    """
    
    def __init__(self, 
                 env: Optional[MockTrafficEnv] = None,
                 n_agents: int = 4,
                 max_queue_length: int = 20,
                 episode_length: int = 3600,
                 decision_frequency: int = 10):
        """
        Initialize dataset builder.
        
        Args:
            env: Pre-configured environment (optional)
            n_agents: Number of agents if creating new environment
            max_queue_length: Maximum queue length per lane
            episode_length: Episode duration in seconds
            decision_frequency: Decision frequency in seconds
        """
        self.env = env if env is not None else MockTrafficEnv(
            n_agents=n_agents,
            max_queue_length=max_queue_length,
            episode_length=episode_length,
            decision_frequency=decision_frequency
        )
        
        self.n_agents = self.env.n_agents
        self.logger = get_logger("dataset_builder")
        self.metrics_tracker = MetricsTracker()
    
    def generate_synthetic_dataset(self, 
                                 episodes: int = 100,
                                 policy: str = 'random',
                                 output_path: str = 'datasets/synthetic/dataset.npz',
                                 seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate fully synthetic dataset with controlled traffic patterns.
        
        Args:
            episodes: Number of episodes to generate
            policy: Policy type ('random', 'fixed_time', 'greedy')
            output_path: Path to save dataset
            seed: Random seed for reproducibility
            
        Returns:
            Dataset statistics
        """
        self.logger.info(f"Generating synthetic dataset: {episodes} episodes with {policy} policy")
        
        if seed is not None:
            self.env.seed(seed)
        
        dataset = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': [],
            'episode_ids': [],
            'timesteps': [],
            'agent_ids': []
        }
        
        episode_stats = []
        
        for episode in range(episodes):
            episode_data = self._collect_episode(episode, policy)
            
            # Add episode data to dataset
            for key in dataset.keys():
                if key in episode_data:
                    dataset[key].extend(episode_data[key])
            
            # Track episode statistics
            episode_stats.append(episode_data['metrics'])
            
            if (episode + 1) % 20 == 0:
                self.logger.info(f"Generated {episode + 1}/{episodes} episodes")
        
        # Convert lists to numpy arrays and save
        for key, value in dataset.items():
            if key != 'agent_ids':  # Keep agent_ids as strings
                dataset[key] = np.array(value)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, **dataset)
        
        # Save metadata
        metadata = self._create_metadata(episodes, policy, episode_stats, 'synthetic')
        metadata_path = output_path.replace('.npz', '_metadata.json')
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Synthetic dataset saved to {output_path}")
        return metadata
    
    def generate_semi_synthetic_dataset(self,
                                      episodes: int = 100,
                                      traffic_patterns: List[str] = ['rush_hour', 'off_peak'],
                                      output_path: str = 'datasets/semi_synthetic/dataset.npz',
                                      seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate semi-synthetic dataset with realistic traffic patterns.
        
        Args:
            episodes: Number of episodes to generate
            traffic_patterns: List of traffic pattern types
            output_path: Path to save dataset
            seed: Random seed
            
        Returns:
            Dataset statistics
        """
        self.logger.info(f"Generating semi-synthetic dataset: {episodes} episodes")
        
        if seed is not None:
            self.env.seed(seed)
        
        dataset = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': [],
            'episode_ids': [],
            'timesteps': [],
            'agent_ids': [],
            'traffic_patterns': []
        }
        
        episode_stats = []
        
        for episode in range(episodes):
            # Select traffic pattern for this episode
            pattern = np.random.choice(traffic_patterns)
            
            episode_data = self._collect_episode(
                episode, 
                policy='adaptive', 
                traffic_pattern=pattern
            )
            
            # Add episode data to dataset
            for key in dataset.keys():
                if key == 'traffic_patterns':
                    dataset[key].extend([pattern] * len(episode_data['observations']))
                elif key in episode_data:
                    dataset[key].extend(episode_data[key])
            
            episode_stats.append(episode_data['metrics'])
            
            if (episode + 1) % 20 == 0:
                self.logger.info(f"Generated {episode + 1}/{episodes} episodes")
        
        # Convert to numpy arrays and save
        for key, value in dataset.items():
            if key not in ['agent_ids', 'traffic_patterns']:
                dataset[key] = np.array(value)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, **dataset)
        
        # Save metadata
        metadata = self._create_metadata(episodes, 'adaptive', episode_stats, 'semi_synthetic')
        metadata['traffic_patterns'] = traffic_patterns
        metadata_path = output_path.replace('.npz', '_metadata.json')
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Semi-synthetic dataset saved to {output_path}")
        return metadata
    
    def _collect_episode(self, 
                        episode_id: int, 
                        policy: str = 'random',
                        traffic_pattern: Optional[str] = None) -> Dict[str, Any]:
        """
        Collect data from one episode using specified policy.
        
        Args:
            episode_id: Episode identifier
            policy: Policy to use for action selection
            traffic_pattern: Traffic pattern type (for semi-synthetic)
            
        Returns:
            Episode data dictionary
        """
        observations_dict, _ = self.env.reset()
        
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': [],
            'episode_ids': [],
            'timesteps': [],
            'agent_ids': []
        }
        
        # Track metrics for this episode
        episode_queues = []
        episode_waiting_times = []
        total_vehicles_passed = 0
        
        timestep = 0
        done = False
        
        while not done:
            # Select actions based on policy
            actions = self._select_actions(observations_dict, policy, timestep)
            
            # Take environment step
            next_observations_dict, rewards_dict, terminated, truncated, info = self.env.step(actions)
            
            done = bool(terminated.get('__all__', False) or truncated.get('__all__', False))
            
            # Store transition data for each agent
            for agent_id in self.env.agent_names:
                if agent_id in observations_dict and agent_id in actions:
                    episode_data['observations'].append(observations_dict[agent_id])
                    episode_data['actions'].append(actions[agent_id])
                    episode_data['rewards'].append(rewards_dict.get(agent_id, 0.0))
                    episode_data['next_observations'].append(
                        next_observations_dict.get(agent_id, observations_dict[agent_id])
                    )
                    episode_data['dones'].append(int(done))
                    episode_data['episode_ids'].append(episode_id)
                    episode_data['timesteps'].append(timestep)
                    episode_data['agent_ids'].append(agent_id)
            
            # Track episode metrics
            for agent_id in self.env.agent_names:
                if agent_id in self.env.agent_states:
                    state = self.env.agent_states[agent_id]
                    episode_queues.extend(state['queue_lengths'])
                    episode_waiting_times.extend(state.get('waiting_times', []))
                    total_vehicles_passed += state.get('vehicles_passed', 0)
            
            observations_dict = next_observations_dict
            timestep += 1
        
        # Calculate episode metrics
        episode_metrics = {
            'episode_id': episode_id,
            'timesteps': timestep,
            'total_reward': sum(r for transitions in episode_data['rewards'] for r in [transitions]),
            'avg_queue_length': np.mean(episode_queues) if episode_queues else 0,
            'avg_waiting_time': np.mean(episode_waiting_times) if episode_waiting_times else 0,
            'total_vehicles_passed': total_vehicles_passed,
            'traffic_pattern': traffic_pattern
        }
        
        episode_data['metrics'] = episode_metrics
        return episode_data
    
    def _select_actions(self, 
                       observations: Dict[str, np.ndarray], 
                       policy: str, 
                       timestep: int) -> Dict[str, int]:
        """
        Select actions based on specified policy.
        
        Args:
            observations: Current observations per agent
            policy: Policy type
            timestep: Current timestep
            
        Returns:
            Actions per agent
        """
        actions = {}
        
        for agent_id in observations.keys():
            if policy == 'random':
                actions[agent_id] = np.random.randint(0, 4)
            elif policy == 'fixed_time':
                # Simple fixed-time policy (cycle through phases)
                phase = (timestep // 5) % 4  # Change phase every 5 steps
                actions[agent_id] = phase
            elif policy == 'greedy':
                # Greedy policy based on queue lengths
                obs = observations[agent_id]
                queue_lengths = obs[:4] * self.env.max_queue_length  # Denormalize
                
                # Simple heuristic: activate phase for direction with most vehicles
                ns_total = queue_lengths[0] + queue_lengths[1]  # North + South
                ew_total = queue_lengths[2] + queue_lengths[3]  # East + West
                
                if ns_total > ew_total:
                    actions[agent_id] = 0 if timestep % 10 < 5 else 1  # NS straight then left
                else:
                    actions[agent_id] = 2 if timestep % 10 < 5 else 3  # EW straight then left
            elif policy == 'adaptive':
                # More sophisticated adaptive policy
                obs = observations[agent_id]
                queue_lengths = obs[:4] * self.env.max_queue_length
                turning_props = obs[4:16].reshape(4, 3)  # [lane, direction]
                
                # Calculate weighted demand per phase
                phase_demands = []
                for phase in range(4):
                    demand = 0
                    green_lanes = self._get_green_lanes(phase)
                    for lane in green_lanes:
                        if phase % 2 == 0:  # Straight/right phases
                            demand += queue_lengths[lane] * (turning_props[lane][0] + turning_props[lane][2])
                        else:  # Left turn phases
                            demand += queue_lengths[lane] * turning_props[lane][1]
                    phase_demands.append(demand)
                
                actions[agent_id] = int(np.argmax(phase_demands))
            else:
                actions[agent_id] = 0  # Default action
        
        return actions
    
    def _get_green_lanes(self, phase: int) -> List[int]:
        """Get lanes that are green for given phase."""
        if phase == 0:  # NS straight/right
            return [0, 1]
        elif phase == 1:  # NS left
            return [0, 1]
        elif phase == 2:  # EW straight/right
            return [2, 3]
        elif phase == 3:  # EW left
            return [2, 3]
        else:
            return []
    
    def _create_metadata(self, 
                        episodes: int, 
                        policy: str, 
                        episode_stats: List[Dict], 
                        dataset_type: str) -> Dict[str, Any]:
        """Create dataset metadata."""
        
        # Aggregate statistics
        total_rewards = [ep['total_reward'] for ep in episode_stats]
        avg_queue_lengths = [ep['avg_queue_length'] for ep in episode_stats]
        avg_waiting_times = [ep['avg_waiting_time'] for ep in episode_stats]
        total_vehicles = [ep['total_vehicles_passed'] for ep in episode_stats]
        
        metadata = {
            'dataset_info': {
                'type': dataset_type,
                'creation_date': datetime.now().isoformat(),
                'episodes': episodes,
                'policy': policy,
                'n_agents': self.n_agents,
                'episode_length': self.env.episode_length,
                'decision_frequency': self.env.decision_frequency,
                'max_queue_length': self.env.max_queue_length
            },
            'statistics': {
                'total_transitions': len(total_rewards),
                'avg_episode_reward': {
                    'mean': float(np.mean(total_rewards)),
                    'std': float(np.std(total_rewards)),
                    'min': float(np.min(total_rewards)),
                    'max': float(np.max(total_rewards))
                },
                'avg_queue_length': {
                    'mean': float(np.mean(avg_queue_lengths)),
                    'std': float(np.std(avg_queue_lengths)),
                    'min': float(np.min(avg_queue_lengths)),
                    'max': float(np.max(avg_queue_lengths))
                },
                'avg_waiting_time': {
                    'mean': float(np.mean(avg_waiting_times)),
                    'std': float(np.std(avg_waiting_times)),
                    'min': float(np.min(avg_waiting_times)),
                    'max': float(np.max(avg_waiting_times))
                },
                'total_vehicles_passed': {
                    'mean': float(np.mean(total_vehicles)),
                    'std': float(np.std(total_vehicles)),
                    'min': float(np.min(total_vehicles)),
                    'max': float(np.max(total_vehicles))
                }
            }
        }
        
        return metadata
