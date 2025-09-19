"""
Evaluation runner for trained traffic signal control agents.
Computes all required metrics: stopped ratio, wait time, queue length, throughput.
"""

import torch
import numpy as np
import random
import yaml
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from ..envs.mock_env import MockTrafficEnv
from ..agents.dqn_agent import DQNAgent
from ..utils.metrics import MetricsTracker, stopped_vehicle_ratio, average_waiting_time, average_queue_length, throughput
from ..utils.logger import get_logger
from ..utils.seed import set_seed


class EvaluationRunner:
    """
    Evaluation runner for multi-agent traffic signal control.
    """
    
    def __init__(self, config_path: str, checkpoint_path: str = None, device_override: Optional[str] = None):
        """
        Initialize evaluation runner.
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint (optional)
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = get_logger("evaluation")
        self.checkpoint_path = checkpoint_path
        
        # Set random seed for reproducible evaluation
        seed = self.config.get('seed', 42)
        set_seed(seed)
        
        # Initialize device with override
        self.device = self._select_device(device_override)
        
        # Initialize environment
        self.env = self._initialize_environment()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Load trained models if checkpoint provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        self.logger.info("Evaluation runner initialized")
    
    def _select_device(self, device_override: Optional[str]) -> torch.device:
        """Select compute device with optional override and safe fallback."""
        # Explicit CPU
        if device_override == 'cpu':
            return torch.device('cpu')
        # Explicit CUDA
        if device_override == 'cuda':
            if torch.cuda.is_available():
                try:
                    _ = torch.zeros(1).to('cuda')
                    return torch.device('cuda')
                except Exception:
                    return torch.device('cpu')
            return torch.device('cpu')
        # Auto
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1).to('cuda')
                return torch.device('cuda')
            except Exception:
                return torch.device('cpu')
        return torch.device('cpu')
    
    def _initialize_environment(self) -> MockTrafficEnv:
        """Initialize evaluation environment."""
        env_config = self.config['env']
        
        env = MockTrafficEnv(
            n_agents=env_config['n_agents'],
            max_queue_length=env_config['max_queue_length'],
            episode_length=env_config['episode_length'],
            decision_frequency=env_config['decision_frequency']
        )
        
        return env
    
    def _initialize_agents(self) -> Dict[str, DQNAgent]:
        """Initialize agents for evaluation."""
        agents = {}
        agent_config = self.config.get('agent', {})
        n_agents = self.config['env']['n_agents']
        
        for i in range(n_agents):
            agent_id = f"agent_{i}"
            agent = DQNAgent(
                agent_id=agent_id,
                state_dim=self.config['env']['state_dim'],
                action_dim=self.config['env']['action_dim'],
                hidden_dim=agent_config.get('hidden_dim', 128),
                learning_rate=agent_config.get('learning_rate', 0.001),
                gamma=agent_config.get('gamma', 0.99),
                device=str(self.device)
            )
            # Set to evaluation mode
            agent.set_eval_mode()
            agents[agent_id] = agent
        
        return agents
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load trained model checkpoint."""
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            # Load main checkpoint to get agent checkpoint paths
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            except Exception:
                # Retry with weights_only disabled explicitly for compatibility
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            agent_checkpoints = checkpoint.get('agent_checkpoints', {})
            
            # Load individual agent checkpoints
            for agent_id, agent_path in agent_checkpoints.items():
                if agent_id in self.agents:
                    self.agents[agent_id].load(agent_path)
                    self.logger.info(f"Loaded trained weights for {agent_id}")
        
        except Exception as e:
            self.logger.warning(f"Could not load checkpoint: {e}")
            self.logger.info("Using randomly initialized agents")
    
    def evaluate(self, 
                num_episodes: int = 100,
                deterministic: bool = True,
                render: bool = False) -> Dict[str, Any]:
        """
        Run evaluation episodes and compute metrics.
        
        Args:
            num_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy
            render: Whether to render episodes
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info(f"Starting evaluation: {num_episodes} episodes")
        
        # Reset metrics tracker
        self.metrics_tracker.reset()
        
        episode_results = []
        all_rewards = []
        
        for episode in range(num_episodes):
            episode_result = self._run_episode(episode, deterministic, render)
            episode_results.append(episode_result)
            all_rewards.append(episode_result['total_reward'])
            
            # Add to metrics tracker
            self.metrics_tracker.add_episode_metrics(
                queue_lengths=episode_result['queue_lengths'],
                waiting_times=episode_result['waiting_times'],
                passed_vehicles=episode_result['vehicles_passed'],
                episode_duration=self.config['env']['episode_length']
            )
            
            if (episode + 1) % 20 == 0:
                self.logger.info(f"Completed {episode + 1}/{num_episodes} episodes")
        
        # Compute final metrics
        results = self._compute_evaluation_results(episode_results)
        
        self.logger.info("Evaluation completed")
        self.logger.info(f"Results: {results['summary']}")
        
        return results
    
    def _run_episode(self, 
                    episode_id: int, 
                    deterministic: bool = True,
                    render: bool = False) -> Dict[str, Any]:
        """
        Run a single evaluation episode.
        
        Args:
            episode_id: Episode identifier
            deterministic: Whether to use deterministic actions
            render: Whether to render the episode
            
        Returns:
            Episode results
        """
        observations, _ = self.env.reset()
        
        episode_data = {
            'episode_id': episode_id,
            'rewards': [],
            'actions': [],
            'queue_lengths': [],
            'waiting_times': [],
            'vehicles_passed': 0,
            'timesteps': 0
        }
        
        done = False
        timestep = 0
        
        while not done:
            # Select actions
            actions = {}
            for agent_id, obs in observations.items():
                if agent_id in self.agents:
                    action = self.agents[agent_id].act(obs, deterministic=deterministic)
                    actions[agent_id] = action
                else:
                    actions[agent_id] = random.randrange(4)  # Random fallback without numpy
            
            # Take environment step
            next_observations, rewards, terminated, truncated, info = self.env.step(actions)
            
            done = bool(terminated.get('__all__', False) or truncated.get('__all__', False))
            
            # Collect episode data
            episode_data['actions'].append(actions.copy())
            episode_data['rewards'].append(rewards.copy())
            
            # Collect traffic metrics
            for agent_id in self.env.agent_names:
                if agent_id in self.env.agent_states:
                    state = self.env.agent_states[agent_id]
                    episode_data['queue_lengths'].extend(state['queue_lengths'].tolist())
                    episode_data['waiting_times'].extend(state.get('waiting_times', []))
                    episode_data['vehicles_passed'] += state.get('vehicles_passed', 0)
            
            observations = next_observations
            timestep += 1
            
            if render and timestep % 10 == 0:
                self.env.render()
        
        episode_data['timesteps'] = timestep
        episode_data['total_reward'] = sum(
            sum(reward_dict.values()) for reward_dict in episode_data['rewards']
        )
        
        return episode_data
    
    def _compute_evaluation_results(self, episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute comprehensive evaluation results."""
        
        # Extract metrics from all episodes
        all_queue_lengths = []
        all_waiting_times = []
        all_rewards = []
        all_vehicles_passed = []
        
        for result in episode_results:
            all_queue_lengths.extend(result['queue_lengths'])
            all_waiting_times.extend(result['waiting_times'])
            all_rewards.append(result['total_reward'])
            all_vehicles_passed.append(result['vehicles_passed'])
        
        # Compute primary and supporting metrics
        metrics = {
            'stopped_vehicle_ratio': stopped_vehicle_ratio(
                all_queue_lengths, 
                [self.config['env']['max_queue_length']] * len(all_queue_lengths)
            ),
            'average_waiting_time': average_waiting_time(all_waiting_times),
            'average_queue_length': average_queue_length(all_queue_lengths),
            'throughput': throughput(
                int(np.mean(all_vehicles_passed)), 
                self.config['env']['episode_length']
            )
        }
        
        # Compute reward statistics
        reward_stats = {
            'mean': float(np.mean(all_rewards)),
            'std': float(np.std(all_rewards)),
            'min': float(np.min(all_rewards)),
            'max': float(np.max(all_rewards))
        }
        
        # Get metrics tracker summary
        metrics_summary = self.metrics_tracker.get_summary()
        
        # Action distribution analysis
        action_analysis = self._analyze_actions(episode_results)
        
        # Performance by episode
        episode_performance = []
        for i, result in enumerate(episode_results):
            episode_metrics = {
                'episode': i,
                'total_reward': result['total_reward'],
                'avg_queue_length': np.mean(result['queue_lengths']) if result['queue_lengths'] else 0,
                'avg_waiting_time': np.mean(result['waiting_times']) if result['waiting_times'] else 0,
                'vehicles_passed': result['vehicles_passed']
            }
            episode_performance.append(episode_metrics)
        
        results = {
            'summary': {
                'num_episodes': len(episode_results),
                'primary_metric': metrics['stopped_vehicle_ratio'],
                'supporting_metrics': {
                    'average_waiting_time': metrics['average_waiting_time'],
                    'average_queue_length': metrics['average_queue_length'],
                    'throughput': metrics['throughput']
                }
            },
            'detailed_metrics': metrics,
            'reward_statistics': reward_stats,
            'metrics_tracker_summary': metrics_summary,
            'action_analysis': action_analysis,
            'episode_performance': episode_performance,
            'config': self.config
        }
        
        return results
    
    def _analyze_actions(self, episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze action distributions and patterns."""
        
        # Collect all actions by agent
        agent_actions = defaultdict(list)
        
        for result in episode_results:
            for action_dict in result['actions']:
                for agent_id, action in action_dict.items():
                    agent_actions[agent_id].append(action)
        
        # Compute action distributions
        action_distributions = {}
        for agent_id, actions in agent_actions.items():
            unique, counts = np.unique(actions, return_counts=True)
            total = len(actions)
            distribution = {int(action): count / total for action, count in zip(unique, counts)}
            action_distributions[agent_id] = distribution
        
        # Overall action distribution
        all_actions = [action for actions in agent_actions.values() for action in actions]
        if all_actions:
            unique, counts = np.unique(all_actions, return_counts=True)
            total = len(all_actions)
            overall_distribution = {int(action): count / total for action, count in zip(unique, counts)}
        else:
            overall_distribution = {}
        
        # Phase transition analysis
        phase_transitions = self._analyze_phase_transitions(episode_results)
        
        return {
            'agent_distributions': action_distributions,
            'overall_distribution': overall_distribution,
            'phase_transitions': phase_transitions
        }
    
    def _analyze_phase_transitions(self, episode_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze traffic light phase transitions."""
        
        transitions = defaultdict(int)
        
        for result in episode_results:
            actions = result['actions']
            for agent_id in self.env.agent_names:
                agent_actions = [action_dict.get(agent_id, 0) for action_dict in actions]
                
                # Count transitions
                for i in range(1, len(agent_actions)):
                    from_phase = agent_actions[i-1]
                    to_phase = agent_actions[i]
                    if from_phase != to_phase:  # Only count actual transitions
                        transitions[f'{from_phase}->{to_phase}'] += 1
        
        # Convert to percentages
        total_transitions = sum(transitions.values())
        if total_transitions > 0:
            transition_percentages = {
                key: (count / total_transitions) * 100 
                for key, count in transitions.items()
            }
        else:
            transition_percentages = {}
        
        return {
            'raw_counts': dict(transitions),
            'percentages': transition_percentages,
            'total_transitions': total_transitions
        }
    
    def compare_with_baseline(self, 
                            baseline_policy: str = 'random',
                            num_episodes: int = 50) -> Dict[str, Any]:
        """
        Compare trained agents with baseline policy.
        
        Args:
            baseline_policy: Baseline policy type ('random', 'fixed_time')
            num_episodes: Number of episodes for comparison
            
        Returns:
            Comparison results
        """
        self.logger.info(f"Running baseline comparison with {baseline_policy} policy")
        
        # Evaluate baseline
        baseline_results = self._evaluate_baseline(baseline_policy, num_episodes)
        
        # Evaluate trained agents
        trained_results = self.evaluate(num_episodes, deterministic=True)
        
        # Compute improvement
        comparison = {
            'baseline_policy': baseline_policy,
            'num_episodes': num_episodes,
            'baseline_metrics': baseline_results['detailed_metrics'],
            'trained_metrics': trained_results['detailed_metrics'],
            'improvements': {},
            'baseline_eval_results': baseline_results,
            'trained_eval_results': trained_results
        }
        
        # Calculate improvements (negative values mean worse performance)
        for metric in ['stopped_vehicle_ratio', 'average_waiting_time', 'average_queue_length']:
            baseline_val = baseline_results['detailed_metrics'][metric]
            trained_val = trained_results['detailed_metrics'][metric]
            
            if baseline_val != 0:
                improvement = ((baseline_val - trained_val) / baseline_val) * 100
            else:
                improvement = 0
            
            comparison['improvements'][metric] = improvement
        
        # Throughput improvement (higher is better)
        baseline_throughput = baseline_results['detailed_metrics']['throughput']
        trained_throughput = trained_results['detailed_metrics']['throughput']
        
        if baseline_throughput != 0:
            throughput_improvement = ((trained_throughput - baseline_throughput) / baseline_throughput) * 100
        else:
            throughput_improvement = 0
        
        comparison['improvements']['throughput'] = throughput_improvement
        
        self.logger.info(f"Comparison completed. Improvements: {comparison['improvements']}")
        
        return comparison
    
    def _evaluate_baseline(self, policy: str, num_episodes: int) -> Dict[str, Any]:
        """Evaluate baseline policy."""
        
        # Temporarily use random policy
        episode_results = []
        
        for episode in range(num_episodes):
            observations, _ = self.env.reset()
            
            episode_data = {
                'episode_id': episode,
                'queue_lengths': [],
                'waiting_times': [],
                'vehicles_passed': 0,
                'total_reward': 0,
                'actions': []
            }
            
            done = False
            timestep = 0
            
            while not done:
                # Select baseline actions
                actions = {}
                for agent_id in observations.keys():
                    if policy == 'random':
                        actions[agent_id] = random.randrange(4)
                    elif policy == 'fixed_time':
                        phase = (timestep // 5) % 4
                        actions[agent_id] = phase
                    else:
                        actions[agent_id] = 0
                
                next_observations, rewards, terminated, truncated, info = self.env.step(actions)
                done = bool(terminated.get('__all__', False) or truncated.get('__all__', False))
                
                # Collect metrics
                episode_data['actions'].append(actions.copy())
                episode_data['total_reward'] += sum(rewards.values())
                
                for agent_id in self.env.agent_names:
                    if agent_id in self.env.agent_states:
                        state = self.env.agent_states[agent_id]
                        episode_data['queue_lengths'].extend(state['queue_lengths'].tolist())
                        episode_data['waiting_times'].extend(state.get('waiting_times', []))
                        episode_data['vehicles_passed'] += state.get('vehicles_passed', 0)
                
                observations = next_observations
                timestep += 1
            
            episode_results.append(episode_data)
        
        # Compute results
        return self._compute_evaluation_results(episode_results)
