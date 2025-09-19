"""
Offline trainer for multi-agent traffic signal control.
Implements CTDE (Centralized Training with Decentralized Execution) approach.
"""

import os
import yaml
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import json
from collections import defaultdict

from ..agents.dqn_agent import DQNAgent
from ..agents.central_critic import CentralCritic
from ..data.loader import MultiAgentDataLoader, create_multi_agent_loader
from ..utils.logger import create_experiment_logger
from ..utils.seed import set_seed
from ..utils.metrics import MetricsTracker


class OfflineTrainer:
    """
    Offline trainer implementing CTDE for multi-agent traffic signal control.
    """
    
    def __init__(self, config_path: str, device_override: Optional[str] = None):
        """
        Initialize offline trainer.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up experiment
        self.experiment_name = self.config.get('experiment', {}).get('name', 'default_experiment')
        self.experiment_dir = os.path.join('experiments', self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Set up logging
        self.logger = create_experiment_logger(self.experiment_name)
        self.logger.info(f"Initializing offline trainer for experiment: {self.experiment_name}")
        
        # Set random seed
        seed = self.config.get('seed', 42)
        set_seed(seed)
        self.logger.info(f"Set random seed: {seed}")
        
        # Initialize device (safe CUDA probe and override)
        self.device = self._select_device(device_override)
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize central critic (if using CTDE)
        self.central_critic = self._initialize_central_critic()
        
        # Initialize data loader
        self.data_loader = self._initialize_data_loader()
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.training_history = defaultdict(list)
        
        self.logger.info("Offline trainer initialized successfully")

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
    
    def _initialize_agents(self) -> Dict[str, DQNAgent]:
        """Initialize DQN agents for each intersection."""
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
                epsilon_start=agent_config.get('epsilon_start', 1.0),
                epsilon_end=agent_config.get('epsilon_end', 0.05),
                epsilon_decay=agent_config.get('epsilon_decay', 10000),
                buffer_size=agent_config.get('buffer_size', 50000),
                target_update_freq=agent_config.get('target_update_freq', 1000),
                device=str(self.device)
            )
            agents[agent_id] = agent
            self.logger.info(f"Initialized agent: {agent_id}")
        
        return agents
    
    def _initialize_central_critic(self) -> Optional[CentralCritic]:
        """Initialize central critic for CTDE training."""
        if not self.config.get('ctde', {}).get('central_critic', False):
            self.logger.info("Central critic disabled, using independent learning")
            return None
        
        ctde_config = self.config.get('ctde', {})
        central_critic = CentralCritic(
            n_agents=self.config['env']['n_agents'],
            local_state_dim=self.config['env']['state_dim'],
            hidden_dim=ctde_config.get('critic_hidden_dim', 256),
            learning_rate=ctde_config.get('critic_lr', 5e-4),
            gamma=self.config.get('agent', {}).get('gamma', 0.99),
            device=str(self.device)
        )
        
        self.logger.info("Initialized central critic for CTDE training")
        return central_critic
    
    def _initialize_data_loader(self) -> MultiAgentDataLoader:
        """Initialize data loader for offline training."""
        dataset_config = self.config.get('dataset', {})
        training_config = self.config['training']
        
        # Get dataset paths
        synthetic_path = dataset_config.get('synthetic_path', 'datasets/synthetic')
        semi_synthetic_path = dataset_config.get('semi_synthetic_path', 'datasets/semi_synthetic')
        
        dataset_paths = []
        
        # Look for dataset files
        for base_path in [synthetic_path, semi_synthetic_path]:
            if os.path.exists(base_path):
                for file in os.listdir(base_path):
                    if file.endswith('.npz'):
                        dataset_paths.append(os.path.join(base_path, file))
        
        if not dataset_paths:
            raise FileNotFoundError("No dataset files found. Please generate datasets first.")
        
        self.logger.info(f"Found {len(dataset_paths)} dataset files: {dataset_paths}")
        
        # Create data loader
        data_loader = create_multi_agent_loader(
            dataset_paths=dataset_paths,
            batch_size=training_config['batch_size'],
            shuffle=True,
            normalize_rewards=True,
            normalize_observations=True,
            clip_rewards=True,
            reward_clip_value=self.config.get('reward', {}).get('clip_reward', 1.0)
        )
        
        # Log dataset statistics
        stats = data_loader.get_dataset_stats()
        self.logger.info(f"Total transitions: {stats['total_transitions']}")
        
        return data_loader
    
    def train(self) -> Dict[str, Any]:
        """
        Run offline training loop.
        
        Returns:
            Training results and metrics
        """
        self.logger.info("Starting offline training")
        
        training_config = self.config['training']
        num_epochs = training_config.get('episodes', 1000)  # Using episodes as epochs
        save_frequency = self.config.get('logging', {}).get('save_frequency', 100)
        eval_frequency = self.config.get('evaluation', {}).get('frequency', 50)
        
        # Set agents to training mode
        for agent in self.agents.values():
            agent.set_train_mode()
        
        best_performance = float('-inf')
        
        try:
            for epoch in range(num_epochs):
                self.epoch = epoch
                
                # Train one epoch
                epoch_metrics = self._train_epoch()
                
                # Log metrics
                self._log_epoch_metrics(epoch, epoch_metrics)
                
                # Evaluate periodically
                if (epoch + 1) % eval_frequency == 0:
                    eval_metrics = self._evaluate()
                    self.logger.info(f"Epoch {epoch + 1} - Evaluation metrics: {eval_metrics}")
                    
                    # Save best model
                    current_performance = eval_metrics.get('reward_mean', float('-inf'))
                    if current_performance > best_performance:
                        best_performance = current_performance
                        self._save_checkpoint('best_model.pt')
                        self.logger.info(f"New best model saved (performance: {best_performance:.3f})")
                
                # Save checkpoint periodically
                if (epoch + 1) % save_frequency == 0:
                    self._save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
                    self.logger.info(f"Checkpoint saved at epoch {epoch + 1}")
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            raise
        
        # Save final model
        self._save_checkpoint('final_model.pt')
        
        # Get final training results
        results = self._get_training_results()
        self.logger.info("Training completed successfully")
        
        return results
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = defaultdict(list)
        
        progress_bar = tqdm(self.data_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Train agents on batch
            agent_metrics = self._train_batch(batch)
            
            # Aggregate metrics
            for key, value in agent_metrics.items():
                epoch_metrics[key].append(value)
            
            self.global_step += 1
            
            # Update progress bar
            if batch_idx % 10 == 0:
                avg_loss = np.mean(epoch_metrics['loss']) if epoch_metrics['loss'] else 0
                progress_bar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
        
        # Average metrics over epoch
        averaged_metrics = {}
        for key, values in epoch_metrics.items():
            averaged_metrics[key] = np.mean(values)
        
        return averaged_metrics
    
    def _train_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train agents on a single batch."""
        batch_metrics = defaultdict(list)
        
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        
        # Group batch by agents
        agent_batches = self._group_batch_by_agent(batch)
        
        # Train individual agents
        for agent_id, agent_batch in agent_batches.items():
            if agent_id in self.agents:
                # Normalize keys and ensure shapes for DQN
                if 'states' in agent_batch and agent_batch['states'].dim() == 3:
                    # If sequences were provided, flatten batch and time
                    B, T, S = agent_batch['states'].shape
                    agent_batch['states'] = agent_batch['states'].view(B*T, S)
                    agent_batch['next_states'] = agent_batch['next_states'].view(B*T, S)
                    agent_batch['actions'] = agent_batch['actions'].view(B*T)
                    agent_batch['rewards'] = agent_batch['rewards'].view(B*T)
                    agent_batch['dones'] = agent_batch['dones'].view(B*T)

                metrics = self.agents[agent_id].learn(agent_batch)
                for key, value in metrics.items():
                    batch_metrics[f'{agent_id}_{key}'].append(value)
        
        # Train central critic if available
        if self.central_critic is not None:
            critic_metrics = self._train_central_critic(batch)
            for key, value in critic_metrics.items():
                batch_metrics[f'critic_{key}'].append(value)
        
        # Average metrics
        averaged_metrics = {}
        for key, values in batch_metrics.items():
            averaged_metrics[key] = np.mean(values)
        
        # Compute overall loss
        agent_losses = [v for k, v in averaged_metrics.items() if k.endswith('_loss')]
        averaged_metrics['loss'] = np.mean(agent_losses) if agent_losses else 0.0
        
        return averaged_metrics
    
    def _group_batch_by_agent(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Group batch data by agent."""
        agent_batches = {}
        
        # Get agent IDs from batch
        agent_ids = batch.get('agent_ids', [])
        if not agent_ids:
            return agent_batches
        
        # Group indices by agent
        agent_indices = defaultdict(list)
        for idx, agent_id in enumerate(agent_ids):
            agent_indices[agent_id].append(idx)
        
        # Create agent-specific batches
        for agent_id, indices in agent_indices.items():
            agent_batch = {}
            indices_tensor = torch.LongTensor(indices).to(self.device)
            
            for key, value in batch.items():
                if key not in ['agent_ids', 'episode_ids'] and isinstance(value, torch.Tensor):
                    if len(value.shape) > 1:
                        agent_batch[key] = torch.index_select(value, 0, indices_tensor)
                    else:
                        agent_batch[key] = value[indices_tensor]
            
            # Rename keys for agent learning
            if 'observations' in agent_batch:
                agent_batch['states'] = agent_batch.pop('observations')
            if 'next_observations' in agent_batch:
                agent_batch['next_states'] = agent_batch.pop('next_observations')
            
            agent_batches[agent_id] = agent_batch
        
        return agent_batches
    
    def _train_central_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train central critic using global information."""
        # Create global states from individual agent states by grouping per (episode_id, timestep)
        # and concatenating observations in a consistent agent order.

        observations = batch['observations']  # [B, state_dim]
        next_observations = batch['next_observations']  # [B, state_dim]
        rewards_tensor = batch['rewards']  # [B] or [B, 1]
        dones_tensor = batch['dones']  # [B]

        agent_ids_list = batch.get('agent_ids', [])  # List[str] length B
        episode_ids_list = batch.get('episode_ids', [])  # List[int] length B
        timesteps_list = batch.get('timesteps', [])  # List[int] length B

        # If required metadata is missing, skip
        if not agent_ids_list or not episode_ids_list or not timesteps_list:
            return {}

        n_agents = self.config['env']['n_agents']
        state_dim = self.config['env']['state_dim']
        expected_global_dim = n_agents * state_dim

        # Map agent string id to index (expects format 'agent_i')
        def agent_to_index(agent_str: str) -> int:
            try:
                return int(str(agent_str).split('_')[-1])
            except Exception:
                return -1

        # Group per (episode, timestep)
        groups = {}
        B = observations.shape[0]
        for i in range(B):
            ep = int(episode_ids_list[i])
            ts = int(timesteps_list[i])
            aid = agent_to_index(agent_ids_list[i])
            if aid < 0 or aid >= n_agents:
                continue
            key = (ep, ts)
            if key not in groups:
                groups[key] = {
                    'obs': [None] * n_agents,
                    'next_obs': [None] * n_agents,
                    'rewards': [None] * n_agents,
                    'dones': []
                }
            groups[key]['obs'][aid] = observations[i]
            groups[key]['next_obs'][aid] = next_observations[i]
            # rewards_tensor may be shape [B] or [B, 1]
            r = rewards_tensor[i]
            if r.dim() > 0:
                r = r.squeeze()
            groups[key]['rewards'][aid] = r
            groups[key]['dones'].append(dones_tensor[i])

        # Build global batches from complete groups only (all agents present)
        global_states_list = []
        next_global_states_list = []
        rewards_list = []
        dones_list = []

        for key, data in groups.items():
            if any(x is None for x in data['obs']) or any(x is None for x in data['next_obs']) or any(x is None for x in data['rewards']):
                continue
            # Concatenate in agent index order
            global_state = torch.cat(data['obs'], dim=-1)  # [n_agents*state_dim]
            next_global_state = torch.cat(data['next_obs'], dim=-1)
            # Stack rewards -> [n_agents]
            rewards_vec = torch.stack([torch.as_tensor(rv, device=self.device, dtype=next_global_state.dtype) for rv in data['rewards']], dim=0)
            # Done for the joint transition (any agent done)
            done_flag = torch.stack([d.to(self.device) if isinstance(d, torch.Tensor) else torch.tensor(bool(d), device=self.device) for d in data['dones']]).any()

            # Ensure correct dims
            if global_state.shape[-1] != expected_global_dim or next_global_state.shape[-1] != expected_global_dim:
                continue

            global_states_list.append(global_state)
            next_global_states_list.append(next_global_state)
            rewards_list.append(rewards_vec)
            dones_list.append(done_flag)

        if not global_states_list:
            # Nothing to train critic on in this batch
            return {}

        global_states = torch.stack(global_states_list, dim=0).to(self.device)  # [G, n_agents*state_dim]
        next_global_states = torch.stack(next_global_states_list, dim=0).to(self.device)
        rewards = torch.stack(rewards_list, dim=0).to(self.device)  # [G, n_agents]
        dones = torch.stack(dones_list, dim=0).to(self.device)  # [G]

        critic_metrics = self.central_critic.train_step(
            current_states=global_states,
            rewards=rewards,
            next_states=next_global_states,
            dones=dones
        )

        return critic_metrics
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate current policy on a subset of data."""
        # Set agents to evaluation mode
        for agent in self.agents.values():
            agent.set_eval_mode()
        
        eval_metrics = defaultdict(list)
        eval_batches = 0
        max_eval_batches = 50  # Limit evaluation to avoid long delays
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                if batch_idx >= max_eval_batches:
                    break
                
                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # Evaluate agents
                agent_batches = self._group_batch_by_agent(batch)
                
                for agent_id, agent_batch in agent_batches.items():
                    if agent_id in self.agents:
                        # Get Q-values and compute metrics
                        states = agent_batch['states']
                        actions = agent_batch['actions']
                        rewards = agent_batch['rewards']
                        
                        q_values = self.agents[agent_id].q_network(states)
                        predicted_actions = torch.argmax(q_values, dim=1)
                        
                        # Compute accuracy
                        accuracy = (predicted_actions == actions.squeeze()).float().mean().item()
                        eval_metrics[f'{agent_id}_accuracy'].append(accuracy)
                        eval_metrics[f'{agent_id}_reward'].append(rewards.mean().item())
                        eval_metrics[f'{agent_id}_q_value'].append(q_values.mean().item())
                
                eval_batches += 1
        
        # Set agents back to training mode
        for agent in self.agents.values():
            agent.set_train_mode()
        
        # Average metrics
        averaged_metrics = {}
        for key, values in eval_metrics.items():
            averaged_metrics[key] = np.mean(values)
        
        # Compute overall metrics
        agent_rewards = [v for k, v in averaged_metrics.items() if k.endswith('_reward')]
        agent_accuracies = [v for k, v in averaged_metrics.items() if k.endswith('_accuracy')]
        
        averaged_metrics['reward_mean'] = np.mean(agent_rewards) if agent_rewards else 0.0
        averaged_metrics['accuracy_mean'] = np.mean(agent_accuracies) if agent_accuracies else 0.0
        
        return averaged_metrics
    
    def _log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an epoch."""
        # Store in training history
        for key, value in metrics.items():
            self.training_history[key].append(value)
        
        # Log to file
        log_entry = {'epoch': epoch, **metrics}
        log_file = os.path.join(self.experiment_dir, 'logs', 'training_log.jsonl')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Log key metrics
        if epoch % 10 == 0:
            loss = metrics.get('loss', 0)
            self.logger.info(f"Epoch {epoch + 1}: Loss = {loss:.4f}")
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        # Save individual agent checkpoints
        agent_checkpoints = {}
        for agent_id, agent in self.agents.items():
            agent_path = checkpoint_path.replace('.pt', f'_{agent_id}.pt')
            agent.save(agent_path)
            agent_checkpoints[agent_id] = agent_path
        
        # Save central critic if available
        critic_checkpoint = None
        if self.central_critic is not None:
            critic_path = checkpoint_path.replace('.pt', '_critic.pt')
            self.central_critic.save(critic_path)
            critic_checkpoint = critic_path
        
        # Save training state
        training_state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'training_history': dict(self.training_history),
            'config': self.config,
            'agent_checkpoints': agent_checkpoints,
            'critic_checkpoint': critic_checkpoint
        }
        
        torch.save(training_state, checkpoint_path)
        self.logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def _get_training_results(self) -> Dict[str, Any]:
        """Get final training results."""
        return {
            'experiment_name': self.experiment_name,
            'total_epochs': self.epoch + 1,
            'total_steps': self.global_step,
            'training_history': dict(self.training_history),
            'final_metrics': {
                key: values[-1] if values else 0 
                for key, values in self.training_history.items()
            },
            'config': self.config
        }
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.training_history = defaultdict(list, checkpoint['training_history'])
        
        # Load agent checkpoints
        agent_checkpoints = checkpoint.get('agent_checkpoints', {})
        for agent_id, agent_path in agent_checkpoints.items():
            if agent_id in self.agents and os.path.exists(agent_path):
                self.agents[agent_id].load(agent_path)
                self.logger.info(f"Loaded agent checkpoint: {agent_id}")
        
        # Load central critic checkpoint
        critic_checkpoint = checkpoint.get('critic_checkpoint')
        if critic_checkpoint and self.central_critic and os.path.exists(critic_checkpoint):
            self.central_critic.load(critic_checkpoint)
            self.logger.info("Loaded central critic checkpoint")
        
        self.logger.info("Checkpoint loaded successfully")
