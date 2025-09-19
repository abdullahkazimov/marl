"""
Deep Q-Network (DQN) Agent for traffic signal control.
Implements DQN with experience replay and target network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple, Optional

from .base_agent import BaseAgent


class QNetwork(nn.Module):
    """Deep Q-Network for value function approximation."""
    
    def __init__(self, 
                 state_dim: int = 25, 
                 action_dim: int = 4, 
                 hidden_dim: int = 128,
                 dropout_rate: float = 0.0):
        """
        Initialize Q-Network.
        
        Args:
            state_dim: State space dimensionality
            action_dim: Action space dimensionality
            hidden_dim: Hidden layer size
            dropout_rate: Dropout rate for regularization
        """
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 50000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, 
             state: np.ndarray, 
             action: int, 
             reward: float, 
             next_state: np.ndarray, 
             done: bool):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.tensor(done, dtype=torch.bool)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    DQN Agent for traffic signal control.
    Supports both online and offline learning.
    """
    
    def __init__(self,
                 agent_id: str,
                 state_dim: int = 25,
                 action_dim: int = 4,
                 hidden_dim: int = 128,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: int = 10000,
                 buffer_size: int = 50000,
                 target_update_freq: int = 1000,
                 device: str = 'cuda'):
        """
        Initialize DQN Agent.
        
        Args:
            agent_id: Unique identifier
            state_dim: State space dimensionality
            action_dim: Action space dimensionality
            hidden_dim: Hidden layer size
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay steps
            buffer_size: Replay buffer size
            target_update_freq: Target network update frequency
            device: Torch device
        """
        super().__init__(agent_id, state_dim, action_dim, device)
        
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network
        self.update_target_network()
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training state
        self.training_mode = True
        self.loss_history = []

    def _maybe_adapt_input_dim(self, input_states: torch.Tensor):
        """
        If the incoming state dimension does not match the configured one,
        adapt the networks to the observed input dimension to avoid matmul errors.
        """
        observed_dim = input_states.shape[-1]
        if observed_dim != self.state_dim:
            # Rebuild networks with new input dimension
            self.state_dim = int(observed_dim)
            old_state = self.q_network.state_dict()
            self.q_network = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
            self.target_network = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
            self.update_target_network()
            # Reinitialize optimizer
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
    def get_epsilon(self) -> float:
        """Get current epsilon value for exploration."""
        return max(
            self.epsilon_end,
            self.epsilon_start - (self.training_step / self.epsilon_decay) * 
            (self.epsilon_start - self.epsilon_end)
        )
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            deterministic: If True, always select greedy action
            
        Returns:
            Selected action
        """
        if not deterministic and self.training_mode and random.random() < self.get_epsilon():
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self._maybe_adapt_input_dim(state_tensor)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, 
                state: np.ndarray, 
                action: int, 
                reward: float, 
                next_state: np.ndarray, 
                done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def learn(self, batch: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Learn from experience batch or replay buffer.
        
        Args:
            batch: Optional pre-formatted batch for offline learning
            
        Returns:
            Training metrics
        """
        if batch is not None:
            # Offline learning from provided batch
            return self._learn_from_batch(batch)
        else:
            # Online learning from replay buffer
            if len(self.replay_buffer) < 1000:  # Minimum buffer size
                return {}
            
            batch_size = min(64, len(self.replay_buffer))
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
            
            batch_dict = {
                'states': states.to(self.device),
                'actions': actions.to(self.device),
                'rewards': rewards.to(self.device),
                'next_states': next_states.to(self.device),
                'dones': dones.to(self.device)
            }
            
            return self._learn_from_batch(batch_dict)
    
    def _learn_from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Learn from a formatted batch of experiences."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Ensure correct shapes: states [B, S], actions [B], rewards [B], next_states [B, S], dones [B]
        if actions.dim() > 1:
            actions = actions.view(-1)
        if rewards.dim() > 1:
            rewards = rewards.view(-1)
        if dones.dim() > 1:
            dones = dones.view(-1)

        # Adapt to observed input dimension if needed
        self._maybe_adapt_input_dim(states)

        # Current Q values
        q_all = self.q_network(states)  # [B, A]
        current_q_values = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]
        
        # Target Q values
        with torch.no_grad():
            # Double DQN target: argmax from online, value from target
            next_q_online = self.q_network(next_states)
            next_actions = torch.argmax(next_q_online, dim=1)
            next_q_target_all = self.target_network(next_states)
            next_q_values = next_q_target_all.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            not_done = (~dones).to(next_q_values.dtype)
            target_q_values = rewards + (self.gamma * next_q_values * not_done)
        
        # Compute loss (Huber/ Smooth L1)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update training statistics
        self.training_step += 1
        self.loss_history.append(loss.item())
        
        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        return {
            'loss': loss.item(),
            'epsilon': self.get_epsilon(),
            'q_values_mean': current_q_values.mean().item()
        }
    
    def update_target_network(self):
        """Copy parameters from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def set_train_mode(self):
        """Set agent to training mode."""
        self.training_mode = True
        self.q_network.train()
        self.target_network.train()
    
    def set_eval_mode(self):
        """Set agent to evaluation mode."""
        self.training_mode = False
        self.q_network.eval()
        self.target_network.eval()
    
    def save(self, filepath: str):
        """Save agent checkpoint."""
        checkpoint = {
            'agent_id': self.agent_id,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'loss_history': self.loss_history,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay
            }
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.loss_history = checkpoint.get('loss_history', [])
    
    def get_training_stats(self) -> Dict[str, any]:
        """Get detailed training statistics."""
        stats = super().get_training_stats()
        stats.update({
            'epsilon': self.get_epsilon(),
            'buffer_size': len(self.replay_buffer),
            'recent_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            'total_loss_history': len(self.loss_history)
        })
        return stats
