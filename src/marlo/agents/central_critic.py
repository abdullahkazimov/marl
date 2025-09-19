"""
Central Critic for Centralized Training with Decentralized Execution (CTDE).
Implements a central value function that has access to global state information
during training but allows decentralized execution during deployment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class CentralCriticNetwork(nn.Module):
    """
    Central critic network that observes global state information.
    Used for centralized training in CTDE paradigm.
    """
    
    def __init__(self,
                 global_state_dim: int,
                 n_agents: int = 4,
                 hidden_dim: int = 256,
                 dropout_rate: float = 0.1):
        """
        Initialize central critic network.
        
        Args:
            global_state_dim: Dimensionality of global state
            n_agents: Number of agents in the system
            hidden_dim: Hidden layer size
            dropout_rate: Dropout rate for regularization
        """
        super(CentralCriticNetwork, self).__init__()
        
        self.n_agents = n_agents
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Agent-specific value heads
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(n_agents)
        ])
        
        # Global value function (optional, for baseline)
        self.global_value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, global_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through central critic.
        
        Args:
            global_state: Global state observation
            
        Returns:
            Dictionary with value estimates per agent and global value
        """
        # Extract shared features
        shared_features = self.shared_layers(global_state)
        
        # Compute agent-specific values
        agent_values = []
        for i, value_head in enumerate(self.value_heads):
            agent_value = value_head(shared_features)
            agent_values.append(agent_value)
        
        # Compute global value
        global_value = self.global_value_head(shared_features)
        
        return {
            'agent_values': torch.stack(agent_values, dim=1).squeeze(-1),  # [batch_size, n_agents]
            'global_value': global_value.squeeze(-1)  # [batch_size]
        }


class CentralCritic:
    """
    Central Critic implementation for CTDE training.
    Coordinates value function learning across multiple agents.
    """
    
    def __init__(self,
                 n_agents: int = 4,
                 local_state_dim: int = 25,
                 hidden_dim: int = 256,
                 learning_rate: float = 0.0005,
                 gamma: float = 0.99,
                 target_update_freq: int = 1000,
                 device: str = 'cuda'):
        """
        Initialize central critic.
        
        Args:
            n_agents: Number of agents
            local_state_dim: Dimensionality of each agent's local state
            hidden_dim: Hidden layer size
            learning_rate: Learning rate for critic
            gamma: Discount factor
            target_update_freq: Target network update frequency
            device: Torch device
        """
        self.n_agents = n_agents
        self.local_state_dim = local_state_dim
        self.global_state_dim = n_agents * local_state_dim  # Concatenated states
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        
        # Networks
        self.critic_network = CentralCriticNetwork(
            self.global_state_dim, n_agents, hidden_dim
        ).to(self.device)
        
        self.target_critic_network = CentralCriticNetwork(
            self.global_state_dim, n_agents, hidden_dim
        ).to(self.device)
        
        # Initialize target network
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.critic_network.parameters(), 
            lr=learning_rate
        )
        
        # Training state
        self.training_step = 0
        self.loss_history = []
    
    def create_global_state(self, agent_states: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Create global state from individual agent states.
        
        Args:
            agent_states: Dictionary of agent states
            
        Returns:
            Global state tensor
        """
        # Sort agent states by agent name for consistent ordering
        sorted_states = []
        for i in range(self.n_agents):
            agent_name = f"agent_{i}"
            if agent_name in agent_states:
                sorted_states.append(agent_states[agent_name])
            else:
                # Handle missing agents with zero states
                sorted_states.append(np.zeros(self.local_state_dim))
        
        global_state = np.concatenate(sorted_states)
        return torch.FloatTensor(global_state).to(self.device)
    
    def get_value_estimates(self, 
                          agent_states: Union[Dict[str, np.ndarray], torch.Tensor]
                          ) -> Dict[str, torch.Tensor]:
        """
        Get value estimates for all agents given global state.
        
        Args:
            agent_states: Either dict of agent states or pre-formed global state
            
        Returns:
            Dictionary with value estimates
        """
        if isinstance(agent_states, dict):
            global_state = self.create_global_state(agent_states).unsqueeze(0)
        else:
            global_state = agent_states
            if global_state.dim() == 1:
                global_state = global_state.unsqueeze(0)
        
        with torch.no_grad():
            return self.critic_network(global_state)
    
    def train_step(self, 
                  current_states: torch.Tensor,
                  rewards: torch.Tensor,
                  next_states: torch.Tensor,
                  dones: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step for the central critic.
        
        Args:
            current_states: Current global states [batch_size, global_state_dim]
            rewards: Rewards for each agent [batch_size, n_agents]
            next_states: Next global states [batch_size, global_state_dim]
            dones: Done flags [batch_size]
            
        Returns:
            Training metrics
        """
        # Current value estimates
        current_values = self.critic_network(current_states)
        current_agent_values = current_values['agent_values']  # [batch_size, n_agents]
        current_global_value = current_values['global_value']  # [batch_size]
        
        # Target value estimates
        with torch.no_grad():
            next_values = self.target_critic_network(next_states)
            next_agent_values = next_values['agent_values']  # [batch_size, n_agents]
            next_global_value = next_values['global_value']  # [batch_size]
            
            # Compute targets
            agent_targets = rewards + (self.gamma * next_agent_values * ~dones.unsqueeze(1))
            global_targets = rewards.mean(dim=1) + (self.gamma * next_global_value * ~dones)
        
        # Compute losses
        agent_loss = F.mse_loss(current_agent_values, agent_targets)
        global_loss = F.mse_loss(current_global_value, global_targets)
        total_loss = agent_loss + 0.5 * global_loss  # Weight global loss less
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update training statistics
        self.training_step += 1
        self.loss_history.append(total_loss.item())
        
        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        return {
            'critic_loss': total_loss.item(),
            'agent_loss': agent_loss.item(),
            'global_loss': global_loss.item(),
            'value_estimates_mean': current_agent_values.mean().item()
        }
    
    def update_target_network(self):
        """Update target critic network."""
        self.target_critic_network.load_state_dict(
            self.critic_network.state_dict()
        )
    
    def compute_advantages(self,
                          states: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          dones: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages for policy gradient methods.
        
        Args:
            states: Current states
            rewards: Rewards
            next_states: Next states  
            dones: Done flags
            
        Returns:
            Advantage estimates
        """
        with torch.no_grad():
            current_values = self.critic_network(states)['agent_values']
            next_values = self.target_critic_network(next_states)['agent_values']
            
            # Compute TD targets
            targets = rewards + (self.gamma * next_values * ~dones.unsqueeze(1))
            
            # Advantages = targets - current_values
            advantages = targets - current_values
            
        return advantages
    
    def save(self, filepath: str):
        """Save central critic checkpoint."""
        checkpoint = {
            'n_agents': self.n_agents,
            'local_state_dim': self.local_state_dim,
            'global_state_dim': self.global_state_dim,
            'critic_network_state_dict': self.critic_network.state_dict(),
            'target_critic_network_state_dict': self.target_critic_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'loss_history': self.loss_history,
            'hyperparameters': {
                'gamma': self.gamma,
                'target_update_freq': self.target_update_freq
            }
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str):
        """Load central critic checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.critic_network.load_state_dict(checkpoint['critic_network_state_dict'])
        self.target_critic_network.load_state_dict(checkpoint['target_critic_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.loss_history = checkpoint.get('loss_history', [])
    
    def get_training_stats(self) -> Dict[str, any]:
        """Get training statistics."""
        return {
            'training_step': self.training_step,
            'recent_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            'total_updates': len(self.loss_history)
        }
