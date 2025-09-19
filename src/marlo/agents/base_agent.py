"""
Base agent class for multi-agent traffic signal control.
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple


class BaseAgent(ABC):
    """
    Abstract base class for traffic signal control agents.
    """
    
    def __init__(self, 
                 agent_id: str,
                 state_dim: int = 25,
                 action_dim: int = 4,
                 device: str = 'cuda'):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            state_dim: State space dimensionality
            action_dim: Action space dimensionality
            device: Torch device
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        
    @abstractmethod
    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: Current state observation
            deterministic: Whether to act deterministically
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Learn from a batch of experience.
        
        Args:
            batch: Batch of experience tuples
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """
        Save agent parameters.
        
        Args:
            filepath: Path to save checkpoint
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """
        Load agent parameters.
        
        Args:
            filepath: Path to checkpoint
        """
        pass
    
    def update_target_network(self):
        """Update target network (if applicable)."""
        pass
    
    def set_train_mode(self):
        """Set agent to training mode."""
        pass
    
    def set_eval_mode(self):
        """Set agent to evaluation mode."""
        pass
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'agent_id': self.agent_id,
            'training_step': self.training_step,
            'episode_count': self.episode_count
        }
