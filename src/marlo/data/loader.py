"""
Dataset loader for offline reinforcement learning.
Loads and preprocesses datasets for training multi-agent traffic signal controllers.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
import json

from ..utils.logger import get_logger


class TrafficDataset(Dataset):
    """
    PyTorch Dataset for traffic signal control data.
    Handles both synthetic and semi-synthetic datasets.
    """
    
    def __init__(self, 
                 dataset_path: str,
                 agent_filter: Optional[List[str]] = None,
                 sequence_length: Optional[int] = None,
                 normalize_rewards: bool = True,
                 normalize_observations: bool = True,
                 clip_rewards: bool = True,
                 reward_clip_value: float = 1.0):
        """
        Initialize traffic dataset.
        
        Args:
            dataset_path: Path to .npz dataset file
            agent_filter: List of agent IDs to include (None = all agents)
            sequence_length: If specified, return sequences of this length
            normalize_rewards: Whether to normalize rewards
        """
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.normalize_rewards = normalize_rewards
        self.normalize_observations = normalize_observations
        self.clip_rewards = clip_rewards
        self.reward_clip_value = reward_clip_value
        self.logger = get_logger("dataset_loader")
        
        # Load dataset
        self.data = np.load(dataset_path, allow_pickle=True)
        
        # Load metadata if available
        metadata_path = dataset_path.replace('.npz', '_metadata.json')
        self.metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Filter by agents if specified
        if agent_filter is not None:
            self._filter_by_agents(agent_filter)
        
        # Preprocess data
        self._preprocess_data()
        
        self.logger.info(f"Loaded dataset: {len(self)} transitions from {dataset_path}")
    
    def _filter_by_agents(self, agent_filter: List[str]):
        """Filter dataset by specific agents."""
        if 'agent_ids' not in self.data:
            self.logger.warning("No agent_ids found in dataset, cannot filter by agents")
            return
        
        agent_ids = self.data['agent_ids']
        mask = np.isin(agent_ids, agent_filter)
        
        for key in self.data.keys():
            if key != 'agent_ids' and len(self.data[key]) == len(mask):
                self.data[key] = self.data[key][mask]
        
        self.logger.info(f"Filtered dataset to {np.sum(mask)} transitions for agents: {agent_filter}")
    
    def _preprocess_data(self):
        """Preprocess dataset for training."""
        # Convert observations to float32
        self.observations = self.data['observations'].astype(np.float32)
        self.next_observations = self.data['next_observations'].astype(np.float32)
        if self.normalize_observations:
            obs_mean = np.mean(self.observations, axis=0, keepdims=True)
            obs_std = np.std(self.observations, axis=0, keepdims=True)
            obs_std[obs_std < 1e-6] = 1.0
            self.observations = (self.observations - obs_mean) / obs_std
            self.next_observations = (self.next_observations - obs_mean) / obs_std
            self.logger.info("Normalized observations per-dimension (z-score)")
        
        # Convert actions to long (for DQN)
        self.actions = self.data['actions'].astype(np.int64)
        
        # Process rewards
        self.rewards = self.data['rewards'].astype(np.float32)
        if self.normalize_rewards:
            reward_mean = np.mean(self.rewards)
            reward_std = np.std(self.rewards)
            if reward_std > 0:
                self.rewards = (self.rewards - reward_mean) / reward_std
                self.logger.info(f"Normalized rewards: mean={reward_mean:.3f}, std={reward_std:.3f}")
        if self.clip_rewards and self.reward_clip_value is not None:
            v = float(self.reward_clip_value)
            self.rewards = np.clip(self.rewards, -v, v)
        
        # Convert dones to boolean (ensure uint8 type for compatibility)
        self.dones = self.data['dones'].astype(np.uint8)
        
        # Store additional info if available
        self.episode_ids = self.data.get('episode_ids', np.zeros(len(self.observations), dtype=np.int32))
        self.timesteps = self.data.get('timesteps', np.arange(len(self.observations)))
        self.agent_ids = self.data.get('agent_ids', ['agent_0'] * len(self.observations))
        
        # Handle sequence data if requested
        if self.sequence_length is not None:
            self._create_sequences()
    
    def _create_sequences(self):
        """Create sequences of transitions for sequential models."""
        self.logger.info(f"Creating sequences of length {self.sequence_length}")
        
        # Group by episode and agent
        episode_agent_groups = {}
        for i, (ep_id, agent_id) in enumerate(zip(self.episode_ids, self.agent_ids)):
            key = (ep_id, agent_id)
            if key not in episode_agent_groups:
                episode_agent_groups[key] = []
            episode_agent_groups[key].append(i)
        
        # Create sequences
        sequence_indices = []
        for indices in episode_agent_groups.values():
            indices = sorted(indices)  # Ensure temporal order
            for i in range(len(indices) - self.sequence_length + 1):
                sequence_indices.append(indices[i:i + self.sequence_length])
        
        self.sequence_indices = sequence_indices
        self.logger.info(f"Created {len(sequence_indices)} sequences")
    
    def __len__(self):
        if self.sequence_length is not None:
            return len(self.sequence_indices)
        return len(self.observations)
    
    def __getitem__(self, idx):
        if self.sequence_length is not None:
            # Return sequence
            indices = self.sequence_indices[idx]
            return {
                'observations': torch.FloatTensor(self.observations[indices]),
                'actions': torch.LongTensor(self.actions[indices]),
                'rewards': torch.FloatTensor(self.rewards[indices]),
                'next_observations': torch.FloatTensor(self.next_observations[indices]),
                'dones': torch.tensor(self.dones[indices], dtype=torch.bool),
                'episode_ids': self.episode_ids[indices[0]],  # Episode ID of first transition
                'agent_ids': self.agent_ids[indices[0]]  # Agent ID
            }
        else:
            # Return single transition
            return {
                'observations': torch.FloatTensor(self.observations[idx]),
                'actions': torch.tensor(self.actions[idx], dtype=torch.long),  # scalar action
                'rewards': torch.tensor(self.rewards[idx], dtype=torch.float32),  # scalar reward
                'next_observations': torch.FloatTensor(self.next_observations[idx]),
                'dones': torch.tensor(bool(self.dones[idx]), dtype=torch.bool),  # scalar bool
                'episode_ids': self.episode_ids[idx],
                'agent_ids': self.agent_ids[idx]
            }
    
    def get_state_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about state dimensions."""
        stats = {}
        for dim in range(self.observations.shape[1]):
            dim_values = self.observations[:, dim]
            stats[f'dim_{dim}'] = {
                'mean': float(np.mean(dim_values)),
                'std': float(np.std(dim_values)),
                'min': float(np.min(dim_values)),
                'max': float(np.max(dim_values))
            }
        return stats
    
    def get_action_distribution(self) -> Dict[int, float]:
        """Get distribution of actions in dataset."""
        unique, counts = np.unique(self.actions, return_counts=True)
        total = len(self.actions)
        return {int(action): float(count / total) for action, count in zip(unique, counts)}


class MultiAgentDataLoader:
    """
    Data loader for multi-agent training.
    Handles batching and sampling across multiple agents.
    """
    
    def __init__(self,
                 dataset_paths: Union[str, List[str]],
                 batch_size: int = 64,
                 shuffle: bool = True,
                 agent_sampling: str = 'uniform',  # 'uniform', 'round_robin', 'proportional'
                 num_workers: int = 0,
                 **dataset_kwargs):
        """
        Initialize multi-agent data loader.
        
        Args:
            dataset_paths: Path(s) to dataset files
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            agent_sampling: How to sample agents ('uniform', 'round_robin', 'proportional')
            num_workers: Number of worker processes
            **dataset_kwargs: Additional arguments for TrafficDataset
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.agent_sampling = agent_sampling
        self.logger = get_logger("multi_agent_dataloader")
        
        # Load datasets
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]
        
        self.datasets = []
        for path in dataset_paths:
            dataset = TrafficDataset(path, **dataset_kwargs)
            self.datasets.append(dataset)
            self.logger.info(f"Loaded dataset from {path}: {len(dataset)} transitions")
        
        # Create combined dataset
        self.combined_dataset = torch.utils.data.ConcatDataset(self.datasets)
        
        # Create data loader
        self.dataloader = DataLoader(
            self.combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        
        self.logger.info(f"Created multi-agent data loader: {len(self.combined_dataset)} total transitions")
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for handling variable-length data."""
        # Stack tensors
        collated = {}
        for key in batch[0].keys():
            if key in ['episode_ids', 'agent_ids']:
                collated[key] = [item[key] for item in batch]
            else:
                try:
                    collated[key] = torch.stack([item[key] for item in batch])
                except RuntimeError:
                    # Handle variable-length sequences by padding
                    sequences = [item[key] for item in batch]
                    collated[key] = torch.nn.utils.rnn.pad_sequence(
                        sequences, batch_first=True
                    )
        
        return collated
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the combined dataset."""
        total_transitions = len(self.combined_dataset)
        
        # Aggregate stats from individual datasets
        all_stats = []
        for i, dataset in enumerate(self.datasets):
            stats = {
                'dataset_idx': i,
                'transitions': len(dataset),
                'state_stats': dataset.get_state_stats(),
                'action_distribution': dataset.get_action_distribution(),
                'metadata': dataset.metadata
            }
            all_stats.append(stats)
        
        return {
            'total_transitions': total_transitions,
            'num_datasets': len(self.datasets),
            'batch_size': self.batch_size,
            'datasets': all_stats
        }


def load_dataset(dataset_path: str, 
                batch_size: int = 64,
                **kwargs) -> DataLoader:
    """
    Convenience function to load a single dataset.
    
    Args:
        dataset_path: Path to dataset file
        batch_size: Batch size
        **kwargs: Additional arguments
        
    Returns:
        DataLoader for the dataset
    """
    dataset = TrafficDataset(dataset_path, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def create_multi_agent_loader(dataset_paths: Union[str, List[str]], 
                             **kwargs) -> MultiAgentDataLoader:
    """
    Convenience function to create multi-agent data loader.
    
    Args:
        dataset_paths: Path(s) to dataset files
        **kwargs: Additional arguments
        
    Returns:
        MultiAgentDataLoader
    """
    return MultiAgentDataLoader(dataset_paths, **kwargs)
