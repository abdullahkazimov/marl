"""
Test suite for dataset loading functionality.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.marlo.data.loader import TrafficDataset, MultiAgentDataLoader, load_dataset


class TestTrafficDataset:
    """Test TrafficDataset class."""
    
    @pytest.fixture
    def sample_dataset_file(self):
        """Create a temporary dataset file for testing."""
        # Create sample data
        n_transitions = 100
        n_agents = 2
        
        data = {
            'observations': np.random.rand(n_transitions, 25).astype(np.float32),
            'actions': np.random.randint(0, 4, n_transitions).astype(np.int64),
            'rewards': np.random.randn(n_transitions).astype(np.float32),
            'next_observations': np.random.rand(n_transitions, 25).astype(np.float32),
            'dones': np.random.choice([True, False], n_transitions),
            'episode_ids': np.random.randint(0, 10, n_transitions),
            'timesteps': np.arange(n_transitions),
            'agent_ids': np.random.choice(['agent_0', 'agent_1'], n_transitions)
        }
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
        np.savez_compressed(temp_file.name, **data)
        temp_file.close()
        
        yield temp_file.name, data
        
        # Cleanup
        os.unlink(temp_file.name)
    
    def test_dataset_initialization(self, sample_dataset_file):
        """Test dataset initialization."""
        filepath, expected_data = sample_dataset_file
        
        dataset = TrafficDataset(filepath)
        
        assert len(dataset) == len(expected_data['observations'])
        assert dataset.dataset_path == filepath
        assert isinstance(dataset.observations, np.ndarray)
        assert isinstance(dataset.actions, np.ndarray)
        assert isinstance(dataset.rewards, np.ndarray)
        assert isinstance(dataset.next_observations, np.ndarray)
        assert isinstance(dataset.dones, np.ndarray)
    
    def test_dataset_getitem(self, sample_dataset_file):
        """Test dataset item retrieval."""
        filepath, expected_data = sample_dataset_file
        
        dataset = TrafficDataset(filepath, normalize_rewards=False)
        
        # Test single item
        item = dataset[0]
        
        assert isinstance(item, dict)
        assert 'observations' in item
        assert 'actions' in item
        assert 'rewards' in item
        assert 'next_observations' in item
        assert 'dones' in item
        
        # Check shapes and types
        assert item['observations'].shape == (25,)
        assert item['actions'].shape == (1,)
        assert item['rewards'].shape == (1,)
        assert item['next_observations'].shape == (25,)
        assert item['dones'].shape == (1,)
        
        # Check data matches original
        np.testing.assert_array_equal(
            item['observations'].numpy(), 
            expected_data['observations'][0]
        )
        assert item['actions'].item() == expected_data['actions'][0]
    
    def test_agent_filtering(self, sample_dataset_file):
        """Test filtering dataset by specific agents."""
        filepath, expected_data = sample_dataset_file
        
        # Filter to only agent_0
        dataset = TrafficDataset(filepath, agent_filter=['agent_0'])
        
        # Should have fewer items than original
        agent_0_mask = expected_data['agent_ids'] == 'agent_0'
        expected_length = np.sum(agent_0_mask)
        
        assert len(dataset) == expected_length
        
        # All remaining items should be from agent_0
        for i in range(min(10, len(dataset))):  # Check first 10 items
            item = dataset[i]
            assert item['agent_ids'] == 'agent_0'
    
    def test_reward_normalization(self, sample_dataset_file):
        """Test reward normalization."""
        filepath, expected_data = sample_dataset_file
        
        # Test with normalization
        dataset_norm = TrafficDataset(filepath, normalize_rewards=True)
        
        # Test without normalization  
        dataset_no_norm = TrafficDataset(filepath, normalize_rewards=False)
        
        # Normalized rewards should have different values
        item_norm = dataset_norm[0]
        item_no_norm = dataset_no_norm[0]
        
        # Original rewards
        original_reward = expected_data['rewards'][0]
        
        # Non-normalized should match original
        assert abs(item_no_norm['rewards'].item() - original_reward) < 1e-6
        
        # Normalized should be different (unless reward was already normalized)
        normalized_reward = item_norm['rewards'].item()
        
        # Check that normalization was applied (mean and std based)
        all_rewards = expected_data['rewards']
        expected_mean = np.mean(all_rewards)
        expected_std = np.std(all_rewards)
        
        if expected_std > 0:
            expected_normalized = (original_reward - expected_mean) / expected_std
            assert abs(normalized_reward - expected_normalized) < 1e-6
    
    def test_state_stats(self, sample_dataset_file):
        """Test state statistics calculation."""
        filepath, expected_data = sample_dataset_file
        
        dataset = TrafficDataset(filepath)
        stats = dataset.get_state_stats()
        
        # Should have stats for all 25 dimensions
        assert len(stats) == 25
        
        for dim in range(25):
            dim_key = f'dim_{dim}'
            assert dim_key in stats
            
            dim_stats = stats[dim_key]
            assert 'mean' in dim_stats
            assert 'std' in dim_stats
            assert 'min' in dim_stats
            assert 'max' in dim_stats
            
            # Verify stats are reasonable
            assert dim_stats['min'] <= dim_stats['max']
            assert dim_stats['std'] >= 0
    
    def test_action_distribution(self, sample_dataset_file):
        """Test action distribution calculation."""
        filepath, expected_data = sample_dataset_file
        
        dataset = TrafficDataset(filepath)
        action_dist = dataset.get_action_distribution()
        
        # Should have distribution for actions 0-3
        assert isinstance(action_dist, dict)
        
        # Probabilities should sum to 1
        total_prob = sum(action_dist.values())
        assert abs(total_prob - 1.0) < 1e-6
        
        # All probabilities should be positive
        for action, prob in action_dist.items():
            assert 0 <= prob <= 1
            assert isinstance(action, (int, np.integer))
            assert 0 <= action <= 3


class TestMultiAgentDataLoader:
    """Test MultiAgentDataLoader class."""
    
    @pytest.fixture
    def sample_dataset_files(self):
        """Create multiple temporary dataset files."""
        files = []
        
        for i in range(2):
            n_transitions = 50 + i * 25
            
            data = {
                'observations': np.random.rand(n_transitions, 25).astype(np.float32),
                'actions': np.random.randint(0, 4, n_transitions).astype(np.int64),
                'rewards': np.random.randn(n_transitions).astype(np.float32),
                'next_observations': np.random.rand(n_transitions, 25).astype(np.float32),
                'dones': np.random.choice([True, False], n_transitions),
                'episode_ids': np.random.randint(0, 5, n_transitions),
                'agent_ids': np.random.choice(['agent_0', 'agent_1'], n_transitions)
            }
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
            np.savez_compressed(temp_file.name, **data)
            temp_file.close()
            
            files.append((temp_file.name, data))
        
        yield files
        
        # Cleanup
        for filepath, _ in files:
            os.unlink(filepath)
    
    def test_multi_agent_dataloader_init(self, sample_dataset_files):
        """Test MultiAgentDataLoader initialization."""
        filepaths = [filepath for filepath, _ in sample_dataset_files]
        
        dataloader = MultiAgentDataLoader(
            dataset_paths=filepaths,
            batch_size=16,
            shuffle=True
        )
        
        assert dataloader.batch_size == 16
        assert dataloader.shuffle == True
        assert len(dataloader.datasets) == 2
        
        # Test with single path
        single_loader = MultiAgentDataLoader(
            dataset_paths=filepaths[0],  # Single path as string
            batch_size=8
        )
        
        assert len(single_loader.datasets) == 1
    
    def test_multi_agent_dataloader_iteration(self, sample_dataset_files):
        """Test iterating through MultiAgentDataLoader."""
        filepaths = [filepath for filepath, _ in sample_dataset_files]
        
        dataloader = MultiAgentDataLoader(
            dataset_paths=filepaths,
            batch_size=10,
            shuffle=False
        )
        
        # Test iteration
        batch_count = 0
        for batch in dataloader:
            assert isinstance(batch, dict)
            
            # Check required keys
            required_keys = ['observations', 'actions', 'rewards', 
                           'next_observations', 'dones']
            for key in required_keys:
                assert key in batch
            
            # Check batch dimensions
            batch_size = batch['observations'].shape[0]
            assert batch_size <= 10  # Should not exceed specified batch size
            
            # Check tensor types
            assert batch['observations'].dtype == torch.float32
            assert batch['actions'].dtype == torch.int64
            assert batch['rewards'].dtype == torch.float32
            
            batch_count += 1
            
            if batch_count >= 3:  # Limit test to first few batches
                break
        
        assert batch_count > 0
    
    def test_dataset_stats(self, sample_dataset_files):
        """Test dataset statistics from MultiAgentDataLoader."""
        filepaths = [filepath for filepath, _ in sample_dataset_files]
        
        dataloader = MultiAgentDataLoader(
            dataset_paths=filepaths,
            batch_size=8
        )
        
        stats = dataloader.get_dataset_stats()
        
        assert 'total_transitions' in stats
        assert 'num_datasets' in stats
        assert 'batch_size' in stats
        assert 'datasets' in stats
        
        assert stats['num_datasets'] == 2
        assert stats['batch_size'] == 8
        assert stats['total_transitions'] > 0
        
        # Check individual dataset stats
        dataset_stats = stats['datasets']
        assert len(dataset_stats) == 2
        
        for ds_stat in dataset_stats:
            assert 'dataset_idx' in ds_stat
            assert 'transitions' in ds_stat
            assert 'state_stats' in ds_stat
            assert 'action_distribution' in ds_stat


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.fixture
    def sample_dataset_file(self):
        """Create a sample dataset file."""
        n_transitions = 50
        
        data = {
            'observations': np.random.rand(n_transitions, 25).astype(np.float32),
            'actions': np.random.randint(0, 4, n_transitions).astype(np.int64),
            'rewards': np.random.randn(n_transitions).astype(np.float32),
            'next_observations': np.random.rand(n_transitions, 25).astype(np.float32),
            'dones': np.random.choice([True, False], n_transitions),
        }
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
        np.savez_compressed(temp_file.name, **data)
        temp_file.close()
        
        yield temp_file.name, data
        
        os.unlink(temp_file.name)
    
    def test_load_dataset_function(self, sample_dataset_file):
        """Test load_dataset convenience function."""
        filepath, _ = sample_dataset_file
        
        dataloader = load_dataset(filepath, batch_size=16)
        
        # Should return a DataLoader
        assert hasattr(dataloader, '__iter__')
        assert hasattr(dataloader, '__len__')
        
        # Test iteration
        for batch in dataloader:
            assert isinstance(batch, dict)
            assert 'observations' in batch
            break  # Just test one batch


def test_missing_dataset_file():
    """Test handling of missing dataset files."""
    with pytest.raises(FileNotFoundError):
        TrafficDataset("non_existent_file.npz")


def test_empty_dataset_paths():
    """Test handling of empty dataset paths."""
    with pytest.raises(FileNotFoundError):
        MultiAgentDataLoader([])


# Import required modules for tests
import torch
from torch.utils.data import DataLoader


if __name__ == "__main__":
    pytest.main([__file__])
