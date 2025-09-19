"""
Test suite for MockTrafficEnv.
"""

import pytest
import numpy as np
import gymnasium as gym
from src.marlo.envs.mock_env import MockTrafficEnv


class TestMockTrafficEnv:
    """Test MockTrafficEnv functionality."""
    
    def test_env_initialization(self):
        """Test environment initialization."""
        # Test default initialization
        env = MockTrafficEnv()
        
        assert env.n_agents == 4
        assert env.max_queue_length == 20
        assert env.episode_length == 3600
        assert env.decision_frequency == 10
        assert env.max_steps == 360  # 3600 / 10
        
        # Test custom initialization
        env_custom = MockTrafficEnv(
            n_agents=2,
            max_queue_length=15,
            episode_length=1800,
            decision_frequency=5
        )
        
        assert env_custom.n_agents == 2
        assert env_custom.max_queue_length == 15
        assert env_custom.episode_length == 1800
        assert env_custom.decision_frequency == 5
        assert env_custom.max_steps == 360  # 1800 / 5
        
        # Check spaces
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 4
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (25,)
    
    def test_env_reset(self):
        """Test environment reset functionality."""
        env = MockTrafficEnv(n_agents=2, seed=42)
        
        observations, info = env.reset(seed=42)
        
        # Check return types
        assert isinstance(observations, dict)
        assert isinstance(info, dict)
        
        # Check agent observations
        assert len(observations) == 2
        assert 'agent_0' in observations
        assert 'agent_1' in observations
        
        # Check observation shape and values
        for agent_id, obs in observations.items():
            assert isinstance(obs, np.ndarray)
            assert obs.shape == (25,)
            assert obs.dtype == np.float32
            assert np.all(obs >= 0.0) and np.all(obs <= 1.0)  # Normalized values
        
        # Check internal state initialization
        assert env.current_step == 0
        assert len(env.agent_states) == 2
        assert len(env.agent_phases) == 2
        assert len(env.agent_phase_timers) == 2
        
        # Test reset consistency with same seed
        observations2, _ = env.reset(seed=42)
        for agent_id in observations:
            assert np.allclose(observations[agent_id], observations2[agent_id])
    
    def test_observation_structure(self):
        """Test that observations have correct 25-dimensional structure."""
        env = MockTrafficEnv(n_agents=1, seed=42)
        observations, _ = env.reset()
        
        obs = observations['agent_0']
        
        # Check total dimension
        assert len(obs) == 25
        
        # Check components (based on implementation):
        # - Queue lengths: 4 values
        # - Turning proportions: 12 values (4 lanes × 3 directions)
        # - Current phase one-hot: 4 values
        # - Neighbor queues: 4 values
        # - Time in phase: 1 value
        
        queue_lengths = obs[:4]
        turning_props = obs[4:16]
        phase_onehot = obs[16:20]
        neighbor_queues = obs[20:24]
        time_in_phase = obs[24]
        
        # Queue lengths should be normalized
        assert np.all(queue_lengths >= 0.0) and np.all(queue_lengths <= 1.0)
        
        # Turning proportions should sum to 1 for each lane
        turning_props_reshaped = turning_props.reshape(4, 3)
        for lane_props in turning_props_reshaped:
            assert abs(np.sum(lane_props) - 1.0) < 1e-6
        
        # Phase one-hot should have exactly one 1
        assert np.sum(phase_onehot) == 1.0
        assert np.all((phase_onehot == 0) | (phase_onehot == 1))
        
        # Neighbor queues should be normalized
        assert np.all(neighbor_queues >= 0.0) and np.all(neighbor_queues <= 1.0)
        
        # Time in phase should be normalized
        assert 0.0 <= time_in_phase <= 1.0
    
    def test_env_step(self):
        """Test environment step functionality."""
        env = MockTrafficEnv(n_agents=2, seed=42)
        observations, _ = env.reset()
        
        # Create valid actions
        actions = {'agent_0': 0, 'agent_1': 2}
        
        # Take step
        next_observations, rewards, terminated, truncated, info = env.step(actions)
        
        # Check return types and structure
        assert isinstance(next_observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminated, dict)
        assert isinstance(truncated, dict)
        assert isinstance(info, dict)
        
        # Check agent data
        assert len(next_observations) == 2
        assert len(rewards) == 2
        assert len(terminated) == 2
        assert len(truncated) == 2
        
        # Check __all__ flags
        assert '__all__' in terminated
        assert '__all__' in truncated
        
        # Check observation shapes
        for agent_id, obs in next_observations.items():
            assert isinstance(obs, np.ndarray)
            assert obs.shape == (25,)
            assert obs.dtype == np.float32
        
        # Check rewards are floats
        for agent_id, reward in rewards.items():
            if agent_id != '__all__':
                assert isinstance(reward, (int, float))
        
        # Check done flags are booleans
        for agent_id, done in terminated.items():
            assert isinstance(done, bool)
        
        # Check that step counter incremented
        assert env.current_step == 1
    
    def test_action_validation(self):
        """Test that environment handles various action inputs."""
        env = MockTrafficEnv(n_agents=2)
        observations, _ = env.reset()
        
        # Test valid actions
        valid_actions = {'agent_0': 0, 'agent_1': 3}
        next_obs, rewards, terminated, truncated, info = env.step(valid_actions)
        assert not terminated['__all__']
        
        # Test with extra agents (should be ignored)
        extra_actions = {'agent_0': 1, 'agent_1': 2, 'agent_999': 0}
        next_obs, rewards, terminated, truncated, info = env.step(extra_actions)
        assert len(rewards) == 2  # Only 2 agents should get rewards
        
        # Test with missing agents (should still work)
        partial_actions = {'agent_0': 1}
        next_obs, rewards, terminated, truncated, info = env.step(partial_actions)
        # Environment should handle missing agents gracefully
    
    def test_episode_termination(self):
        """Test that episodes terminate correctly."""
        # Short episode for testing
        env = MockTrafficEnv(n_agents=1, episode_length=10, decision_frequency=5)
        observations, _ = env.reset()
        
        step_count = 0
        done = False
        
        while not done and step_count < 10:  # Safety limit
            actions = {'agent_0': 0}
            observations, rewards, terminated, truncated, info = env.step(actions)
            done = terminated.get('__all__', False) or truncated.get('__all__', False)
            step_count += 1
        
        assert done
        assert step_count == 2  # Should terminate after 2 steps (10/5 = 2)
        assert env.current_step == env.max_steps
    
    def test_reward_calculation(self):
        """Test that rewards are calculated correctly."""
        env = MockTrafficEnv(n_agents=2, seed=42)
        observations, _ = env.reset()
        
        actions = {'agent_0': 0, 'agent_1': 1}
        next_observations, rewards, terminated, truncated, info = env.step(actions)
        
        # Rewards should be negative (based on stopped vehicle ratio)
        for agent_id, reward in rewards.items():
            if agent_id != '__all__':
                assert isinstance(reward, (int, float))
                # Reward should typically be negative or zero
                assert reward <= 0.0
    
    def test_traffic_light_phases(self):
        """Test that traffic light phases are applied correctly."""
        env = MockTrafficEnv(n_agents=1, seed=42)
        observations, _ = env.reset()
        
        # Test each phase
        for phase in range(4):
            actions = {'agent_0': phase}
            next_observations, rewards, terminated, truncated, info = env.step(actions)
            
            # Check that phase was set
            assert env.agent_phases['agent_0'] == phase
            
            # Check that observation reflects the phase
            obs = next_observations['agent_0']
            phase_onehot = obs[16:20]  # Phase one-hot encoding
            assert phase_onehot[phase] == 1.0
            
            if not terminated.get('__all__', False):
                observations = next_observations
            else:
                break
    
    def test_queue_length_updates(self):
        """Test that queue lengths update during simulation."""
        env = MockTrafficEnv(n_agents=1, seed=42)
        observations, _ = env.reset()
        
        initial_queues = observations['agent_0'][:4] * env.max_queue_length
        
        # Run several steps
        for _ in range(5):
            actions = {'agent_0': 0}  # NS straight/right
            observations, rewards, terminated, truncated, info = env.step(actions)
            
            if terminated.get('__all__', False):
                break
        
        final_queues = observations['agent_0'][:4] * env.max_queue_length
        
        # Queue lengths should have changed (due to arrivals/departures)
        # Note: This might occasionally fail due to randomness, but should usually pass
        assert not np.allclose(initial_queues, final_queues, atol=0.1)
    
    def test_render_method(self):
        """Test that render method works without errors."""
        env = MockTrafficEnv(n_agents=2)
        observations, _ = env.reset()
        
        # Should not raise an exception
        env.render('human')
        
        # Take a step and render again
        actions = {'agent_0': 0, 'agent_1': 1}
        observations, rewards, terminated, truncated, info = env.step(actions)
        env.render('human')
    
    def test_close_method(self):
        """Test that close method works."""
        env = MockTrafficEnv(n_agents=1)
        env.close()  # Should not raise an exception
    
    def test_seed_reproducibility(self):
        """Test that seeding produces reproducible results."""
        # Create two environments with same seed
        env1 = MockTrafficEnv(n_agents=2, seed=123)
        env2 = MockTrafficEnv(n_agents=2, seed=123)
        
        # Reset with same seed
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        
        # Should be identical
        for agent_id in obs1:
            assert np.allclose(obs1[agent_id], obs2[agent_id])
        
        # Take same actions
        actions = {'agent_0': 1, 'agent_1': 3}
        next_obs1, rewards1, _, _, _ = env1.step(actions)
        next_obs2, rewards2, _, _, _ = env2.step(actions)
        
        # Results should be identical
        for agent_id in next_obs1:
            assert np.allclose(next_obs1[agent_id], next_obs2[agent_id])
            assert abs(rewards1[agent_id] - rewards2[agent_id]) < 1e-10


@pytest.fixture
def sample_env():
    """Provide a sample environment for testing."""
    return MockTrafficEnv(n_agents=2, max_queue_length=10, seed=42)


def test_full_episode_run(sample_env):
    """Integration test for a full episode."""
    env = sample_env
    observations, _ = env.reset()
    
    total_rewards = {agent_id: 0.0 for agent_id in env.agent_names}
    step_count = 0
    done = False
    
    while not done and step_count < env.max_steps + 10:  # Safety limit
        # Random valid actions
        actions = {agent_id: np.random.randint(0, 4) for agent_id in observations.keys()}
        
        next_observations, rewards, terminated, truncated, info = env.step(actions)
        
        # Accumulate rewards
        for agent_id in total_rewards:
            if agent_id in rewards:
                total_rewards[agent_id] += rewards[agent_id]
        
        observations = next_observations
        done = terminated.get('__all__', False) or truncated.get('__all__', False)
        step_count += 1
    
    # Episode should have terminated
    assert done
    assert step_count <= env.max_steps
    
    # Should have collected rewards for all agents
    for agent_id in env.agent_names:
        assert agent_id in total_rewards
        assert isinstance(total_rewards[agent_id], (int, float))


if __name__ == "__main__":
    pytest.main([__file__])
