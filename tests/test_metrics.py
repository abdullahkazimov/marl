"""
Test suite for traffic metrics calculations.
"""

import pytest
import numpy as np
from src.marlo.utils.metrics import (
    stopped_vehicle_ratio,
    average_waiting_time,
    average_queue_length,
    throughput,
    calculate_reward,
    MetricsTracker
)


class TestMetricsFunctions:
    """Test individual metrics functions."""
    
    def test_stopped_vehicle_ratio(self):
        """Test stopped vehicle ratio calculation."""
        # Test basic calculation
        queue_lengths = [5, 10, 0, 3]
        lane_capacities = [20, 20, 20, 20]
        ratio = stopped_vehicle_ratio(queue_lengths, lane_capacities)
        expected = (5 + 10 + 0 + 3) / (4 * 20)  # 18/80 = 0.225
        assert abs(ratio - expected) < 1e-6
        
        # Test with default capacities
        ratio_default = stopped_vehicle_ratio(queue_lengths)
        assert abs(ratio_default - expected) < 1e-6
        
        # Test empty queues
        empty_queues = [0, 0, 0, 0]
        ratio_empty = stopped_vehicle_ratio(empty_queues)
        assert ratio_empty == 0.0
        
        # Test with numpy arrays
        queue_array = np.array([2, 4, 6, 8])
        capacity_array = np.array([10, 10, 10, 10])
        ratio_array = stopped_vehicle_ratio(queue_array, capacity_array)
        expected_array = 20 / 40  # 0.5
        assert abs(ratio_array - expected_array) < 1e-6
    
    def test_average_waiting_time(self):
        """Test average waiting time calculation."""
        # Test basic calculation
        waiting_times = [10, 20, 30, 15, 5]
        avg_wait = average_waiting_time(waiting_times)
        expected = 16.0  # (10+20+30+15+5)/5
        assert abs(avg_wait - expected) < 1e-6
        
        # Test empty list
        avg_wait_empty = average_waiting_time([])
        assert avg_wait_empty == 0.0
        
        # Test with numpy array
        wait_array = np.array([1, 2, 3, 4, 5])
        avg_wait_array = average_waiting_time(wait_array)
        assert abs(avg_wait_array - 3.0) < 1e-6
    
    def test_average_queue_length(self):
        """Test average queue length calculation."""
        # Test basic calculation
        queue_lengths = [5, 0, 10, 15]
        avg_queue = average_queue_length(queue_lengths)
        expected = 7.5  # (5+0+10+15)/4
        assert abs(avg_queue - expected) < 1e-6
        
        # Test empty list
        avg_queue_empty = average_queue_length([])
        assert avg_queue_empty == 0.0
        
        # Test single value
        avg_queue_single = average_queue_length([8])
        assert avg_queue_single == 8.0
    
    def test_throughput(self):
        """Test throughput calculation."""
        # Test basic calculation (vehicles per minute)
        passed_vehicles = 120
        episode_duration = 3600  # 1 hour
        tp = throughput(passed_vehicles, episode_duration)
        expected = 120 / 60  # 2 vehicles per minute
        assert abs(tp - expected) < 1e-6
        
        # Test different duration
        tp_30min = throughput(60, 1800)  # 30 minutes
        assert abs(tp_30min - 2.0) < 1e-6
        
        # Test zero duration
        tp_zero = throughput(100, 0)
        assert tp_zero == 0.0
        
        # Test zero vehicles
        tp_no_vehicles = throughput(0, 3600)
        assert tp_no_vehicles == 0.0
    
    def test_calculate_reward(self):
        """Test reward function calculation."""
        # Test basic reward calculation
        queue_lengths = [10, 5, 0, 15]
        alpha = 2.0
        lane_capacities = [20, 20, 20, 20]
        
        reward = calculate_reward(queue_lengths, alpha, lane_capacities)
        expected_ratio = 30 / 80  # 0.375
        expected_reward = -alpha * expected_ratio  # -0.75
        assert abs(reward - expected_reward) < 1e-6
        
        # Test with default alpha
        reward_default = calculate_reward(queue_lengths, lane_capacities=lane_capacities)
        expected_default = -1.0 * expected_ratio
        assert abs(reward_default - expected_default) < 1e-6


class TestMetricsTracker:
    """Test MetricsTracker class."""
    
    def test_metrics_tracker_initialization(self):
        """Test metrics tracker initialization."""
        tracker = MetricsTracker()
        
        assert len(tracker.stopped_ratios) == 0
        assert len(tracker.waiting_times) == 0
        assert len(tracker.queue_lengths) == 0
        assert len(tracker.throughputs) == 0
        assert len(tracker.rewards) == 0
    
    def test_add_episode_metrics(self):
        """Test adding episode metrics."""
        tracker = MetricsTracker()
        
        # Add first episode
        queue_lengths = [5, 10, 8, 3]
        waiting_times = [12, 8, 15, 20, 5]
        passed_vehicles = 45
        episode_duration = 1800  # 30 minutes
        
        tracker.add_episode_metrics(
            queue_lengths=queue_lengths,
            waiting_times=waiting_times,
            passed_vehicles=passed_vehicles,
            episode_duration=episode_duration
        )
        
        assert len(tracker.stopped_ratios) == 1
        assert len(tracker.waiting_times) == 1
        assert len(tracker.queue_lengths) == 1
        assert len(tracker.throughputs) == 1
        assert len(tracker.rewards) == 1
        
        # Check values
        assert tracker.queue_lengths[0] == np.mean(queue_lengths)
        assert tracker.waiting_times[0] == np.mean(waiting_times)
        assert tracker.throughputs[0] == passed_vehicles / (episode_duration / 60)
        
        # Add second episode
        tracker.add_episode_metrics(
            queue_lengths=[2, 4, 6, 1],
            waiting_times=[5, 10],
            passed_vehicles=50,
            episode_duration=1800
        )
        
        assert len(tracker.stopped_ratios) == 2
        assert len(tracker.waiting_times) == 2
    
    def test_get_summary(self):
        """Test getting summary statistics."""
        tracker = MetricsTracker()
        
        # Add multiple episodes
        for i in range(5):
            queue_lengths = [i+1, i+2, i+3, i+1]
            waiting_times = [10+i, 15+i, 20+i]
            passed_vehicles = 40 + i*5
            
            tracker.add_episode_metrics(
                queue_lengths=queue_lengths,
                waiting_times=waiting_times,
                passed_vehicles=passed_vehicles,
                episode_duration=3600
            )
        
        summary = tracker.get_summary()
        
        # Check that all required keys are present
        required_keys = [
            'stopped_vehicle_ratio_mean', 'stopped_vehicle_ratio_std',
            'average_waiting_time_mean', 'average_waiting_time_std',
            'average_queue_length_mean', 'average_queue_length_std',
            'throughput_mean', 'throughput_std',
            'reward_mean', 'reward_std'
        ]
        
        for key in required_keys:
            assert key in summary
            assert isinstance(summary[key], (int, float))
    
    def test_get_latest_metrics(self):
        """Test getting latest episode metrics."""
        tracker = MetricsTracker()
        
        # No episodes yet
        latest_empty = tracker.get_latest_metrics()
        assert latest_empty == {}
        
        # Add an episode
        tracker.add_episode_metrics(
            queue_lengths=[1, 2, 3, 4],
            waiting_times=[10, 20],
            passed_vehicles=30,
            episode_duration=3600
        )
        
        latest = tracker.get_latest_metrics()
        assert 'stopped_vehicle_ratio' in latest
        assert 'average_waiting_time' in latest
        assert 'average_queue_length' in latest
        assert 'throughput' in latest
        assert 'reward' in latest
        
        # Add another episode
        tracker.add_episode_metrics(
            queue_lengths=[5, 6, 7, 8],
            waiting_times=[5, 15, 25],
            passed_vehicles=40,
            episode_duration=3600
        )
        
        latest_updated = tracker.get_latest_metrics()
        # Should reflect the most recent episode
        assert latest_updated['average_queue_length'] == np.mean([5, 6, 7, 8])
        assert latest_updated['average_waiting_time'] == np.mean([5, 15, 25])
    
    def test_reset(self):
        """Test resetting metrics tracker."""
        tracker = MetricsTracker()
        
        # Add some data
        tracker.add_episode_metrics(
            queue_lengths=[1, 2, 3],
            waiting_times=[10, 20],
            passed_vehicles=25,
            episode_duration=1800
        )
        
        assert len(tracker.stopped_ratios) == 1
        
        # Reset
        tracker.reset()
        
        # Should be empty again
        assert len(tracker.stopped_ratios) == 0
        assert len(tracker.waiting_times) == 0
        assert len(tracker.queue_lengths) == 0
        assert len(tracker.throughputs) == 0
        assert len(tracker.rewards) == 0


@pytest.fixture
def sample_metrics_data():
    """Provide sample metrics data for testing."""
    return {
        'queue_lengths': [5, 10, 8, 3, 12, 6, 9, 2],
        'waiting_times': [12, 18, 15, 22, 8, 25, 10, 30],
        'passed_vehicles': 45,
        'episode_duration': 3600
    }


def test_metrics_integration(sample_metrics_data):
    """Integration test for metrics calculation pipeline."""
    # Test the full pipeline of metrics calculation
    queue_lengths = sample_metrics_data['queue_lengths']
    waiting_times = sample_metrics_data['waiting_times']
    passed_vehicles = sample_metrics_data['passed_vehicles']
    episode_duration = sample_metrics_data['episode_duration']
    
    # Calculate individual metrics
    stopped_ratio = stopped_vehicle_ratio(queue_lengths)
    avg_wait = average_waiting_time(waiting_times)
    avg_queue = average_queue_length(queue_lengths)
    tp = throughput(passed_vehicles, episode_duration)
    reward = calculate_reward(queue_lengths)
    
    # Verify all metrics are reasonable
    assert 0 <= stopped_ratio <= 1
    assert avg_wait >= 0
    assert avg_queue >= 0
    assert tp >= 0
    assert reward <= 0  # Reward should be negative or zero
    
    # Test with metrics tracker
    tracker = MetricsTracker()
    tracker.add_episode_metrics(
        queue_lengths=queue_lengths,
        waiting_times=waiting_times,
        passed_vehicles=passed_vehicles,
        episode_duration=episode_duration
    )
    
    latest = tracker.get_latest_metrics()
    
    # Verify tracker produces same results
    assert abs(latest['stopped_vehicle_ratio'] - stopped_ratio) < 1e-6
    assert abs(latest['average_waiting_time'] - avg_wait) < 1e-6
    assert abs(latest['average_queue_length'] - avg_queue) < 1e-6
    assert abs(latest['throughput'] - tp) < 1e-6
    assert abs(latest['reward'] - reward) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])
