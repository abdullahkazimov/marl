"""
Mock Traffic Environment for MARLO.
Implements the exact specifications from the problem:
- 25-dimensional state space per agent
- 4 discrete actions (traffic light phases)
- Multi-agent setup with independent intersection control
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from gymnasium.spaces import Box, Discrete


class MockTrafficEnv(gym.Env):
    """
    Mock traffic environment that simulates multi-agent traffic signal control.
    
    State Space (25-dimensional per agent):
    - Queue Lengths: [N, S, E, W] lanes (4 values)
    - Turning Proportions: For each lane [straight, left, right] (4×3 = 12 values)
    - Current Signal Phase: One-hot encoded (4 values)
    - Neighbor Queue Lengths: Adjacent intersections' relevant lanes (4 values)
    - Additional Traffic Info: [time_in_phase] (1 value)
    
    Action Space: 4 discrete actions
    - 0: North-South straight/right green, East-West red
    - 1: North-South left green, East-West red  
    - 2: East-West straight/right green, North-South red
    - 3: East-West left green, North-South red
    """
    
    def __init__(self, 
                 n_agents: int = 4,
                 max_queue_length: int = 20,
                 episode_length: int = 3600,
                 decision_frequency: int = 10,
                 seed: Optional[int] = None):
        """
        Initialize the mock traffic environment.
        
        Args:
            n_agents: Number of intersections (agents)
            max_queue_length: Maximum vehicles per lane
            episode_length: Episode duration in seconds
            decision_frequency: Time between decisions in seconds
            seed: Random seed
        """
        super().__init__()
        
        self.n_agents = n_agents
        self.max_queue_length = max_queue_length
        self.episode_length = episode_length
        self.decision_frequency = decision_frequency
        self.max_steps = episode_length // decision_frequency
        
        # Define action and observation spaces
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(25,), dtype=np.float32
        )
        
        # Agent names
        self.agent_names = [f"agent_{i}" for i in range(n_agents)]
        
        # Initialize state variables
        self.current_step = 0
        self.agent_states = {}
        self.agent_phases = {}
        self.agent_phase_timers = {}
        self.traffic_patterns = {}
        
        # Initialize RNG
        if seed is not None:
            self.seed(seed)
        else:
            self._np_random = np.random.RandomState()
    
    def seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._np_random = np.random.RandomState(seed)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            observations: Dictionary of agent observations
            info: Additional information
        """
        if seed is not None:
            self.seed(seed)
        
        self.current_step = 0
        
        # Initialize agent states
        for agent in self.agent_names:
            self.agent_states[agent] = self._generate_initial_state()
            self.agent_phases[agent] = 0  # Start with phase 0
            self.agent_phase_timers[agent] = 0
            self.traffic_patterns[agent] = self._generate_traffic_pattern()
        
        observations = {agent: self._get_observation(agent) for agent in self.agent_names}
        info = self._get_info()
        
        return observations, info
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], 
                                                   Dict[str, float], 
                                                   Dict[str, bool], 
                                                   Dict[str, bool],
                                                   Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            actions: Dictionary of actions per agent
            
        Returns:
            observations: Next observations per agent
            rewards: Rewards per agent
            terminated: Termination flags per agent
            truncated: Truncation flags per agent
            info: Additional information
        """
        self.current_step += 1
        
        # Update agent states based on actions
        for agent, action in actions.items():
            if agent in self.agent_names:
                self._update_agent_state(agent, action)
        
        # Generate observations and rewards
        observations = {agent: self._get_observation(agent) for agent in self.agent_names}
        rewards = {agent: self._calculate_reward(agent) for agent in self.agent_names}
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        terminated = {agent: done for agent in self.agent_names}
        terminated["__all__"] = done
        
        truncated = {agent: False for agent in self.agent_names}
        truncated["__all__"] = False
        
        info = self._get_info()
        
        return observations, rewards, terminated, truncated, info
    
    def _generate_initial_state(self) -> Dict[str, Any]:
        """Generate initial state for an agent."""
        return {
            'queue_lengths': self._np_random.randint(0, self.max_queue_length // 2, 4),
            'turning_props': self._generate_turning_proportions(),
            'neighbor_queues': self._np_random.randint(0, self.max_queue_length // 2, 4),
            'vehicles_passed': 0,
            'waiting_times': [],
            'phase_start_time': 0
        }
    
    def _generate_turning_proportions(self) -> np.ndarray:
        """Generate realistic turning proportions for each lane."""
        props = []
        for _ in range(4):  # For each lane (N, S, E, W)
            # Generate proportions that sum to 1
            raw_props = self._np_random.dirichlet([2, 1, 1])  # Bias toward straight
            props.extend(raw_props)
        return np.array(props)
    
    def _generate_traffic_pattern(self) -> Dict[str, Any]:
        """Generate traffic pattern parameters for an intersection."""
        return {
            'base_arrival_rate': self._np_random.uniform(0.1, 0.5),
            'rush_hour_multiplier': self._np_random.uniform(1.5, 3.0),
            'direction_bias': self._np_random.uniform(0.8, 1.2, 4)
        }
    
    def _update_agent_state(self, agent: str, action: int):
        """Update agent state based on action and traffic dynamics."""
        state = self.agent_states[agent]
        pattern = self.traffic_patterns[agent]
        
        # Update phase if it changed
        old_phase = self.agent_phases[agent]
        new_phase = action
        
        if old_phase != new_phase:
            self.agent_phases[agent] = new_phase
            self.agent_phase_timers[agent] = 0
            state['phase_start_time'] = self.current_step
        else:
            self.agent_phase_timers[agent] += self.decision_frequency
        
        # Simulate traffic flow based on current phase
        self._simulate_traffic_flow(agent, new_phase)
        
        # Update neighbor information (simplified)
        state['neighbor_queues'] = self._get_neighbor_queues(agent)
    
    def _simulate_traffic_flow(self, agent: str, phase: int):
        """Simulate traffic flow for current phase."""
        state = self.agent_states[agent]
        pattern = self.traffic_patterns[agent]
        
        # Calculate arrival rates (simplified)
        time_factor = self._get_time_factor()
        base_arrivals = pattern['base_arrival_rate'] * time_factor
        
        # Add new vehicles to queues
        for i in range(4):  # For each lane
            arrivals = self._np_random.poisson(
                base_arrivals * pattern['direction_bias'][i]
            )
            state['queue_lengths'][i] = min(
                state['queue_lengths'][i] + arrivals,
                self.max_queue_length
            )
        
        # Process vehicles based on current phase
        vehicles_processed = self._process_vehicles_by_phase(state, phase)
        state['vehicles_passed'] += vehicles_processed
        
        # Update waiting times (simplified)
        avg_wait = np.mean(state['queue_lengths']) * 2  # Rough estimate
        state['waiting_times'].append(avg_wait)
    
    def _process_vehicles_by_phase(self, state: Dict, phase: int) -> int:
        """Process vehicles through intersection based on traffic light phase."""
        processed = 0
        
        # Define which lanes are green for each phase
        green_lanes = self._get_green_lanes(phase)
        
        for lane_idx in green_lanes:
            if state['queue_lengths'][lane_idx] > 0:
                # Process some vehicles (simplified model)
                to_process = min(
                    state['queue_lengths'][lane_idx],
                    self._np_random.randint(1, 4)  # 1-3 vehicles per decision
                )
                state['queue_lengths'][lane_idx] -= to_process
                processed += to_process
        
        return processed
    
    def _get_green_lanes(self, phase: int) -> List[int]:
        """Get which lanes have green light for given phase."""
        if phase == 0:  # NS straight/right
            return [0, 1]  # North, South lanes
        elif phase == 1:  # NS left
            return [0, 1]  # North, South lanes (left turns)
        elif phase == 2:  # EW straight/right
            return [2, 3]  # East, West lanes
        elif phase == 3:  # EW left
            return [2, 3]  # East, West lanes (left turns)
        else:
            return []
    
    def _get_time_factor(self) -> float:
        """Get time-based traffic factor (rush hour simulation)."""
        # Simulate rush hour patterns
        hour = (self.current_step * self.decision_frequency) // 3600
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            return 2.0
        elif 22 <= hour <= 6:  # Night
            return 0.3
        else:  # Regular hours
            return 1.0
    
    def _get_neighbor_queues(self, agent: str) -> np.ndarray:
        """Get simplified neighbor queue information."""
        # In a real implementation, this would access neighboring agents
        # For mock environment, generate realistic neighbor data
        return self._np_random.randint(0, self.max_queue_length // 3, 4)
    
    def _get_observation(self, agent: str) -> np.ndarray:
        """
        Get 25-dimensional observation for agent.
        
        State composition:
        - Queue lengths (4)
        - Turning proportions (12)  
        - Current phase one-hot (4)
        - Neighbor queues (4)
        - Time in current phase (1)
        """
        state = self.agent_states[agent]
        
        # Normalize queue lengths to [0, 1]
        norm_queues = state['queue_lengths'] / self.max_queue_length
        
        # Turning proportions (already normalized)
        turning_props = state['turning_props']
        
        # One-hot encode current phase
        phase_onehot = np.zeros(4)
        phase_onehot[self.agent_phases[agent]] = 1.0
        
        # Normalize neighbor queues
        norm_neighbors = state['neighbor_queues'] / self.max_queue_length
        
        # Normalize time in phase
        time_in_phase = min(self.agent_phase_timers[agent] / 60.0, 1.0)  # Max 1 minute
        
        # Concatenate all features (should be 25-dimensional)
        observation = np.concatenate([
            norm_queues,           # 4
            turning_props,         # 12
            phase_onehot,         # 4
            norm_neighbors,       # 4
            [time_in_phase]       # 1
        ]).astype(np.float32)
        
        assert len(observation) == 25, f"Observation should be 25-dim, got {len(observation)}"
        
        return observation
    
    def _calculate_reward(self, agent: str) -> float:
        """
        Calculate reward based on stopped vehicle ratio.
        R = -alpha * stopped_vehicle_ratio
        """
        state = self.agent_states[agent]
        
        # Calculate stopped vehicle ratio
        total_queued = np.sum(state['queue_lengths'])
        total_capacity = 4 * self.max_queue_length
        stopped_ratio = total_queued / total_capacity if total_capacity > 0 else 0
        
        # Reward is negative stopped ratio (lower ratio = higher reward)
        alpha = 1.0  # Can be configured
        reward = -alpha * stopped_ratio
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional environment information."""
        return {
            'step': self.current_step,
            'time': self.current_step * self.decision_frequency,
            'episode_progress': self.current_step / self.max_steps
        }
    
    def render(self, mode: str = 'human'):
        """Render the environment (simplified for mock)."""
        if mode == 'human':
            print(f"Step {self.current_step}/{self.max_steps}")
            for agent in self.agent_names:
                state = self.agent_states[agent]
                phase = self.agent_phases[agent]
                queues = state['queue_lengths']
                print(f"{agent}: Phase {phase}, Queues {queues}")
    
    def close(self):
        """Clean up environment resources."""
        pass
