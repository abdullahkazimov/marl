# Datasets for MARLO Traffic Optimization

This directory contains offline datasets for training multi-agent traffic signal controllers.

## Dataset Types

### Synthetic Datasets (`synthetic/`)
- **Controlled and idealized traffic scenarios**
- No collisions, balanced flow, minimal external disruptions
- Designed for baseline agent behavior testing in stable conditions
- Generated using simple traffic flow models

### Semi-Synthetic Datasets (`semi_synthetic/`)
- **Realistic traffic patterns with temporal variations**
- Incorporates rush hour patterns, irregular queues, varied turning ratios
- Simulates real-world traffic irregularities and driver behavior
- More challenging scenarios for robust policy learning

## Dataset Format

Each dataset is stored as a compressed NumPy archive (`.npz`) with the following structure:

```python
{
  "observations": np.ndarray,      # shape = (transitions, 25)
  "actions": np.ndarray,           # shape = (transitions,)
  "rewards": np.ndarray,           # shape = (transitions,)
  "next_observations": np.ndarray, # shape = (transitions, 25)
  "dones": np.ndarray,             # shape = (transitions,)
  "episode_ids": np.ndarray,       # shape = (transitions,)
  "timesteps": np.ndarray,         # shape = (transitions,)
  "agent_ids": np.ndarray          # shape = (transitions,)
}
```

### Field Descriptions

- **observations**: 25-dimensional state vectors for each agent
- **actions**: Selected traffic light phases (0-3)
- **rewards**: Scalar rewards based on stopped vehicle ratio
- **next_observations**: Post-action state vectors
- **dones**: Episode termination flags
- **episode_ids**: Episode identifiers for transition grouping
- **timesteps**: Time step within each episode
- **agent_ids**: Agent identifiers (e.g., "agent_0", "agent_1")

## State Vector Components (25-dimensional)

1. **Queue Lengths** (4 values): Number of vehicles waiting in each lane [N, S, E, W]
2. **Turning Proportions** (12 values): For each lane, proportion of vehicles going [straight, left, right]
3. **Current Signal Phase** (4 values): One-hot encoded current traffic light phase
4. **Neighbor Queue Lengths** (4 values): Queue lengths from adjacent intersections
5. **Time in Phase** (1 value): Normalized time spent in current phase

## Traffic Light Phases

- **Phase 0**: North-South straight/right green, East-West red
- **Phase 1**: North-South left green, East-West red
- **Phase 2**: East-West straight/right green, North-South red
- **Phase 3**: East-West left green, North-South red

## Reward Function

All datasets use the reward function specified in the research:

```
R = -α × stopped_vehicle_ratio
```

Where:
- `α` is a tunable hyperparameter (typically 1.0)
- `stopped_vehicle_ratio` is the primary performance metric

## Dataset Generation

### Generating Synthetic Data

```bash
python -m src.marlo.cli collect-data \
  --config configs/base.yaml \
  --dataset-type synthetic \
  --episodes 100 \
  --policy random
```

### Generating Semi-Synthetic Data

```bash
python -m src.marlo.cli collect-data \
  --config configs/base.yaml \
  --dataset-type semi-synthetic \
  --episodes 100 \
  --policy adaptive
```

### Available Policies

- **random**: Random phase selection
- **fixed_time**: Traditional fixed-duration signals
- **greedy**: Simple heuristic based on queue lengths
- **adaptive**: Sophisticated policy considering turning proportions

## Dataset Statistics

Each dataset includes a companion metadata file (`*_metadata.json`) containing:

```json
{
  "dataset_info": {
    "type": "synthetic",
    "creation_date": "2024-01-01T00:00:00",
    "episodes": 100,
    "policy": "random",
    "n_agents": 4,
    "episode_length": 3600,
    "decision_frequency": 10
  },
  "statistics": {
    "total_transitions": 14400,
    "avg_episode_reward": {
      "mean": -2.45,
      "std": 1.23,
      "min": -5.67,
      "max": -0.12
    },
    "avg_queue_length": {
      "mean": 4.56,
      "std": 2.34,
      "min": 0.0,
      "max": 18.9
    }
  }
}
```

## Loading Datasets

### Python API

```python
from src.marlo.data.loader import TrafficDataset, MultiAgentDataLoader

# Load single dataset
dataset = TrafficDataset("datasets/synthetic/dataset_01.npz")

# Load multiple datasets
loader = MultiAgentDataLoader([
    "datasets/synthetic/dataset_01.npz",
    "datasets/semi_synthetic/rush_hour_01.npz"
], batch_size=64)

# Iterate through data
for batch in loader:
    observations = batch['observations']
    actions = batch['actions']
    # ... training code
```

### Dataset Analysis

```python
# Get dataset statistics
stats = dataset.get_state_stats()
action_dist = dataset.get_action_distribution()

# Filter by specific agents
agent_dataset = TrafficDataset(
    "dataset.npz", 
    agent_filter=["agent_0", "agent_1"]
)
```

## Best Practices

### For Training
1. **Combine Multiple Datasets**: Use both synthetic and semi-synthetic data for robust training
2. **Balance Data**: Ensure good representation of different traffic conditions
3. **Preprocessing**: Consider reward normalization for stable training

### For Evaluation
1. **Hold-out Sets**: Reserve some datasets for evaluation only
2. **Cross-validation**: Test on different traffic patterns than training
3. **Baseline Comparison**: Compare against simple policy datasets

## Storage Requirements

Typical dataset sizes:
- **100 episodes, 4 agents**: ~50MB compressed
- **1000 episodes, 4 agents**: ~500MB compressed
- Compression ratio: ~70% reduction from raw arrays

## Future Extensions

When integrating with SUMO (Simulation of Urban MObility):
1. Replace mock environment data with SUMO outputs
2. Maintain same dataset structure for compatibility
3. Add real-world traffic pattern support
4. Include vehicle type information
5. Support for irregular intersection geometries

---

**Note**: Mock datasets are used for development and testing. For production deployment, replace with datasets generated from realistic traffic simulation tools like SUMO.
