# MARLO: Multi-Agent Reinforcement Learning for Traffic Optimization

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

*A complete end-to-end pipeline for multi-agent reinforcement learning applied to traffic signal optimization*

</div>

## 🚦 Overview

MARLO implements a comprehensive multi-agent reinforcement learning system for optimizing traffic signal control at urban intersections. The system uses **Centralized Training with Decentralized Execution (CTDE)** approach with **offline learning** from pre-collected traffic data.

## ⚡ TL;DR Quick Guide

Follow these four steps end-to-end (CPU-safe):

1) Generate datasets
```bash
python -m src.marlo.cli collect-data --config configs/base.yaml --episodes 200 --dataset-type synthetic --policy random
```

2) Train DQN (writes to `experiments/experiment_01/`)
```bash
python -m src.marlo.cli train --config configs/experiment_01.yaml --experiment-name experiment_01 --device cpu
```

3) Evaluate (creates `experiments/experiment_01/eval_results.json`)
```bash
python -m src.marlo.cli eval --config configs/experiment_01.yaml \
  --checkpoint experiments/experiment_01/checkpoints/best_model.pt \
  --episodes 30 --deterministic --output-file experiments/experiment_01/eval_results.json \
  --device cpu
```

4) Visualize the full dashboard (learning curve, metrics, heatmaps, action distributions)
```bash
python -m src.marlo.cli visualize --experiment experiment_01
```

Notes:
- If your PyTorch build has no CUDA, always add `--device cpu` to train/eval.
- The dashboard shows evaluation charts only after step 3 (evaluation).

### Key Features

- **Multi-Agent System**: Independent agents for each intersection with coordinated training
- **CTDE Architecture**: Centralized training with decentralized execution for scalable deployment
- **Offline Learning**: Train on pre-collected datasets without real-world risks
- **Comprehensive Metrics**: Primary and supporting metrics for thorough evaluation
- **Complete Pipeline**: From data generation to training, evaluation, and visualization

## 🎯 Problem Definition

**Research Goal**: Minimize the average ratio of vehicles waiting at red lights to total vehicles that might occupy each given lane, using multi-agent reinforcement learning.

### Performance Metrics

#### Primary Metric
- **Average Stopped Vehicle Ratio**: Ratio of vehicles waiting at red lights to total vehicle capacity per lane (lower is better)

#### Supporting Metrics
- **Average Waiting Time**: Duration vehicles spend idle at red lights (lower is better)
- **Average Queue Length**: Mean number of vehicles lined up per lane (lower is better)
- **Throughput**: Total vehicles passing through intersections per minute (higher is better)

## 🏗️ Architecture

### Agent Design
- **State Space**: 25-dimensional vector per agent including:
  - Queue lengths (4 values: N, S, E, W lanes)
  - Turning proportions (12 values: 4 lanes × 3 directions)
  - Current signal phase (4 values: one-hot encoded)
  - Neighbor queue information (4 values)
  - Time in current phase (1 value)

- **Action Space**: 4 discrete actions representing traffic light phases:
  - 0: North-South straight/right green, East-West red
  - 1: North-South left green, East-West red
  - 2: East-West straight/right green, North-South red
  - 3: East-West left green, North-South red

- **Reward Function**: `R = -α × stopped_vehicle_ratio`

### Learning Algorithm
- **Base Algorithm**: Deep Q-Network (DQN) with experience replay
- **Multi-Agent Coordination**: CTDE with central critic for shared learning
- **Training Mode**: Offline learning from synthetic and semi-synthetic datasets

## 📁 Project Structure

```
marlo-traffic/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── configs/                      # Configuration files
│   ├── base.yaml                # Base training configuration
│   └── experiment_01.yaml       # Experiment-specific overrides
├── src/marlo/                   # Main source code
│   ├── __init__.py
│   ├── cli.py                   # Command-line interface
│   ├── utils/                   # Utility functions
│   │   ├── logger.py           # Logging utilities
│   │   ├── metrics.py          # Traffic metrics calculation
│   │   └── seed.py             # Reproducibility utilities
│   ├── envs/                    # Environment implementations
│   │   └── mock_env.py         # Mock traffic environment
│   ├── agents/                  # RL agents
│   │   ├── base_agent.py       # Abstract base agent
│   │   ├── dqn_agent.py        # DQN implementation
│   │   └── central_critic.py   # Central critic for CTDE
│   ├── data/                    # Dataset handling
│   │   ├── dataset_builder.py  # Dataset generation
│   │   └── loader.py           # Data loading utilities
│   ├── training/                # Training and evaluation
│   │   ├── trainer.py          # Offline training pipeline
│   │   └── eval_runner.py      # Evaluation system
│   └── viz/                     # Visualization
│       └── plotter.py          # Plotting utilities
├── datasets/                    # Generated datasets
│   ├── synthetic/              # Controlled traffic scenarios
│   └── semi_synthetic/         # Realistic traffic patterns
├── experiments/                 # Experiment results
│   └── experiment_01/          # Example experiment
│       ├── checkpoints/        # Saved models
│       ├── logs/              # Training logs
│       └── plots/             # Generated visualizations
├── scripts/                     # Utility scripts
│   └── generate_dataset.py    # Standalone dataset generation
└── tests/                      # Unit tests
    ├── test_metrics.py         # Test metrics calculations
    ├── test_mock_env.py        # Test environment
    └── test_data_loader.py     # Test data loading
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/marlo-traffic.git
cd marlo-traffic

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
# Generate synthetic dataset for training
python -m src.marlo.cli collect-data --config configs/base.yaml --episodes 100

# Or use the standalone script
python scripts/generate_dataset.py
```

### 3. Train Agents

```bash
# Train using base configuration
python -m src.marlo.cli train --config configs/base.yaml

# Train with custom experiment name
python -m src.marlo.cli train --config configs/base.yaml --experiment-name my_experiment
```

### 4. Evaluate Performance

```bash
# Evaluate trained model
python -m src.marlo.cli eval --config configs/base.yaml \
  --checkpoint experiments/experiment_01/checkpoints/best_model.pt \
  --episodes 100

# Compare with baseline
python -m src.marlo.cli eval --config configs/base.yaml \
  --checkpoint experiments/experiment_01/checkpoints/best_model.pt \
  --baseline-comparison random
```

### 5. Generate Visualizations

```bash
# Create complete dashboard
python -m src.marlo.cli visualize --experiment experiment_01

# Generate specific plots
python -m src.marlo.cli visualize --experiment experiment_01 \
  --plots learning metrics heatmap
```

## 📊 Visualization

The system generates comprehensive visualizations:

1. **Learning Curve**: Training reward progression over episodes
2. **Metrics Progression**: Evolution of all 4 traffic metrics during evaluation
3. **Turn Distribution**: Pie charts showing traffic light phase usage per intersection
4. **Queue Length Heatmap**: Queue lengths across episodes and lanes

## ⚙️ Configuration

Configuration files use YAML format. Key parameters:

```yaml
# Agent configuration
agent:
  type: dqn
  learning_rate: 0.001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 10000

# Training configuration
training:
  algorithm: offline
  episodes: 1000
  batch_size: 64
  
# CTDE settings
ctde:
  central_critic: true
  critic_lr: 0.0005
  
# Environment settings
env:
  n_agents: 4
  episode_length: 3600  # 1 hour
  decision_frequency: 10  # seconds
```

## 🧪 Testing

Run the test suite to verify installation:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_metrics.py -v
python -m pytest tests/test_mock_env.py -v
python -m pytest tests/test_data_loader.py -v
```

## 📈 Results and Analysis

### Expected Outcomes

- **Primary Metric Improvement**: Significant reduction in stopped vehicle ratio compared to baseline policies
- **Supporting Metrics**: Improvements in waiting time, queue length, and throughput
- **Learning Convergence**: Stable learning curves showing policy improvement over episodes

### Baseline Comparisons

The system supports comparison with standard baseline policies:
- **Random Policy**: Random phase selection
- **Fixed-Time Policy**: Traditional fixed-duration signals

## 🔧 Extending the System

### Adding New Environments

1. Inherit from `BaseAgent` in `src/marlo/agents/base_agent.py`
2. Implement required methods: `act()`, `learn()`, `save()`, `load()`
3. Register in configuration files

### Custom Metrics

1. Add metric functions to `src/marlo/utils/metrics.py`
2. Update `MetricsTracker` class for automatic tracking
3. Modify evaluation pipeline to include new metrics

### New Visualizations

1. Add plotting functions to `src/marlo/viz/plotter.py`
2. Integrate with CLI visualization commands
3. Update dashboard generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635)
- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [Deep Q-Network](https://www.nature.com/articles/nature14236)

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact [your-email@example.com](mailto:your-email@example.com).

---

<div align="center">

**MARLO** - Making urban intersections smarter, one signal at a time 🚦

</div>
