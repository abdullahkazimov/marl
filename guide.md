I’ll give you a **complete folder/file structure** (no SUMO), with **short explanations after each file/folder** so you know *why* it exists.
This will give you a **clean, small, and professional skeleton** to build multi-agent RL for traffic signals. Dataset files are also included, with realistic structures you can mock now and later replace with SUMO outputs.

---

# 📂 Project Structure

```
marlo-traffic/                           # project root
├─ README.md                             # explain how to run/train/eval the project
├─ requirements.txt                      # Python dependencies (gymnasium, numpy, torch, matplotlib, etc.)
├─ configs/                              # keep hyperparameters, paths, experiment configs
│  ├─ base.yaml                          # base training config (alpha, decision freq, episode length, etc.)
│  └─ experiment_01.yaml                 # override configs for specific runs
│
├─ src/                                  # all source code (Python package style)
│  └─ marlo/
│     ├─ __init__.py                     # makes this a package
│     ├─ cli.py                          # simple CLI: train / eval / collect-data
│     │
│     ├─ utils/                          # helper functions
│     │  ├─ __init__.py
│     │  ├─ logger.py                    # unified logging & checkpoint naming
│     │  ├─ metrics.py                   # reward + evaluation metrics (stopped ratio, wait time, throughput)
│     │  └─ seed.py                      # set random seeds for reproducibility
│     │
│     ├─ envs/                           # environments (mock now, SUMO later)
│     │  ├─ __init__.py
│     │  └─ mock_env.py                  # minimal fake Gym env producing 25-dim states & random rewards
│     │
│     ├─ data/                           # dataset generation + loading
│     │  ├─ __init__.py
│     │  ├─ dataset_builder.py           # creates offline dataset from env rollouts (mocked now)
│     │  └─ loader.py                    # loads dataset into PyTorch DataLoader / replay buffer
│     │
│     ├─ agents/                         # agents for multi-agent RL
│     │  ├─ __init__.py
│     │  ├─ base_agent.py                # abstract base (act, learn, save, load)
│     │  ├─ dqn_agent.py                 # simple DQN implementation (offline-capable)
│     │  └─ central_critic.py            # placeholder for CTDE training (central critic model)
│     │
│     ├─ training/                       # training & evaluation logic
│     │  ├─ __init__.py
│     │  ├─ trainer.py                   # training loop (offline dataset → agent update)
│     │  └─ eval_runner.py               # run eval episodes & compute metrics
│     │
│     └─ viz/                            # visualization tools
│        ├─ __init__.py
│        └─ plotter.py                   # line plots, heatmaps, pie charts for results
│
├─ datasets/                             # offline datasets (mock for now, replace with SUMO later)
│  ├─ README.md                          # describes dataset format & fields
│  ├─ synthetic/                         # controlled datasets (balanced traffic, no noise)
│  │  ├─ dataset_01.npz                  # compressed numpy: dict-of-arrays (obs, actions, rewards, next_obs, done)
│  │  └─ dataset_02.npz
│  └─ semi_synthetic/                    # more realistic datasets (random turn ratios, irregular queues)
│     ├─ rush_hour_01.npz
│     └─ off_peak_01.npz
│
├─ experiments/                          # logs, checkpoints, plots per experiment
│  └─ experiment_01/
│     ├─ checkpoints/                    # saved models
│     ├─ logs/                           # training logs (JSON, CSV, txt)
│     └─ plots/                          # generated figures (learning curves, heatmaps)
│
└─ tests/                                # unit tests
   ├─ test_metrics.py                    # test correctness of reward & metrics functions
   ├─ test_mock_env.py                   # test env produces valid 25-dim states & rewards
   └─ test_data_loader.py                # test dataset loader returns correct batches
```

---

# 📂 Dataset File Structure

Since SUMO isn’t available yet, you can **mock datasets** in `.npz` (NumPy compressed) or `.parquet`.
Each dataset stores tuples `(s, a, r, s’, done)` for **multiple agents & timesteps**.

### Example: `dataset_01.npz`

```python
{
  "observations": np.ndarray,   # shape = (episodes, steps, agents, 25)
  "actions": np.ndarray,        # shape = (episodes, steps, agents)
  "rewards": np.ndarray,        # shape = (episodes, steps, agents)
  "next_observations": np.ndarray,  # shape = (episodes, steps, agents, 25)
  "dones": np.ndarray           # shape = (episodes, steps, agents)
}
```

* **observations**: each agent’s 25-dim state vector
* **actions**: chosen phase (0–3)
* **rewards**: scalar reward `-alpha * stopped_vehicle_ratio`
* **next\_observations**: post-step states
* **dones**: episode termination flags

This structure works for both **synthetic** and **semi-synthetic** traffic. Later, when SUMO is integrated, the dataset builder will populate it with real transitions.

---

✅ With this skeleton, you can already:

* Create **mock datasets** (random numbers shaped like real data).
* Train/test agents offline on these fake datasets.
* Develop metrics & plotting fully.
* Drop SUMO in later without breaking structure.

---

Here is a short preview that you should follow while coding:

---

```
marlo-traffic/
├─ README.md
```

````markdown
# MARLO Traffic
Multi-agent RL traffic signal control (mock env now, SUMO later).

## Run
```bash
python -m src.marlo.cli train --config configs/base.yaml
````

```

---

```

├─ requirements.txt

````
```text
gymnasium
numpy
torch
matplotlib
pyyaml
````

---

```
├─ configs/base.yaml
```

```yaml
agent:
  type: dqn
  learning_rate: 0.001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 10000

training:
  episodes: 10
  batch_size: 32
```

```
├─ configs/experiment_01.yaml
```

```yaml
# override base.yaml
agent:
  learning_rate: 0.0005
training:
  episodes: 20
```

---

```
├─ src/marlo/__init__.py
```

```python
# make marlo a package
```

```
├─ src/marlo/cli.py
```

```python
import argparse
from marlo.training.trainer import Trainer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval"])
    p.add_argument("--config", default="configs/base.yaml")
    args = p.parse_args()
    if args.mode == "train":
        Trainer(config_path=args.config).train()
    else:
        Trainer(config_path=args.config).evaluate()

if __name__ == "__main__":
    main()
```

---

```
├─ src/marlo/utils/logger.py
```

```python
import logging

def get_logger(name="marlo"):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger(name)
```

```
├─ src/marlo/utils/metrics.py
```

```python
import numpy as np

def avg_wait_time(rewards):
    return -np.mean(rewards)

def stopped_vehicle_ratio(states):
    return np.mean([s[0] for s in states])  # mock: first dim = stopped ratio
```

```
├─ src/marlo/utils/seed.py
```

```python
import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

---

```
├─ src/marlo/envs/mock_env.py
```

```python
import gymnasium as gym
import numpy as np

class MockTrafficEnv(gym.Env):
    def __init__(self, n_agents=2):
        self.n_agents = n_agents
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 1, (25,), np.float32)

    def reset(self, *, seed=None, options=None):
        obs = {f"agent_{i}": self._rand_obs() for i in range(self.n_agents)}
        return obs, {}

    def step(self, actions):
        obs = {k: self._rand_obs() for k in actions}
        rewards = {k: np.random.randn() for k in actions}
        dones = {k: False for k in actions}
        dones["__all__"] = False
        return obs, rewards, dones, {}

    def _rand_obs(self):
        return self.observation_space.sample()
```

---

```
├─ src/marlo/data/dataset_builder.py
```

```python
import numpy as np

def build_dataset(env, episodes=5):
    data = {"obs": [], "actions": [], "rewards": [], "next_obs": [], "done": []}
    for _ in range(episodes):
        obs, _ = env.reset()
        done = {"__all__": False}
        while not done["__all__"]:
            actions = {k: env.action_space.sample() for k in obs}
            next_obs, rewards, done, _ = env.step(actions)
            data["obs"].append(obs)
            data["actions"].append(actions)
            data["rewards"].append(rewards)
            data["next_obs"].append(next_obs)
            data["done"].append(done)
            obs = next_obs
    np.savez("datasets/synthetic/dataset_01.npz", **data)
```

```
├─ src/marlo/data/loader.py
```

```python
import numpy as np
import torch

def load_dataset(path):
    arrs = np.load(path, allow_pickle=True)
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in arrs.items()}
```

---

```
├─ src/marlo/agents/base_agent.py
```

```python
class BaseAgent:
    def act(self, state): raise NotImplementedError
    def learn(self, batch): raise NotImplementedError
    def save(self, path): pass
    def load(self, path): pass
```

```
├─ src/marlo/agents/dqn_agent.py
```

```python
import torch, torch.nn as nn, torch.optim as optim
from .base_agent import BaseAgent

class QNet(nn.Module):
    def __init__(self, in_dim=25, out_dim=4):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, out_dim))
    def forward(self, x): return self.layers(x)

class DQNAgent(BaseAgent):
    def __init__(self, lr=1e-3):
        self.q = QNet()
        self.opt = optim.Adam(self.q.parameters(), lr=lr)

    def act(self, state):
        with torch.no_grad():
            return self.q(torch.tensor(state)).argmax().item()

    def learn(self, batch): pass  # stub
```

```
├─ src/marlo/agents/central_critic.py
```

```python
# placeholder for central critic training
class CentralCritic:
    def __init__(self): pass
```

---

```
├─ src/marlo/training/trainer.py
```

```python
import yaml
from marlo.envs.mock_env import MockTrafficEnv
from marlo.agents.dqn_agent import DQNAgent
from marlo.utils.logger import get_logger

class Trainer:
    def __init__(self, config_path):
        self.config = yaml.safe_load(open(config_path))
        self.env = MockTrafficEnv()
        self.agent = DQNAgent(lr=self.config["agent"]["learning_rate"])
        self.log = get_logger()

    def train(self):
        self.log.info("Training started")
        for ep in range(self.config["training"]["episodes"]):
            obs, _ = self.env.reset()
            self.log.info(f"Episode {ep} obs={list(obs.values())[0].shape}")

    def evaluate(self):
        self.log.info("Evaluating...")
```

```
├─ src/marlo/training/eval_runner.py
```

```python
def run_eval(env, agent, episodes=2):
    results = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = {"__all__": False}
        while not done["__all__"]:
            acts = {k: agent.act(v) for k, v in obs.items()}
            obs, r, done, _ = env.step(acts)
            results.append(sum(r.values()))
    return results
```

---

```
├─ src/marlo/viz/plotter.py
```

```python
import matplotlib.pyplot as plt

def plot_curve(values, path="curve.png"):
    plt.plot(values)
    plt.savefig(path)
```

---

```
├─ datasets/README.md
```

```markdown
Datasets store offline RL data.
Each `.npz` contains: obs, actions, rewards, next_obs, done.
```

---

```
├─ tests/test_metrics.py
```

```python
from marlo.utils.metrics import avg_wait_time

def test_avg_wait_time():
    rewards = [-1, -2, -3]
    assert round(avg_wait_time(rewards), 2) == 2.0
```

```
├─ tests/test_mock_env.py
```

```python
from marlo.envs.mock_env import MockTrafficEnv

def test_env_shapes():
    env = MockTrafficEnv()
    obs, _ = env.reset()
    assert len(list(obs.values())[0]) == 25
```

```
├─ tests/test_data_loader.py
```

```python
from marlo.data.loader import load_dataset

def test_loader(tmp_path):
    import numpy as np
    np.savez(tmp_path/"mock.npz", obs=[1], actions=[0])
    data = load_dataset(tmp_path/"mock.npz")
    assert "obs" in data
```

---

Perfect 👌 — let’s add a **tiny dataset generator** so you can immediately run training/eval without touching SUMO.

I’ll place it in `scripts/` (to keep project root clean) and wire it so you can just run one command.

---

### New file:

```
scripts/generate_dataset.py
```

```python
"""
Generate a small synthetic offline dataset using MockTrafficEnv.
Saves dataset in datasets/synthetic/dataset_01.npz
"""

import os
import numpy as np
from src.marlo.envs.mock_env import MockTrafficEnv

def generate_dataset(path="datasets/synthetic/dataset_01.npz", episodes=3):
    env = MockTrafficEnv(n_agents=2)
    data = {"obs": [], "actions": [], "rewards": [], "next_obs": [], "done": []}

    for _ in range(episodes):
        obs, _ = env.reset()
        done = {"__all__": False}
        while not done["__all__"]:
            actions = {k: env.action_space.sample() for k in obs}
            next_obs, rewards, done, _ = env.step(actions)

            data["obs"].append(obs)
            data["actions"].append(actions)
            data["rewards"].append(rewards)
            data["next_obs"].append(next_obs)
            data["done"].append(done)

            obs = next_obs

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **data)
    print(f"✅ Dataset saved at {path}")

if __name__ == "__main__":
    generate_dataset()
```

---

### Update CLI (`src/marlo/cli.py`)

Add a `collect-data` mode:

```python
elif args.mode == "collect-data":
    from scripts.generate_dataset import generate_dataset
    generate_dataset()
```

---

### Usage

```bash
# generate toy dataset
python -m src.marlo.cli collect-data

# train using config
python -m src.marlo.cli train --config configs/base.yaml

# evaluate
python -m src.marlo.cli eval --config configs/base.yaml
```

---

Got it 👍 — let’s wire up the **offline dataset → replay buffer → training loop**.
I’ll keep it **tiny and clean**, so you can extend later with SUMO or more complex agents.

---

### 🔹 Update `src/marlo/data/loader.py`

```python
"""
Loader for offline datasets saved as .npz
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrafficDataset(Dataset):
    def __init__(self, path: str):
        raw = np.load(path, allow_pickle=True)
        self.obs = raw["obs"]
        self.actions = raw["actions"]
        self.rewards = raw["rewards"]
        self.next_obs = raw["next_obs"]
        self.done = raw["done"]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return {
            "obs": self.obs[idx].item(),          # dict per agent
            "actions": self.actions[idx].item(),  # dict per agent
            "rewards": self.rewards[idx].item(),  # dict per agent
            "next_obs": self.next_obs[idx].item(),
            "done": self.done[idx].item()
        }

def load_dataset(path: str, batch_size: int = 32):
    ds = TrafficDataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
```

---

### 🔹 Update `src/marlo/training/trainer.py`

```python
"""
Offline trainer: updates agents using pre-collected dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from src.marlo.data.loader import load_dataset
from src.marlo.agents.dqn_agent import DQNAgent
from src.marlo.utils.logger import get_logger

def train_offline(config, dataset_path):
    logger = get_logger()

    # single agent for now (extend to multi-agent easily)
    agent = DQNAgent(
        state_dim=25,
        action_dim=4,
        lr=config["agent"]["learning_rate"],
        gamma=config["agent"]["gamma"]
    )

    dataloader = load_dataset(dataset_path, batch_size=config["training"]["batch_size"])
    optimizer = optim.Adam(agent.q_net.parameters(), lr=config["agent"]["learning_rate"])
    criterion = nn.MSELoss()

    for epoch in range(config["training"]["epochs"]):
        total_loss = 0
        for batch in dataloader:
            # offline RL: pick one agent’s perspective (say A1)
            obs = torch.tensor(batch["obs"]["A1"], dtype=torch.float32)
            actions = torch.tensor(batch["actions"]["A1"], dtype=torch.long)
            rewards = torch.tensor(batch["rewards"]["A1"], dtype=torch.float32)
            next_obs = torch.tensor(batch["next_obs"]["A1"], dtype=torch.float32)
            done = torch.tensor(batch["done"]["__all__"], dtype=torch.float32)

            # compute Q-targets
            q_values = agent.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze()
            with torch.no_grad():
                max_next_q = agent.target_net(next_obs).max(1)[0]
                targets = rewards + (1 - done) * config["agent"]["gamma"] * max_next_q

            loss = criterion(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1} | Loss: {total_loss:.3f}")

    agent.save("experiments/experiment_01/checkpoints/final.pt")
    logger.info("✅ Training complete, model saved.")
```

---

### 🔹 Update `src/marlo/cli.py`

```python
elif args.mode == "train":
    from src.marlo.training.trainer import train_offline
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_offline(cfg, dataset_path="datasets/synthetic/dataset_01.npz")
```

---

### 🔹 Example `configs/base.yaml`

```yaml
agent:
  type: dqn
  learning_rate: 0.001
  gamma: 0.99

training:
  batch_size: 32
  epochs: 5
```

---

### ✅ Workflow Now

```bash
# Step 1: Generate dataset
python -m src.marlo.cli collect-data

# Step 2: Train offline from dataset
python -m src.marlo.cli train --config configs/base.yaml

# Step 3: (later) Evaluate agent
python -m src.marlo.cli eval --config configs/base.yaml
```

---

Perfect 🚦 let’s extend `eval_runner.py` so you can actually run **trained policies** in the mock env and compute your 4 metrics:

* **Stopped Vehicle Ratio** (primary)
* **Average Waiting Time**
* **Average Queue Length**
* **Throughput**

---

### 🔹 `src/marlo/training/eval_runner.py`

```python
"""
Evaluation runner for trained agents on MockTrafficEnv
Computes: stopped ratio, wait time, queue length, throughput
"""

import torch
import numpy as np
from src.marlo.envs.mock_env import MockTrafficEnv
from src.marlo.agents.dqn_agent import DQNAgent
from src.marlo.utils.metrics import (
    stopped_vehicle_ratio,
    average_wait_time,
    average_queue_length,
    throughput,
)
from src.marlo.utils.logger import get_logger

def evaluate(config, model_path, episodes=3):
    logger = get_logger()
    env = MockTrafficEnv(n_agents=2)

    # Load agent (single-agent demo with A1’s perspective)
    agent = DQNAgent(
        state_dim=25,
        action_dim=4,
        lr=config["agent"]["learning_rate"],
        gamma=config["agent"]["gamma"]
    )
    agent.load(model_path)

    all_metrics = {"stopped_ratio": [], "wait_time": [], "queue_length": [], "throughput": []}

    for ep in range(episodes):
        obs, _ = env.reset()
        done = {"__all__": False}

        ep_rewards = []
        ep_queues = []
        ep_wait_times = []
        passed_cars = 0

        while not done["__all__"]:
            # choose action for A1, random for others
            state = torch.tensor(list(obs["A1"]), dtype=torch.float32).unsqueeze(0)
            action_A1 = agent.act(state)

            actions = {k: env.action_space.sample() for k in obs}
            actions["A1"] = action_A1

            next_obs, rewards, done, _ = env.step(actions)

            ep_rewards.append(rewards["A1"])
            ep_queues.append(sum(obs["A1"][:4]))  # first 4 entries = queue lengths
            ep_wait_times.append(np.mean(obs["A1"][:4]))  # proxy wait time
            passed_cars += np.random.randint(0, 3)  # mock throughput

            obs = next_obs

        # compute metrics for this episode
        all_metrics["stopped_ratio"].append(stopped_vehicle_ratio(ep_queues))
        all_metrics["wait_time"].append(average_wait_time(ep_wait_times))
        all_metrics["queue_length"].append(average_queue_length(ep_queues))
        all_metrics["throughput"].append(throughput(passed_cars))

        logger.info(f"[Eval] Ep {ep+1} | Metrics: "
                    f"Stopped={all_metrics['stopped_ratio'][-1]:.3f}, "
                    f"Wait={all_metrics['wait_time'][-1]:.3f}, "
                    f"Queue={all_metrics['queue_length'][-1]:.3f}, "
                    f"Throughput={all_metrics['throughput'][-1]}")

    return all_metrics
```

---

### 🔹 `src/marlo/utils/metrics.py`

```python
"""
Metric functions for evaluation
"""

import numpy as np

def stopped_vehicle_ratio(queues):
    total_capacity = len(queues) * 10  # assume lane cap=10
    return np.mean(queues) / total_capacity

def average_wait_time(wait_times):
    return float(np.mean(wait_times))

def average_queue_length(queues):
    return float(np.mean(queues))

def throughput(passed_cars):
    return passed_cars  # total cars passed
```

---

### 🔹 Update `src/marlo/cli.py`

```python
elif args.mode == "eval":
    from src.marlo.training.eval_runner import evaluate
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    metrics = evaluate(cfg, "experiments/experiment_01/checkpoints/final.pt")
    print("✅ Evaluation complete. Metrics:", metrics)
```

---

### ✅ Workflow

```bash
# Step 1: Generate dataset
python -m src.marlo.cli collect-data

# Step 2: Train offline
python -m src.marlo.cli train --config configs/base.yaml

# Step 3: Evaluate trained agent
python -m src.marlo.cli eval --config configs/base.yaml
```

---

This will print episode-wise metrics and return a dictionary of averages.
Later, `viz/plotter.py` can take this output and make learning curves, heatmaps, etc.

---