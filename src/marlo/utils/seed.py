"""
Seed utilities for reproducible experiments.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducible experiments.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
