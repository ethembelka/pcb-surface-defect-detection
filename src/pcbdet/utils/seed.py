"""Seed fixing for reproducibility.

Every run (split, training, evaluation) calls ``set_seed`` first so that the same config
produces the same result again. Note: Ultralytics also takes its own ``seed`` argument; the
trainer passes the config seed there as well.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Fix Python, NumPy and (if installed) PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is not needed for data-only steps such as the split
        pass
