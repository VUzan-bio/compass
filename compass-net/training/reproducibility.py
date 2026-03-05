"""Reproducibility utilities for deterministic training.

Sets seeds for all random number generators and configures PyTorch
for deterministic operation. Required for credible ablation tables --
results must be reproducible across runs.

Usage:
    from compass_ml.training.reproducibility import seed_everything
    seed_everything(42)
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Configures: Python random, numpy, PyTorch (CPU + CUDA), and
    cuDNN deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN operations (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For torch >= 1.8: warn on nondeterministic ops
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch versions don't support warn_only
            pass
