"""Differentiable Spearman rank correlation.

Primary: torchsort (Blondel et al., ICML 2020) -- exact differentiable
sorting/ranking via projections onto the permutahedron with isotonic
regression. O(n log n) complexity.

Fallback: sigmoid-based soft ranking when torchsort is not installed.

The regularization_strength parameter controls the smoothness of the
soft ranking approximation:
    - Higher = smoother (more gradient signal, less accurate ranks)
    - Lower  = sharper (more accurate ranks, risk of vanishing gradients)
    - Anneal from 1.0 -> 0.1 over training for curriculum effect.

References:
    Blondel, Teboul, Berthet, Djolonga. "Fast Differentiable Sorting
    and Ranking." ICML 2020. arXiv:2002.08871.
    PyTorch implementation: https://github.com/teddykoker/torchsort
"""

from __future__ import annotations

import torch

_HAS_TORCHSORT = False
try:
    import torchsort as _torchsort
    _HAS_TORCHSORT = True
except ImportError:
    pass


def differentiable_spearman(
    pred: torch.Tensor,
    target: torch.Tensor,
    regularization_strength: float = 1.0,
) -> torch.Tensor:
    """Compute differentiable Spearman rho.

    Args:
        pred:   (N,) predicted values.
        target: (N,) ground truth values.
        regularization_strength: smoothness parameter. Anneal 1.0 -> 0.1.

    Returns:
        Scalar Spearman rho (differentiable, on same device as inputs).
    """
    if _HAS_TORCHSORT:
        return _spearman_torchsort(pred, target, regularization_strength)
    return _spearman_sigmoid(pred, target, regularization_strength)


def _spearman_torchsort(
    pred: torch.Tensor,
    target: torch.Tensor,
    regularization_strength: float,
) -> torch.Tensor:
    """Exact differentiable Spearman via torchsort."""
    # torchsort expects (batch, n)
    pred_2d = pred.unsqueeze(0)
    target_2d = target.unsqueeze(0)

    pred_ranks = _torchsort.soft_rank(
        pred_2d, regularization_strength=regularization_strength,
    )
    target_ranks = _torchsort.soft_rank(
        target_2d, regularization_strength=regularization_strength,
    )

    pred_ranks = pred_ranks - pred_ranks.mean()
    target_ranks = target_ranks - target_ranks.mean()

    pred_norm = pred_ranks / (pred_ranks.norm() + 1e-8)
    target_norm = target_ranks / (target_ranks.norm() + 1e-8)

    return (pred_norm * target_norm).sum()


def _spearman_sigmoid(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Fallback: sigmoid-based soft ranking."""
    def soft_rank(x: torch.Tensor) -> torch.Tensor:
        pairwise = x.unsqueeze(-1) - x.unsqueeze(-2)
        return torch.sigmoid(pairwise / max(temperature, 0.01)).sum(dim=-1)

    pr = soft_rank(pred)
    tr = soft_rank(target)
    pr_c = pr - pr.mean()
    tr_c = tr - tr.mean()
    return (pr_c * tr_c).sum() / (
        torch.sqrt((pr_c ** 2).sum() * (tr_c ** 2).sum()) + 1e-8
    )
