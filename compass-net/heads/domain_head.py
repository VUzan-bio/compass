"""Domain-adversarial head with gradient reversal layer (GRL).

The domain head predicts which dataset a sample came from. The GRL reverses
gradients during backpropagation, forcing the shared encoder to produce
features that are discriminative for efficiency prediction but invariant
to the source domain (dataset / experimental batch).

This addresses the distribution shift problem: the encoder cannot rely on
batch-specific correlations because the adversarial head penalises any
domain-discriminative signal in the learned representations.

Reference:
    Ganin et al. "Domain-Adversarial Training of Neural Networks"
    JMLR 2016, 17(59):1-35. arXiv:1505.07818.

Lambda schedule (Ganin et al. Eq. 8):
    lambda(p) = 2 / (1 + exp(-10 * p)) - 1
    where p = training_progress in [0, 1]

    At p=0:   lambda=0   (no adaptation, let encoder learn task features)
    At p=0.5: lambda~0.73
    At p=1.0: lambda~1.0 (full domain invariance pressure)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradientReversal(Function):
    """Gradient Reversal Layer (GRL).

    Forward: identity.
    Backward: multiply gradient by -lambda (reverse direction).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_val = 0.0

    def set_lambda(self, val: float) -> None:
        self.lambda_val = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversal.apply(x, self.lambda_val)


class DomainHead(nn.Module):
    """Domain discriminator with gradient reversal.

    Architecture: GRL -> Linear -> ReLU -> Dropout -> Linear -> logits

    The GRL ensures the shared encoder maximises domain classification loss
    (i.e. produces domain-invariant features) while this head minimises it.
    """

    def __init__(self, input_dim: int, n_domains: int, hidden_dim: int = 64):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_domains),
        )

    def set_lambda(self, lambda_val: float) -> None:
        self.grl.set_lambda(lambda_val)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict domain from pooled encoder features.

        Args:
            features: (batch, input_dim) pooled encoder output.
        Returns:
            (batch, n_domains) domain logits (un-normalised).
        """
        return self.classifier(self.grl(features))


def domain_adaptation_lambda(progress: float) -> float:
    """Ganin et al. sigmoid schedule for GRL lambda.

    Args:
        progress: training progress in [0, 1].
    Returns:
        lambda value for the GRL.
    """
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
