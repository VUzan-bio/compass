"""Calibration check and temperature scaling for Compass-ML.

Active learning requires meaningful uncertainty estimates. If MC-dropout
predictions are miscalibrated, the acquisition function will select
suboptimal candidates.

Workflow:
    1. After Phase 2 training, compute reliability diagram on val set
    2. If ECE > threshold, fit temperature scaling on val set
    3. Apply temperature scaling before MC-dropout in Phase 3

Temperature scaling (Guo et al., ICML 2017): a single learnable scalar
that rescales the logits before sigmoid. Simple, effective, and doesn't
change the ranking -- only the confidence calibration.

References:
    Guo et al., "On Calibration of Modern Neural Networks." ICML 2017.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS


class TemperatureScaling(nn.Module):
    """Post-hoc temperature scaling for calibration.

    Applied AFTER the efficiency head's final linear layer but BEFORE
    the sigmoid activation. Higher temperature = softer predictions
    (closer to 0.5), lower = sharper.

    Fitting: optimise temperature on a held-out validation set by
    minimising NLL (or MSE for regression).
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature before activation."""
        return logits / self.temperature.clamp(min=0.01)

    def fit(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
        max_iter: int = 50,
    ) -> float:
        """Fit temperature on validation data using L-BFGS.

        Args:
            val_logits: (N,) pre-sigmoid logits from the model.
            val_targets: (N,) true efficiency values.
            max_iter: L-BFGS iterations.

        Returns:
            Fitted temperature value.
        """
        self.temperature.data.fill_(1.0)
        criterion = nn.MSELoss()

        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = torch.sigmoid(val_logits / self.temperature.clamp(min=0.01))
            loss = criterion(scaled, val_targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()


def reliability_diagram(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute reliability diagram data for regression.

    Bins predictions by value, computes mean predicted vs mean observed
    in each bin. Well-calibrated models have points on the diagonal.

    Args:
        predictions: (N,) predicted efficiencies.
        targets: (N,) true efficiencies.
        n_bins: number of bins.

    Returns:
        Dict with 'bins' (list of dicts) and 'ece' (expected calibration error).
    """
    bin_edges = np.linspace(predictions.min(), predictions.max(), n_bins + 1)
    bins: list[dict] = []
    total_count = len(predictions)

    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        # Include right edge in last bin
        if i == n_bins - 1:
            mask = mask | (predictions == bin_edges[i + 1])

        count = mask.sum()
        if count > 0:
            bins.append({
                "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                "mean_predicted": float(predictions[mask].mean()),
                "mean_observed": float(targets[mask].mean()),
                "count": int(count),
            })

    # Expected Calibration Error (weighted by bin population)
    ece = sum(
        b["count"] * abs(b["mean_predicted"] - b["mean_observed"])
        for b in bins
    ) / max(total_count, 1)

    return {"bins": bins, "ece": float(ece)}


def check_calibration(
    predictions: np.ndarray,
    targets: np.ndarray,
    ece_threshold: float = 0.05,
) -> tuple[bool, float]:
    """Check if model predictions are well-calibrated.

    Args:
        predictions: (N,) predicted efficiencies.
        targets: (N,) true efficiencies.
        ece_threshold: maximum acceptable ECE.

    Returns:
        (is_calibrated, ece_value)
    """
    result = reliability_diagram(predictions, targets)
    ece = result["ece"]
    return ece <= ece_threshold, ece
