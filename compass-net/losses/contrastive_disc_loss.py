"""Contrastive discrimination loss for Enhancement A.

Combines regression (Huber on log-ratios) with a margin-based contrastive
term that teaches the model which of the MUT/WT pair should have higher
activity — a ranking signal more robust to fluorescence noise.

L = alpha * Huber(log(1+pred), log(1+true)) + (1-alpha) * L_contrastive

The contrastive term:
  - High-disc pairs (ratio > 2): push z_mut and z_wt apart (margin gap)
  - Low-disc pairs (ratio <= 1.5): pull embeddings together
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveDiscriminationLoss(nn.Module):
    """Combined regression + contrastive loss for discrimination prediction."""

    def __init__(self, margin: float = 0.5, alpha: float = 0.6):
        """
        Args:
            margin: Minimum desired gap between z_mut and z_wt norms
                    for high-discrimination pairs.
            alpha: Balance between regression (alpha) and contrastive
                   (1-alpha). 0.6 = slight preference for regression.
        """
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.huber = nn.HuberLoss(delta=1.0, reduction="none")

    def forward(
        self,
        pred_ratio: torch.Tensor,
        true_ratio: torch.Tensor,
        z_mut: torch.Tensor,
        z_wt: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_ratio: (batch,) Softplus output from disc head.
            true_ratio: (batch,) ground truth MUT/WT ratio.
            z_mut:      (batch, D) pooled MUT embedding.
            z_wt:       (batch, D) pooled WT embedding.
            weights:    (batch,) optional per-sample weights.

        Returns:
            Scalar loss.
        """
        # --- Regression term (per-sample) ---
        log_pred = torch.log1p(pred_ratio)
        log_true = torch.log1p(true_ratio)
        l_reg = self.huber(log_pred, log_true)  # (batch,)

        # --- Contrastive term ---
        diff_norm = torch.norm(z_mut - z_wt, dim=-1)  # (batch,)

        # High disc (ratio > 2): push embeddings apart
        high_disc = (true_ratio > 2.0).float()
        l_push = high_disc * F.relu(self.margin - diff_norm)

        # Low disc (ratio <= 1.5): pull embeddings together
        low_disc = (true_ratio <= 1.5).float()
        l_pull = low_disc * F.relu(diff_norm - self.margin * 0.3)

        l_contrastive = l_push + l_pull  # (batch,)

        # --- Combine ---
        per_sample = self.alpha * l_reg + (1 - self.alpha) * l_contrastive

        if weights is not None:
            per_sample = per_sample * weights

        return per_sample.mean()
