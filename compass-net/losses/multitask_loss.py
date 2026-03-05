"""Multi-task loss for Compass-ML.

L_total = L_efficiency + lambda_rank * (1 - rho_soft) + lambda_disc * L_discrimination

- Efficiency: Huber loss (robust to outlier failed reactions)
- Ranking: differentiable Spearman (directly optimises the evaluation metric)
- Discrimination: Huber in log-space (ratios span 1x to 100x)

Spearman regularization_strength is annealed during training via
set_spearman_strength() -- called from the training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .spearman_loss import differentiable_spearman


class MultiTaskLoss(nn.Module):
    """Combined efficiency + discrimination loss with ranking regulariser.

    Phase 1: lambda_disc=0 (efficiency only)
    Phase 2: lambda_disc=0.1 (proxy discrimination as regulariser)
    Phase 3: lambda_disc=0.3-0.5 (real discrimination labels)
    """

    def __init__(
        self,
        lambda_disc: float = 0.0,
        lambda_rank: float = 0.5,
        huber_delta: float = 0.5,
    ):
        super().__init__()
        self.lambda_disc = lambda_disc
        self.lambda_rank = lambda_rank
        self.huber = nn.HuberLoss(delta=huber_delta)
        self._spearman_strength = 1.0

    def set_spearman_strength(self, strength: float) -> None:
        """Anneal Spearman regularization: call from training loop.

        Schedule: strength = max(0.1, 1.0 - 0.9 * epoch / n_epochs)
        """
        self._spearman_strength = max(0.01, strength)

    def forward(
        self,
        pred_eff: torch.Tensor,
        true_eff: torch.Tensor,
        pred_disc: torch.Tensor | None = None,
        true_disc: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-task loss.

        Args:
            pred_eff:  (batch, 1) predicted efficiency.
            true_eff:  (batch,) true efficiency.
            pred_disc: (batch, 1) predicted discrimination ratio (optional).
            true_disc: (batch,) true discrimination ratio (optional).

        Returns:
            Dict with keys: "total", "efficiency", "ranking",
            and optionally "discrimination".
        """
        pred_flat = pred_eff.squeeze(-1)

        l_eff = self.huber(pred_flat, true_eff)
        l_rank = 1.0 - differentiable_spearman(
            pred_flat, true_eff,
            regularization_strength=self._spearman_strength,
        )

        total = l_eff + self.lambda_rank * l_rank
        losses: dict[str, torch.Tensor] = {
            "efficiency": l_eff,
            "ranking": l_rank,
        }

        if (
            self.lambda_disc > 0
            and pred_disc is not None
            and true_disc is not None
        ):
            # Huber in log-space: ratios span 1x-100x, log compresses range
            l_disc = self.huber(
                torch.log1p(pred_disc.squeeze(-1)),
                torch.log1p(true_disc),
            )
            total = total + self.lambda_disc * l_disc
            losses["discrimination"] = l_disc

        losses["total"] = total
        return losses
