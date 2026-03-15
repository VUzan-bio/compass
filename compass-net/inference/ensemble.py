"""Ensemble model averaging for Compass-ML.

Runs N independently trained models and averages predictions.
Provides uncertainty estimates for free (prediction variance).

Benefits (standard in ML):
    - ~5-10% Spearman improvement from variance reduction
    - Prediction uncertainty from ensemble disagreement
    - Better calibration without temperature scaling
    - Robustness to random seed sensitivity

Usage:
    ensemble = EnsemblePredictor(
        checkpoint_paths=["model_seed42.pt", "model_seed123.pt", "model_seed456.pt"],
    )
    mean, std = ensemble.predict(target_onehot, rnafm_emb)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Average predictions from multiple Compass-ML checkpoints.

    Each model is loaded independently and run in inference mode.
    Predictions are averaged; standard deviation provides uncertainty.
    """

    def __init__(
        self,
        checkpoint_paths: list[str | Path],
        device: str = "cpu",
        model_class: Optional[type] = None,
    ):
        self.device = device
        self.models: list[torch.nn.Module] = []

        if model_class is None:
            from compass_ml import CompassML
            model_class = CompassML

        for path in checkpoint_paths:
            path = Path(path)
            if not path.exists():
                logger.warning("Ensemble: checkpoint not found: %s", path)
                continue

            checkpoint = torch.load(path, map_location=device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            # Auto-detect model config from checkpoint
            config = checkpoint.get("config", {})
            model = model_class(**config)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            self.models.append(model)

        logger.info("Ensemble loaded: %d models", len(self.models))

    @property
    def n_models(self) -> int:
        return len(self.models)

    @torch.no_grad()
    def predict(
        self,
        target_onehot: torch.Tensor,
        crrna_rnafm_emb: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict with ensemble averaging.

        Args:
            target_onehot: (batch, 4, 34) one-hot target DNA.
            crrna_rnafm_emb: (batch, 20, 640) RNA-FM embeddings.

        Returns:
            (mean_predictions, std_predictions): each (batch,)
        """
        if not self.models:
            raise RuntimeError("No models loaded in ensemble")

        all_preds = []
        for model in self.models:
            output = model(
                target_onehot=target_onehot.to(self.device),
                crrna_rnafm_emb=crrna_rnafm_emb.to(self.device) if crrna_rnafm_emb is not None else None,
            )
            eff = output["efficiency"].squeeze(-1).cpu().numpy()
            all_preds.append(eff)

        stacked = np.stack(all_preds, axis=0)  # (n_models, batch)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)

        return mean, std

    @torch.no_grad()
    def predict_with_discrimination(
        self,
        mut_onehot: torch.Tensor,
        wt_onehot: torch.Tensor,
        crrna_rnafm_emb: torch.Tensor | None = None,
        thermo_feats: torch.Tensor | None = None,
        mm_position: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict efficiency and discrimination with uncertainty.

        Returns:
            (eff_mean, eff_std, disc_mean, disc_std): each (batch,)
        """
        eff_preds, disc_preds = [], []

        for model in self.models:
            output = model(
                target_onehot=mut_onehot.to(self.device),
                crrna_rnafm_emb=crrna_rnafm_emb.to(self.device) if crrna_rnafm_emb is not None else None,
                wt_target_onehot=wt_onehot.to(self.device),
                thermo_feats=thermo_feats.to(self.device) if thermo_feats is not None else None,
                mm_position=mm_position.to(self.device) if mm_position is not None else None,
            )
            eff_preds.append(output["efficiency"].squeeze(-1).cpu().numpy())
            if "discrimination" in output:
                disc_preds.append(output["discrimination"].squeeze(-1).cpu().numpy())

        eff_stack = np.stack(eff_preds, axis=0)
        eff_mean, eff_std = eff_stack.mean(axis=0), eff_stack.std(axis=0)

        if disc_preds:
            disc_stack = np.stack(disc_preds, axis=0)
            disc_mean, disc_std = disc_stack.mean(axis=0), disc_stack.std(axis=0)
        else:
            disc_mean = np.zeros_like(eff_mean)
            disc_std = np.zeros_like(eff_mean)

        return eff_mean, eff_std, disc_mean, disc_std
