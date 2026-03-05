"""Level 2 — Sequence-based ML prediction (Seq-deepCpf1 equivalent).

CNN trained on Kim et al. 2018 HT-PAMDA data + Huang et al. 2024
EasyDesign diagnostic data.

Input: one-hot encoded 34-nt context (upstream + PAM + spacer + downstream).
Output: predicted Cas12a activity score in [0, 1].

Falls back to heuristic scoring when model weights are unavailable.

Temperature calibration (T > 1) spreads saturated sigmoid outputs.
Ensemble weight alpha blends calibrated CNN with heuristic scoring.

Reference: Kim et al., Nature Biotechnology (2018). "Deep learning improves
prediction of CRISPR-Cpf1 guide RNA activity."
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from compass.core.types import (
    CrRNACandidate,
    HeuristicScore,
    MLScore,
    OffTargetReport,
    ScoredCandidate,
)
from compass.scoring.base import Scorer
from compass.scoring.preprocessing import extract_input_window, one_hot_encode

logger = logging.getLogger(__name__)

_CONTEXT_LENGTH = 34  # upstream + PAM(4) + spacer(20-23) + downstream
_DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "seq_cnn_best.pt"
_DEFAULT_CALIBRATION = Path(__file__).resolve().parent.parent / "weights" / "calibration.json"


class SequenceMLScorer(Scorer):
    """Sequence-based CNN scorer with temperature calibration and ensemble.

    Wraps a trained SeqCNN model. Falls back to heuristic if model
    is unavailable.

    Temperature calibration: sigmoid(logit / T) spreads saturated outputs.
    Ensemble: alpha * heuristic + (1 - alpha) * calibrated_cnn.

    Usage:
        scorer = SequenceMLScorer(model_path="weights/seq_cnn_best.pt")
        scored = scorer.score_batch(candidates, offtargets)
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        heuristic_fallback: Optional[Scorer] = None,
        device: Optional[str] = None,
        calibration_path: Optional[str | Path] = None,
    ) -> None:
        self.model = None
        self.model_path = model_path
        self._fallback = heuristic_fallback
        self._device_name = device
        self._device = None
        self._val_rho: Optional[float] = None

        # Calibration parameters
        self.temperature: float = 1.0
        self.alpha: float = 0.0
        self.calibrated: bool = False
        self._calibration_meta: dict = {}

        if model_path is None and _DEFAULT_WEIGHTS.exists():
            model_path = _DEFAULT_WEIGHTS
        if model_path is not None:
            self._load_model(Path(model_path))

        # Load calibration
        cal_path = Path(calibration_path) if calibration_path else _DEFAULT_CALIBRATION
        self._load_calibration(cal_path)

    def score(
        self,
        candidate: CrRNACandidate,
        offtarget: OffTargetReport,
    ) -> ScoredCandidate:
        # Always compute heuristic as baseline
        if self._fallback:
            base = self._fallback.score(candidate, offtarget)
        else:
            from compass.scoring.heuristic import HeuristicScorer
            base = HeuristicScorer().score(candidate, offtarget)

        # Add ML prediction if model available
        if self.model is not None:
            prediction = self._predict(candidate)
            base.ml_scores.append(MLScore(
                model_name="seq_cnn",
                predicted_efficiency=prediction,
            ))

        return base

    def score_batch(
        self,
        candidates: list[CrRNACandidate],
        offtargets: list[OffTargetReport],
    ) -> list[ScoredCandidate]:
        """Override for GPU-batched inference."""
        if self.model is None:
            return super().score_batch(candidates, offtargets)

        # Batch encode all contexts
        contexts = [self._encode_context(c) for c in candidates]
        predictions = self._predict_batch(contexts)

        scored = []
        for c, o, pred in zip(candidates, offtargets, predictions):
            s = self.score(c, o)
            # Replace the individual prediction with batch result
            s.ml_scores = [MLScore(model_name="seq_cnn", predicted_efficiency=pred)]
            scored.append(s)

        scored.sort(key=lambda s: self._sort_key(s), reverse=True)
        for i, s in enumerate(scored):
            s.rank = i + 1
        return scored

    @property
    def validation_rho(self) -> Optional[float]:
        """Spearman rho from training validation set."""
        return self._val_rho

    @property
    def calibration_meta(self) -> dict:
        """Calibration metadata (T, alpha, rho values)."""
        return dict(self._calibration_meta)

    def calibrated_score(self, raw_score: float) -> float:
        """Apply temperature calibration to a raw CNN score.

        Inverts sigmoid to get logit, divides by T, re-applies sigmoid.
        """
        if not self.calibrated or self.temperature <= 0:
            return raw_score
        clamped = max(1e-7, min(1 - 1e-7, raw_score))
        logit = math.log(clamped / (1 - clamped))
        return 1.0 / (1.0 + math.exp(-logit / self.temperature))

    def ensemble_score(self, heuristic_score: float, cnn_calibrated: float) -> float:
        """Compute ensemble: alpha * heuristic + (1 - alpha) * calibrated_cnn."""
        return self.alpha * heuristic_score + (1 - self.alpha) * cnn_calibrated

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_calibration(self, path: Path) -> None:
        """Load temperature and ensemble weight from calibration JSON."""
        if not path.exists():
            logger.info("No calibration file at %s — using raw scores", path)
            return
        try:
            with open(path) as f:
                cal = json.load(f)
            self.temperature = cal.get("temperature", 1.0)
            self.alpha = cal.get("alpha", 0.0)
            self.calibrated = True
            self._calibration_meta = cal
            logger.info(
                "Loaded calibration: T=%.2f, alpha=%.4f (val ensemble rho=%.4f)",
                self.temperature,
                self.alpha,
                cal.get("val_rho_ensemble", 0.0),
            )
        except Exception as e:
            logger.warning("Failed to load calibration from %s: %s", path, e)

    def _load_model(self, path: Path) -> None:
        """Load a trained SeqCNN model from checkpoint."""
        try:
            import torch
            from compass.scoring.seq_cnn import SeqCNN

            self._device = torch.device(
                self._device_name
                or ("cuda" if torch.cuda.is_available() else "cpu")
            )

            self.model = SeqCNN()
            checkpoint = torch.load(
                str(path), map_location=self._device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self._device)
            self.model.eval()
            self._val_rho = checkpoint.get("val_rho")
            logger.info(
                "Loaded SeqCNN from %s (val rho=%.4f)",
                path,
                self._val_rho or 0.0,
            )
        except Exception as e:
            logger.warning("Failed to load SeqCNN from %s: %s", path, e)
            self.model = None

    def _encode_context(self, candidate: CrRNACandidate) -> np.ndarray:
        """Build the 34-nt context window and one-hot encode it.

        Uses extract_input_window for proper flanking context handling.
        Falls back to PAM + spacer + N-padding if flanking unavailable.
        """
        pam = candidate.pam_seq
        spacer = candidate.spacer_seq

        # Build input window with flanking context if available
        window = extract_input_window(
            pam=pam,
            spacer=spacer,
            upstream_flank="",
            downstream_flank="",
            total_len=_CONTEXT_LENGTH,
        )
        return one_hot_encode(window, max_len=_CONTEXT_LENGTH)

    def _predict(self, candidate: CrRNACandidate) -> float:
        """Single-sample prediction."""
        encoded = self._encode_context(candidate)
        predictions = self._predict_batch([encoded])
        return predictions[0]

    def _predict_batch(self, contexts: list[np.ndarray]) -> list[float]:
        """Batch prediction. Returns list of efficiency scores in [0, 1]."""
        if self.model is None:
            return [0.5] * len(contexts)

        import torch

        batch = torch.tensor(np.stack(contexts), dtype=torch.float32)
        if self._device is not None:
            batch = batch.to(self._device)
        with torch.no_grad():
            output = self.model(batch)
        return output.squeeze(-1).clamp(0, 1).tolist()
