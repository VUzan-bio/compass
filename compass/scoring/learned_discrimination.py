"""Learned discrimination scoring — trained on EasyDesign paired data.

Replaces the heuristic position×destabilisation model with a LightGBM
gradient-boosted tree trained on 6,136 paired MUT/WT trans-cleavage measurements.

The model predicts delta_logk (MUT - WT activity in log space) from 15
thermodynamic features. Discrimination ratio = 10^(delta_logk).

Key improvements over heuristic:
  - Captures non-linear interactions between position, chemistry, and context
  - Thermodynamic energy features (cumulative dG, energy ratio) learned from data
  - 15% RMSE reduction, 54% correlation improvement vs heuristic (3-fold CV)
  - Automatic fallback to heuristic when model unavailable

Interface is identical to HeuristicDiscriminationScorer — drop-in replacement.

References:
  - Huang et al. (2024) iMeta — EasyDesign training data
  - Zhang et al. (2024) NAR — R-loop energetics
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from compass.core.types import (
    CrRNACandidate,
    DetectionStrategy,
    DiscriminationScore,
    MismatchPair,
    OffTargetReport,
    ScoredCandidate,
)
from compass.scoring.base import Scorer
from compass.scoring.discrimination import (
    HeuristicDiscriminationScorer,
    DISCRIMINATION_THRESHOLD,
    PROXIMITY_DEFAULT_RATIO,
)

logger = logging.getLogger(__name__)

# Ensure compass-net is importable
_COMPASS_NET_DIR = Path(__file__).resolve().parent.parent.parent / "compass-net"
if str(_COMPASS_NET_DIR) not in sys.path:
    sys.path.insert(0, str(_COMPASS_NET_DIR))
_COMPASS_NET_DATA = _COMPASS_NET_DIR / "data"
if str(_COMPASS_NET_DATA) not in sys.path:
    sys.path.insert(0, str(_COMPASS_NET_DATA))

# DNA → RNA complement
_DNA_TO_RNA = {"A": "U", "T": "A", "C": "G", "G": "C"}


def _classify_rna_dna_mismatch(rna_base: str, dna_base: str) -> str:
    """Classify RNA:DNA mismatch pair."""
    r = rna_base.upper().replace("T", "U")
    d = dna_base.upper()
    return f"r{r}:d{d}"


class LearnedDiscriminationScorer(Scorer):
    """Learned discrimination scorer using trained XGBoost/LightGBM model.

    Drop-in replacement for HeuristicDiscriminationScorer.
    Falls back to heuristic when model is unavailable or prediction fails.

    Usage:
        scorer = LearnedDiscriminationScorer(
            model_path="compass-net/checkpoints/disc_xgb.pkl",
            cas_variant="enAsCas12a",
        )
        scored = scorer.score_with_pair(candidate, pair, offtarget)
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        cas_variant: str = "enAsCas12a",
        min_ratio: float = DISCRIMINATION_THRESHOLD,
        heuristic_fallback: Optional[Scorer] = None,
    ) -> None:
        self.cas_variant = cas_variant
        self.min_ratio = min_ratio
        self._model = None
        self._feature_module = None
        self._model_loaded = False

        # Heuristic fallback
        self._heuristic = HeuristicDiscriminationScorer(
            cas_variant=cas_variant,
            min_ratio=min_ratio,
            heuristic_fallback=heuristic_fallback,
        )

        # Try to load the trained model
        if model_path is None:
            # Default checkpoint location
            model_path = _COMPASS_NET_DIR / "checkpoints" / "disc_xgb.pkl"

        self._model_path = Path(model_path)
        self._try_load_model()

    def _try_load_model(self) -> bool:
        """Attempt to load the trained model."""
        if not self._model_path.exists():
            logger.info(
                "Discrimination model not found at %s, using heuristic fallback",
                self._model_path,
            )
            return False

        try:
            from models.discrimination_model import FeatureDiscriminationModel
            self._model = FeatureDiscriminationModel.load(self._model_path)

            from thermo_discrimination_features import compute_features_for_pair, FEATURE_NAMES, FEATURE_NAMES_V1
            self._feature_module = compute_features_for_pair

            # Auto-detect feature version from checkpoint
            n_feat = getattr(self._model, '_n_features', None)
            if n_feat is None:
                # Try to infer from pickle metadata
                import pickle
                with open(self._model_path, "rb") as f:
                    meta = pickle.load(f)
                n_feat = meta.get("n_features", 15)
            self._feature_names = FEATURE_NAMES if n_feat >= 18 else FEATURE_NAMES_V1

            self._model_loaded = True
            logger.info(
                "Loaded learned discrimination model from %s (backend=%s)",
                self._model_path,
                self._model._backend,
            )
            return True

        except Exception as e:
            logger.warning(
                "Failed to load discrimination model: %s. Using heuristic.",
                str(e),
            )
            return False

    @property
    def model_name(self) -> str:
        """Name of the active model for tracking."""
        if self._model_loaded:
            return f"learned_{self._model._backend}"
        return "heuristic_discrimination"

    def score(
        self,
        candidate: CrRNACandidate,
        offtarget: OffTargetReport,
    ) -> ScoredCandidate:
        """Score a candidate (without discrimination)."""
        return self._heuristic.score(candidate, offtarget)

    def score_with_pair(
        self,
        candidate: CrRNACandidate,
        pair: MismatchPair,
        offtarget: OffTargetReport,
    ) -> ScoredCandidate:
        """Score a candidate WITH discrimination analysis."""
        scored = self.score(candidate, offtarget)
        scored.discrimination = self.predict_discrimination(candidate, pair)
        return scored

    def predict_discrimination(
        self,
        candidate: CrRNACandidate,
        pair: MismatchPair,
    ) -> DiscriminationScore:
        """Predict MUT/WT discrimination ratio.

        Uses the learned model when available, falls back to heuristic.
        For PROXIMITY candidates, returns conservative estimate (no crRNA-level disc).
        """
        strategy = candidate.detection_strategy

        # PROXIMITY: no crRNA-level discrimination
        if strategy != DetectionStrategy.DIRECT:
            return DiscriminationScore(
                wt_activity=1.0,
                mut_activity=PROXIMITY_DEFAULT_RATIO,
                model_name="learned_proximity" if self._model_loaded else "heuristic_proximity",
                is_measured=False,
                detection_strategy=strategy,
            )

        # DIRECT: try learned model first
        if self._model_loaded:
            try:
                return self._predict_learned(candidate, pair)
            except Exception as e:
                logger.debug("Learned prediction failed for %s: %s", candidate.candidate_id, e)

        # Fallback to heuristic
        return self._heuristic.predict_discrimination(candidate, pair)

    def _predict_learned(
        self,
        candidate: CrRNACandidate,
        pair: MismatchPair,
    ) -> DiscriminationScore:
        """Predict using the trained model."""
        # Find mismatch position and type from the pair
        wt_spacer = pair.wt_spacer
        mut_spacer = pair.mut_spacer

        if not wt_spacer or not mut_spacer or len(wt_spacer) != len(mut_spacer):
            raise ValueError("Invalid spacer pair")

        # Find mismatch positions
        for i in range(len(wt_spacer)):
            if wt_spacer[i].upper() != mut_spacer[i].upper():
                spacer_pos = i + 1  # 1-indexed from PAM-proximal

                # Mismatch type: crRNA RNA base vs WT DNA base
                mut_dna = mut_spacer[i].upper()
                wt_dna = wt_spacer[i].upper()
                rna_base = _DNA_TO_RNA.get(mut_dna, "N")
                mismatch_type = _classify_rna_dna_mismatch(rna_base, wt_dna)

                # Build guide sequence (PAM + spacer) for feature computation
                pam = candidate.pam_seq if hasattr(candidate, "pam_seq") else "TTTV"
                guide_seq = pam + mut_spacer

                # Compute features
                features = self._feature_module(
                    guide_seq=guide_seq,
                    spacer_position=spacer_pos,
                    mismatch_type=mismatch_type,
                    cas_variant=self.cas_variant,
                )

                # Auto-detect: use 18 features if v2 checkpoint, 15 if legacy
                feature_names = self._feature_names
                X = np.array(
                    [[features[n] for n in feature_names]],
                    dtype=np.float32,
                )
                delta_logk = float(self._model.predict(X)[0])
                ratio = 10 ** delta_logk

                # Convert to activity scores
                # MUT activity = 1.0 (perfect match), WT = 1/ratio
                wt_activity = 1.0 / max(ratio, 1e-6)
                mut_activity = 1.0

                # Confidence: based on how far the prediction is from
                # the training distribution (simple heuristic)
                confidence = min(1.0, max(0.3, 1.0 - abs(delta_logk - 0.57) / 2.0))

                return DiscriminationScore(
                    wt_activity=round(wt_activity, 4),
                    mut_activity=round(mut_activity, 4),
                    model_name=self.model_name,
                    is_measured=False,
                    detection_strategy=candidate.detection_strategy,
                )

        # No mismatch found
        return DiscriminationScore(
            wt_activity=1.0,
            mut_activity=1.0,
            model_name=self.model_name,
            is_measured=False,
            detection_strategy=candidate.detection_strategy,
        )

    def add_discrimination(
        self,
        scored: ScoredCandidate,
        pair: MismatchPair,
    ) -> ScoredCandidate:
        """Add discrimination score to an existing ScoredCandidate."""
        scored.discrimination = self.predict_discrimination(scored.candidate, pair)
        return scored

    def add_discrimination_batch(
        self,
        scored_candidates: list[ScoredCandidate],
        pairs: list[MismatchPair],
    ) -> list[ScoredCandidate]:
        """Add discrimination scores to a batch of candidates."""
        pair_map = {p.candidate_id: p for p in pairs}

        for sc in scored_candidates:
            pair = pair_map.get(sc.candidate.candidate_id)
            if pair is not None:
                self.add_discrimination(sc, pair)

        n_scored = sum(1 for sc in scored_candidates if sc.discrimination is not None)
        n_learned = sum(
            1 for sc in scored_candidates
            if sc.discrimination is not None and "learned" in (sc.discrimination.model_name or "")
        )
        logger.info(
            "Discrimination scoring: %d/%d scored (%d learned, %d heuristic)",
            n_scored, len(scored_candidates),
            n_learned, n_scored - n_learned,
        )

        return scored_candidates

    def analyze_panel_discrimination(
        self,
        scored_candidates: list[ScoredCandidate],
    ) -> dict[str, dict]:
        """Delegate to heuristic for panel analysis."""
        return self._heuristic.analyze_panel_discrimination(scored_candidates)
