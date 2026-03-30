"""Learned discrimination scoring — trained on EasyDesign paired data.

Replaces the heuristic position*destabilisation model with a LightGBM
gradient-boosted tree trained on 6,136 paired MUT/WT trans-cleavage measurements.

The model predicts delta_logk (MUT - WT activity in log space) from 15
thermodynamic features. Discrimination ratio = 10^(delta_logk).

A physics-based R-loop discrimination estimate (D_rloop) is computed from
first principles and blended with the learned prediction. D_rloop uses the
cumulative R-loop free energy at the mismatch position and the mismatch
ddG to compute a Boltzmann propagation probability. This is entirely
deterministic, generalises across Cas enzymes, and provides a strong
baseline that the learned model refines.

Key improvements over heuristic:
  - Captures non-linear interactions between position, chemistry, and context
  - Thermodynamic energy features (cumulative dG, energy ratio) learned from data
  - R-loop physics prior (D_rloop) constrains predictions in low-data regimes
  - 15% RMSE reduction, 54% correlation improvement vs heuristic (3-fold CV)
  - Automatic fallback to heuristic when model unavailable

Interface is identical to HeuristicDiscriminationScorer — drop-in replacement.

References:
  - Huang et al. (2024) iMeta — EasyDesign training data
  - Zhang et al. (2024) NAR — R-loop energetics
  - Strohkendl et al. (2018) Mol Cell — position-dependent R-loop sensitivity
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

# Pre-computed ESM-2 protein log-likelihood ratios for AMR targets.
# |LLR| correlates with fitness cost (MTB: rho=-0.706, p=0.023).
# Low |LLR| = prevalent, conservative mutation = harder to discriminate.
# High |LLR| = rare, costly mutation = easier to discriminate.
_ESM2_LLR_LOOKUP: dict[str, float] = {}
_ESM2_LLR_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "research" / "esm2_llr" / "llr_results.csv"


def _load_esm2_llr() -> dict[str, float]:
    """Load ESM-2 LLR lookup table from pre-computed results."""
    global _ESM2_LLR_LOOKUP
    if _ESM2_LLR_LOOKUP:
        return _ESM2_LLR_LOOKUP
    if not _ESM2_LLR_PATH.exists():
        return {}
    try:
        import csv
        with open(_ESM2_LLR_PATH) as f:
            for row in csv.DictReader(f):
                if row.get("esm2_llr") and row["esm2_llr"] != "":
                    _ESM2_LLR_LOOKUP[row["label"]] = float(row["esm2_llr"])
        logger.info("Loaded ESM-2 LLR for %d targets", len(_ESM2_LLR_LOOKUP))
    except Exception as e:
        logger.debug("Could not load ESM-2 LLR: %s", e)
    return _ESM2_LLR_LOOKUP


# Ensure compass-net modules (features.thermodynamic, etc.) are importable.
from compass.scoring._netpath import ensure_importable as _ensure_compass_net, _NET_DIR as _COMPASS_NET_DIR
_ensure_compass_net()

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

        # Load ESM-2 LLR lookup (lazy, no-op if file missing)
        self._esm2_llr = _load_esm2_llr()

    def _try_load_model(self) -> bool:
        """Attempt to load the trained model."""
        if not self._model_path.exists():
            logger.info(
                "Discrimination model not found at %s, using heuristic fallback",
                self._model_path,
            )
            return False

        try:
            import pickle

            # Prefer the native-format JSON counterpart when available —
            # it is version-portable across xgboost/Python upgrades.
            # Run compass-net/scripts/migrate_disc_model_format.py to generate it.
            json_path = self._model_path.with_suffix(".json")
            if json_path.exists():
                return self._try_load_json_model(json_path)

            logger.warning(
                "Loading discrimination model from pickle (%s). "
                "Pickle is not forward-portable across xgboost/scikit-learn versions. "
                "Run compass-net/scripts/migrate_disc_model_format.py to convert "
                "to the stable XGBoost JSON format.",
                self._model_path.name,
            )

            # Pickle deserialization of local model checkpoint.
            # These files are shipped with the repo, not user-uploaded.
            with open(self._model_path, "rb") as f:
                checkpoint = pickle.load(f)

            if isinstance(checkpoint, dict) and "model" in checkpoint:
                self._model = checkpoint["model"]
                n_feat = checkpoint.get("n_features", 15)
                backend = checkpoint.get("backend", "unknown")
            else:
                # Legacy format: FeatureDiscriminationModel wrapper
                from models.discrimination_model import FeatureDiscriminationModel
                self._model = FeatureDiscriminationModel.load(self._model_path)
                n_feat = getattr(self._model, '_n_features', 15)
                backend = getattr(self._model, '_backend', 'unknown')

            from thermo_discrimination_features import compute_features_for_pair, FEATURE_NAMES, FEATURE_NAMES_V1
            self._feature_module = compute_features_for_pair
            self._feature_names = FEATURE_NAMES if n_feat >= 18 else FEATURE_NAMES_V1
            self._backend = backend

            self._model_loaded = True
            logger.info(
                "Loaded learned discrimination model from %s (backend=%s, %d features)",
                self._model_path, backend, n_feat,
            )
            return True

        except ImportError as e:
            # XGBoost/LightGBM not installed — try JSON format fallback
            json_path = self._model_path.with_suffix(".json")
            if json_path.exists():
                return self._try_load_json_model(json_path)
            logger.warning(
                "Discrimination model requires %s which is not installed. "
                "Re-export model as JSON on a machine with the library, or install it. "
                "Using heuristic fallback.",
                str(e).split("'")[1] if "'" in str(e) else str(e),
            )
            return False

        except Exception as e:
            logger.warning(
                "Failed to load discrimination model: %s. Using heuristic.",
                str(e),
            )
            return False

    def _try_load_json_model(self, json_path: Path) -> bool:
        """Load model from XGBoost native JSON format (version-portable).

        XGBoost's native JSON format is stable across library versions,
        unlike pickle. Use compass-net/scripts/migrate_disc_model_format.py
        to convert existing .pkl checkpoints.
        """
        try:
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(str(json_path))
            self._model = booster
            self._backend = "xgboost_json"
            self._model_loaded = True

            # Load accompanying feature metadata if present
            import json as _json
            meta_path = json_path.with_name(json_path.stem + "_meta.json")
            n_feat = 15
            if meta_path.exists():
                with open(meta_path) as mf:
                    meta = _json.load(mf)
                n_feat = int(meta.get("n_features", 15))

            from thermo_discrimination_features import (  # type: ignore[import]
                compute_features_for_pair, FEATURE_NAMES, FEATURE_NAMES_V1,
            )
            self._feature_module = compute_features_for_pair
            self._feature_names = FEATURE_NAMES if n_feat >= 18 else FEATURE_NAMES_V1

            logger.info(
                "Loaded discrimination model from XGBoost JSON %s (%d features)",
                json_path, n_feat,
            )
            return True
        except Exception as e:
            logger.warning("Failed to load JSON discrimination model: %s", e)
            return False

    @property
    def model_name(self) -> str:
        """Name of the active model for tracking."""
        if self._model_loaded:
            return f"learned_{getattr(self, '_backend', 'xgboost')}"
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
        """Predict using XGBoost model blended with R-loop physics prior."""
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

                # Compute features for XGBoost
                features = self._feature_module(
                    guide_seq=guide_seq,
                    spacer_position=spacer_pos,
                    mismatch_type=mismatch_type,
                    cas_variant=self.cas_variant,
                )

                # XGBoost prediction
                feature_names = self._feature_names
                X = np.array(
                    [[features[n] for n in feature_names]],
                    dtype=np.float32,
                )
                delta_logk = float(self._model.predict(X)[0])
                ratio_xgb = 10 ** delta_logk

                # R-loop physics-based discrimination (deterministic)
                try:
                    from thermo_discrimination_features import (
                        compute_rloop_discrimination,
                    )
                    rloop = compute_rloop_discrimination(
                        guide_seq=guide_seq,
                        spacer_position=spacer_pos,
                        mismatch_type=mismatch_type,
                    )
                    d_rloop = rloop["d_rloop"]
                except Exception:
                    d_rloop = ratio_xgb  # fallback: no blending

                # Blend: geometric mean of learned and physics predictions.
                # Physics prior validated (rho=0.23 standalone) but blend
                # at alpha=0.35 reduces XGBoost performance (0.519 -> 0.488).
                # CV-optimised alpha=0.03 on EasyDesign 7K paired data.
                # Keep small alpha for soft regularisation without noise.
                alpha = 0.05
                import math
                ratio = math.exp(
                    (1 - alpha) * math.log(max(ratio_xgb, 1e-6))
                    + alpha * math.log(max(d_rloop, 1e-6))
                )

                # Convert to activity scores
                wt_activity = 1.0 / max(ratio, 1e-6)
                mut_activity = 1.0

                # Confidence: higher when physics and learned model agree
                agreement = 1.0 - min(1.0, abs(math.log10(max(ratio_xgb, 0.1)) - math.log10(max(d_rloop, 0.1))) / 2.0)
                base_conf = min(1.0, max(0.3, 1.0 - abs(delta_logk - 0.57) / 2.0))
                confidence = 0.6 * base_conf + 0.4 * agreement

                # ESM-2 evolutionary constraint modulation.
                # |LLR| correlates with fitness cost (MTB: rho=-0.706, p=0.023).
                # High |LLR| → costly mutation → structurally different → more
                # confident that crRNA can discriminate MUT from WT.
                esm2_llr = self._esm2_llr.get(candidate.target_label)
                esm2_abs = None
                if esm2_llr is not None:
                    esm2_abs = abs(esm2_llr)
                    # Sigmoid scaling: |LLR|=5 → +0.05 confidence, |LLR|=10 → +0.09
                    evo_boost = 0.1 * (1.0 - math.exp(-esm2_abs / 5.0))
                    confidence = min(1.0, confidence + evo_boost)

                # Store features + physics values for diagnostics
                feature_values = [features.get(n, 0.0) for n in feature_names]
                feat_dict = dict(zip(feature_names, feature_values))
                feat_dict["d_rloop"] = round(d_rloop, 2)
                feat_dict["d_xgboost"] = round(ratio_xgb, 2)
                feat_dict["d_blended"] = round(ratio, 2)
                if esm2_abs is not None:
                    feat_dict["esm2_llr"] = round(esm2_llr, 4)
                    feat_dict["esm2_abs_llr"] = round(esm2_abs, 4)

                return DiscriminationScore(
                    wt_activity=round(wt_activity, 4),
                    mut_activity=round(mut_activity, 4),
                    model_name="learned_xgboost+rloop",
                    is_measured=False,
                    detection_strategy=candidate.detection_strategy,
                    confidence=round(confidence, 4),
                    feature_vector=feat_dict,
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
