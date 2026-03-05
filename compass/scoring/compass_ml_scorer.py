"""Compass-ML scorer adapter for the COMPASS pipeline.

Bridges the standalone Compass-ML model (compass-net/) with the pipeline's
Module 5 scoring interface. Replaces SequenceMLScorer when Compass-ML
weights are available.

Architecture: dual-branch CNN + RNA-FM with optional RLPA attention.
The CNN branch processes target DNA (34-nt one-hot), the RNA-FM branch
processes crRNA spacer embeddings (pre-cached). RLPA adds biophysically-
informed causal attention encoding Cas12a R-loop directionality.

When RNA-FM embeddings are unavailable (cache miss), falls back to
CNN-only scoring which is still superior to the old SeqCNN v1.

Usage in runner.py:
    scorer = CompassMlScorer(
        weights_path="compass/weights/compass_ml_best.pt",
        rnafm_cache_dir="compass/data/embeddings/rnafm",
    )
    scored = scorer.score_batch(candidates, offtargets)
"""

from __future__ import annotations

import json
import logging
import math
import sys
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

_CONTEXT_LENGTH = 34
_COMPASS_NET_DIR = Path(__file__).resolve().parent.parent.parent / "compass-net"
_DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "compass_ml_diagnostic.pt"
_DEFAULT_CALIBRATION = Path(__file__).resolve().parent.parent / "weights" / "compass_ml_calibration.json"

# Complement tables
_DNA_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
_DNA_TO_RNA_RC = {"A": "U", "T": "A", "C": "G", "G": "C", "N": "N"}


class CompassMlScorer(Scorer):
    """Pipeline-compatible scorer using Compass-ML.

    Implements the Scorer interface (score/score_batch) so the pipeline
    can use it as a drop-in replacement for SequenceMLScorer.

    The model runs on CPU by default — at 235K params, single-sample
    inference takes <1ms, and batched inference is negligible.
    """

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        heuristic_fallback: Optional[Scorer] = None,
        rnafm_cache_dir: Optional[str | Path] = None,
        calibration_path: Optional[str | Path] = None,
        use_rlpa: bool = True,
        use_rnafm: bool = True,
        multitask: bool = False,
        device: str = "cpu",
        collect_embeddings: bool = False,
    ) -> None:
        self.model = None
        self._fallback = heuristic_fallback
        self.collect_embeddings = collect_embeddings
        self._collected_embeddings: list[dict] = []
        self._device_name = device
        self._device = None
        self._val_rho: Optional[float] = None
        self._use_rnafm = use_rnafm
        self._use_rlpa = use_rlpa
        self._multitask = multitask
        self._rnafm_cache = None

        # Calibration parameters
        self.temperature: float = 1.0
        self.alpha: float = 0.0
        self.calibrated: bool = False
        self._calibration_meta: dict = {}

        # Resolve weights path
        if weights_path is None:
            weights_path = _DEFAULT_WEIGHTS
        weights_path = Path(weights_path)

        if weights_path.exists():
            self._load_model(weights_path, use_rnafm, use_rlpa, multitask)
        else:
            logger.warning(
                "Compass-ML weights not found at %s — scorer will use heuristic fallback",
                weights_path,
            )

        # Load calibration
        cal_path = Path(calibration_path) if calibration_path else _DEFAULT_CALIBRATION
        self._load_calibration(cal_path)

        # Load RNA-FM embedding cache
        if use_rnafm and rnafm_cache_dir is not None:
            cache_path = Path(rnafm_cache_dir)
            if cache_path.exists():
                self._load_rnafm_cache(cache_path)
            else:
                # Keep _use_rnafm True — the model architecture expects
                # 128-dim fused input (64 CNN + 64 RNA-FM projection).
                # Zero embeddings degrade gracefully.
                logger.info(
                    "RNA-FM cache dir %s not found — using zero embeddings (model needs RNA-FM dim)",
                    cache_path,
                )

    @property
    def validation_rho(self) -> Optional[float]:
        return self._val_rho

    @property
    def calibration_meta(self) -> dict:
        return dict(self._calibration_meta)

    def calibrated_score(self, raw_score: float) -> float:
        """Apply temperature calibration: sigmoid(logit(raw) / T).

        Spreads compressed sigmoid outputs so threshold decisions
        (efficiency >= 0.4, etc.) work on a properly scaled range.
        """
        if not self.calibrated or self.temperature <= 0:
            return raw_score
        clamped = max(1e-7, min(1 - 1e-7, raw_score))
        logit = math.log(clamped / (1 - clamped))
        return 1.0 / (1.0 + math.exp(-logit / self.temperature))

    def ensemble_score_val(self, heuristic_score: float, gn_calibrated: float) -> float:
        """Compute ensemble: alpha * heuristic + (1 - alpha) * calibrated_gn."""
        return self.alpha * heuristic_score + (1 - self.alpha) * gn_calibrated

    # Alias so the pipeline runner's manual loop can call the same interface
    # as SequenceMLScorer (which defines ensemble_score without the _val suffix).
    def ensemble_score(self, heuristic_score: float, gn_calibrated: float) -> float:
        return self.ensemble_score_val(heuristic_score, gn_calibrated)

    def score(
        self,
        candidate: CrRNACandidate,
        offtarget: OffTargetReport,
    ) -> ScoredCandidate:
        """Score a single candidate.

        Always computes heuristic as the base score (composite_score uses it).
        Adds Compass-ML prediction as an MLScore if model is available.
        When calibrated, populates cnn_calibrated and ensemble_score fields
        so composite_score and Block 3 thresholds use calibrated values.
        """
        # Heuristic baseline (required — composite_score delegates to it)
        if self._fallback:
            base = self._fallback.score(candidate, offtarget)
        else:
            from compass.scoring.heuristic import HeuristicScorer
            base = HeuristicScorer().score(candidate, offtarget)

        # Add Compass-ML prediction
        if self.model is not None:
            prediction = self._predict_single(candidate)
            base.ml_scores.append(MLScore(
                model_name="compass_ml",
                predicted_efficiency=prediction,
            ))
            base.cnn_score = prediction

            # Apply calibration
            if self.calibrated:
                cal = self.calibrated_score(prediction)
                base.cnn_calibrated = cal
                base.ensemble_score = self.ensemble_score_val(
                    base.heuristic.composite, cal,
                )

        return base

    def score_batch(
        self,
        candidates: list[CrRNACandidate],
        offtargets: list[OffTargetReport],
    ) -> list[ScoredCandidate]:
        """Score and rank a batch of candidates."""
        if self.model is None:
            return super().score_batch(candidates, offtargets)

        # Batch encode and predict
        contexts = [self._encode_context(c) for c in candidates]
        rnafm_embs = [self._get_rnafm_embedding(c) for c in candidates]
        predictions = self._predict_batch(contexts, rnafm_embs)

        scored = []
        for c, o, pred in zip(candidates, offtargets, predictions):
            s = self.score(c, o)
            s.ml_scores = [MLScore(model_name="compass_ml", predicted_efficiency=pred)]
            s.cnn_score = pred

            # Apply calibration to batch predictions
            if self.calibrated:
                cal = self.calibrated_score(pred)
                s.cnn_calibrated = cal
                s.ensemble_score = self.ensemble_score_val(
                    s.heuristic.composite, cal,
                )

            scored.append(s)

        scored.sort(key=lambda s: self._sort_key(s), reverse=True)
        for i, s in enumerate(scored):
            s.rank = i + 1
        return scored

    def predict_efficiency(self, candidate: CrRNACandidate) -> float:
        """Predict efficiency score only (no heuristic, no ScoredCandidate wrapper)."""
        if self.model is None:
            return 0.5
        return self._predict_single(candidate)

    def predict_with_discrimination(
        self,
        candidate: CrRNACandidate,
        wt_context_34: Optional[str] = None,
        mm_position: Optional[int] = None,
    ) -> dict[str, float]:
        """Predict efficiency and optionally discrimination ratio.

        Args:
            candidate: The crRNA candidate (mutant target).
            wt_context_34: 34-nt wildtype target DNA string. If provided
                and model has multitask head, returns discrimination too.
            mm_position: PAM-relative mismatch position (1-24), or None.

        Returns:
            dict with "efficiency" and optionally "neural_disc", "disc_method".
        """
        if self.model is None:
            return {"efficiency": 0.5}

        import torch

        context = self._encode_context(candidate)
        target_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        target_tensor = target_tensor.to(self._device)

        rnafm_emb = self._get_rnafm_embedding(candidate)
        rnafm_tensor = None
        if rnafm_emb is not None:
            rnafm_tensor = torch.tensor(rnafm_emb, dtype=torch.float32).unsqueeze(0)
            rnafm_tensor = rnafm_tensor.to(self._device)

        wt_tensor = None
        if wt_context_34 is not None and self._multitask:
            wt_onehot = one_hot_encode(wt_context_34, max_len=_CONTEXT_LENGTH)
            wt_tensor = torch.tensor(wt_onehot, dtype=torch.float32).unsqueeze(0)
            wt_tensor = wt_tensor.to(self._device)

        # Compute thermo features if model has them
        thermo_tensor = None
        if self._n_thermo > 0 and wt_context_34 is not None and mm_position is not None:
            thermo_tensor = self._compute_thermo_feats(
                candidate, wt_context_34, mm_position,
            )
            if thermo_tensor is not None:
                thermo_tensor = thermo_tensor.to(self._device)

        # Mismatch position tensor
        mm_tensor = None
        if self._pos_embed_dim > 0 and mm_position is not None:
            mm_tensor = torch.tensor([mm_position], dtype=torch.long, device=self._device)

        with torch.no_grad():
            output = self.model(
                target_onehot=target_tensor,
                crrna_rnafm_emb=rnafm_tensor,
                wt_target_onehot=wt_tensor,
                thermo_feats=thermo_tensor,
                mm_position=mm_tensor,
            )

        result = {"efficiency": output["efficiency"].item()}
        if "discrimination" in output:
            result["neural_disc"] = output["discrimination"].item()
            result["disc_method"] = "neural_enhanced" if (self._n_thermo > 0 or self._pos_embed_dim > 0) else "neural"
            if mm_position is not None:
                result["mm_position_pam"] = mm_position
            if thermo_tensor is not None:
                # Return raw ddG (first feature, un-normalized)
                raw_ddg = thermo_tensor[0, 0].item() * self._thermo_norm_std[0] + self._thermo_norm_mean[0]
                result["thermo_ddg"] = round(raw_ddg, 3)
        return result

    def _compute_thermo_feats(
        self,
        candidate: CrRNACandidate,
        wt_context_34: str,
        mm_position: int,
    ) -> Optional["torch.Tensor"]:
        """Compute 3 thermodynamic features for a MUT/WT pair."""
        import torch
        try:
            compass_ml_str = str(_COMPASS_NET_DIR)
            if compass_ml_str not in sys.path:
                sys.path.insert(0, compass_ml_str)
            from features.thermodynamic import RNA_DNA_NN, compute_hybrid_dg

            # Extract crRNA from candidate spacer
            spacer_dna = candidate.spacer_seq
            crrna = "".join(
                _DNA_TO_RNA_RC.get(b, "N") for b in reversed(spacer_dna.upper())
            )

            # ddg_hybrid: dinucleotide contribution at mismatch position
            T = 37.0 + 273.15
            idx = mm_position - 1
            penalty = 0.0
            for i in [idx - 1, idx]:
                if 0 <= i < len(crrna) - 1:
                    dinuc = crrna[i:i + 2]
                    if dinuc in RNA_DNA_NN:
                        dH, dS = RNA_DNA_NN[dinuc]
                        penalty += dH - T * (dS / 1000.0)
            ddg_hybrid = penalty

            # cumulative_dg_at_mm
            cumulative = [0.0]
            running = 0.0
            for i in range(len(crrna) - 1):
                dinuc = crrna[i:i + 2]
                if dinuc in RNA_DNA_NN:
                    dH, dS = RNA_DNA_NN[dinuc]
                    step = dH - T * (dS / 1000.0)
                else:
                    step = -1.0
                running += step
                cumulative.append(running)
            cum_idx = min(mm_position, len(cumulative) - 1)
            cumulative_dg = cumulative[cum_idx]

            # local_dg
            steps = []
            for i in [idx - 1, idx]:
                if 0 <= i < len(crrna) - 1:
                    dinuc = crrna[i:i + 2]
                    if dinuc in RNA_DNA_NN:
                        dH, dS = RNA_DNA_NN[dinuc]
                        steps.append(dH - T * (dS / 1000.0))
            local_dg = sum(steps) / max(len(steps), 1)

            feats = torch.tensor([ddg_hybrid, cumulative_dg, local_dg], dtype=torch.float32)

            # Z-score normalise using training stats
            if self._thermo_norm_mean and self._thermo_norm_std:
                mean = torch.tensor(self._thermo_norm_mean, dtype=torch.float32)
                std = torch.tensor(self._thermo_norm_std, dtype=torch.float32)
                feats = (feats - mean) / (std + 1e-8)

            return feats.unsqueeze(0)  # (1, 3)
        except Exception as e:
            logger.warning("Failed to compute thermo features: %s", e)
            return None

    # ------------------------------------------------------------------
    # Private — model loading
    # ------------------------------------------------------------------

    def _load_model(
        self,
        path: Path,
        use_rnafm: bool,
        use_rlpa: bool,
        multitask: bool,
    ) -> None:
        """Load Compass-ML from checkpoint.

        Auto-detects multitask capability: if checkpoint contains disc_head
        keys, enables multitask even if not explicitly requested.
        """
        try:
            import torch
            import importlib

            # compass-net/ has a hyphen so can't be imported directly.
            # Register it as 'compass_ml' package in sys.modules.
            compass_ml_str = str(_COMPASS_NET_DIR)
            if "compass_ml" not in sys.modules:
                if compass_ml_str not in sys.path:
                    sys.path.insert(0, compass_ml_str)
                spec = importlib.util.spec_from_file_location(
                    "compass_ml",
                    str(_COMPASS_NET_DIR / "__init__.py"),
                    submodule_search_locations=[compass_ml_str],
                )
                mod = importlib.util.module_from_spec(spec)
                sys.modules["compass_ml"] = mod
                spec.loader.exec_module(mod)

            from compass_ml import CompassML

            self._device = torch.device(self._device_name)

            checkpoint = torch.load(
                str(path), map_location=self._device, weights_only=False,
            )

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Auto-detect multitask: if disc_head keys present, enable it
            has_disc_head = any("disc_head" in k for k in state_dict.keys())
            if has_disc_head and not multitask:
                logger.info(
                    "Checkpoint contains disc_head — auto-enabling multitask mode"
                )
                multitask = True
                self._multitask = True

            # Auto-detect enhanced disc head features from checkpoint
            n_thermo = 0
            pos_embed_dim = 0
            if has_disc_head:
                for k in state_dict:
                    if "pos_embedding.weight" in k:
                        pos_embed_dim = state_dict[k].shape[1]
                    if "disc_head.head.0.weight" in k:
                        disc_input = state_dict[k].shape[1]
                        # base = 512 (4 * 128), extra = thermo + pos
                for k in state_dict:
                    if "disc_head.head.0.weight" in k:
                        disc_input = state_dict[k].shape[1]
                        n_thermo = disc_input - 512 - pos_embed_dim
                        if n_thermo < 0:
                            n_thermo = 0
                        break

            self._n_thermo = n_thermo
            self._pos_embed_dim = pos_embed_dim

            self.model = CompassML(
                use_rnafm=use_rnafm,
                use_rloop_attention=use_rlpa,
                multitask=multitask,
                n_thermo=n_thermo,
                pos_embed_dim=pos_embed_dim,
            )

            # Handle partial loading: filter out unexpected keys (e.g.
            # domain_head from training) and allow missing keys (e.g.
            # disc_head when loading RLPA checkpoint into multitask model)
            model_keys = set(self.model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            missing = model_keys - ckpt_keys
            unexpected = ckpt_keys - model_keys
            if unexpected:
                logger.info(
                    "Compass-ML: filtering %d unexpected keys from checkpoint: %s",
                    len(unexpected),
                    list(unexpected)[:5],
                )
                state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            if missing:
                logger.info(
                    "Compass-ML: %d keys missing from checkpoint (expected for partial load): %s",
                    len(missing),
                    list(missing)[:5],
                )
            self.model.load_state_dict(state_dict, strict=False)

            self.model.to(self._device)
            self.model.eval()

            self._val_rho = (
                checkpoint.get("val_rho")
                or checkpoint.get("val_eff_rho")
                or checkpoint.get("best_val_rho")
            )
            n_params = sum(p.numel() for p in self.model.parameters())

            # Store disc head metadata
            self._disc_val_r = checkpoint.get("val_disc_r")
            self._checkpoint_meta = {
                k: v for k, v in checkpoint.items()
                if k != "model_state_dict"
            }

            # Thermo normalisation params from enhanced checkpoint
            metadata = checkpoint.get("metadata", {})
            self._thermo_norm_mean = metadata.get("thermo_norm_mean", [])
            self._thermo_norm_std = metadata.get("thermo_norm_std", [])

            logger.info(
                "Loaded Compass-ML from %s (%d params, val_rho=%.4f, rnafm=%s, rlpa=%s, mt=%s)",
                path, n_params, self._val_rho or 0.0,
                use_rnafm, use_rlpa, multitask,
            )
            if has_disc_head:
                logger.info(
                    "  Discrimination head active (val_disc_r=%.4f)",
                    self._disc_val_r or 0.0,
                )
        except Exception as e:
            logger.warning("Failed to load Compass-ML from %s: %s", path, e)
            self.model = None

    def _load_rnafm_cache(self, cache_dir: Path) -> None:
        """Load RNA-FM embedding cache."""
        try:
            # Import EmbeddingCache from compass-net's data subpackage
            compass_ml_str = str(_COMPASS_NET_DIR)
            if compass_ml_str not in sys.path:
                sys.path.insert(0, compass_ml_str)

            # Direct file import to avoid subpackage registration issues
            import importlib.util
            cache_module_path = _COMPASS_NET_DIR / "data" / "embedding_cache.py"
            spec = importlib.util.spec_from_file_location(
                "compass_ml_embedding_cache", str(cache_module_path),
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            EmbeddingCache = mod.EmbeddingCache

            self._rnafm_cache = EmbeddingCache(str(cache_dir))
            logger.info(
                "Loaded RNA-FM cache from %s (%d sequences)",
                cache_dir, len(self._rnafm_cache),
            )
        except Exception as e:
            logger.warning("Failed to load RNA-FM cache: %s", e)

    def _load_calibration(self, path: Path) -> None:
        """Load temperature and ensemble weight from calibration JSON."""
        if not path.exists():
            logger.info("No Compass-ML calibration at %s — using raw scores", path)
            return
        try:
            with open(path) as f:
                cal = json.load(f)
            self.temperature = cal.get("temperature", 1.0)
            self.alpha = cal.get("alpha", 0.0)
            self.calibrated = True
            self._calibration_meta = cal
            logger.info(
                "Loaded Compass-ML calibration: T=%.2f, alpha=%.4f (val ensemble rho=%.4f)",
                self.temperature,
                self.alpha,
                cal.get("val_rho_ensemble", 0.0),
            )
        except Exception as e:
            logger.warning("Failed to load Compass-ML calibration from %s: %s", path, e)

    # ------------------------------------------------------------------
    # Private — encoding
    # ------------------------------------------------------------------

    def _encode_context(self, candidate: CrRNACandidate) -> np.ndarray:
        """Build the 34-nt context window and one-hot encode it.

        Layout: [PAM 4nt] [spacer 20-23nt] [downstream padding to 34nt]
        Same convention as SequenceMLScorer for consistency.
        """
        window = extract_input_window(
            pam=candidate.pam_seq,
            spacer=candidate.spacer_seq,
            upstream_flank="",
            downstream_flank="",
            total_len=_CONTEXT_LENGTH,
        )
        return one_hot_encode(window, max_len=_CONTEXT_LENGTH)

    def _get_rnafm_embedding(self, candidate: CrRNACandidate) -> Optional[np.ndarray]:
        """Look up RNA-FM embedding for a candidate's crRNA spacer.

        The crRNA spacer is the reverse complement of the protospacer
        (target DNA), with T→U conversion for RNA.

        Returns (20, 640) numpy array or None on cache miss.
        """
        if self._rnafm_cache is None:
            return None

        # crRNA spacer = reverse complement of DNA spacer, T→U
        spacer_dna = candidate.spacer_seq
        crrna_rna = "".join(
            _DNA_TO_RNA_RC.get(b, "N") for b in reversed(spacer_dna.upper())
        )

        emb = self._rnafm_cache.get(crrna_rna)
        if emb is None:
            return None
        return emb.numpy()

    # ------------------------------------------------------------------
    # Private — prediction
    # ------------------------------------------------------------------

    def _predict_single(self, candidate: CrRNACandidate) -> float:
        """Single-sample prediction."""
        context = self._encode_context(candidate)
        rnafm_emb = self._get_rnafm_embedding(candidate)
        predictions = self._predict_batch([context], [rnafm_emb])
        return predictions[0]

    # Alias for pipeline runner compatibility (calls _predict on ml_scorer)
    def _predict(self, candidate: CrRNACandidate) -> float:
        return self._predict_single(candidate)

    def _predict_batch(
        self,
        contexts: list[np.ndarray],
        rnafm_embs: list[Optional[np.ndarray]],
    ) -> list[float]:
        """Batch prediction. Returns list of efficiency scores in [0, 1].

        Side effect: if collect_embeddings is True, stores the last batch's
        128-dim pooled embeddings in self._last_batch_embeddings.
        """
        self._last_batch_embeddings = None

        if self.model is None:
            return [0.5] * len(contexts)

        import torch

        # Stack target DNA one-hot tensors
        batch = torch.tensor(
            np.stack(contexts), dtype=torch.float32,
        ).to(self._device)

        # Stack RNA-FM embeddings (use zeros for cache misses)
        rnafm_batch = None
        if self._use_rnafm:
            emb_list = []
            for emb in rnafm_embs:
                if emb is not None:
                    emb_list.append(emb)
                else:
                    # Zero embedding for cache miss — model degrades gracefully
                    emb_list.append(np.zeros((20, 640), dtype=np.float32))
            rnafm_batch = torch.tensor(
                np.stack(emb_list), dtype=torch.float32,
            ).to(self._device)

        with torch.no_grad():
            output = self.model(
                target_onehot=batch,
                crrna_rnafm_emb=rnafm_batch,
            )

        # Capture embeddings if requested (128-dim pooled RLPA vectors)
        if self.collect_embeddings and "embedding" in output:
            self._last_batch_embeddings = output["embedding"].cpu().numpy()

        return output["efficiency"].squeeze(-1).clamp(0, 1).tolist()

    def get_collected_embeddings(self) -> list[dict]:
        """Return all collected embeddings."""
        return self._collected_embeddings

    def clear_embeddings(self) -> None:
        """Free collected embeddings memory."""
        self._collected_embeddings = []
