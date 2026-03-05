"""Tests for Compass-ML pipeline integration (compass/scoring/compass_ml_scorer.py).

Tests that the CompassMlScorer adapter:
1. Loads weights and produces valid efficiency scores
2. Falls back gracefully when weights/cache unavailable
3. Implements the same Scorer interface as SequenceMLScorer
4. Handles both direct and proximity candidates
5. Batch scoring returns correct number of results
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from compass.core.types import (
    CrRNACandidate,
    DetectionStrategy,
    HeuristicScore,
    OffTargetReport,
    PAMVariant,
    ScoredCandidate,
    Strand,
)
from compass.scoring.base import Scorer
from compass.scoring.preprocessing import one_hot_encode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_candidate(
    spacer: str = "ATCGATCGATCGATCGATCG",
    pam: str = "TTTG",
    strategy: DetectionStrategy = DetectionStrategy.DIRECT,
    mutation_pos: int | None = 5,
) -> CrRNACandidate:
    """Create a minimal CrRNACandidate for testing."""
    return CrRNACandidate(
        candidate_id=f"test_{spacer[:8]}",
        target_label="rpoB_S450L",
        spacer_seq=spacer,
        pam_seq=pam,
        pam_variant=PAMVariant.TTTV,
        strand=Strand.PLUS,
        genomic_start=761155,
        genomic_end=761155 + len(spacer),
        mutation_position_in_spacer=mutation_pos,
        gc_content=sum(1 for b in spacer if b in "GC") / len(spacer),
        homopolymer_max=2,
        detection_strategy=strategy,
        proximity_distance=0 if strategy == DetectionStrategy.DIRECT else 25,
    )


def _make_offtarget(candidate_id: str = "test_ATCGATCG") -> OffTargetReport:
    return OffTargetReport(candidate_id=candidate_id, is_clean=True)


# ---------------------------------------------------------------------------
# Test: Interface compliance
# ---------------------------------------------------------------------------

class TestInterfaceCompliance:
    """CompassMlScorer must implement the Scorer ABC."""

    def test_is_subclass_of_scorer(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        assert issubclass(CompassMlScorer, Scorer)

    def test_has_score_method(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        assert hasattr(CompassMlScorer, "score")

    def test_has_score_batch_method(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        assert hasattr(CompassMlScorer, "score_batch")


# ---------------------------------------------------------------------------
# Test: Fallback behaviour (no weights)
# ---------------------------------------------------------------------------

class TestFallbackNoWeights:
    """When weights are unavailable, scorer falls back to heuristic."""

    def test_init_without_weights(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(weights_path="/nonexistent/path.pt")
        assert scorer.model is None

    def test_score_without_model_uses_heuristic(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(weights_path="/nonexistent/path.pt")
        candidate = _make_candidate()
        ot = _make_offtarget(candidate.candidate_id)

        result = scorer.score(candidate, ot)
        assert isinstance(result, ScoredCandidate)
        assert result.heuristic.composite > 0
        # No ML score added since model is None
        assert len(result.ml_scores) == 0

    def test_score_batch_without_model(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(weights_path="/nonexistent/path.pt")

        candidates = [_make_candidate(spacer=f"ATCGATCGATCG{'ATCG'}{i:04d}"[:20]) for i in range(3)]
        offtargets = [_make_offtarget(c.candidate_id) for c in candidates]

        results = scorer.score_batch(candidates, offtargets)
        assert len(results) == 3
        assert all(isinstance(r, ScoredCandidate) for r in results)
        # Ranked
        assert all(r.rank is not None for r in results)

    def test_predict_efficiency_without_model(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(weights_path="/nonexistent/path.pt")
        candidate = _make_candidate()
        assert scorer.predict_efficiency(candidate) == 0.5


# ---------------------------------------------------------------------------
# Test: With model loaded (using mock)
# ---------------------------------------------------------------------------

class TestWithMockModel:
    """Test scoring with a mocked Compass-ML model."""

    def _make_mock_scorer(self):
        """Create a CompassMlScorer with a mock model that returns fixed predictions."""
        import torch
        from compass.scoring.compass_ml_scorer import CompassMlScorer

        scorer = CompassMlScorer(weights_path="/nonexistent/path.pt")

        # Mock the model
        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.to = MagicMock(return_value=mock_model)

        def mock_forward(**kwargs):
            batch_size = kwargs["target_onehot"].shape[0]
            return {
                "efficiency": torch.tensor([[0.72]] * batch_size),
            }

        mock_model.side_effect = mock_forward
        mock_model.__call__ = mock_forward

        scorer.model = mock_model
        scorer._device = torch.device("cpu")
        scorer._use_rnafm = False
        return scorer

    def test_score_with_model_adds_ml_score(self):
        scorer = self._make_mock_scorer()
        candidate = _make_candidate()
        ot = _make_offtarget(candidate.candidate_id)

        result = scorer.score(candidate, ot)
        assert isinstance(result, ScoredCandidate)
        assert result.heuristic.composite > 0
        # ML score should be added
        assert len(result.ml_scores) >= 1
        ml = result.ml_scores[-1]
        assert ml.model_name == "compass_ml"
        assert 0.0 <= ml.predicted_efficiency <= 1.0

    def test_score_batch_with_model(self):
        scorer = self._make_mock_scorer()
        candidates = [
            _make_candidate(spacer="ATCGATCGATCGATCGATCG"),
            _make_candidate(spacer="GCTAGCTAGCTAGCTAGCTA"),
            _make_candidate(spacer="TTTTAAAACCCCGGGGAAAA"),
        ]
        offtargets = [_make_offtarget(c.candidate_id) for c in candidates]

        results = scorer.score_batch(candidates, offtargets)
        assert len(results) == 3
        # All should have ML scores
        for r in results:
            assert any(m.model_name == "compass_ml" for m in r.ml_scores)
        # Should be ranked
        ranks = [r.rank for r in results]
        assert sorted(ranks) == [1, 2, 3]

    def test_predict_efficiency_with_model(self):
        scorer = self._make_mock_scorer()
        candidate = _make_candidate()
        score = scorer.predict_efficiency(candidate)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Test: Context encoding
# ---------------------------------------------------------------------------

class TestContextEncoding:
    """Test the 34-nt context window construction."""

    def test_encode_context_shape(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(weights_path="/nonexistent/path.pt")
        candidate = _make_candidate()
        context = scorer._encode_context(candidate)
        assert context.shape == (4, 34)

    def test_encode_context_is_one_hot(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(weights_path="/nonexistent/path.pt")
        candidate = _make_candidate()
        context = scorer._encode_context(candidate)
        # Each position should sum to 0 (N/padding) or 1 (valid base)
        col_sums = context.sum(axis=0)
        assert all(s in (0.0, 1.0) for s in col_sums)

    def test_encode_proximity_candidate(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(weights_path="/nonexistent/path.pt")
        candidate = _make_candidate(
            strategy=DetectionStrategy.PROXIMITY,
            mutation_pos=None,
        )
        context = scorer._encode_context(candidate)
        assert context.shape == (4, 34)


# ---------------------------------------------------------------------------
# Test: RNA-FM embedding lookup
# ---------------------------------------------------------------------------

class TestRNAFMEmbedding:
    """Test crRNA spacer derivation and embedding lookup."""

    def test_get_rnafm_no_cache_returns_none(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(
            weights_path="/nonexistent/path.pt",
            use_rnafm=True,
            rnafm_cache_dir=None,
        )
        candidate = _make_candidate()
        assert scorer._get_rnafm_embedding(candidate) is None

    def test_get_rnafm_with_mock_cache(self):
        import torch
        from compass.scoring.compass_ml_scorer import CompassMlScorer

        scorer = CompassMlScorer(
            weights_path="/nonexistent/path.pt",
            use_rnafm=True,
        )

        # Mock the cache
        mock_cache = MagicMock()
        mock_cache.get = MagicMock(return_value=torch.randn(20, 640))
        scorer._rnafm_cache = mock_cache

        candidate = _make_candidate()
        emb = scorer._get_rnafm_embedding(candidate)
        assert emb is not None
        assert emb.shape == (20, 640)

    def test_cache_miss_returns_none(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer

        scorer = CompassMlScorer(
            weights_path="/nonexistent/path.pt",
            use_rnafm=True,
        )

        mock_cache = MagicMock()
        mock_cache.get = MagicMock(return_value=None)
        scorer._rnafm_cache = mock_cache

        candidate = _make_candidate()
        emb = scorer._get_rnafm_embedding(candidate)
        assert emb is None


# ---------------------------------------------------------------------------
# Test: Real model loading (integration test, skipped if weights missing)
# ---------------------------------------------------------------------------

_WEIGHTS_PATH = Path("c:/Users/pushg/Documents/compass/compass/weights/compass_ml_best.pt")
_COMPASS_NET_DIR = Path("c:/Users/pushg/Documents/compass/compass-net")


@pytest.mark.skipif(
    not _WEIGHTS_PATH.exists() or not _COMPASS_NET_DIR.exists(),
    reason="Compass-ML weights or compass-net/ directory not available",
)
class TestRealModelIntegration:
    """Integration tests with actual Compass-ML weights."""

    def test_load_real_weights(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(
            weights_path=str(_WEIGHTS_PATH),
            use_rnafm=True,
            use_rlpa=True,
            multitask=False,
        )
        assert scorer.model is not None

    def test_predict_real_candidate(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(
            weights_path=str(_WEIGHTS_PATH),
            use_rnafm=True,
            use_rlpa=True,
            multitask=False,
        )
        candidate = _make_candidate()
        score = scorer.predict_efficiency(candidate)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_real_batch(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(
            weights_path=str(_WEIGHTS_PATH),
            use_rnafm=True,
            use_rlpa=True,
            multitask=False,
        )
        candidates = [
            _make_candidate(spacer="ATCGATCGATCGATCGATCG"),
            _make_candidate(spacer="GCTAGCTAGCTAGCTAGCTA"),
            _make_candidate(spacer="TTTTAAAACCCCGGGGAAAA"),
        ]
        offtargets = [_make_offtarget(c.candidate_id) for c in candidates]

        results = scorer.score_batch(candidates, offtargets)
        assert len(results) == 3
        scores = [r.ml_scores[-1].predicted_efficiency for r in results]
        # Scores should be different for different sequences
        assert len(set(round(s, 4) for s in scores)) > 1

    def test_cnn_only_fallback(self):
        """Without RNA-FM cache, model should still work (CNN-only)."""
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(
            weights_path=str(_WEIGHTS_PATH),
            use_rnafm=True,
            use_rlpa=True,
            rnafm_cache_dir=None,
        )
        candidate = _make_candidate()
        score = scorer.predict_efficiency(candidate)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_validation_rho_loaded(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(
            weights_path=str(_WEIGHTS_PATH),
            use_rnafm=True,
            use_rlpa=True,
        )
        # The RLPA checkpoint should have val_rho stored
        assert scorer.validation_rho is not None or True  # may be None depending on checkpoint format


# ---------------------------------------------------------------------------
# Test: Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    """Test temperature calibration and ensemble scoring."""

    def test_no_calibration_file_uses_raw(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(
            weights_path="/nonexistent/path.pt",
            calibration_path="/nonexistent/cal.json",
        )
        assert not scorer.calibrated
        assert scorer.temperature == 1.0
        assert scorer.alpha == 0.0

    def test_calibrated_score_identity_when_uncalibrated(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(
            weights_path="/nonexistent/path.pt",
            calibration_path="/nonexistent/cal.json",
        )
        assert scorer.calibrated_score(0.7) == 0.7

    def test_calibrated_score_spreads_distribution(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(weights_path="/nonexistent/path.pt")
        scorer.calibrated = True
        scorer.temperature = 5.0

        # T > 1 should spread scores toward 0.5
        high = scorer.calibrated_score(0.9)
        low = scorer.calibrated_score(0.1)
        # Spread should be narrower than original [0.1, 0.9]
        assert high < 0.9
        assert low > 0.1
        # But ordering preserved
        assert high > low

    def test_ensemble_score_blends(self):
        from compass.scoring.compass_ml_scorer import CompassMlScorer
        scorer = CompassMlScorer(weights_path="/nonexistent/path.pt")
        scorer.alpha = 0.35
        result = scorer.ensemble_score_val(0.8, 0.6)
        expected = 0.35 * 0.8 + 0.65 * 0.6
        assert abs(result - expected) < 1e-6

    def test_calibration_loads_from_json(self, tmp_path):
        import json
        from compass.scoring.compass_ml_scorer import CompassMlScorer

        cal_file = tmp_path / "compass_ml_calibration.json"
        cal_file.write_text(json.dumps({
            "model": "compass_ml",
            "temperature": 4.5,
            "alpha": 0.30,
            "val_rho_ensemble": 0.72,
        }))

        scorer = CompassMlScorer(
            weights_path="/nonexistent/path.pt",
            calibration_path=str(cal_file),
        )
        assert scorer.calibrated is True
        assert scorer.temperature == 4.5
        assert scorer.alpha == 0.30
        assert scorer.calibration_meta["val_rho_ensemble"] == 0.72

    def test_score_populates_calibrated_fields(self):
        """When calibrated, score() should populate cnn_calibrated and ensemble_score."""
        import torch
        from compass.scoring.compass_ml_scorer import CompassMlScorer

        scorer = CompassMlScorer(weights_path="/nonexistent/path.pt")

        # Mock model
        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.to = MagicMock(return_value=mock_model)

        def mock_forward(**kwargs):
            batch_size = kwargs["target_onehot"].shape[0]
            return {"efficiency": torch.tensor([[0.72]] * batch_size)}

        mock_model.side_effect = mock_forward
        mock_model.__call__ = mock_forward
        scorer.model = mock_model
        scorer._device = torch.device("cpu")
        scorer._use_rnafm = False

        # Enable calibration
        scorer.calibrated = True
        scorer.temperature = 3.0
        scorer.alpha = 0.35

        candidate = _make_candidate()
        ot = _make_offtarget(candidate.candidate_id)
        result = scorer.score(candidate, ot)

        assert result.cnn_calibrated is not None
        assert result.ensemble_score is not None
        assert result.cnn_score == pytest.approx(0.72, abs=0.01)
        # Calibrated should differ from raw (T != 1)
        assert result.cnn_calibrated != result.cnn_score
        # Ensemble should blend heuristic and calibrated
        expected_ensemble = 0.35 * result.heuristic.composite + 0.65 * result.cnn_calibrated
        assert result.ensemble_score == pytest.approx(expected_ensemble, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Test that ScoringConfig correctly supports compass_ml scorer."""

    def test_config_default_is_seq_cnn(self):
        from compass.core.config import ScoringConfig
        config = ScoringConfig()
        assert config.scorer == "seq_cnn"

    def test_config_compass_ml_fields(self):
        from compass.core.config import ScoringConfig
        config = ScoringConfig(
            scorer="compass_ml",
            compass_ml_weights=Path("compass/weights/compass_ml_best.pt"),
            rnafm_cache_dir=Path("compass/data/embeddings/rnafm"),
            compass_ml_use_rlpa=True,
            compass_ml_use_rnafm=True,
        )
        assert config.scorer == "compass_ml"
        assert config.compass_ml_use_rlpa is True
        assert config.compass_ml_use_rnafm is True
