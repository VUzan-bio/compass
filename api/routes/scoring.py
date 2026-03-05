"""Scoring model info and JEPA placeholder endpoints (Phase 4)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from api.schemas import JEPAScoreRequest, JEPAScoreResponse, ScoringModelInfo

router = APIRouter(prefix="/api/scoring", tags=["scoring"])


@router.get("/models", response_model=list[ScoringModelInfo])
async def list_models() -> list[ScoringModelInfo]:
    """List available scoring models and their status."""
    models = [
        ScoringModelInfo(
            name="heuristic",
            status="ready",
            description="Rule-based scoring (Kim 2018 features): seed position, GC, structure, off-targets",
        ),
    ]

    # Check for CNN weights
    cnn_status = "no_weights"
    try:
        from compass.scoring.sequence_ml import SequenceMLScorer
        if Path("models/seq_cnn.pt").exists():
            cnn_status = "ready"
    except ImportError:
        cnn_status = "disabled"
    models.append(ScoringModelInfo(
        name="seq_cnn",
        status=cnn_status,
        description="Sequence-based CNN (Seq-deepCpf1 equivalent)",
    ))

    # Check for JEPA weights
    jepa_status = "no_weights"
    try:
        from compass.scoring.jepa import JEPAScorer
        if Path("models/jepa_encoder.pt").exists():
            jepa_status = "ready"
    except ImportError:
        jepa_status = "disabled"
    models.append(ScoringModelInfo(
        name="jepa",
        status=jepa_status,
        description="bDNA-JEPA fine-tuned predictor (Paths A/B/C)",
    ))

    return models


@router.post("/jepa", response_model=JEPAScoreResponse)
async def score_with_jepa(req: JEPAScoreRequest) -> JEPAScoreResponse:
    """Score spacer sequences with JEPA model (placeholder)."""
    # Placeholder: returns dummy predictions until JEPA weights are available
    predictions = []
    for seq in req.spacer_sequences:
        predictions.append({
            "spacer": seq,
            "efficiency": 0.0,
            "embedding": [],
            "status": "no_weights",
            "message": "JEPA encoder not yet available. Train and place weights at models/jepa_encoder.pt",
        })
    return JEPAScoreResponse(predictions=predictions)
