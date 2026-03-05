"""Experimental validation and active learning endpoints (Phase 4)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import UncertaintyCandidate, ValidationUpload

router = APIRouter(prefix="/api/validation", tags=["validation"])

VALIDATION_DIR = Path("results/validation")


@router.post("/upload", status_code=201)
async def upload_measurements(req: ValidationUpload) -> dict:
    """Upload experimental measurements for active learning."""
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    batch_id = uuid.uuid4().hex[:12]
    batch_path = VALIDATION_DIR / f"{batch_id}.json"

    data = {
        "batch_id": batch_id,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "n_measurements": len(req.measurements),
        "measurements": req.measurements,
    }

    with open(batch_path, "w") as f:
        json.dump(data, f, indent=2)

    return {
        "batch_id": batch_id,
        "n_measurements": len(req.measurements),
        "status": "stored",
    }


@router.get("/uncertainty", response_model=list[UncertaintyCandidate])
async def get_uncertainty_ranking() -> list[UncertaintyCandidate]:
    """Return candidates ranked by model uncertainty (placeholder).

    When JEPA is trained, this will return candidates where the model
    is least confident — the optimal next batch for wet-lab validation.
    """
    return []
