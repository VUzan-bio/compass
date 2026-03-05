"""Block 3: Sensitivity-Specificity Optimization API endpoints.

GET  /api/v1/presets                            — list available parameter presets
GET  /api/v1/panel/{job_id}/diagnostics         — panel-level diagnostics
GET  /api/v1/panel/{job_id}/who_compliance      — per-drug-class WHO TPP compliance
GET  /api/v1/panel/{job_id}/top_k/{target_label} — top-K candidates for one target
POST /api/v1/panel/{job_id}/sweep               — sweep one parameter
POST /api/v1/panel/{job_id}/pareto              — compute Pareto frontier

All endpoints operate on COMPLETED pipeline runs. They re-evaluate
existing candidates against different thresholds — they do NOT re-run
the pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from compass.optimisation.profiles import list_presets, get_preset, ParameterProfile
from compass.optimisation.sweep import sweep_parameter
from compass.optimisation.pareto import pareto_frontier
from compass.optimisation.top_k import collect_top_k
from compass.optimisation.metrics import compute_diagnostic_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["optimisation"])

# In-memory cache for pipeline results, keyed by job_id
_job_cache: dict[str, dict] = {}


def cache_pipeline_result(
    job_id: str,
    members: list,
    candidates_by_target: dict,
) -> None:
    """Called by the pipeline route after panel assembly to cache results."""
    _job_cache[job_id] = {
        "members": members,
        "candidates": candidates_by_target,
    }


def _get_job(job_id: str) -> dict:
    """Retrieve cached pipeline result or raise 404."""
    if job_id not in _job_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No pipeline results for job '{job_id}'. Run the pipeline first.",
        )
    return _job_cache[job_id]


# --- Request/Response models ---

class SweepRequest(BaseModel):
    parameter_name: str = Field(
        description="Parameter to sweep: 'efficiency_threshold' or 'discrimination_threshold'"
    )
    values: list[float] = Field(
        description="Values to evaluate",
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )
    base_preset: str = Field(
        default="balanced",
        description="Base profile preset name",
    )


class ParetoRequest(BaseModel):
    disc_values: Optional[list[float]] = Field(
        default=None,
        description="Custom discrimination threshold grid (optional)",
    )
    score_values: Optional[list[float]] = Field(
        default=None,
        description="Custom efficiency threshold grid (optional)",
    )


# --- Endpoints ---

@router.get("/presets")
async def get_presets() -> list[dict]:
    """Return all available parameter presets with descriptions."""
    return list_presets()


@router.get("/panel/{job_id}/diagnostics")
async def get_diagnostics(job_id: str, preset: str = "balanced") -> dict:
    """Return DiagnosticMetrics.summary() for a completed pipeline run.

    Uses the specified preset's thresholds to determine which targets
    are covered. Default: balanced (WHO TPP).
    """
    job = _get_job(job_id)
    try:
        profile = get_preset(preset)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    metrics = compute_diagnostic_metrics(
        members=job["members"],
        candidates_by_target=job["candidates"],
        efficiency_threshold=profile.efficiency_threshold,
        discrimination_threshold=profile.discrimination_threshold,
    )
    return metrics.summary()


@router.get("/panel/{job_id}/who_compliance")
async def get_who_compliance(job_id: str, preset: str = "balanced") -> dict:
    """Return per-drug-class WHO TPP compliance status."""
    job = _get_job(job_id)
    try:
        profile = get_preset(preset)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    metrics = compute_diagnostic_metrics(
        members=job["members"],
        candidates_by_target=job["candidates"],
        efficiency_threshold=profile.efficiency_threshold,
        discrimination_threshold=profile.discrimination_threshold,
    )
    return {
        "preset": preset,
        "who_compliance": metrics.who_compliance,
        "panel_sensitivity": round(metrics.sensitivity, 3),
        "panel_specificity": round(metrics.specificity, 3),
    }


@router.get("/panel/{job_id}/top_k/{target_label}")
async def get_target_top_k(job_id: str, target_label: str, k: int = 5) -> dict:
    """Return top-K candidates for one target with tradeoff annotations."""
    job = _get_job(job_id)

    results = collect_top_k(
        members=job["members"],
        candidates_by_target=job["candidates"],
        k=k,
    )

    for r in results:
        if r.target_label == target_label:
            return r.to_dict()

    raise HTTPException(
        status_code=404,
        detail=f"Target '{target_label}' not found in panel.",
    )


@router.post("/panel/{job_id}/sweep")
async def run_sweep(job_id: str, request: SweepRequest) -> dict:
    """Sweep one parameter and return the sensitivity/specificity curve."""
    job = _get_job(job_id)

    try:
        base_profile = get_preset(request.base_preset)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = sweep_parameter(
        parameter_name=request.parameter_name,
        values=request.values,
        members=job["members"],
        candidates_by_target=job["candidates"],
        base_profile=base_profile,
    )

    return result.to_dict()


@router.post("/panel/{job_id}/pareto")
async def run_pareto(job_id: str, request: ParetoRequest) -> dict:
    """Compute the Pareto frontier over efficiency and discrimination thresholds."""
    job = _get_job(job_id)

    frontier = pareto_frontier(
        members=job["members"],
        candidates_by_target=job["candidates"],
        disc_values=request.disc_values,
        score_values=request.score_values,
    )

    return {
        "n_points": len(frontier),
        "frontier": [p.to_dict() for p in frontier],
    }
