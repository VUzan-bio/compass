"""Pipeline submission and job tracking endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import JobResponse, PipelineRunRequest
from api.state import AppState, PipelineJob
from compass.core.enzyme import list_enzymes, get_enzyme, DEFAULT_ENZYME_ID

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# Injected at startup by main.py
_state: AppState | None = None


def init(state: AppState) -> None:
    global _state
    _state = state


def _get_state() -> AppState:
    if _state is None:
        raise RuntimeError("AppState not initialized")
    return _state


@router.post("/run", response_model=JobResponse, status_code=202)
async def submit_run(req: PipelineRunRequest) -> JobResponse:
    """Submit a pipeline run. Returns immediately with job ID."""
    state = _get_state()

    # Merge enzyme_id into config_overrides if provided
    overrides = dict(req.config_overrides)
    if req.enzyme_id:
        overrides.setdefault("candidates", {})
        if isinstance(overrides["candidates"], dict):
            overrides["candidates"]["enzyme_id"] = req.enzyme_id

    job = PipelineJob(
        name=req.name,
        mode=req.mode,
        mutations=req.mutations,
        config_overrides=overrides,
    )
    state.submit_job(job)
    return JobResponse(**job.to_response())


@router.get("/jobs", response_model=list[JobResponse])
async def list_jobs() -> list[JobResponse]:
    """List all jobs, newest first."""
    state = _get_state()
    return [JobResponse(**j.to_response()) for j in state.list_jobs()]


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    """Get status of a specific job."""
    state = _get_state()
    job = state.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return JobResponse(**job.to_response())


@router.get("/enzymes")
async def get_enzymes() -> dict:
    """Return available Cas12a enzyme variants with PAM specs."""
    enzymes = list_enzymes()
    return {
        "enzymes": [e.to_dict() for e in enzymes],
        "default": DEFAULT_ENZYME_ID,
    }
