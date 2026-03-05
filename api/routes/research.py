"""Research sandbox endpoints — experimental scoring R&D."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from api.state import AppState, JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])

_state: AppState | None = None


def init(state: AppState) -> None:
    global _state
    _state = state


def _get_state() -> AppState:
    if _state is None:
        raise RuntimeError("AppState not initialized")
    return _state


# ── Schemas ──

class CompareRequest(BaseModel):
    job_id: str
    model_a: str = "heuristic"
    model_b: str = "compass_ml"


class AblationRow(BaseModel):
    label: str
    features: str
    kim_rho: float
    ed_rho: float | None = None
    notes: str = ""


# ── POST /api/research/compare ──

@router.post("/compare")
async def compare_scorers_endpoint(req: CompareRequest) -> dict[str, Any]:
    """Compare two scoring models on a completed panel result."""
    state = _get_state()
    job = state.get_job(req.job_id)
    if job is None:
        raise HTTPException(404, f"Job {req.job_id} not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, f"Job {req.job_id} not completed")

    result = job.result
    if result is None:
        result_path = state.results_dir / f"{req.job_id}.json"
        if result_path.exists():
            with open(result_path) as f:
                result = json.load(f)
        else:
            raise HTTPException(500, "Result data not found")

    from compass.research.scorer_compare import compare_scorers
    return compare_scorers(result, req.model_a, req.model_b)


# ── GET /api/research/thermo/{job_id}/{target_label} ──

@router.get("/thermo/{job_id}/{target_label}")
async def get_thermo_profile(job_id: str, target_label: str) -> dict[str, Any]:
    """Get per-position thermodynamic profile for a target's selected spacer."""
    state = _get_state()
    job = state.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, f"Job {job_id} not completed")

    result = job.result
    if result is None:
        result_path = state.results_dir / f"{job_id}.json"
        if result_path.exists():
            with open(result_path) as f:
                result = json.load(f)
        else:
            raise HTTPException(500, "Result data not found")

    # Find the target
    spacer = None
    pam = None
    snp_pos = None

    members = result.get("members", [])
    for member in members:
        target = member.get("target", {})
        mutation = target.get("mutation", {})
        label = mutation.get("label", f"{mutation.get('gene', '')}_{mutation.get('ref_aa', '')}{mutation.get('position', '')}{mutation.get('alt_aa', '')}")
        if label == target_label:
            selected = member.get("selected_candidate", {})
            candidate = selected.get("candidate", {})
            spacer = candidate.get("spacer_seq")
            pam = candidate.get("pam_seq")
            snp_pos = candidate.get("mutation_position_in_spacer")
            break

    if not members:
        # Basic mode
        targets_data = result.get("targets", {})
        scored_list = targets_data.get(target_label, [])
        if scored_list:
            first = scored_list[0]
            cand = first.get("candidate", {})
            spacer = cand.get("spacer_seq")
            pam = cand.get("pam_seq")
            snp_pos = cand.get("mutation_position_in_spacer")

    if not spacer:
        raise HTTPException(404, f"Target {target_label} not found in job {job_id}")

    from compass.research.thermo_profile import get_thermo_profile
    return get_thermo_profile(spacer, pam or "", snp_pos)


# ── GET /api/research/thermo/standalone ──

@router.get("/thermo/standalone")
async def get_thermo_standalone(
    spacer: str = Query(..., min_length=15, max_length=30, description="DNA spacer sequence (15-30 nt)"),
    pam: str = Query("TTTV", description="PAM sequence"),
) -> dict[str, Any]:
    """Compute thermodynamic profile from a raw spacer sequence (no panel needed)."""
    spacer = spacer.upper().strip()
    invalid = set(spacer) - {"A", "T", "C", "G"}
    if invalid:
        raise HTTPException(400, f"Invalid bases in spacer: {invalid}")

    from compass.research.thermo_profile import get_thermo_profile
    return get_thermo_profile(spacer, pam, snp_position=None)


# ── GET /api/research/ablation ──

@router.get("/ablation")
async def get_ablation() -> list[dict[str, Any]]:
    """Get ablation study results."""
    from compass.research.ablation_store import load_ablation_rows
    return load_ablation_rows()


# ── POST /api/research/ablation ──

@router.post("/ablation")
async def add_ablation(row: AblationRow) -> dict[str, Any]:
    """Add a new ablation row."""
    from compass.research.ablation_store import add_ablation_row
    added = add_ablation_row(row.model_dump())
    return {"ok": True, "row": added}


# ── GET /api/research/nuclease/profiles ──

@router.get("/nuclease/profiles")
async def list_nuclease_profiles() -> dict[str, Any]:
    """List all available nuclease profiles with summaries."""
    from compass.nuclease.profile_loader import NucleaseProfile
    return {
        "profiles": [
            NucleaseProfile.load(vid).to_summary()
            for vid in NucleaseProfile.available()
        ]
    }


# ── GET /api/research/nuclease/comparison ──

@router.get("/nuclease/comparison")
async def nuclease_comparison() -> dict[str, Any]:
    """Compare PAM coverage across all loaded nuclease profiles."""
    from compass.nuclease.profile_loader import NucleaseProfile
    from compass.nuclease.pam_coverage import compare_pam_coverage

    profiles = {
        vid: NucleaseProfile.load(vid).to_summary()
        for vid in NucleaseProfile.available()
    }
    coverage = compare_pam_coverage()

    return {"profiles": profiles, "coverage": coverage}
