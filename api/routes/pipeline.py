"""Pipeline submission and job tracking endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import JobResponse, PipelineRunRequest
from api.state import AppState, PipelineJob
from compass.core.enzyme import list_enzymes, get_enzyme, DEFAULT_ENZYME_ID
from compass.core.organisms import list_organisms, load_organism

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


@router.get("/organisms")
async def get_organisms() -> dict:
    """Return available organism profiles for multi-species panel design."""
    organism_ids = list_organisms()
    organisms = []
    for oid in sorted(organism_ids):
        try:
            profile = load_organism(oid)
            sc = profile.species_control
            organisms.append({
                "id": oid,
                "name": profile.name,
                "reference_accession": profile.reference_accession,
                "genome_gc": profile.genome_gc,
                "genome_length": profile.genome_length,
                "gene_count": len(profile.gene_synonyms),
                "species_control": sc.name if sc else None,
            })
        except Exception:
            organisms.append({"id": oid, "name": oid, "error": "failed to load"})
    return {"organisms": organisms, "default": "mtb"}


@router.get("/parameters")
async def get_parameters():
    """Return available config overrides with types, ranges, and defaults.

    Frontend can use this to render dynamic parameter controls
    without hardcoding parameter names or constraints.
    """
    return {
        "overrides": {
            "gc_min": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.40, "section": "candidates", "description": "Minimum GC content for spacer"},
            "gc_max": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.60, "section": "candidates", "description": "Maximum GC content for spacer"},
            "homopolymer_max": {"type": "int", "min": 2, "max": 8, "default": 4, "section": "candidates", "description": "Maximum homopolymer run length"},
            "mfe_threshold": {"type": "float", "min": -10.0, "max": 0.0, "default": -2.0, "section": "candidates", "description": "Minimum folding energy threshold (kcal/mol)"},
            "spacer_lengths": {"type": "list[int]", "min": 18, "max": 23, "default": [20, 21, 23], "section": "candidates", "description": "Allowed spacer lengths (nt)"},
            "scorer": {"type": "str", "choices": ["compass_ml", "seq_cnn", "heuristic"], "default": "compass_ml", "section": "scoring", "description": "Efficiency scoring backend"},
            "compass_ml_use_rlpa": {"type": "bool", "default": False, "section": "scoring", "description": "Enable R-loop attention (recommended: off for Phase 1)"},
            "compass_ml_use_rnafm": {"type": "bool", "default": True, "section": "scoring", "description": "Use RNA-FM embeddings for crRNA features"},
            "discrimination_min_ratio": {"type": "float", "min": 1.0, "max": 20.0, "default": 2.0, "section": "scoring", "description": "Minimum discrimination ratio for clinical use"},
            "max_plex": {"type": "int", "min": 1, "max": 30, "default": 14, "section": "multiplex", "description": "Maximum panel size (number of targets)"},
            "efficiency_weight": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5, "section": "multiplex", "description": "Weight for efficiency in panel optimization"},
            "discrimination_weight": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.2, "section": "multiplex", "description": "Weight for discrimination in panel optimization"},
            "tm_min": {"type": "float", "min": 40.0, "max": 80.0, "default": 57.0, "section": "primers", "description": "Minimum primer melting temperature (°C)"},
            "tm_max": {"type": "float", "min": 50.0, "max": 85.0, "default": 72.0, "section": "primers", "description": "Maximum primer melting temperature (°C)"},
            "amplicon_min": {"type": "int", "min": 50, "max": 200, "default": 80, "section": "primers", "description": "Minimum amplicon length (bp)"},
            "amplicon_max": {"type": "int", "min": 80, "max": 500, "default": 250, "section": "primers", "description": "Maximum amplicon length (bp)"},
            "sample_type": {"type": "str", "choices": ["genomic", "cfDNA"], "default": "genomic", "section": "primers", "description": "Sample type (cfDNA caps amplicon at 120bp)"},
            "sm_enabled": {"type": "bool", "default": True, "section": "synthetic_mismatch", "description": "Enable synthetic mismatch enhancement"},
        }
    }
