"""Figure generation endpoints.

Renders publication-quality matplotlib figures from pipeline results.
Figures are cached on disk after first generation.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from api.state import AppState, JobStatus

router = APIRouter(prefix="/api/figures", tags=["figures"])

_state: AppState | None = None


def init(state: AppState) -> None:
    global _state
    _state = state


def _get_state() -> AppState:
    if _state is None:
        raise RuntimeError("AppState not initialized")
    return _state


def _load_result(job_id: str) -> dict[str, Any]:
    state = _get_state()
    job = state.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, f"Job not completed (status: {job.status.value})")

    if job.result:
        return job.result

    result_path = state.results_dir / f"{job_id}.json"
    if result_path.exists():
        with open(result_path) as f:
            return json.load(f)

    raise HTTPException(500, "Result data not found")


def _get_cached_or_generate(job_id: str, fig_type: str, generator: Any) -> Response:
    """Check cache, generate if missing, return PNG."""
    state = _get_state()
    cache_dir = state.results_dir / job_id / "figures"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{fig_type}.png"

    if cache_path.exists():
        return Response(
            content=cache_path.read_bytes(),
            media_type="image/png",
            headers={"Cache-Control": "max-age=3600"},
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = generator()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()

    cache_path.write_bytes(png_bytes)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "max-age=3600"},
    )


@router.get("/{job_id}/discrimination")
async def discrimination_figure(job_id: str) -> Response:
    """Panel discrimination summary bar chart."""
    result = _load_result(job_id)

    def generate():
        from compass.viz.discrimination import DiscriminationHeatmap

        members = result.get("members", [])
        ratios = {}
        for m in members:
            sel = m.get("selected_candidate", {})
            disc = sel.get("discrimination")
            mutation = m.get("target", {}).get("mutation", {})
            gene = mutation.get("gene", "")
            ref = mutation.get("ref_aa", "")
            pos = mutation.get("position", "")
            alt = mutation.get("alt_aa", "")
            label = f"{gene}_{ref}{pos}{alt}"

            if disc:
                wt = disc.get("wt_activity", 0)
                mut = disc.get("mut_activity", 0)
                ratio = (mut / wt) if wt > 0 else (10.0 if mut > 0 else 0.0)
                ratios[label] = ratio
            else:
                ratios[label] = 0.0

        hmap = DiscriminationHeatmap()
        return hmap.plot_panel_summary(ratios)

    return _get_cached_or_generate(job_id, "discrimination", generate)


@router.get("/{job_id}/ranking")
async def ranking_figure(job_id: str) -> Response:
    """Candidate ranking for the top candidate per target."""
    result = _load_result(job_id)

    def generate():
        from compass.viz.ranking import CandidateRankingPlot

        members = result.get("members", [])
        top_per_target = {}
        for m in members:
            sel = m.get("selected_candidate", {})
            heur = sel.get("heuristic", {})
            cand = sel.get("candidate", {})
            mutation = m.get("target", {}).get("mutation", {})
            gene = mutation.get("gene", "")
            ref = mutation.get("ref_aa", "")
            pos = mutation.get("position", "")
            alt = mutation.get("alt_aa", "")
            label = f"{gene}_{ref}{pos}{alt}"

            top_per_target[label] = {
                "composite": heur.get("composite", 0),
                "status": sel.get("validation_status", "untested"),
            }

        plot = CandidateRankingPlot()
        return plot.plot_multi_target_top(top_per_target)

    return _get_cached_or_generate(job_id, "ranking", generate)


@router.get("/{job_id}/multiplex")
async def multiplex_figure(job_id: str) -> Response:
    """Cross-reactivity matrix visualization."""
    result = _load_result(job_id)

    def generate():
        import numpy as np
        from compass.viz.multiplex import MultiplexMatrixPlot

        matrix = result.get("cross_reactivity_matrix")
        members = result.get("members", [])
        labels = []
        for m in members:
            mutation = m.get("target", {}).get("mutation", {})
            gene = mutation.get("gene", "")
            ref = mutation.get("ref_aa", "")
            pos = mutation.get("position", "")
            alt = mutation.get("alt_aa", "")
            labels.append(f"{gene}_{ref}{pos}{alt}")

        if matrix is None:
            n = len(labels) or 1
            matrix = np.zeros((n, n))
        else:
            matrix = np.array(matrix)

        plot = MultiplexMatrixPlot()
        return plot.plot_cross_reactivity(matrix, labels)

    return _get_cached_or_generate(job_id, "multiplex", generate)


@router.get("/{job_id}/dashboard")
async def dashboard_figure(job_id: str) -> Response:
    """Target dashboard overview."""
    result = _load_result(job_id)

    def generate():
        from compass.viz.target_overview import TargetDashboard

        members = result.get("members", [])
        targets_data = []
        for m in members:
            mutation = m.get("target", {}).get("mutation", {})
            sel = m.get("selected_candidate", {})
            heur = sel.get("heuristic", {})

            targets_data.append({
                "gene": mutation.get("gene", ""),
                "mutation": f"{mutation.get('ref_aa', '')}{mutation.get('position', '')}{mutation.get('alt_aa', '')}",
                "drug": mutation.get("drug", "OTHER"),
                "n_candidates": 1,
                "top_score": heur.get("composite", 0),
                "status": sel.get("validation_status", "untested"),
            })

        dash = TargetDashboard()
        return dash.plot_dashboard(targets_data)

    return _get_cached_or_generate(job_id, "dashboard", generate)
