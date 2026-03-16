"""Result retrieval and export endpoints."""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.schemas import (
    CandidateSummary,
    ExportFormat,
    ModuleStats,
    PanelSummary,
    PipelineResultResponse,
    TargetResult,
)
from api.readiness import compute_readiness_scores
from api.state import AppState, JobStatus

router = APIRouter(prefix="/api/results", tags=["results"])

_state: AppState | None = None


def init(state: AppState) -> None:
    global _state
    _state = state


def _get_state() -> AppState:
    if _state is None:
        raise RuntimeError("AppState not initialized")
    return _state


def _safe_disc_ratio(wt: float, mut: float) -> float | None:
    """Compute discrimination ratio with safety checks.

    - div-by-zero (wt=0, mut>0) → cap at 999.0
    - both zero → None
    - NaN / inf → None
    - floor at 0.1
    - round to 1 decimal place
    """
    if wt > 0:
        ratio = mut / wt
    elif mut > 0:
        return 999.0
    else:
        return None

    if math.isnan(ratio) or math.isinf(ratio):
        return None

    ratio = max(ratio, 0.1)
    return round(ratio, 1)


def _get_completed_result(job_id: str) -> dict[str, Any]:
    """Get result dict from a completed job."""
    state = _get_state()
    job = state.get_job(job_id)
    if job is None:
        # Job not in memory — try loading result directly from disk
        result_path = state.results_dir / f"{job_id}.json"
        if result_path.exists():
            with open(result_path) as f:
                return json.load(f)
        raise HTTPException(404, f"Job {job_id} not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, f"Job {job_id} status is {job.status.value}, not completed")
    if job.result is None:
        # Try loading from disk
        result_path = state.results_dir / f"{job_id}.json"
        if result_path.exists():
            with open(result_path) as f:
                job.result = json.load(f)
        else:
            raise HTTPException(500, f"Result data not found for job {job_id}")
    return job.result


def _build_target_result(member: dict[str, Any]) -> TargetResult:
    """Build TargetResult from a MultiplexPanel member dict."""
    target = member.get("target", {})
    mutation = target.get("mutation", {})
    selected = member.get("selected_candidate", {})
    candidate = selected.get("candidate", {})
    heuristic = selected.get("heuristic", {})
    disc = selected.get("discrimination")
    primers = member.get("primers")

    disc_ratio = None
    if disc:
        disc_ratio = _safe_disc_ratio(
            disc.get("wt_activity", 0),
            disc.get("mut_activity", 0),
        )

    # Discrimination method and neural/feature scores
    disc_method = None
    neural_disc = None
    feature_disc = None
    if disc:
        neural_disc = disc.get("neural_disc")
        feature_disc = disc.get("feature_disc")
        disc_method = disc.get("disc_method")
        # Infer method from available data
        if disc_method is None:
            if neural_disc is not None:
                disc_method = "neural"
            elif feature_disc is not None:
                disc_method = "feature"
            elif disc_ratio is not None:
                disc_method = "heuristic"

    # Off-target report (nested under selected_candidate)
    offtarget = selected.get("offtarget", {})

    # Count off-target hits: sum of mtb + human + cross-reactivity hits
    offtarget_hits = None
    if offtarget:
        mtb_hits = offtarget.get("mtb_hits", [])
        human_hits = offtarget.get("human_hits", [])
        xr_hits = offtarget.get("cross_reactivity_hits", [])
        offtarget_hits = len(mtb_hits) + len(human_hits) + len(xr_hits)

    # ML confidence: best confidence from ml_scores list
    ml_confidence = None
    for ml_entry in selected.get("ml_scores", []):
        c = ml_entry.get("confidence")
        if c is not None:
            if ml_confidence is None or c > ml_confidence:
                ml_confidence = c

    # Mismatch type classification (transition vs transversion)
    mismatch_type = None
    ref_base_at_mut = candidate.get("ref_base_at_mutation")
    wt_spacer = candidate.get("wt_spacer_seq")
    spacer_seq = candidate.get("spacer_seq")
    mm_pos = candidate.get("mutation_position_in_spacer")
    if ref_base_at_mut and spacer_seq and mm_pos is not None:
        # Derive mismatch_type from ref vs alt base
        try:
            alt_base = spacer_seq[mm_pos - 1] if mm_pos > 0 else None
        except (IndexError, TypeError):
            alt_base = None
        if ref_base_at_mut and alt_base:
            purines = {"A", "G"}
            pyrimidines = {"C", "T"}
            ref_upper = ref_base_at_mut.upper()
            alt_upper = alt_base.upper()
            if (ref_upper in purines and alt_upper in purines) or \
               (ref_upper in pyrimidines and alt_upper in pyrimidines):
                mismatch_type = "transition"
            elif ref_upper != alt_upper:
                mismatch_type = "transversion"

    candidate_summary = CandidateSummary(
        candidate_id=candidate.get("candidate_id", ""),
        spacer_seq=candidate.get("spacer_seq", ""),
        pam_seq=candidate.get("pam_seq", ""),
        pam_variant=candidate.get("pam_variant", ""),
        strand=candidate.get("strand", ""),
        gc_content=candidate.get("gc_content", 0),
        detection_strategy=candidate.get("detection_strategy", "direct"),
        mutation_position_in_spacer=candidate.get("mutation_position_in_spacer"),
        wt_spacer_seq=candidate.get("wt_spacer_seq"),
        composite_score=heuristic.get("composite", 0),
        cnn_score=selected.get("cnn_score"),
        cnn_calibrated=selected.get("cnn_calibrated"),
        ensemble_score=selected.get("ensemble_score"),
        discrimination_ratio=disc_ratio,
        discrimination=disc,
        disc_method=disc_method,
        neural_disc=neural_disc,
        feature_disc=feature_disc,
        thermo_ddg=disc.get("thermo_ddg") if disc else None,
        mm_position_pam=disc.get("mm_position_pam") if disc else None,
        pam_penalty=heuristic.get("pam_penalty", candidate.get("pam_activity_weight")),
        is_canonical_pam=candidate.get("pam_variant") == "TTTV" if candidate.get("pam_variant") else None,
        activity_qc=heuristic.get("activity_qc"),
        discrimination_qc=heuristic.get("discrimination_qc"),
        mismatch_type_score=heuristic.get("mismatch_type_score"),
        flanking_gc_score=heuristic.get("flanking_gc_score"),
        enzyme_id=candidate.get("enzyme_id") or member.get("enzyme_id"),
        pam_disrupted=selected.get("pam_disrupted", False),
        pam_disruption_type=selected.get("pam_disruption_type"),
        ml_scores=selected.get("ml_scores", []),
        rank=selected.get("rank"),
        # Heuristic sub-scores
        seed_position_score=heuristic.get("seed_position_score"),
        gc_penalty=heuristic.get("gc_penalty"),
        structure_penalty=heuristic.get("structure_penalty"),
        homopolymer_penalty=heuristic.get("homopolymer_penalty"),
        offtarget_penalty=heuristic.get("offtarget_penalty"),
        proximity_bonus=heuristic.get("proximity_bonus"),
        # Thermodynamic data
        mfe=candidate.get("mfe"),
        cumulative_dg=disc.get("cumulative_dg") if disc else None,
        local_dg=disc.get("local_dg") if disc else None,
        # Off-target details
        offtarget_hits=offtarget_hits,
        offtarget_is_clean=offtarget.get("is_clean") if offtarget else None,
        # Discrimination model details
        disc_confidence=disc.get("confidence") if disc else None,
        disc_feature_vector=disc.get("feature_vector") if disc else None,
        disc_model_name=disc.get("model_name") if disc else None,
        # ML metadata
        ml_confidence=ml_confidence,
        # Mismatch details
        mismatch_type=mismatch_type,
        ref_base_at_mutation=ref_base_at_mut,
    ) if candidate.get("candidate_id") else None

    fwd_primer = None
    rev_primer = None
    amplicon_length = None
    fwd_tm = None
    rev_tm = None
    fwd_gc = None
    rev_gc = None
    primer_dimer_dg = None
    amplicon_seq = None
    if primers:
        fwd = primers.get("fwd", {})
        rev = primers.get("rev", {})
        fwd_primer = fwd.get("seq")
        rev_primer = rev.get("seq")
        fwd_tm = fwd.get("tm")
        rev_tm = rev.get("tm")
        fwd_gc = fwd.get("gc_content")
        rev_gc = rev.get("gc_content")
        primer_dimer_dg = primers.get("dimer_dg")
        amplicon_seq = primers.get("amplicon_seq")
        if fwd.get("amplicon_start") is not None and rev.get("amplicon_end") is not None:
            amplicon_length = rev["amplicon_end"] - fwd["amplicon_start"]

    return TargetResult(
        label=mutation.get("label", f"{mutation.get('gene', '')}_{mutation.get('ref_aa', '')}{mutation.get('position', '')}{mutation.get('alt_aa', '')}"),
        gene=mutation.get("gene", ""),
        mutation=f"{mutation.get('ref_aa', '')}{mutation.get('position', '')}{mutation.get('alt_aa', '')}",
        drug=mutation.get("drug", "OTHER"),
        detection_strategy=candidate.get("detection_strategy", "direct"),
        n_candidates=1,  # panel only contains selected candidate
        selected_candidate=candidate_summary,
        has_primers=primers is not None,
        fwd_primer=fwd_primer,
        rev_primer=rev_primer,
        amplicon_length=amplicon_length,
        fwd_tm=fwd_tm,
        rev_tm=rev_tm,
        fwd_gc=fwd_gc,
        rev_gc=rev_gc,
        primer_dimer_dg=primer_dimer_dg,
        amplicon_seq=amplicon_seq,
        proximity_distance=candidate.get("proximity_distance"),
        has_sm=member.get("has_sm", False),
        sm_enhanced_spacer=member.get("sm_enhanced_spacer"),
        sm_position=member.get("sm_position"),
        sm_discrimination_score=member.get("sm_discrimination_score"),
        sm_improvement_factor=member.get("sm_improvement_factor"),
        sm_original_base=member.get("sm_original_base"),
        sm_replacement_base=member.get("sm_replacement_base"),
        asrpa_discrimination=member.get("asrpa_discrimination"),
    )


@router.get("/{job_id}", response_model=PipelineResultResponse)
async def get_results(job_id: str) -> PipelineResultResponse:
    """Get full pipeline results for a completed job."""
    result = _get_completed_result(job_id)

    # Handle both basic and full mode results
    if result.get("mode") == "basic":
        targets_data = result.get("targets", {})
        target_results = []
        for label, scored_list in targets_data.items():
            if scored_list:
                first = scored_list[0]
                cand = first.get("candidate", {})
                heur = first.get("heuristic", {})
                target_results.append(TargetResult(
                    label=label,
                    gene=cand.get("target_label", label).split("_")[0],
                    mutation=label,
                    drug="",
                    detection_strategy=cand.get("detection_strategy", "direct"),
                    n_candidates=len(scored_list),
                    selected_candidate=CandidateSummary(
                        candidate_id=cand.get("candidate_id", ""),
                        spacer_seq=cand.get("spacer_seq", ""),
                        pam_seq=cand.get("pam_seq", ""),
                        pam_variant=cand.get("pam_variant", ""),
                        strand=cand.get("strand", ""),
                        gc_content=cand.get("gc_content", 0),
                        detection_strategy=cand.get("detection_strategy", "direct"),
                        wt_spacer_seq=cand.get("wt_spacer_seq"),
                        composite_score=heur.get("composite", 0),
                        cnn_score=first.get("cnn_score"),
                        cnn_calibrated=first.get("cnn_calibrated"),
                        ensemble_score=first.get("ensemble_score"),
                        ml_scores=first.get("ml_scores", []),
                        rank=first.get("rank"),
                    ),
                ))

        return PipelineResultResponse(
            job_id=job_id,
            panel=PanelSummary(
                plex=len(target_results),
                complete_targets=len(target_results),
            ),
            targets=target_results,
        )

    # Full mode: MultiplexPanel
    members = result.get("members", [])
    target_results = [_build_target_result(m) for m in members]

    # Compute readiness scores (operates on dicts, maps back to models)
    tr_dicts = [tr.model_dump() for tr in target_results]
    compute_readiness_scores(tr_dicts)
    target_results = [TargetResult(**d) for d in tr_dicts]

    # Compute panel summary
    disc_ratios = []
    direct_count = 0
    proximity_count = 0
    for tr in target_results:
        if tr.selected_candidate and tr.selected_candidate.discrimination_ratio is not None:
            ratio = tr.selected_candidate.discrimination_ratio
            if ratio < 999.0:
                disc_ratios.append(ratio)
        if tr.detection_strategy == "direct":
            direct_count += 1
        else:
            proximity_count += 1

    panel = PanelSummary(
        plex=len(members),
        complete_targets=sum(1 for tr in target_results if tr.has_primers),
        panel_score=result.get("panel_score"),
        mean_discrimination=sum(disc_ratios) / len(disc_ratios) if disc_ratios else None,
        direct_count=direct_count,
        proximity_count=proximity_count,
    )

    # Module statistics (populated by run_full)
    module_stats_raw = result.get("module_stats", [])
    module_stats = [ModuleStats(**s) for s in module_stats_raw]
    total_duration_ms = result.get("total_duration_ms", 0)

    # Calibration metadata: look in module_stats for ML scorer calibration info
    calibration = None
    for ms in module_stats_raw:
        breakdown = ms.get("breakdown", {})
        cal = breakdown.get("calibration")
        if cal is not None:
            calibration = cal
            break
    # Fallback: top-level calibration key
    if calibration is None:
        calibration = result.get("calibration")

    # SA optimizer convergence trace
    optimizer_score_trace = result.get("optimizer_score_trace")

    return PipelineResultResponse(
        job_id=job_id,
        panel=panel,
        targets=target_results,
        module_stats=module_stats,
        total_duration_ms=total_duration_ms,
        primer_dimer_matrix=result.get("primer_dimer_matrix"),
        primer_dimer_labels=result.get("primer_dimer_labels"),
        primer_dimer_report=result.get("primer_dimer_report"),
        calibration=calibration,
        optimizer_score_trace=optimizer_score_trace,
    )


@router.get("/{job_id}/export")
async def export_results(
    job_id: str,
    format: ExportFormat = Query(ExportFormat.JSON),
) -> StreamingResponse:
    """Export pipeline results in various formats."""
    result = _get_completed_result(job_id)

    if format == ExportFormat.JSON:
        content = json.dumps(result, indent=2, default=str)
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=compass_{job_id}.json"},
        )

    # Build flat rows for TSV/CSV/FASTA
    members = result.get("members", [])
    rows = []
    for m in members:
        target = m.get("target", {})
        mutation = target.get("mutation", {})
        selected = m.get("selected_candidate", {})
        candidate = selected.get("candidate", {})
        heuristic = selected.get("heuristic", {})
        disc = selected.get("discrimination")
        primers = m.get("primers")

        disc_ratio = ""
        if disc:
            ratio = _safe_disc_ratio(
                disc.get("wt_activity", 0),
                disc.get("mut_activity", 0),
            )
            if ratio is not None:
                disc_ratio = f"{ratio:.1f}"

        label = f"{mutation.get('gene', '')}_{mutation.get('ref_aa', '')}{mutation.get('position', '')}{mutation.get('alt_aa', '')}"

        rows.append({
            "target": label,
            "gene": mutation.get("gene", ""),
            "drug": mutation.get("drug", ""),
            "strategy": candidate.get("detection_strategy", ""),
            "spacer": candidate.get("spacer_seq", ""),
            "pam": candidate.get("pam_seq", ""),
            "score": f"{heuristic.get('composite', 0):.4f}",
            "discrimination": disc_ratio,
            "has_primers": str(primers is not None),
            "fwd_primer": primers.get("fwd", {}).get("seq", "") if primers else "",
            "rev_primer": primers.get("rev", {}).get("seq", "") if primers else "",
        })

    if format == ExportFormat.FASTA:
        lines = []
        for row in rows:
            lines.append(f">{row['target']} pam={row['pam']} score={row['score']}")
            lines.append(row["spacer"])
        content = "\n".join(lines) + "\n"
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=compass_{job_id}.fasta"},
        )

    # TSV or CSV
    sep = "\t" if format == ExportFormat.TSV else ","
    ext = "tsv" if format == ExportFormat.TSV else "csv"
    media = "text/tab-separated-values" if format == ExportFormat.TSV else "text/csv"

    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys(), delimiter=sep)
        writer.writeheader()
        writer.writerows(rows)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type=media,
        headers={"Content-Disposition": f"attachment; filename=compass_{job_id}.{ext}"},
    )


@router.get("/{job_id}/pools")
async def get_pools(job_id: str) -> dict:
    """Return primer pool assignments for the spatially-addressed electrode array."""
    result = _get_completed_result(job_id)

    dimer_matrix = result.get("primer_dimer_matrix")
    dimer_labels = result.get("primer_dimer_labels")
    dimer_report = result.get("primer_dimer_report")

    from compass.multiplex.pooling import compute_primer_pools, compute_amplicon_pad_specificity
    from compass.multiplex.kinetics import estimate_all_targets

    # Compute pools
    pooling = compute_primer_pools(
        dimer_matrix=dimer_matrix,
        dimer_labels=dimer_labels,
        dimer_report=dimer_report,
    )

    # Compute kinetics
    targets = list(pooling.target_to_pool.keys())
    kinetics = estimate_all_targets(targets=targets)

    # Compute amplicon-pad specificity
    specificity = compute_amplicon_pad_specificity()

    return {
        "pooling": pooling.to_dict(),
        "kinetics": kinetics,
        "specificity": specificity,
    }


@router.get("/{job_id}/umap")
async def get_umap_data(job_id: str) -> dict:
    """Return UMAP embedding coordinates for all candidates in a run."""
    state = _get_state()

    # Look for UMAP JSON in the results directory
    umap_path = state.results_dir / job_id / "umap_embeddings.json"
    if not umap_path.exists():
        raise HTTPException(
            404,
            "UMAP data not available. Re-run the pipeline with Compass-ML scoring.",
        )

    with open(umap_path) as f:
        return json.load(f)


@router.get("/{job_id}/cross-reactivity")
async def get_cross_reactivity(job_id: str) -> dict:
    """Score every crRNA against every other target's amplicon for off-target risk."""
    from compass.scoring.cross_reactivity import compute_cross_reactivity_matrix

    result = _get_completed_result(job_id)
    members = result.get("members", [])

    spacers: list[str] = []
    amplicons: list[str] = []
    labels: list[str] = []

    for m in members:
        target = m.get("target", {})
        mutation = target.get("mutation", {})
        selected = m.get("selected_candidate", {})
        candidate = selected.get("candidate", {})
        primers = m.get("primers")

        label = mutation.get(
            "label",
            f"{mutation.get('gene', '')}_{mutation.get('ref_aa', '')}"
            f"{mutation.get('position', '')}{mutation.get('alt_aa', '')}",
        )
        spacer = candidate.get("spacer_seq", "")
        amplicon = primers.get("amplicon_seq", "") if primers else ""

        if spacer and amplicon:
            spacers.append(spacer)
            amplicons.append(amplicon)
            labels.append(label)

    if len(spacers) < 2:
        raise HTTPException(
            400,
            "Cross-reactivity analysis requires at least 2 panel members "
            "with both spacer and amplicon sequences.",
        )

    return compute_cross_reactivity_matrix(spacers, amplicons, labels)
