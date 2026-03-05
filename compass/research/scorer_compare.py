"""Compare two scoring models on the same panel results.

Computes per-target score deltas, rank changes, Kendall tau rank
correlation, and diagnostic impact analysis.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _kendall_tau(ranks_a: list[int], ranks_b: list[int]) -> float:
    """Compute Kendall tau rank correlation coefficient.

    tau = 1 - 2 * n_discordant / (n * (n-1) / 2)
    Returns value in [-1, 1]. 1 = identical ranking.
    """
    n = len(ranks_a)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff_a = ranks_a[i] - ranks_a[j]
            diff_b = ranks_b[i] - ranks_b[j]
            product = diff_a * diff_b
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return round((concordant - discordant) / total, 4)


def compare_scorers(
    panel_result: dict[str, Any],
    model_a: str,
    model_b: str,
) -> dict[str, Any]:
    """Compare two scoring approaches on a completed panel.

    Returns comparison dict with per-target scores, ranks, Kendall tau,
    and diagnostic impact summary.
    """
    members = panel_result.get("members", [])
    if not members:
        return _compare_basic_mode(panel_result, model_a, model_b)

    targets = []
    for member in members:
        target = member.get("target", {})
        mutation = target.get("mutation", {})
        selected = member.get("selected_candidate", {})
        candidate = selected.get("candidate", {})
        heuristic = selected.get("heuristic", {})
        disc = selected.get("discrimination")

        label = mutation.get("label", f"{mutation.get('gene', '')}_{mutation.get('ref_aa', '')}{mutation.get('position', '')}{mutation.get('alt_aa', '')}")
        drug = mutation.get("drug", candidate.get("drug", ""))
        strategy = candidate.get("detection_strategy", "")

        score_a = _get_score(selected, model_a, heuristic)
        score_b = _get_score(selected, model_b, heuristic)

        disc_ratio = None
        if disc:
            wt = disc.get("wt_activity", 0)
            mut = disc.get("mut_activity", 0)
            if wt > 0:
                disc_ratio = round(mut / wt, 1)

        targets.append({
            "label": label,
            "drug": drug,
            "strategy": strategy,
            "model_a": {"score": score_a, "disc": disc_ratio},
            "model_b": {"score": score_b, "disc": disc_ratio},
            "score_delta": round(score_b - score_a, 4) if score_a is not None and score_b is not None else None,
        })

    # Rank by score for each model
    sorted_a = sorted(range(len(targets)), key=lambda i: targets[i]["model_a"]["score"] or 0, reverse=True)
    sorted_b = sorted(range(len(targets)), key=lambda i: targets[i]["model_b"]["score"] or 0, reverse=True)

    for rank, idx in enumerate(sorted_a, 1):
        targets[idx]["model_a"]["rank"] = rank
    for rank, idx in enumerate(sorted_b, 1):
        targets[idx]["model_b"]["rank"] = rank

    for t in targets:
        rank_a = t["model_a"]["rank"]
        rank_b = t["model_b"]["rank"]
        t["rank_delta"] = rank_a - rank_b  # positive = improved in B
        t["rank_changed"] = abs(t["rank_delta"]) >= 3

    # Kendall tau
    ranks_a = [t["model_a"]["rank"] for t in targets]
    ranks_b = [t["model_b"]["rank"] for t in targets]
    tau = _kendall_tau(ranks_a, ranks_b)

    # Diagnostic impact: check which targets cross 0.4 threshold
    threshold = 0.4
    dropped = []
    gained = []
    for t in targets:
        sa = t["model_a"]["score"] or 0
        sb = t["model_b"]["score"] or 0
        if sa >= threshold and sb < threshold:
            dropped.append(t["label"])
        elif sa < threshold and sb >= threshold:
            gained.append(t["label"])

    above_a = sum(1 for t in targets if (t["model_a"]["score"] or 0) >= threshold)
    above_b = sum(1 for t in targets if (t["model_b"]["score"] or 0) >= threshold)

    deltas = [t["score_delta"] for t in targets if t["score_delta"] is not None]

    return {
        "model_a": model_a,
        "model_b": model_b,
        "targets": targets,
        "summary": {
            "kendall_tau": tau,
            "rank_agreement": sum(1 for t in targets if abs(t["rank_delta"]) == 0),
            "total_targets": len(targets),
            "mean_score_delta": round(sum(deltas) / len(deltas), 4) if deltas else 0,
            "above_threshold_a": above_a,
            "above_threshold_b": above_b,
            "dropped": dropped,
            "gained": gained,
        },
    }


def _get_score(selected: dict, model: str, heuristic: dict) -> float | None:
    """Extract score for a given model from a scored candidate."""
    if model == "heuristic":
        return heuristic.get("composite", 0)

    ml_scores = selected.get("ml_scores", [])
    for ml in ml_scores:
        if ml.get("model_name") == model:
            return ml.get("predicted_efficiency")

    if model in ("compass_ml", "compass_ml_diagnostic"):
        if selected.get("ensemble_score") is not None:
            return selected["ensemble_score"]
        if selected.get("cnn_calibrated") is not None:
            return selected["cnn_calibrated"]
        if selected.get("cnn_score") is not None:
            return selected["cnn_score"]

    return heuristic.get("composite", 0)


def _compare_basic_mode(result: dict, model_a: str, model_b: str) -> dict:
    """Handle basic mode results (dict of target -> scored list)."""
    targets_data = result.get("targets", {})
    targets = []

    for label, scored_list in targets_data.items():
        if not scored_list:
            continue
        first = scored_list[0]
        heuristic = first.get("heuristic", {})
        candidate = first.get("candidate", {})
        score_a = _get_score(first, model_a, heuristic)
        score_b = _get_score(first, model_b, heuristic)

        targets.append({
            "label": label,
            "drug": candidate.get("drug", ""),
            "strategy": candidate.get("detection_strategy", ""),
            "model_a": {"score": score_a, "disc": None, "rank": 0},
            "model_b": {"score": score_b, "disc": None, "rank": 0},
            "score_delta": round(score_b - score_a, 4) if score_a is not None and score_b is not None else None,
            "rank_delta": 0,
            "rank_changed": False,
        })

    return {
        "model_a": model_a,
        "model_b": model_b,
        "targets": targets,
        "summary": {
            "kendall_tau": 1.0,
            "rank_agreement": len(targets),
            "total_targets": len(targets),
            "mean_score_delta": 0,
            "above_threshold_a": len(targets),
            "above_threshold_b": len(targets),
            "dropped": [],
            "gained": [],
        },
    }
