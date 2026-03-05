"""Multi-axis Diagnostic Readiness Score computation.

Computes a percentile-rank composite that forces spread among candidates,
plus a traffic-light risk matrix and experimental priority ranking.
"""

from __future__ import annotations

import math
from typing import Any


def _percentile_rank(values: list[float], higher_is_better: bool = True) -> list[float]:
    """Rank values as percentiles (0-1). Handles ties via average rank."""
    n = len(values)
    if n <= 1:
        return [0.5] * n

    # Sort ascending by value, assign ranks with tie averaging
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    raw_ranks = [0.0] * n

    # Group by value for tie handling
    i = 0
    while i < n:
        j = i
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_pos = (i + j - 1) / 2.0
        for k in range(i, j):
            orig_idx = indexed[k][0]
            if higher_is_better:
                raw_ranks[orig_idx] = avg_pos / max(n - 1, 1)
            else:
                raw_ranks[orig_idx] = 1.0 - avg_pos / max(n - 1, 1)
        i = j
    return raw_ranks


def _risk_level(value: float, green_min: float, amber_min: float) -> str:
    if value >= green_min:
        return "green"
    elif value >= amber_min:
        return "amber"
    return "red"


# Weights for composite readiness score
W_EFF = 0.20
W_DISC = 0.40
W_PRIMER = 0.15
W_SAFETY = 0.15
W_GC = 0.10


def compute_readiness_scores(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute multi-axis Diagnostic Readiness Score for each target.

    Modifies targets in-place, adding readiness_score, readiness_components,
    experimental_priority, risk_profile, and priority_reason.
    """
    n = len(targets)
    if n == 0:
        return targets

    efficiencies: list[float] = []
    discriminations: list[float] = []
    primer_qualities: list[float] = []
    offtarget_counts: list[float] = []
    gc_deviations: list[float] = []
    strategies: list[str] = []

    for t in targets:
        cand = t.get("selected_candidate") or {}

        # Efficiency
        eff = (cand.get("ensemble_score")
               or cand.get("cnn_calibrated")
               or cand.get("composite_score")
               or 0.5)
        efficiencies.append(eff)

        # Discrimination
        strategy = t.get("detection_strategy", "direct")
        strategies.append(strategy)
        if strategy.lower() == "direct":
            disc = cand.get("discrimination_ratio") or 1.0
        else:
            asrpa = t.get("asrpa_discrimination") or {}
            disc = asrpa.get("disc_ratio", 1.0)
        disc = min(disc, 100.0)
        discriminations.append(disc)

        # Primer quality
        has_primers = t.get("has_primers", False)
        amp_len = t.get("amplicon_length") or 150
        amp_deviation = abs(amp_len - 100) / 100.0
        primer_qual = 1.0 if has_primers else 0.0
        if has_primers:
            primer_qual = max(0.0, 1.0 - amp_deviation * 0.3)
        primer_qualities.append(primer_qual)

        # Off-target count
        disc_data = cand.get("discrimination") or {}
        ot = disc_data.get("offtarget_count", 0)
        offtarget_counts.append(ot)

        # GC deviation from optimal
        gc = cand.get("gc_content", 0.5)
        gc_dev = abs(gc - 0.50)
        gc_deviations.append(gc_dev)

    # Percentile ranks
    eff_ranks = _percentile_rank(efficiencies, higher_is_better=True)
    disc_ranks = _percentile_rank(discriminations, higher_is_better=True)
    primer_ranks = _percentile_rank(primer_qualities, higher_is_better=True)
    safety_ranks = _percentile_rank(offtarget_counts, higher_is_better=False)
    gc_ranks = _percentile_rank(gc_deviations, higher_is_better=False)

    # Composite readiness scores
    readiness_scores: list[float] = []
    for i in range(n):
        score = (
            W_EFF * eff_ranks[i]
            + W_DISC * disc_ranks[i]
            + W_PRIMER * primer_ranks[i]
            + W_SAFETY * safety_ranks[i]
            + W_GC * gc_ranks[i]
        )
        readiness_scores.append(score)

    # Priority ranking: direct candidates first (they test crRNA discrimination
    # directly), then proximity candidates (AS-RPA validation track).
    # Within each group, rank by readiness score.
    priority_scores = []
    for i in range(n):
        strategy_boost = 1.0 if strategies[i].lower() == "direct" else 0.0
        priority_scores.append((strategy_boost, readiness_scores[i]))
    priority_order = sorted(range(n), key=lambda i: priority_scores[i], reverse=True)
    priority_map = {idx: rank + 1 for rank, idx in enumerate(priority_order)}

    # Check for shared amplicons (same fwd + rev primer pair)
    primer_groups: dict[str, list[int]] = {}
    for i, t in enumerate(targets):
        fwd = t.get("fwd_primer") or ""
        rev = t.get("rev_primer") or ""
        if fwd and rev:
            key = f"{fwd}|{rev}"
            primer_groups.setdefault(key, []).append(i)

    shared_amplicon: dict[int, list[str]] = {}
    for indices in primer_groups.values():
        if len(indices) > 1:
            for idx in indices:
                partners = [targets[j].get("label", "") for j in indices if j != idx]
                shared_amplicon[idx] = partners

    # Assign to targets
    for i, t in enumerate(targets):
        disc_val = discriminations[i]
        eff_val = efficiencies[i]
        gc_val = gc_deviations[i]
        ot_val = offtarget_counts[i]
        strategy = strategies[i]

        # Risk profile
        if strategy.lower() == "direct":
            disc_risk = _risk_level(disc_val, 3.0, 2.0)
        else:
            disc_risk = _risk_level(disc_val, 10.0, 1.5)

        risk = {
            "activity": _risk_level(eff_val, 0.7, 0.4),
            "discrimination": disc_risk,
            "primers": "green" if primer_qualities[i] > 0.5 else ("amber" if primer_qualities[i] > 0 else "red"),
            "gc_risk": _risk_level(1.0 - gc_val, 0.85, 0.70),
            "off_target": _risk_level(
                1.0 if ot_val == 0 else (0.5 if ot_val <= 2 else 0.0),
                0.8, 0.4,
            ),
        }

        # Overall risk = worst axis
        risk_values = {"green": 2, "amber": 1, "red": 0}
        worst = min(risk_values[v] for v in risk.values())
        risk["overall"] = {2: "green", 1: "amber", 0: "red"}[worst]

        # Priority reasons
        reasons: list[str] = []
        if disc_ranks[i] >= 0.8:
            reasons.append("top discrimination")
        if eff_ranks[i] >= 0.8:
            reasons.append("highest activity")
        if i in shared_amplicon:
            partners = shared_amplicon[i]
            reasons.append(f"shares amplicon with {', '.join(partners[:2])}")
        if risk["discrimination"] == "red":
            reasons.append("needs SM enhancement or alternative strategy")
        elif risk["discrimination"] == "amber":
            reasons.append(f"borderline discrimination ({disc_val:.1f}x)")
        if not reasons:
            reasons.append("standard priority")

        t["readiness_score"] = round(readiness_scores[i], 3)
        t["readiness_components"] = {
            "efficiency": round(eff_ranks[i], 3),
            "discrimination": round(disc_ranks[i], 3),
            "primers": round(primer_ranks[i], 3),
            "safety": round(safety_ranks[i], 3),
            "gc": round(gc_ranks[i], 3),
        }
        t["experimental_priority"] = priority_map[i]
        t["risk_profile"] = risk
        t["priority_reason"] = "; ".join(reasons)

    return targets
