"""Top-K alternative candidates per target with tradeoff annotations.

After the multiplex optimizer selects one candidate per target, this module
collects the next-best alternatives and annotates each with the tradeoff
it represents relative to the selected candidate.

This enables:
1. Experimental fallback (if #1 fails in the lab, try #2)
2. Pareto analysis (different candidates optimise different metrics)
3. Active learning (test alternatives with highest uncertainty)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from compass.core.types import ScoredCandidate, PanelMember
from compass.optimisation.metrics import TARGET_DRUG_CLASS

logger = logging.getLogger(__name__)


@dataclass
class AlternativeCandidate:
    """An alternative candidate with tradeoff annotation."""
    candidate: ScoredCandidate
    rank: int
    tradeoffs: list[str]        # e.g. ["higher_discrimination", "fewer_offtargets"]
    tradeoff_summary: str       # human-readable summary
    delta_efficiency: float
    delta_discrimination: float

    def to_dict(self) -> dict:
        return {
            "spacer_seq": self.candidate.candidate.spacer_seq,
            "rank": self.rank,
            "efficiency": self.candidate.composite_score,
            "discrimination_ratio": (
                self.candidate.discrimination.ratio
                if self.candidate.discrimination else None
            ),
            "offtarget_count": self.candidate.offtarget.total_risky_hits,
            "tradeoffs": self.tradeoffs,
            "tradeoff_summary": self.tradeoff_summary,
            "delta_efficiency": round(self.delta_efficiency, 4),
            "delta_discrimination": round(self.delta_discrimination, 4),
        }


@dataclass
class TargetCandidateSet:
    """Selected candidate + ranked alternatives for one target."""
    target_label: str
    drug_class: str
    selected: ScoredCandidate
    alternatives: list[AlternativeCandidate]
    selection_reason: str

    @property
    def top_k(self) -> list:
        return [self.selected] + [a.candidate for a in self.alternatives]

    def to_dict(self) -> dict:
        return {
            "target_label": self.target_label,
            "drug_class": self.drug_class,
            "selected_spacer": self.selected.candidate.spacer_seq,
            "selected_efficiency": self.selected.composite_score,
            "selected_discrimination": (
                self.selected.discrimination.ratio
                if self.selected.discrimination else None
            ),
            "selection_reason": self.selection_reason,
            "n_alternatives": len(self.alternatives),
            "alternatives": [a.to_dict() for a in self.alternatives],
        }


def _build_tradeoff_summary(
    selected: ScoredCandidate,
    alternative: ScoredCandidate,
    tradeoffs: list[str],
) -> str:
    """Build a human-readable tradeoff summary string."""
    sel_eff = selected.composite_score
    alt_eff = alternative.composite_score
    sel_disc = selected.discrimination.ratio if selected.discrimination else 0.0
    alt_disc = alternative.discrimination.ratio if alternative.discrimination else 0.0
    sel_ot = selected.offtarget.total_risky_hits
    alt_ot = alternative.offtarget.total_risky_hits

    parts = []
    if "higher_discrimination" in tradeoffs:
        parts.append(f"higher disc ({alt_disc:.1f}x vs {sel_disc:.1f}x)")
    if "higher_efficiency" in tradeoffs:
        parts.append(f"higher score ({alt_eff:.2f} vs {sel_eff:.2f})")
    if "fewer_offtargets" in tradeoffs:
        parts.append(f"fewer off-targets ({alt_ot} vs {sel_ot})")
    if "different_pam" in tradeoffs:
        parts.append("different PAM site")

    if not parts:
        return "comparable performance"

    # Add cost
    costs = []
    if alt_eff < sel_eff - 0.02 and "higher_efficiency" not in tradeoffs:
        costs.append(f"lower score ({alt_eff:.2f} vs {sel_eff:.2f})")
    if alt_disc < sel_disc * 0.8 and "higher_discrimination" not in tradeoffs:
        costs.append(f"lower disc ({alt_disc:.1f}x vs {sel_disc:.1f}x)")

    summary = " + ".join(parts)
    if costs:
        summary += " but " + " and ".join(costs)
    return summary


def _annotate_tradeoffs(
    selected: ScoredCandidate,
    alternative: ScoredCandidate,
) -> list[str]:
    """Determine what tradeoffs an alternative represents vs the selected."""
    tradeoffs: list[str] = []

    sel_eff = selected.composite_score
    alt_eff = alternative.composite_score
    sel_disc = selected.discrimination.ratio if selected.discrimination else 0.0
    alt_disc = alternative.discrimination.ratio if alternative.discrimination else 0.0
    sel_ot = selected.offtarget.total_risky_hits
    alt_ot = alternative.offtarget.total_risky_hits

    if alt_eff > sel_eff + 0.02:
        tradeoffs.append("higher_efficiency")

    if alt_disc > sel_disc * 1.2 and alt_disc > sel_disc + 0.5:
        tradeoffs.append("higher_discrimination")

    if alt_ot < sel_ot:
        tradeoffs.append("fewer_offtargets")

    sel_pos = selected.candidate.genomic_start
    alt_pos = alternative.candidate.genomic_start
    if abs(sel_pos - alt_pos) > 5:
        tradeoffs.append("different_pam")

    if not tradeoffs:
        tradeoffs.append("comparable")

    return tradeoffs


def _selection_reason(selected: ScoredCandidate) -> str:
    """Generate a human-readable reason why this candidate was selected."""
    eff = selected.composite_score
    disc = selected.discrimination.ratio if selected.discrimination else 0.0
    ot = selected.offtarget.total_risky_hits
    strategy = selected.candidate.detection_strategy.value

    parts = [f"score={eff:.2f}"]
    if disc > 0:
        parts.append(f"disc={disc:.1f}x")
    parts.append(f"off-targets={ot}")
    if strategy != "direct":
        parts.append(f"strategy={strategy}")
    if selected.candidate.in_seed:
        parts.append(f"seed pos {selected.candidate.mutation_position_in_spacer}")

    return "Best composite: " + ", ".join(parts)


def collect_top_k(
    members: list[PanelMember],
    candidates_by_target: dict[str, list[ScoredCandidate]],
    k: int = 5,
) -> list[TargetCandidateSet]:
    """Collect top-K alternative candidates per target with tradeoff annotations.

    Args:
        members: Panel members with selected candidates.
        candidates_by_target: All scored candidates per target label.
        k: Maximum number of alternatives to return per target (3-5 typical).

    Returns:
        List of TargetCandidateSet, one per panel member.
    """
    results: list[TargetCandidateSet] = []

    for member in members:
        selected = member.selected_candidate
        label = member.label
        drug_class = TARGET_DRUG_CLASS.get(label, "unknown")
        all_candidates = candidates_by_target.get(label, [])

        ranked = sorted(
            all_candidates,
            key=lambda sc: sc.composite_score,
            reverse=True,
        )

        alternatives: list[AlternativeCandidate] = []
        rank = 0
        for sc in ranked:
            if sc.candidate.spacer_seq == selected.candidate.spacer_seq:
                continue
            rank += 1

            sel_disc = selected.discrimination.ratio if selected.discrimination else 0.0
            alt_disc = sc.discrimination.ratio if sc.discrimination else 0.0

            tradeoffs = _annotate_tradeoffs(selected, sc)
            summary = _build_tradeoff_summary(selected, sc, tradeoffs)

            alt = AlternativeCandidate(
                candidate=sc,
                rank=rank,
                tradeoffs=tradeoffs,
                tradeoff_summary=summary,
                delta_efficiency=sc.composite_score - selected.composite_score,
                delta_discrimination=alt_disc - sel_disc,
            )
            alternatives.append(alt)

            if len(alternatives) >= k:
                break

        results.append(TargetCandidateSet(
            target_label=label,
            drug_class=drug_class,
            selected=selected,
            alternatives=alternatives,
            selection_reason=_selection_reason(selected),
        ))

    logger.info(
        "Collected top-%d alternatives for %d targets (avg %.1f alternatives/target)",
        k, len(results),
        sum(len(r.alternatives) for r in results) / max(len(results), 1),
    )

    return results
