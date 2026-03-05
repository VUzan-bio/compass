"""Parameter sweep engine for sensitivity-specificity analysis.

The sweep engine re-evaluates existing panel candidates against
different thresholds WITHOUT re-running the full pipeline. This is fast
because candidates are already generated and scored — we just change
which ones pass the threshold.

Two modes:
1. Threshold sweep: vary disc_threshold or score_threshold, measure how
   many targets remain covered. This traces the ROC-like curve.
2. Pipeline re-run sweep: change upstream params (GC filter, off-target
   stringency) and re-run candidate generation. Slower but more thorough.

For Block 3 MVP: threshold sweep only.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

from compass.core.types import PanelMember, ScoredCandidate
from compass.optimisation.metrics import DiagnosticMetrics, compute_diagnostic_metrics
from compass.optimisation.profiles import ParameterProfile, BALANCED

logger = logging.getLogger(__name__)


@dataclass
class SweepPoint:
    """One point on a parameter sweep curve."""
    parameter_name: str
    parameter_value: float
    metrics: DiagnosticMetrics

    def to_dict(self) -> dict:
        s = self.metrics.summary()
        return {
            "parameter_name": self.parameter_name,
            "parameter_value": round(self.parameter_value, 4),
            "sensitivity": s["panel_sensitivity"],
            "specificity": s["panel_specificity"],
            "drug_class_coverage": s["drug_class_coverage"],
            "cost": s["cost"],
        }


@dataclass
class SweepResult:
    """Complete result of a parameter sweep."""
    parameter_name: str
    points: list[SweepPoint]
    base_profile: ParameterProfile

    def to_dict(self) -> dict:
        return {
            "parameter_name": self.parameter_name,
            "n_points": len(self.points),
            "base_profile": self.base_profile.name,
            "points": [p.to_dict() for p in self.points],
        }


def sweep_parameter(
    parameter_name: str,
    values: list[float],
    members: list[PanelMember],
    candidates_by_target: dict[str, list[ScoredCandidate]],
    base_profile: ParameterProfile | None = None,
) -> SweepResult:
    """Sweep one parameter over a range and compute metrics at each point.

    Args:
        parameter_name: Which threshold to sweep. One of:
            "efficiency_threshold", "discrimination_threshold".
        values: List of values to evaluate.
        members: Panel members with selected candidates.
        candidates_by_target: All candidates per target.
        base_profile: Starting profile (other params held constant).
            Defaults to BALANCED.

    Returns:
        SweepResult with sensitivity/specificity at each value.
    """
    if base_profile is None:
        base_profile = copy.deepcopy(BALANCED)

    if parameter_name not in ("efficiency_threshold", "discrimination_threshold"):
        raise ValueError(
            f"Cannot sweep '{parameter_name}'. "
            f"Supported: efficiency_threshold, discrimination_threshold"
        )

    points: list[SweepPoint] = []

    for val in sorted(values):
        profile = copy.deepcopy(base_profile)
        setattr(profile, parameter_name, val)

        metrics = compute_diagnostic_metrics(
            members=members,
            candidates_by_target=candidates_by_target,
            efficiency_threshold=profile.efficiency_threshold,
            discrimination_threshold=profile.discrimination_threshold,
        )

        points.append(SweepPoint(
            parameter_name=parameter_name,
            parameter_value=val,
            metrics=metrics,
        ))

    return SweepResult(
        parameter_name=parameter_name,
        points=points,
        base_profile=base_profile,
    )
