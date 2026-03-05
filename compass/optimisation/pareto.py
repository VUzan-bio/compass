"""Compute the Pareto frontier of sensitivity vs specificity.

A profile is Pareto-optimal if no other profile achieves BOTH higher
sensitivity AND higher specificity. The frontier represents the
fundamental tradeoff: you can have more coverage (sensitivity) or
more stringent discrimination (specificity), but not both.

For the paper: the Pareto frontier is a figure showing how different
parameter configurations trace the tradeoff curve. Each point is a
(sensitivity, specificity) pair from a specific ParameterProfile.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from compass.core.types import PanelMember, ScoredCandidate
from compass.optimisation.metrics import DiagnosticMetrics, compute_diagnostic_metrics
from compass.optimisation.profiles import ParameterProfile

logger = logging.getLogger(__name__)


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""
    profile: ParameterProfile
    metrics: DiagnosticMetrics

    def to_dict(self) -> dict:
        s = self.metrics.summary()
        return {
            "profile": self.profile.to_dict(),
            "sensitivity": s["panel_sensitivity"],
            "specificity": s["panel_specificity"],
            "drug_class_coverage": s["drug_class_coverage"],
            "cost": s["cost"],
        }


def generate_profile_grid(
    disc_values: list[float] | None = None,
    score_values: list[float] | None = None,
) -> list[ParameterProfile]:
    """Generate a grid of profiles for Pareto analysis.

    Returns len(disc) x len(score) profiles.
    """
    if disc_values is None:
        disc_values = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    if score_values is None:
        score_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    profiles: list[ParameterProfile] = []
    for disc in disc_values:
        for score in score_values:
            profiles.append(ParameterProfile(
                name=f"grid_d{disc:.1f}_s{score:.1f}",
                description=f"eff>={score:.2f}, disc>={disc:.2f}",
                efficiency_threshold=score,
                discrimination_threshold=disc,
            ))
    return profiles


def pareto_frontier(
    members: list[PanelMember],
    candidates_by_target: dict[str, list[ScoredCandidate]],
    disc_values: list[float] | None = None,
    score_values: list[float] | None = None,
) -> list[ParetoPoint]:
    """Compute the Pareto frontier over efficiency and discrimination thresholds.

    Evaluates a grid of (efficiency_threshold, discrimination_threshold) pairs
    and returns the non-dominated set — points where improving sensitivity
    necessarily reduces specificity, and vice versa.

    Args:
        members: Panel members with selected candidates.
        candidates_by_target: All candidates per target.
        disc_values: Discrimination threshold values for grid.
        score_values: Efficiency threshold values for grid.

    Returns:
        List of ParetoPoint on the non-dominated frontier, sorted by
        decreasing sensitivity.
    """
    profiles = generate_profile_grid(disc_values, score_values)

    all_points: list[ParetoPoint] = []
    for profile in profiles:
        metrics = compute_diagnostic_metrics(
            members=members,
            candidates_by_target=candidates_by_target,
            efficiency_threshold=profile.efficiency_threshold,
            discrimination_threshold=profile.discrimination_threshold,
        )
        all_points.append(ParetoPoint(profile=profile, metrics=metrics))

    # Extract non-dominated set (maximize both sensitivity and specificity)
    frontier: list[ParetoPoint] = []
    for p in all_points:
        dominated = False
        for q in all_points:
            if (
                q.metrics.sensitivity >= p.metrics.sensitivity
                and q.metrics.specificity >= p.metrics.specificity
                and (
                    q.metrics.sensitivity > p.metrics.sensitivity
                    or q.metrics.specificity > p.metrics.specificity
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(p)

    # Sort by decreasing sensitivity
    frontier.sort(key=lambda p: -p.metrics.sensitivity)

    logger.info(
        "Pareto frontier: %d non-dominated points from %d grid points",
        len(frontier), len(all_points),
    )

    return frontier


def plot_pareto(
    frontier: list[ParetoPoint],
    all_points: list[ParetoPoint] | None = None,
    save_path: str = "pareto_frontier.png",
) -> None:
    """Generate publication-quality Pareto frontier figure.

    - Scatter all evaluated profiles in grey
    - Highlight Pareto-optimal points in colour
    - Connect frontier with a line
    - Mark WHO TPP target zone (sensitivity>=0.95, specificity>=0.98)
    - Axis labels: "Panel Sensitivity" vs "Panel Specificity"
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # WHO TPP target zone
    ax.axhspan(0.98, 1.02, alpha=0.08, color="green", label=None)
    ax.axvspan(0.95, 1.02, alpha=0.08, color="green", label=None)
    ax.add_patch(mpatches.Rectangle(
        (0.95, 0.98), 0.05, 0.02,
        linewidth=1.5, edgecolor="green", facecolor="green",
        alpha=0.15, label="WHO TPP zone",
    ))

    # All evaluated points (grey)
    if all_points:
        ax.scatter(
            [p.metrics.sensitivity for p in all_points],
            [p.metrics.specificity for p in all_points],
            c="lightgrey", s=20, alpha=0.6, zorder=1,
            label=f"Grid ({len(all_points)} profiles)",
        )

    # Pareto frontier
    front_sens = [p.metrics.sensitivity for p in frontier]
    front_spec = [p.metrics.specificity for p in frontier]
    ax.plot(front_sens, front_spec, "o-", color="royalblue", markersize=8,
            linewidth=2, zorder=3, label=f"Pareto frontier ({len(frontier)} pts)")

    # Label frontier points
    for p in frontier:
        label = f"d≥{p.profile.discrimination_threshold:.0f},s≥{p.profile.efficiency_threshold:.1f}"
        ax.annotate(
            label,
            (p.metrics.sensitivity, p.metrics.specificity),
            textcoords="offset points", xytext=(5, 5),
            fontsize=7, alpha=0.8,
        )

    ax.set_xlabel("Panel Sensitivity (mutation coverage)", fontsize=12)
    ax.set_ylabel("Panel Specificity (1 - false positive rate)", fontsize=12)
    ax.set_title("Sensitivity-Specificity Pareto Frontier", fontsize=14)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Pareto frontier plot saved to %s", save_path)
