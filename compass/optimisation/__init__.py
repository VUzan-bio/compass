"""Block 3: Sensitivity-Specificity Optimization Framework.

Provides tools for exploring the tradeoff between diagnostic sensitivity
(detecting all resistance mutations) and specificity (distinguishing
resistant from susceptible). Clinicians can select parameter profiles
that match their deployment context:

- High sensitivity (field screening): maximize coverage, tolerate lower
  discrimination ratios. Catches rare mutations at the cost of false positives.
- Balanced (WHO TPP): meet WHO Target Product Profile thresholds.
  RIF sens >= 95%, INH/FQ >= 90%, spec >= 98% for drug-resistant TB.
- High specificity (confirmatory): maximize discrimination ratios,
  accept fewer covered targets. Minimizes false resistance calls.

Key components:
    DiagnosticMetrics: panel-level metrics with per-drug-class WHO compliance
    ParameterProfile: preset threshold configurations
    compute_diagnostic_metrics: bridge between pipeline output and metrics
    sweep_parameter: 1D parameter sweep with sens/spec curves
    pareto_frontier: multi-objective Pareto-optimal profiles
    collect_top_k: ranked alternatives per target with tradeoff annotations
"""

from compass.optimisation.metrics import (
    DiagnosticMetrics,
    DrugClassMetrics,
    TargetMetrics,
    compute_diagnostic_metrics,
    WHO_TPP_SENSITIVITY,
    WHO_TPP_SPECIFICITY,
    TARGET_DRUG_CLASS,
)
from compass.optimisation.profiles import ParameterProfile, get_preset, list_presets
from compass.optimisation.sweep import sweep_parameter
from compass.optimisation.pareto import pareto_frontier, generate_profile_grid
from compass.optimisation.top_k import TargetCandidateSet, collect_top_k

__all__ = [
    "DiagnosticMetrics",
    "DrugClassMetrics",
    "TargetMetrics",
    "compute_diagnostic_metrics",
    "WHO_TPP_SENSITIVITY",
    "WHO_TPP_SPECIFICITY",
    "TARGET_DRUG_CLASS",
    "ParameterProfile",
    "get_preset",
    "list_presets",
    "sweep_parameter",
    "pareto_frontier",
    "generate_profile_grid",
    "TargetCandidateSet",
    "collect_top_k",
]
