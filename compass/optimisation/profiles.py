"""Parameter profiles for sensitivity-specificity optimization.

Three presets cover the clinical deployment spectrum:

1. HIGH_SENSITIVITY (field screening)
   - Low thresholds: catch everything, accept false positives
   - For: resource-limited settings, initial screening, ruling out susceptibility
   - disc >= 2x, score >= 0.3, top_k = 5
   - Clinical rationale: In field screening at peripheral health centres,
     missing a resistant case (false negative) is more dangerous than a
     false positive, which can be resolved by confirmatory testing at a
     reference lab. Maximise mutation coverage.

2. BALANCED (WHO TPP)
   - WHO Target Product Profile for drug-resistant TB diagnostics
   - disc >= 3x, score >= 0.4, top_k = 3
   - Clinical rationale: Meets WHO 2024 TPP requirements — RIF sensitivity
     >= 95%, INH/FQ >= 90%, specificity >= 98%. The default for clinical
     diagnostic deployment in settings with confirmatory capacity.

3. HIGH_SPECIFICITY (confirmatory)
   - Strict thresholds: only high-confidence calls
   - disc >= 5x, score >= 0.6, top_k = 3
   - For: reference labs, confirmatory testing, clinical decision-making
   - Clinical rationale: When the result directly informs treatment decisions
     (e.g., switching from first-line to MDR-TB regimen), false positives
     carry high cost (unnecessary toxic drugs, prolonged treatment). Accept
     lower coverage in exchange for high discrimination confidence.

Each profile defines thresholds for:
    - efficiency_threshold: minimum predicted cleavage activity
    - discrimination_threshold: minimum MUT/WT ratio
    - offtarget_max_hits: maximum allowed off-target sites
    - cross_reactivity_max: maximum allowed cross-reactivity in multiplex
    - top_k: number of alternative candidates to retain per target
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParameterProfile:
    """Threshold configuration for panel optimization.

    Controls the sensitivity-specificity tradeoff by adjusting
    minimum quality thresholds for candidate selection.
    """

    name: str
    description: str

    # Efficiency: minimum predicted cleavage activity (0-1 scale)
    efficiency_threshold: float = 0.4

    # Discrimination: minimum MUT/WT activity ratio
    discrimination_threshold: float = 3.0

    # Off-target: maximum allowed off-target sites (Cas-OFFinder)
    offtarget_max_hits: int = 5

    # Cross-reactivity: maximum allowed pairwise spacer similarity in panel
    cross_reactivity_max: float = 0.3

    # Multiplex optimizer weights
    efficiency_weight: float = 0.4
    discrimination_weight: float = 0.3
    cross_reactivity_weight: float = 0.3

    # Target sensitivity and specificity
    target_sensitivity: float = 0.95
    target_specificity: float = 0.98

    # Top-K alternatives to retain per target
    top_k: int = 3

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "efficiency_threshold": self.efficiency_threshold,
            "discrimination_threshold": self.discrimination_threshold,
            "offtarget_max_hits": self.offtarget_max_hits,
            "cross_reactivity_max": self.cross_reactivity_max,
            "efficiency_weight": self.efficiency_weight,
            "discrimination_weight": self.discrimination_weight,
            "cross_reactivity_weight": self.cross_reactivity_weight,
            "target_sensitivity": self.target_sensitivity,
            "target_specificity": self.target_specificity,
            "top_k": self.top_k,
        }


# --- Presets ---

HIGH_SENSITIVITY = ParameterProfile(
    name="high_sensitivity",
    description="Field screening — maximise coverage, tolerate lower discrimination.",
    efficiency_threshold=0.3,
    discrimination_threshold=2.0,
    offtarget_max_hits=10,
    cross_reactivity_max=0.4,
    efficiency_weight=0.5,
    discrimination_weight=0.2,
    cross_reactivity_weight=0.3,
    target_sensitivity=0.98,
    target_specificity=0.90,
    top_k=5,
)

BALANCED = ParameterProfile(
    name="balanced",
    description="WHO TPP-aligned — clinical diagnostic deployment.",
    efficiency_threshold=0.4,
    discrimination_threshold=3.0,
    offtarget_max_hits=5,
    cross_reactivity_max=0.3,
    efficiency_weight=0.4,
    discrimination_weight=0.3,
    cross_reactivity_weight=0.3,
    target_sensitivity=0.95,
    target_specificity=0.98,
    top_k=3,
)

HIGH_SPECIFICITY = ParameterProfile(
    name="high_specificity",
    description="Confirmatory — minimise false calls, reference lab use.",
    efficiency_threshold=0.6,
    discrimination_threshold=5.0,
    offtarget_max_hits=2,
    cross_reactivity_max=0.2,
    efficiency_weight=0.3,
    discrimination_weight=0.5,
    cross_reactivity_weight=0.2,
    target_sensitivity=0.90,
    target_specificity=0.99,
    top_k=3,
)

_PRESETS = {
    "high_sensitivity": HIGH_SENSITIVITY,
    "balanced": BALANCED,
    "high_specificity": HIGH_SPECIFICITY,
}


def get_preset(name: str) -> ParameterProfile:
    """Get a named preset profile.

    Args:
        name: One of "high_sensitivity", "balanced", "high_specificity".

    Raises:
        KeyError if name not found.
    """
    if name not in _PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. Available: {list(_PRESETS.keys())}"
        )
    return _PRESETS[name]


def list_presets() -> list[dict]:
    """Return all presets as serializable dicts."""
    return [p.to_dict() for p in _PRESETS.values()]
