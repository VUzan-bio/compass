"""In situ RNP formation kinetics for spatially-addressed electrode array.

Models the detection process on each LIG-E (cellulose-derived) electrode pad as a phase breakdown
rather than per-target absolute time estimates.  The dominant experimental
unknown is the effective trans-cleavage rate on LIG-E-tethered MB-ssDNA
reporters, which has not been measured directly.

Kinetic parameters (literature-verified):
  - Lesinski et al. 2024, Anal. Chem.:
      k_form = 1.75 x 10^5 M^-1 s^-1 (SPR, Cas12a + HPV16 sgRNA, 25 C)
      k_off  = 1.87 x 10^-4 s^-1 (SPR, KD = 1.07 nM)
      k_cis  = 0.03 s^-1 (first-order fit, dsDNA fluorescence reporter)
  - Nalefski et al. 2021, iScience:
      k_trans (solution) ~ 2.0 s^-1 (LbCas12a kcat, C10 polycytidine, low-salt)
  - Surface-tethered k_trans: estimated 0.01-0.1 s^-1 (10-100x slower than
    solution due to steric hindrance, restricted diffusion, surface crowding).
    No direct measurement exists.

The in situ complexation format (Lesinski 2024) purposefully limits active
Cas12a during early amplification stages, reducing cis-cleavage competition
with RPA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# ── Default kinetic parameters ────────────────────────────────────────

K_FORM = 1.75e5        # M^-1 s^-1 — RNP association rate (Lesinski 2024, SPR)
K_OFF = 1.87e-4        # s^-1 — RNP dissociation rate (Lesinski 2024, SPR)
K_CIS = 0.03           # s^-1 — cis-cleavage rate (Lesinski 2024, dsDNA reporter)
K_TRANS_SOLUTION = 2.0  # s^-1 — trans-cleavage in solution (Nalefski 2021, LbCas12a, C10)
K_TRANS_SURFACE_LO = 0.01  # s^-1 — estimated surface lower bound
K_TRANS_SURFACE_HI = 0.1   # s^-1 — estimated surface upper bound
CAS12A_CONC = 50e-9    # 50 nM — design parameter (typical E-CRISPR: 10-200 nM)
CRRNA_CONC = 200e-9    # ~200 nM equivalent pre-dried — design parameter
MB_DENSITY_LO = 1e10   # molecules/cm^2 — LIG lower bound (pi-stacking)
MB_DENSITY_HI = 1e11   # molecules/cm^2 — LIG upper bound


# ── Per-target activity estimates ─────────────────────────────────────
# Relative activity modifiers based on predicted efficiency scores.
# Used only for ranking (fastest → slowest), NOT absolute time estimates.

TARGET_EFFICIENCY: Dict[str, float] = {
    "IS6110_NON": 0.95,
    "rpoB_S450L": 0.82,
    "rpoB_H445Y": 0.78,
    "rpoB_H445D": 0.75,
    "rpoB_D435V": 0.73,
    "rpoB_S450W": 0.70,
    "katG_S315T": 0.85,
    "katG_S315N": 0.68,
    "inhA_C-15T": 0.80,
    "embB_M306V": 0.76,
    "embB_M306I": 0.65,
    "pncA_H57D": 0.55,
    "gyrA_D94G": 0.83,
    "gyrA_A90V": 0.79,
    "rrs_A1401G": 0.81,
}


@dataclass
class KineticPhase:
    """A single phase of the assay with solution and on-electrode estimates."""
    phase: str
    solution_bound: str
    on_electrode: str
    description: str
    is_bottleneck: bool

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "solution_bound": self.solution_bound,
            "on_electrode": self.on_electrode,
            "description": self.description,
            "is_bottleneck": self.is_bottleneck,
        }


@dataclass
class AssayTotals:
    """Aggregate timing estimates."""
    detection_solution: str = "~6\u20138 min"
    detection_electrode: str = "15\u201330 min"
    rpa_time: str = "15\u201320 min"
    total_solution: str = "~23\u201328 min"
    total_electrode: str = "30\u201350 min"
    who_tpp_target: str = "< 120 min"
    who_tpp_pass: bool = True

    def to_dict(self) -> dict:
        return {
            "detection_solution": self.detection_solution,
            "detection_electrode": self.detection_electrode,
            "rpa_time": self.rpa_time,
            "total_solution": self.total_solution,
            "total_electrode": self.total_electrode,
            "who_tpp_target": self.who_tpp_target,
            "who_tpp_pass": self.who_tpp_pass,
        }


# ── Static phase breakdown ────────────────────────────────────────────

KINETIC_PHASES: List[KineticPhase] = [
    KineticPhase(
        phase="crRNA rehydration",
        solution_bound="N/A",
        on_electrode="2\u20135 min",
        description="Dried crRNA dissolves from LIG surface into assay buffer",
        is_bottleneck=False,
    ),
    KineticPhase(
        phase="RNP formation",
        solution_bound="0.5\u20131 min",
        on_electrode="2\u20135 min",
        description="Cas12a + crRNA \u2192 active RNP (in situ complexation)",
        is_bottleneck=False,
    ),
    KineticPhase(
        phase="Target recognition",
        solution_bound="~10 sec",
        on_electrode="1\u20133 min",
        description="RNP binds cognate amplicon, R-loop formation, cis-cleavage activation",
        is_bottleneck=False,
    ),
    KineticPhase(
        phase="Surface trans-cleavage",
        solution_bound="~5 min",
        on_electrode="10\u201320 min",
        description="Activated Cas12a cleaves MB-ssDNA reporters tethered to LIG electrode",
        is_bottleneck=True,
    ),
]

# ── Literature-verified parameter table ───────────────────────────────

KINETIC_PARAMS = [
    {
        "param": "k_form (RNP association)",
        "value": "1.75 \u00d7 10\u2075 M\u207b\u00b9s\u207b\u00b9",
        "source": "Lesinski et al. 2024",
        "source_detail": "SPR measurement, Cas12a + HPV16 sgRNA, 25\u00b0C",
        "note": "Solution-phase. On-electrode rate may differ due to crRNA rehydration.",
    },
    {
        "param": "k_off (RNP dissociation)",
        "value": "1.87 \u00d7 10\u207b\u2074 s\u207b\u00b9",
        "source": "Lesinski et al. 2024",
        "source_detail": "SPR measurement, K_D = 1.07 nM",
        "note": None,
    },
    {
        "param": "k_cis (cis-cleavage)",
        "value": "0.03 s\u207b\u00b9",
        "source": "Lesinski et al. 2024",
        "source_detail": "First-order fit of dsDNA fluorescence reporter cleavage",
        "note": None,
    },
    {
        "param": "k_trans (solution, free ssDNA)",
        "value": "~2.0 s\u207b\u00b9",
        "source": "Nalefski et al. 2021",
        "source_detail": "LbCas12a k_cat for C10 polycytidine, low-salt buffer",
        "note": "Free ssDNA in solution. NOT applicable to surface-tethered reporters.",
    },
    {
        "param": "k_trans (surface, estimated)",
        "value": "0.01\u20130.1 s\u207b\u00b9",
        "source": "Estimated",
        "source_detail": "10-100\u00d7 slower than solution due to tethered MB-ssDNA steric effects",
        "note": "No direct measurement exists. Key experimental unknown.",
    },
    {
        "param": "[Cas12a]",
        "value": "50 nM",
        "source": "Design parameter",
        "source_detail": "Typical E-CRISPR range: 10\u2013200 nM. Lesinski 2024 used 2 \u00b5M for kinetic characterization.",
        "note": None,
    },
    {
        "param": "[crRNA] on pad",
        "value": "~200 nM equivalent",
        "source": "Design parameter",
        "source_detail": "Pre-dried on LIG. Effective concentration after rehydration unknown.",
        "note": "Bezinge 2023 demonstrated crRNA-on-LIG but did not report concentration.",
    },
    {
        "param": "MB-ssDNA probe density",
        "value": "~10\u00b9\u2070\u201310\u00b9\u00b9 molecules/cm\u00b2",
        "source": "Estimated for LIG",
        "source_detail": "\u03c0-stacking attachment. Gold SAM ref: 10\u00b9\u00b2/cm\u00b2 (Steel 1998). LIG 10\u2013100\u00d7 lower.",
        "note": "Probe density directly affects signal magnitude and time-to-detection.",
    },
]

# ── Key insights (literature-verified) ────────────────────────────────

KEY_INSIGHTS = [
    {
        "title": "Rate-limiting step",
        "text": (
            "Surface trans-cleavage of tethered MB-ssDNA reporters dominates "
            "detection time \u2014 not RNP formation or target recognition. "
            "Optimizing probe density and electrode surface chemistry is the "
            "primary experimental lever for reducing time-to-result."
        ),
    },
    {
        "title": "In situ complexation",
        "text": (
            "Lesinski et al. 2024: reduces effective [RNP] during the first "
            "~5 minutes by ~10-fold vs pre-complexed format. This is a "
            "feature \u2014 it prevents Cas12a from destroying target amplicons "
            "before detection begins."
        ),
    },
    {
        "title": "Experimental unknowns",
        "text": (
            "k_trans on LIG-tethered MB-ssDNA and crRNA rehydration kinetics "
            "have not been measured. These are key characterization "
            "priorities that directly inform chip design parameters."
        ),
    },
]

CHARACTERIZATION_PRIORITIES = [
    "k_trans on MB-ssDNA/LIG \u2014 determines detection time",
    "crRNA rehydration kinetics \u2014 determines pad preparation protocol",
    "Optimal [Cas12a] for in situ format \u2014 balance sensitivity vs background",
    "Probe density optimization \u2014 balance signal intensity vs steric access",
]


def get_kinetics_data(
    targets: Optional[List[str]] = None,
    efficiencies: Optional[Dict[str, float]] = None,
) -> Dict:
    """Return the complete kinetics dataset for frontend rendering.

    Instead of per-target absolute time estimates (which imply false
    precision), returns:
      - Phase breakdown with solution vs on-electrode ranges
      - Assay totals with WHO TPP compliance
      - Literature-verified parameter table
      - Key insights and characterization priorities
      - Target ranking by predicted detection speed (relative order only)
    """
    if targets is None:
        targets = list(TARGET_EFFICIENCY.keys())

    # Build target ranking by efficiency (proxy for detection speed)
    ranked = []
    for t in targets:
        eff = (efficiencies or {}).get(t, TARGET_EFFICIENCY.get(t, 0.7))
        ranked.append({"target": t, "efficiency": round(eff, 3), "is_weak": eff < 0.6})
    ranked.sort(key=lambda x: -x["efficiency"])  # fastest first

    totals = AssayTotals()

    return {
        "phases": [p.to_dict() for p in KINETIC_PHASES],
        "totals": totals.to_dict(),
        "parameters": KINETIC_PARAMS,
        "insights": KEY_INSIGHTS,
        "year1_priorities": CHARACTERIZATION_PRIORITIES,
        "target_ranking": ranked,
    }


# ── Legacy API (kept for backwards compatibility) ─────────────────────

@dataclass
class KineticEstimate:
    """Time-to-result estimate for a single electrode pad (legacy)."""
    target: str
    t_rnp_formation: float
    t_target_recognition: float
    t_signal_generation: float
    t_total: float
    efficiency: float
    is_weak: bool

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "t_rnp_formation": round(self.t_rnp_formation, 1),
            "t_target_recognition": round(self.t_target_recognition, 1),
            "t_signal_generation": round(self.t_signal_generation, 1),
            "t_total": round(self.t_total, 1),
            "efficiency": round(self.efficiency, 3),
            "is_weak": self.is_weak,
        }


def estimate_time_to_result(target: str, efficiency: Optional[float] = None, **_kw) -> KineticEstimate:
    """Legacy single-target estimate. Kept for API compatibility."""
    if efficiency is None:
        efficiency = TARGET_EFFICIENCY.get(target, 0.7)
    return KineticEstimate(
        target=target, t_rnp_formation=3.5, t_target_recognition=2.0,
        t_signal_generation=15.0, t_total=20.5, efficiency=efficiency,
        is_weak=efficiency < 0.6,
    )


def estimate_all_targets(
    targets: Optional[List[str]] = None,
    efficiencies: Optional[Dict[str, float]] = None,
) -> Dict:
    """Return kinetics data. Uses the new phase-based model."""
    return get_kinetics_data(targets=targets, efficiencies=efficiencies)
