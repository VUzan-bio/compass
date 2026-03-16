"""
AS-RPA thermodynamic discrimination estimation.

Lightweight estimation of allele-specific RPA primer discrimination
based on terminal mismatch identity and optional penultimate mismatch.
This is NOT a full nearest-neighbour thermodynamic calculator; it uses
empirical mismatch penalty data specific to recombinase polymerase
amplification (RPA) to predict selectivity.

The key insight is that RPA's strand-displacement mechanism makes it
more tolerant of internal mismatches than PCR, but 3'-terminal mismatches
still block extension by the Bsu polymerase.  Discrimination therefore
depends almost entirely on (1) the identity of the 3' mismatch and
(2) whether a deliberate penultimate mismatch is introduced.

References
----------
- Ye et al. (2019) Allele-Specific LAMP/RPA primer design.
- PMC12179515 (2025) RPA-specific terminal mismatch penalty measurements.
- Ayyadevara et al. (2000) Penultimate mismatch enhancement of
  allele-specific amplification selectivity.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RT_37C = 0.616  # kcal/mol at 37 deg C (310.15 K)

_PENULTIMATE_MM_BONUS = 2.5  # kcal/mol additional penalty from deliberate
                              # penultimate mismatch (Ayyadevara et al., 2000)

_RATIO_FLOOR = 1.0
_RATIO_CAP = 100.0   # Hard ceiling — empirical AS-RPA rarely exceeds 100× (Ye et al. 2019)
_EMPIRICAL_SCALE = 0.50  # Damping factor: maps Boltzmann → empirical range (10-100×)
                          # Accounts for kinetic effects in RPA that reduce thermodynamic selectivity

# ---------------------------------------------------------------------------
# Terminal mismatch penalty table (primer 3' base : WT template base)
# Values in kcal/mol, from RPA-specific data (PMC12179515, 2025).
# Key format: (primer_3prime_base, wt_template_base)
# These represent the DDG penalty for the mismatch relative to a
# perfectly matched primer.
# ---------------------------------------------------------------------------

_TERMINAL_PENALTY: Dict[tuple[str, str], float] = {
    ("C", "C"): 3.8,   # strongest block
    ("G", "A"): 3.2,
    ("A", "A"): 3.0,
    ("C", "T"): 2.8,
    ("G", "G"): 2.5,
    ("T", "T"): 2.2,
    ("A", "G"): 2.0,
    ("T", "C"): 1.8,
    ("C", "A"): 1.5,
    ("A", "C"): 0.8,   # tolerated
    ("T", "G"): 0.5,   # wobble — worst for discrimination
    ("G", "T"): 0.5,   # wobble — worst for discrimination
}

# Complementary pairs (not mismatches) — no penalty.
_COMPLEMENTS = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}

# ---------------------------------------------------------------------------
# Block-class thresholds (kcal/mol)
# ---------------------------------------------------------------------------

_STRONG_THRESHOLD = 5.0
_MODERATE_THRESHOLD = 3.0


def _classify_block(ddg: float) -> str:
    """Classify the discrimination block strength."""
    if ddg > _STRONG_THRESHOLD:
        return "strong"
    if ddg >= _MODERATE_THRESHOLD:
        return "moderate"
    return "weak"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_asrpa_discrimination(
    primer_3prime_base: str,
    wt_template_base: str,
    has_penultimate_mm: bool = True,
) -> Dict[str, Any]:
    """Estimate discrimination ratio for an AS-RPA primer.

    Parameters
    ----------
    primer_3prime_base : str
        The 3'-terminal base of the AS-RPA primer (A, T, C, or G).
    wt_template_base : str
        The wild-type template base opposite the primer 3' end.
        When the primer matches the mutant allele, this is the WT base
        that creates the discriminating mismatch.
    has_penultimate_mm : bool, default True
        Whether the primer carries a deliberate penultimate (N-1)
        mismatch to enhance selectivity (Ayyadevara et al., 2000).

    Returns
    -------
    dict
        terminal_mismatch : str   e.g. "C:C"
        ddg_kcal : float          total DDG penalty (kcal/mol)
        disc_ratio : float        predicted fold-discrimination (1-1000x)
        block_class : str         "strong", "moderate", or "weak"
        has_penultimate_mm : bool
        estimated_specificity : float  fraction, e.g. 0.998

    Raises
    ------
    ValueError
        If bases are not valid DNA characters.
    """
    p = primer_3prime_base.upper().strip()
    w = wt_template_base.upper().strip()

    if p not in "ACGT" or len(p) != 1:
        raise ValueError(f"Invalid primer 3' base: {primer_3prime_base!r}")
    if w not in "ACGT" or len(w) != 1:
        raise ValueError(f"Invalid WT template base: {wt_template_base!r}")

    # If the pair is complementary there is no mismatch — no discrimination.
    if (p, w) in _COMPLEMENTS:
        return {
            "terminal_mismatch": f"{p}:{w}",
            "ddg_kcal": 0.0,
            "disc_ratio": _RATIO_FLOOR,
            "block_class": "none",
            "has_penultimate_mm": has_penultimate_mm,
            "estimated_specificity": 0.0,
        }

    base_penalty = _TERMINAL_PENALTY.get((p, w))
    if base_penalty is None:
        raise ValueError(
            f"No penalty data for mismatch {p}:{w}. "
            "This should not happen for valid DNA bases."
        )

    ddg = base_penalty
    if has_penultimate_mm:
        ddg += _PENULTIMATE_MM_BONUS

    # Empirically-damped Boltzmann ratio: exp(scale * DDG / RT)
    # Pure Boltzmann overestimates because RPA strand exchange kinetics
    # partially compensate for terminal mismatches. The damping factor (0.40)
    # maps theoretical ratios (~100-30000×) to the empirical range (~10-100×)
    # observed in AS-RPA experiments (Ye et al. 2019, Li et al. 2019).
    ratio = math.exp(_EMPIRICAL_SCALE * ddg / _RT_37C)
    ratio = max(_RATIO_FLOOR, min(_RATIO_CAP, ratio))

    block_class = _classify_block(ddg)
    specificity = 1.0 - (1.0 / ratio) if ratio > 1.0 else 0.0

    return {
        "terminal_mismatch": f"{p}:{w}",
        "ddg_kcal": round(ddg, 2),
        "disc_ratio": round(ratio, 1),
        "block_class": block_class,
        "has_penultimate_mm": has_penultimate_mm,
        "estimated_specificity": round(specificity, 4),
    }


# ---------------------------------------------------------------------------
# Penultimate substitution ΔΔG table
# The penultimate mismatch bonus depends on the specific substitution.
# Different mismatches at N-1 have different destabilisation effects.
# Values from Ayyadevara et al. (2000) and RPA empirical data.
# Key: (original_base, substitution_base) → ΔΔG bonus (kcal/mol)
# ---------------------------------------------------------------------------

_PENULTIMATE_SUBSTITUTION_DDG: Dict[tuple[str, str], float] = {
    # Strong destabilisers (purine↔pyrimidine transversions)
    ("A", "C"): 3.0, ("A", "T"): 2.8,
    ("T", "G"): 3.0, ("T", "A"): 2.5,
    ("G", "T"): 2.8, ("G", "A"): 2.2,
    ("C", "A"): 3.2, ("C", "G"): 2.5,
    # Weaker destabilisers (transitions)
    ("A", "G"): 1.8, ("G", "C"): 1.5,
    ("T", "C"): 1.5, ("C", "T"): 1.8,
}


def optimize_penultimate_mismatch(
    primer_seq: str,
    wt_template_base: str,
    penultimate_template_base: str,
) -> Dict[str, Any]:
    """Test all 3 possible penultimate substitutions and return the best.

    For each non-original base at the N-1 position:
      1. Compute the specific ΔΔG bonus for that substitution
      2. Add it to the terminal mismatch ΔΔG
      3. Return the substitution that maximizes total ΔΔG (discrimination)

    Parameters
    ----------
    primer_seq : str
        Full primer sequence (5'→3'). Last base is the 3' terminal;
        second-to-last is the penultimate position.
    wt_template_base : str
        WT template base at the primer 3' end (creates the terminal mismatch).
    penultimate_template_base : str
        Template base at the penultimate (N-1) position.

    Returns
    -------
    dict with keys:
        best_substitution : str       the base to use at N-1
        original_base : str           the original N-1 base
        ddg_bonus : float             ΔΔG from the penultimate mismatch
        total_ddg : float             terminal + penultimate combined
        disc_ratio : float            predicted fold-discrimination
        block_class : str
        all_substitutions : list      results for all 3 options
    """
    primer_3prime = primer_seq[-1].upper()
    original_penult = primer_seq[-2].upper() if len(primer_seq) >= 2 else "N"
    bases = {"A", "T", "G", "C"}

    # Terminal mismatch base ΔΔG
    terminal_ddg = _TERMINAL_PENALTY.get((primer_3prime, wt_template_base.upper()), 0.0)

    # Try all 3 non-original substitutions at N-1
    candidates = []
    for sub_base in sorted(bases - {original_penult}):
        # The penultimate mismatch ΔΔG depends on what substitution we make
        pen_ddg = _PENULTIMATE_SUBSTITUTION_DDG.get(
            (original_penult, sub_base), _PENULTIMATE_MM_BONUS
        )
        total = terminal_ddg + pen_ddg
        ratio = math.exp(_EMPIRICAL_SCALE * total / _RT_37C)
        ratio = max(_RATIO_FLOOR, min(_RATIO_CAP, ratio))

        candidates.append({
            "substitution": sub_base,
            "penultimate_ddg": round(pen_ddg, 2),
            "total_ddg": round(total, 2),
            "disc_ratio": round(ratio, 1),
            "block_class": _classify_block(total),
        })

    # Also test no penultimate mismatch (original base)
    no_pen = {
        "substitution": original_penult,
        "penultimate_ddg": 0.0,
        "total_ddg": round(terminal_ddg, 2),
        "disc_ratio": round(max(_RATIO_FLOOR, min(_RATIO_CAP, math.exp(_EMPIRICAL_SCALE * terminal_ddg / _RT_37C))), 1),
        "block_class": _classify_block(terminal_ddg),
    }

    # Pick the best (highest discrimination ratio)
    best = max(candidates, key=lambda c: c["disc_ratio"])

    return {
        "best_substitution": best["substitution"],
        "original_base": original_penult,
        "ddg_bonus": best["penultimate_ddg"],
        "total_ddg": best["total_ddg"],
        "disc_ratio": best["disc_ratio"],
        "block_class": best["block_class"],
        "no_penultimate": no_pen,
        "all_substitutions": candidates,
    }


def score_panel_asrpa(
    members: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Batch-score a panel of AS-RPA proximity candidates.

    Parameters
    ----------
    members : sequence of dict
        Each dict must contain at minimum:
            - ``primer_3prime_base`` (str)
            - ``wt_template_base`` (str)
        Optional keys:
            - ``has_penultimate_mm`` (bool, default True)
            - any additional keys are passed through unchanged.

    Returns
    -------
    list of dict
        One entry per member, each containing the discrimination results
        (disc_ratio, terminal_mismatch, block_class, estimated_specificity)
        merged with any extra keys from the input dict.
    """
    results: List[Dict[str, Any]] = []
    for member in members:
        p3 = member["primer_3prime_base"]
        wt = member["wt_template_base"]
        pen = member.get("has_penultimate_mm", True)

        disc = compute_asrpa_discrimination(p3, wt, has_penultimate_mm=pen)

        # Merge input keys (pass-through) with discrimination results.
        merged: Dict[str, Any] = {}
        for k, v in member.items():
            if k not in ("primer_3prime_base", "wt_template_base",
                         "has_penultimate_mm"):
                merged[k] = v
        merged.update(disc)
        results.append(merged)

    return results
