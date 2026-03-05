"""Thermodynamic feature computation for discrimination prediction.

Computes 15 features encoding the biophysical determinants of Cas12a
mismatch discrimination:

Position features (4):
  1. spacer_position:         1-indexed from PAM-proximal (seed = 1-8)
  2. in_seed:                 binary, position ≤ 8
  3. position_sensitivity:    from empirical Cas12a mismatch profiling
  4. region_code:             0=seed(1-8), 1=trunk(9-14), 2=tail(15+)

Mismatch chemistry features (4):
  5. mismatch_destab:         destabilisation score (0-1, purine-purine highest)
  6. is_wobble:               binary, rG:dT or rU:dG (tolerated by Cas12a)
  7. is_purine_purine:        binary, rA:dA/rG:dA/rA:dG/rG:dG (most destabilising)
  8. is_transition:           binary, purine↔purine or pyrimidine↔pyrimidine

Thermodynamic features (5):
  9. mismatch_ddg:            ΔΔG penalty at mismatch position (kcal/mol)
  10. cumulative_dg_at_mm:    cumulative R-loop dG up to mismatch position
  11. seed_dg:                cumulative dG of seed region (positions 1-8)
  12. total_hybrid_dg:        total RNA:DNA hybrid dG
  13. energy_ratio:           |cumulative_dg(pos)| / |ΔΔG_mismatch|

Context features (2):
  14. gc_content:             GC fraction of spacer (0-1)
  15. local_gc:               GC fraction in ±2 nt window around mismatch

References:
  - Sugimoto et al. (2000) Biochemistry — RNA:DNA mismatch ΔΔG
  - Zhang et al. (2024) NAR — R-loop energetics and trans-cleavage
  - Strohkendl et al. (2018) Mol Cell — position-dependent sensitivity
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Ensure compass-net is importable
_COMPASS_NET_DIR = Path(__file__).resolve().parent.parent
if str(_COMPASS_NET_DIR) not in sys.path:
    sys.path.insert(0, str(_COMPASS_NET_DIR))

_COMPASS_DIR = _COMPASS_NET_DIR.parent
if str(_COMPASS_DIR) not in sys.path:
    sys.path.insert(0, str(_COMPASS_DIR))

# Import existing infrastructure
try:
    from features.thermodynamic import RNA_DNA_NN, INIT_DH, INIT_DS
except ImportError:
    from compass.research.thermo_profile import RNA_DNA_NN, INIT_DH, INIT_DS

from compass.candidates.synthetic_mismatch import (
    MISMATCH_DESTABILISATION,
    POSITION_SENSITIVITY_PROFILES,
    MismatchType,
    _classify_mismatch,
)

# ──────────────────────────────────────────────────────────────
# RNA:DNA mismatch ΔΔG penalties (Sugimoto et al. 2000)
# These are thermodynamic destabilisation costs (kcal/mol) for
# replacing a Watson-Crick pair with a mismatch in an RNA:DNA duplex.
# Positive values = destabilising (unfavourable for R-loop).
# ──────────────────────────────────────────────────────────────

MISMATCH_DDG: dict[str, float] = {
    # Purine-purine: severely destabilising
    "rA:dA": 3.8,    # large steric clash
    "rG:dA": 3.5,
    "rA:dG": 3.2,
    "rG:dG": 4.0,    # largest purines
    # Pyrimidine-pyrimidine
    "rC:dC": 3.5,    # steric clash (small bases)
    "rU:dC": 2.8,
    "rC:dT": 2.5,
    "rU:dT": 2.2,    # small, weak destabilisation
    # Wobble pairs: partially stable
    "rG:dT": 0.8,    # G:T wobble — thermodynamically tolerated
    "rU:dG": 1.0,    # U:G wobble — also tolerated
    # Purine-pyrimidine transversions
    "rA:dC": 2.9,
    "rC:dA": 3.0,
}

_DNA_TO_RNA = {"A": "U", "T": "A", "C": "G", "G": "C"}


def _compute_cumulative_dg(rna_spacer: str, temperature_c: float = 37.0) -> list[float]:
    """Cumulative RNA:DNA hybrid dG at each spacer position."""
    T = temperature_c + 273.15
    seq = rna_spacer.upper()
    cumulative = [0.0]
    running = 0.0
    for i in range(len(seq) - 1):
        dinuc = seq[i:i + 2]
        if dinuc in RNA_DNA_NN:
            dH, dS = RNA_DNA_NN[dinuc]
            step_dg = dH - T * (dS / 1000.0)
        else:
            step_dg = -1.0
        running += step_dg
        cumulative.append(round(running, 3))
    return cumulative


def _compute_total_hybrid_dg(rna_spacer: str, temperature_c: float = 37.0) -> float:
    """Total RNA:DNA hybrid formation dG."""
    T = temperature_c + 273.15
    dH, dS = INIT_DH, INIT_DS
    seq = rna_spacer.upper()
    for i in range(len(seq) - 1):
        dinuc = seq[i:i + 2]
        if dinuc in RNA_DNA_NN:
            h, s = RNA_DNA_NN[dinuc]
            dH += h
            dS += s
    return dH - T * (dS / 1000.0)


def compute_features_for_pair(
    guide_seq: str,
    spacer_position: int,
    mismatch_type: str,
    cas_variant: str = "LbCas12a",
) -> dict[str, float]:
    """Compute 15 thermodynamic discrimination features for a single pair.

    Args:
        guide_seq: 25-nt guide (4 PAM + 21 spacer) or 20-nt spacer
        spacer_position: 1-indexed mismatch position in spacer (from PAM-proximal)
        mismatch_type: e.g., "rA:dG" — RNA:DNA mismatch classification
        cas_variant: Cas12a variant for position sensitivity lookup

    Returns:
        Dict of 15 named features.
    """
    # Extract spacer (skip PAM if present)
    if len(guide_seq) >= 24 and guide_seq[:3].upper() == "TTT":
        spacer_dna = guide_seq[4:24]  # 20-nt spacer
    else:
        spacer_dna = guide_seq[:20]

    spacer_len = len(spacer_dna)

    # Convert spacer to crRNA (RNA complement)
    crrna = "".join(_DNA_TO_RNA.get(b, "N") for b in reversed(spacer_dna.upper()))

    # ── Position features ──
    in_seed = 1.0 if spacer_position <= 8 else 0.0

    profile = POSITION_SENSITIVITY_PROFILES.get(
        cas_variant, POSITION_SENSITIVITY_PROFILES["enAsCas12a"]
    )
    pos_sensitivity = profile.get(spacer_position, 0.05)

    if spacer_position <= 8:
        region_code = 0  # seed
    elif spacer_position <= 14:
        region_code = 1  # trunk
    else:
        region_code = 2  # tail

    # ── Mismatch chemistry features ──
    # Look up destabilisation score from existing module
    mm_destab = 0.5
    for mt in MismatchType:
        if mt.value == mismatch_type:
            mm_destab = MISMATCH_DESTABILISATION.get(mt, 0.5)
            break

    is_wobble = 1.0 if mismatch_type in ("rG:dT", "rU:dG") else 0.0
    is_purine_purine = 1.0 if mismatch_type in ("rA:dA", "rG:dA", "rA:dG", "rG:dG") else 0.0
    is_transition = 1.0 if mismatch_type in (
        "rA:dG", "rG:dA", "rC:dT", "rU:dC",  # same-class substitutions
    ) else 0.0

    # ── Thermodynamic features ──
    mismatch_ddg = MISMATCH_DDG.get(mismatch_type, 2.5)

    cumulative = _compute_cumulative_dg(crrna)
    # Position in cumulative array (crrna is reversed, so position 1 = index 0)
    # But cumulative[0] = 0 (start), cumulative[1] = first dinuc, etc.
    cum_idx = min(spacer_position, len(cumulative) - 1)
    cumulative_dg_at_mm = cumulative[cum_idx]

    seed_end = min(8, len(cumulative) - 1)
    seed_dg = cumulative[seed_end]

    total_hybrid_dg = _compute_total_hybrid_dg(crrna)

    # Energy ratio: how much of the R-loop energy has accumulated
    # relative to the mismatch penalty. High ratio → mismatch has less
    # impact because the R-loop is already very stable at that point.
    if abs(mismatch_ddg) > 0.01:
        energy_ratio = abs(cumulative_dg_at_mm) / mismatch_ddg
    else:
        energy_ratio = 10.0  # effectively infinite tolerance

    # ── Context features ──
    gc_count = sum(1 for b in spacer_dna.upper() if b in "GC")
    gc_content = gc_count / max(spacer_len, 1)

    # Local GC in ±2 window around mismatch
    mm_idx = spacer_position - 1  # 0-indexed
    window_start = max(0, mm_idx - 2)
    window_end = min(spacer_len, mm_idx + 3)
    local_seq = spacer_dna[window_start:window_end].upper()
    local_gc = sum(1 for b in local_seq if b in "GC") / max(len(local_seq), 1)

    return {
        # Position (4)
        "spacer_position": float(spacer_position),
        "in_seed": in_seed,
        "position_sensitivity": pos_sensitivity,
        "region_code": float(region_code),
        # Mismatch chemistry (4)
        "mismatch_destab": mm_destab,
        "is_wobble": is_wobble,
        "is_purine_purine": is_purine_purine,
        "is_transition": is_transition,
        # Thermodynamics (5)
        "mismatch_ddg": mismatch_ddg,
        "cumulative_dg_at_mm": cumulative_dg_at_mm,
        "seed_dg": seed_dg,
        "total_hybrid_dg": total_hybrid_dg,
        "energy_ratio": energy_ratio,
        # Context (2)
        "gc_content": gc_content,
        "local_gc": local_gc,
    }


FEATURE_NAMES = [
    "spacer_position", "in_seed", "position_sensitivity", "region_code",
    "mismatch_destab", "is_wobble", "is_purine_purine", "is_transition",
    "mismatch_ddg", "cumulative_dg_at_mm", "seed_dg", "total_hybrid_dg",
    "energy_ratio",
    "gc_content", "local_gc",
]


def compute_features_batch(
    pairs: list,
    cas_variant: str = "LbCas12a",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute features and targets for a batch of DiscriminationPair objects.

    Returns:
        X: (n_pairs, 15) feature matrix
        y: (n_pairs,) target array — delta_logk (MUT - WT activity in log space)
    """
    X_list = []
    y_list = []

    for pair in pairs:
        feats = compute_features_for_pair(
            guide_seq=pair.guide_seq,
            spacer_position=pair.spacer_position,
            mismatch_type=pair.mismatch_type,
            cas_variant=cas_variant,
        )
        X_list.append([feats[name] for name in FEATURE_NAMES])
        y_list.append(pair.delta_logk)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info("Computed features: X=%s, y=%s", X.shape, y.shape)
    return X, y


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from compass_ml_data_extract import extract_discrimination_pairs

    # Handle import from different locations
    import importlib
    try:
        from extract_discrimination_pairs import extract_discrimination_pairs
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from extract_discrimination_pairs import extract_discrimination_pairs

    pairs = extract_discrimination_pairs()
    X, y = compute_features_batch(pairs)

    print(f"\nFeature matrix: {X.shape}")
    print(f"Target vector: {y.shape}")
    print(f"\nFeature statistics:")
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        print(f"  {name:25s}: min={col.min():.3f}, max={col.max():.3f}, "
              f"mean={col.mean():.3f}, std={col.std():.3f}")
    print(f"\nTarget (delta_logk): min={y.min():.3f}, max={y.max():.3f}, "
          f"mean={y.mean():.3f}, std={y.std():.3f}")
