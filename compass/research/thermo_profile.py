"""R-loop free energy landscape computation.

Computes the sequence-dependent free-energy profile for Cas12a R-loop
formation using nearest-neighbor thermodynamic parameters.

Three energy components following Zhang et al. (NAR 2024):
1. RNA:DNA hybrid formation (Sugimoto et al., Biochemistry 1995)
2. dsDNA target unwinding (SantaLucia, PNAS 1998)
3. crRNA spacer unfolding (ViennaRNA or GC estimate)

The CRISPRzip framework (Offerhaus et al., bioRxiv 2025) additionally
includes protein-mediated contributions that we do not model here.
Our profiles represent the nucleic acid component only.

References:
- Sugimoto et al. (1995) Biochemistry 34(35):11211-11216
  RNA:DNA nearest-neighbor parameters
- SantaLucia (1998) PNAS 95(4):1460-1465
  Unified DNA:DNA nearest-neighbor parameters
- Zhang et al. (2024) NAR 52(22):14077-14092
  DOI: 10.1093/nar/gkae1124
  Linear correlation: trans-cleavage rate ~ dG(unwinding)
- Offerhaus et al. (2025) bioRxiv 2025.12.12.691775v2
  CRISPRzip: mechanistic free-energy model for R-loop formation
- Aris et al. (2025) Nat Commun 16:2939
  DOI: 10.1038/s41467-025-57703-y
  Four-state kinetic model for Cas12a R-loop dynamics
- Strohkendl et al. (2024) Molecular Cell 84(14):2717-2731
  Cryo-EM of Cas12a R-loop intermediates, staged formation from 5-bp seed
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Import NN parameters from compass-net thermodynamic module
_COMPASS_NET_DIR = Path(__file__).resolve().parent.parent.parent / "compass-net"
if str(_COMPASS_NET_DIR) not in sys.path:
    sys.path.insert(0, str(_COMPASS_NET_DIR))

try:
    from features.thermodynamic import (
        RNA_DNA_NN, INIT_DH, INIT_DS,
        compute_spacer_folding_dg, compute_hybrid_dg, compute_melting_temperature,
    )
except ImportError:
    # Fallback inline parameters (Sugimoto et al. 1995)
    RNA_DNA_NN = {
        "AA": (-7.8, -21.9), "AC": (-5.9, -12.3),
        "AG": (-9.1, -23.5), "AU": (-8.3, -23.9),
        "CA": (-9.0, -26.1), "CC": (-9.3, -23.2),
        "CG": (-16.3, -47.1), "CU": (-7.0, -19.7),
        "GA": (-5.5, -13.5), "GC": (-8.0, -17.1),
        "GG": (-12.8, -31.9), "GU": (-7.8, -21.6),
        "UA": (-7.8, -23.2), "UC": (-8.6, -22.9),
        "UG": (-10.4, -28.4), "UU": (-11.5, -36.4),
    }
    INIT_DH = 1.9
    INIT_DS = -3.9

    def compute_spacer_folding_dg(seq):
        gc = sum(1 for b in seq.upper() if b in "GC") / max(len(seq), 1)
        return -2.0 * gc

    def compute_hybrid_dg(seq, temperature_celsius=37.0, na_concentration=0.15):
        T = temperature_celsius + 273.15
        dH, dS = INIT_DH, INIT_DS
        s = seq.upper()
        for i in range(len(s) - 1):
            d = s[i:i+2]
            if d in RNA_DNA_NN:
                h, ss = RNA_DNA_NN[d]
                dH += h
                dS += ss
        dS += 0.368 * len(s) * math.log(na_concentration)
        return dH - T * (dS / 1000.0)

    def compute_melting_temperature(seq, na_concentration=0.15, oligo_concentration=250e-9):
        R = 1.987
        dH, dS = INIT_DH, INIT_DS
        s = seq.upper()
        for i in range(len(s) - 1):
            d = s[i:i+2]
            if d in RNA_DNA_NN:
                h, ss = RNA_DNA_NN[d]
                dH += h
                dS += ss
        dS += 0.368 * len(s) * math.log(na_concentration)
        dS_total = dS + R * math.log(oligo_concentration / 4.0)
        if dS_total == 0:
            return 0.0
        return (dH * 1000.0) / dS_total - 273.15


# DNA:DNA nearest-neighbor parameters (SantaLucia 1998)
DNA_DNA_NN = {
    "AA": (-7.9, -22.2), "AT": (-7.2, -20.4),
    "AG": (-7.8, -21.0), "AC": (-8.4, -22.4),
    "TA": (-7.2, -21.3), "TT": (-7.9, -22.2),
    "TG": (-8.5, -22.7), "TC": (-8.2, -22.2),
    "GA": (-8.2, -22.2), "GT": (-8.4, -22.4),
    "GG": (-8.0, -19.9), "GC": (-10.6, -27.2),
    "CA": (-8.5, -22.7), "CT": (-7.8, -21.0),
    "CG": (-10.6, -27.2), "CC": (-8.0, -19.9),
}

# Mismatch penalties by type (kcal/mol)
MISMATCH_PENALTIES = {
    "purine_pyrimidine": 3.5,   # A->C, G->T: severe steric clash
    "pyrimidine_purine": 2.8,   # C->A, T->G: moderate clash
    "transition": 1.8,          # A->G, G->A, C->T, T->C: wobble
    "default": 2.5,
}

_DNA_TO_RNA_RC = {"A": "U", "T": "A", "C": "G", "G": "C", "N": "N"}
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
_PURINES = set("AG")
_PYRIMIDINES = set("CT")


def _dna_to_crrna(spacer_dna: str) -> str:
    """Convert DNA spacer to crRNA (reverse complement, T->U)."""
    return "".join(_DNA_TO_RNA_RC.get(b, "N") for b in reversed(spacer_dna.upper()))


def _classify_mismatch(ref_base: str, alt_base: str) -> str:
    """Classify mismatch type for penalty lookup."""
    r, a = ref_base.upper(), alt_base.upper()
    if r in _PURINES and a in _PYRIMIDINES:
        return "purine_pyrimidine"
    if r in _PYRIMIDINES and a in _PURINES:
        return "pyrimidine_purine"
    if (r in _PURINES and a in _PURINES) or (r in _PYRIMIDINES and a in _PYRIMIDINES):
        return "transition"
    return "default"


def compute_cumulative_dg_profile(
    rna_spacer: str,
    temperature_celsius: float = 37.0,
) -> list[float]:
    """Cumulative dG at each position of the RNA:DNA hybrid.

    Position 1 = PAM-proximal (seed start).
    Returns list of len(rna_spacer) floats.
    """
    T = temperature_celsius + 273.15
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
        cumulative.append(round(running, 2))

    return cumulative


def compute_per_position_dg(rna_spacer: str, temperature_celsius: float = 37.0) -> list[float]:
    """Per-position dG contribution (not cumulative)."""
    T = temperature_celsius + 273.15
    seq = rna_spacer.upper()
    per_pos = [0.0]

    for i in range(len(seq) - 1):
        dinuc = seq[i:i + 2]
        if dinuc in RNA_DNA_NN:
            dH, dS = RNA_DNA_NN[dinuc]
            step_dg = dH - T * (dS / 1000.0)
        else:
            step_dg = -1.0
        per_pos.append(round(step_dg, 2))

    return per_pos


def compute_wt_profile(
    rna_spacer: str,
    snp_position: int,
    mismatch_type: str = "default",
    temperature_celsius: float = 37.0,
) -> list[float]:
    """Cumulative dG profile for the wildtype target (mismatch at SNP).

    At snp_position, the RNA:DNA pair is mismatched — adds a penalty.
    All downstream positions shift by this penalty (cumulative).
    """
    profile = compute_cumulative_dg_profile(rna_spacer, temperature_celsius)
    penalty = MISMATCH_PENALTIES.get(mismatch_type, 2.5)
    if 0 < snp_position < len(profile):
        for i in range(snp_position, len(profile)):
            profile[i] = round(profile[i] + penalty, 2)
    return profile


def compute_target_unwinding_cost(target_dna: str) -> float:
    """Cost to denature dsDNA at the target site (SantaLucia 1998).

    Returns positive value (cost, unfavourable).
    """
    T = 310.15  # 37 C
    dH = 0.1  # initiation
    dS = -2.8
    seq = target_dna.upper()
    for i in range(len(seq) - 1):
        dinuc = seq[i:i + 2]
        if dinuc in DNA_DNA_NN:
            h, s = DNA_DNA_NN[dinuc]
            dH += h
            dS += s
    duplex_dg = dH - T * (dS / 1000.0)
    return round(-duplex_dg, 2)


def compute_spacer_unfolding_cost(rna_spacer: str) -> float:
    """Cost to unfold the crRNA spacer from its self-folded state.

    Uses compute_spacer_folding_dg if available, else GC estimate.
    Returns positive value (cost, unfavourable).
    Zhang et al. (NAR 2024): self-folded spacers are slow.
    """
    mfe = compute_spacer_folding_dg(rna_spacer)
    return round(-mfe, 2)  # negate: unfolding costs energy


def get_thermo_profile(
    spacer_dna: str,
    pam_seq: str,
    snp_position: int | None = None,
) -> dict:
    """Full thermodynamic profile for a candidate.

    Returns dict with:
    - Cumulative R-loop profiles (MUT and WT)
    - Per-position energy contributions
    - Energy budget decomposition (Zhang et al. 2024)
    - Scalar metrics
    """
    crrna = _dna_to_crrna(spacer_dna)
    positions = list(range(1, len(crrna) + 1))

    # Profiles
    mut_cumulative = compute_cumulative_dg_profile(crrna)
    per_pos = compute_per_position_dg(crrna)

    # WT profile with mismatch
    wt_cumulative = None
    snp_barrier = None
    mismatch_type = "default"
    if snp_position and 0 < snp_position <= len(crrna):
        wt_cumulative = compute_wt_profile(crrna, snp_position, mismatch_type)
        snp_barrier = round(wt_cumulative[snp_position] - mut_cumulative[snp_position], 2)

    # Energy budget (Zhang et al. 2024)
    protospacer = spacer_dna[:20] if len(spacer_dna) >= 20 else spacer_dna
    hybrid_dg = compute_hybrid_dg(crrna)
    target_unwinding = compute_target_unwinding_cost(protospacer)
    spacer_unfolding = compute_spacer_unfolding_cost(crrna)
    # Net nucleic acid dG: hybrid_dg is negative (gain), costs are positive
    # ΔG_total = ΔG_hybrid + ΔG_unwind + ΔG_unfold
    net_dg = round(hybrid_dg + target_unwinding + spacer_unfolding, 2)

    # Scalar features
    melting_tm = compute_melting_temperature(crrna)
    seed_len = min(8, len(mut_cumulative) - 1)
    seed_dg = mut_cumulative[seed_len] if seed_len > 0 else 0.0
    gc_content = sum(1 for b in spacer_dna.upper() if b in "GC") / max(len(spacer_dna), 1)

    result = {
        "spacer_dna": spacer_dna,
        "pam_seq": pam_seq,
        "crrna_spacer": crrna,
        "snp_position": snp_position,
        "mutant_profile": {
            "positions": positions,
            "cumulative_dg": mut_cumulative,
        },
        "per_position_dg": per_pos,
        "energy_budget": {
            "spacer_unfolding_cost": round(spacer_unfolding, 2),
            "target_unwinding_cost": round(target_unwinding, 2),
            "hybrid_formation_dg": round(hybrid_dg, 2),
            "net_dg": net_dg,
            "note": (
                "Protein-mediated stabilisation not included. Cas12a provides "
                "additional binding energy (~10\u201330 kcal/mol from PAM recognition, "
                "REC domain contacts, and conformational coupling) that makes "
                "R-loop formation favourable in vivo."
            ),
        },
        "scalars": {
            "net_dg": net_dg,
            "seed_dg": round(seed_dg, 2),
            "hybrid_dg": round(hybrid_dg, 2),
            "melting_tm": round(melting_tm, 1),
            "target_unwinding": target_unwinding,
            "spacer_unfolding": spacer_unfolding,
            "gc_content": round(gc_content * 100, 1),
            "snp_barrier": snp_barrier,
        },
        "references": [
            "Sugimoto et al. (1995) Biochemistry 34:11211",
            "SantaLucia (1998) PNAS 95:1460",
            "Zhang et al. (2024) NAR 52:14077, DOI: 10.1093/nar/gkae1124",
        ],
    }

    if wt_cumulative is not None:
        result["wildtype_profile"] = {
            "positions": positions,
            "cumulative_dg": wt_cumulative,
        }

    return result
