"""Thermodynamic features for CRISPR-Cas12a guide scoring.

Three scalar features encoding biophysical knowledge:

1. Spacer folding dG: self-complementarity of crRNA spacer.
   Strong self-folding (dG << 0) competes with Cas12a loading.

2. RNA-DNA hybrid dG: predicted binding energy of the crRNA-target duplex.
   Stronger binding (more negative dG) = more stable R-loop = higher activity.
   Uses nearest-neighbor parameters from Sugimoto et al. Biochemistry 1995.

3. Melting temperature Tm: temperature at which 50% of RNA-DNA hybrids dissociate.

References:
    Sugimoto et al., Biochemistry 1995, 34(35):11211-11216.
    SantaLucia, PNAS 1998, 95(4):1460-1465.
"""

from __future__ import annotations

import math
import subprocess
import re


# RNA-DNA nearest-neighbor parameters (Sugimoto et al., Biochemistry 1995)
# (dH in kcal/mol, dS in cal/mol/K) for 5'-XY-3' RNA / 3'-X'Y'-5' DNA
RNA_DNA_NN = {
    "AA": (-7.8, -21.9),  "AC": (-5.9, -12.3),
    "AG": (-9.1, -23.5),  "AU": (-8.3, -23.9),
    "CA": (-9.0, -26.1),  "CC": (-9.3, -23.2),
    "CG": (-16.3, -47.1), "CU": (-7.0, -19.7),
    "GA": (-5.5, -13.5),  "GC": (-8.0, -17.1),
    "GG": (-12.8, -31.9), "GU": (-7.8, -21.6),
    "UA": (-7.8, -23.2),  "UC": (-8.6, -22.9),
    "UG": (-10.4, -28.4), "UU": (-11.5, -36.4),
}

INIT_DH = 1.9    # kcal/mol
INIT_DS = -3.9   # cal/mol/K


def compute_spacer_folding_dg(rna_sequence: str) -> float:
    """Compute MFE of crRNA spacer self-folding using RNAfold.

    Falls back to GC-content approximation if ViennaRNA not installed.
    """
    try:
        result = subprocess.run(
            ["RNAfold", "--noPS"],
            input=rna_sequence,
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            match = re.search(r'\(\s*(-?\d+\.\d+)\s*\)', lines[1])
            if match:
                return float(match.group(1))
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Fallback: GC-content approximation
    gc = sum(1 for b in rna_sequence.upper() if b in "GC") / max(len(rna_sequence), 1)
    return -2.0 * gc


def compute_hybrid_dg(
    rna_sequence: str,
    temperature_celsius: float = 37.0,
    na_concentration: float = 0.15,
) -> float:
    """Compute RNA-DNA hybrid dG using nearest-neighbor model."""
    T = temperature_celsius + 273.15
    dH = INIT_DH
    dS = INIT_DS

    seq = rna_sequence.upper()
    for i in range(len(seq) - 1):
        dinuc = seq[i:i+2]
        if dinuc in RNA_DNA_NN:
            h, s = RNA_DNA_NN[dinuc]
            dH += h
            dS += s

    # Salt correction (Owczarzy et al., Biochemistry 2004)
    dS += 0.368 * len(seq) * math.log(na_concentration)

    return dH - T * (dS / 1000.0)


def compute_melting_temperature(
    rna_sequence: str,
    na_concentration: float = 0.15,
    oligo_concentration: float = 250e-9,
) -> float:
    """Compute Tm of RNA-DNA hybrid. Returns degrees Celsius."""
    R = 1.987  # cal/mol/K
    dH = INIT_DH
    dS = INIT_DS

    seq = rna_sequence.upper()
    for i in range(len(seq) - 1):
        dinuc = seq[i:i+2]
        if dinuc in RNA_DNA_NN:
            h, s = RNA_DNA_NN[dinuc]
            dH += h
            dS += s

    dS += 0.368 * len(seq) * math.log(na_concentration)
    dS_total = dS + R * math.log(oligo_concentration / 4.0)

    if dS_total == 0:
        return 0.0

    Tm_K = (dH * 1000.0) / dS_total
    return Tm_K - 273.15


def compute_all_features(target_dna_34: str) -> dict[str, float]:
    """Compute all 3 thermodynamic features from a 34-nt target DNA sequence.

    Extracts the 20-nt protospacer (positions 4-23), converts to crRNA spacer
    (reverse complement, T->U), then computes all features.
    """
    protospacer = target_dna_34[4:24]
    complement = {"A": "U", "T": "A", "C": "G", "G": "C"}
    crrna_spacer = "".join(complement.get(b, b) for b in reversed(protospacer))

    return {
        "folding_dg": compute_spacer_folding_dg(crrna_spacer),
        "hybrid_dg": compute_hybrid_dg(crrna_spacer),
        "melting_tm": compute_melting_temperature(crrna_spacer),
    }
