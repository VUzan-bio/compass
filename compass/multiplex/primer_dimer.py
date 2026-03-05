"""
Thermodynamic primer-dimer prediction for multiplex RPA panels.

Uses SantaLucia nearest-neighbor (NN) parameters to estimate the free energy
of duplex formation between any two oligonucleotides in a panel.  The module
reports both *full-length* (internal) dimers and *3'-anchored* (extensible)
dimers — the latter being the primary concern for multiplex RPA because
recombinase-mediated extension of a misprimed 3' end produces amplification
artifacts that compete with genuine targets.

Thermodynamic model
-------------------
- NN parameters: SantaLucia & Hicks, *Annu. Rev. Biophys. Biomol. Struct.*
  **33**, 415-440 (2004).  Unified DNA/DNA set (Table 2) with initiation
  penalties for terminal AT pairs.
- Salt correction: SantaLucia (1998) log-linear form,
  ΔG_salt = ΔG_1M + 0.368 × (N/2) × ln([Na⁺]).  Default [Na⁺] is set to
  100 mM to approximate the ionic environment of an RPA reaction buffer
  (magnesium acetate + potassium acetate contributions).

Thresholds
----------
Adapted from PrimerROC (Johnston *et al.*, 2019, *Sci. Rep.* **9**, 209) and
the SADDLE framework (Xie *et al.*, 2022, *PLOS Comput. Biol.* **18**,
e1010474), recalibrated for the lower-temperature, recombinase-driven
extension in RPA:

- ΔG_3' < −6.0 kcal/mol  →  HIGH risk extensible dimer
- ΔG_3' < −4.0 kcal/mol  →  MODERATE risk extensible dimer
- ΔG_full < −8.0 kcal/mol →  stable internal dimer (sequestration risk)

References
----------
1. SantaLucia J Jr & Hicks D (2004). The thermodynamics of DNA structural
   motifs. *Annu. Rev. Biophys. Biomol. Struct.* 33:415-440.
2. SantaLucia J Jr (1998). A unified view of polymer, dumbbell, and
   oligonucleotide DNA nearest-neighbor thermodynamics. *PNAS* 95:1460-1465.
3. Johnston AD *et al.* (2019). PrimerROC: accurate condition-independent
   dimer prediction using ROC analysis. *Sci. Rep.* 9:209.
4. Xie K *et al.* (2022). SADDLE: structure-aware dimer likelihood
   estimation for primer design. *PLOS Comput. Biol.* 18:e1010474.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# SantaLucia & Hicks (2004) unified nearest-neighbour parameters
# ΔH in kcal/mol, ΔS in cal/(mol·K)
# ---------------------------------------------------------------------------

_NN_DH: Dict[str, float] = {
    "AA": -7.9,  "TT": -7.9,
    "AT": -7.2,
    "TA": -7.2,
    "CA": -8.5,  "TG": -8.5,
    "GT": -8.4,  "AC": -8.4,
    "CT": -7.8,  "AG": -7.8,
    "GA": -8.2,  "TC": -8.2,
    "CG": -10.6,
    "GC": -9.8,
    "GG": -8.0,  "CC": -8.0,
}

_NN_DS: Dict[str, float] = {
    "AA": -22.2, "TT": -22.2,
    "AT": -20.4,
    "TA": -21.3,
    "CA": -22.7, "TG": -22.7,
    "GT": -22.4, "AC": -22.4,
    "CT": -21.0, "AG": -21.0,
    "GA": -22.2, "TC": -22.2,
    "CG": -27.2,
    "GC": -24.4,
    "GG": -19.9, "CC": -19.9,
}

# Initiation parameters (SantaLucia & Hicks 2004, Table 2)
_INIT_DH = 0.2   # kcal/mol — per terminal AT pair
_INIT_DS = -5.7   # cal/(mol·K) — per terminal AT pair
_INIT_GC_DH = 0.1
_INIT_GC_DS = -2.8

# Watson-Crick complement
_WC = str.maketrans("ACGT", "TGCA")

# Thresholds (kcal/mol)
THRESHOLD_3PRIME_HIGH = -6.0
THRESHOLD_3PRIME_MODERATE = -4.0
THRESHOLD_FULL_STABLE = -8.0


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _reverse_complement(seq: str) -> str:
    return seq.upper().translate(_WC)[::-1]


def _is_wc_pair(a: str, b: str) -> bool:
    """Return True if *a* and *b* form a Watson-Crick pair."""
    return (a, b) in {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}


def _consecutive_wc_runs(seq1_segment: str, seq2_segment: str) -> List[Tuple[int, int]]:
    """Find all runs of consecutive WC pairs between two equal-length segments.

    Returns a list of (start, length) tuples for each consecutive run.
    """
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(seq1_segment)
    while i < n:
        if _is_wc_pair(seq1_segment[i], seq2_segment[i]):
            start = i
            while i < n and _is_wc_pair(seq1_segment[i], seq2_segment[i]):
                i += 1
            runs.append((start, i - start))
        else:
            i += 1
    return runs


def _dg_for_run(
    seq1_segment: str,
    seq2_segment: str,
    start: int,
    length: int,
    temperature_c: float,
    na_conc: float,
) -> float:
    """Compute ΔG (kcal/mol) for a consecutive WC run using NN parameters.

    *seq2_segment* is read 3'→5' (i.e. already the complement direction).
    Only called when length >= 2 (a single base pair has no NN dinucleotide).
    """
    if length < 2:
        return 0.0

    temp_k = temperature_c + 273.15
    dh_total = 0.0
    ds_total = 0.0

    # NN dinucleotide steps along the run on seq1
    for j in range(start, start + length - 1):
        dinuc = seq1_segment[j] + seq1_segment[j + 1]
        dh_total += _NN_DH.get(dinuc, 0.0)
        ds_total += _NN_DS.get(dinuc, 0.0)

    # Initiation penalties for the terminal pairs of this run
    for idx in (start, start + length - 1):
        pair = seq1_segment[idx] + seq2_segment[idx]
        if pair in {"AT", "TA"}:
            dh_total += _INIT_DH
            ds_total += _INIT_DS
        else:
            dh_total += _INIT_GC_DH
            ds_total += _INIT_GC_DS

    # ΔG at 1 M NaCl
    dg_1m = dh_total - temp_k * ds_total / 1000.0

    # Salt correction — SantaLucia (1998) log-linear approximation
    n_bp = length
    dg_salt = dg_1m + 0.368 * (n_bp / 2.0) * math.log(na_conc)

    return dg_salt


# ---------------------------------------------------------------------------
# Public API — single pair
# ---------------------------------------------------------------------------

@dataclass
class DimerResult:
    """Result of a dimer ΔG calculation for a single pair of oligos."""
    dg_full: float            # most stable ΔG across all alignment offsets
    dg_3prime: float          # most stable ΔG for 3'-anchored alignments only
    full_run_length: int      # length of the run producing dg_full
    prime3_run_length: int    # length of the run producing dg_3prime
    is_self: bool = False     # True when seq1 == seq2 (homodimer)


def compute_dimer_dg(
    seq1: str,
    seq2: str,
    anchor_3prime: bool = False,
    temperature: float = 37.0,
    na_conc: float = 0.1,
) -> DimerResult:
    """Compute the most stable dimer ΔG between two primers.

    The algorithm slides the reverse-complement of *seq2* along *seq1* (and
    vice-versa) to enumerate every possible alignment.  At each offset only
    runs of **consecutive** Watson-Crick base pairs are scored — mismatches
    and bulges break a run.

    Parameters
    ----------
    seq1, seq2 : str
        Primer sequences (5'→3').
    anchor_3prime : bool
        If *True*, only return the 3'-anchored ΔG (extensible dimer mode).
    temperature : float
        Reaction temperature in °C (default 37 for RPA).
    na_conc : float
        Monovalent-cation concentration in M (default 0.1 for RPA buffer).

    Returns
    -------
    DimerResult
    """
    seq1 = seq1.upper().strip()
    seq2 = seq2.upper().strip()
    rc2 = _reverse_complement(seq2)

    is_self = seq1 == seq2

    best_full_dg = 0.0
    best_full_len = 0
    best_3p_dg = 0.0
    best_3p_len = 0

    def _scan(s1: str, s2_rc: str, s2_orig: str) -> None:
        """Slide s2_rc along s1 and update outer best values."""
        nonlocal best_full_dg, best_full_len, best_3p_dg, best_3p_len

        len1 = len(s1)
        len2 = len(s2_rc)

        # Offsets: negative means s2_rc hangs off the left of s1
        for offset in range(-(len2 - 1), len1):
            # Overlapping region indices
            start1 = max(0, offset)
            start2 = max(0, -offset)
            end1 = min(len1, offset + len2)
            overlap = end1 - start1
            if overlap < 2:
                continue

            seg1 = s1[start1:end1]
            seg2 = s2_rc[start2:start2 + overlap]

            runs = _consecutive_wc_runs(seg1, seg2)
            for rstart, rlen in runs:
                if rlen < 2:
                    continue
                dg = _dg_for_run(seg1, seg2, rstart, rlen, temperature, na_conc)
                if dg < best_full_dg:
                    best_full_dg = dg
                    best_full_len = rlen

                # 3'-anchored check: the run must include the 3' end of
                # either s1 or s2_orig.
                # s1 3'-end is at index len1-1 → in segment coords: len1-1 - start1
                s1_3p_seg = len1 - 1 - start1
                # s2_orig 3'-end maps to position 0 in s2_rc → seg2 index: -start2
                s2_3p_seg = -start2  # == 0 when start2 == 0

                run_end = rstart + rlen - 1
                anchored = False
                if rstart <= s1_3p_seg <= run_end and s1_3p_seg == overlap - 1:
                    anchored = True
                if rstart <= s2_3p_seg <= run_end and s2_3p_seg == 0 and start2 == 0:
                    # s2 3' end is at the beginning of s2_rc, which aligns to
                    # start1 in s1.  The run must start at index 0 of the
                    # overlap to be 3'-anchored on s2.
                    if rstart == 0:
                        anchored = True

                if anchored and dg < best_3p_dg:
                    best_3p_dg = dg
                    best_3p_len = rlen

    # Scan both orientations (seq1 as template, then seq2 as template) to
    # capture 3'-anchored dimers from either primer's perspective.
    _scan(seq1, rc2, seq2)
    if not is_self:
        rc1 = _reverse_complement(seq1)
        _scan(seq2, rc1, seq1)

    if anchor_3prime:
        return DimerResult(
            dg_full=best_full_dg,
            dg_3prime=best_3p_dg,
            full_run_length=best_full_len,
            prime3_run_length=best_3p_len,
            is_self=is_self,
        )

    return DimerResult(
        dg_full=best_full_dg,
        dg_3prime=best_3p_dg,
        full_run_length=best_full_len,
        prime3_run_length=best_3p_len,
        is_self=is_self,
    )


# ---------------------------------------------------------------------------
# Public API — panel analysis
# ---------------------------------------------------------------------------

@dataclass
class PanelDimerReport:
    """Summary of pairwise dimer interactions across a multiplex RPA panel."""
    dg_matrix_full: np.ndarray        # (N, N) full-length ΔG matrix
    dg_matrix_3prime: np.ndarray      # (N, N) 3'-anchored ΔG matrix
    oligo_labels: List[str]           # label for each row/column
    flagged_pairs: List[Dict]         # MODERATE risk (ΔG_3' < -4.0)
    high_risk_pairs: List[Dict]       # HIGH risk (ΔG_3' < -6.0)
    internal_dimers: List[Dict]       # stable internal dimers (ΔG_full < -8.0)
    panel_dimer_score: float          # 0-1 composite score (0 = clean)
    recommendations: List[str]


def analyse_panel_dimers(
    primers: Sequence[Dict[str, str]],
    temperature: float = 37.0,
    na_conc: float = 0.1,
) -> PanelDimerReport:
    """Check all pairwise (and self-) dimer interactions in a primer panel.

    Parameters
    ----------
    primers : list of dict
        Each dict must have keys ``target``, ``fwd``, ``rev`` where *fwd* and
        *rev* are 5'→3' primer sequences.
    temperature : float
        Reaction temperature in °C.
    na_conc : float
        Monovalent-cation concentration in M.

    Returns
    -------
    PanelDimerReport
    """
    # Build flat oligo list with labels
    oligos: List[Tuple[str, str]] = []  # (label, sequence)
    for entry in primers:
        target = entry["target"]
        oligos.append((f"{target}_F", entry["fwd"].upper().strip()))
        oligos.append((f"{target}_R", entry["rev"].upper().strip()))

    n = len(oligos)
    dg_full = np.zeros((n, n), dtype=np.float64)
    dg_3p = np.zeros((n, n), dtype=np.float64)

    flagged: List[Dict] = []
    high_risk: List[Dict] = []
    internal: List[Dict] = []

    for i in range(n):
        for j in range(i, n):
            res = compute_dimer_dg(
                oligos[i][1], oligos[j][1],
                anchor_3prime=True,
                temperature=temperature,
                na_conc=na_conc,
            )
            dg_full[i, j] = dg_full[j, i] = res.dg_full
            dg_3p[i, j] = dg_3p[j, i] = res.dg_3prime

            pair_info = {
                "oligo_a": oligos[i][0],
                "oligo_b": oligos[j][0],
                "dg_full": round(res.dg_full, 2),
                "dg_3prime": round(res.dg_3prime, 2),
                "run_length_full": res.full_run_length,
                "run_length_3prime": res.prime3_run_length,
                "is_self": i == j,
            }

            if res.dg_3prime < THRESHOLD_3PRIME_HIGH:
                high_risk.append(pair_info)
            elif res.dg_3prime < THRESHOLD_3PRIME_MODERATE:
                flagged.append(pair_info)

            if res.dg_full < THRESHOLD_FULL_STABLE:
                internal.append(pair_info)

    # --- Panel dimer score (0 = clean, 1 = worst) --------------------------
    # Weighted count of problematic pairs normalised by total pair count.
    total_pairs = n * (n + 1) // 2  # including self
    weighted = len(high_risk) * 1.0 + len(flagged) * 0.4 + len(internal) * 0.3
    raw_score = weighted / max(total_pairs, 1)
    panel_score = min(1.0, raw_score)

    # --- Recommendations ---------------------------------------------------
    recs: List[str] = []
    if high_risk:
        recs.append(
            f"{len(high_risk)} HIGH-risk extensible dimer(s) detected "
            f"(ΔG_3' < {THRESHOLD_3PRIME_HIGH} kcal/mol). "
            "Redesign at least one primer in each pair — shift the 3' end "
            "or introduce a deliberate mismatch at the penultimate base."
        )
    if flagged:
        recs.append(
            f"{len(flagged)} MODERATE-risk extensible dimer(s) detected "
            f"(ΔG_3' < {THRESHOLD_3PRIME_MODERATE} kcal/mol). "
            "Consider testing empirically; these may be tolerable at lower "
            "primer concentrations."
        )
    if internal:
        recs.append(
            f"{len(internal)} stable internal dimer(s) detected "
            f"(ΔG_full < {THRESHOLD_FULL_STABLE} kcal/mol). "
            "These can sequester primer and reduce sensitivity — check "
            "whether affected targets show low amplification efficiency."
        )
    if not recs:
        recs.append("Panel looks clean — no problematic dimers detected.")

    labels = [label for label, _ in oligos]

    return PanelDimerReport(
        dg_matrix_full=dg_full,
        dg_matrix_3prime=dg_3p,
        oligo_labels=labels,
        flagged_pairs=flagged,
        high_risk_pairs=high_risk,
        internal_dimers=internal,
        panel_dimer_score=round(panel_score, 4),
        recommendations=recs,
    )
