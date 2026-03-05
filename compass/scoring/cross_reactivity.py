"""Cross-reactivity scoring for multiplex CRISPR diagnostic panels.

Evaluates every crRNA spacer against every other target's amplicon to
identify potential off-target cleavage that could cause false positives
in a multiplexed detection assay.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Position-weighted mismatch penalty model
# ---------------------------------------------------------------------------
# Index 0 = PAM position 1 (5' end of PAM), index 23 = spacer position 20.
# The PAM+spacer window is 24 nt total: 4 nt PAM + 20 nt spacer.

def _build_penalty_vector() -> list[float]:
    """Return a 24-element penalty vector (4 PAM + 20 spacer positions)."""
    penalties = [0.0] * 24

    # PAM positions 1-2 (indices 0-1)
    for i in range(0, 2):
        penalties[i] = 0.01
    # PAM positions 3-4 (indices 2-3)
    for i in range(2, 4):
        penalties[i] = 0.05

    # Spacer positions 1-3 / seed (indices 4-6)
    for i in range(4, 7):
        penalties[i] = 0.05
    # Spacer positions 4-5 (indices 7-8)
    for i in range(7, 9):
        penalties[i] = 0.15
    # Spacer positions 6-8 (indices 9-11)
    for i in range(9, 12):
        penalties[i] = 0.30
    # Spacer positions 9-12 (indices 12-15)
    for i in range(12, 16):
        penalties[i] = 0.50
    # Spacer positions 13-20 (indices 16-23)
    for i in range(16, 24):
        penalties[i] = 0.70

    return penalties


_PENALTY_VECTOR = _build_penalty_vector()

# Valid TTTV PAM: T at positions 0-2, then A/C/G at position 3
_VALID_PAM_PREFIX = {"T"}
_VALID_PAM_V = {"A", "C", "G"}


def _is_valid_pam(pam: str) -> bool:
    """Check whether a 4-nt string is a valid TTTV PAM."""
    if len(pam) != 4:
        return False
    return (
        pam[0] in _VALID_PAM_PREFIX
        and pam[1] in _VALID_PAM_PREFIX
        and pam[2] in _VALID_PAM_PREFIX
        and pam[3] in _VALID_PAM_V
    )


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


def _score_window(spacer: str, window_24: str) -> tuple[float, int]:
    """Score a 24-nt window (4 PAM + 20 spacer) against a spacer.

    Returns (predicted_activity, n_mismatches).
    Activity = 1.0 - sum(penalties for mismatched positions).
    If total mismatches >= 6, activity is forced to 0.
    """
    pam = window_24[:4]
    target_spacer = window_24[4:]

    if not _is_valid_pam(pam):
        return 0.0, 24  # no valid PAM → no activity

    n_mismatches = 0
    penalty_sum = 0.0

    # PAM mismatches (positions 0-3 of window vs canonical TTTV)
    canonical_pam = "TTT" + pam[3]  # the V base is already correct
    for i in range(4):
        if pam[i] != canonical_pam[i]:
            penalty_sum += _PENALTY_VECTOR[i]
            n_mismatches += 1

    # Spacer mismatches (positions 4-23)
    for i in range(20):
        if spacer[i].upper() != target_spacer[i].upper():
            penalty_sum += _PENALTY_VECTOR[i + 4]
            n_mismatches += 1

    if n_mismatches >= 6:
        return 0.0, n_mismatches

    activity = max(0.0, 1.0 - penalty_sum)
    return activity, n_mismatches


def _classify_risk(activity: float) -> str:
    """Classify predicted off-target activity into a risk tier."""
    if activity < 0.01:
        return "none"
    elif activity <= 0.05:
        return "low"
    elif activity <= 0.15:
        return "medium"
    else:
        return "high"


def _best_off_target_score(spacer: str, amplicon: str) -> dict:
    """Slide a 24-nt window across the amplicon (both strands) and return the
    best (highest activity) off-target hit.

    Uses simple sliding-window alignment, not full Smith-Waterman.
    """
    spacer_upper = spacer.upper()
    if len(spacer_upper) < 20:
        return {"activity": 0.0, "mismatches": 24, "risk": "none", "position": -1, "strand": "+"}

    # Truncate or pad spacer to 20 nt
    spacer_20 = spacer_upper[:20]
    window_len = 24  # 4 PAM + 20 spacer

    best = {"activity": 0.0, "mismatches": 24, "risk": "none", "position": -1, "strand": "+"}

    for strand_label, seq in [("+", amplicon.upper()), ("-", _reverse_complement(amplicon).upper())]:
        if len(seq) < window_len:
            continue
        for start in range(len(seq) - window_len + 1):
            window = seq[start : start + window_len]
            activity, mm = _score_window(spacer_20, window)
            if activity > best["activity"]:
                best = {
                    "activity": round(activity, 4),
                    "mismatches": mm,
                    "risk": _classify_risk(activity),
                    "position": start,
                    "strand": strand_label,
                }
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_cross_reactivity_matrix(
    spacers: list[str],
    amplicons: list[str],
    labels: list[str],
) -> dict:
    """Score every crRNA spacer against every other target's amplicon.

    Parameters
    ----------
    spacers : list[str]
        crRNA spacer sequences (one per panel member).
    amplicons : list[str]
        Amplicon sequences (one per panel member).
    labels : list[str]
        Human-readable labels for each panel member (e.g. gene_mutation).

    Returns
    -------
    dict with keys:
        matrix        – list of pair dicts with scoring details
        n_targets     – number of panel members
        n_pairs       – number of off-target pairs evaluated
        high_risk_pairs – count of pairs classified as "high"
        same_gene_pairs – count of pairs sharing the same gene prefix
        panel_safe    – True if no high-risk pairs detected
        interpretation – human-readable summary string
    """
    n = len(spacers)
    matrix: list[dict] = []
    high_risk_count = 0
    same_gene_count = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # skip self

            hit = _best_off_target_score(spacers[i], amplicons[j])

            # Check if same gene (first token before underscore)
            gene_i = labels[i].split("_")[0] if "_" in labels[i] else labels[i]
            gene_j = labels[j].split("_")[0] if "_" in labels[j] else labels[j]
            same_gene = gene_i == gene_j

            if same_gene:
                same_gene_count += 1

            pair = {
                "crRNA_index": i,
                "crRNA_label": labels[i],
                "target_index": j,
                "target_label": labels[j],
                "same_gene": same_gene,
                **hit,
            }
            matrix.append(pair)

            if hit["risk"] == "high":
                high_risk_count += 1

    n_pairs = len(matrix)
    panel_safe = high_risk_count == 0

    # Build interpretation string
    if panel_safe:
        interpretation = (
            f"Panel of {n} targets has no high-risk cross-reactive pairs. "
            f"{n_pairs} off-target pairs evaluated."
        )
    else:
        interpretation = (
            f"WARNING: {high_risk_count} high-risk cross-reactive pair(s) detected "
            f"among {n_pairs} off-target pairs in a {n}-target panel. "
            f"Consider redesigning affected crRNAs or adjusting amplicon boundaries."
        )

    return {
        "matrix": matrix,
        "n_targets": n,
        "n_pairs": n_pairs,
        "high_risk_pairs": high_risk_count,
        "same_gene_pairs": same_gene_count,
        "panel_safe": panel_safe,
        "interpretation": interpretation,
    }
