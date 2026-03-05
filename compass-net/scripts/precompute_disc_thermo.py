"""Precompute 3 thermodynamic features for all discrimination pairs.

For each MUT/WT pair, computes:
  1. ddg_hybrid:         dG(crRNA:MUT) - dG(crRNA:WT) — energetic cost of mismatch
  2. cumulative_dg_at_mm: cumulative R-loop dG up to mismatch position
  3. local_dg:           average of flanking dinucleotide stacking dGs

Saves as: compass-net/data/disc_thermo_features.pt
  dict with 'features' (N x 3 tensor), 'pair_ids' (list),
  'mean' (3,), 'std' (3,) for z-score normalisation at inference.

Usage (from compass/ root):
    python compass-net/scripts/precompute_disc_thermo.py
"""

from __future__ import annotations

import logging
import os
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMPASS_NET_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT_DIR = os.path.dirname(_COMPASS_NET_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _COMPASS_NET_DIR)

from features.thermodynamic import RNA_DNA_NN, INIT_DH, INIT_DS, compute_hybrid_dg
from data.extract_discrimination_pairs import extract_discrimination_pairs

logger = logging.getLogger(__name__)

_DNA_TO_RNA = {"A": "U", "T": "A", "C": "G", "G": "C"}


def _guide_to_crrna(guide_seq: str) -> str:
    """Convert 25-nt guide (4 PAM + 21 spacer) to 20-nt crRNA spacer (RNA)."""
    spacer_dna = guide_seq[4:24] if len(guide_seq) >= 24 else guide_seq[:20]
    return "".join(_DNA_TO_RNA.get(b, "N") for b in reversed(spacer_dna.upper()))


def _compute_cumulative_dg(rna_spacer: str, temperature_c: float = 37.0) -> list[float]:
    """Cumulative RNA:DNA hybrid dG at each spacer position (PAM-proximal first)."""
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


def _compute_local_dg(rna_spacer: str, spacer_position: int, temperature_c: float = 37.0) -> float:
    """Average of the two dinucleotide stacking dGs flanking the mismatch position."""
    T = temperature_c + 273.15
    seq = rna_spacer.upper()
    idx = spacer_position - 1  # 0-indexed

    steps = []
    for i in [idx - 1, idx]:
        if 0 <= i < len(seq) - 1:
            dinuc = seq[i:i + 2]
            if dinuc in RNA_DNA_NN:
                dH, dS = RNA_DNA_NN[dinuc]
                steps.append(dH - T * (dS / 1000.0))
            else:
                steps.append(-1.0)

    return sum(steps) / max(len(steps), 1)


def compute_thermo_for_pair(guide_seq: str, spacer_position: int) -> list[float]:
    """Compute 3 thermo features for a single discrimination pair.

    Returns [ddg_hybrid, cumulative_dg_at_mm, local_dg].
    """
    crrna = _guide_to_crrna(guide_seq)

    # ddg_hybrid: dG difference between perfect-match and mismatch hybrid
    # For the perfect match (MUT), the crRNA fully complements the target.
    # For the mismatch (WT), one position differs.
    # We approximate ddg as the hybrid dG of the full crRNA vs a modified version
    # where the mismatched position is replaced.
    # Simpler: use the mismatch ddG lookup from thermo_discrimination_features
    dg_full = compute_hybrid_dg(crrna)

    # Create a crRNA with the mismatched base replaced (simulating weaker binding)
    # The WT target has a different base at spacer_position, so the crRNA:WT
    # hybrid has a mismatch there. We approximate by computing dG without that position's
    # nearest-neighbor contribution.
    idx = spacer_position - 1  # 0-indexed in crRNA
    # Remove the contribution of dinucleotides spanning the mismatch
    T = 37.0 + 273.15
    penalty = 0.0
    for i in [idx - 1, idx]:
        if 0 <= i < len(crrna) - 1:
            dinuc = crrna[i:i + 2]
            if dinuc in RNA_DNA_NN:
                dH, dS = RNA_DNA_NN[dinuc]
                penalty += dH - T * (dS / 1000.0)
    # The mismatch costs ~+2-4 kcal/mol relative to the match
    # ddg = dG(MUT) - dG(WT) ≈ penalty (the matching contribution that's lost)
    ddg_hybrid = penalty  # negative value = MUT binds more strongly (expected)

    # Cumulative dG up to mismatch position
    cumulative = _compute_cumulative_dg(crrna)
    cum_idx = min(spacer_position, len(cumulative) - 1)
    cumulative_dg_at_mm = cumulative[cum_idx]

    # Local dG around mismatch
    local_dg = _compute_local_dg(crrna, spacer_position)

    return [ddg_hybrid, cumulative_dg_at_mm, local_dg]


def main():
    logging.basicConfig(level=logging.INFO)

    output_path = os.path.join(_ROOT_DIR, "compass-net/data/disc_thermo_features.pt")

    logger.info("Extracting discrimination pairs...")
    pairs = extract_discrimination_pairs()
    logger.info("Got %d pairs", len(pairs))

    features = []
    pair_ids = []
    for p in pairs:
        feats = compute_thermo_for_pair(p.guide_seq, p.spacer_position)
        features.append(feats)
        pair_ids.append(p.guide_id)

    features_tensor = torch.tensor(features, dtype=torch.float32)
    logger.info("Features tensor: %s", features_tensor.shape)

    # Z-score normalisation stats
    mean = features_tensor.mean(dim=0)
    std = features_tensor.std(dim=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    logger.info("Feature stats (raw):")
    names = ["ddg_hybrid", "cumulative_dg_at_mm", "local_dg"]
    for i, name in enumerate(names):
        logger.info("  %s: mean=%.4f, std=%.4f", name, mean[i].item(), std[i].item())

    # Normalise
    features_norm = (features_tensor - mean) / std

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "features": features_norm,
        "features_raw": features_tensor,
        "pair_ids": pair_ids,
        "mean": mean,
        "std": std,
        "feature_names": names,
    }, output_path)
    logger.info("Saved to %s", output_path)

    # Sanity check: correlation with discrimination ratio
    import numpy as np
    from scipy.stats import pearsonr

    ratios = np.array([p.ratio_linear for p in pairs])
    log_ratios = np.log1p(ratios)
    for i, name in enumerate(names):
        r, _ = pearsonr(features_tensor[:, i].numpy(), log_ratios)
        logger.info("  %s vs log(1+ratio): r=%.4f", name, r)


if __name__ == "__main__":
    main()
