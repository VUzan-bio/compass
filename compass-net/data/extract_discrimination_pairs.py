"""Extract paired MUT/WT discrimination measurements from EasyDesign.

For each crRNA guide tested against both a perfect-match target (HD=0)
and single-mismatch targets (HD=1), we extract:
  - MUT activity: trans-cleavage with perfect match (guide == target)
  - WT activity:  trans-cleavage with 1 mismatch (guide != target at 1 pos)
  - Mismatch position (1-indexed from PAM-proximal end)
  - Mismatch type (RNA:DNA pair)
  - Discrimination ratio = 10^(MUT_logk) / 10^(WT_logk)

The EasyDesign dataset uses LbCas12a with log-k fluorescence readout.
Positive log-k = high activity; negative = low. The ratio of activities
in linear space is what matters for discrimination.

Source: Huang et al., iMeta 2024 — Table S2 Training data.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# RNA complement of DNA base (for crRNA)
_DNA_TO_RNA = {"A": "U", "T": "A", "C": "G", "G": "C"}
_PURINES = {"A", "G"}
_PYRIMIDINES = {"C", "T", "U"}


@dataclass
class DiscriminationPair:
    """A single MUT/WT paired measurement for discrimination training."""

    guide_seq: str          # 25-nt guide (4 PAM + 21 spacer)
    mut_target: str         # perfect-match target (= guide)
    wt_target: str          # mismatched target (1 nt differs)
    mut_activity: float     # log-k fluorescence, perfect match
    wt_activity: float      # log-k fluorescence, mismatched
    mismatch_position: int  # 1-indexed from PAM-proximal (pos 5 in 25-nt = spacer pos 1)
    mismatch_ref: str       # WT DNA base at mismatch position
    mismatch_alt: str       # MUT DNA base at mismatch position (= guide base)
    rna_base: str           # crRNA RNA base at mismatch position
    dna_base_wt: str        # WT target DNA base (mismatched with crRNA)
    mismatch_type: str      # e.g., "rA:dG" RNA:DNA mismatch classification
    spacer_position: int    # 1-indexed position within the 20-nt spacer (PAM-proximal)
    guide_id: str = ""      # unique identifier

    @property
    def ratio_linear(self) -> float:
        """Discrimination ratio in linear activity space.

        10^(mut_logk) / 10^(wt_logk) = 10^(mut - wt).
        Higher = better discrimination.
        """
        return 10 ** (self.mut_activity - self.wt_activity)

    @property
    def delta_logk(self) -> float:
        """Activity difference in log-k space (MUT - WT)."""
        return self.mut_activity - self.wt_activity

    @property
    def in_seed(self) -> bool:
        """Mismatch in seed region (spacer positions 1-8)."""
        return 1 <= self.spacer_position <= 8


def _classify_rna_dna_mismatch(rna_base: str, dna_base: str) -> str:
    """Classify RNA:DNA mismatch pair."""
    r = rna_base.upper().replace("T", "U")
    d = dna_base.upper()
    return f"r{r}:d{d}"


def extract_discrimination_pairs(
    xlsx_path: str = "compass-net/data/external/easydesign/Table_S2.xlsx",
    max_hamming: int = 1,
    min_pairs_per_guide: int = 1,
) -> list[DiscriminationPair]:
    """Extract paired MUT/WT measurements from EasyDesign Training data.

    Strategy:
      1. Find all guides with HD=0 measurement (perfect match = MUT baseline)
      2. For each, find HD=1 measurements (single mismatch = WT scenario)
      3. Pair them: same guide, different target → discrimination pair

    Args:
        xlsx_path: path to Table S2.xlsx
        max_hamming: maximum hamming distance for WT targets (1 for single-mismatch)
        min_pairs_per_guide: minimum number of WT targets per guide to include

    Returns:
        List of DiscriminationPair objects.
    """
    if not os.path.isabs(xlsx_path):
        xlsx_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", xlsx_path,
        )

    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(
            f"EasyDesign data not found at {xlsx_path}. "
            f"Download from https://github.com/scRNA-Compt/EasyDesign"
        )

    df = pd.read_excel(xlsx_path, sheet_name="Training data")
    logger.info("Loaded %d rows from EasyDesign Training data", len(df))

    # Step 1: Index perfect-match (HD=0) rows by guide
    hd0 = df[df["guide_target_hamming_dist"] == 0].copy()
    # Average activity if multiple HD=0 measurements for same guide
    guide_mut_activity = hd0.groupby("guide_seq")["30 min"].mean().to_dict()

    # Step 2: Get single-mismatch (HD=1) rows
    hd1 = df[
        (df["guide_target_hamming_dist"] >= 1)
        & (df["guide_target_hamming_dist"] <= max_hamming)
    ].copy()

    # Step 3: Build pairs
    pairs: list[DiscriminationPair] = []
    pair_id = 0

    for _, row in hd1.iterrows():
        guide = str(row["guide_seq"]).upper()
        target = str(row["target_at_guide"]).upper()
        wt_activity = float(row["30 min"])

        if guide not in guide_mut_activity:
            continue  # no HD=0 baseline for this guide

        mut_activity = guide_mut_activity[guide]

        # Find mismatch position(s)
        if len(guide) != len(target):
            continue

        diffs = []
        for i in range(len(guide)):
            if guide[i] != target[i]:
                diffs.append(i)

        if len(diffs) != 1:
            continue  # expect exactly 1 mismatch for HD=1

        mm_idx = diffs[0]  # 0-indexed in the 25-nt sequence

        # PAM is positions 0-3 (TTTV), spacer starts at position 4
        pam_len = 4
        if mm_idx < pam_len:
            continue  # mismatch in PAM, not relevant for crRNA discrimination

        # Spacer position: 1-indexed from PAM-proximal end
        spacer_pos = mm_idx - pam_len + 1  # position 4 → spacer_pos 1

        # Bases
        mut_dna = guide[mm_idx]    # guide base = matches MUT target
        wt_dna = target[mm_idx]    # WT target base (mismatched)

        # crRNA base: RNA complement of the guide/MUT DNA base
        rna_base = _DNA_TO_RNA.get(mut_dna, "N")

        # Mismatch type: crRNA (RNA) vs WT target (DNA)
        mm_type = _classify_rna_dna_mismatch(rna_base, wt_dna)

        pair_id += 1
        pair = DiscriminationPair(
            guide_seq=guide,
            mut_target=guide,  # perfect match
            wt_target=target,
            mut_activity=mut_activity,
            wt_activity=wt_activity,
            mismatch_position=mm_idx + 1,  # 1-indexed in full 25-nt
            mismatch_ref=wt_dna,
            mismatch_alt=mut_dna,
            rna_base=rna_base,
            dna_base_wt=wt_dna,
            mismatch_type=mm_type,
            spacer_position=spacer_pos,
            guide_id=f"ED_disc_{pair_id:05d}",
        )
        pairs.append(pair)

    # Filter guides with min_pairs
    if min_pairs_per_guide > 1:
        from collections import Counter
        guide_counts = Counter(p.guide_seq for p in pairs)
        pairs = [p for p in pairs if guide_counts[p.guide_seq] >= min_pairs_per_guide]

    logger.info(
        "Extracted %d discrimination pairs from %d unique guides",
        len(pairs),
        len(set(p.guide_seq for p in pairs)),
    )

    # Summary stats
    ratios = [p.ratio_linear for p in pairs]
    seed_pairs = [p for p in pairs if p.in_seed]
    logger.info(
        "Ratio stats: median=%.2f, mean=%.2f, seed_pairs=%d/%d",
        np.median(ratios),
        np.mean(ratios),
        len(seed_pairs),
        len(pairs),
    )

    return pairs


def pairs_to_dataframe(pairs: list[DiscriminationPair]) -> pd.DataFrame:
    """Convert pairs to a DataFrame for analysis and training."""
    records = []
    for p in pairs:
        records.append({
            "guide_id": p.guide_id,
            "guide_seq": p.guide_seq,
            "mut_target": p.mut_target,
            "wt_target": p.wt_target,
            "mut_activity": p.mut_activity,
            "wt_activity": p.wt_activity,
            "delta_logk": p.delta_logk,
            "ratio_linear": p.ratio_linear,
            "mismatch_position": p.mismatch_position,
            "spacer_position": p.spacer_position,
            "mismatch_ref": p.mismatch_ref,
            "mismatch_alt": p.mismatch_alt,
            "rna_base": p.rna_base,
            "dna_base_wt": p.dna_base_wt,
            "mismatch_type": p.mismatch_type,
            "in_seed": p.in_seed,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pairs = extract_discrimination_pairs()
    df = pairs_to_dataframe(pairs)
    print(f"\nExtracted {len(df)} discrimination pairs")
    print(f"\nMismatch type distribution:")
    print(df["mismatch_type"].value_counts())
    print(f"\nSpacer position distribution:")
    print(df["spacer_position"].value_counts().sort_index())
    print(f"\nRatio stats by seed vs non-seed:")
    for is_seed, grp in df.groupby("in_seed"):
        label = "seed (1-8)" if is_seed else "non-seed (9+)"
        print(f"  {label}: n={len(grp)}, median_ratio={grp['ratio_linear'].median():.3f}, "
              f"mean_ratio={grp['ratio_linear'].mean():.3f}")
