"""Sequence-level data augmentation for CRISPR guide activity prediction.

Three augmentation strategies, each biologically justified:

1. Reverse complement (RC):
   DNA is double-stranded; the Cas12a target can be on either strand.
   RC of the 34-nt target should predict the same activity. This doubles
   the effective training set and teaches strand-invariant features.

2. Flanking region shuffle:
   Positions 24-33 (10 nt downstream of spacer) do not contact the
   R-loop and should not affect activity. Randomly shuffling these
   positions teaches the model to ignore irrelevant flanking context.

3. Synthetic mismatch injection:
   Generate artificial MUT/WT pairs by introducing single-nucleotide
   changes at random spacer positions. This creates additional
   discrimination training pairs from efficiency-only data, following
   the self-distillation principle from Phase 2 training.

References:
    Cas-FM (arxiv:2506.11182) — augmentation for out-of-distribution robustness
    Kim et al., Nature Biotechnology 2018 — original Cas12a activity dataset
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


# DNA complement map
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
_BASES = ["A", "C", "G", "T"]


def reverse_complement_onehot(target_onehot: np.ndarray) -> np.ndarray:
    """Reverse complement a one-hot encoded DNA sequence.

    Args:
        target_onehot: (4, 34) one-hot encoded target DNA [A, C, G, T].

    Returns:
        (4, 34) one-hot of the reverse complement.
    """
    # Reverse along position axis
    rc = target_onehot[:, ::-1].copy()
    # Complement: swap A↔T (channels 0↔3) and C↔G (channels 1↔2)
    rc = rc[[3, 2, 1, 0], :]
    return rc


def shuffle_flanking(
    target_onehot: np.ndarray,
    spacer_end: int = 24,
) -> np.ndarray:
    """Shuffle downstream flanking positions (don't contact R-loop).

    Positions spacer_end to 33 are permuted randomly. PAM (0-3) and
    spacer (4-23) are preserved exactly.

    Args:
        target_onehot: (4, 34) one-hot encoded target DNA.
        spacer_end: first flanking position (default 24).

    Returns:
        (4, 34) one-hot with shuffled flanking region.
    """
    aug = target_onehot.copy()
    flank = aug[:, spacer_end:]
    perm = np.random.permutation(flank.shape[1])
    aug[:, spacer_end:] = flank[:, perm]
    return aug


def inject_synthetic_mismatch(
    target_onehot: np.ndarray,
    avoid_positions: Optional[list[int]] = None,
    spacer_start: int = 4,
    spacer_end: int = 24,
) -> tuple[np.ndarray, int]:
    """Inject a single-nucleotide mismatch at a random spacer position.

    Creates a synthetic WT target from a MUT target by changing one
    nucleotide within the spacer region. Used to generate artificial
    discrimination training pairs.

    Args:
        target_onehot: (4, 34) one-hot encoded MUT target DNA.
        avoid_positions: list of absolute positions to skip (e.g., existing SNP).
        spacer_start: first spacer position (default 4).
        spacer_end: last spacer position exclusive (default 24).

    Returns:
        (wt_onehot, mismatch_position):
            wt_onehot is (4, 34) with one changed position.
            mismatch_position is 1-indexed from PAM-proximal.
    """
    avoid = set(avoid_positions or [])
    candidates = [p for p in range(spacer_start, spacer_end) if p not in avoid]
    if not candidates:
        return target_onehot.copy(), 0

    pos = random.choice(candidates)
    wt = target_onehot.copy()

    # Current base at this position
    current_idx = np.argmax(target_onehot[:, pos])
    # Choose a different base
    alternatives = [i for i in range(4) if i != current_idx]
    new_idx = random.choice(alternatives)

    wt[:, pos] = 0
    wt[new_idx, pos] = 1

    # Convert to 1-indexed from spacer start
    mm_pos = pos - spacer_start + 1
    return wt, mm_pos


class SequenceAugmenter:
    """Apply sequence-level augmentations during training.

    Usage:
        aug = SequenceAugmenter(rc_prob=0.5, shuffle_prob=0.3)
        augmented = aug(target_onehot)  # for efficiency training
        mut, wt, pos = aug.generate_disc_pair(target_onehot)  # for discrimination
    """

    def __init__(
        self,
        rc_prob: float = 0.5,
        shuffle_prob: float = 0.3,
        mismatch_prob: float = 0.0,
    ):
        self.rc_prob = rc_prob
        self.shuffle_prob = shuffle_prob
        self.mismatch_prob = mismatch_prob

    def __call__(self, target_onehot: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a single target.

        Args:
            target_onehot: (4, 34) one-hot encoded target DNA.

        Returns:
            (4, 34) augmented one-hot.
        """
        x = target_onehot
        if random.random() < self.rc_prob:
            x = reverse_complement_onehot(x)
        if random.random() < self.shuffle_prob:
            x = shuffle_flanking(x)
        return x

    def augment_batch(
        self,
        batch: np.ndarray,
    ) -> np.ndarray:
        """Apply augmentations to a batch.

        Args:
            batch: (N, 4, 34) one-hot encoded targets.

        Returns:
            (N, 4, 34) augmented batch.
        """
        return np.stack([self(x) for x in batch])

    def generate_disc_pair(
        self,
        target_onehot: np.ndarray,
        avoid_positions: Optional[list[int]] = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Generate a synthetic MUT/WT pair for discrimination training.

        Args:
            target_onehot: (4, 34) MUT target (perfect guide match).
            avoid_positions: positions to skip for mismatch injection.

        Returns:
            (mut_onehot, wt_onehot, mm_position_1indexed)
        """
        wt, mm_pos = inject_synthetic_mismatch(
            target_onehot, avoid_positions=avoid_positions,
        )
        return target_onehot.copy(), wt, mm_pos
