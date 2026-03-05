"""Preprocessing utilities for sequence-based ML scoring.

One-hot encoding, input window construction, and label normalisation
for CNN and transformer-based Cas12a activity predictors.

Input convention: channels-first (4, L) for PyTorch Conv1d.
Channel order: A=0, C=1, G=2, T=3.
"""

from __future__ import annotations

import numpy as np

NUCLEOTIDE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot_encode(seq: str, max_len: int = 34) -> np.ndarray:
    """Encode DNA sequence as one-hot matrix.

    Args:
        seq: DNA sequence string (A/C/G/T). Unknown bases are zeros.
        max_len: Fixed output length. Shorter sequences are right-padded
                 with zeros; longer sequences are truncated.

    Returns:
        np.ndarray of shape (4, max_len) — channels-first for Conv1d.
    """
    seq = seq.upper().strip()
    mat = np.zeros((4, max_len), dtype=np.float32)
    for i, nt in enumerate(seq[:max_len]):
        idx = NUCLEOTIDE_MAP.get(nt)
        if idx is not None:
            mat[idx, i] = 1.0
    return mat


def encode_dataset(
    sequences: list[str],
    labels: np.ndarray,
    max_len: int = 34,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a list of sequences into a batch tensor.

    Returns:
        X: np.ndarray of shape (N, 4, max_len)
        y: np.ndarray of shape (N,)
    """
    X = np.stack([one_hot_encode(s, max_len) for s in sequences])
    return X, labels.astype(np.float32)


def extract_input_window(
    pam: str,
    spacer: str,
    upstream_flank: str = "",
    downstream_flank: str = "",
    total_len: int = 34,
) -> str:
    """Construct the 34-nt input window for the CNN.

    Window layout (5' to 3'):
      [upstream_context] [PAM 4nt] [spacer 18-23nt] [downstream_context]

    The window is centered on PAM + spacer. Flanking context fills to
    total_len. If spacer < 23 nt, more downstream context is included.

    For COMPASS integration: the scanner provides PAM, spacer, and
    flanking context from the target's flanking_seq.
    """
    core = pam + spacer
    need_upstream = max(0, total_len - len(core) - len(downstream_flank))
    need_downstream = max(
        0, total_len - len(core) - min(need_upstream, len(upstream_flank))
    )

    up = upstream_flank[-need_upstream:] if need_upstream > 0 else ""
    down = downstream_flank[:need_downstream] if need_downstream > 0 else ""

    window = up + core + down
    # Trim or pad to exact total_len
    window = window[:total_len].ljust(total_len, "N")
    return window


def normalise_labels(
    raw_labels: np.ndarray,
    transform: str = "log",
) -> np.ndarray:
    """Normalise activity labels to [0, 1].

    Args:
        raw_labels: Raw activity measurements (indel freq or signal intensity).
        transform: ``"log"`` for log2(x + 1) then min-max (Kim et al. 2018),
                   ``"minmax"`` for direct min-max scaling.

    Returns:
        np.ndarray of shape matching input, values in [0, 1].
    """
    if transform == "log":
        labels = np.log2(raw_labels + 1)
    else:
        labels = raw_labels.copy()
    lo, hi = labels.min(), labels.max()
    if hi - lo > 1e-8:
        labels = (labels - lo) / (hi - lo)
    return labels.astype(np.float32)
