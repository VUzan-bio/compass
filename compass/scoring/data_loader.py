"""Data loaders for CNN training datasets.

Loads Kim et al. 2018 (Nature Biotech) supplementary data and prepares
train/val/test splits following the original paper's protocol.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from compass.scoring.preprocessing import normalise_labels, one_hot_encode

logger = logging.getLogger(__name__)

# Column names in the Kim 2018 source data Excel file
_COL_SEQ34 = "34 bp synthetic target and target context sequence\n(4 bp + PAM + 23 bp protospacer + 3 bp)"
_COL_INDEL = "Indel freqeuncy\n(Background substracted, %)"  # sic — typo in original


def _load_sheet(
    xlsx_path: Path, sheet_name: str, max_len: int = 34,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one sheet from the Kim 2018 source data.

    Returns:
        X: (N, 4, max_len) one-hot encoded sequences
        y: (N,) normalised indel frequencies in [0, 1]
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=1)

    # Find the 34-nt sequence column
    seq_col = None
    for c in df.columns:
        if "34 bp" in str(c) or "34bp" in str(c):
            seq_col = c
            break
    if seq_col is None:
        # Fall back to second column (index 1)
        seq_col = df.columns[1]

    # Find the indel frequency column (background subtracted)
    indel_col = None
    for c in df.columns:
        if "Background substracted" in str(c) or "Background subtracted" in str(c):
            indel_col = c
            break
    if indel_col is None:
        # Fall back to last column
        indel_col = df.columns[-1]

    # Clean data
    sequences = df[seq_col].dropna().astype(str).values
    indels = df[indel_col].dropna().values.astype(np.float64)

    # Ensure same length (drop any NaN rows)
    valid = pd.DataFrame({"seq": df[seq_col], "indel": df[indel_col]}).dropna()
    sequences = valid["seq"].astype(str).values
    indels = valid["indel"].values.astype(np.float64)

    # Filter: valid DNA sequences only (34 nt, ACGT only)
    mask = np.array([
        len(s) == max_len and all(c in "ACGTacgt" for c in s)
        for s in sequences
    ])
    sequences = sequences[mask]
    indels = indels[mask]

    # Clip negative indel values to 0 (background subtraction artifacts)
    indels = np.clip(indels, 0, None)

    # Normalise
    y = normalise_labels(indels, transform="log")

    # One-hot encode
    X = np.stack([one_hot_encode(s, max_len) for s in sequences])

    return X, y


def load_kim2018(
    xlsx_path: str = "compass/data/kim2018/nbt4061_source_data.xlsx",
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],  # (X_train, y_train) — HT 1-1
    Tuple[np.ndarray, np.ndarray],  # (X_val, y_val)   — HT 1-2
    Tuple[np.ndarray, np.ndarray],  # (X_test, y_test)  — HT 2 + HT 3
]:
    """Load Kim et al. 2018 Cas12a activity dataset.

    Splits follow the original paper:
        - HT 1-1 (15,000 sequences): training
        - HT 1-2 (1,292 sequences): validation
        - HT 2 + HT 3 (4,214 sequences): held-out test

    Returns:
        Three tuples of (X, y) where X is (N, 4, 34) and y is (N,).
    """
    path = Path(xlsx_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Kim 2018 source data not found at {path}. "
            "Download from: https://www.nature.com/articles/nbt.4061 "
            "(Supplementary Source Data)"
        )

    logger.info("Loading HT 1-1 (training)...")
    X_train, y_train = _load_sheet(path, "Data set HT 1-1")
    logger.info("  %d sequences", len(X_train))

    logger.info("Loading HT 1-2 (validation)...")
    X_val, y_val = _load_sheet(path, "Data set HT 1-2")
    logger.info("  %d sequences", len(X_val))

    logger.info("Loading HT 2 (test set 1)...")
    X_test1, y_test1 = _load_sheet(path, "Data set HT 2")
    logger.info("  %d sequences", len(X_test1))

    logger.info("Loading HT 3 (test set 2)...")
    X_test2, y_test2 = _load_sheet(path, "Data set HT 3")
    logger.info("  %d sequences", len(X_test2))

    # Combine HT 2 + HT 3 as test
    X_test = np.concatenate([X_test1, X_test2], axis=0)
    y_test = np.concatenate([y_test1, y_test2], axis=0)

    logger.info(
        "Kim 2018 loaded: train=%d, val=%d, test=%d",
        len(X_train), len(X_val), len(X_test),
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
