"""Pre-compute thermodynamic features for Kim et al. 2018 dataset.

Saves a (N, 3) tensor: [folding_dg, hybrid_dg, melting_tm] per sequence,
plus normalization stats (mean, std) for each feature.

Usage:
    python compass-net/scripts/precompute_thermo.py
"""

from __future__ import annotations

import sys, os, json, logging
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMPASS_NET_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT_DIR = os.path.dirname(_COMPASS_NET_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _COMPASS_NET_DIR)

from run_phase1 import _setup, load_kim2018_sequences
from features.thermodynamic import compute_all_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    xlsx_path = "compass/data/kim2018/nbt4061_source_data.xlsx"
    (seqs_train, y_train), (seqs_val, y_val), (seqs_test, y_test) = load_kim2018_sequences(xlsx_path)
    all_seqs = seqs_train + seqs_val + seqs_test
    all_labels = np.concatenate([y_train, y_val, y_test])
    logger.info("Total sequences: %d", len(all_seqs))

    features = []
    for i, seq in enumerate(all_seqs):
        feat = compute_all_features(seq)
        features.append([feat["folding_dg"], feat["hybrid_dg"], feat["melting_tm"]])
        if (i + 1) % 5000 == 0 or i + 1 == len(all_seqs):
            logger.info("  Computed %d / %d", i + 1, len(all_seqs))

    features_array = np.array(features, dtype=np.float32)

    # Print statistics
    names = ["folding_dg", "hybrid_dg", "melting_tm"]
    logger.info("Feature statistics:")
    for j, name in enumerate(names):
        col = features_array[:, j]
        logger.info("  %-12s mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
                     name, col.mean(), col.std(), col.min(), col.max())

    # Correlation with activity
    from scipy.stats import spearmanr
    logger.info("Correlations with activity (Spearman rho):")
    for j, name in enumerate(names):
        rho, pval = spearmanr(features_array[:, j], all_labels)
        logger.info("  %-12s rho=%.4f  p=%.2e", name, rho, pval)

    # Normalize
    mean = features_array.mean(axis=0)
    std = features_array.std(axis=0)
    std[std < 1e-8] = 1.0
    features_norm = (features_array - mean) / std

    # Save
    out_path = os.path.join("E:/compass-net-data/cache", "thermo_features.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "features": torch.from_numpy(features_norm),
        "raw_features": torch.from_numpy(features_array),
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
        "feature_names": names,
        "n_train": len(seqs_train),
        "n_val": len(seqs_val),
        "n_test": len(seqs_test),
    }, out_path)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
