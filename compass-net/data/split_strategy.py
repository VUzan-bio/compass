"""Sequence-identity-based data splitting to prevent leakage.

Random train/val/test splits allow near-identical guides to appear in both
training and test sets, inflating reported performance. This module clusters
guides by sequence identity and splits at the cluster level.

Two backends:
    1. CD-HIT-EST (preferred): external tool, handles large datasets efficiently.
       Install: conda install -c bioconda cd-hit
    2. Pure Python fallback: pairwise Hamming distance (O(n^2), fine for <20K seqs).

References:
    Wessels et al., Nature Biotechnology 2024 -- showed sequence leakage
    inflates CRISPR model performance.
    Graph-CRISPR (Jiang et al., Briefings in Bioinformatics 2025) -- found
    2.2-2.5% redundancy even after filtering.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def cluster_sequences(
    sequences: list[str],
    identity_threshold: float = 0.80,
    word_size: int = 5,
) -> dict[int, list[int]]:
    """Cluster sequences by identity.

    Tries CD-HIT-EST first; falls back to Hamming-based clustering.

    Args:
        sequences: list of DNA sequences (same length preferred).
        identity_threshold: minimum identity to be in the same cluster.
        word_size: CD-HIT word size parameter (-n).

    Returns:
        {cluster_id: [sequence_indices]}
    """
    try:
        return _cluster_cdhit(sequences, identity_threshold, word_size)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return _cluster_hamming(sequences, identity_threshold)


def _cluster_cdhit(
    sequences: list[str],
    identity_threshold: float,
    word_size: int,
) -> dict[int, list[int]]:
    """Cluster using CD-HIT-EST (external tool)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False,
    ) as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")
        fasta_path = f.name

    output_path = fasta_path + ".cdhit"

    subprocess.run(
        [
            "cd-hit-est",
            "-i", fasta_path,
            "-o", output_path,
            "-c", str(identity_threshold),
            "-n", str(word_size),
            "-d", "0",
            "-M", "0",
        ],
        check=True,
        capture_output=True,
    )

    clusters: dict[int, list[int]] = defaultdict(list)
    cluster_id = -1
    with open(output_path + ".clstr") as f:
        for line in f:
            if line.startswith(">Cluster"):
                cluster_id = int(line.strip().split()[-1])
            elif ">" in line:
                seq_name = line.split(">")[1].split("...")[0]
                seq_idx = int(seq_name.replace("seq_", ""))
                clusters[cluster_id].append(seq_idx)

    # Cleanup temp files
    for p in [fasta_path, output_path, output_path + ".clstr"]:
        Path(p).unlink(missing_ok=True)

    return dict(clusters)


def _cluster_hamming(
    sequences: list[str],
    identity_threshold: float,
) -> dict[int, list[int]]:
    """Pure Python fallback: greedy clustering by Hamming distance.

    O(n^2) but fine for <20K sequences of length 34.
    """
    n = len(sequences)
    seq_len = len(sequences[0]) if sequences else 0
    min_matches = int(identity_threshold * seq_len)

    assigned = [-1] * n
    cluster_id = 0
    clusters: dict[int, list[int]] = {}

    for i in range(n):
        if assigned[i] >= 0:
            continue
        # Start new cluster with sequence i as representative
        assigned[i] = cluster_id
        clusters[cluster_id] = [i]

        for j in range(i + 1, n):
            if assigned[j] >= 0:
                continue
            # Count matches
            matches = sum(
                1 for a, b in zip(sequences[i], sequences[j]) if a == b
            )
            if matches >= min_matches:
                assigned[j] = cluster_id
                clusters[cluster_id].append(j)

        cluster_id += 1

    return clusters


def clustered_split(
    sequences: list[str],
    identity_threshold: float = 0.80,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split data by sequence clusters to prevent leakage.

    No cluster appears in more than one split.

    Args:
        sequences: list of DNA sequences.
        identity_threshold: clustering identity threshold.
        train_frac: fraction of clusters for training.
        val_frac: fraction of clusters for validation.
        random_state: random seed for reproducibility.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    clusters = cluster_sequences(sequences, identity_threshold)
    cluster_ids = list(clusters.keys())

    rng = np.random.RandomState(random_state)
    rng.shuffle(cluster_ids)

    n_clusters = len(cluster_ids)
    n_train = int(n_clusters * train_frac)
    n_val = int(n_clusters * val_frac)

    train_clusters = cluster_ids[:n_train]
    val_clusters = cluster_ids[n_train:n_train + n_val]
    test_clusters = cluster_ids[n_train + n_val:]

    train_idx = [i for c in train_clusters for i in clusters[c]]
    val_idx = [i for c in val_clusters for i in clusters[c]]
    test_idx = [i for c in test_clusters for i in clusters[c]]

    return train_idx, val_idx, test_idx


def kfold_clustered_splits(
    sequences: list[str],
    n_folds: int = 5,
    identity_threshold: float = 0.80,
    random_state: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """K-fold cross-validation with sequence-clustered splits.

    Each fold: train on (K-1) cluster groups, validate on 1.
    No cluster appears in multiple folds.

    Args:
        sequences: list of DNA sequences.
        n_folds: number of CV folds.
        identity_threshold: clustering identity threshold.
        random_state: random seed for reproducibility.

    Returns:
        List of (train_indices, val_indices) tuples, one per fold.
    """
    clusters = cluster_sequences(sequences, identity_threshold)
    cluster_ids = list(clusters.keys())

    rng = np.random.RandomState(random_state)
    rng.shuffle(cluster_ids)

    fold_size = len(cluster_ids) // n_folds
    folds: list[tuple[list[int], list[int]]] = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(cluster_ids)
        val_cluster_ids = set(cluster_ids[start:end])
        train_cluster_ids = [c for c in cluster_ids if c not in val_cluster_ids]

        val_idx = [i for c in val_cluster_ids for i in clusters[c]]
        train_idx = [i for c in train_cluster_ids for i in clusters[c]]
        folds.append((train_idx, val_idx))

    return folds
