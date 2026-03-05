"""Kim et al. 2018 data loader for multi-domain training.

Training domain:
    Domain 0: HT 1-1 (15,000 guides)

Held-out splits (NEVER in training):
    Validation: HT 1-2 (1,292 guides) — same as single-dataset rows
    Test: HT 2 + HT 3 (4,214 guides) — same as single-dataset rows

Reference:
    Kim et al. "Genome-wide analysis reveals specificities of Cpf1
    endonucleases in human cells" Nature Biotechnology 2018, 36(9):863.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


def _load_sheet(xlsx_path: str, sheet_name: str) -> tuple[list[str], np.ndarray]:
    """Load one HT sheet, returning raw 34-nt sequences and raw indel frequencies."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=1)

    seq_col = None
    for c in df.columns:
        if "34 bp" in str(c) or "34bp" in str(c):
            seq_col = c
            break
    if seq_col is None:
        seq_col = df.columns[1]

    indel_col = None
    for c in df.columns:
        if "Background substracted" in str(c) or "Background subtracted" in str(c):
            indel_col = c
            break
    if indel_col is None:
        indel_col = df.columns[-1]

    valid = pd.DataFrame({"seq": df[seq_col], "indel": df[indel_col]}).dropna()
    sequences = valid["seq"].astype(str).values
    indels = valid["indel"].values.astype(np.float64)

    mask = np.array([
        len(s) == 34 and all(c in "ACGTacgt" for c in s)
        for s in sequences
    ])
    sequences = [s.upper() for s in sequences[mask]]
    indels = indels[mask]

    # Raw indel frequencies (not normalised — MultiDatasetLoader does quantile norm)
    indels = np.clip(indels, 0, None)

    return sequences, indels


def load_kim2018_domains(
    xlsx_path: str = "compass/data/kim2018/nbt4061_source_data.xlsx",
) -> dict:
    """Load Kim 2018 as multi-domain data.

    Returns dict with keys: "train_domains", "test_sequences", "test_activities"
    """
    if not os.path.isabs(xlsx_path):
        xlsx_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "..", xlsx_path,
        )

    # Training: HT 1-1 only (HT 1-2 is validation — must NOT leak into training)
    seqs_ht1_1, acts_ht1_1 = _load_sheet(xlsx_path, "Data set HT 1-1")

    # Validation (held out)
    seqs_ht1_2, acts_ht1_2 = _load_sheet(xlsx_path, "Data set HT 1-2")

    # Test (held out)
    seqs_ht2, acts_ht2 = _load_sheet(xlsx_path, "Data set HT 2")
    seqs_ht3, acts_ht3 = _load_sheet(xlsx_path, "Data set HT 3")

    return {
        "train_domains": [
            {
                "name": "Kim2018_HT1-1",
                "variant": "AsCas12a",
                "readout_type": "indel_pct",
                "cell_context": "HEK293T",
                "seq_format": "34bp",
                "sequences": seqs_ht1_1,
                "activities": acts_ht1_1.tolist(),
            },
        ],
        "val_sequences": seqs_ht1_2,
        "val_activities": np.clip(acts_ht1_2, 0, None).tolist(),
        "test_sequences": seqs_ht2 + seqs_ht3,
        "test_activities": np.clip(np.concatenate([acts_ht2, acts_ht3]), 0, None).tolist(),
    }
