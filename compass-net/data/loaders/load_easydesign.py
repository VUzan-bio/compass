"""EasyDesign dataset loader (Huang et al., iMeta 2024).

11,992 Cas12a trans-cleavage diagnostic cases across multiple pathogens.
Training data: 10,634 samples (log-k fluorescence activity).
Test data: 1,358 samples (raw fluorescence values, 45-nt context).

This is the most relevant external dataset for COMPASS because it measures
TRANS-CLEAVAGE (the same readout as our electrochemical diagnostic),
not cis-cleavage indels like Kim 2018.

Source: https://github.com/scRNA-Compt/EasyDesign/data/Table S2.xlsx

Format:
    Training: 25-nt (4 PAM + 21 spacer/context), log-k activity
    Test: 45-nt DNA target, 25-nt crRNA, raw fluorescence

Standardisation to 34-nt: take first 24 nt (4 PAM + 20 spacer), pad 10 N's
for flanking context. This loses the 1 extra spacer nt and all flanking,
but preserves the PAM + protospacer — the most important region.

Reference:
    Huang et al. "EasyDesign: an efficient and comprehensive approach for
    CRISPR-Cas12a-based nucleic acid diagnostics" iMeta 2024.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


def load_easydesign(
    xlsx_path: str = "compass-net/data/external/easydesign/Table_S2.xlsx",
    use_augmented: bool = False,
) -> dict:
    """Load EasyDesign data for multi-domain training.

    Args:
        xlsx_path: path to Table S2.xlsx
        use_augmented: if True, include the 31,993 augmented samples

    Returns dict with keys: "sequences", "activities", "test_sequences", "test_activities"
    """
    if not os.path.isabs(xlsx_path):
        xlsx_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "..", xlsx_path,
        )

    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(
            f"EasyDesign data not found at {xlsx_path}. "
            f"Download from https://github.com/scRNA-Compt/EasyDesign/data/Table%20S2.xlsx"
        )

    # Training data
    df_train = pd.read_excel(xlsx_path, sheet_name="Training data")

    sequences = []
    activities = []
    for _, row in df_train.iterrows():
        guide = str(row["guide_seq"]).upper()
        if len(guide) < 24 or not guide.startswith("TTT"):
            continue

        # Standardise: take first 24 nt (PAM + 20 spacer), pad to 34
        seq_34 = guide[:24].ljust(34, "N")
        # Activity: 30-min fluorescence (log-k scale)
        act = float(row["30 min"])
        sequences.append(seq_34)
        activities.append(act)

    # Optionally add augmented data (same format)
    if use_augmented:
        df_aug = pd.read_excel(xlsx_path, sheet_name="Augment data")
        for _, row in df_aug.iterrows():
            guide = str(row["guide_seq"]).upper()
            if len(guide) < 24 or not guide.startswith("TTT"):
                continue
            seq_34 = guide[:24].ljust(34, "N")
            act = float(row["out_logk_measurement"])
            sequences.append(seq_34)
            activities.append(act)

    # Test data (45-nt context — extract 34 nt)
    df_test = pd.read_excel(xlsx_path, sheet_name="Test data")
    test_sequences = []
    test_activities = []
    for _, row in df_test.iterrows():
        dna = str(row["DNA"]).upper()
        if len(dna) < 34:
            continue

        # Find TTTV PAM in the 45-nt context
        pam_pos = _find_pam(dna)
        if pam_pos is not None:
            seq_34 = dna[pam_pos:pam_pos + 34]
            if len(seq_34) < 34:
                seq_34 = seq_34.ljust(34, "N")
        else:
            seq_34 = dna[:34]

        test_sequences.append(seq_34)
        test_activities.append(float(row["true value"]))

    return {
        "name": "EasyDesign",
        "variant": "LbCas12a",
        "readout_type": "fluorescence_logk",
        "cell_context": "in_vitro_diagnostic",
        "seq_format": "34bp",  # already standardised
        "sequences": sequences,
        "activities": activities,
        "test_sequences": test_sequences,
        "test_activities": test_activities,
    }


def _find_pam(seq: str) -> int | None:
    for i in range(len(seq) - 3):
        if seq[i:i+3] == "TTT" and seq[i+3] in "ACG":
            return i
    return None
