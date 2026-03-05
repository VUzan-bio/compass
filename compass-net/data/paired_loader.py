"""Paired (mutant_target, wildtype_target) data loader with FIXED crRNA guide.

Biology: the crRNA is designed to perfectly complement the resistant (mutant)
target DNA. In the diagnostic assay, the same crRNA is used against both:
    - Mutant target: perfect match -> high trans-cleavage signal
    - Wildtype target: one mismatch at the SNP -> reduced signal
    - Discrimination = activity(mutant) / activity(wildtype)

The GUIDE does not change. Only the TARGET DNA changes.

For the Kim et al. 2018 dataset: each entry is (target_DNA, measured_activity).
To simulate discrimination, we take each target, introduce a single mismatch
at seed positions (simulating the "other allele"), and create paired samples.
The crRNA spacer is derived as the reverse complement of the protospacer.

References:
    Chen et al., Science 2018 (PMID 29449511) -- Cas12a trans-cleavage.
    Strohkendl et al., Mol Cell 2018 (PMID 30078724) -- R-loop kinetics.
    Kim et al., Nature Biotechnology 2018 (PMID 29431740) -- HT-PAMDA data.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class PairedTargetDataset(Dataset):
    """Paired (mutant_target, wildtype_target, crRNA) dataset.

    Each sample:
        mutant_target_onehot:    (4, 34) target DNA with resistance SNP
        wildtype_target_onehot:  (4, 34) target DNA without SNP (one mismatch)
        crrna_spacer:            str, the crRNA spacer (RNA, for RNA-FM cache lookup)
        efficiency:              float, measured activity on mutant target
        mismatch_pos:            int, position of the SNP in the target

    Key insight: the crRNA (RNA-FM input) is IDENTICAL for mutant and wildtype.
    Only the target DNA (CNN input) changes. This correctly models the assay.
    """

    def __init__(
        self,
        target_sequences: list[str],
        activities: list[float],
        mismatch_positions: list[int] | None = None,
    ):
        self.pairs: list[dict] = []

        # Default: seed positions (4-11 in 34-nt encoding = positions 1-8 of spacer)
        # Seed mismatches cause the strongest activity reduction (Strohkendl 2018)
        if mismatch_positions is None:
            mismatch_positions = list(range(4, 12))

        for target_dna, activity in zip(target_sequences, activities):
            if len(target_dna) < 24:
                continue

            # crRNA spacer = reverse complement of protospacer (positions 4-23)
            # Convert to RNA (T -> U) for RNA-FM
            protospacer = target_dna[4:24]
            crrna_spacer = _reverse_complement(protospacer).replace("T", "U")

            for pos in mismatch_positions:
                if pos >= len(target_dna):
                    continue

                # Mutant target = original (perfect match to crRNA by design)
                # Wildtype target = one mismatch at the SNP position
                wildtype_target = _introduce_target_mismatch(target_dna, pos)

                self.pairs.append({
                    "mutant_target": target_dna,
                    "wildtype_target": wildtype_target,
                    "crrna_spacer": crrna_spacer,
                    "efficiency": activity,
                    "mismatch_pos": pos,
                })

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        p = self.pairs[idx]
        return {
            "mutant_target_onehot": _one_hot(p["mutant_target"]),
            "wildtype_target_onehot": _one_hot(p["wildtype_target"]),
            "crrna_spacer": p["crrna_spacer"],
            "efficiency": torch.tensor(p["efficiency"], dtype=torch.float32),
            "mismatch_pos": p["mismatch_pos"],
        }


class SingleTargetDataset(Dataset):
    """Standard single-target dataset for Phase 1 (efficiency only).

    Each sample: (target_onehot, crrna_spacer, efficiency).
    No paired wildtype -- used when multi-task is OFF.
    """

    def __init__(
        self,
        target_sequences: list[str],
        activities: list[float],
    ):
        self.targets = target_sequences
        self.activities = activities

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        target = self.targets[idx]
        protospacer = target[4:24] if len(target) >= 24 else target
        crrna_spacer = _reverse_complement(protospacer).replace("T", "U")

        return {
            "target_onehot": _one_hot(target),
            "crrna_spacer": crrna_spacer,
            "efficiency": torch.tensor(
                self.activities[idx], dtype=torch.float32,
            ),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _reverse_complement(seq: str) -> str:
    return "".join(_COMPLEMENT.get(b, b) for b in reversed(seq.upper()))


def _introduce_target_mismatch(target_dna: str, pos: int) -> str:
    """Change one nucleotide in the target DNA at the given position.

    Uses deterministic replacement (first alternative base) for reproducibility.
    """
    bases = "ACGT"
    original = target_dna[pos].upper()
    replacement = next(b for b in bases if b != original)
    return target_dna[:pos] + replacement + target_dna[pos + 1:]


def _one_hot(seq: str, max_len: int = 34) -> torch.Tensor:
    """One-hot encode DNA sequence. Channels-first (4, L) for Conv1d."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    mat = torch.zeros(4, max_len)
    for i, nt in enumerate(seq[:max_len].upper()):
        idx = mapping.get(nt)
        if idx is not None:
            mat[idx, i] = 1.0
    return mat
