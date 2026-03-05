"""Multi-dataset loader with domain-adversarial training support.

Harmonises multiple Cas12a activity datasets into a common format:
- Target DNA: standardised to 34-nt encoding (4 PAM + 20 spacer + 10 flanking)
- Activity: quantile-normalised to [0, 1] per dataset (preserves rank order)
- Domain ID: integer identifying the source dataset
- Variant ID: integer identifying the Cas12a variant

Quantile normalisation rationale:
    Kim 2018 measures indel frequency (0-100%).
    EasyDesign measures fluorescence (arbitrary units).
    Chen 2025 measures editing percentage (0-100%).
    These scales are incomparable. Quantile normalisation maps each dataset's
    activity distribution to a uniform [0,1] range, preserving rank order
    within each dataset while making magnitudes comparable.

Reference:
    Bolstad et al. "A comparison of normalization methods for high density
    oligonucleotide array data" Bioinformatics 2003, 19(2):185-193.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.stats import rankdata


@dataclass
class DatasetMeta:
    """Metadata for one source dataset."""
    name: str
    domain_id: int
    variant: str
    readout_type: str
    seq_format: str
    cell_context: str
    n_samples: int = 0


def quantile_normalise(values: np.ndarray) -> np.ndarray:
    """Map values to [0, 1] preserving rank order. Ties get averaged ranks."""
    if len(values) <= 1:
        return np.array([0.5] * len(values), dtype=np.float32)
    ranks = rankdata(values, method="average")
    return ((ranks - 1) / (len(ranks) - 1)).astype(np.float32)


def standardise_target_sequence(
    target_seq: str,
    source_format: str,
    target_length: int = 34,
) -> str | None:
    """Standardise target sequences to 34-nt encoding.

    Returns None if the sequence can't be standardised (no PAM found, too short).
    """
    seq = target_seq.upper().strip()

    if source_format == "34bp":
        if len(seq) >= target_length:
            return seq[:target_length]
        return None

    if source_format == "45bp":
        pam_pos = _find_tttv_pam(seq)
        if pam_pos is not None:
            start = pam_pos
            end = start + target_length
            if end <= len(seq):
                return seq[start:end]
            # Pad with N if flanking is too short
            fragment = seq[start:]
            return fragment.ljust(target_length, "N")[:target_length]
        # Fallback: assume PAM starts at position 0
        if len(seq) >= target_length:
            return seq[:target_length]
        return None

    # Generic format
    pam_pos = _find_tttv_pam(seq)
    if pam_pos is not None:
        start = pam_pos
        fragment = seq[start:]
        if len(fragment) >= target_length:
            return fragment[:target_length]
        return fragment.ljust(target_length, "N")[:target_length]

    if len(seq) >= target_length:
        return seq[:target_length]
    return None


def _find_tttv_pam(seq: str) -> int | None:
    """Find TTTV PAM in sequence (Cas12a canonical PAM)."""
    for i in range(len(seq) - 3):
        if seq[i:i+3] == "TTT" and seq[i+3] in "ACG":
            return i
    return None


_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


class MultiDatasetLoader(Dataset):
    """Unified dataset combining multiple Cas12a activity datasets.

    Each sample:
        target_onehot: (4, 34) standardised target DNA
        crrna_spacer:  str (20-nt RNA, for embedding lookup or live LoRA)
        activity:      float [0, 1] quantile-normalised
        domain_id:     int (source dataset)
        variant_id:    int (Cas12a variant)
    """

    def __init__(self, datasets: list[dict]):
        """
        Args:
            datasets: list of dicts, each with:
                "metadata": DatasetMeta
                "sequences": list[str]  (raw target DNA)
                "activities": list[float] (raw activity)
        """
        self.samples: list[dict] = []
        self.n_domains = len(datasets)
        self.variant_map: dict[str, int] = {}
        self.domain_ids: list[int] = []

        for ds in datasets:
            meta = ds["metadata"]
            seqs = ds["sequences"]
            acts = np.array(ds["activities"], dtype=np.float64)
            acts_norm = quantile_normalise(acts)

            if meta.variant not in self.variant_map:
                self.variant_map[meta.variant] = len(self.variant_map)
            variant_id = self.variant_map[meta.variant]

            n_added = 0
            for seq, act in zip(seqs, acts_norm):
                std_seq = standardise_target_sequence(seq, meta.seq_format)
                if std_seq is None:
                    continue

                protospacer = std_seq[4:24]
                crrna = "".join(_COMPLEMENT.get(b, "N") for b in reversed(protospacer))
                crrna = crrna.replace("T", "U")

                self.samples.append({
                    "target_seq": std_seq,
                    "crrna_spacer": crrna,
                    "activity": float(act),
                    "domain_id": meta.domain_id,
                    "variant_id": variant_id,
                })
                self.domain_ids.append(meta.domain_id)
                n_added += 1

            meta.n_samples = n_added

        print(f"MultiDatasetLoader: {len(self.samples)} samples, "
              f"{self.n_domains} domains, {len(self.variant_map)} variants")
        for ds in datasets:
            m = ds["metadata"]
            print(f"  [{m.domain_id}] {m.name}: {m.n_samples} samples "
                  f"({m.variant}, {m.readout_type})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "target_onehot": _one_hot(s["target_seq"]),
            "crrna_spacer": s["crrna_spacer"],
            "activity": torch.tensor(s["activity"], dtype=torch.float32),
            "domain_id": torch.tensor(s["domain_id"], dtype=torch.long),
            "variant_id": torch.tensor(s["variant_id"], dtype=torch.long),
        }


def collate_multi(batch: list[dict]) -> dict:
    """Collate function for MultiDatasetLoader."""
    return {
        "target_onehot": torch.stack([b["target_onehot"] for b in batch]),
        "crrna_spacers": [b["crrna_spacer"] for b in batch],
        "activity": torch.stack([b["activity"] for b in batch]),
        "domain_id": torch.stack([b["domain_id"] for b in batch]),
        "variant_id": torch.stack([b["variant_id"] for b in batch]),
    }


def _one_hot(seq: str, max_len: int = 34) -> torch.Tensor:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    mat = torch.zeros(4, max_len)
    for i, nt in enumerate(seq[:max_len].upper()):
        idx = mapping.get(nt)
        if idx is not None:
            mat[idx, i] = 1.0
    return mat
