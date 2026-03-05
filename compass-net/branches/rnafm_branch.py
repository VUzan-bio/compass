"""RNA-FM branch — frozen foundation model embeddings for crRNA structure.

RNA-FM (Chen et al., 2022) is a 100M-parameter transformer pre-trained on
23M non-coding RNA sequences from RNAcentral. It produces 640-dim per-
nucleotide embeddings encoding secondary structure propensity, folding
stability, and base-stacking energetics.

Why RNA-FM and not a DNA model:
    Cas-FM (ICML 2025 Workshop) showed RNA-FM -> rho=0.76 vs DNABERT-2 -> rho=0.49.
    crRNAs are functional RNA molecules whose activity depends on RNA-level
    properties (folding, Cas12a loading) that RNA-FM captures.

CRITICAL: the crRNA spacer is 20 nt, but the CNN branch processes 34 nt
of target DNA. To fuse them, we align the 20-nt crRNA embeddings to
positions 4-23 of the 34-nt target (the protospacer region) and zero-pad
positions 0-3 (PAM) and 24-33 (flanking). This preserves positional
correspondence: crRNA position i <-> target protospacer position i.

Embedding extraction runs ONCE and is cached. Not run during training.

References:
    Chen et al., arXiv:2204.00300 (2022) — RNA-FM.
    Huang et al., ICML 2025 Workshop — Cas-FM benchmark.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RNAFMBranch(nn.Module):
    """Frozen RNA-FM embeddings -> learned projection -> per-position features.

    Only the projection layer is trainable (~82K params for default dims).
    RNA-FM itself is never loaded during training -- we use cached embeddings.

    The branch handles the 20-nt -> 34-nt alignment internally:
    input embeddings are (batch, 20, 640) for the crRNA spacer, and the
    output is (batch, 34, proj_dim) with zeros at PAM and flanking positions.
    """

    def __init__(
        self,
        embed_dim: int = 640,
        proj_dim: int = 64,
        dropout: float = 0.1,
        target_len: int = 34,
        spacer_start: int = 4,
        spacer_len: int = 20,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.target_len = target_len
        self.spacer_start = spacer_start
        self.spacer_len = spacer_len
        self.proj_dim = proj_dim

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project crRNA embeddings and align to 34-nt target positions.

        Args:
            embeddings: (batch, spacer_len, 640) pre-computed RNA-FM embeddings
                        for the 20-nt crRNA spacer.

        Returns:
            (batch, target_len, proj_dim) with crRNA features at positions
            spacer_start..spacer_start+spacer_len-1, zeros elsewhere.
        """
        batch_size = embeddings.size(0)
        projected = self.proj(embeddings)  # (batch, spacer_len, proj_dim)

        # Allocate full-length output with zeros at PAM and flanking positions
        out = torch.zeros(
            batch_size, self.target_len, self.proj_dim,
            device=embeddings.device, dtype=projected.dtype,
        )
        end = self.spacer_start + min(self.spacer_len, projected.size(1))
        out[:, self.spacer_start:end, :] = projected[:, :end - self.spacer_start, :]
        return out


def extract_rnafm_embeddings(
    sequences: list[str],
    model_path: str = "RNA-FM_pretrained.pth",
    batch_size: int = 64,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Extract RNA-FM embeddings for a list of RNA sequences.

    Run ONCE to populate the embedding cache. Not called during training.

    Args:
        sequences: list of RNA sequences (use U not T).
        model_path: path to RNA-FM checkpoint.
        batch_size: inference batch size.
        device: torch device.

    Returns:
        List of tensors, each (seq_len, 640).
    """
    # Lazy import — RNA-FM only needed for embedding extraction
    try:
        import fm  # RNA-FM package
    except ImportError:
        raise ImportError(
            "RNA-FM not installed. Install via: pip install rna-fm\n"
            "Or clone: https://github.com/ml4bio/RNA-FM"
        )

    import torch as _torch

    model, alphabet = fm.pretrained.rna_fm_t12(model_path)
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()

    all_embeddings: list[_torch.Tensor] = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        # RNA-FM expects (label, sequence) tuples
        data = [(f"seq_{i + j}", seq) for j, seq in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with _torch.no_grad():
            results = model(tokens, repr_layers=[12])
            # Shape: (batch, seq_len+2, 640) — includes BOS/EOS tokens
            emb = results["representations"][12]

        for j in range(len(batch_seqs)):
            # Strip BOS (position 0) and EOS (last position)
            seq_len = len(batch_seqs[j])
            all_embeddings.append(emb[j, 1:seq_len + 1, :].cpu())

    return all_embeddings
