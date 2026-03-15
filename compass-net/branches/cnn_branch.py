"""CNN branch — multi-scale Conv1d feature extraction from one-hot target DNA.

Architecturally identical to SeqCNN v1's convolutional layers so that
pre-trained v1 weights transfer directly (see training/transfer_weights.py).

v1 architecture:
    MultiScaleConvBlock(4, 40) -> 120 channels
    DilatedConvBlock(120)      -> 120 channels (residual)
    Conv1d(120, 96, k=1)       -> 96 channels

This branch preserves the same layer structure but parameterises the
output dimension for flexibility in Compass-ML fusion.

Gap 7: Optional PAM embedding — for enAsCas12a with 9 PAM variants and
activity penalties (0.25-1.0), a dedicated PAM encoder provides explicit
PAM-type information that is otherwise diluted across 34-position convolutions.

Input:  (batch, 4, 34) one-hot encoded target DNA
        (batch,) optional PAM class index (0-8 for enAsCas12a variants)
Output: (batch, 34, out_dim) per-position feature vectors

References:
    Kim et al., Nature Biotechnology 36:239-241 (2018). PMID: 29431740.
    Kleinstiver et al., Nature Biotechnology 37:276-282 (2019) — enAsCas12a PAMs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNBranch(nn.Module):
    """Multi-scale Conv1d feature extraction from one-hot DNA.

    Three parallel branches (k=3, k=5, k=7) capture motifs at different
    scales. Dilated convolutions expand the receptive field with residual
    connections. A 1x1 conv reduces to the desired output dimension.

    Default hyperparameters match SeqCNN v1 exactly (branches=40, reduced=96)
    for weight transfer compatibility.
    """

    def __init__(
        self,
        in_channels: int = 4,
        branches: int = 40,
        out_dim: int = 96,
        # Gap 7: explicit PAM encoding
        n_pam_classes: int = 0,
        pam_embed_dim: int = 8,
    ):
        super().__init__()
        concat_ch = branches * 3  # 120 for default

        # Multi-scale parallel convolutions
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branches, kernel_size=3, padding=1),
            nn.BatchNorm1d(branches),
            nn.GELU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, branches, kernel_size=5, padding=2),
            nn.BatchNorm1d(branches),
            nn.GELU(),
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, branches, kernel_size=7, padding=3),
            nn.BatchNorm1d(branches),
            nn.GELU(),
        )

        # Dilated convolutions with residual
        self.dilated1 = nn.Sequential(
            nn.Conv1d(concat_ch, concat_ch, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(concat_ch),
            nn.GELU(),
        )
        self.dilated2 = nn.Sequential(
            nn.Conv1d(concat_ch, concat_ch, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(concat_ch),
            nn.GELU(),
        )

        # Gap 7: PAM embedding
        self.n_pam_classes = n_pam_classes
        if n_pam_classes > 0:
            self.pam_embedding = nn.Embedding(n_pam_classes, pam_embed_dim)
            # Project PAM embedding to match CNN channels, broadcast to all positions
            self.pam_proj = nn.Linear(pam_embed_dim, concat_ch)
            # Note: PAM embedding is ADDED to conv features before reduction,
            # so it modulates all positions equally (PAM type affects global activity).

        # Channel reduction
        self.reduce = nn.Sequential(
            nn.Conv1d(concat_ch, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        pam_class: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, 4, seq_len) one-hot encoded target DNA.
            pam_class: (batch,) int tensor, PAM variant index 0..n_pam_classes-1.
                       None if PAM encoding is disabled.
        Returns:
            (batch, seq_len, out_dim) per-position features.
        """
        h = torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)
        h = h + self.dilated2(self.dilated1(h))  # residual around both dilated layers

        # Gap 7: Add PAM embedding (broadcast to all positions)
        if self.n_pam_classes > 0 and pam_class is not None:
            pam_emb = self.pam_embedding(pam_class)       # (batch, pam_embed_dim)
            pam_feat = self.pam_proj(pam_emb)             # (batch, concat_ch)
            h = h + pam_feat.unsqueeze(-1)                # broadcast to seq_len

        h = self.reduce(h)          # (batch, out_dim, seq_len)
        return h.permute(0, 2, 1)   # (batch, seq_len, out_dim)
