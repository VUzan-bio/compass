"""CNN branch — multi-scale Conv1d feature extraction from one-hot target DNA.

Architecturally identical to SeqCNN v1's convolutional layers so that
pre-trained v1 weights transfer directly (see training/transfer_weights.py).

v1 architecture:
    MultiScaleConvBlock(4, 40) -> 120 channels
    DilatedConvBlock(120)      -> 120 channels (residual)
    Conv1d(120, 96, k=1)       -> 96 channels

This branch preserves the same layer structure but parameterises the
output dimension for flexibility in Compass-ML fusion.

Input:  (batch, 4, 34) one-hot encoded target DNA
Output: (batch, 34, out_dim) per-position feature vectors

References:
    Kim et al., Nature Biotechnology 36:239-241 (2018). PMID: 29431740.
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

        # Channel reduction
        self.reduce = nn.Sequential(
            nn.Conv1d(concat_ch, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 4, seq_len) one-hot encoded target DNA.
        Returns:
            (batch, seq_len, out_dim) per-position features.
        """
        h = torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)
        h = h + self.dilated2(self.dilated1(h))  # residual around both dilated layers
        h = self.reduce(h)          # (batch, out_dim, seq_len)
        return h.permute(0, 2, 1)   # (batch, seq_len, out_dim)
