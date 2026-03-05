"""SeqCNN — Cas12a guide activity prediction via convolutional neural network.

Multi-scale parallel convolutions capture motifs at different scales:
  - Kernel 3: dinucleotide stacking energies
  - Kernel 5: seed-region patterns (half of 8-nt seed)
  - Kernel 7: broader context (secondary structure propensity)

Dilated convolutions expand receptive field without pooling away
positional information. Adaptive pooling handles variable-length input.

Architecture:
    Input (batch, 4, L) one-hot DNA
    -> MultiScaleConvBlock (parallel k=3,5,7) -> 120 channels
    -> DilatedConvBlock (k=3 d=1, k=3 d=2 + residual) -> 120 channels
    -> Channel reduction Conv1d(120 -> 96)
    -> AdaptiveAvgPool1d(1) -> 96-dim vector
    -> Dense(96 -> 64) -> GELU -> Dropout(0.3)
    -> Dense(64 -> 32) -> GELU -> Dropout(0.2)
    -> Dense(32 -> 1) -> Sigmoid

~110K parameters. Trainable on CPU in minutes.

References:
    Kim et al., Nature Biotechnology 36:239-241 (2018). PMID: 29431740.
    Huang et al., iMeta 3(4):e214 (2024). PMID: 39135699.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiScaleConvBlock(nn.Module):
    """Parallel convolutions with kernel sizes 3, 5, 7.

    Each branch: Conv1d -> BatchNorm -> GELU.
    Outputs concatenated along channel dimension.
    """

    def __init__(self, in_channels: int, out_per_branch: int = 64):
        super().__init__()
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_per_branch),
            nn.GELU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_per_branch),
            nn.GELU(),
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_per_branch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)


class DilatedConvBlock(nn.Module):
    """Two dilated convolutions with residual connection.

    Dilation expands the receptive field without pooling:
      - Conv(k=3, d=1): receptive field = 3
      - Conv(k=3, d=2): effective receptive field = 5
      - Combined with residual: receptive field = 7
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.conv2(h)
        return h + x  # residual


class SeqCNN(nn.Module):
    """COMPASS seq_cnn: Cas12a guide activity prediction from sequence.

    Args:
        in_channels: Input channels (4 for one-hot DNA).
        branches: Channels per multi-scale branch. Total = 3 * branches.
        dilated_channels: Must equal 3 * branches.
        reduced_channels: Channels after 1x1 reduction conv.
        fc1: First dense layer size.
        fc2: Second dense layer size.
        dropout1: Dropout after fc1.
        dropout2: Dropout after fc2.
    """

    def __init__(
        self,
        in_channels: int = 4,
        branches: int = 40,
        dilated_channels: int = 120,
        reduced_channels: int = 96,
        fc1: int = 64,
        fc2: int = 32,
        dropout1: float = 0.3,
        dropout2: float = 0.2,
    ):
        super().__init__()

        self.multi_scale = MultiScaleConvBlock(in_channels, branches)
        self.dilated = DilatedConvBlock(dilated_channels)
        self.reduce = nn.Sequential(
            nn.Conv1d(dilated_channels, reduced_channels, kernel_size=1),
            nn.BatchNorm1d(reduced_channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(reduced_channels, fc1),
            nn.GELU(),
            nn.Dropout(dropout1),
            nn.Linear(fc1, fc2),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(fc2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, 4, L) one-hot encoded DNA sequence.

        Returns:
            (batch, 1) predicted activity score in [0, 1].
        """
        h = self.multi_scale(x)  # (batch, 120, L)
        h = self.dilated(h)      # (batch, 120, L)
        h = self.reduce(h)       # (batch, 96, L)
        h = self.pool(h)         # (batch, 96, 1)
        h = h.squeeze(-1)        # (batch, 96)
        return self.head(h)      # (batch, 1)

    def predict(self, x: torch.Tensor) -> float:
        """Single-sequence prediction (convenience method)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x.unsqueeze(0)).item()
