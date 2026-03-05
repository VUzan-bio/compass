"""R-Loop Propagation Attention (RLPA).

Novel contribution: biophysically-informed causal attention for
CRISPR-Cas12a guide activity prediction.

Biology: Cas12a R-loop formation is DIRECTIONAL. The enzyme initiates
strand invasion at the PAM-proximal end (position 1) and zips toward
the PAM-distal end (position 20+). A mismatch at position 3 prevents
the R-loop from ever reaching position 18 -- this is the kinetic basis
for seed-region dominance.

Standard self-attention treats positions symmetrically (3->18 = 18->3).
RLPA encodes directionality via a learnable bias matrix initialised
from Cas12a R-loop kinetics:

    Standard: Attention = softmax(QK^T / sqrt(d)) V
    RLPA:     Attention = softmax(QK^T / sqrt(d) + alpha * B_rloop) V

where B_rloop encodes:
    - Upstream (PAM-proximal) positions: strong positive bias
      (position i depends on all upstream positions the R-loop must traverse)
    - Downstream (PAM-distal) positions: soft negative bias
      (decaying relevance if R-loop stalls before reaching them)
    - alpha is a learnable scalar controlling bias strength
    - B_rloop is learnable -- initialised from biophysics, refined by data

References:
    Strohkendl et al., Mol Cell 2018 (PMID 30078724) -- Cas12a R-loop kinetics.
    Xiong et al., ICML 2020 -- Pre-norm transformer stability.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RLoopBias(nn.Module):
    """Learnable R-loop propagation bias matrix.

    Initialised from Cas12a kinetics (Strohkendl et al. 2018), then
    refined during training. The biophysical prior is a starting point,
    not a hard constraint -- the model can override it if the data
    warrants.

    Matrix layout for 34-nt input:
        Positions 0-3:   PAM (always relevant, strong positive bias)
        Positions 4-11:  Seed region (strongest directional dependency)
        Positions 12-23: Mid/distal spacer (moderate dependency)
        Positions 24-33: Flanking context (weak)
    """

    def __init__(self, max_len: int = 34, spacer_start: int = 4, seed_len: int = 8):
        super().__init__()

        bias = torch.zeros(max_len, max_len)
        seed_end = spacer_start + seed_len  # position 12

        for i in range(max_len):
            for j in range(max_len):
                if j <= i:
                    # j is upstream of or at i: R-loop propagated through j to reach i
                    distance = i - j
                    if j < spacer_start:
                        # PAM region: always relevant regardless of distance
                        bias[i, j] = 1.5
                    elif j < seed_end:
                        # Seed (positions 4-11): strongest dependency
                        # Exponential decay with distance, but high baseline
                        bias[i, j] = 2.0 * math.exp(-distance / 10.0)
                    else:
                        # Mid/distal spacer
                        bias[i, j] = 1.0 * math.exp(-distance / 8.0)
                else:
                    # j is downstream of i: soft penalty (not hard mask)
                    # R-loop may not reach j if it stalls at i
                    distance = j - i
                    bias[i, j] = -0.5 * distance

        self.bias = nn.Parameter(bias)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return bias matrix for the given sequence length."""
        return self.bias[:seq_len, :seq_len]


class RLoopAttention(nn.Module):
    """Single-head attention with R-loop propagation bias.

    Why single-head:
        - 34 positions, ~15K samples -> multi-head overfits
        - Single head = interpretable attention map showing R-loop dynamics
        - ~25K params total (Q/K/V projections + bias + FFN)

    Why pre-norm:
        - More stable with small datasets (Xiong et al., ICML 2020)
        - Gradient flow is better in pre-norm configuration
    """

    def __init__(
        self,
        d_model: int = 128,
        d_k: int = 32,
        d_ff: int = 64,
        dropout: float = 0.1,
        max_len: int = 34,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.scale = math.sqrt(d_k)

        # Bottleneck Q/K/V: d_model -> d_k (saves ~90% params vs full-rank)
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
        self.W_o = nn.Linear(d_k, d_model, bias=False)  # project back

        # R-loop bias: learnable, initialised from biophysics
        self.rloop_bias = RLoopBias(max_len=max_len)
        self.bias_scale = nn.Parameter(torch.tensor(1.0))

        # Pre-norm + compact FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model) -- same shape as input
            attn_weights: (batch, seq_len, seq_len) -- for visualisation
        """
        seq_len = x.size(1)
        x_norm = self.norm1(x)

        Q = self.W_q(x_norm)  # (batch, seq_len, d_k)
        K = self.W_k(x_norm)  # (batch, seq_len, d_k)
        V = self.W_v(x_norm)  # (batch, seq_len, d_k)

        # Scaled dot-product + R-loop propagation bias
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        bias = self.rloop_bias(seq_len)
        scores = scores + self.bias_scale * bias.unsqueeze(0)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = self.W_o(self.attn_dropout(torch.matmul(attn_weights, V)))

        # Residual + FFN with pre-norm
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        return x, attn_weights
