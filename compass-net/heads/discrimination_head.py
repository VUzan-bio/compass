"""Discrimination head -- predict MUT/WT activity ratio.

Computes the discrimination ratio D = A_MUT / A_WT from the contrastive
difference between mutant-target and wildtype-target pooled representations.

The head sees [mut, wt, mut-wt, mut*wt]. The difference and interaction
terms force the shared encoder to learn features sensitive to single-
nucleotide changes in the TARGET DNA. This is exactly what diagnostic
SNP detection requires.

Biology: the crRNA guide is FIXED. The encoder sees the same crRNA
(via RNA-FM) for both mutant and wildtype. Only the CNN features differ
because the target DNA differs by one nucleotide at the SNP position.

Enhanced features (optional, added incrementally):
  - thermo_feats: (batch, n_thermo) thermodynamic features (ddG, cumulative dG, local dG)
  - mm_position:  (batch,) PAM-relative mismatch position (1-24) -> learnable embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiscriminationHead(nn.Module):
    """Predict discrimination ratio from paired encoder representations.

    Input: two pooled vectors (mut_pooled, wt_pooled), each of size input_dim.
    Base features: [mut, wt, mut-wt, mut*wt] = 4x input_dim.
    Optional: + n_thermo thermo features + pos_embed_dim position embedding.
    Output: scalar > 0 (Softplus ensures positivity).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        n_thermo: int = 0,
        n_positions: int = 24,
        pos_embed_dim: int = 0,
    ):
        super().__init__()
        self.n_thermo = n_thermo
        self.pos_embed_dim = pos_embed_dim

        # Position embedding (Enhancement C)
        if pos_embed_dim > 0:
            self.pos_embedding = nn.Embedding(n_positions + 1, pos_embed_dim)
            # +1 for position 0 (unknown/padding)
            self.pos_dropout = nn.Dropout(0.3)

        # Total input dim: base + thermo + position
        total_input = input_dim * 4 + n_thermo + pos_embed_dim

        self.head = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(32, 1),
            nn.Softplus(),  # disc ratio > 0 always
        )

    def forward(
        self,
        mut_pooled: torch.Tensor,
        wt_pooled: torch.Tensor,
        thermo_feats: torch.Tensor | None = None,
        mm_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            mut_pooled:  (batch, input_dim) encoder output for mutant target.
            wt_pooled:   (batch, input_dim) encoder output for wildtype target.
            thermo_feats: (batch, n_thermo) thermodynamic features, or None.
            mm_position: (batch,) int tensor, PAM-relative position 1-24, or 0 if unknown.

        Returns:
            (batch, 1) predicted discrimination ratio.
        """
        combined = torch.cat([
            mut_pooled,
            wt_pooled,
            mut_pooled - wt_pooled,
            mut_pooled * wt_pooled,
        ], dim=-1)

        # Append thermo features (Enhancement B)
        if self.n_thermo > 0:
            if thermo_feats is not None:
                combined = torch.cat([combined, thermo_feats], dim=-1)
            else:
                combined = torch.cat([
                    combined,
                    torch.zeros(mut_pooled.size(0), self.n_thermo, device=mut_pooled.device),
                ], dim=-1)

        # Append position embedding (Enhancement C)
        if self.pos_embed_dim > 0:
            if mm_position is not None:
                pos_emb = self.pos_embedding(mm_position.clamp(0, 24))
                pos_emb = self.pos_dropout(pos_emb)
                combined = torch.cat([combined, pos_emb], dim=-1)
            else:
                combined = torch.cat([
                    combined,
                    torch.zeros(mut_pooled.size(0), self.pos_embed_dim, device=mut_pooled.device),
                ], dim=-1)

        return self.head(combined)
