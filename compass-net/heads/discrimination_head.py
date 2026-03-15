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
  - rloop_bias:   Reuse RLPA R-loop positional prior for position-aware discrimination
  - cross_attention: Cross-attention between MUT and WT per-position features
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DiscriminationCrossAttention(nn.Module):
    """Cross-attention between MUT and WT per-position features.

    Learns position-specific interactions: e.g., "mismatch at position 5
    matters more when seed GC is high." Standard in paired-sequence models
    (AlphaFold-Multimer uses similar paired attention).

    The MUT features attend to WT features and vice versa, producing
    a difference-aware representation before pooling.
    """

    def __init__(self, d_model: int = 128, d_k: int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.scale = math.sqrt(d_k)

        # MUT attends to WT
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
        self.W_o = nn.Linear(d_k, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        mut_feat: torch.Tensor,
        wt_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attend MUT to WT, return difference-weighted features.

        Args:
            mut_feat: (batch, seq_len, d_model) MUT per-position features.
            wt_feat:  (batch, seq_len, d_model) WT per-position features.

        Returns:
            (batch, seq_len, d_model) cross-attended difference features.
        """
        x = self.norm(mut_feat)
        wt_normed = self.norm(wt_feat)

        Q = self.W_q(x)            # MUT queries
        K = self.W_k(wt_normed)    # WT keys
        V = self.W_v(wt_normed)    # WT values

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        cross_out = self.W_o(torch.matmul(attn, V))

        # Return difference: how MUT differs from what WT attention suggests
        return mut_feat - cross_out


class DiscriminationHead(nn.Module):
    """Predict discrimination ratio from paired encoder representations.

    Input: two pooled vectors (mut_pooled, wt_pooled), each of size input_dim.
    Base features: [mut, wt, mut-wt, mut*wt] = 4x input_dim.
    Optional: + n_thermo thermo features + pos_embed_dim position embedding.
    Optional: + rloop_pos_dim R-loop positional prior.
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
        # Gap 3: R-loop positional prior for discrimination
        rloop_pos_dim: int = 0,
        # Gap 4: cross-attention between MUT/WT (pre-pooling)
        use_cross_attention: bool = False,
        cross_attn_d_model: int = 128,
    ):
        super().__init__()
        self.n_thermo = n_thermo
        self.pos_embed_dim = pos_embed_dim
        self.rloop_pos_dim = rloop_pos_dim
        self.use_cross_attention = use_cross_attention

        # Position embedding (Enhancement C — flat)
        if pos_embed_dim > 0:
            self.pos_embedding = nn.Embedding(n_positions + 1, pos_embed_dim)
            self.pos_dropout = nn.Dropout(0.3)

        # Gap 3: R-loop positional prior for discrimination
        # Reuses the biophysical insight from RLPA: seed positions dominate
        # discrimination. Learns a position-dependent scaling of the
        # contrastive signal, initialised from Cas12a sensitivity profiles.
        if rloop_pos_dim > 0:
            self.rloop_pos_proj = nn.Sequential(
                nn.Embedding(n_positions + 1, rloop_pos_dim),
                nn.Linear(rloop_pos_dim, rloop_pos_dim),
                nn.GELU(),
            )

        # Gap 4: cross-attention between MUT and WT per-position features
        if use_cross_attention:
            self.cross_attn = DiscriminationCrossAttention(
                d_model=cross_attn_d_model, d_k=32, dropout=0.1,
            )
            self.cross_pool = nn.AdaptiveAvgPool1d(1)

        # Total input dim: base + thermo + position + rloop + cross_attn
        total_input = input_dim * 4 + n_thermo + pos_embed_dim + rloop_pos_dim
        if use_cross_attention:
            total_input += cross_attn_d_model  # pooled cross-attention output

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
        # Gap 4: per-position features for cross-attention
        mut_feat_seq: torch.Tensor | None = None,
        wt_feat_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            mut_pooled:   (batch, input_dim) encoder output for mutant target.
            wt_pooled:    (batch, input_dim) encoder output for wildtype target.
            thermo_feats: (batch, n_thermo) thermodynamic features, or None.
            mm_position:  (batch,) int tensor, PAM-relative position 1-24.
            mut_feat_seq: (batch, 34, d_model) MUT per-position features (for cross-attn).
            wt_feat_seq:  (batch, 34, d_model) WT per-position features (for cross-attn).

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

        # Append position embedding (Enhancement C — flat)
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

        # Gap 3: R-loop positional prior
        if self.rloop_pos_dim > 0:
            if mm_position is not None:
                rloop_emb = self.rloop_pos_proj(mm_position.clamp(0, 24))
                combined = torch.cat([combined, rloop_emb], dim=-1)
            else:
                combined = torch.cat([
                    combined,
                    torch.zeros(mut_pooled.size(0), self.rloop_pos_dim, device=mut_pooled.device),
                ], dim=-1)

        # Gap 4: cross-attention between MUT and WT per-position features
        if self.use_cross_attention and mut_feat_seq is not None and wt_feat_seq is not None:
            cross_diff = self.cross_attn(mut_feat_seq, wt_feat_seq)
            # Pool the cross-attention output: (batch, 34, d_model) -> (batch, d_model)
            cross_pooled = self.cross_pool(
                cross_diff.permute(0, 2, 1)
            ).squeeze(-1)
            combined = torch.cat([combined, cross_pooled], dim=-1)
        elif self.use_cross_attention:
            # Fallback: zero-pad if per-position features not provided
            d = self.cross_attn.W_o.out_features
            combined = torch.cat([
                combined,
                torch.zeros(mut_pooled.size(0), d, device=mut_pooled.device),
            ], dim=-1)

        return self.head(combined)
