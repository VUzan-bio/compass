"""RNA-FM with LoRA adapters for CRISPR-specific fine-tuning.

Loads RNA-FM weights from HuggingFace (multimolecule/rnafm) into a custom
pre-norm BERT implementation, then applies LoRA (rank 4) on Q and V
projections of the last 2 transformer layers.

This avoids the multimolecule package dependency (version conflicts) by
loading the safetensors weights directly into a minimal PyTorch model.

Why LoRA instead of full fine-tuning:
    - RNA-FM has 99M params. Full fine-tuning on 15K samples = catastrophic overfitting.
    - LoRA adds ~20K trainable params (rank 4 on Q,V in layers 10-11).
    - The base model stays frozen, preserving general RNA knowledge.

References:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" ICLR 2022.
    Chen et al., "Interpretable RNA Foundation Model from Unannotated Data" Nature Methods 2022.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# RNA-FM vocabulary (from multimolecule/rnafm tokenizer_config.json)
# Standard nucleotides + special tokens
RNAFM_VOCAB = {
    "<pad>": 0, "<cls>": 1, "<eos>": 2, "<unk>": 3, "<mask>": 4, "<null>": 5,
    "A": 6, "C": 7, "G": 8, "U": 9,
    "R": 10, "Y": 11, "S": 12, "W": 13, "K": 14, "M": 15,
    "B": 16, "D": 17, "H": 18, "V": 19, "N": 20,
    ".": 21, "*": 22, "-": 23, "+": 24, "X": 25,
}


def tokenize_rna(sequences: list[str], max_len: int = 22) -> torch.Tensor:
    """Tokenize RNA sequences for RNA-FM.

    Adds <cls> at start and <eos> at end, pads to max_len.
    For 20-nt spacers: max_len = 22 (20 + cls + eos).
    """
    batch = []
    for seq in sequences:
        ids = [RNAFM_VOCAB["<cls>"]]
        for ch in seq.upper():
            ids.append(RNAFM_VOCAB.get(ch, RNAFM_VOCAB["<unk>"]))
        ids.append(RNAFM_VOCAB["<eos>"])
        # Pad
        while len(ids) < max_len:
            ids.append(RNAFM_VOCAB["<pad>"])
        batch.append(ids[:max_len])
    return torch.tensor(batch, dtype=torch.long)


# ======================================================================
# LoRA layer
# ======================================================================

class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 4,
        alpha: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base = base_linear
        self.base.requires_grad_(False)  # freeze base

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * (1.0 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return base_out + lora_out


# ======================================================================
# Minimal RNA-FM architecture (pre-norm BERT variant)
# ======================================================================

class RNAFMAttention(nn.Module):
    def __init__(self, hidden_size: int = 640, num_heads: int = 20, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output_dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)

        B, L, _ = x.shape
        q = self.query(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask[:, None, None, :] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, -1)
        out = self.output_dense(out)
        out = self.dropout(out)
        return residual + out


class RNAFMLayer(nn.Module):
    def __init__(self, hidden_size: int = 640, intermediate_size: int = 5120, dropout: float = 0.1):
        super().__init__()
        self.attention = RNAFMAttention(hidden_size, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention(x, pad_mask)
        residual = x
        x = self.layer_norm(x)
        x = self.intermediate(x)
        x = F.gelu(x)
        x = self.output_dense(x)
        x = self.dropout(x)
        return residual + x


class RNAFMEncoder(nn.Module):
    def __init__(self, num_layers: int = 12, hidden_size: int = 640, intermediate_size: int = 5120):
        super().__init__()
        self.word_embeddings = nn.Embedding(26, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(1026, hidden_size)
        self.embed_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        self.layers = nn.ModuleList([
            RNAFMLayer(hidden_size, intermediate_size) for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_ids: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.word_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.embed_layer_norm(x)

        for layer in self.layers:
            x = layer(x, pad_mask)

        x = self.final_layer_norm(x)
        return x

    def load_from_safetensors(self, state_dict: dict[str, torch.Tensor]) -> int:
        """Load RNA-FM weights from HuggingFace safetensors format.

        Returns number of keys loaded.
        """
        key_map = {
            "model.embeddings.word_embeddings.weight": "word_embeddings.weight",
            "model.embeddings.position_embeddings.weight": "position_embeddings.weight",
            "model.embeddings.layer_norm.weight": "embed_layer_norm.weight",
            "model.embeddings.layer_norm.bias": "embed_layer_norm.bias",
            "model.encoder.layer_norm.weight": "final_layer_norm.weight",
            "model.encoder.layer_norm.bias": "final_layer_norm.bias",
        }

        for i in range(12):
            prefix_src = f"model.encoder.layer.{i}"
            prefix_dst = f"layers.{i}"
            layer_map = {
                f"{prefix_src}.attention.self.query.weight": f"{prefix_dst}.attention.query.weight",
                f"{prefix_src}.attention.self.query.bias": f"{prefix_dst}.attention.query.bias",
                f"{prefix_src}.attention.self.key.weight": f"{prefix_dst}.attention.key.weight",
                f"{prefix_src}.attention.self.key.bias": f"{prefix_dst}.attention.key.bias",
                f"{prefix_src}.attention.self.value.weight": f"{prefix_dst}.attention.value.weight",
                f"{prefix_src}.attention.self.value.bias": f"{prefix_dst}.attention.value.bias",
                f"{prefix_src}.attention.output.dense.weight": f"{prefix_dst}.attention.output_dense.weight",
                f"{prefix_src}.attention.output.dense.bias": f"{prefix_dst}.attention.output_dense.bias",
                f"{prefix_src}.attention.layer_norm.weight": f"{prefix_dst}.attention.layer_norm.weight",
                f"{prefix_src}.attention.layer_norm.bias": f"{prefix_dst}.attention.layer_norm.bias",
                f"{prefix_src}.intermediate.dense.weight": f"{prefix_dst}.intermediate.weight",
                f"{prefix_src}.intermediate.dense.bias": f"{prefix_dst}.intermediate.bias",
                f"{prefix_src}.output.dense.weight": f"{prefix_dst}.output_dense.weight",
                f"{prefix_src}.output.dense.bias": f"{prefix_dst}.output_dense.bias",
                f"{prefix_src}.layer_norm.weight": f"{prefix_dst}.layer_norm.weight",
                f"{prefix_src}.layer_norm.bias": f"{prefix_dst}.layer_norm.bias",
            }
            key_map.update(layer_map)

        my_state = self.state_dict()
        loaded = 0
        for src_key, dst_key in key_map.items():
            if src_key in state_dict and dst_key in my_state:
                if state_dict[src_key].shape == my_state[dst_key].shape:
                    my_state[dst_key] = state_dict[src_key]
                    loaded += 1

        self.load_state_dict(my_state)
        return loaded


# ======================================================================
# Full branch: RNA-FM + LoRA + projection
# ======================================================================

class RNAFMLoRABranch(nn.Module):
    """Live RNA-FM with LoRA adapters + projection to Compass-ML feature space.

    Input: list of RNA strings (crRNA spacers, 20-nt)
    Output: (batch, 34, proj_dim) per-position features aligned to target DNA

    LoRA applied to Q and V projections of layers 10-11 only.
    """

    def __init__(
        self,
        proj_dim: int = 64,
        lora_rank: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        lora_target_layers: tuple[int, ...] = (10, 11),
        target_len: int = 34,
        spacer_start: int = 4,
        spacer_len: int = 20,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.target_len = target_len
        self.spacer_start = spacer_start
        self.spacer_len = spacer_len

        # Build RNA-FM encoder
        self.encoder = RNAFMEncoder()

        # Load weights from HuggingFace
        self._load_pretrained_weights()

        # Freeze everything in the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Apply LoRA to Q and V in target layers
        self.lora_target_layers = lora_target_layers
        for layer_idx in lora_target_layers:
            layer = self.encoder.layers[layer_idx]
            layer.attention.query = LoRALinear(
                layer.attention.query, rank=lora_rank,
                alpha=lora_alpha, dropout=lora_dropout,
            )
            layer.attention.value = LoRALinear(
                layer.attention.value, rank=lora_rank,
                alpha=lora_alpha, dropout=lora_dropout,
            )

        # Projection: 640 -> proj_dim (same as frozen branch)
        self.proj = nn.Sequential(
            nn.Linear(640, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "RNA-FM LoRA: %d trainable / %d total (%.2f%%)",
            trainable, total, 100 * trainable / total,
        )

    def _load_pretrained_weights(self) -> None:
        """Load RNA-FM weights from HuggingFace cache or download."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download("multimolecule/rnafm", "model.safetensors")
        state_dict = load_file(path)
        n_loaded = self.encoder.load_from_safetensors(state_dict)
        logger.info("RNA-FM: loaded %d weight tensors from %s", n_loaded, path)

    def forward(self, sequences: list[str]) -> torch.Tensor:
        """Forward pass: tokenize -> RNA-FM -> LoRA -> project -> align.

        Args:
            sequences: list of RNA strings (crRNA spacers, 20-nt each)
        Returns:
            (batch, target_len, proj_dim) with features at spacer positions
        """
        device = next(self.parameters()).device
        input_ids = tokenize_rna(sequences, max_len=self.spacer_len + 2).to(device)
        pad_mask = (input_ids != RNAFM_VOCAB["<pad>"]).long()

        # Run RNA-FM with LoRA-adapted layers
        hidden = self.encoder(input_ids, pad_mask)  # (B, 22, 640)

        # Strip <cls> and <eos>
        hidden = hidden[:, 1:self.spacer_len + 1, :]  # (B, 20, 640)

        # Project
        projected = self.proj(hidden)  # (B, 20, proj_dim)

        # Align to 34-nt target positions (zero-pad PAM + flanking)
        batch_size = projected.size(0)
        out = torch.zeros(
            batch_size, self.target_len, self.proj_dim,
            device=device, dtype=projected.dtype,
        )
        end = self.spacer_start + min(self.spacer_len, projected.size(1))
        out[:, self.spacer_start:end, :] = projected[:, :end - self.spacer_start, :]
        return out

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
