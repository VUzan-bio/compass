"""Pre-compute and cache RNA-FM embeddings for all training sequences.

RNA-FM inference is too slow to run every training batch. This cache
stores pre-computed embeddings on disk for O(1) lookup during training.

Usage (one-time):
    python -m compass_ml.data.embedding_cache \\
        --sequences sequences.txt \\
        --cache_dir compass-net/cache/rnafm \\
        --model_path RNA-FM_pretrained.pth

During training:
    cache = EmbeddingCache("compass-net/cache/rnafm")
    emb = cache.get("AUGCCGAUUCGA...")  # (20, 640) tensor
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


class EmbeddingCache:
    """Manages pre-computed RNA-FM embeddings.

    Storage: {seq_hash: tensor(seq_len, 640)} in batched .pt files.
    Index file maps seq_hash -> file_idx for O(1) lookup.
    Separate from model code -- cache once, train many times.
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, int] = {}
        self._batch_cache: dict[int, dict[str, torch.Tensor]] = {}
        self._load_index()

    def get(self, sequence: str) -> torch.Tensor | None:
        """Retrieve cached embedding for a sequence.

        Args:
            sequence: RNA sequence (crRNA spacer, e.g. "AUGCCG...").

        Returns:
            (seq_len, 640) tensor or None if not cached.
        """
        seq_hash = self._hash(sequence)
        if seq_hash not in self.index:
            return None
        file_idx = self.index[seq_hash]

        # In-memory LRU for hot batches during training
        if file_idx not in self._batch_cache:
            path = self.cache_dir / f"batch_{file_idx}.pt"
            self._batch_cache[file_idx] = torch.load(
                path, map_location="cpu", weights_only=True,
            )
        return self._batch_cache[file_idx][seq_hash]

    def get_or_zeros(self, sequence: str, seq_len: int = 20, dim: int = 640) -> torch.Tensor:
        """Get cached embedding, or return zeros if not cached.

        Useful during development/testing when RNA-FM is not available.
        """
        emb = self.get(sequence)
        if emb is not None:
            return emb
        return torch.zeros(seq_len, dim)

    def put_batch(
        self, sequences: list[str], embeddings: list[torch.Tensor],
    ) -> None:
        """Store a batch of embeddings to disk.

        Args:
            sequences: list of RNA sequences.
            embeddings: list of (seq_len, 640) tensors.
        """
        file_idx = len(list(self.cache_dir.glob("batch_*.pt")))
        data: dict[str, torch.Tensor] = {}
        for seq, emb in zip(sequences, embeddings):
            seq_hash = self._hash(seq)
            data[seq_hash] = emb.cpu()
            self.index[seq_hash] = file_idx
        torch.save(data, self.cache_dir / f"batch_{file_idx}.pt")
        self._save_index()

    def has(self, sequence: str) -> bool:
        return self._hash(sequence) in self.index

    def __len__(self) -> int:
        return len(self.index)

    def _hash(self, seq: str) -> str:
        return hashlib.sha256(seq.encode()).hexdigest()[:16]

    def _load_index(self) -> None:
        idx_path = self.cache_dir / "index.pt"
        if idx_path.exists():
            self.index = torch.load(idx_path, map_location="cpu", weights_only=True)

    def _save_index(self) -> None:
        torch.save(self.index, self.cache_dir / "index.pt")

    def clear_memory_cache(self) -> None:
        """Free in-memory batch cache (call between epochs if memory-tight)."""
        self._batch_cache.clear()
