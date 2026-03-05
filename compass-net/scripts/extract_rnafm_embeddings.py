"""Extract RNA-FM embeddings for all Kim 2018 crRNA spacers.

For each 34-nt target DNA:
    1. Extract protospacer (positions 4-23)
    2. Reverse complement -> crRNA spacer
    3. Replace T->U (RNA)
    4. Run through RNA-FM -> (20, 640) embedding
    5. Cache using EmbeddingCache

Usage (from compass/ root):
    python compass-net/scripts/extract_rnafm_embeddings.py
    python compass-net/scripts/extract_rnafm_embeddings.py --batch-size 64 --device cuda
"""

from __future__ import annotations

import sys
import os
import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch

# Add compass/ root to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMPASS_NET_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT_DIR = os.path.dirname(_COMPASS_NET_DIR)
sys.path.insert(0, _ROOT_DIR)

logger = logging.getLogger(__name__)

_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def reverse_complement(seq: str) -> str:
    return "".join(_COMPLEMENT.get(b, b) for b in reversed(seq.upper()))


def load_all_sequences(xlsx_path: str) -> list[str]:
    """Load all 34-nt target sequences from Kim 2018 data."""
    sheets = ["Data set HT 1-1", "Data set HT 1-2", "Data set HT 2", "Data set HT 3"]
    all_seqs = []

    for sheet in sheets:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, header=1)
        seq_col = None
        for c in df.columns:
            if "34 bp" in str(c) or "34bp" in str(c):
                seq_col = c
                break
        if seq_col is None:
            seq_col = df.columns[1]

        sequences = df[seq_col].dropna().astype(str).values
        for s in sequences:
            s = s.upper().strip()
            if len(s) == 34 and all(c in "ACGT" for c in s):
                all_seqs.append(s)

    return all_seqs


def target_to_crrna_spacer(target_dna: str) -> str:
    """Convert 34-nt target DNA to 20-nt crRNA spacer (RNA)."""
    protospacer = target_dna[4:24]
    return reverse_complement(protospacer).replace("T", "U")


def main():
    parser = argparse.ArgumentParser(description="Extract RNA-FM embeddings for Kim 2018")
    parser.add_argument(
        "--data", type=str,
        default=os.path.join(_ROOT_DIR, "compass/data/kim2018/nbt4061_source_data.xlsx"),
    )
    parser.add_argument("--cache-dir", type=str, default=os.path.join(_COMPASS_NET_DIR, "cache", "rnafm"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # --- Load sequences ---
    logger.info("Loading sequences from %s", args.data)
    all_targets = load_all_sequences(args.data)
    logger.info("Loaded %d target sequences", len(all_targets))

    # --- Derive unique crRNA spacers ---
    spacer_set: dict[str, str] = {}  # spacer -> first target (for logging)
    for target in all_targets:
        spacer = target_to_crrna_spacer(target)
        if spacer not in spacer_set:
            spacer_set[spacer] = target
    unique_spacers = list(spacer_set.keys())
    logger.info("Unique crRNA spacers: %d (from %d targets)", len(unique_spacers), len(all_targets))

    # --- Setup cache ---
    os.makedirs(args.cache_dir, exist_ok=True)

    # Bootstrap compass_ml.data.embedding_cache
    import importlib.util
    cache_path = os.path.join(_COMPASS_NET_DIR, "data", "embedding_cache.py")
    spec = importlib.util.spec_from_file_location("embedding_cache", cache_path)
    cache_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cache_mod)
    EmbeddingCache = cache_mod.EmbeddingCache

    cache = EmbeddingCache(args.cache_dir)
    logger.info("Cache dir: %s (existing entries: %d)", args.cache_dir, len(cache))

    # Filter out already-cached spacers
    to_extract = [s for s in unique_spacers if not cache.has(s)]
    logger.info("Already cached: %d, to extract: %d", len(unique_spacers) - len(to_extract), len(to_extract))

    if not to_extract:
        logger.info("All embeddings already cached. Nothing to do.")
        return

    # --- Load RNA-FM ---
    logger.info("Loading RNA-FM model...")
    import fm
    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(args.device)
    model.eval()
    logger.info("RNA-FM loaded on %s", args.device)

    # --- Extract embeddings ---
    start_time = time.time()
    total = len(to_extract)

    for i in range(0, total, args.batch_size):
        batch_spacers = to_extract[i:i + args.batch_size]

        # RNA-FM expects (label, sequence) tuples
        data = [(f"seq_{i + j}", seq) for j, seq in enumerate(batch_spacers)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(args.device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[12])
            emb = results["representations"][12]  # (batch, seq_len+2, 640)

        # Extract per-sequence embeddings (strip BOS/EOS tokens)
        batch_embs = []
        for j in range(len(batch_spacers)):
            seq_len = len(batch_spacers[j])
            batch_embs.append(emb[j, 1:seq_len + 1, :].cpu())

        # Store in cache
        cache.put_batch(batch_spacers, batch_embs)

        if (i + len(batch_spacers)) % 500 < args.batch_size or i + len(batch_spacers) >= total:
            elapsed = time.time() - start_time
            done = i + len(batch_spacers)
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            logger.info(
                "  %d / %d (%.1f%%) | %.1f seq/s | ETA: %.0fs",
                done, total, 100 * done / total, rate, eta,
            )

    elapsed = time.time() - start_time
    logger.info("Done. Extracted %d embeddings in %.1fs (%.1f seq/s)", total, elapsed, total / elapsed)
    logger.info("Cache now has %d entries", len(cache))


if __name__ == "__main__":
    main()
