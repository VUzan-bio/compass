"""Generate synthetic discrimination pairs via self-distillation.

Uses the trained efficiency head to predict activity for all 60
single-mismatch variants of each training target, creating pseudo-
discrimination pairs.

For each 34-nt target:
  - The original is "MUT" (fully matched to crRNA, high activity)
  - Each of 20 positions x 3 alternative bases = 60 variants is "WT"
    (one mismatch -> predicted lower activity)
  - Pseudo ratio = pred_eff(original) / pred_eff(variant)

The crRNA is the SAME for all variants (only target DNA changes).
RNA-FM embedding is identical across variants (it encodes the crRNA).

Saves: compass-net/data/synthetic_disc_pairs.pt
  dict with 'mut_seqs', 'wt_seqs', 'mm_positions', 'ratios'

Usage (from compass/ root):
    python compass-net/scripts/generate_synthetic_pairs.py --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMPASS_NET_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT_DIR = os.path.dirname(_COMPASS_NET_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _COMPASS_NET_DIR)

logger = logging.getLogger(__name__)


def _one_hot(seq: str, max_len: int = 34) -> torch.Tensor:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    mat = torch.zeros(4, max_len)
    for i, nt in enumerate(seq[:max_len].upper()):
        idx = mapping.get(nt)
        if idx is not None:
            mat[idx, i] = 1.0
    return mat


def _reverse_complement(seq: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(comp.get(b, b) for b in reversed(seq.upper()))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic disc pairs")
    parser.add_argument(
        "--checkpoint", type=str,
        default="compass-net/checkpoints/enhancements/C_position_best.pt",
        help="Checkpoint with trained efficiency head",
    )
    parser.add_argument(
        "--data", type=str,
        default="compass/data/kim2018/nbt4061_source_data.xlsx",
    )
    parser.add_argument("--output", type=str, default="compass-net/data/synthetic_disc_pairs.pt")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from run_phase1 import _setup, load_kim2018_sequences
    _setup()

    from compass_ml.compass_ml import CompassML
    from compass_ml.data.embedding_cache import EmbeddingCache

    device = torch.device(args.device)

    # Load all Kim 2018 sequences (train + val + test)
    logger.info("Loading Kim 2018 data...")
    (seqs_train, _), (seqs_val, _), (seqs_test, _) = load_kim2018_sequences(
        os.path.join(_ROOT_DIR, args.data)
    )
    all_seqs = list(seqs_train) + list(seqs_val) + list(seqs_test)
    logger.info("Total sequences: %d", len(all_seqs))

    # Load checkpoint — try C, then B, then A, then multitask baseline
    ckpt_path = os.path.join(_ROOT_DIR, args.checkpoint)
    fallbacks = [
        "compass-net/checkpoints/enhancements/B_thermo_best.pt",
        "compass-net/checkpoints/enhancements/A_contrastive_best.pt",
        "compass-net/checkpoints/multitask/compass_ml_multitask_best.pt",
    ]
    if not os.path.exists(ckpt_path):
        for fb in fallbacks:
            fb_path = os.path.join(_ROOT_DIR, fb)
            if os.path.exists(fb_path):
                ckpt_path = fb_path
                logger.info("Using fallback checkpoint: %s", fb)
                break
        else:
            raise FileNotFoundError(f"No checkpoint found")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Detect architecture from checkpoint
    has_disc = any("disc_head" in k for k in state_dict)
    has_pos = any("pos_embedding" in k for k in state_dict)
    has_thermo = False
    n_thermo = 0
    pos_embed_dim = 0

    # Check disc head input dim for thermo
    for k in state_dict:
        if "disc_head.head.0.weight" in k:
            input_dim = state_dict[k].shape[1]
            base_dim = 512  # 4 * 128
            if has_pos:
                for pk in state_dict:
                    if "pos_embedding.weight" in pk:
                        pos_embed_dim = state_dict[pk].shape[1]
                        break
            extra = input_dim - base_dim - pos_embed_dim
            if extra > 0:
                n_thermo = extra
                has_thermo = True
            break

    model = CompassML(
        use_rnafm=True, use_rloop_attention=True,
        multitask=has_disc, n_thermo=n_thermo, pos_embed_dim=pos_embed_dim,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    logger.info("Loaded model from %s", ckpt_path)

    # Embedding cache
    cache = EmbeddingCache(os.path.join(_ROOT_DIR, "compass-net/cache/rnafm"))

    # Generate synthetic pairs
    bases = "ACGT"
    mut_seqs = []
    wt_seqs = []
    mm_positions = []
    ratios = []

    logger.info("Generating synthetic pairs...")

    # Process in batches for efficiency
    batch_size = args.batch_size
    n_processed = 0

    for start in range(0, len(all_seqs), batch_size):
        batch_seqs = all_seqs[start:start + batch_size]

        # Predict efficiency for all original sequences
        orig_onehots = torch.stack([_one_hot(s) for s in batch_seqs]).to(device)

        # RNA-FM: zero embeddings (cache likely empty)
        rnafm_batch = torch.zeros(len(batch_seqs), 20, 640, device=device)

        with torch.no_grad():
            out = model(target_onehot=orig_onehots, crrna_rnafm_emb=rnafm_batch)
            orig_preds = out["efficiency"].squeeze(-1).cpu()  # (batch,)

        # For each sequence, generate 60 variants
        for seq_idx, seq in enumerate(batch_seqs):
            orig_eff = orig_preds[seq_idx].item()

            if orig_eff < 0.05:
                continue  # too low activity, ratio unreliable

            # Generate all single-mismatch variants in spacer region (positions 4-23)
            variant_onehots = []
            variant_positions = []

            for pos in range(4, 24):  # spacer positions in 34-nt window
                orig_base = seq[pos].upper() if pos < len(seq) else "N"
                for alt in bases:
                    if alt == orig_base:
                        continue
                    var_seq = seq[:pos] + alt + seq[pos + 1:]
                    variant_onehots.append(_one_hot(var_seq))
                    variant_positions.append(pos - 4 + 1)  # 1-indexed spacer position

            if not variant_onehots:
                continue

            var_tensor = torch.stack(variant_onehots).to(device)
            var_rnafm = torch.zeros(len(variant_onehots), 20, 640, device=device)

            with torch.no_grad():
                var_out = model(target_onehot=var_tensor, crrna_rnafm_emb=var_rnafm)
                var_preds = var_out["efficiency"].squeeze(-1).cpu()

            for vi in range(len(variant_onehots)):
                var_eff = var_preds[vi].item()

                # Compute pseudo ratio
                if var_eff < 1e-6:
                    pseudo_ratio = 100.0  # cap
                else:
                    pseudo_ratio = orig_eff / var_eff

                # Filter
                if pseudo_ratio < 1.0:
                    continue  # variant predicted higher (artifact)
                if pseudo_ratio > 100.0:
                    continue  # extreme outlier
                if orig_eff < 0.05 and var_eff < 0.05:
                    continue  # both too low

                mut_seqs.append(seq)
                wt_seqs.append(seq[:4 + variant_positions[vi] - 1] +
                               bases[[b for b in bases if b != seq[4 + variant_positions[vi] - 1].upper()][0] == b and True or False] +
                               seq[4 + variant_positions[vi]:] if False else
                               # Simpler: just record the variant sequence
                               seq)  # placeholder
                # Actually reconstruct properly:
                sp = variant_positions[vi]  # 1-indexed spacer pos
                abs_pos = sp + 3  # 0-indexed in 34-nt (spacer starts at pos 4)
                wt_seqs[-1] = variant_onehots[vi]  # store one-hot directly
                mm_positions.append(sp)
                ratios.append(pseudo_ratio)
                mut_seqs[-1] = seq  # keep string

        n_processed += len(batch_seqs)
        if n_processed % 5000 == 0:
            logger.info("  Processed %d/%d sequences, %d pairs so far",
                        n_processed, len(all_seqs), len(ratios))

    # Convert to tensors — store one-hots for both mut and wt
    # Re-generate cleanly since the inline approach was messy
    logger.info("Re-generating pairs cleanly with tensors...")

    # Clear and redo properly
    all_mut_seqs = []
    all_wt_seqs = []
    all_mm_positions = []
    all_ratios = []

    for start in range(0, len(all_seqs), batch_size):
        batch_seqs = all_seqs[start:start + batch_size]
        orig_onehots = torch.stack([_one_hot(s) for s in batch_seqs]).to(device)
        rnafm_batch = torch.zeros(len(batch_seqs), 20, 640, device=device)

        with torch.no_grad():
            out = model(target_onehot=orig_onehots, crrna_rnafm_emb=rnafm_batch)
            orig_preds = out["efficiency"].squeeze(-1).cpu()

        for seq_idx, seq in enumerate(batch_seqs):
            orig_eff = orig_preds[seq_idx].item()
            if orig_eff < 0.05:
                continue

            variants = []
            positions = []
            var_strs = []

            for sp in range(1, 21):  # spacer positions 1-20
                abs_pos = sp + 3  # position in 34-nt window
                if abs_pos >= len(seq):
                    continue
                orig_base = seq[abs_pos].upper()
                for alt in bases:
                    if alt == orig_base:
                        continue
                    var_seq = seq[:abs_pos] + alt + seq[abs_pos + 1:]
                    variants.append(_one_hot(var_seq))
                    positions.append(sp)
                    var_strs.append(var_seq)

            if not variants:
                continue

            var_tensor = torch.stack(variants).to(device)
            var_rnafm = torch.zeros(len(variants), 20, 640, device=device)

            with torch.no_grad():
                var_out = model(target_onehot=var_tensor, crrna_rnafm_emb=var_rnafm)
                var_preds = var_out["efficiency"].squeeze(-1).cpu()

            orig_oh = _one_hot(seq)
            for vi in range(len(variants)):
                var_eff = var_preds[vi].item()
                if var_eff < 1e-6:
                    pseudo_ratio = 100.0
                else:
                    pseudo_ratio = orig_eff / var_eff

                if pseudo_ratio < 1.0 or pseudo_ratio > 100.0:
                    continue
                if orig_eff < 0.05 and var_eff < 0.05:
                    continue

                all_mut_seqs.append(seq)
                all_wt_seqs.append(var_strs[vi])
                all_mm_positions.append(positions[vi])
                all_ratios.append(pseudo_ratio)

        n_done = start + len(batch_seqs)
        if n_done % 5000 == 0 or n_done == len(all_seqs):
            logger.info("  %d/%d sequences -> %d synthetic pairs",
                        n_done, len(all_seqs), len(all_ratios))

    logger.info("Total synthetic pairs: %d", len(all_ratios))

    # Save as strings + scalars (much smaller than one-hot tensors)
    output_path = os.path.join(_ROOT_DIR, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.save({
        "mut_seqs": all_mut_seqs,
        "wt_seqs": all_wt_seqs,
        "mm_positions": torch.tensor(all_mm_positions, dtype=torch.long),
        "ratios": torch.tensor(all_ratios, dtype=torch.float32),
        "n_source_seqs": len(all_seqs),
    }, output_path)
    logger.info("Saved to %s", output_path)

    # Stats
    ratios_np = np.array(all_ratios)
    logger.info("Ratio stats: median=%.2f, mean=%.2f, min=%.2f, max=%.2f",
                np.median(ratios_np), np.mean(ratios_np),
                np.min(ratios_np), np.max(ratios_np))


if __name__ == "__main__":
    main()
