"""Phase 1 ablation row 3: CNN + RNA-FM + RLPA.

Builds on the CNN+RNA-FM result (phase1_rnafm_best.pt) by adding
R-Loop Propagation Attention. Lower LR (5e-4) because RLPA is new
while CNN+RNA-FM layers are pre-trained.

Key question: does the biophysics-informed attention prior improve
generalization on 15K samples, or does it overfit?

Usage (from compass/ root):
    python compass-net/scripts/run_phase1_rlpa.py
    python compass-net/scripts/run_phase1_rlpa.py --device cuda
"""

from __future__ import annotations

import sys
import os
import argparse
import logging
import json
import random

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------------
# Bootstrap: reuse run_phase1.py's setup
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMPASS_NET_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT_DIR = os.path.dirname(_COMPASS_NET_DIR)

# Add compass-net/scripts/../.. to path so run_phase1 is importable
sys.path.insert(0, _ROOT_DIR)

# Import the bootstrap and data loading from run_phase1
sys.path.insert(0, _COMPASS_NET_DIR)
from run_phase1 import _setup, load_kim2018_sequences, evaluate

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compass-ML Phase 1 + RLPA (ablation row 3)")
    parser.add_argument(
        "--data", type=str,
        default="compass/data/kim2018/nbt4061_source_data.xlsx",
    )
    parser.add_argument("--cache-dir", type=str, default="E:/compass-net-data/cache/rnafm")
    parser.add_argument(
        "--pretrained", type=str,
        default="E:/compass-net-data/weights/phase1_rnafm_best.pt",
        help="CNN+RNA-FM checkpoint to transfer weights from",
    )
    parser.add_argument("--output", type=str, default="E:/compass-net-data/weights/phase1_rlpa_best.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Bootstrap compass_ml package
    _setup()

    from compass_ml.compass_ml import CompassML
    from compass_ml.data.paired_loader import SingleTargetDataset
    from compass_ml.data.embedding_cache import EmbeddingCache
    from compass_ml.training.train_compass_ml import train_phase, collate_single_target
    from compass_ml.training.reproducibility import seed_everything

    seed_everything(args.seed)
    device = torch.device(args.device)
    logger.info("Device: %s", device)

    # --- Load data (same split, same seed) ---
    logger.info("Loading Kim 2018 data...")
    (seqs_train, y_train), (seqs_val, y_val), (seqs_test, y_test) = \
        load_kim2018_sequences(args.data)
    logger.info("Loaded: train=%d, val=%d, test=%d",
                len(seqs_train), len(seqs_val), len(seqs_test))

    # --- Create datasets ---
    train_ds = SingleTargetDataset(seqs_train, y_train.tolist())
    val_ds = SingleTargetDataset(seqs_val, y_val.tolist())
    test_ds = SingleTargetDataset(seqs_test, y_test.tolist())

    # --- Build overrides ---
    overrides = {
        "lr": args.lr or 5e-4,      # lower LR: RLPA is new, CNN+RNA-FM pre-trained
        "patience": 25,              # more patience for attention to converge
    }
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size

    batch_size = overrides.get("batch_size", 256)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, collate_fn=collate_single_target,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_single_target,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_single_target,
    )

    # --- Build model with RLPA ---
    model = CompassML(
        use_rnafm=True,
        use_rloop_attention=True,
        multitask=False,
    )
    logger.info(
        "CompassML: %d params | RNA-FM=True | RLPA=True",
        model.count_trainable_params(),
    )

    # --- Transfer weights from CNN+RNA-FM checkpoint ---
    if os.path.exists(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        pretrained_state = ckpt["model_state_dict"]
        model_state = model.state_dict()

        transferred = 0
        skipped = 0
        for name, param in pretrained_state.items():
            if name in model_state and model_state[name].shape == param.shape:
                model_state[name] = param
                transferred += 1
            else:
                skipped += 1

        model.load_state_dict(model_state)
        logger.info(
            "Weight transfer from %s: %d transferred, %d skipped (new RLPA layers)",
            args.pretrained, transferred, skipped,
        )
    else:
        logger.warning("No pretrained checkpoint found at %s, training from scratch", args.pretrained)

    # --- Embedding cache ---
    cache = EmbeddingCache(args.cache_dir)
    logger.info("Embedding cache: %d entries", len(cache))

    # --- Train ---
    logger.info("=" * 60)
    logger.info("Starting Phase 1 + RLPA training")
    logger.info("=" * 60)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    model, history = train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        phase=1,
        save_path=args.output,
        embedding_cache=cache,
        device=device,
        seed=args.seed,
        **overrides,
    )

    # --- Evaluate ---
    logger.info("=" * 60)
    logger.info("Evaluation")
    logger.info("=" * 60)

    val_metrics = evaluate(model, val_loader, device, cache, use_rnafm=True)
    test_metrics = evaluate(model, test_loader, device, cache, use_rnafm=True)

    logger.info("Validation metrics:")
    for k, v in val_metrics.items():
        logger.info("  %-20s %.4f", k, v)

    logger.info("Test metrics (HT2 + HT3):")
    for k, v in test_metrics.items():
        logger.info("  %-20s %.4f", k, v)

    # --- Save attention maps for 10 random test sequences ---
    logger.info("Saving attention maps for visualization...")
    model.eval()
    attn_data = []
    random.seed(args.seed)
    sample_indices = random.sample(range(len(test_ds)), min(10, len(test_ds)))

    with torch.no_grad():
        for idx in sample_indices:
            sample = test_ds[idx]
            target_onehot = sample["target_onehot"].unsqueeze(0).to(device)
            crrna_spacer = sample["crrna_spacer"]

            from compass_ml.training.train_compass_ml import _get_batch_embeddings
            crrna_emb = _get_batch_embeddings([crrna_spacer], cache, device)

            output = model(target_onehot=target_onehot, crrna_rnafm_emb=crrna_emb)
            attn_weights = output.get("attn_weights")

            attn_data.append({
                "index": idx,
                "sequence": seqs_test[idx],
                "crrna_spacer": crrna_spacer,
                "efficiency_true": float(y_test[idx]),
                "efficiency_pred": float(output["efficiency"].item()),
                "attn_weights": attn_weights.cpu() if attn_weights is not None else None,
            })

    attn_path = args.output.replace(".pt", "_attention_maps.pt")
    torch.save(attn_data, attn_path)
    logger.info("Attention maps saved to %s", attn_path)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Compass-ML Phase 1 + RLPA Results")
    print("=" * 60)
    print(f"Config:     CNN + RNA-FM + RLPA")
    print(f"Params:     {model.count_trainable_params():,}")
    print(f"Val rho:    {val_metrics['spearman_rho']:.4f}")
    print(f"Test rho:   {test_metrics['spearman_rho']:.4f}")
    print(f"Test r:     {test_metrics['pearson_r']:.4f}")
    print(f"Test MSE:   {test_metrics['mse']:.4f}")
    print(f"Top-20%%:   {test_metrics['top_k_precision']:.3f}")
    print(f"Best epoch: {history['val_rho'].index(max(history['val_rho'])) + 1}")
    print(f"Best val:   {max(history['val_rho']):.4f}")
    print()
    print("Comparison to previous rows:")
    print(f"  CNN only:      test rho = 0.4959")
    print(f"  CNN + RNA-FM:  test rho = 0.5009")
    print(f"  + RLPA:        test rho = {test_metrics['spearman_rho']:.4f}  (delta = {test_metrics['spearman_rho'] - 0.5009:+.4f})")
    print("=" * 60)

    # Save results JSON
    results = {
        "config": "CNN + RNA-FM + RLPA",
        "params": model.count_trainable_params(),
        "val": val_metrics,
        "test": test_metrics,
        "best_epoch": history["val_rho"].index(max(history["val_rho"])) + 1,
        "seed": args.seed,
    }
    results_path = args.output.replace(".pt", "_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
