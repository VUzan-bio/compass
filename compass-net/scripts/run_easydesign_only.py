"""Train Compass-ML exclusively on EasyDesign trans-cleavage data.

EasyDesign (Huang et al., iMeta 2024) measures Cas12a TRANS-CLEAVAGE
fluorescence — the same readout mechanism as the COMPASS electrochemical
biosensor. This is the most relevant training signal for diagnostic
guide scoring, unlike Kim 2018 which measures cis-cleavage indels.

IMPORTANT: The original EasyDesign train/test split has a measurement
mismatch — training data uses log-k kinetic rates while test data uses
raw fluorescence intensities. These are different measurement domains
that prevent valid cross-split evaluation.

Solution: We discard the original test split and create an internal
80/10/10 random split on the 10,634 log-k training samples. This gives
a consistent trans-cleavage benchmark with uniform labels.

Data: 10,634 log-k samples -> 8,508 train / 1,063 val / 1,063 test
Cross-benchmark: Kim 2018 cis-cleavage test set (HT2 + HT3)

Usage (from compass/ root):
    python compass-net/scripts/run_easydesign_only.py
    python compass-net/scripts/run_easydesign_only.py --device cuda
"""

from __future__ import annotations

import sys
import os
import argparse
import logging
import json

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMPASS_NET_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT_DIR = os.path.dirname(_COMPASS_NET_DIR)

sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _COMPASS_NET_DIR)

from run_phase1 import _setup, load_kim2018_sequences, evaluate

logger = logging.getLogger(__name__)


def normalise_minmax(activities: np.ndarray) -> np.ndarray:
    """Normalise activities to [0, 1] via min-max scaling."""
    lo, hi = activities.min(), activities.max()
    if hi - lo > 1e-8:
        activities = (activities - lo) / (hi - lo)
    return activities.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Compass-ML EasyDesign-only training")
    parser.add_argument(
        "--ed-data", type=str,
        default="compass-net/data/external/easydesign/Table_S2.xlsx",
    )
    parser.add_argument(
        "--kim-data", type=str,
        default="compass/data/kim2018/nbt4061_source_data.xlsx",
        help="Kim 2018 data for cross-benchmark evaluation",
    )
    parser.add_argument("--cache-dir", type=str, default="E:/compass-net-data/cache/rnafm")
    parser.add_argument("--output", type=str, default="E:/compass-net-data/weights/easydesign_best.pt")
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

    _setup()

    from compass_ml.compass_ml import CompassML
    from compass_ml.data.paired_loader import SingleTargetDataset
    from compass_ml.data.embedding_cache import EmbeddingCache
    from compass_ml.training.train_compass_ml import train_phase, collate_single_target
    from compass_ml.training.reproducibility import seed_everything
    from compass_ml.data.loaders.load_easydesign import load_easydesign

    seed_everything(args.seed)
    device = torch.device(args.device)
    logger.info("Device: %s", device)

    # --- Load EasyDesign log-k data only (discard original test split) ---
    logger.info("Loading EasyDesign data (log-k training partition only)...")
    ed = load_easydesign(args.ed_data)
    all_seqs = ed["sequences"]
    all_acts = np.array(ed["activities"], dtype=np.float64)

    logger.info("EasyDesign log-k pool: %d samples (range: [%.3f, %.3f])",
                len(all_seqs), all_acts.min(), all_acts.max())
    logger.info("NOTE: Original test split discarded (measurement mismatch: "
                "log-k train vs raw fluorescence test)")

    # Normalise all log-k activities to [0, 1] BEFORE splitting
    # so train/val/test share the same normalisation
    all_acts_norm = normalise_minmax(all_acts)

    # --- Internal 80/10/10 split (random, seeded) ---
    rng = np.random.RandomState(args.seed)
    n = len(all_seqs)
    indices = rng.permutation(n)
    n_test = n // 10       # 10%
    n_val = n // 10        # 10%
    n_train = n - n_val - n_test  # 80%

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    seqs_train = [all_seqs[i] for i in train_idx]
    y_train = all_acts_norm[train_idx]
    seqs_val = [all_seqs[i] for i in val_idx]
    y_val = all_acts_norm[val_idx]
    seqs_test = [all_seqs[i] for i in test_idx]
    y_test = all_acts_norm[test_idx]

    logger.info("Internal split: train=%d, val=%d, test=%d",
                len(seqs_train), len(seqs_val), len(seqs_test))

    # --- Create datasets ---
    train_ds = SingleTargetDataset(seqs_train, y_train.tolist())
    val_ds = SingleTargetDataset(seqs_val, y_val.tolist())
    test_ds = SingleTargetDataset(seqs_test, y_test.tolist())

    # --- DataLoaders ---
    from torch.utils.data import DataLoader

    batch_size = args.batch_size or 256
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

    # --- Build model: same architecture as production (CNN + RNA-FM + RLPA) ---
    model = CompassML(
        use_rnafm=True,
        use_rloop_attention=True,
        multitask=False,
    )
    logger.info(
        "CompassML: %d params | RNA-FM=True | RLPA=True",
        model.count_trainable_params(),
    )

    # --- Embedding cache ---
    cache = EmbeddingCache(args.cache_dir)
    logger.info("Embedding cache: %d entries", len(cache))

    # --- Training config ---
    overrides = {
        "lr": args.lr or 1e-3,
        "patience": 25,
    }
    if args.epochs is not None:
        overrides["epochs"] = args.epochs

    # --- Train ---
    logger.info("=" * 60)
    logger.info("Starting EasyDesign-only training (log-k, internal split)")
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

    # --- Evaluate on internal EasyDesign test set (log-k, same domain) ---
    logger.info("=" * 60)
    logger.info("Evaluation: EasyDesign internal test (log-k)")
    logger.info("=" * 60)

    ed_val_metrics = evaluate(model, val_loader, device, cache, use_rnafm=True)
    ed_test_metrics = evaluate(model, test_loader, device, cache, use_rnafm=True)

    logger.info("EasyDesign val (log-k):")
    for k, v in ed_val_metrics.items():
        logger.info("  %-20s %.4f", k, v)

    logger.info("EasyDesign test (log-k):")
    for k, v in ed_test_metrics.items():
        logger.info("  %-20s %.4f", k, v)

    # --- Cross-benchmark: evaluate on Kim 2018 test ---
    logger.info("=" * 60)
    logger.info("Cross-benchmark: Kim 2018 test set (cis-cleavage)")
    logger.info("=" * 60)

    kim_metrics = {"spearman_rho": 0.0, "pearson_r": 0.0}
    try:
        (_, _), (_, _), (kim_test_seqs, kim_test_y) = load_kim2018_sequences(args.kim_data)
        kim_test_ds = SingleTargetDataset(kim_test_seqs, kim_test_y.tolist())
        kim_test_loader = DataLoader(
            kim_test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_single_target,
        )
        kim_metrics = evaluate(model, kim_test_loader, device, cache, use_rnafm=True)
        logger.info("Kim 2018 test (cis-cleavage):")
        for k, v in kim_metrics.items():
            logger.info("  %-20s %.4f", k, v)
    except Exception as e:
        logger.warning("Could not evaluate on Kim 2018: %s", e)

    # --- Also evaluate existing checkpoints on the same ED internal test ---
    logger.info("=" * 60)
    logger.info("Cross-benchmark: existing checkpoints on ED internal test")
    logger.info("=" * 60)

    existing_checkpoints = [
        ("Kim-only (RLPA)", "E:/compass-net-data/weights/phase1_rlpa_best.pt"),
        ("Multi-dataset no DA", "E:/compass-net-data/weights/multidataset_noda_best.pt"),
        ("Multi-dataset + DA", "E:/compass-net-data/weights/multidataset_da_best.pt"),
        ("Phase1 + RNA-FM", "E:/compass-net-data/weights/phase1_rnafm_best.pt"),
    ]

    all_results = []
    for ckpt_name, ckpt_path in existing_checkpoints:
        if not os.path.exists(ckpt_path):
            logger.warning("Checkpoint not found: %s", ckpt_path)
            continue
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            has_rnafm = any(k.startswith("rnafm.") for k in state_dict)
            has_rlpa = any(k.startswith("attention.") for k in state_dict)

            ckpt_model = CompassML(
                use_rnafm=has_rnafm,
                use_rloop_attention=has_rlpa,
                multitask=False,
            )
            ckpt_model.load_state_dict(state_dict, strict=False)
            ckpt_model.to(device)

            ckpt_ed_metrics = evaluate(ckpt_model, test_loader, device, cache, use_rnafm=has_rnafm)
            logger.info("%s on ED internal test: rho=%.4f, r=%.4f",
                        ckpt_name, ckpt_ed_metrics["spearman_rho"], ckpt_ed_metrics["pearson_r"])

            all_results.append({
                "name": ckpt_name,
                "ed_internal_rho": ckpt_ed_metrics["spearman_rho"],
                "ed_internal_r": ckpt_ed_metrics["pearson_r"],
            })
        except Exception as e:
            logger.warning("Failed to evaluate %s: %s", ckpt_name, e)

    # Add our new model
    all_results.append({
        "name": "EasyDesign-only",
        "ed_internal_rho": ed_test_metrics["spearman_rho"],
        "ed_internal_r": ed_test_metrics["pearson_r"],
    })

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Compass-ML EasyDesign-Only Results")
    print("=" * 60)
    print(f"Config:         CNN + RNA-FM + RLPA (trained on EasyDesign log-k only)")
    print(f"Params:         {model.count_trainable_params():,}")
    print(f"Train samples:  {len(seqs_train)}")
    print(f"Val samples:    {len(seqs_val)}")
    print(f"Test samples:   {len(seqs_test)}")
    print(f"ED val rho:     {ed_val_metrics['spearman_rho']:.4f}")
    print(f"ED test rho:    {ed_test_metrics['spearman_rho']:.4f}")
    print(f"ED test r:      {ed_test_metrics['pearson_r']:.4f}")
    print(f"Kim test rho:   {kim_metrics['spearman_rho']:.4f}")
    print(f"Kim test r:     {kim_metrics['pearson_r']:.4f}")
    print(f"Best epoch:     {history['val_rho'].index(max(history['val_rho'])) + 1}")
    print(f"Best val rho:   {max(history['val_rho']):.4f}")
    print()
    print("Cross-benchmark (all models on ED internal log-k test):")
    for r in all_results:
        print(f"  {r['name']:<25s} ED rho={r['ed_internal_rho']:+.4f}  ED r={r['ed_internal_r']:+.4f}")
    print("=" * 60)

    # --- Save results ---
    results = {
        "config": "CNN + RNA-FM + RLPA (EasyDesign-only, internal log-k split)",
        "params": model.count_trainable_params(),
        "split": {
            "method": "random 80/10/10 on log-k training partition",
            "note": "Original EasyDesign test split discarded due to "
                    "measurement mismatch (log-k train vs fluorescence test)",
            "train": len(seqs_train),
            "val": len(seqs_val),
            "test": len(seqs_test),
            "seed": args.seed,
        },
        "ed_val": ed_val_metrics,
        "ed_test": ed_test_metrics,
        "kim_test": kim_metrics,
        "cross_benchmark": all_results,
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
