"""Evaluate Phase 2 multi-task checkpoint (eval-only, no training)."""
from __future__ import annotations

import sys, os, json, logging

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMPASS_NET_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT_DIR = os.path.dirname(_COMPASS_NET_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _COMPASS_NET_DIR)

import torch
from torch.utils.data import DataLoader
from run_phase1 import _setup, load_kim2018_sequences, evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="E:/compass-net-data/weights/phase2_multitask_best.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    _setup()

    from compass_ml.compass_ml import CompassML
    from compass_ml.data.paired_loader import SingleTargetDataset
    from compass_ml.training.train_compass_ml import collate_single_target
    from compass_ml.data.embedding_cache import EmbeddingCache

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    xlsx_path = "compass/data/kim2018/nbt4061_source_data.xlsx"
    (seqs_train, y_train), (seqs_val, y_val), (seqs_test, y_test) = load_kim2018_sequences(xlsx_path)
    logger.info("Loaded: train=%d, val=%d, test=%d", len(seqs_train), len(seqs_val), len(seqs_test))

    cache = EmbeddingCache("E:/compass-net-data/cache/rnafm")
    logger.info("Embedding cache: %d entries", len(cache))

    model = CompassML(use_rnafm=True, use_rloop_attention=True, multitask=True)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info("Loaded checkpoint from epoch %d (val_rho=%.4f)", ckpt["epoch"] + 1, ckpt["val_rho"])
    logger.info("Params: %d", model.count_trainable_params())

    val_ds = SingleTargetDataset(seqs_val, y_val.tolist())
    test_ds = SingleTargetDataset(seqs_test, y_test.tolist())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_single_target)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_single_target)

    val_metrics = evaluate(model, val_loader, device, cache, use_rnafm=True)
    test_metrics = evaluate(model, test_loader, device, cache, use_rnafm=True)

    logger.info("Validation metrics:")
    for k, v in val_metrics.items():
        logger.info("  %-20s %.4f", k, v)
    logger.info("Test metrics (HT2 + HT3):")
    for k, v in test_metrics.items():
        logger.info("  %-20s %.4f", k, v)

    print("\n" + "=" * 60)
    print("Compass-ML Phase 2 Multi-task Results")
    print("=" * 60)
    print(f"Config:     CNN + RNA-FM + RLPA + Multi-task")
    print(f"Params:     {model.count_trainable_params():,}")
    print(f"Val rho:    {val_metrics['spearman_rho']:.4f}")
    print(f"Test rho:   {test_metrics['spearman_rho']:.4f}")
    print(f"Test r:     {test_metrics['pearson_r']:.4f}")
    print(f"Test MSE:   {test_metrics['mse']:.4f}")
    print(f"Top-20%%:   {test_metrics['top_k_precision']:.3f}")
    print(f"Best epoch: {ckpt['epoch'] + 1}")
    print(f"Best val:   {ckpt['val_rho']:.4f}")
    print()
    print("Comparison to previous rows:")
    print(f"  CNN only:              test rho = 0.4959")
    print(f"  CNN + RNA-FM:          test rho = 0.5009")
    print(f"  CNN + RNA-FM + RLPA:   test rho = 0.5336")
    print(f"  + Multi-task:          test rho = {test_metrics['spearman_rho']:.4f}")
    print("=" * 60)

    results = {
        "config": "CNN + RNA-FM + RLPA + Multi-task",
        "params": model.count_trainable_params(),
        "val": val_metrics,
        "test": test_metrics,
        "best_epoch": ckpt["epoch"] + 1,
    }
    results_path = args.checkpoint.replace(".pt", "_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
