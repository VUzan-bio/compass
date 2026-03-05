"""Phase 1 training: CNN-only baseline on Kim et al. 2018 data.

Sanity check: this should reproduce SeqCNN v1 performance.
Expected: val rho ~0.74, test rho ~0.53.

Usage (from compass/ root):
    python compass-net/run_phase1.py
    python compass-net/run_phase1.py --use-rnafm --cache-dir compass-net/cache/rnafm
"""

from __future__ import annotations

import sys
import os
import argparse
import logging
import importlib
import importlib.util

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------------
# Bootstrap: make compass-net/ importable as "compass_ml"
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)


def _register(full_name: str, file_path: str, search_paths: list[str] | None = None):
    spec = importlib.util.spec_from_file_location(
        full_name, file_path, submodule_search_locations=search_paths,
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = full_name if search_paths else full_name.rsplit(".", 1)[0]
    if search_paths:
        mod.__path__ = search_paths
    sys.modules[full_name] = mod
    return spec, mod


def _exec(full_name: str, silent: bool = True):
    mod = sys.modules.get(full_name)
    if mod is None:
        return
    spec = mod.__spec__
    if spec is None or spec.loader is None:
        return
    prefix = full_name + "."
    children = [
        k for k in sys.modules
        if k.startswith(prefix) and k.count(".") == full_name.count(".") + 1
    ]
    # Two-pass exec: first pass may fail due to inter-sibling deps,
    # second pass resolves them (e.g. multitask_loss depends on spearman_loss)
    failed = []
    for child_name in sorted(children):
        child = sys.modules[child_name]
        if child.__spec__ and child.__spec__.loader:
            try:
                child.__spec__.loader.exec_module(child)
            except Exception:
                failed.append(child_name)
    for child_name in failed:
        child = sys.modules[child_name]
        if child.__spec__ and child.__spec__.loader:
            try:
                child.__spec__.loader.exec_module(child)
            except Exception as e:
                if not silent:
                    print(f"  WARN exec {child_name}: {e}")
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        if not silent:
            print(f"  WARN exec {full_name}: {e}")


def _setup():
    pkg_init = os.path.join(_SCRIPT_DIR, "__init__.py")
    _register("compass_ml", pkg_init, [_SCRIPT_DIR])
    for sub in ["branches", "attention", "heads", "losses", "data",
                "training", "evaluation", "features"]:
        sub_dir = os.path.join(_SCRIPT_DIR, sub)
        sub_init = os.path.join(sub_dir, "__init__.py")
        if os.path.isfile(sub_init):
            _register(f"compass_ml.{sub}", sub_init, [sub_dir])
    _register("compass_ml.compass_ml", os.path.join(_SCRIPT_DIR, "compass_ml.py"))
    for sub in ["branches", "attention", "heads", "losses", "data",
                "training", "evaluation", "features"]:
        sub_dir = os.path.join(_SCRIPT_DIR, sub)
        if not os.path.isdir(sub_dir):
            continue
        for fname in sorted(os.listdir(sub_dir)):
            if fname.endswith(".py") and fname != "__init__.py":
                _register(
                    f"compass_ml.{sub}.{fname[:-3]}",
                    os.path.join(sub_dir, fname),
                )
    # Execute in dependency order, two passes for cross-package deps
    exec_order = [
        "compass_ml.branches", "compass_ml.attention", "compass_ml.heads",
        "compass_ml.losses", "compass_ml.data", "compass_ml.features",
        "compass_ml.compass_ml", "compass_ml.evaluation", "compass_ml.training",
        "compass_ml",
    ]
    for m in exec_order:
        _exec(m, silent=True)
    # Second pass: retry any modules that failed due to cross-package deps
    for m in exec_order:
        _exec(m, silent=False)


# ---------------------------------------------------------------------------
# Kim 2018 data loading (returns raw sequences + normalised labels)
# ---------------------------------------------------------------------------

def load_kim2018_sequences(
    xlsx_path: str,
) -> tuple[
    tuple[list[str], np.ndarray],  # train
    tuple[list[str], np.ndarray],  # val
    tuple[list[str], np.ndarray],  # test
]:
    """Load Kim 2018 data as raw 34-nt sequences + normalised labels.

    Same logic as compass/scoring/data_loader.py but preserves raw sequences
    so they can flow through SingleTargetDataset (which does its own encoding).
    """
    path = os.path.join(_ROOT_DIR, xlsx_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Kim 2018 data not found at {path}")

    def _load_sheet(sheet_name: str) -> tuple[list[str], np.ndarray]:
        df = pd.read_excel(path, sheet_name=sheet_name, header=1)

        # Find columns (same heuristic as data_loader.py)
        seq_col = None
        for c in df.columns:
            if "34 bp" in str(c) or "34bp" in str(c):
                seq_col = c
                break
        if seq_col is None:
            seq_col = df.columns[1]

        indel_col = None
        for c in df.columns:
            if "Background substracted" in str(c) or "Background subtracted" in str(c):
                indel_col = c
                break
        if indel_col is None:
            indel_col = df.columns[-1]

        valid = pd.DataFrame({"seq": df[seq_col], "indel": df[indel_col]}).dropna()
        sequences = valid["seq"].astype(str).values
        indels = valid["indel"].values.astype(np.float64)

        # Filter: valid 34-nt DNA
        mask = np.array([
            len(s) == 34 and all(c in "ACGTacgt" for c in s)
            for s in sequences
        ])
        sequences = sequences[mask].tolist()
        indels = indels[mask]

        # Clip negatives, normalise with log transform
        indels = np.clip(indels, 0, None)
        labels = np.log2(indels + 1)
        lo, hi = labels.min(), labels.max()
        if hi - lo > 1e-8:
            labels = (labels - lo) / (hi - lo)
        labels = labels.astype(np.float32)

        # Uppercase
        sequences = [s.upper() for s in sequences]

        return sequences, labels

    seqs_train, y_train = _load_sheet("Data set HT 1-1")
    seqs_val, y_val = _load_sheet("Data set HT 1-2")
    seqs_test1, y_test1 = _load_sheet("Data set HT 2")
    seqs_test2, y_test2 = _load_sheet("Data set HT 3")

    seqs_test = seqs_test1 + seqs_test2
    y_test = np.concatenate([y_test1, y_test2])

    return (seqs_train, y_train), (seqs_val, y_val), (seqs_test, y_test)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model,
    loader,
    device,
    embedding_cache=None,
    use_rnafm=False,
) -> dict[str, float]:
    """Evaluate on a dataset. Returns Spearman, Pearson, MSE, MAE, top-k."""
    from compass_ml.training.train_compass_ml import _get_batch_embeddings

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            target_onehot = batch["target_onehot"].to(device)
            efficiency = batch["efficiency"]

            crrna_emb = None
            if use_rnafm and embedding_cache is not None:
                crrna_emb = _get_batch_embeddings(
                    batch["crrna_spacer"], embedding_cache, device,
                )

            output = model(
                target_onehot=target_onehot,
                crrna_rnafm_emb=crrna_emb,
            )

            all_preds.extend(output["efficiency"].squeeze(-1).cpu().tolist())
            all_targets.extend(efficiency.cpu().tolist())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    rho, _ = spearmanr(preds, targets)
    r, _ = pearsonr(preds, targets)
    mse = float(mean_squared_error(targets, preds))
    mae = float(mean_absolute_error(targets, preds))

    k = max(1, len(targets) // 5)
    top_pred = set(np.argsort(preds)[-k:])
    top_true = set(np.argsort(targets)[-k:])
    top_k_prec = len(top_pred & top_true) / k

    return {
        "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
        "pearson_r": float(r) if not np.isnan(r) else 0.0,
        "mse": mse,
        "mae": mae,
        "top_k_precision": top_k_prec,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compass-ML Phase 1 training")
    parser.add_argument(
        "--data", type=str,
        default="compass/data/kim2018/nbt4061_source_data.xlsx",
        help="Path to Kim 2018 Excel (relative to compass/ root)",
    )
    parser.add_argument("--use-rnafm", action="store_true")
    parser.add_argument("--use-rlpa", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="compass-net/cache/rnafm")
    parser.add_argument("--output", type=str, default="compass-net/weights/phase1_best.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transfer-v1", type=str, default=None,
                        help="Path to SeqCNN v1 checkpoint for weight transfer")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Bootstrap imports
    _setup()

    from compass_ml.compass_ml import CompassML
    from compass_ml.data.paired_loader import SingleTargetDataset
    from compass_ml.data.embedding_cache import EmbeddingCache
    from compass_ml.training.train_compass_ml import train_phase, collate_single_target
    from compass_ml.training.reproducibility import seed_everything

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Load data ---
    logger.info("Loading Kim 2018 data...")
    (seqs_train, y_train), (seqs_val, y_val), (seqs_test, y_test) = \
        load_kim2018_sequences(args.data)
    logger.info(
        "Loaded: train=%d, val=%d, test=%d",
        len(seqs_train), len(seqs_val), len(seqs_test),
    )

    # --- Create datasets ---
    train_ds = SingleTargetDataset(seqs_train, y_train.tolist())
    val_ds = SingleTargetDataset(seqs_val, y_val.tolist())
    test_ds = SingleTargetDataset(seqs_test, y_test.tolist())

    # --- Build overrides ---
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr

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

    # --- Build model ---
    model = CompassML(
        use_rnafm=args.use_rnafm,
        use_rloop_attention=args.use_rlpa,
        multitask=False,
    )
    logger.info(
        "CompassML: %d params | RNA-FM=%s | RLPA=%s",
        model.count_trainable_params(), args.use_rnafm, args.use_rlpa,
    )

    # --- Optional v1 weight transfer ---
    if args.transfer_v1:
        from compass_ml.training.transfer_weights import transfer_v1_to_compass_ml
        n_transferred = transfer_v1_to_compass_ml(model, args.transfer_v1)
        logger.info("Transferred %d layers from v1 checkpoint", n_transferred)

    # --- Embedding cache (for RNA-FM mode) ---
    cache = None
    if args.use_rnafm:
        cache = EmbeddingCache(args.cache_dir)
        logger.info("Embedding cache: %d entries", len(cache))

    # --- Train Phase 1 ---
    logger.info("=" * 60)
    logger.info("Starting Phase 1 training (efficiency only)")
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

    val_metrics = evaluate(model, val_loader, device, cache, args.use_rnafm)
    test_metrics = evaluate(model, test_loader, device, cache, args.use_rnafm)

    logger.info("Validation metrics:")
    for k, v in val_metrics.items():
        logger.info("  %-20s %.4f", k, v)

    logger.info("Test metrics (HT2 + HT3):")
    for k, v in test_metrics.items():
        logger.info("  %-20s %.4f", k, v)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Compass-ML Phase 1 Results")
    print("=" * 60)
    config_str = "CNN only"
    if args.use_rnafm:
        config_str += " + RNA-FM"
    if args.use_rlpa:
        config_str += " + RLPA"
    print(f"Config:     {config_str}")
    print(f"Params:     {model.count_trainable_params():,}")
    print(f"Val rho:    {val_metrics['spearman_rho']:.4f}")
    print(f"Test rho:   {test_metrics['spearman_rho']:.4f}")
    print(f"Test r:     {test_metrics['pearson_r']:.4f}")
    print(f"Test MSE:   {test_metrics['mse']:.4f}")
    print(f"Top-20%%:   {test_metrics['top_k_precision']:.3f}")
    print(f"Best epoch: {history['val_rho'].index(max(history['val_rho'])) + 1}")
    print(f"Best val:   {max(history['val_rho']):.4f}")
    print("=" * 60)

    # Save results summary
    import json
    results = {
        "config": config_str,
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
