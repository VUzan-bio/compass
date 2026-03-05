"""Phase 2 ablation row 4: CNN + RNA-FM + RLPA + Multi-task.

Two-step process:
    Step A: Generate proxy discrimination labels using frozen Phase 1 RLPA model.
            For each training sequence, create paired targets (mutant + wildtype
            with mismatch at seed positions 4-11). Run both through frozen
            efficiency head. Proxy disc = pred_eff(mutant) / pred_eff(wildtype).

    Step B: Train with multi-task loss (efficiency + proxy discrimination).
            lambda_disc=0.1 (low weight -- proxy labels are self-distilled).
            Early stopping on val Spearman rho (EFFICIENCY only).

Usage (from compass/ root):
    python compass-net/scripts/run_phase2_multitask.py
    python compass-net/scripts/run_phase2_multitask.py --device cuda
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


def train_multitask_epoch(
    model,
    train_loader,
    proxy_disc_labels,
    loss_fn,
    optimizer,
    device,
    embedding_cache,
    label_noise_sigma=0.02,
):
    """Train one epoch with multi-task loss (efficiency + proxy discrimination).

    proxy_disc_labels is a dict mapping sequence index -> list of
    (wildtype_onehot, proxy_disc_ratio) pairs across mismatch positions.
    We sample ONE random mismatch position per sequence per epoch.
    """
    import random
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        target_onehot = batch["target_onehot"].to(device)
        efficiency = batch["efficiency"].to(device)

        # Gaussian label noise
        if label_noise_sigma > 0:
            noise = torch.randn_like(efficiency) * label_noise_sigma
            efficiency = (efficiency + noise).clamp(0.0, 1.0)

        # RNA-FM embeddings
        from compass_ml.training.train_compass_ml import _get_batch_embeddings
        crrna_emb = _get_batch_embeddings(
            batch["crrna_spacer"], embedding_cache, device,
        )

        # Build wildtype targets and proxy disc labels for this batch
        batch_wt_onehot = []
        batch_disc_labels = []
        has_disc = False

        for i, spacer in enumerate(batch["crrna_spacer"]):
            if spacer in proxy_disc_labels:
                pairs = proxy_disc_labels[spacer]
                # Sample one random mismatch position per sequence
                wt_onehot, disc_ratio = random.choice(pairs)
                batch_wt_onehot.append(wt_onehot)
                batch_disc_labels.append(disc_ratio)
                has_disc = True
            else:
                # Fallback: use mutant as wildtype (disc=1.0, no signal)
                batch_wt_onehot.append(target_onehot[i].cpu())
                batch_disc_labels.append(1.0)

        wt_onehot = torch.stack(batch_wt_onehot).to(device)
        disc_labels = torch.tensor(batch_disc_labels, dtype=torch.float32).to(device)

        output = model(
            target_onehot=target_onehot,
            crrna_rnafm_emb=crrna_emb,
            wt_target_onehot=wt_onehot,
        )

        losses = loss_fn(
            pred_eff=output["efficiency"],
            true_eff=efficiency,
            pred_disc=output.get("discrimination"),
            true_disc=disc_labels if has_disc else None,
        )

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses["total"].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Compass-ML Phase 2 Multi-task (ablation row 4)",
    )
    parser.add_argument(
        "--data", type=str,
        default="compass/data/kim2018/nbt4061_source_data.xlsx",
    )
    parser.add_argument("--cache-dir", type=str, default="E:/compass-net-data/cache/rnafm")
    parser.add_argument(
        "--pretrained", type=str,
        default="E:/compass-net-data/weights/phase1_rlpa_best.pt",
        help="Phase 1 RLPA checkpoint to transfer from and generate proxy labels",
    )
    parser.add_argument("--output", type=str, default="E:/compass-net-data/weights/phase2_multitask_best.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    _setup()

    from compass_ml.compass_ml import CompassML
    from compass_ml.data.paired_loader import (
        SingleTargetDataset, PairedTargetDataset,
        _one_hot, _introduce_target_mismatch, _reverse_complement,
    )
    from compass_ml.data.embedding_cache import EmbeddingCache
    from compass_ml.training.train_compass_ml import (
        collate_single_target, collate_paired_target,
        _get_batch_embeddings, validate,
    )
    from compass_ml.losses.multitask_loss import MultiTaskLoss
    from compass_ml.training.reproducibility import seed_everything

    seed_everything(args.seed)
    device = torch.device(args.device)
    logger.info("Device: %s", device)

    # --- Load data ---
    logger.info("Loading Kim 2018 data...")
    (seqs_train, y_train), (seqs_val, y_val), (seqs_test, y_test) = \
        load_kim2018_sequences(args.data)
    logger.info("Loaded: train=%d, val=%d, test=%d",
                len(seqs_train), len(seqs_val), len(seqs_test))

    # --- Embedding cache ---
    cache = EmbeddingCache(args.cache_dir)
    logger.info("Embedding cache: %d entries", len(cache))

    # =====================================================================
    # Step A: Generate proxy discrimination labels
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step A: Generating proxy discrimination labels")
    logger.info("=" * 60)

    # Load Phase 1 RLPA model (frozen for proxy generation)
    phase1_model = CompassML(
        use_rnafm=True,
        use_rloop_attention=True,
        multitask=False,
    )
    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    phase1_model.load_state_dict(ckpt["model_state_dict"])
    phase1_model = phase1_model.to(device)
    phase1_model.eval()
    logger.info("Loaded Phase 1 RLPA model from %s", args.pretrained)

    # Generate proxy labels: for each training sequence, create paired targets
    # at each seed mismatch position (4-11), predict efficiency on both,
    # compute ratio. Store as dict: crRNA_spacer -> [(wt_onehot, disc_ratio), ...]
    seed_positions = list(range(4, 12))  # positions 4-11 in 34-nt encoding
    proxy_disc_labels = {}  # crRNA_spacer -> [(wt_onehot, disc_ratio)]

    logger.info("Generating proxy labels for %d training sequences at %d mismatch positions...",
                len(seqs_train), len(seed_positions))

    batch_size_proxy = 512
    with torch.no_grad():
        for i in range(0, len(seqs_train), batch_size_proxy):
            batch_seqs = seqs_train[i:i + batch_size_proxy]

            for pos in seed_positions:
                # Mutant targets (original)
                mut_onehots = torch.stack([_one_hot(s) for s in batch_seqs]).to(device)

                # Wildtype targets (mismatch at pos)
                wt_seqs = [_introduce_target_mismatch(s, pos) for s in batch_seqs]
                wt_onehots = torch.stack([_one_hot(s) for s in wt_seqs]).to(device)

                # crRNA spacers
                spacers = []
                for s in batch_seqs:
                    protospacer = s[4:24]
                    spacers.append(_reverse_complement(protospacer).replace("T", "U"))

                crrna_emb = _get_batch_embeddings(spacers, cache, device)

                # Predict efficiency on both
                mut_out = phase1_model(target_onehot=mut_onehots, crrna_rnafm_emb=crrna_emb)
                wt_out = phase1_model(target_onehot=wt_onehots, crrna_rnafm_emb=crrna_emb)

                mut_eff = mut_out["efficiency"].squeeze(-1)
                wt_eff = wt_out["efficiency"].squeeze(-1)
                proxy_ratios = (mut_eff / wt_eff.clamp(min=0.01)).cpu()

                for j, spacer in enumerate(spacers):
                    if spacer not in proxy_disc_labels:
                        proxy_disc_labels[spacer] = []
                    proxy_disc_labels[spacer].append(
                        (_one_hot(wt_seqs[j]), float(proxy_ratios[j]))
                    )

            if (i + len(batch_seqs)) % 5000 < batch_size_proxy or i + len(batch_seqs) >= len(seqs_train):
                logger.info("  Proxy labels: %d / %d sequences processed",
                            min(i + len(batch_seqs), len(seqs_train)), len(seqs_train))

    logger.info("Proxy disc labels generated for %d unique crRNA spacers", len(proxy_disc_labels))

    # Free Phase 1 model memory
    del phase1_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =====================================================================
    # Step B: Train with multi-task
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step B: Multi-task training (efficiency + proxy discrimination)")
    logger.info("=" * 60)

    # Build multi-task model
    model = CompassML(
        use_rnafm=True,
        use_rloop_attention=True,
        multitask=True,
    )
    logger.info("CompassML: %d params | RNA-FM=True | RLPA=True | MT=True",
                model.count_trainable_params())

    # Transfer ALL shared weights from Phase 1 RLPA checkpoint
    # (discrimination head is new, randomly initialized)
    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    pretrained_state = ckpt["model_state_dict"]
    model_state = model.state_dict()

    transferred = 0
    new_layers = []
    for name, param in model_state.items():
        if name in pretrained_state and pretrained_state[name].shape == param.shape:
            model_state[name] = pretrained_state[name]
            transferred += 1
        else:
            new_layers.append(name)

    model.load_state_dict(model_state)
    logger.info("Weight transfer: %d shared layers transferred", transferred)
    logger.info("New layers (randomly initialized): %s", new_layers)

    model = model.to(device)

    # --- Create datasets and loaders ---
    train_ds = SingleTargetDataset(seqs_train, y_train.tolist())
    val_ds = SingleTargetDataset(seqs_val, y_val.tolist())
    test_ds = SingleTargetDataset(seqs_test, y_test.tolist())

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, collate_fn=collate_single_target,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_single_target,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_single_target,
    )

    # --- Loss: multi-task with proxy disc ---
    loss_fn = MultiTaskLoss(
        lambda_disc=0.1,   # low weight -- proxy labels are self-distilled
        lambda_rank=0.5,
    )

    # --- Optimizer ---
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6,
    )

    # --- Training loop ---
    # Custom loop because we need the proxy disc labels in the training step
    best_rho = -1.0
    patience_counter = 0
    patience = 25
    n_epochs = args.epochs

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rho": [],
        "lr": [],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for epoch in range(n_epochs):
        # Anneal Spearman regularization
        spearman_strength = max(0.1, 1.0 - 0.9 * epoch / n_epochs)
        loss_fn.set_spearman_strength(spearman_strength)

        # Train with multi-task
        train_loss = train_multitask_epoch(
            model, train_loader, proxy_disc_labels, loss_fn,
            optimizer, device, cache,
            label_noise_sigma=0.02,
        )
        scheduler.step()

        # Validate on EFFICIENCY ONLY (not discrimination -- proxy labels aren't ground truth)
        val_loss, val_rho = validate(
            model, val_loader, loss_fn, device,
            embedding_cache=cache, use_rnafm=True,
        )

        current_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rho"].append(val_rho)
        history["lr"].append(current_lr)

        # Early stopping on val Spearman rho (EFFICIENCY ONLY)
        if val_rho > best_rho:
            best_rho = val_rho
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_rho": val_rho,
                    "val_loss": val_loss,
                    "phase": 2,
                    "config": {
                        "lr": args.lr, "epochs": n_epochs,
                        "batch_size": args.batch_size,
                        "lambda_disc": 0.1, "lambda_rank": 0.5,
                    },
                },
                args.output,
            )
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            logger.info(
                "Phase 2 | Epoch %3d | Train: %.4f | Val: %.4f | "
                "Rho: %.4f | Best: %.4f | LR: %.2e | Spearman-s: %.2f",
                epoch + 1, train_loss, val_loss,
                val_rho, best_rho, current_lr, spearman_strength,
            )

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d. Best rho = %.4f",
                        epoch + 1, best_rho)
            break

    # Load best model
    best_ckpt = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    logger.info("Loaded best Phase 2 model from epoch %d (rho = %.4f)",
                best_ckpt["epoch"] + 1, best_ckpt["val_rho"])

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

    # --- Summary ---
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
    print(f"Best epoch: {history['val_rho'].index(max(history['val_rho'])) + 1}")
    print(f"Best val:   {max(history['val_rho']):.4f}")
    print()
    print("Comparison to previous rows:")
    print(f"  CNN only:              test rho = 0.4959")
    print(f"  CNN + RNA-FM:          test rho = 0.5009")
    print(f"  CNN + RNA-FM + RLPA:   (see row 3 results)")
    print(f"  + Multi-task:          test rho = {test_metrics['spearman_rho']:.4f}")
    print("=" * 60)

    # Save results JSON
    results = {
        "config": "CNN + RNA-FM + RLPA + Multi-task",
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
