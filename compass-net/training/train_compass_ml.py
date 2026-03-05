"""Training pipeline for Compass-ML.

Three-phase training strategy:

Phase 1 (Kim et al. data): EFFICIENCY ONLY
    - Multi-task OFF (lambda_disc=0)
    - Train CNN + RNA-FM projection -> efficiency head
    - Loss = Huber + soft_spearman
    - Benchmark CNN-only vs CNN+RNA-FM vs +RLPA

Phase 2 (Kim et al. data): PROXY DISCRIMINATION AS REGULARISER
    - Multi-task ON with low weight (lambda_disc=0.1)
    - Proxy disc labels = self-distilled from Phase 1 efficiency head
    - Purpose: pre-train discrimination head, force encoder to be
      sensitive to single-nucleotide differences
    - Low lambda ensures proxy labels don't corrupt efficiency signal

Phase 3 (electrochemical data): REAL DISCRIMINATION LABELS
    - Multi-task ON with full weight (lambda_disc=0.3-0.5)
    - Measured trans-cleavage on mutant AND wildtype templates
    - Fine-tune all trainable params (lr=1e-4)

Training features:
    - AdamW with weight decay (L2 regularisation)
    - CosineAnnealingWarmRestarts scheduler
    - Gradient clipping (max_norm=1.0)
    - Early stopping on validation Spearman rho (efficiency)
    - Spearman regularization_strength annealing (1.0 -> 0.1)
    - Gaussian label noise augmentation (sigma=0.02)

Usage:
    python -m compass_ml.training.train_compass_ml \\
        --data compass/data/kim2018/nbt4061_source_data.xlsx \\
        --output compass-net/weights/compass_ml_best.pt \\
        --phase 1 --use-rnafm --epochs 200
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

from ..compass_ml import CompassML
from ..losses.multitask_loss import MultiTaskLoss
from ..data.embedding_cache import EmbeddingCache
from .reproducibility import seed_everything

logger = logging.getLogger(__name__)


# ======================================================================
# Training configuration
# ======================================================================

PHASE_CONFIGS = {
    1: {
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "epochs": 200,
        "patience": 20,
        "batch_size": 256,
        "lambda_disc": 0.0,
        "lambda_rank": 0.5,
        "label_noise_sigma": 0.02,
    },
    2: {
        "lr": 5e-4,
        "weight_decay": 1e-3,
        "epochs": 150,
        "patience": 20,
        "batch_size": 256,
        "lambda_disc": 0.1,
        "lambda_rank": 0.5,
        "label_noise_sigma": 0.02,
    },
    3: {
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "epochs": 50,
        "patience": 15,
        "batch_size": 32,
        "lambda_disc": 0.3,
        "lambda_rank": 0.5,
        "label_noise_sigma": 0.0,
    },
}


# ======================================================================
# Core training loop
# ======================================================================


def train_one_epoch(
    model: CompassML,
    loader: DataLoader,
    loss_fn: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    embedding_cache: EmbeddingCache | None = None,
    label_noise_sigma: float = 0.0,
    use_rnafm: bool = True,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        target_onehot = batch["target_onehot"].to(device)
        efficiency = batch["efficiency"].to(device)

        # Gaussian label noise augmentation (only during training)
        if label_noise_sigma > 0:
            noise = torch.randn_like(efficiency) * label_noise_sigma
            efficiency = (efficiency + noise).clamp(0.0, 1.0)

        # Get RNA-FM embeddings from cache (batch of crRNA spacer strings)
        crrna_emb = None
        if use_rnafm and embedding_cache is not None:
            crrna_emb = _get_batch_embeddings(
                batch["crrna_spacer"], embedding_cache, device,
            )

        output = model(
            target_onehot=target_onehot,
            crrna_rnafm_emb=crrna_emb,
        )

        losses = loss_fn(
            pred_eff=output["efficiency"],
            true_eff=efficiency,
        )

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses["total"].item()

    return total_loss / max(len(loader), 1)


def validate(
    model: CompassML,
    loader: DataLoader,
    loss_fn: MultiTaskLoss,
    device: torch.device,
    embedding_cache: EmbeddingCache | None = None,
    use_rnafm: bool = True,
) -> tuple[float, float]:
    """Validate. Returns (avg_loss, spearman_rho)."""
    model.eval()
    total_loss = 0.0
    all_preds: list[float] = []
    all_targets: list[float] = []

    with torch.no_grad():
        for batch in loader:
            target_onehot = batch["target_onehot"].to(device)
            efficiency = batch["efficiency"].to(device)

            crrna_emb = None
            if use_rnafm and embedding_cache is not None:
                crrna_emb = _get_batch_embeddings(
                    batch["crrna_spacer"], embedding_cache, device,
                )

            output = model(
                target_onehot=target_onehot,
                crrna_rnafm_emb=crrna_emb,
            )

            losses = loss_fn(
                pred_eff=output["efficiency"],
                true_eff=efficiency,
            )
            total_loss += losses["total"].item()

            all_preds.extend(output["efficiency"].squeeze(-1).cpu().tolist())
            all_targets.extend(efficiency.cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    rho, _ = spearmanr(all_preds, all_targets)
    return avg_loss, float(rho) if not np.isnan(rho) else 0.0


def train_phase(
    model: CompassML,
    train_loader: DataLoader,
    val_loader: DataLoader,
    phase: int = 1,
    save_path: str = "compass_ml_best.pt",
    embedding_cache: EmbeddingCache | None = None,
    device: torch.device | None = None,
    seed: int = 42,
    **overrides,
) -> tuple[CompassML, dict[str, list[float]]]:
    """Full training for one phase.

    Args:
        model: CompassML instance.
        train_loader: training DataLoader.
        val_loader: validation DataLoader.
        phase: training phase (1, 2, or 3).
        save_path: path to save best checkpoint.
        embedding_cache: RNA-FM embedding cache (None if CNN-only).
        device: torch device.
        seed: random seed.
        **overrides: override any PHASE_CONFIGS value.

    Returns:
        (model_with_best_weights, history_dict)
    """
    seed_everything(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    cfg = {**PHASE_CONFIGS[phase], **overrides}
    use_rnafm = model.use_rnafm

    # Loss function
    loss_fn = MultiTaskLoss(
        lambda_disc=cfg["lambda_disc"],
        lambda_rank=cfg["lambda_rank"],
    )

    # Optimiser with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    # LR scheduler: cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6,
    )

    # Early stopping on validation Spearman rho (efficiency)
    best_rho = -1.0
    patience_counter = 0
    n_epochs = cfg["epochs"]
    patience = cfg["patience"]

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_rho": [],
        "lr": [],
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(n_epochs):
        # Anneal Spearman regularization: 1.0 -> 0.1 over training
        spearman_strength = max(0.1, 1.0 - 0.9 * epoch / n_epochs)
        loss_fn.set_spearman_strength(spearman_strength)

        # Train
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            embedding_cache=embedding_cache,
            label_noise_sigma=cfg["label_noise_sigma"],
            use_rnafm=use_rnafm,
        )
        scheduler.step()

        # Validate
        val_loss, val_rho = validate(
            model, val_loader, loss_fn, device,
            embedding_cache=embedding_cache,
            use_rnafm=use_rnafm,
        )

        current_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rho"].append(val_rho)
        history["lr"].append(current_lr)

        # Early stopping on val Spearman rho (primary task metric)
        if val_rho > best_rho:
            best_rho = val_rho
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_rho": val_rho,
                    "val_loss": val_loss,
                    "phase": phase,
                    "config": cfg,
                },
                save_path,
            )
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            logger.info(
                "Phase %d | Epoch %3d | Train: %.4f | Val: %.4f | "
                "Rho: %.4f | Best: %.4f | LR: %.2e | Spearman-s: %.2f",
                phase, epoch + 1, train_loss, val_loss,
                val_rho, best_rho, current_lr, spearman_strength,
            )

        if patience_counter >= patience:
            logger.info(
                "Early stopping at epoch %d. Best rho = %.4f",
                epoch + 1, best_rho,
            )
            break

    # Load best model
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        "Loaded best Phase %d model from epoch %d (rho = %.4f)",
        phase, checkpoint["epoch"] + 1, checkpoint["val_rho"],
    )

    return model, history


# ======================================================================
# Phase 2: proxy discrimination label generation
# ======================================================================


def generate_proxy_disc_labels(
    model: CompassML,
    paired_loader: DataLoader,
    embedding_cache: EmbeddingCache | None,
    device: torch.device,
) -> list[float]:
    """Use Phase 1 efficiency head to generate proxy discrimination labels.

    For each (mutant_target, wildtype_target) pair:
        1. Predict efficiency on mutant target (perfect match)
        2. Predict efficiency on wildtype target (one mismatch)
        3. Proxy disc = pred_mut / pred_wt

    These are self-distilled labels, not ground truth. Used as a regulariser
    (lambda_disc=0.1) to pre-train the discrimination head.
    """
    model.eval()
    proxy_labels: list[float] = []

    with torch.no_grad():
        for batch in paired_loader:
            mut_onehot = batch["mutant_target_onehot"].to(device)
            wt_onehot = batch["wildtype_target_onehot"].to(device)

            crrna_emb = None
            if model.use_rnafm and embedding_cache is not None:
                crrna_emb = _get_batch_embeddings(
                    batch["crrna_spacer"], embedding_cache, device,
                )

            mut_out = model(target_onehot=mut_onehot, crrna_rnafm_emb=crrna_emb)
            wt_out = model(target_onehot=wt_onehot, crrna_rnafm_emb=crrna_emb)

            mut_eff = mut_out["efficiency"].squeeze(-1)
            wt_eff = wt_out["efficiency"].squeeze(-1)

            # Avoid division by zero
            proxy = mut_eff / wt_eff.clamp(min=0.01)
            proxy_labels.extend(proxy.cpu().tolist())

    return proxy_labels


# ======================================================================
# Helpers
# ======================================================================


def _get_batch_embeddings(
    crrna_spacers: list[str] | tuple[str, ...],
    cache: EmbeddingCache,
    device: torch.device,
    seq_len: int = 20,
    dim: int = 640,
) -> torch.Tensor:
    """Look up RNA-FM embeddings from cache for a batch of crRNA spacers.

    Returns (batch, seq_len, dim) tensor. Falls back to zeros if not cached.
    """
    embeddings = []
    for spacer in crrna_spacers:
        emb = cache.get_or_zeros(spacer, seq_len=seq_len, dim=dim)
        embeddings.append(emb)
    return torch.stack(embeddings).to(device)


def collate_single_target(batch: list[dict]) -> dict:
    """Custom collate for SingleTargetDataset (handles string fields)."""
    return {
        "target_onehot": torch.stack([b["target_onehot"] for b in batch]),
        "crrna_spacer": [b["crrna_spacer"] for b in batch],
        "efficiency": torch.stack([b["efficiency"] for b in batch]),
    }


def collate_paired_target(batch: list[dict]) -> dict:
    """Custom collate for PairedTargetDataset (handles string fields)."""
    return {
        "mutant_target_onehot": torch.stack(
            [b["mutant_target_onehot"] for b in batch]
        ),
        "wildtype_target_onehot": torch.stack(
            [b["wildtype_target_onehot"] for b in batch]
        ),
        "crrna_spacer": [b["crrna_spacer"] for b in batch],
        "efficiency": torch.stack([b["efficiency"] for b in batch]),
        "mismatch_pos": [b["mismatch_pos"] for b in batch],
    }


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Compass-ML")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--use-rnafm", action="store_true")
    parser.add_argument("--use-rlpa", action="store_true")
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="compass-net/cache/rnafm")
    parser.add_argument("--output", type=str, default="compass-net/weights/compass_ml_best.pt")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Build model
    model = CompassML(
        use_rnafm=args.use_rnafm,
        use_rloop_attention=args.use_rlpa,
        multitask=args.multitask,
    )
    logger.info(
        "Compass-ML: %d trainable params | RNA-FM: %s | RLPA: %s | MT: %s",
        model.count_trainable_params(),
        args.use_rnafm, args.use_rlpa, args.multitask,
    )

    # Load checkpoint if resuming
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint from %s", args.checkpoint)

    # Embedding cache
    cache = None
    if args.use_rnafm:
        cache = EmbeddingCache(args.cache_dir)
        logger.info("Embedding cache: %d entries", len(cache))

    # Data loading placeholder -- in practice, integrate with Kim 2018 loader
    logger.info(
        "Data loading not implemented in CLI. Use train_phase() directly "
        "with your DataLoaders. See compass-net/training/train_compass_ml.py."
    )

    # Build overrides from CLI args
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr

    logger.info("Phase %d config: %s", args.phase, {**PHASE_CONFIGS[args.phase], **overrides})


if __name__ == "__main__":
    main()
