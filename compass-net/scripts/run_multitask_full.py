"""Full multi-task Compass-ML training with REAL discrimination pairs.

Unlike Phase 2 (proxy disc from self-distillation), this uses measured
EasyDesign discrimination pairs (HD=0 vs HD=1, ~6K pairs) alongside
Kim 2018 efficiency data (~15K sequences).

Training strategy:
    - Load Phase 1 checkpoint (efficiency-only, ~200K params)
    - Add discrimination head (~35K new params, randomly initialized)
    - Differential learning rates: 1e-4 for shared encoder, 1e-3 for disc head
    - Combined early stopping: eff_rho + 0.5 * disc_r
    - lambda_disc=0.3, lambda_rank=0.5
    - 100 epochs, patience=20, batch_size=128

Data:
    - Efficiency: Kim 2018 HT1-1 (train), HT1-2 (val), HT2+HT3 (test)
    - Discrimination: EasyDesign Table_S2.xlsx HD=0/HD=1 pairs (~6,136 pairs)
      Split 80/10/10 by guide (no guide leakage between splits)

Output:
    compass-net/checkpoints/multitask/compass_ml_multitask_best.pt

Usage (from compass/ root):
    python compass-net/scripts/run_multitask_full.py
    python compass-net/scripts/run_multitask_full.py --device cuda
"""

from __future__ import annotations

import sys
import os
import argparse
import logging
import json
import random
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset

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


# ---------------------------------------------------------------------------
# Discrimination pair dataset from EasyDesign
# ---------------------------------------------------------------------------

class DiscPairDataset(Dataset):
    """Dataset of real MUT/WT discrimination pairs from EasyDesign.

    Each sample provides:
        - mut_target_onehot: (4, 34) perfect-match target
        - wt_target_onehot:  (4, 34) single-mismatch target
        - crrna_spacer:      str (for RNA-FM cache lookup)
        - disc_ratio:        float (10^(mut_logk - wt_logk))
        - mut_efficiency:    float (mut log-k, normalised)
    """

    def __init__(self, pairs):
        """Args: pairs — list of DiscriminationPair objects."""
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]

        # EasyDesign guides are 25-nt (4 PAM + 21 spacer)
        # Pad to 34-nt with flanking context (zeros)
        mut_target = p.mut_target
        wt_target = p.wt_target

        # crRNA spacer: reverse complement of protospacer (positions 4-24), as RNA
        protospacer = mut_target[4:24] if len(mut_target) >= 24 else mut_target[4:]
        crrna_spacer = _reverse_complement(protospacer).replace("T", "U")

        return {
            "mut_target_onehot": _one_hot(mut_target),
            "wt_target_onehot": _one_hot(wt_target),
            "crrna_spacer": crrna_spacer,
            "disc_ratio": torch.tensor(p.ratio_linear, dtype=torch.float32),
            "mut_efficiency": torch.tensor(p.mut_activity, dtype=torch.float32),
        }


def collate_disc_pairs(batch):
    """Custom collate for DiscPairDataset (handles string fields)."""
    return {
        "mut_target_onehot": torch.stack([b["mut_target_onehot"] for b in batch]),
        "wt_target_onehot": torch.stack([b["wt_target_onehot"] for b in batch]),
        "crrna_spacer": [b["crrna_spacer"] for b in batch],
        "disc_ratio": torch.stack([b["disc_ratio"] for b in batch]),
        "mut_efficiency": torch.stack([b["mut_efficiency"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Helpers (from paired_loader.py)
# ---------------------------------------------------------------------------

_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _reverse_complement(seq: str) -> str:
    return "".join(_COMPLEMENT.get(b, b) for b in reversed(seq.upper()))


def _one_hot(seq: str, max_len: int = 34) -> torch.Tensor:
    """One-hot encode DNA sequence. Channels-first (4, L) for Conv1d."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    mat = torch.zeros(4, max_len)
    for i, nt in enumerate(seq[:max_len].upper()):
        idx = mapping.get(nt)
        if idx is not None:
            mat[idx, i] = 1.0
    return mat


# ---------------------------------------------------------------------------
# Split disc pairs by guide (no guide leakage)
# ---------------------------------------------------------------------------

def split_disc_pairs(pairs, train_frac=0.8, val_frac=0.1, seed=42):
    """Split discrimination pairs by guide_seq to prevent leakage.

    All pairs from the same guide go into the same split.
    Returns (train_pairs, val_pairs, test_pairs).
    """
    rng = random.Random(seed)

    # Group by guide
    guide_to_pairs = defaultdict(list)
    for p in pairs:
        guide_to_pairs[p.guide_seq].append(p)

    guides = list(guide_to_pairs.keys())
    rng.shuffle(guides)

    n_train = int(len(guides) * train_frac)
    n_val = int(len(guides) * val_frac)

    train_guides = set(guides[:n_train])
    val_guides = set(guides[n_train:n_train + n_val])
    test_guides = set(guides[n_train + n_val:])

    train_pairs = [p for g in train_guides for p in guide_to_pairs[g]]
    val_pairs = [p for g in val_guides for p in guide_to_pairs[g]]
    test_pairs = [p for g in test_guides for p in guide_to_pairs[g]]

    return train_pairs, val_pairs, test_pairs


# ---------------------------------------------------------------------------
# Multi-task training epoch
# ---------------------------------------------------------------------------

def train_multitask_epoch(
    model,
    eff_loader,
    disc_loader,
    loss_fn,
    optimizer,
    device,
    embedding_cache,
    label_noise_sigma=0.02,
):
    """Train one epoch with interleaved efficiency and discrimination batches.

    Strategy: alternate between efficiency-only batches (Kim 2018) and
    discrimination batches (EasyDesign pairs). This ensures both tasks
    get gradient signal every epoch.
    """
    from compass_ml.training.train_compass_ml import _get_batch_embeddings

    model.train()
    total_loss = 0.0
    total_eff_loss = 0.0
    total_disc_loss = 0.0
    n_batches = 0

    disc_iter = iter(disc_loader)

    for eff_batch in eff_loader:
        # --- Efficiency batch ---
        target_onehot = eff_batch["target_onehot"].to(device)
        efficiency = eff_batch["efficiency"].to(device)

        if label_noise_sigma > 0:
            noise = torch.randn_like(efficiency) * label_noise_sigma
            efficiency = (efficiency + noise).clamp(0.0, 1.0)

        crrna_emb = None
        if embedding_cache is not None:
            crrna_emb = _get_batch_embeddings(
                eff_batch["crrna_spacer"], embedding_cache, device,
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
        total_eff_loss += losses["efficiency"].item()
        n_batches += 1

        # --- Discrimination batch (interleaved) ---
        try:
            disc_batch = next(disc_iter)
        except StopIteration:
            disc_iter = iter(disc_loader)
            disc_batch = next(disc_iter)

        mut_onehot = disc_batch["mut_target_onehot"].to(device)
        wt_onehot = disc_batch["wt_target_onehot"].to(device)
        disc_ratio = disc_batch["disc_ratio"].to(device)

        disc_crrna_emb = None
        if embedding_cache is not None:
            disc_crrna_emb = _get_batch_embeddings(
                disc_batch["crrna_spacer"], embedding_cache, device,
            )

        disc_output = model(
            target_onehot=mut_onehot,
            crrna_rnafm_emb=disc_crrna_emb,
            wt_target_onehot=wt_onehot,
        )

        # Discrimination-only loss (no efficiency target for EasyDesign pairs)
        if "discrimination" in disc_output:
            disc_pred = disc_output["discrimination"]
            l_disc = torch.nn.functional.huber_loss(
                torch.log1p(disc_pred.squeeze(-1)),
                torch.log1p(disc_ratio),
                delta=0.5,
            )
            disc_loss = loss_fn.lambda_disc * l_disc

            optimizer.zero_grad()
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_disc_loss += l_disc.item()

    avg_total = total_loss / max(n_batches, 1)
    avg_eff = total_eff_loss / max(n_batches, 1)
    avg_disc = total_disc_loss / max(n_batches, 1)
    return avg_total, avg_eff, avg_disc


# ---------------------------------------------------------------------------
# Validation with discrimination correlation
# ---------------------------------------------------------------------------

def validate_multitask(
    model,
    eff_loader,
    disc_loader,
    loss_fn,
    device,
    embedding_cache,
):
    """Validate both efficiency and discrimination.

    Returns:
        (val_loss, eff_rho, disc_r, disc_rho)
    """
    from compass_ml.training.train_compass_ml import _get_batch_embeddings

    model.eval()
    total_loss = 0.0
    all_eff_preds = []
    all_eff_targets = []
    all_disc_preds = []
    all_disc_targets = []

    with torch.no_grad():
        # Efficiency validation
        for batch in eff_loader:
            target_onehot = batch["target_onehot"].to(device)
            efficiency = batch["efficiency"].to(device)

            crrna_emb = None
            if embedding_cache is not None:
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

            all_eff_preds.extend(output["efficiency"].squeeze(-1).cpu().tolist())
            all_eff_targets.extend(efficiency.cpu().tolist())

        # Discrimination validation
        for batch in disc_loader:
            mut_onehot = batch["mut_target_onehot"].to(device)
            wt_onehot = batch["wt_target_onehot"].to(device)
            disc_ratio = batch["disc_ratio"]

            crrna_emb = None
            if embedding_cache is not None:
                crrna_emb = _get_batch_embeddings(
                    batch["crrna_spacer"], embedding_cache, device,
                )

            output = model(
                target_onehot=mut_onehot,
                crrna_rnafm_emb=crrna_emb,
                wt_target_onehot=wt_onehot,
            )

            if "discrimination" in output:
                all_disc_preds.extend(
                    output["discrimination"].squeeze(-1).cpu().tolist()
                )
                all_disc_targets.extend(disc_ratio.tolist())

    avg_loss = total_loss / max(len(eff_loader), 1)

    eff_rho, _ = spearmanr(all_eff_preds, all_eff_targets)
    eff_rho = float(eff_rho) if not np.isnan(eff_rho) else 0.0

    disc_r = 0.0
    disc_rho = 0.0
    if len(all_disc_preds) > 2:
        r, _ = pearsonr(all_disc_preds, all_disc_targets)
        disc_r = float(r) if not np.isnan(r) else 0.0
        rho, _ = spearmanr(all_disc_preds, all_disc_targets)
        disc_rho = float(rho) if not np.isnan(rho) else 0.0

    return avg_loss, eff_rho, disc_r, disc_rho


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full multi-task Compass-ML (efficiency + real discrimination)",
    )
    parser.add_argument(
        "--data", type=str,
        default="compass/data/kim2018/nbt4061_source_data.xlsx",
    )
    parser.add_argument(
        "--disc-data", type=str,
        default="compass-net/data/external/easydesign/Table_S2.xlsx",
    )
    parser.add_argument("--cache-dir", type=str, default="compass-net/cache/rnafm")
    parser.add_argument(
        "--pretrained", type=str,
        default="compass/weights/compass_ml_best.pt",
        help="Phase 1 checkpoint to transfer shared encoder weights from",
    )
    parser.add_argument(
        "--output", type=str,
        default="compass-net/checkpoints/multitask/compass_ml_multitask_best.pt",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-shared", type=float, default=1e-4,
                        help="LR for shared encoder (Phase 1 params)")
    parser.add_argument("--lr-disc", type=float, default=1e-3,
                        help="LR for discrimination head (new params)")
    parser.add_argument("--lambda-disc", type=float, default=0.3)
    parser.add_argument("--lambda-rank", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    _setup()

    from compass_ml.compass_ml import CompassML
    from compass_ml.data.paired_loader import SingleTargetDataset
    from compass_ml.data.embedding_cache import EmbeddingCache
    from compass_ml.data.extract_discrimination_pairs import extract_discrimination_pairs
    from compass_ml.training.train_compass_ml import collate_single_target
    from compass_ml.losses.multitask_loss import MultiTaskLoss
    from compass_ml.training.reproducibility import seed_everything

    seed_everything(args.seed)
    device = torch.device(args.device)
    logger.info("Device: %s", device)

    # =====================================================================
    # Load efficiency data (Kim 2018)
    # =====================================================================
    logger.info("Loading Kim 2018 efficiency data...")
    (seqs_train, y_train), (seqs_val, y_val), (seqs_test, y_test) = \
        load_kim2018_sequences(args.data)
    logger.info("Efficiency data: train=%d, val=%d, test=%d",
                len(seqs_train), len(seqs_val), len(seqs_test))

    # =====================================================================
    # Load real discrimination pairs (EasyDesign)
    # =====================================================================
    logger.info("Extracting EasyDesign discrimination pairs...")
    disc_pairs = extract_discrimination_pairs(args.disc_data)
    logger.info("Total discrimination pairs: %d", len(disc_pairs))

    # Split by guide (no leakage)
    disc_train, disc_val, disc_test = split_disc_pairs(
        disc_pairs, train_frac=0.8, val_frac=0.1, seed=args.seed,
    )
    logger.info("Disc splits: train=%d, val=%d, test=%d",
                len(disc_train), len(disc_val), len(disc_test))

    # =====================================================================
    # Create datasets and loaders
    # =====================================================================
    eff_train_ds = SingleTargetDataset(seqs_train, y_train.tolist())
    eff_val_ds = SingleTargetDataset(seqs_val, y_val.tolist())
    eff_test_ds = SingleTargetDataset(seqs_test, y_test.tolist())

    disc_train_ds = DiscPairDataset(disc_train)
    disc_val_ds = DiscPairDataset(disc_val)
    disc_test_ds = DiscPairDataset(disc_test)

    eff_train_loader = DataLoader(
        eff_train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, collate_fn=collate_single_target,
    )
    eff_val_loader = DataLoader(
        eff_val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_single_target,
    )
    eff_test_loader = DataLoader(
        eff_test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_single_target,
    )

    disc_train_loader = DataLoader(
        disc_train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, collate_fn=collate_disc_pairs,
    )
    disc_val_loader = DataLoader(
        disc_val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_disc_pairs,
    )
    disc_test_loader = DataLoader(
        disc_test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_disc_pairs,
    )

    # =====================================================================
    # Embedding cache (may be empty — falls back to zeros)
    # =====================================================================
    cache = EmbeddingCache(args.cache_dir)
    logger.info("Embedding cache: %d entries", len(cache))
    if len(cache) == 0:
        logger.warning(
            "No RNA-FM embeddings cached. Training will use zero embeddings. "
            "This is fine — CNN branch carries the signal. RNA-FM can be "
            "added later via cache generation + fine-tuning."
        )

    # =====================================================================
    # Build multi-task model
    # =====================================================================
    model = CompassML(
        use_rnafm=True,
        use_rloop_attention=True,
        multitask=True,
    )
    logger.info("CompassML multi-task: %d params", model.count_trainable_params())

    # Transfer shared weights from Phase 1 checkpoint
    pretrained_path = os.path.join(_ROOT_DIR, args.pretrained)
    if not os.path.exists(pretrained_path):
        pretrained_path = args.pretrained
    if os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
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
        logger.info("Weight transfer: %d shared layers from Phase 1", transferred)
        logger.info("New layers (randomly init): %s", new_layers)
        logger.info("Phase 1 val_rho: %.4f", ckpt.get("val_rho", 0.0))
    else:
        logger.warning("No pretrained checkpoint found at %s — training from scratch",
                        pretrained_path)

    model = model.to(device)

    # Count disc head params
    disc_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "disc_head" in n and p.requires_grad
    )
    shared_params = model.count_trainable_params() - disc_params
    logger.info("Shared encoder: %d params | Disc head: %d params | Total: %d",
                shared_params, disc_params, model.count_trainable_params())

    # =====================================================================
    # Differential learning rates
    # =====================================================================
    disc_head_params = [p for n, p in model.named_parameters() if "disc_head" in n]
    shared_encoder_params = [p for n, p in model.named_parameters() if "disc_head" not in n]

    optimizer = AdamW([
        {"params": shared_encoder_params, "lr": args.lr_shared},
        {"params": disc_head_params, "lr": args.lr_disc},
    ], weight_decay=1e-3)

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6,
    )

    # =====================================================================
    # Loss function
    # =====================================================================
    loss_fn = MultiTaskLoss(
        lambda_disc=args.lambda_disc,
        lambda_rank=args.lambda_rank,
    )

    # =====================================================================
    # Training loop
    # =====================================================================
    best_combined = -float("inf")
    best_eff_rho = -1.0
    best_disc_r = -1.0
    patience_counter = 0
    n_epochs = args.epochs

    history = {
        "train_loss": [], "train_eff_loss": [], "train_disc_loss": [],
        "val_loss": [], "val_eff_rho": [], "val_disc_r": [], "val_disc_rho": [],
        "combined_metric": [], "lr_shared": [], "lr_disc": [],
    }

    output_path = os.path.join(_ROOT_DIR, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("=" * 70)
    logger.info("Starting multi-task training")
    logger.info("  Epochs: %d | Patience: %d | Batch: %d",
                n_epochs, args.patience, args.batch_size)
    logger.info("  LR shared: %.1e | LR disc: %.1e", args.lr_shared, args.lr_disc)
    logger.info("  lambda_disc: %.2f | lambda_rank: %.2f",
                args.lambda_disc, args.lambda_rank)
    logger.info("  Early stopping: eff_rho + 0.5 * disc_r")
    logger.info("=" * 70)

    for epoch in range(n_epochs):
        # Anneal Spearman regularization: 1.0 -> 0.1
        spearman_strength = max(0.1, 1.0 - 0.9 * epoch / n_epochs)
        loss_fn.set_spearman_strength(spearman_strength)

        # Train
        train_loss, train_eff, train_disc = train_multitask_epoch(
            model, eff_train_loader, disc_train_loader,
            loss_fn, optimizer, device, cache,
            label_noise_sigma=0.02,
        )
        scheduler.step()

        # Validate
        val_loss, eff_rho, disc_r, disc_rho = validate_multitask(
            model, eff_val_loader, disc_val_loader,
            loss_fn, device, cache,
        )

        # Combined metric: eff_rho + 0.5 * disc_r
        combined = eff_rho + 0.5 * disc_r

        lr_shared = scheduler.get_last_lr()[0]
        lr_disc = scheduler.get_last_lr()[1] if len(scheduler.get_last_lr()) > 1 else lr_shared

        history["train_loss"].append(train_loss)
        history["train_eff_loss"].append(train_eff)
        history["train_disc_loss"].append(train_disc)
        history["val_loss"].append(val_loss)
        history["val_eff_rho"].append(eff_rho)
        history["val_disc_r"].append(disc_r)
        history["val_disc_rho"].append(disc_rho)
        history["combined_metric"].append(combined)
        history["lr_shared"].append(lr_shared)
        history["lr_disc"].append(lr_disc)

        # Early stopping on combined metric
        if combined > best_combined:
            best_combined = combined
            best_eff_rho = eff_rho
            best_disc_r = disc_r
            patience_counter = 0

            # Save checkpoint with full metadata
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_eff_rho": eff_rho,
                    "val_disc_r": disc_r,
                    "val_disc_rho": disc_rho,
                    "combined_metric": combined,
                    "val_loss": val_loss,
                    "phase": "multitask_full",
                    "config": {
                        "lr_shared": args.lr_shared,
                        "lr_disc": args.lr_disc,
                        "lambda_disc": args.lambda_disc,
                        "lambda_rank": args.lambda_rank,
                        "batch_size": args.batch_size,
                        "epochs": n_epochs,
                        "patience": args.patience,
                        "label_noise_sigma": 0.02,
                    },
                    "architecture": {
                        "use_rnafm": True,
                        "use_rloop_attention": True,
                        "multitask": True,
                        "cnn_branches": 32,
                        "cnn_out_dim": 64,
                        "rnafm_proj_dim": 64,
                        "fused_dim": 128,
                        "hidden_dim": 64,
                    },
                    "data": {
                        "efficiency_train": len(seqs_train),
                        "efficiency_val": len(seqs_val),
                        "disc_train": len(disc_train),
                        "disc_val": len(disc_val),
                        "disc_test": len(disc_test),
                    },
                    "params": {
                        "total": model.count_trainable_params(),
                        "shared_encoder": shared_params,
                        "disc_head": disc_params,
                    },
                },
                output_path,
            )
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or patience_counter == 0:
            logger.info(
                "Epoch %3d | Loss: %.4f (eff=%.4f, disc=%.4f) | "
                "Val eff_rho: %.4f | disc_r: %.4f | disc_rho: %.4f | "
                "Combined: %.4f | Best: %.4f | LR: %.1e/%.1e",
                epoch + 1, train_loss, train_eff, train_disc,
                eff_rho, disc_r, disc_rho,
                combined, best_combined, lr_shared, lr_disc,
            )

        if patience_counter >= args.patience:
            logger.info(
                "Early stopping at epoch %d. Best combined = %.4f "
                "(eff_rho=%.4f, disc_r=%.4f)",
                epoch + 1, best_combined, best_eff_rho, best_disc_r,
            )
            break

    # =====================================================================
    # Load best model and evaluate
    # =====================================================================
    best_ckpt = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    logger.info(
        "Loaded best model from epoch %d (eff_rho=%.4f, disc_r=%.4f)",
        best_ckpt["epoch"] + 1,
        best_ckpt["val_eff_rho"],
        best_ckpt["val_disc_r"],
    )

    # Efficiency evaluation
    val_metrics = evaluate(model, eff_val_loader, device, cache, use_rnafm=True)
    test_metrics = evaluate(model, eff_test_loader, device, cache, use_rnafm=True)

    # Discrimination evaluation on test set
    _, _, test_disc_r, test_disc_rho = validate_multitask(
        model, eff_test_loader, disc_test_loader, loss_fn, device, cache,
    )

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("Compass-ML Multi-Task Training Results")
    print("=" * 70)
    print(f"Config:         CNN + RNA-FM + RLPA + Multi-task (real disc)")
    print(f"Total params:   {model.count_trainable_params():,}")
    print(f"  Shared:       {shared_params:,}")
    print(f"  Disc head:    {disc_params:,}")
    print(f"")
    print(f"Efficiency:")
    print(f"  Val rho:      {val_metrics['spearman_rho']:.4f}")
    print(f"  Test rho:     {test_metrics['spearman_rho']:.4f}")
    print(f"  Test r:       {test_metrics['pearson_r']:.4f}")
    print(f"  Test MSE:     {test_metrics['mse']:.4f}")
    print(f"  Top-20%:      {test_metrics['top_k_precision']:.3f}")
    print(f"")
    print(f"Discrimination:")
    print(f"  Val disc_r:   {best_disc_r:.4f}")
    print(f"  Test disc_r:  {test_disc_r:.4f}")
    print(f"  Test disc_rho:{test_disc_rho:.4f}")
    print(f"")
    print(f"Combined:       {best_combined:.4f} (eff_rho + 0.5*disc_r)")
    print(f"Best epoch:     {best_ckpt['epoch'] + 1}")
    print(f"Checkpoint:     {output_path}")
    print("=" * 70)

    # =====================================================================
    # Deployment checks
    # =====================================================================
    passed = True
    checks = []

    if val_metrics["spearman_rho"] >= 0.52:
        checks.append(f"  [PASS] eff_rho >= 0.52: {val_metrics['spearman_rho']:.4f}")
    else:
        checks.append(f"  [FAIL] eff_rho >= 0.52: {val_metrics['spearman_rho']:.4f}")
        passed = False

    if best_disc_r > 0:
        checks.append(f"  [PASS] disc_r > 0: {best_disc_r:.4f}")
    else:
        checks.append(f"  [FAIL] disc_r > 0: {best_disc_r:.4f}")
        passed = False

    has_disc = any("disc_head" in k for k in best_ckpt["model_state_dict"].keys())
    if has_disc:
        checks.append(f"  [PASS] disc_head present in checkpoint")
    else:
        checks.append(f"  [FAIL] disc_head NOT in checkpoint")
        passed = False

    print("\nDeployment Checks:")
    for c in checks:
        print(c)
    print(f"\n{'READY FOR DEPLOYMENT' if passed else 'DEPLOYMENT BLOCKED'}")

    # Save results JSON
    results = {
        "config": "CNN + RNA-FM + RLPA + Multi-task (real disc)",
        "params": {
            "total": model.count_trainable_params(),
            "shared": shared_params,
            "disc_head": disc_params,
        },
        "val_efficiency": val_metrics,
        "test_efficiency": test_metrics,
        "val_disc_r": best_disc_r,
        "test_disc_r": test_disc_r,
        "test_disc_rho": test_disc_rho,
        "combined_metric": best_combined,
        "best_epoch": best_ckpt["epoch"] + 1,
        "deployment_passed": passed,
        "seed": args.seed,
    }
    results_path = output_path.replace(".pt", "_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
