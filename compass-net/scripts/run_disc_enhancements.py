"""Discrimination head enhancements A→B→C→D.

Each enhancement builds on the previous checkpoint:
  A: Contrastive discrimination loss (from multitask baseline)
  B: Thermodynamic feature injection (from A)
  C: Position-aware mismatch embedding (from B)
  D: Synthetic pair augmentation (from C)

Usage (from compass/ root):
    python compass-net/scripts/run_disc_enhancements.py --enhancement A --device cuda
    python compass-net/scripts/run_disc_enhancements.py --enhancement B --device cuda
    python compass-net/scripts/run_disc_enhancements.py --enhancement C --device cuda
    python compass-net/scripts/run_disc_enhancements.py --enhancement D --device cuda
    python compass-net/scripts/run_disc_enhancements.py --enhancement ALL --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import random
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMPASS_NET_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT_DIR = os.path.dirname(_COMPASS_NET_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _COMPASS_NET_DIR)

from run_phase1 import _setup, load_kim2018_sequences, evaluate
from scripts.run_multitask_full import (
    DiscPairDataset, collate_disc_pairs, split_disc_pairs,
    _one_hot, _reverse_complement,
    validate_multitask,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enhanced disc pair dataset (adds spacer_position and thermo_feats)
# ---------------------------------------------------------------------------

class EnhancedDiscPairDataset(Dataset):
    """Discrimination pairs with spacer position and optional thermo features."""

    def __init__(self, pairs, thermo_data=None):
        self.pairs = pairs
        # Build index from guide_id -> thermo features
        self.thermo_map = {}
        if thermo_data is not None:
            for i, pid in enumerate(thermo_data["pair_ids"]):
                self.thermo_map[pid] = thermo_data["features"][i]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        mut_target = p.mut_target
        wt_target = p.wt_target
        protospacer = mut_target[4:24] if len(mut_target) >= 24 else mut_target[4:]
        crrna_spacer = _reverse_complement(protospacer).replace("T", "U")

        item = {
            "mut_target_onehot": _one_hot(mut_target),
            "wt_target_onehot": _one_hot(wt_target),
            "crrna_spacer": crrna_spacer,
            "disc_ratio": torch.tensor(p.ratio_linear, dtype=torch.float32),
            "mut_efficiency": torch.tensor(p.mut_activity, dtype=torch.float32),
            "spacer_position": torch.tensor(p.spacer_position, dtype=torch.long),
        }

        if p.guide_id in self.thermo_map:
            item["thermo_feats"] = self.thermo_map[p.guide_id]
        else:
            item["thermo_feats"] = torch.zeros(3, dtype=torch.float32)

        return item


def collate_enhanced(batch):
    return {
        "mut_target_onehot": torch.stack([b["mut_target_onehot"] for b in batch]),
        "wt_target_onehot": torch.stack([b["wt_target_onehot"] for b in batch]),
        "crrna_spacer": [b["crrna_spacer"] for b in batch],
        "disc_ratio": torch.stack([b["disc_ratio"] for b in batch]),
        "mut_efficiency": torch.stack([b["mut_efficiency"] for b in batch]),
        "spacer_position": torch.stack([b["spacer_position"] for b in batch]),
        "thermo_feats": torch.stack([b["thermo_feats"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Mixed dataset for Enhancement D (real + synthetic pairs)
# ---------------------------------------------------------------------------

class MixedDiscDataset(Dataset):
    """Combines real EasyDesign pairs with synthetic self-distillation pairs."""

    def __init__(self, real_dataset, synthetic_path, thermo_data=None):
        self.real = real_dataset
        self.syn = torch.load(synthetic_path, map_location="cpu", weights_only=False)
        self.n_real = len(real_dataset)
        self.n_syn = len(self.syn["ratios"])
        self.thermo_data = thermo_data
        logger.info("Mixed dataset: %d real + %d synthetic", self.n_real, self.n_syn)

    def __len__(self):
        return self.n_real * 2  # 2x to see all real + synthetic samples

    def __getitem__(self, idx):
        if idx < self.n_real:
            item = self.real[idx]
            item["weight"] = torch.tensor(1.0)
            return item
        else:
            # Random synthetic sample
            si = torch.randint(0, self.n_syn, (1,)).item()
            # Synthetic data stored as strings — one-hot encode on the fly
            mut_seq = self.syn["mut_seqs"][si]
            wt_seq = self.syn["wt_seqs"][si]
            item = {
                "mut_target_onehot": _one_hot(mut_seq),
                "wt_target_onehot": _one_hot(wt_seq),
                "crrna_spacer": "",  # Not used for synthetic
                "disc_ratio": self.syn["ratios"][si],
                "mut_efficiency": torch.tensor(0.5),
                "spacer_position": self.syn["mm_positions"][si],
                "thermo_feats": torch.zeros(3),
                "weight": torch.tensor(0.1),
            }
            return item


def collate_mixed(batch):
    result = collate_enhanced(batch)
    result["weight"] = torch.stack([b["weight"] for b in batch])
    return result


# ---------------------------------------------------------------------------
# Training loop for enhancements
# ---------------------------------------------------------------------------

def train_enhancement_epoch(
    model, eff_loader, disc_loader, optimizer, device,
    embedding_cache, contrastive_loss, lambda_disc=0.3,
    use_thermo=False, use_position=False, use_weights=False,
):
    """Train one epoch with contrastive disc loss and optional enhanced features."""
    from compass_ml.training.train_compass_ml import _get_batch_embeddings
    from compass_ml.losses.multitask_loss import MultiTaskLoss

    model.train()
    total_eff_loss = 0.0
    total_disc_loss = 0.0
    n_eff = 0
    n_disc = 0

    loss_fn = MultiTaskLoss(lambda_disc=0, lambda_rank=0.5)

    disc_iter = iter(disc_loader)

    for eff_batch in eff_loader:
        # --- Efficiency batch ---
        target_onehot = eff_batch["target_onehot"].to(device)
        efficiency = eff_batch["efficiency"].to(device)

        crrna_emb = None
        if embedding_cache is not None:
            crrna_emb = _get_batch_embeddings(
                eff_batch["crrna_spacer"], embedding_cache, device,
            )

        output = model(target_onehot=target_onehot, crrna_rnafm_emb=crrna_emb)
        losses = loss_fn(pred_eff=output["efficiency"], true_eff=efficiency)

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_eff_loss += losses["efficiency"].item()
        n_eff += 1

        # --- Discrimination batch ---
        try:
            disc_batch = next(disc_iter)
        except StopIteration:
            disc_iter = iter(disc_loader)
            disc_batch = next(disc_iter)

        mut_oh = disc_batch["mut_target_onehot"].to(device)
        wt_oh = disc_batch["wt_target_onehot"].to(device)
        disc_ratio = disc_batch["disc_ratio"].to(device)

        disc_crrna = None
        if embedding_cache is not None:
            # Always provide RNA-FM embeddings (zeros for empty/synthetic spacers)
            disc_crrna = _get_batch_embeddings(
                disc_batch["crrna_spacer"], embedding_cache, device,
            )

        # Build optional enhanced features
        thermo = disc_batch.get("thermo_feats")
        if use_thermo and thermo is not None:
            thermo = thermo.to(device)
        else:
            thermo = None

        mm_pos = disc_batch.get("spacer_position")
        if use_position and mm_pos is not None:
            mm_pos = mm_pos.to(device)
        else:
            mm_pos = None

        disc_output = model(
            target_onehot=mut_oh,
            crrna_rnafm_emb=disc_crrna,
            wt_target_onehot=wt_oh,
            thermo_feats=thermo,
            mm_position=mm_pos,
        )

        if "discrimination" in disc_output:
            pred_disc = disc_output["discrimination"].squeeze(-1)
            z_mut = disc_output["z_mut"]
            z_wt = disc_output["z_wt"]

            weights = disc_batch.get("weight")
            if use_weights and weights is not None:
                weights = weights.to(device)
            else:
                weights = None

            l_disc = contrastive_loss(pred_disc, disc_ratio, z_mut, z_wt, weights)
            disc_loss = lambda_disc * l_disc

            optimizer.zero_grad()
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_disc_loss += l_disc.item()
            n_disc += 1

    avg_eff = total_eff_loss / max(n_eff, 1)
    avg_disc = total_disc_loss / max(n_disc, 1)
    return avg_eff, avg_disc


def validate_disc(model, disc_loader, device, embedding_cache, use_thermo=False, use_position=False):
    """Validate discrimination only, returning Pearson r and Spearman rho."""
    from compass_ml.training.train_compass_ml import _get_batch_embeddings

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in disc_loader:
            mut_oh = batch["mut_target_onehot"].to(device)
            wt_oh = batch["wt_target_onehot"].to(device)
            disc_ratio = batch["disc_ratio"]

            crrna_emb = None
            if embedding_cache is not None and batch["crrna_spacer"][0]:
                crrna_emb = _get_batch_embeddings(
                    batch["crrna_spacer"], embedding_cache, device,
                )

            thermo = batch.get("thermo_feats")
            if use_thermo and thermo is not None:
                thermo = thermo.to(device)
            else:
                thermo = None

            mm_pos = batch.get("spacer_position")
            if use_position and mm_pos is not None:
                mm_pos = mm_pos.to(device)
            else:
                mm_pos = None

            output = model(
                target_onehot=mut_oh, crrna_rnafm_emb=crrna_emb,
                wt_target_onehot=wt_oh, thermo_feats=thermo, mm_position=mm_pos,
            )

            if "discrimination" in output:
                all_preds.extend(output["discrimination"].squeeze(-1).cpu().tolist())
                all_targets.extend(disc_ratio.tolist())

    if len(all_preds) < 3:
        return 0.0, 0.0

    r, _ = pearsonr(all_preds, all_targets)
    rho, _ = spearmanr(all_preds, all_targets)
    return float(r) if not np.isnan(r) else 0.0, float(rho) if not np.isnan(rho) else 0.0


# ---------------------------------------------------------------------------
# Build model from checkpoint with correct architecture
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(ckpt_path, device, n_thermo=0, pos_embed_dim=0):
    """Load model, auto-detecting architecture from checkpoint."""
    from compass_ml.compass_ml import CompassML

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    has_disc = any("disc_head" in k for k in state_dict)

    # Detect existing thermo/position dims from checkpoint
    ckpt_n_thermo = 0
    ckpt_pos_dim = 0
    for k in state_dict:
        if "pos_embedding.weight" in k:
            ckpt_pos_dim = state_dict[k].shape[1]
        if "disc_head.head.0.weight" in k:
            input_dim = state_dict[k].shape[1]
            base = 512
            # Will subtract pos_dim after we know it

    if "disc_head.head.0.weight" in {k for k in state_dict}:
        input_dim = state_dict["disc_head.head.0.weight"].shape[1]
        ckpt_n_thermo = input_dim - 512 - ckpt_pos_dim

    # Use max of checkpoint and requested
    actual_thermo = max(n_thermo, max(ckpt_n_thermo, 0))
    actual_pos = max(pos_embed_dim, ckpt_pos_dim)

    model = CompassML(
        use_rnafm=True, use_rloop_attention=True,
        multitask=True, n_thermo=actual_thermo, pos_embed_dim=actual_pos,
    )

    # Transfer weights (strict=False to handle new layers)
    model_sd = model.state_dict()
    transferred = 0
    new_keys = []
    for name, param in model_sd.items():
        if name in state_dict and state_dict[name].shape == param.shape:
            model_sd[name] = state_dict[name]
            transferred += 1
        else:
            new_keys.append(name)

    model.load_state_dict(model_sd)
    model.to(device)

    logger.info("Loaded %d/%d weights from %s", transferred, len(model_sd), ckpt_path)
    if new_keys:
        logger.info("New layers (randomly init): %s", new_keys)

    return model, ckpt


def save_checkpoint(model, optimizer, epoch, metrics, config, output_path):
    """Save checkpoint with full metadata."""
    disc_params = sum(p.numel() for n, p in model.named_parameters() if "disc_head" in n)
    total_params = model.count_trainable_params()

    # Detect disc head input dim
    disc_input_dim = 0
    for k, v in model.state_dict().items():
        if "disc_head.head.0.weight" in k:
            disc_input_dim = v.shape[1]
            break

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": {
            "enhancement": config["enhancement"],
            "phase": "discrimination_enhancement",
            "epoch": epoch,
            "val_eff_rho": metrics["eff_rho"],
            "val_disc_r": metrics["disc_r"],
            "val_disc_rho": metrics["disc_rho"],
            "combined_metric": metrics["combined"],
            "lambda_disc": config.get("lambda_disc", 0.3),
            "lambda_rank": config.get("lambda_rank", 0.5),
            "total_params": total_params,
            "disc_head_params": disc_params,
            "disc_head_input_dim": disc_input_dim,
            "n_thermo": config.get("n_thermo", 0),
            "pos_embed_dim": config.get("pos_embed_dim", 0),
            "margin": config.get("margin", 0.5),
            "alpha": config.get("alpha", 0.6),
            "thermo_norm_mean": config.get("thermo_norm_mean", []),
            "thermo_norm_std": config.get("thermo_norm_std", []),
            "synthetic_pairs_count": config.get("synthetic_pairs_count", 0),
            "synthetic_weight": config.get("synthetic_weight", 0.0),
            "training_data": config.get("training_data", ""),
            "enhancements_applied": config.get("enhancements_applied", []),
            "parent_checkpoint": config.get("parent_checkpoint", ""),
        },
        # Also store at top level for backward compat
        "val_eff_rho": metrics["eff_rho"],
        "val_disc_r": metrics["disc_r"],
        "epoch": epoch,
        "architecture": {
            "use_rnafm": True,
            "use_rloop_attention": True,
            "multitask": True,
            "cnn_branches": 32,
            "cnn_out_dim": 64,
            "rnafm_proj_dim": 64,
            "fused_dim": 128,
            "hidden_dim": 64,
            "n_thermo": config.get("n_thermo", 0),
            "pos_embed_dim": config.get("pos_embed_dim", 0),
        },
    }, output_path)


# ---------------------------------------------------------------------------
# Run a single enhancement
# ---------------------------------------------------------------------------

def run_enhancement(
    enhancement: str,
    parent_ckpt: str,
    output_path: str,
    device: torch.device,
    n_epochs: int,
    patience: int,
    lr: float,
    lr_disc: float,
    lambda_disc: float,
    margin: float,
    alpha: float,
    n_thermo: int,
    pos_embed_dim: int,
    batch_size: int,
    data_path: str,
    cache_dir: str,
    seed: int,
    synthetic_path: str | None = None,
    thermo_data_path: str | None = None,
):
    """Run one enhancement training phase."""
    from compass_ml.data.embedding_cache import EmbeddingCache
    from compass_ml.data.extract_discrimination_pairs import extract_discrimination_pairs
    from compass_ml.training.train_compass_ml import collate_single_target
    from compass_ml.losses.contrastive_disc_loss import ContrastiveDiscriminationLoss
    from compass_ml.training.reproducibility import seed_everything

    seed_everything(seed)

    logger.info("=" * 70)
    logger.info("Enhancement %s", enhancement)
    logger.info("  Parent: %s", parent_ckpt)
    logger.info("  Output: %s", output_path)
    logger.info("  n_thermo=%d, pos_embed_dim=%d", n_thermo, pos_embed_dim)
    logger.info("=" * 70)

    # Load data
    (seqs_train, y_train), (seqs_val, y_val), (seqs_test, y_test) = \
        load_kim2018_sequences(os.path.join(_ROOT_DIR, data_path))

    disc_pairs = extract_discrimination_pairs(
        os.path.join(_ROOT_DIR, "compass-net/data/external/easydesign/Table_S2.xlsx")
    )
    disc_train, disc_val, disc_test = split_disc_pairs(disc_pairs, seed=seed)

    # Load thermo data if needed
    thermo_data = None
    thermo_norm_mean = []
    thermo_norm_std = []
    if n_thermo > 0 and thermo_data_path:
        td_path = os.path.join(_ROOT_DIR, thermo_data_path)
        if os.path.exists(td_path):
            thermo_data = torch.load(td_path, map_location="cpu", weights_only=False)
            thermo_norm_mean = thermo_data["mean"].tolist()
            thermo_norm_std = thermo_data["std"].tolist()
            logger.info("Loaded thermo features from %s", td_path)
        else:
            logger.warning("Thermo data not found at %s — using zeros", td_path)

    # Create datasets
    from compass_ml.data.paired_loader import SingleTargetDataset
    eff_train_ds = SingleTargetDataset(seqs_train, y_train.tolist())
    eff_val_ds = SingleTargetDataset(seqs_val, y_val.tolist())

    disc_train_ds = EnhancedDiscPairDataset(disc_train, thermo_data)
    disc_val_ds = EnhancedDiscPairDataset(disc_val, thermo_data)
    disc_test_ds = EnhancedDiscPairDataset(disc_test, thermo_data)

    # Handle Enhancement D: mixed real + synthetic
    use_weights = False
    synthetic_count = 0
    if enhancement == "D" and synthetic_path:
        syn_path = os.path.join(_ROOT_DIR, synthetic_path)
        if os.path.exists(syn_path):
            disc_train_ds = MixedDiscDataset(disc_train_ds, syn_path, thermo_data)
            synthetic_count = disc_train_ds.n_syn
            use_weights = True
            logger.info("Using mixed dataset with %d synthetic pairs", synthetic_count)
            disc_collate = collate_mixed
        else:
            logger.warning("Synthetic data not found at %s — using real only", syn_path)
            disc_collate = collate_enhanced
    else:
        disc_collate = collate_enhanced

    eff_train_loader = DataLoader(
        eff_train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, collate_fn=collate_single_target,
    )
    eff_val_loader = DataLoader(
        eff_val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_single_target,
    )

    disc_train_loader = DataLoader(
        disc_train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, collate_fn=disc_collate,
    )
    disc_val_loader = DataLoader(
        disc_val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_enhanced,
    )
    disc_test_loader = DataLoader(
        disc_test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_enhanced,
    )

    # Embedding cache
    cache = EmbeddingCache(os.path.join(_ROOT_DIR, cache_dir))

    # Load model
    model, parent_data = load_model_from_checkpoint(
        os.path.join(_ROOT_DIR, parent_ckpt), device,
        n_thermo=n_thermo, pos_embed_dim=pos_embed_dim,
    )

    disc_params_list = [p for n, p in model.named_parameters() if "disc_head" in n]
    other_params = [p for n, p in model.named_parameters() if "disc_head" not in n]

    optimizer = AdamW([
        {"params": other_params, "lr": lr},
        {"params": disc_params_list, "lr": lr_disc},
    ], weight_decay=1e-3)

    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    contrastive_loss = ContrastiveDiscriminationLoss(margin=margin, alpha=alpha)

    use_thermo = n_thermo > 0
    use_position = pos_embed_dim > 0

    # Training loop
    best_combined = -float("inf")
    best_metrics = {}
    patience_counter = 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for epoch in range(n_epochs):
        avg_eff, avg_disc = train_enhancement_epoch(
            model, eff_train_loader, disc_train_loader,
            optimizer, device, cache, contrastive_loss,
            lambda_disc=lambda_disc,
            use_thermo=use_thermo, use_position=use_position,
            use_weights=use_weights,
        )
        scheduler.step()

        # Validate efficiency
        val_metrics = evaluate(model, eff_val_loader, device, cache, use_rnafm=True)
        eff_rho = val_metrics["spearman_rho"]

        # Validate discrimination
        disc_r, disc_rho = validate_disc(
            model, disc_val_loader, device, cache,
            use_thermo=use_thermo, use_position=use_position,
        )

        combined = eff_rho + 0.5 * disc_r
        lr_now = scheduler.get_last_lr()[0]

        if combined > best_combined:
            best_combined = combined
            best_metrics = {
                "eff_rho": eff_rho, "disc_r": disc_r,
                "disc_rho": disc_rho, "combined": combined,
            }
            patience_counter = 0

            enhancements_so_far = []
            if enhancement == "A":
                enhancements_so_far = ["A"]
            elif enhancement == "B":
                enhancements_so_far = ["A", "B"]
            elif enhancement == "C":
                enhancements_so_far = ["A", "B", "C"]
            elif enhancement == "D":
                enhancements_so_far = ["A", "B", "C", "D"]

            save_checkpoint(model, optimizer, epoch, best_metrics, {
                "enhancement": enhancement,
                "lambda_disc": lambda_disc,
                "lambda_rank": 0.5,
                "n_thermo": n_thermo,
                "pos_embed_dim": pos_embed_dim,
                "margin": margin,
                "alpha": alpha,
                "thermo_norm_mean": thermo_norm_mean,
                "thermo_norm_std": thermo_norm_std,
                "synthetic_pairs_count": synthetic_count,
                "synthetic_weight": 0.1 if use_weights else 0.0,
                "training_data": "Kim2018+EasyDesign" + ("+synthetic" if synthetic_count > 0 else ""),
                "enhancements_applied": enhancements_so_far,
                "parent_checkpoint": parent_ckpt,
            }, output_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or patience_counter == 0:
            logger.info(
                "Enh %s | Epoch %3d | eff=%.4f disc=%.4f | "
                "eff_rho=%.4f disc_r=%.4f disc_rho=%.4f | "
                "Combined=%.4f Best=%.4f | LR=%.1e",
                enhancement, epoch + 1, avg_eff, avg_disc,
                eff_rho, disc_r, disc_rho,
                combined, best_combined, lr_now,
            )

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    logger.info(
        "Enhancement %s done: eff_rho=%.4f, disc_r=%.4f, disc_rho=%.4f, combined=%.4f",
        enhancement, best_metrics["eff_rho"], best_metrics["disc_r"],
        best_metrics["disc_rho"], best_metrics["combined"],
    )

    # Test set evaluation
    best_ckpt = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_disc_r, test_disc_rho = validate_disc(
        model, disc_test_loader, device, cache,
        use_thermo=use_thermo, use_position=use_position,
    )
    logger.info("Test disc_r=%.4f, disc_rho=%.4f", test_disc_r, test_disc_rho)

    return best_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Discrimination enhancements A→D")
    parser.add_argument("--enhancement", type=str, default="ALL",
                        choices=["A", "B", "C", "D", "ALL"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data", type=str,
                        default="compass/data/kim2018/nbt4061_source_data.xlsx")
    parser.add_argument("--cache-dir", type=str, default="compass-net/cache/rnafm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    _setup()
    device = torch.device(args.device)

    # Read baseline metrics
    baseline_path = os.path.join(
        _ROOT_DIR, "compass-net/checkpoints/multitask/compass_ml_multitask_best.pt"
    )
    if not os.path.exists(baseline_path):
        logger.error("Multitask baseline not found at %s", baseline_path)
        return 1

    baseline_ckpt = torch.load(baseline_path, map_location="cpu", weights_only=False)
    baseline_disc_r = baseline_ckpt.get("val_disc_r", 0.0)
    baseline_eff_rho = baseline_ckpt.get("val_eff_rho", 0.0)
    logger.info("Baseline: eff_rho=%.4f, disc_r=%.4f", baseline_eff_rho, baseline_disc_r)

    results = {
        "Baseline MT": {"disc_r": baseline_disc_r, "eff_rho": baseline_eff_rho},
    }

    enhancements = ["A", "B", "C", "D"] if args.enhancement == "ALL" else [args.enhancement]

    for enh in enhancements:
        if enh == "A":
            # Precompute thermo if B will follow
            if "B" in enhancements:
                thermo_path = os.path.join(_ROOT_DIR, "compass-net/data/disc_thermo_features.pt")
                if not os.path.exists(thermo_path):
                    logger.info("Precomputing thermo features...")
                    from scripts.precompute_disc_thermo import main as precompute_main
                    precompute_main()

            metrics = run_enhancement(
                enhancement="A",
                parent_ckpt="compass-net/checkpoints/multitask/compass_ml_multitask_best.pt",
                output_path=os.path.join(_ROOT_DIR, "compass-net/checkpoints/enhancements/A_contrastive_best.pt"),
                device=device, n_epochs=60, patience=15,
                lr=5e-5, lr_disc=5e-5, lambda_disc=0.3,
                margin=0.5, alpha=0.6,
                n_thermo=0, pos_embed_dim=0,
                batch_size=128, data_path=args.data,
                cache_dir=args.cache_dir, seed=args.seed,
            )
            results["A: Contrastive"] = metrics

            # Fallback: if eff_rho dropped too much
            if metrics["eff_rho"] < 0.50:
                logger.warning("Eff rho dropped to %.4f — retrying with lambda_disc=0.15",
                               metrics["eff_rho"])
                metrics = run_enhancement(
                    enhancement="A",
                    parent_ckpt="compass-net/checkpoints/multitask/compass_ml_multitask_best.pt",
                    output_path=os.path.join(_ROOT_DIR, "compass-net/checkpoints/enhancements/A_contrastive_best.pt"),
                    device=device, n_epochs=60, patience=15,
                    lr=5e-5, lr_disc=5e-5, lambda_disc=0.15,
                    margin=0.3, alpha=0.6,
                    n_thermo=0, pos_embed_dim=0,
                    batch_size=128, data_path=args.data,
                    cache_dir=args.cache_dir, seed=args.seed,
                )
                results["A: Contrastive"] = metrics

        elif enh == "B":
            metrics = run_enhancement(
                enhancement="B",
                parent_ckpt="compass-net/checkpoints/enhancements/A_contrastive_best.pt",
                output_path=os.path.join(_ROOT_DIR, "compass-net/checkpoints/enhancements/B_thermo_best.pt"),
                device=device, n_epochs=60, patience=15,
                lr=5e-5, lr_disc=5e-4, lambda_disc=0.3,
                margin=0.5, alpha=0.6,
                n_thermo=3, pos_embed_dim=0,
                batch_size=128, data_path=args.data,
                cache_dir=args.cache_dir, seed=args.seed,
                thermo_data_path="compass-net/data/disc_thermo_features.pt",
            )
            results["B: Thermo"] = metrics

        elif enh == "C":
            metrics = run_enhancement(
                enhancement="C",
                parent_ckpt="compass-net/checkpoints/enhancements/B_thermo_best.pt",
                output_path=os.path.join(_ROOT_DIR, "compass-net/checkpoints/enhancements/C_position_best.pt"),
                device=device, n_epochs=60, patience=15,
                lr=5e-5, lr_disc=5e-4, lambda_disc=0.3,
                margin=0.5, alpha=0.6,
                n_thermo=3, pos_embed_dim=32,
                batch_size=128, data_path=args.data,
                cache_dir=args.cache_dir, seed=args.seed,
                thermo_data_path="compass-net/data/disc_thermo_features.pt",
            )
            results["C: Position"] = metrics

        elif enh == "D":
            # Generate synthetic pairs first if needed
            syn_path = os.path.join(_ROOT_DIR, "compass-net/data/synthetic_disc_pairs.pt")
            if not os.path.exists(syn_path):
                logger.info("Generating synthetic pairs...")
                os.system(f'python "{os.path.join(_SCRIPT_DIR, "generate_synthetic_pairs.py")}" '
                          f'--device {args.device}')

            metrics = run_enhancement(
                enhancement="D",
                parent_ckpt="compass-net/checkpoints/enhancements/C_position_best.pt",
                output_path=os.path.join(_ROOT_DIR, "compass-net/checkpoints/enhancements/D_augmented_best.pt"),
                device=device, n_epochs=40, patience=10,
                lr=3e-5, lr_disc=3e-5, lambda_disc=0.3,
                margin=0.5, alpha=0.6,
                n_thermo=3, pos_embed_dim=32,
                batch_size=256, data_path=args.data,
                cache_dir=args.cache_dir, seed=args.seed,
                thermo_data_path="compass-net/data/disc_thermo_features.pt",
                synthetic_path="compass-net/data/synthetic_disc_pairs.pt",
            )
            results["D: Augmented"] = metrics

    # Print summary table
    print("\n" + "=" * 75)
    print("DISCRIMINATION ENHANCEMENT RESULTS")
    print("=" * 75)
    print(f"{'Enhancement':<20} {'Disc r':>8} {'Eff rho':>8} {'Disc rho':>8} {'Delta r':>8}")
    print("-" * 75)

    prev_r = baseline_disc_r
    for name, m in results.items():
        disc_r = m.get("disc_r", m.get("disc_r", 0))
        eff_rho = m.get("eff_rho", m.get("eff_rho", 0))
        disc_rho = m.get("disc_rho", 0)
        delta = disc_r - prev_r if name != "Baseline MT" else 0
        print(f"{name:<20} {disc_r:>8.4f} {eff_rho:>8.4f} {disc_rho:>8.4f} {delta:>+8.4f}")
        if name != "Baseline MT":
            prev_r = disc_r

    total_delta = list(results.values())[-1].get("disc_r", 0) - baseline_disc_r
    print("-" * 75)
    print(f"{'Total improvement':<20} {'':>8} {'':>8} {'':>8} {total_delta:>+8.4f}")
    print("=" * 75)

    # Save results
    results_path = os.path.join(_ROOT_DIR, "compass-net/checkpoints/enhancements/enhancement_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
