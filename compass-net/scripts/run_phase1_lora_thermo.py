"""Phase 1 ablation row 5: CNN + RNA-FM (LoRA) + RLPA + thermodynamic dG.

Replaces frozen RNA-FM embeddings with live LoRA-adapted RNA-FM forward pass.
Adds 3 thermodynamic scalar features (folding dG, hybrid dG, melting Tm).

Key changes from row 3:
    - RNA-FM runs live with LoRA adapters on layers 10-11 (rank 4)
    - 3 scalar features appended after pooling
    - Two param groups: LoRA (lr=2e-4) and rest (lr=1e-4)
    - Batch size 64 (RNA-FM runs live, needs more VRAM)
    - Mixed precision (autocast) for RNA-FM forward pass

Usage (from compass/ root):
    python compass-net/scripts/run_phase1_lora_thermo.py --device cuda
"""

from __future__ import annotations

import sys, os, argparse, logging, json

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMPASS_NET_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT_DIR = os.path.dirname(_COMPASS_NET_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _COMPASS_NET_DIR)

from run_phase1 import _setup, load_kim2018_sequences

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compass-ML Row 5: LoRA + thermo")
    parser.add_argument("--data", type=str, default="compass/data/kim2018/nbt4061_source_data.xlsx")
    parser.add_argument("--thermo-cache", type=str, default="E:/compass-net-data/cache/thermo_features.pt")
    parser.add_argument("--pretrained", type=str, default="E:/compass-net-data/weights/phase1_rlpa_best.pt")
    parser.add_argument("--output", type=str, default="E:/compass-net-data/weights/phase1_lora_thermo_best.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--lr-rest", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    _setup()

    from compass_ml.compass_ml import CompassML
    from compass_ml.data.paired_loader import SingleTargetDataset, _reverse_complement
    from compass_ml.training.reproducibility import seed_everything
    from torch.utils.data import DataLoader

    seed_everything(args.seed)
    device = torch.device(args.device)
    logger.info("Device: %s", device)

    # --- Load data ---
    logger.info("Loading Kim 2018 data...")
    (seqs_train, y_train), (seqs_val, y_val), (seqs_test, y_test) = load_kim2018_sequences(args.data)
    logger.info("Loaded: train=%d, val=%d, test=%d", len(seqs_train), len(seqs_val), len(seqs_test))

    # --- Load thermodynamic features ---
    thermo_data = torch.load(args.thermo_cache, map_location="cpu", weights_only=False)
    thermo_features = thermo_data["features"]  # (N, 3) normalized
    n_train = thermo_data["n_train"]
    n_val = thermo_data["n_val"]
    thermo_train = thermo_features[:n_train]
    thermo_val = thermo_features[n_train:n_train + n_val]
    thermo_test = thermo_features[n_train + n_val:]
    logger.info("Thermodynamic features: %d train, %d val, %d test",
                len(thermo_train), len(thermo_val), len(thermo_test))

    # --- Custom dataset that returns crRNA sequences + thermo features ---
    class ThermoDataset(torch.utils.data.Dataset):
        def __init__(self, targets, activities, thermo):
            self.targets = targets
            self.activities = activities
            self.thermo = thermo

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            from compass_ml.data.paired_loader import _one_hot
            target = self.targets[idx]
            protospacer = target[4:24] if len(target) >= 24 else target
            crrna = _reverse_complement(protospacer).replace("T", "U")
            return {
                "target_onehot": _one_hot(target),
                "crrna_sequence": crrna,
                "efficiency": torch.tensor(self.activities[idx], dtype=torch.float32),
                "thermo": self.thermo[idx],
            }

    def collate_thermo(batch):
        return {
            "target_onehot": torch.stack([b["target_onehot"] for b in batch]),
            "crrna_sequences": [b["crrna_sequence"] for b in batch],
            "efficiency": torch.stack([b["efficiency"] for b in batch]),
            "thermo": torch.stack([b["thermo"] for b in batch]),
        }

    train_ds = ThermoDataset(seqs_train, y_train.tolist(), thermo_train)
    val_ds = ThermoDataset(seqs_val, y_val.tolist(), thermo_val)
    test_ds = ThermoDataset(seqs_test, y_test.tolist(), thermo_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, collate_fn=collate_thermo)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_thermo)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_thermo)

    # --- Build model ---
    model = CompassML(
        use_rnafm=True,
        use_rnafm_lora=True,
        use_rloop_attention=True,
        multitask=False,
        n_scalar_features=3,
        lora_rank=4,
        lora_alpha=8,
    )

    # Transfer CNN + RLPA weights from phase1_rlpa checkpoint
    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    pretrained_state = ckpt["model_state_dict"]
    model_state = model.state_dict()

    transferred = 0
    skipped = []
    for name, param in model_state.items():
        if name in pretrained_state and pretrained_state[name].shape == param.shape:
            model_state[name] = pretrained_state[name]
            transferred += 1
        elif name.startswith("rnafm_lora") or name.startswith("efficiency_head"):
            # LoRA branch and new head are initialized fresh
            skipped.append(name)

    model.load_state_dict(model_state)
    logger.info("Weight transfer from %s: %d transferred, %d new (LoRA/head)",
                args.pretrained, transferred, len(skipped))

    model = model.to(device)
    total_params = model.count_trainable_params()
    logger.info("Trainable params: %d", total_params)

    # --- Two param groups ---
    lora_params = []
    rest_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "rnafm_lora" in name:
            lora_params.append(param)
        else:
            rest_params.append(param)

    logger.info("Param groups: LoRA=%d tensors, rest=%d tensors",
                len(lora_params), len(rest_params))

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": rest_params, "lr": args.lr_rest},
    ], weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.HuberLoss(delta=0.5)

    # --- Training loop ---
    best_rho = -1.0
    patience_counter = 0
    patience = 20

    logger.info("=" * 60)
    logger.info("Starting Row 5 training: CNN + RNA-FM (LoRA) + RLPA + thermo")
    logger.info("=" * 60)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            target_oh = batch["target_onehot"].to(device)
            crrna_seqs = batch["crrna_sequences"]
            efficiency = batch["efficiency"].to(device)
            thermo = batch["thermo"].to(device)

            # Label noise
            noise = torch.randn_like(efficiency) * 0.03
            efficiency_noisy = (efficiency + noise).clamp(0, 1)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                output = model(
                    target_onehot=target_oh,
                    crrna_sequences=crrna_seqs,
                    scalar_features=thermo,
                )
                loss = loss_fn(output["efficiency"].squeeze(-1), efficiency_noisy)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(n_batches, 1)

        # --- Validation ---
        model.eval()
        all_preds, all_labels = [], []
        val_loss_total = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                target_oh = batch["target_onehot"].to(device)
                crrna_seqs = batch["crrna_sequences"]
                efficiency = batch["efficiency"].to(device)
                thermo = batch["thermo"].to(device)

                with torch.amp.autocast("cuda"):
                    output = model(
                        target_onehot=target_oh,
                        crrna_sequences=crrna_seqs,
                        scalar_features=thermo,
                    )
                    loss = loss_fn(output["efficiency"].squeeze(-1), efficiency)

                val_loss_total += loss.item()
                n_val_batches += 1
                all_preds.extend(output["efficiency"].squeeze(-1).cpu().tolist())
                all_labels.extend(efficiency.cpu().tolist())

        avg_val_loss = val_loss_total / max(n_val_batches, 1)
        val_rho, _ = spearmanr(all_preds, all_labels)

        # Early stopping
        if val_rho > best_rho:
            best_rho = val_rho
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_rho": val_rho,
                "val_loss": avg_val_loss,
                "config": {
                    "use_rnafm_lora": True,
                    "use_rloop_attention": True,
                    "n_scalar_features": 3,
                    "lora_rank": 4,
                    "lr_lora": args.lr_lora,
                    "lr_rest": args.lr_rest,
                    "batch_size": args.batch_size,
                },
            }, args.output)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or patience_counter == 0 or epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                "Row 5 | Epoch %3d | Train: %.4f | Val: %.4f | "
                "Rho: %.4f | Best: %.4f | LR: %.2e",
                epoch + 1, avg_train_loss, avg_val_loss,
                val_rho, best_rho, current_lr,
            )

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d. Best rho = %.4f", epoch + 1, best_rho)
            break

    # --- Load best and evaluate ---
    best_ckpt = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    best_epoch = best_ckpt["epoch"] + 1
    logger.info("Loaded best model from epoch %d (rho = %.4f)", best_epoch, best_ckpt["val_rho"])

    logger.info("=" * 60)
    logger.info("Evaluation")
    logger.info("=" * 60)

    def evaluate_loader(loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                target_oh = batch["target_onehot"].to(device)
                crrna_seqs = batch["crrna_sequences"]
                thermo = batch["thermo"].to(device)

                with torch.amp.autocast("cuda"):
                    output = model(
                        target_onehot=target_oh,
                        crrna_sequences=crrna_seqs,
                        scalar_features=thermo,
                    )

                all_preds.extend(output["efficiency"].squeeze(-1).cpu().tolist())
                all_labels.extend(batch["efficiency"].tolist())

        preds = np.array(all_preds)
        labels = np.array(all_labels)
        rho, _ = spearmanr(preds, labels)
        r, _ = pearsonr(preds, labels)
        mse = mean_squared_error(labels, preds)
        mae = mean_absolute_error(labels, preds)

        # Top-20% precision
        k = max(1, len(labels) // 5)
        top_pred_idx = np.argsort(preds)[-k:]
        top_true_idx = set(np.argsort(labels)[-k:])
        top_k_prec = len(set(top_pred_idx) & top_true_idx) / k

        return {
            "spearman_rho": round(float(rho), 4),
            "pearson_r": round(float(r), 4),
            "mse": round(float(mse), 4),
            "mae": round(float(mae), 4),
            "top_k_precision": round(float(top_k_prec), 4),
        }

    val_metrics = evaluate_loader(val_loader)
    test_metrics = evaluate_loader(test_loader)

    logger.info("Validation metrics:")
    for k, v in val_metrics.items():
        logger.info("  %-20s %.4f", k, v)
    logger.info("Test metrics (HT2 + HT3):")
    for k, v in test_metrics.items():
        logger.info("  %-20s %.4f", k, v)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Compass-ML Row 5: LoRA + Thermo Results")
    print("=" * 60)
    print(f"Config:     CNN + RNA-FM (LoRA) + RLPA + thermo dG")
    print(f"Params:     {total_params:,}")
    print(f"Val rho:    {val_metrics['spearman_rho']:.4f}")
    print(f"Test rho:   {test_metrics['spearman_rho']:.4f}")
    print(f"Test r:     {test_metrics['pearson_r']:.4f}")
    print(f"Test MSE:   {test_metrics['mse']:.4f}")
    print(f"Top-20%%:   {test_metrics['top_k_precision']:.3f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val:   {best_ckpt['val_rho']:.4f}")
    print()
    print("Comparison to previous rows:")
    print(f"  CNN only:              test rho = 0.4959")
    print(f"  CNN + RNA-FM:          test rho = 0.5009")
    print(f"  CNN + RNA-FM + RLPA:   test rho = 0.5336")
    print(f"  + Multi-task:          test rho = 0.5361")
    print(f"  + LoRA + thermo:       test rho = {test_metrics['spearman_rho']:.4f}")
    print("=" * 60)

    # Save results JSON
    results = {
        "config": "CNN + RNA-FM (LoRA) + RLPA + thermo dG",
        "params": total_params,
        "val": val_metrics,
        "test": test_metrics,
        "best_epoch": best_epoch,
        "seed": args.seed,
    }
    results_path = args.output.replace(".pt", "_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
