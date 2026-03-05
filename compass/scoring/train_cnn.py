"""Training script for SeqCNN — Cas12a guide activity predictor.

Standalone script. Not imported by the COMPASS pipeline at runtime.

Usage:
    python -m compass.scoring.train_cnn \
        --data compass/data/kim2018/sequences.csv \
        --output compass/weights/seq_cnn_best.pt \
        --epochs 200 --patience 20

References:
    Kim et al., Nature Biotechnology 36:239-241 (2018). PMID: 29431740.
    Huang et al., iMeta 3(4):e214 (2024). PMID: 39135699.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

from compass.scoring.preprocessing import encode_dataset, normalise_labels
from compass.scoring.seq_cnn import SeqCNN

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Loss function
# ══════════════════════════════════════════════════════════════════════


class HuberSpearmanLoss(nn.Module):
    """Huber loss with differentiable Spearman correlation regulariser.

    Primary: Huber (delta=1.0) — robust to outliers in activity measurements.
    Auxiliary: Soft-rank Spearman approximation — directly optimises the
    evaluation metric via differentiable ranking (Blondel et al., ICML 2020).

    Args:
        delta: Huber loss transition point.
        spearman_weight: Weight for (1 - rho) term.
    """

    def __init__(self, delta: float = 1.0, spearman_weight: float = 0.1):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.spearman_weight = spearman_weight

    def _soft_rank(self, x: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
        """Differentiable ranking via soft sorting."""
        pairwise = x.unsqueeze(-1) - x.unsqueeze(-2)
        return torch.sigmoid(pairwise / temp).sum(dim=-1)

    def _spearman(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Differentiable Spearman correlation (soft ranking)."""
        pred_rank = self._soft_rank(pred.squeeze())
        target_rank = self._soft_rank(target.squeeze())
        pred_c = pred_rank - pred_rank.mean()
        target_c = target_rank - target_rank.mean()
        num = (pred_c * target_c).sum()
        den = torch.sqrt((pred_c**2).sum() * (target_c**2).sum() + 1e-8)
        return num / den

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        huber = self.huber(pred, target)
        spearman = 1.0 - self._spearman(pred, target)
        return huber + self.spearman_weight * spearman


# ══════════════════════════════════════════════════════════════════════
# Data splitting
# ══════════════════════════════════════════════════════════════════════


def source_based_split(
    sequences: list[str],
    labels: np.ndarray,
    groups: list[str],
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split by genomic group to prevent sequence similarity leakage.

    Each fold's test set contains guides from genes/regions NOT in
    the training set. This mimics real-world usage where the model
    predicts activity for targets it has never seen.
    """
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(sequences, labels, groups=groups))


# ══════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════


def train_seq_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    save_path: str = "seq_cnn_best.pt",
) -> tuple[SeqCNN, dict]:
    """Train SeqCNN with early stopping on validation Spearman rho.

    Optimiser: AdamW (decoupled weight decay).
    Scheduler: Cosine annealing with warm restarts (T_0=50 epochs).
    Early stopping: on validation Spearman rho.

    Args:
        X_train: (N_train, 4, 34) one-hot encoded sequences.
        y_train: (N_train,) normalised activity labels.
        X_val: (N_val, 4, 34) validation sequences.
        y_val: (N_val,) validation labels.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Initial learning rate.
        weight_decay: L2 regularisation coefficient.
        patience: Early stopping patience (epochs without improvement).
        save_path: Path to save best model checkpoint.

    Returns:
        model: Trained SeqCNN with best validation rho weights.
        history: Dict of per-epoch training metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val).unsqueeze(1),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SeqCNN().to(device)
    criterion = HuberSpearmanLoss(delta=1.0, spearman_weight=0.1)
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimiser, T_0=50, T_mult=2, eta_min=1e-6
    )

    best_rho = -1.0
    patience_counter = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_rho": [],
    }

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / max(len(train_loader), 1)

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
                val_preds.append(pred.cpu().numpy())
                val_targets.append(yb.cpu().numpy())

        val_preds_flat = np.concatenate(val_preds).flatten()
        val_targets_flat = np.concatenate(val_targets).flatten()
        rho, _ = spearmanr(val_preds_flat, val_targets_flat)
        avg_val_loss = val_loss / max(len(val_loader), 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_rho"].append(rho)

        # Early stopping
        if rho > best_rho:
            best_rho = rho
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_rho": rho,
                    "val_loss": avg_val_loss,
                },
                save_path,
            )
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            logger.info(
                "Epoch %3d | Train loss: %.4f | Val loss: %.4f | "
                "Val rho: %.4f | Best rho: %.4f | LR: %.6f",
                epoch + 1,
                avg_train_loss,
                avg_val_loss,
                rho,
                best_rho,
                scheduler.get_last_lr()[0],
            )

        if patience_counter >= patience:
            logger.info(
                "Early stopping at epoch %d. Best rho = %.4f",
                epoch + 1,
                best_rho,
            )
            break

    # Load best model
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        "Loaded best model from epoch %d (rho = %.4f)",
        checkpoint["epoch"] + 1,
        checkpoint["val_rho"],
    )

    return model, history


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════


def evaluate_model(
    model: SeqCNN,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Comprehensive evaluation on held-out test set.

    Returns dict with: spearman_rho, pearson_r, mse, mae, top_k_precision.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    with torch.no_grad():
        preds = (
            model(torch.from_numpy(X_test).to(device)).cpu().numpy().flatten()
        )

    rho, rho_p = spearmanr(preds, y_test)
    r, r_p = pearsonr(preds, y_test)
    mse = float(mean_squared_error(y_test, preds))
    mae = float(mean_absolute_error(y_test, preds))

    # Top-k precision: fraction of top-20% predicted guides
    # that are actually in the top-20% measured guides
    k = max(1, len(y_test) // 5)
    top_pred_idx = set(np.argsort(preds)[-k:])
    top_true_idx = set(np.argsort(y_test)[-k:])
    top_k_precision = len(top_pred_idx & top_true_idx) / k

    logger.info("Spearman rho:        %.4f  (p = %.2e)", rho, rho_p)
    logger.info("Pearson r:           %.4f  (p = %.2e)", r, r_p)
    logger.info("MSE:                 %.4f", mse)
    logger.info("MAE:                 %.4f", mae)
    logger.info("Top-20%% precision:  %.3f", top_k_precision)

    return {
        "spearman_rho": float(rho),
        "pearson_r": float(r),
        "mse": mse,
        "mae": mae,
        "top_k_precision": top_k_precision,
    }


# ══════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SeqCNN on Cas12a activity data"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV with columns: sequence, label, [group]",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="compass/weights/seq_cnn_best.pt",
        help="Path to save best model checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    import pandas as pd

    logger.info("Loading data from %s", args.data)
    df = pd.read_csv(args.data)

    sequences = df["sequence"].tolist()
    raw_labels = df["label"].values
    labels = normalise_labels(raw_labels, transform="log")

    # Split: use group column if available, else random
    if "group" in df.columns:
        groups = df["group"].tolist()
        splits = source_based_split(sequences, labels, groups, n_splits=5)
        train_idx, val_idx = splits[0]
    else:
        logger.warning(
            "No 'group' column found — using random split. "
            "This may cause information leakage through sequence similarity."
        )
        n = len(sequences)
        perm = np.random.permutation(n)
        split_point = int(n * (1 - args.val_split))
        train_idx, val_idx = perm[:split_point], perm[split_point:]

    X, y = encode_dataset(sequences, labels, max_len=34)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    logger.info(
        "Training: %d samples, Validation: %d samples",
        len(X_train),
        len(X_val),
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    model, history = train_seq_cnn(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        save_path=args.output,
    )

    logger.info("Evaluating on validation set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_model(model, X_val, y_val, device)
    logger.info("Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
