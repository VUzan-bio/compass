"""Architecture diagnostic: isolate loss function vs pooling strategy.

4 configs:
  1. baseline:       Huber + 0.5*ranking + AvgPool     (current)
  2. mse_avgpool:    MSE + AvgPool                     (loss fix only)
  3. huber_catpool:  Huber + 0.5*ranking + cat(avg,max) (pool fix only)
  4. mse_catpool:    MSE + cat(avg,max)                (both fixes)

All: CNN+RNA-FM, seed 42, Phase 1, raw/100 activities.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from compass_net.compass_ml import CompassML
from compass_net.data.embedding_cache import EmbeddingCache
from compass_net.data.loaders.load_kim2018 import load_kim2018_domains
from compass_net.data.paired_loader import SingleTargetDataset
from compass_net.training.train_compass_ml import (
    collate_single_target,
    _get_batch_embeddings,
)
from compass_net.training.reproducibility import seed_everything
from torch.utils.data import DataLoader

CACHE_DIR = "compass/data/embeddings/rnafm"
SEED = 42
DEVICE = torch.device("cpu")


class ConcatPoolCompassML(CompassML):
    """CompassML with concat(avg, max) pooling instead of avg-only."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dense_dim = self.fused_dim * 2  # avg + max concatenated
        self.efficiency_head = nn.Sequential(
            nn.Linear(dense_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.21),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _pool_and_append_scalars(self, fused, scalar_features=None):
        x = fused.permute(0, 2, 1)  # (batch, fused_dim, seq_len)
        avg = x.mean(dim=-1)
        mx = x.max(dim=-1).values
        pooled = torch.cat([avg, mx], dim=-1)
        return pooled


def load_data():
    data = load_kim2018_domains()
    train_d = data["train_domains"][0]
    train_acts = (np.clip(np.array(train_d["activities"]), 0, 100) / 100.0).tolist()
    val_acts = (np.clip(np.array(data["val_activities"]), 0, 100) / 100.0).tolist()
    test_acts = (np.clip(np.array(data["test_activities"]), 0, 100) / 100.0).tolist()
    return data, train_d, train_acts, val_acts, test_acts


def train_and_eval(model, loss_type, name, data, train_d, train_acts, val_acts, test_acts, cache):
    seed_everything(SEED)
    model = model.to(DEVICE)

    if loss_type == "mse":
        mse_criterion = nn.MSELoss()
    else:
        from compass_net.losses.multitask_loss import MultiTaskLoss
        mt_criterion = MultiTaskLoss(lambda_disc=0.0, lambda_rank=0.5)

    train_ds = SingleTargetDataset(train_d["sequences"], train_acts)
    val_ds = SingleTargetDataset(data["val_sequences"], val_acts)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_single_target)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, collate_fn=collate_single_target)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    best_val_rho = -1.0
    patience_counter = 0
    best_state = None
    n_epochs = 200
    patience = 20

    t0 = time.time()
    for epoch in range(n_epochs):
        if loss_type != "mse":
            s = max(0.1, 1.0 - 0.9 * epoch / n_epochs)
            mt_criterion.set_spearman_strength(s)

        model.train()
        for batch in train_loader:
            target_oh = batch["target_onehot"].to(DEVICE)
            eff = batch["efficiency"].to(DEVICE)
            crrna_emb = _get_batch_embeddings(batch["crrna_spacer"], cache, DEVICE)
            out = model(target_onehot=target_oh, crrna_rnafm_emb=crrna_emb)
            pred = out["efficiency"].squeeze(-1)

            if loss_type == "mse":
                loss = mse_criterion(pred, eff)
            else:
                loss = mt_criterion(pred_eff=out["efficiency"], true_eff=eff)["total"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                crrna_emb = _get_batch_embeddings(batch["crrna_spacer"], cache, DEVICE)
                out = model(target_onehot=batch["target_onehot"].to(DEVICE), crrna_rnafm_emb=crrna_emb)
                val_preds.extend(out["efficiency"].squeeze(-1).cpu().tolist())
                val_targets.extend(batch["efficiency"].tolist())

        val_rho = spearmanr(val_preds, val_targets).statistic
        if np.isnan(val_rho):
            val_rho = 0.0

        if val_rho > best_val_rho:
            best_val_rho = val_rho
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            logger.info(
                "  [%s] Epoch %3d | Val rho: %.4f | Best: %.4f",
                name, epoch + 1, val_rho, best_val_rho,
            )

        if patience_counter >= patience:
            logger.info("  [%s] Early stop at epoch %d", name, epoch + 1)
            break

    train_time = time.time() - t0

    # Load best
    model.load_state_dict(best_state)
    model.eval()

    # Test
    test_ds = SingleTargetDataset(data["test_sequences"], test_acts)
    test_loader = DataLoader(test_ds, batch_size=512, collate_fn=collate_single_target)
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            crrna_emb = _get_batch_embeddings(batch["crrna_spacer"], cache, DEVICE)
            out = model(target_onehot=batch["target_onehot"].to(DEVICE), crrna_rnafm_emb=crrna_emb)
            test_preds.extend(out["efficiency"].squeeze(-1).cpu().tolist())
    test_preds = np.array(test_preds)
    test_rho = float(spearmanr(test_preds, np.array(data["test_activities"])).statistic)

    # Train rho
    train_preds = []
    with torch.no_grad():
        for batch in train_loader:
            crrna_emb = _get_batch_embeddings(batch["crrna_spacer"], cache, DEVICE)
            out = model(target_onehot=batch["target_onehot"].to(DEVICE), crrna_rnafm_emb=crrna_emb)
            train_preds.extend(out["efficiency"].squeeze(-1).cpu().tolist())
    train_rho = float(spearmanr(train_preds, train_d["activities"]).statistic)

    return {
        "name": name,
        "loss": loss_type,
        "pool": "catpool" if isinstance(model, ConcatPoolCompassML) else "avgpool",
        "train_rho": round(train_rho, 4),
        "val_rho": round(best_val_rho, 4),
        "test_rho": round(test_rho, 4),
        "gap_val_test": round(best_val_rho - test_rho, 4),
        "pred_std": round(float(test_preds.std()), 4),
        "pred_mean": round(float(test_preds.mean()), 4),
        "train_time_s": round(train_time, 0),
    }


def main():
    data, train_d, train_acts, val_acts, test_acts = load_data()
    cache = EmbeddingCache(CACHE_DIR)
    logger.info("Data: train=%d, val=%d, test=%d, cache=%d",
                len(train_d["sequences"]), len(data["val_sequences"]),
                len(data["test_sequences"]), len(cache))

    results = []

    # 1. Baseline
    logger.info("\n=== 1/4: Baseline (Huber+Rank, AvgPool) ===")
    m1 = CompassML(use_rnafm=True, use_rloop_attention=False, multitask=False)
    r1 = train_and_eval(m1, "huber_rank", "baseline", data, train_d, train_acts, val_acts, test_acts, cache)
    results.append(r1)
    logger.info("  RESULT: %s", r1)

    # 2. MSE only
    logger.info("\n=== 2/4: MSE only, AvgPool ===")
    m2 = CompassML(use_rnafm=True, use_rloop_attention=False, multitask=False)
    r2 = train_and_eval(m2, "mse", "mse_avgpool", data, train_d, train_acts, val_acts, test_acts, cache)
    results.append(r2)
    logger.info("  RESULT: %s", r2)

    # 3. ConcatPool, Huber+Rank
    logger.info("\n=== 3/4: Huber+Rank, ConcatPool ===")
    m3 = ConcatPoolCompassML(use_rnafm=True, use_rloop_attention=False, multitask=False)
    r3 = train_and_eval(m3, "huber_rank", "huber_catpool", data, train_d, train_acts, val_acts, test_acts, cache)
    results.append(r3)
    logger.info("  RESULT: %s", r3)

    # 4. MSE + ConcatPool
    logger.info("\n=== 4/4: MSE, ConcatPool ===")
    m4 = ConcatPoolCompassML(use_rnafm=True, use_rloop_attention=False, multitask=False)
    r4 = train_and_eval(m4, "mse", "mse_catpool", data, train_d, train_acts, val_acts, test_acts, cache)
    results.append(r4)
    logger.info("  RESULT: %s", r4)

    # Summary
    logger.info("\n" + "=" * 75)
    logger.info("ARCHITECTURE DIAGNOSTIC RESULTS")
    logger.info("=" * 75)
    logger.info("%-18s %8s %8s %8s %10s %8s", "Config", "Train", "Val", "Test", "Gap(v-t)", "PredSD")
    logger.info("-" * 75)
    for r in results:
        logger.info("%-18s %8.4f %8.4f %8.4f %10.4f %8.4f",
                    r["name"], r["train_rho"], r["val_rho"], r["test_rho"],
                    r["gap_val_test"], r["pred_std"])

    Path("results/research/arch_diagnostic").mkdir(parents=True, exist_ok=True)
    with open("results/research/arch_diagnostic/results.json", "w") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda r: r["test_rho"])
    logger.info("\nBEST: %s (test rho=%.4f, gap=%.4f)", best["name"], best["test_rho"], best["gap_val_test"])


if __name__ == "__main__":
    main()
