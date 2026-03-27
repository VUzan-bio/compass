"""Cross-species transfer learning experiment.

Weeks 4-6: quantify generalization gap from Kim 2018 (human cells, AsCpf1,
cis-cleavage) to EasyDesign (in vitro, LbCas12a, trans-cleavage).

Three configurations:
  1. Baseline:  Kim 2018 only, no domain adaptation
  2. Multi:     Kim 2018 + EasyDesign, no domain adaptation
  3. DA + EVO2: Kim 2018 + EasyDesign + domain-adversarial + optional EVO-2 scalar

Reports BOTH test sets: Kim 2018 HT2+3 and EasyDesign test set.
The model must not degrade on Kim while improving on EasyDesign.

Usage:
    # Full experiment (3 configs x 3 seeds):
    python scripts/research/cross_species_transfer.py

    # Single config:
    python scripts/research/cross_species_transfer.py --config baseline

    # Dry run:
    python scripts/research/cross_species_transfer.py --dry-run

Output:
    results/research/cross_species/
        config_{name}/seed_{s}/     — checkpoints + training history
        transfer_results.json       — full results
        transfer_table.md           — comparison table
        mmd_analysis.json           — MMD between embedding spaces
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/research/cross_species")
KIM2018_PATH = "compass/data/kim2018/nbt4061_source_data.xlsx"
EASYDESIGN_PATH = "compass-net/data/external/easydesign/Table_S2.xlsx"
RNAFM_CACHE_DIR = "compass-net/cache/rnafm"

CONFIGS = {
    "baseline": {
        "name": "Baseline (Kim 2018 only)",
        "use_easydesign": False,
        "domain_adversarial": False,
        "use_evo2_scalar": False,
        "lambda_domain": 0.0,
    },
    "multi": {
        "name": "Multi-dataset (Kim + EasyDesign)",
        "use_easydesign": True,
        "domain_adversarial": False,
        "use_evo2_scalar": False,
        "lambda_domain": 0.0,
    },
    "da_evo2": {
        "name": "DA + EVO-2 (Kim + EasyDesign + GRL)",
        "use_easydesign": True,
        "domain_adversarial": True,
        "use_evo2_scalar": True,
        "lambda_domain": 0.05,
    },
}

SEEDS = [42, 123, 456]


# ======================================================================
# Data loading
# ======================================================================


def load_all_data(use_easydesign: bool) -> tuple[list[dict], dict, dict]:
    """Load datasets for multi-domain training.

    Returns:
        (datasets, kim_val_data, kim_test_data)
    """
    from compass_net.data.loaders.load_kim2018 import load_kim2018_domains
    from compass_net.data.multi_dataset import DatasetMeta

    kim = load_kim2018_domains(KIM2018_PATH)

    domain_id = 0
    datasets = []

    # Kim 2018 training domain
    d = kim["train_domains"][0]
    datasets.append({
        "metadata": DatasetMeta(
            name=d["name"],
            domain_id=domain_id,
            variant=d["variant"],
            readout_type=d["readout_type"],
            seq_format=d["seq_format"],
            cell_context=d["cell_context"],
        ),
        "sequences": d["sequences"],
        "activities": d["activities"],
    })
    domain_id += 1

    logger.info("Kim 2018 train: %d samples", len(d["sequences"]))

    # EasyDesign (optional)
    ed_test_data = None
    if use_easydesign:
        from compass_net.data.loaders.load_easydesign import load_easydesign

        ed = load_easydesign(EASYDESIGN_PATH, use_augmented=False)
        datasets.append({
            "metadata": DatasetMeta(
                name=ed["name"],
                domain_id=domain_id,
                variant=ed["variant"],
                readout_type=ed["readout_type"],
                seq_format=ed["seq_format"],
                cell_context=ed["cell_context"],
            ),
            "sequences": ed["sequences"],
            "activities": ed["activities"],
        })
        domain_id += 1
        logger.info("EasyDesign train: %d samples", len(ed["sequences"]))

        ed_test_data = {
            "sequences": ed["test_sequences"],
            "activities": ed["test_activities"],
        }
        logger.info("EasyDesign test: %d samples", len(ed["test_sequences"]))

    kim_val_data = {
        "sequences": kim["val_sequences"],
        "activities": kim["val_activities"],
    }
    kim_test_data = {
        "sequences": kim["test_sequences"],
        "activities": kim["test_activities"],
    }

    return datasets, kim_val_data, kim_test_data, ed_test_data


# ======================================================================
# Training
# ======================================================================


def train_single_config(
    config_key: str,
    seed: int,
    device: torch.device,
) -> dict:
    """Train one config with one seed."""
    from compass_net.compass_ml import CompassML
    from compass_net.data.embedding_cache import EmbeddingCache
    from compass_net.data.multi_dataset import MultiDatasetLoader, collate_multi
    from compass_net.data.balanced_sampler import DomainBalancedSampler
    from compass_net.data.paired_loader import SingleTargetDataset
    from compass_net.training.train_compass_ml import collate_single_target
    from compass_net.training.reproducibility import seed_everything
    from compass_net.losses.multitask_loss import MultiTaskLoss

    cfg = CONFIGS[config_key]
    seed_everything(seed)

    # Load data
    datasets, kim_val, kim_test, ed_test = load_all_data(cfg["use_easydesign"])

    n_domains = len(datasets) if cfg["domain_adversarial"] else None

    # Determine if EVO-2 scalar is available
    n_scalar = 0
    if cfg["use_evo2_scalar"]:
        evo2_path = Path("results/research/evo2_llr/llr_results.csv")
        if evo2_path.exists():
            n_scalar = 1
            logger.info("EVO-2 scalar feature enabled")
        else:
            logger.warning("EVO-2 results not found, proceeding without scalar feature")

    # Build model
    model = CompassML(
        use_rnafm=True,
        use_rloop_attention=True,
        multitask=False,
        n_domains=n_domains,
        n_scalar_features=n_scalar,
    )
    model = model.to(device)

    n_params = model.count_trainable_params()
    logger.info("Config: %s | Seed: %d | Params: %d | Domains: %s",
                cfg["name"], seed, n_params, n_domains)

    # Embedding cache
    cache = EmbeddingCache(RNAFM_CACHE_DIR)
    logger.info("RNA-FM cache: %d entries", len(cache))

    # Build training loader
    train_dataset = MultiDatasetLoader(datasets)
    logger.info("Multi-dataset: %d total samples", len(train_dataset))

    if n_domains and n_domains > 1:
        sampler = DomainBalancedSampler(
            domain_ids=[s["domain_id"] for s in train_dataset.samples],
            batch_size=128,
            n_batches_per_epoch=200,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=collate_multi,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True,
            collate_fn=collate_multi,
        )

    # Validation loader (Kim 2018 HT1-2)
    from compass_net.data.multi_dataset import quantile_normalise
    val_acts = quantile_normalise(np.array(kim_val["activities"])).tolist()
    val_ds = SingleTargetDataset(kim_val["sequences"], val_acts)
    val_loader = DataLoader(
        val_ds, batch_size=512, shuffle=False,
        collate_fn=collate_single_target,
    )

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=40, T_mult=2, eta_min=1e-6,
    )
    loss_fn = MultiTaskLoss(lambda_disc=0.0, lambda_rank=0.5)
    domain_loss_fn = nn.CrossEntropyLoss() if n_domains else None

    # Output
    save_dir = RESULTS_DIR / f"config_{config_key}" / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_rho = -1.0
    patience_counter = 0
    n_epochs = 80
    patience = 20
    history = {"train_loss": [], "val_rho": [], "dom_acc": [], "lr": []}

    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        total_dom_correct = 0
        total_dom_count = 0

        # GRL lambda schedule
        progress = epoch / max(n_epochs - 1, 1)
        grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        if hasattr(model, "domain_head") and model.use_domain_adversarial:
            model.domain_head.set_lambda(grl_lambda)

        for batch in train_loader:
            target_oh = batch["target_onehot"].to(device)
            activity = batch["activity"].to(device)

            # RNA-FM embeddings
            from compass_net.training.train_compass_ml import _get_batch_embeddings
            crrna_emb = _get_batch_embeddings(
                batch["crrna_spacers"], cache, device,
            )

            output = model(
                target_onehot=target_oh,
                crrna_rnafm_emb=crrna_emb,
            )

            # Efficiency loss
            l_eff = loss_fn(
                pred_eff=output["efficiency"],
                true_eff=activity,
            )["total"]

            # Domain loss
            loss = l_eff
            if n_domains and "domain_logits" in output:
                domain_ids = batch["domain_id"].to(device)
                l_dom = domain_loss_fn(output["domain_logits"], domain_ids)
                loss = l_eff + cfg["lambda_domain"] * l_dom

                dom_preds = output["domain_logits"].argmax(dim=-1)
                total_dom_correct += (dom_preds == domain_ids).sum().item()
                total_dom_count += domain_ids.size(0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Validate on Kim 2018 HT1-2
        val_rho = _validate(model, val_loader, cache, device)

        dom_acc = total_dom_correct / max(total_dom_count, 1) if total_dom_count else 0
        lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(total_loss / max(len(train_loader), 1))
        history["val_rho"].append(val_rho)
        history["dom_acc"].append(dom_acc)
        history["lr"].append(lr)

        if val_rho > best_val_rho:
            best_val_rho = val_rho
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_rho": val_rho,
                "config": cfg,
            }, str(save_dir / "best_model.pt"))
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            dom_str = f" | DomAcc: {dom_acc:.2f}" if total_dom_count else ""
            logger.info(
                "Epoch %3d | Loss: %.4f | Val ρ: %.4f | Best: %.4f%s | LR: %.2e",
                epoch + 1, history["train_loss"][-1], val_rho,
                best_val_rho, dom_str, lr,
            )

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d (best ρ=%.4f)", epoch + 1, best_val_rho)
            break

    train_time = time.time() - t0

    # Load best model
    ckpt = torch.load(str(save_dir / "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Evaluate on BOTH test sets
    metrics = _evaluate_both_test_sets(model, kim_test, ed_test, cache, device)
    metrics.update({
        "config": config_key,
        "config_name": cfg["name"],
        "seed": seed,
        "n_params": n_params,
        "train_time_s": round(train_time, 1),
        "best_val_rho": best_val_rho,
        "n_epochs": len(history["train_loss"]),
    })

    # Save
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "  Kim test ρ=%.4f | ED test ρ=%s | %d epochs, %.0fs",
        metrics["kim_test_rho"],
        f"{metrics['ed_test_rho']:.4f}" if metrics.get("ed_test_rho") else "N/A",
        metrics["n_epochs"], train_time,
    )

    return metrics


def _validate(model, val_loader, cache, device) -> float:
    """Validate on Kim 2018 HT1-2, return Spearman rho."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            target_oh = batch["target_onehot"].to(device)
            efficiency = batch["efficiency"]

            crrna_emb = None
            if model.use_rnafm and cache is not None:
                from compass_net.training.train_compass_ml import _get_batch_embeddings
                crrna_emb = _get_batch_embeddings(
                    batch["crrna_spacer"], cache, device,
                )

            output = model(target_onehot=target_oh, crrna_rnafm_emb=crrna_emb)
            all_preds.extend(output["efficiency"].squeeze(-1).cpu().tolist())
            all_targets.extend(efficiency.tolist())

    rho = stats.spearmanr(all_preds, all_targets).statistic
    return float(rho) if not np.isnan(rho) else 0.0


def _evaluate_both_test_sets(
    model, kim_test, ed_test, cache, device,
) -> dict:
    """Evaluate on Kim 2018 HT2+3 and EasyDesign test set."""
    from compass_net.data.paired_loader import SingleTargetDataset
    from compass_net.data.multi_dataset import quantile_normalise
    from compass_net.training.train_compass_ml import collate_single_target
    from compass_net.evaluation.benchmark import evaluate_predictions

    results = {}

    # Kim 2018 test
    kim_acts = quantile_normalise(np.array(kim_test["activities"])).tolist()
    kim_ds = SingleTargetDataset(kim_test["sequences"], kim_acts)
    kim_loader = DataLoader(kim_ds, batch_size=512, collate_fn=collate_single_target)

    kim_preds, kim_targets = _collect_preds(model, kim_loader, cache, device)
    kim_metrics = evaluate_predictions(kim_preds, kim_targets)
    results["kim_test_rho"] = kim_metrics["spearman_rho"]
    results["kim_test_pearson"] = kim_metrics["pearson_r"]
    results["kim_test_top20"] = kim_metrics["top_k_precision"]

    # GC-stratified on Kim test
    gc_vals = np.array([
        sum(1 for c in s.upper() if c in "GC") / len(s)
        for s in kim_test["sequences"]
    ])
    gc_q = np.percentile(gc_vals, [25, 50, 75])
    gc_bins = np.digitize(gc_vals, gc_q)
    gc_strat = {}
    for q in range(4):
        mask = gc_bins == q
        if mask.sum() >= 10:
            rho = float(stats.spearmanr(kim_preds[mask], kim_targets[mask]).statistic)
            gc_strat[f"q{q+1}"] = {"n": int(mask.sum()), "rho": round(rho, 4)}
    results["kim_gc_stratified"] = gc_strat

    # EasyDesign test
    if ed_test is not None and ed_test.get("sequences"):
        ed_acts = quantile_normalise(np.array(ed_test["activities"])).tolist()
        ed_ds = SingleTargetDataset(ed_test["sequences"], ed_acts)
        ed_loader = DataLoader(ed_ds, batch_size=512, collate_fn=collate_single_target)

        ed_preds, ed_targets = _collect_preds(model, ed_loader, cache, device)
        ed_metrics = evaluate_predictions(ed_preds, ed_targets)
        results["ed_test_rho"] = ed_metrics["spearman_rho"]
        results["ed_test_pearson"] = ed_metrics["pearson_r"]
        results["ed_test_top20"] = ed_metrics["top_k_precision"]
    else:
        results["ed_test_rho"] = None
        results["ed_test_pearson"] = None
        results["ed_test_top20"] = None

    return results


def _collect_preds(model, loader, cache, device):
    """Collect predictions from a DataLoader."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            target_oh = batch["target_onehot"].to(device)
            crrna_emb = None
            if model.use_rnafm and cache is not None:
                from compass_net.training.train_compass_ml import _get_batch_embeddings
                crrna_emb = _get_batch_embeddings(
                    batch["crrna_spacer"], cache, device,
                )
            output = model(target_onehot=target_oh, crrna_rnafm_emb=crrna_emb)
            preds.extend(output["efficiency"].squeeze(-1).cpu().tolist())
            targets.extend(batch["efficiency"].tolist())
    return np.array(preds), np.array(targets)


# ======================================================================
# MMD analysis
# ======================================================================


def compute_mmd(
    model,
    datasets: list[dict],
    cache,
    device: torch.device,
) -> dict:
    """Compute Maximum Mean Discrepancy between domain embedding spaces.

    MMD measures how different the learned representations are for Kim 2018
    vs EasyDesign samples. Lower MMD = more domain-invariant features.
    """
    from compass_net.data.multi_dataset import MultiDatasetLoader, collate_multi

    loader = MultiDatasetLoader(datasets)
    dl = DataLoader(loader, batch_size=256, collate_fn=collate_multi)

    embeddings_by_domain: dict[int, list] = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for batch in dl:
            target_oh = batch["target_onehot"].to(device)
            domain_ids = batch["domain_id"].numpy()

            from compass_net.training.train_compass_ml import _get_batch_embeddings
            crrna_emb = _get_batch_embeddings(
                batch["crrna_spacers"], cache, device,
            )

            output = model(target_onehot=target_oh, crrna_rnafm_emb=crrna_emb)
            emb = output["embedding"].cpu().numpy()

            for i, d_id in enumerate(domain_ids):
                embeddings_by_domain[int(d_id)].append(emb[i])

    # Compute MMD between each pair of domains
    mmd_results = {}
    domain_ids = sorted(embeddings_by_domain.keys())

    for i in range(len(domain_ids)):
        for j in range(i + 1, len(domain_ids)):
            d_i = domain_ids[i]
            d_j = domain_ids[j]
            emb_i = np.array(embeddings_by_domain[d_i])
            emb_j = np.array(embeddings_by_domain[d_j])

            mmd_val = _compute_mmd_rbf(emb_i, emb_j)
            key = f"domain_{d_i}_vs_{d_j}"
            mmd_results[key] = {
                "mmd": round(float(mmd_val), 6),
                "n_i": len(emb_i),
                "n_j": len(emb_j),
            }
            logger.info("MMD(%d, %d) = %.6f (n=%d, %d)",
                        d_i, d_j, mmd_val, len(emb_i), len(emb_j))

    return mmd_results


def _compute_mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float | None = None,
) -> float:
    """Compute MMD^2 with RBF kernel.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    where k(a,b) = exp(-gamma * ||a-b||^2)
    """
    if gamma is None:
        # Median heuristic
        XY = np.vstack([X[:500], Y[:500]])
        from scipy.spatial.distance import pdist
        dists = pdist(XY, "sqeuclidean")
        gamma = 1.0 / np.median(dists) if np.median(dists) > 0 else 1.0

    n_x = min(len(X), 1000)
    n_y = min(len(Y), 1000)
    X_sub = X[np.random.choice(len(X), n_x, replace=False)]
    Y_sub = Y[np.random.choice(len(Y), n_y, replace=False)]

    from scipy.spatial.distance import cdist
    XX = np.exp(-gamma * cdist(X_sub, X_sub, "sqeuclidean"))
    YY = np.exp(-gamma * cdist(Y_sub, Y_sub, "sqeuclidean"))
    XY = np.exp(-gamma * cdist(X_sub, Y_sub, "sqeuclidean"))

    mmd2 = XX.mean() + YY.mean() - 2 * XY.mean()
    return max(0.0, mmd2)


# ======================================================================
# Results aggregation
# ======================================================================


def aggregate_results(all_metrics: list[dict]) -> list[dict]:
    """Aggregate across seeds for each config."""
    by_config = defaultdict(list)
    for m in all_metrics:
        by_config[m["config"]].append(m)

    summary = []
    for config_key, runs in by_config.items():
        kim_rhos = [r["kim_test_rho"] for r in runs]
        ed_rhos = [r["ed_test_rho"] for r in runs if r.get("ed_test_rho") is not None]

        entry = {
            "config": config_key,
            "config_name": runs[0]["config_name"],
            "n_params": runs[0]["n_params"],
            "n_seeds": len(runs),
            "kim_test_rho_mean": round(float(np.mean(kim_rhos)), 4),
            "kim_test_rho_std": round(float(np.std(kim_rhos)), 4),
            "ed_test_rho_mean": round(float(np.mean(ed_rhos)), 4) if ed_rhos else None,
            "ed_test_rho_std": round(float(np.std(ed_rhos)), 4) if ed_rhos else None,
            "per_seed": [
                {
                    "seed": r["seed"],
                    "kim_rho": r["kim_test_rho"],
                    "ed_rho": r.get("ed_test_rho"),
                }
                for r in runs
            ],
        }
        summary.append(entry)

    return summary


def format_table(summary: list[dict]) -> str:
    """Format comparison table."""
    lines = [
        "# Cross-Species Transfer Results",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "| Config | Kim ρ (mean±std) | EasyDesign ρ (mean±std) | Δ Kim | Δ ED |",
        "|--------|------------------|-------------------------|-------|------|",
    ]

    baseline_kim = None
    baseline_ed = None

    for s in summary:
        kim_str = f"{s['kim_test_rho_mean']:.3f}±{s['kim_test_rho_std']:.3f}"
        ed_str = (f"{s['ed_test_rho_mean']:.3f}±{s['ed_test_rho_std']:.3f}"
                  if s['ed_test_rho_mean'] is not None else "—")

        if baseline_kim is None:
            baseline_kim = s["kim_test_rho_mean"]
            baseline_ed = s.get("ed_test_rho_mean")
            d_kim = "—"
            d_ed = "—"
        else:
            d_kim = f"{s['kim_test_rho_mean'] - baseline_kim:+.3f}"
            if s.get("ed_test_rho_mean") is not None and baseline_ed is not None:
                d_ed = f"{s['ed_test_rho_mean'] - baseline_ed:+.3f}"
            else:
                d_ed = "—"

        lines.append(f"| {s['config_name']} | {kim_str} | {ed_str} | {d_kim} | {d_ed} |")

    lines.extend([
        "",
        "## Decision Gate",
        "",
    ])

    # Check decision criteria
    if len(summary) >= 2:
        baseline = summary[0]
        best_da = summary[-1]  # DA config
        if (best_da.get("ed_test_rho_mean") is not None and
                baseline.get("ed_test_rho_mean") is not None):
            ed_improvement = best_da["ed_test_rho_mean"] - baseline["ed_test_rho_mean"]
            kim_degradation = baseline["kim_test_rho_mean"] - best_da["kim_test_rho_mean"]

            if ed_improvement >= 0.05 and kim_degradation <= 0.02:
                lines.append(
                    f"**POSITIVE**: DA improves EasyDesign ρ by {ed_improvement:+.3f} "
                    f"with Kim degradation of only {kim_degradation:.3f}."
                )
                lines.append("Domain-adversarial training works for this transfer.")
            else:
                lines.append(
                    f"**NEGATIVE**: ED improvement = {ed_improvement:+.3f}, "
                    f"Kim degradation = {kim_degradation:.3f}."
                )
                lines.append("Domains may be too different. Consider separate models.")

    return "\n".join(lines)


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description="Cross-species transfer experiment")
    parser.add_argument("--config", type=str, default=None, choices=list(CONFIGS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs_to_run = [args.config] if args.config else list(CONFIGS.keys())
    seeds = args.seeds or SEEDS

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info("Configs: %s", configs_to_run)
        logger.info("Seeds: %s", seeds)
        logger.info("Total runs: %d", len(configs_to_run) * len(seeds))
        for ck in configs_to_run:
            c = CONFIGS[ck]
            logger.info("  %s: ED=%s, DA=%s, EVO2=%s, λ=%.3f",
                        c["name"], c["use_easydesign"],
                        c["domain_adversarial"], c["use_evo2_scalar"],
                        c["lambda_domain"])
        return

    # Save experiment config
    with open(RESULTS_DIR / "experiment_config.json", "w") as f:
        json.dump({
            "experiment": "cross_species_transfer",
            "timestamp": datetime.now().isoformat(),
            "configs": configs_to_run,
            "seeds": seeds,
        }, f, indent=2)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    all_metrics = []
    total = len(configs_to_run) * len(seeds)
    idx = 0

    for config_key in configs_to_run:
        for seed in seeds:
            idx += 1
            logger.info("\n%s\n[%d/%d] %s | Seed: %d\n%s",
                        "=" * 60, idx, total,
                        CONFIGS[config_key]["name"], seed, "=" * 60)

            metrics_path = RESULTS_DIR / f"config_{config_key}" / f"seed_{seed}" / "metrics.json"
            if metrics_path.exists():
                logger.info("  Loading existing results...")
                with open(metrics_path) as f:
                    all_metrics.append(json.load(f))
                continue

            metrics = train_single_config(config_key, seed, device)
            all_metrics.append(metrics)

    # Aggregate
    summary = aggregate_results(all_metrics)

    with open(RESULTS_DIR / "transfer_results.json", "w") as f:
        json.dump({"summary": summary, "all_runs": all_metrics}, f, indent=2)

    table = format_table(summary)
    with open(RESULTS_DIR / "transfer_table.md", "w") as f:
        f.write(table)

    logger.info("\n%s", table)


if __name__ == "__main__":
    main()
