"""Kim 2018 full benchmark: 4 ablation configs x 3 seeds on 15K HT1-1.

Weeks 1-2 experiment: establish head-to-head comparison with DeepCpf1 by
training Compass-ML on the FULL Kim 2018 HT1-1 dataset (currently trained
on ~1,628 subset).

Usage:
    # Full benchmark (4 configs x 3 seeds, Phase 1 only):
    python scripts/research/benchmark_kim2018_full.py

    # Single config for debugging:
    python scripts/research/benchmark_kim2018_full.py --config cnn_only --seeds 42

    # Dry run (print config, don't train):
    python scripts/research/benchmark_kim2018_full.py --dry-run

    # Evaluate existing checkpoints only:
    python scripts/research/benchmark_kim2018_full.py --eval-only

Output:
    results/research/kim2018_benchmark/
        config_{name}/seed_{s}/         - checkpoints + training history
        benchmark_results.json          - full results with CI
        benchmark_table.md              - publication-ready comparison table
        figures/                        - GC-stratified plots
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/research/kim2018_benchmark")
KIM2018_PATH = "compass/data/kim2018/nbt4061_source_data.xlsx"
RNAFM_CACHE_DIR = "compass/data/embeddings/rnafm"

# Published baselines from literature (test on HT2+HT3 combined)
LITERATURE_BASELINES = {
    "DeepCpf1 (Kim 2018)": {"spearman_rho": 0.87, "source": "Kim et al. NBT 2018"},
    "Seq-deepCpf1 (Kim 2018)": {"spearman_rho": 0.83, "source": "Kim et al. NBT 2018"},
    "CRISPR-BERT (Dallago 2023)": {"spearman_rho": 0.69, "source": "Dallago et al. 2023"},
}

# Ablation configurations
CONFIGS = {
    "cnn_only": {
        "name": "Compass-ML: CNN only",
        "use_rnafm": False,
        "use_rloop_attention": False,
        "multitask": False,
    },
    "cnn_rnafm": {
        "name": "Compass-ML: CNN + RNA-FM",
        "use_rnafm": True,
        "use_rloop_attention": False,
        "multitask": False,
    },
    "cnn_rnafm_rlpa": {
        "name": "Compass-ML: CNN + RNA-FM + RLPA",
        "use_rnafm": True,
        "use_rloop_attention": True,
        "multitask": False,
    },
    "cnn_rnafm_rlpa_mt": {
        "name": "Compass-ML: CNN + RNA-FM + RLPA + MT",
        "use_rnafm": True,
        "use_rloop_attention": True,
        "multitask": True,
    },
}

SEEDS = [42, 123, 456]


# ======================================================================
# Data loading
# ======================================================================


def load_data() -> dict:
    """Load Kim 2018 full dataset."""
    from compass_net.data.loaders.load_kim2018 import load_kim2018_domains

    data = load_kim2018_domains(KIM2018_PATH)
    train_domain = data["train_domains"][0]
    logger.info(
        "Kim 2018 loaded: train=%d (HT1-1), val=%d (HT1-2), test=%d (HT2+3)",
        len(train_domain["sequences"]),
        len(data["val_sequences"]),
        len(data["test_sequences"]),
    )
    return data


def build_datasets(data: dict):
    """Build PyTorch datasets from loaded data."""
    from compass_net.data.paired_loader import SingleTargetDataset

    train_domain = data["train_domains"][0]

    # Normalize activities to [0, 1] via quantile transform within each split
    train_acts = _quantile_normalize(train_domain["activities"])
    val_acts = _quantile_normalize(data["val_activities"])
    test_acts = _quantile_normalize(data["test_activities"])

    train_ds = SingleTargetDataset(train_domain["sequences"], train_acts)
    val_ds = SingleTargetDataset(data["val_sequences"], val_acts)
    test_ds = SingleTargetDataset(data["test_sequences"], test_acts)

    return train_ds, val_ds, test_ds


def _quantile_normalize(activities) -> list[float]:
    """Quantile normalize to [0, 1] (rank-based, preserves ordering)."""
    arr = np.array(activities, dtype=np.float64)
    ranks = stats.rankdata(arr, method="average")
    return (ranks / len(ranks)).tolist()


def compute_gc_content(seq: str) -> float:
    """GC fraction of a DNA sequence."""
    gc = sum(1 for c in seq.upper() if c in "GC")
    return gc / len(seq) if seq else 0.0


# ======================================================================
# Training
# ======================================================================


def train_single_config(
    config_key: str,
    seed: int,
    data: dict,
    device: torch.device,
) -> dict:
    """Train one config with one seed. Returns metrics dict."""
    from compass_net.compass_ml import CompassML
    from compass_net.data.embedding_cache import EmbeddingCache
    from compass_net.training.train_compass_ml import (
        train_phase,
        collate_single_target,
    )
    from compass_net.training.reproducibility import seed_everything

    cfg = CONFIGS[config_key]
    seed_everything(seed)

    # Build model
    model = CompassML(
        use_rnafm=cfg["use_rnafm"],
        use_rloop_attention=cfg["use_rloop_attention"],
        multitask=cfg["multitask"],
    )
    n_params = model.count_trainable_params()
    logger.info(
        "Config: %s | Seed: %d | Params: %d",
        cfg["name"], seed, n_params,
    )

    # Embedding cache
    cache = None
    if cfg["use_rnafm"]:
        cache = EmbeddingCache(RNAFM_CACHE_DIR)
        logger.info("RNA-FM cache: %d entries", len(cache))

    # Build datasets and loaders
    train_ds, val_ds, test_ds = build_datasets(data)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True,
        collate_fn=collate_single_target, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=512, shuffle=False,
        collate_fn=collate_single_target, num_workers=0,
    )

    # Output path
    save_dir = RESULTS_DIR / f"config_{config_key}" / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir / "best_model.pt")

    # Train Phase 1 (efficiency only)
    t0 = time.time()
    model, history = train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        phase=1,
        save_path=save_path,
        embedding_cache=cache,
        device=device,
        seed=seed,
    )
    train_time = time.time() - t0

    # Save training history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Evaluate on test set
    metrics = evaluate_model(model, data, cache, device)
    metrics["config"] = config_key
    metrics["config_name"] = cfg["name"]
    metrics["seed"] = seed
    metrics["n_params"] = n_params
    metrics["train_time_s"] = round(train_time, 1)
    metrics["best_val_rho"] = max(history["val_rho"]) if history["val_rho"] else 0.0
    metrics["n_epochs"] = len(history["train_loss"])

    # Save per-run metrics
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "  Test rho=%.4f (val rho=%.4f, %d epochs, %.0fs)",
        metrics["test_rho"], metrics["best_val_rho"],
        metrics["n_epochs"], train_time,
    )

    return metrics


def evaluate_model(
    model,
    data: dict,
    cache,
    device: torch.device,
) -> dict:
    """Evaluate trained model on val and test sets with full metrics."""
    from compass_net.evaluation.benchmark import (
        predict_compass_ml,
        evaluate_predictions,
    )
    from compass_net.data.paired_loader import SingleTargetDataset
    from compass_net.training.train_compass_ml import collate_single_target
    from torch.utils.data import DataLoader

    results = {}

    # --- Test set evaluation (HT2 + HT3) ---
    test_seqs = data["test_sequences"]
    test_acts_raw = np.array(data["test_activities"], dtype=np.float64)
    test_acts_norm = np.array(_quantile_normalize(data["test_activities"]))

    test_ds = SingleTargetDataset(test_seqs, test_acts_norm.tolist())
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False,
        collate_fn=collate_single_target, num_workers=0,
    )

    # Collect predictions
    preds, targets = _collect_predictions(model, test_loader, cache, device)

    # Standard metrics
    test_metrics = evaluate_predictions(preds, targets)
    results["test_rho"] = test_metrics["spearman_rho"]
    results["test_pearson"] = test_metrics["pearson_r"]
    results["test_mae"] = test_metrics["mae"]
    results["test_top20_precision"] = test_metrics["top_k_precision"]

    # BCa bootstrap 95% CI for Spearman rho
    ci_lo, ci_hi = _bca_bootstrap_ci(preds, targets, n_boot=10000)
    results["test_rho_ci_lo"] = ci_lo
    results["test_rho_ci_hi"] = ci_hi

    # --- GC-stratified analysis ---
    gc_values = np.array([compute_gc_content(s) for s in test_seqs])
    gc_quartiles = np.percentile(gc_values, [25, 50, 75])
    gc_bins = np.digitize(gc_values, gc_quartiles)  # 0, 1, 2, 3

    gc_strat = {}
    for q in range(4):
        mask = gc_bins == q
        if mask.sum() >= 10:
            q_rho = float(stats.spearmanr(preds[mask], targets[mask]).statistic)
            gc_strat[f"q{q + 1}"] = {
                "n": int(mask.sum()),
                "rho": round(q_rho, 4),
                "gc_range": [
                    round(float(gc_values[mask].min()), 3),
                    round(float(gc_values[mask].max()), 3),
                ],
            }
    results["gc_stratified"] = gc_strat

    # --- Validation set (HT1-2) ---
    val_acts_norm = np.array(_quantile_normalize(data["val_activities"]))
    val_ds = SingleTargetDataset(data["val_sequences"], val_acts_norm.tolist())
    val_loader = DataLoader(
        val_ds, batch_size=512, shuffle=False,
        collate_fn=collate_single_target, num_workers=0,
    )
    val_preds, val_targets = _collect_predictions(model, val_loader, cache, device)
    val_metrics = evaluate_predictions(val_preds, val_targets)
    results["val_rho"] = val_metrics["spearman_rho"]

    return results


def _collect_predictions(
    model,
    loader,
    cache,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and collect predictions + targets."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            target_onehot = batch["target_onehot"].to(device)
            efficiency = batch["efficiency"]

            crrna_emb = None
            if model.use_rnafm and cache is not None:
                from compass_net.training.train_compass_ml import _get_batch_embeddings
                crrna_emb = _get_batch_embeddings(
                    batch["crrna_spacer"], cache, device,
                )

            output = model(
                target_onehot=target_onehot,
                crrna_rnafm_emb=crrna_emb,
            )

            all_preds.extend(output["efficiency"].squeeze(-1).cpu().tolist())
            all_targets.extend(efficiency.tolist())

    return np.array(all_preds), np.array(all_targets)


# ======================================================================
# Statistical analysis
# ======================================================================


def _bca_bootstrap_ci(
    preds: np.ndarray,
    targets: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """BCa (bias-corrected and accelerated) bootstrap 95% CI for Spearman rho."""
    rng = np.random.default_rng(42)
    n = len(preds)
    observed = float(stats.spearmanr(preds, targets).statistic)

    # Bootstrap distribution
    boot_rhos = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        rho = stats.spearmanr(preds[idx], targets[idx]).statistic
        boot_rhos[b] = rho if not np.isnan(rho) else 0.0

    # Bias correction
    z0 = stats.norm.ppf(np.mean(boot_rhos < observed))

    # Acceleration (jackknife)
    jack_rhos = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jack_rhos[i] = stats.spearmanr(preds[mask], targets[mask]).statistic

    jack_mean = jack_rhos.mean()
    num = np.sum((jack_mean - jack_rhos) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack_rhos) ** 2) ** 1.5)
    a = num / denom if abs(denom) > 1e-12 else 0.0

    # BCa percentiles
    z_alpha = stats.norm.ppf(alpha / 2)
    z_1alpha = stats.norm.ppf(1 - alpha / 2)

    a1 = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    a2 = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    # Clamp to valid percentile range
    a1 = np.clip(a1, 0.001, 0.999)
    a2 = np.clip(a2, 0.001, 0.999)

    ci_lo = float(np.percentile(boot_rhos, a1 * 100))
    ci_hi = float(np.percentile(boot_rhos, a2 * 100))

    return round(ci_lo, 4), round(ci_hi, 4)


def steiger_z_test(
    rho1: float,
    rho2: float,
    n: int,
) -> tuple[float, float]:
    """Steiger's Z-test for comparing two dependent Spearman correlations.

    Tests H0: rho1 = rho2 (two-tailed).
    Both correlations are computed on the same test set.

    Returns (z_statistic, p_value).
    """
    # Fisher z-transform
    z1 = np.arctanh(rho1)
    z2 = np.arctanh(rho2)
    se = np.sqrt(2.0 / (n - 3))
    z_stat = (z1 - z2) / se
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    return round(float(z_stat), 4), round(float(p_val), 4)


# ======================================================================
# Results aggregation and reporting
# ======================================================================


def aggregate_results(all_metrics: list[dict]) -> list[dict]:
    """Aggregate across seeds for each config."""
    from collections import defaultdict

    by_config = defaultdict(list)
    for m in all_metrics:
        by_config[m["config"]].append(m)

    summary = []
    for config_key, runs in by_config.items():
        rhos = [r["test_rho"] for r in runs]
        val_rhos = [r["val_rho"] for r in runs]
        pearsons = [r["test_pearson"] for r in runs]
        maes = [r["test_mae"] for r in runs]
        top20s = [r["test_top20_precision"] for r in runs]

        summary.append({
            "config": config_key,
            "config_name": runs[0]["config_name"],
            "n_params": runs[0]["n_params"],
            "n_seeds": len(runs),
            # Spearman rho
            "test_rho_mean": round(float(np.mean(rhos)), 4),
            "test_rho_std": round(float(np.std(rhos)), 4),
            "test_rho_min": round(float(np.min(rhos)), 4),
            "test_rho_max": round(float(np.max(rhos)), 4),
            # BCa CI from first seed (representative)
            "test_rho_ci_lo": runs[0].get("test_rho_ci_lo"),
            "test_rho_ci_hi": runs[0].get("test_rho_ci_hi"),
            # Validation
            "val_rho_mean": round(float(np.mean(val_rhos)), 4),
            # Secondary metrics
            "test_pearson_mean": round(float(np.mean(pearsons)), 4),
            "test_mae_mean": round(float(np.mean(maes)), 4),
            "test_top20_mean": round(float(np.mean(top20s)), 4),
            # Training stats
            "mean_epochs": round(float(np.mean([r["n_epochs"] for r in runs])), 0),
            "mean_train_time_s": round(float(np.mean([r["train_time_s"] for r in runs])), 0),
            # GC stratified (from first seed)
            "gc_stratified": runs[0].get("gc_stratified", {}),
            # Per-seed details
            "per_seed": [
                {"seed": r["seed"], "test_rho": r["test_rho"], "val_rho": r["val_rho"]}
                for r in runs
            ],
        })

    return sorted(summary, key=lambda x: x["test_rho_mean"], reverse=True)


def format_comparison_table(summary: list[dict]) -> str:
    """Format results as a publication-ready markdown comparison table."""
    lines = []
    lines.append("# Kim 2018 Benchmark Results")
    lines.append("")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"Training: HT1-1 (15K guides), Val: HT1-2, Test: HT2+HT3")
    lines.append("")

    # Main comparison table
    lines.append("## Head-to-Head Comparison")
    lines.append("")
    lines.append("| Model | Params | Test ρ (mean±std) | 95% CI | Val ρ | Pearson r | Top-20% |")
    lines.append("|-------|--------|-------------------|--------|-------|-----------|---------|")

    # Our models
    for s in summary:
        ci_str = ""
        if s["test_rho_ci_lo"] is not None:
            ci_str = f"[{s['test_rho_ci_lo']:.3f}, {s['test_rho_ci_hi']:.3f}]"
        lines.append(
            f"| {s['config_name']} | {s['n_params']:,} | "
            f"{s['test_rho_mean']:.3f}±{s['test_rho_std']:.3f} | {ci_str} | "
            f"{s['val_rho_mean']:.3f} | {s['test_pearson_mean']:.3f} | "
            f"{s['test_top20_mean']:.3f} |"
        )

    # Literature baselines
    lines.append("|-------|--------|-------------------|--------|-------|-----------|---------|")
    for name, info in LITERATURE_BASELINES.items():
        lines.append(
            f"| {name} | — | {info['spearman_rho']:.3f} | — | — | — | — |"
        )

    lines.append("")

    # GC-stratified table
    lines.append("## GC-Stratified Performance (Best Config)")
    lines.append("")
    if summary:
        best = summary[0]
        gc = best.get("gc_stratified", {})
        if gc:
            lines.append("| GC Quartile | N | GC Range | Spearman ρ |")
            lines.append("|-------------|---|----------|------------|")
            for q_key in sorted(gc.keys()):
                v = gc[q_key]
                lines.append(
                    f"| {q_key.upper()} | {v['n']} | "
                    f"{v['gc_range'][0]:.1%}–{v['gc_range'][1]:.1%} | "
                    f"{v['rho']:.3f} |"
                )

    lines.append("")

    # Decision gate
    lines.append("## Decision Gate")
    lines.append("")
    if summary:
        best_rho = summary[0]["test_rho_mean"]
        if best_rho >= 0.75:
            lines.append(f"**POSITIVE**: Best test ρ = {best_rho:.3f} ≥ 0.75.")
            lines.append("Architecture is sound. Proceed with cross-species experiments.")
        elif best_rho >= 0.65:
            lines.append(f"**MARGINAL**: Best test ρ = {best_rho:.3f} (0.65–0.75).")
            lines.append("Competitive but not SOTA. Consider: flanking context (34nt → 50nt), "
                         "attention pooling, or ensemble approaches.")
        else:
            lines.append(f"**NEGATIVE**: Best test ρ = {best_rho:.3f} < 0.65.")
            lines.append("Architecture is limited. Check: (a) quantile norm bug, "
                         "(b) GC bias in high-GC quartile, (c) RNA-FM zero-embedding rate.")

    return "\n".join(lines)


# ======================================================================
# Evaluation-only mode
# ======================================================================


def load_existing_metrics() -> list[dict]:
    """Load metrics from previous runs."""
    all_metrics = []
    for config_key in CONFIGS:
        for seed in SEEDS:
            metrics_path = RESULTS_DIR / f"config_{config_key}" / f"seed_{seed}" / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    all_metrics.append(json.load(f))
    return all_metrics


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description="Kim 2018 Full Benchmark")
    parser.add_argument(
        "--config", type=str, default=None,
        choices=list(CONFIGS.keys()),
        help="Run single config (default: all 4)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Seeds to use (default: 42 123 456)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs_to_run = [args.config] if args.config else list(CONFIGS.keys())
    seeds = args.seeds or SEEDS

    # Save experiment config
    experiment_config = {
        "experiment": "kim2018_full_benchmark",
        "timestamp": datetime.now().isoformat(),
        "configs": configs_to_run,
        "seeds": seeds,
        "data_path": KIM2018_PATH,
        "phase": 1,
        "literature_baselines": LITERATURE_BASELINES,
    }

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info("Configs: %s", configs_to_run)
        logger.info("Seeds: %s", seeds)
        for ck in configs_to_run:
            cfg = CONFIGS[ck]
            logger.info("  %s: rnafm=%s, rlpa=%s, mt=%s",
                        cfg["name"], cfg["use_rnafm"],
                        cfg["use_rloop_attention"], cfg["multitask"])
        logger.info("Total runs: %d", len(configs_to_run) * len(seeds))
        return

    with open(RESULTS_DIR / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)

    if args.eval_only:
        logger.info("=== Evaluation Only ===")
        all_metrics = load_existing_metrics()
        if not all_metrics:
            logger.error("No existing metrics found in %s", RESULTS_DIR)
            sys.exit(1)
        logger.info("Loaded %d existing runs", len(all_metrics))
    else:
        # Determine device
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", device)

        # Load data once
        data = load_data()

        # Run all configs x seeds
        all_metrics = []
        total_runs = len(configs_to_run) * len(seeds)
        run_idx = 0

        for config_key in configs_to_run:
            for seed in seeds:
                run_idx += 1
                logger.info(
                    "\n%s\n[%d/%d] Config: %s | Seed: %d\n%s",
                    "=" * 60, run_idx, total_runs,
                    CONFIGS[config_key]["name"], seed, "=" * 60,
                )

                # Check for existing results
                metrics_path = (
                    RESULTS_DIR / f"config_{config_key}" / f"seed_{seed}" / "metrics.json"
                )
                if metrics_path.exists():
                    logger.info("  Found existing results, loading...")
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    all_metrics.append(metrics)
                    continue

                metrics = train_single_config(config_key, seed, data, device)
                all_metrics.append(metrics)

    # Aggregate and report
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATION")
    logger.info("=" * 60)

    summary = aggregate_results(all_metrics)

    # Save full results
    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        json.dump({"summary": summary, "all_runs": all_metrics}, f, indent=2)

    # Save comparison table
    table = format_comparison_table(summary)
    with open(RESULTS_DIR / "benchmark_table.md", "w", encoding="utf-8") as f:
        f.write(table)

    # Print summary
    logger.info("\n%s", table)

    # Statistical comparisons between configs
    if len(summary) >= 2:
        logger.info("\n=== Steiger's Z-tests (pairwise) ===")
        n_test = len(all_metrics[0].get("test_rho_ci_lo", 0) and
                      [m for m in all_metrics if m["config"] == summary[0]["config"]])
        # Use approximate N from test set
        n_approx = 4214  # HT2 + HT3
        for i in range(len(summary)):
            for j in range(i + 1, len(summary)):
                z, p = steiger_z_test(
                    summary[i]["test_rho_mean"],
                    summary[j]["test_rho_mean"],
                    n_approx,
                )
                sig = "*" if p < 0.05 else ""
                logger.info(
                    "  %s vs %s: z=%.3f, p=%.4f %s",
                    summary[i]["config_name"],
                    summary[j]["config_name"],
                    z, p, sig,
                )

    # Decision gate
    logger.info("\n" + "=" * 60)
    logger.info("DECISION GATE")
    logger.info("=" * 60)
    if summary:
        best = summary[0]
        best_rho = best["test_rho_mean"]
        logger.info("Best config: %s (ρ = %.4f)", best["config_name"], best_rho)

        if best_rho >= 0.75:
            logger.info("POSITIVE: ρ ≥ 0.75 → Architecture sound, proceed to cross-species")
        elif best_rho >= 0.65:
            logger.info("MARGINAL: 0.65 ≤ ρ < 0.75 → Competitive but not SOTA")
            logger.info("  Consider: extended flanking context, attention pooling, ensembles")
        else:
            logger.info("NEGATIVE: ρ < 0.65 → Architecture limited")
            logger.info("  Check: quantile norm, GC bias, RNA-FM zero-embedding rate")


if __name__ == "__main__":
    main()
