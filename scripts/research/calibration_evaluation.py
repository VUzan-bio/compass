"""Post-training calibration evaluation and Expected Calibration Error.

"Must add" experiment: after retraining Compass-ML on full 15K Kim 2018,
evaluate calibration quality. A well-calibrated model means P(correct | p=0.7)
≈ 70% — essential for clinical decision-making when COMPASS ranks crRNAs
for diagnostic panel inclusion.

Calibration methods evaluated:
  1. Raw model output (sigmoid, uncalibrated)
  2. Temperature scaling (Platt 1999; Guo et al. ICML 2017)
  3. Isotonic regression (Zadrozny & Elkan, ICML 2002)
  4. Histogram binning (Zadrozny & Elkan, KDD 2001)

Metrics:
  - ECE: Expected Calibration Error (primary; Naeini et al. AAAI 2015)
  - MCE: Maximum Calibration Error (worst-case bin)
  - Brier score: proper scoring rule decomposition
  - Reliability diagrams: predicted vs observed frequency

Key insight: Spearman rho (our primary efficiency metric) is rank-invariant
and thus UNAFFECTED by calibration. But calibrated scores enable:
  - Meaningful score thresholds for panel inclusion
  - Cross-dataset comparison (Kim 2018 scores vs EasyDesign scores)
  - Confidence intervals on individual crRNA predictions

References:
  - Guo et al. "On Calibration of Modern Neural Networks." ICML 2017.
  - Platt. "Probabilistic outputs for SVMs." 1999.
  - Naeini et al. "Obtaining well calibrated probabilities using
    Bayesian binning into quantiles." AAAI 2015.

Usage:
    python scripts/research/calibration_evaluation.py
    python scripts/research/calibration_evaluation.py --checkpoint results/research/kim2018_benchmark/config_cnn_rnafm_rlpa/seed_42/best_model.pt

Output:
    results/research/calibration/
        calibration_results.json    — ECE, MCE, Brier for each method
        reliability_diagram.csv     — binned predicted vs observed
        calibration_report.md       — summary table
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/research/calibration")
KIM2018_PATH = "compass/data/kim2018/nbt4061_source_data.xlsx"
RNAFM_CACHE_DIR = "compass/data/embeddings/rnafm"


# ======================================================================
# Calibration methods
# ======================================================================


def temperature_scaling(
    logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Temperature scaling (Guo et al. ICML 2017).

    Finds T* = argmin_T MSE(sigma(logit / T), target)
    on the validation set. Preserves ranking (monotonic).

    Returns: (calibrated_probs, optimal_temperature)
    """
    def _logit(p, eps=1e-7):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    raw_logits = _logit(logits)

    def objective(T):
        calibrated = _sigmoid(raw_logits / T)
        return float(np.mean((calibrated - targets) ** 2))

    result = minimize_scalar(objective, bounds=(0.1, 20.0), method="bounded")
    T_opt = result.x

    calibrated = _sigmoid(raw_logits / T_opt)
    return calibrated, float(T_opt)


def isotonic_regression(
    predictions: np.ndarray,
    targets: np.ndarray,
    test_predictions: np.ndarray,
) -> np.ndarray:
    """Isotonic regression calibration (Zadrozny & Elkan, ICML 2002).

    Non-parametric, monotonic mapping. Preserves ranking.
    Fit on val, transform test.
    """
    from sklearn.isotonic import IsotonicRegression

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(predictions, targets)
    return ir.predict(test_predictions)


def histogram_binning(
    predictions: np.ndarray,
    targets: np.ndarray,
    test_predictions: np.ndarray,
    n_bins: int = 15,
) -> np.ndarray:
    """Histogram binning (Zadrozny & Elkan, KDD 2001).

    Non-parametric, piecewise-constant. Bins predictions and replaces
    with bin-level mean observed frequency. May NOT preserve ranking
    within bins (different from temperature scaling and isotonic).
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means[i] = targets[mask].mean()
        else:
            bin_means[i] = (bin_edges[i] + bin_edges[i + 1]) / 2

    # Map test predictions to calibrated values
    calibrated = np.zeros_like(test_predictions)
    for i in range(n_bins):
        mask = (test_predictions >= bin_edges[i]) & (test_predictions < bin_edges[i + 1])
        calibrated[mask] = bin_means[i]
    # Handle edge case: predictions exactly at 1.0
    calibrated[test_predictions >= bin_edges[-1]] = bin_means[-1]

    return calibrated


# ======================================================================
# Evaluation metrics
# ======================================================================


def expected_calibration_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, float, list[dict]]:
    """ECE and MCE with reliability diagram data.

    ECE = sum_b (|B_b|/N) * |avg_pred(B_b) - avg_target(B_b)|
    MCE = max_b |avg_pred(B_b) - avg_target(B_b)|

    Ref: Naeini et al. AAAI 2015.

    Returns: (ece, mce, bin_data)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    mce = 0.0
    n = len(predictions)

    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (predictions >= bin_edges[i + 1])

        count = int(mask.sum())
        if count == 0:
            bin_data.append({
                "bin_lo": round(float(bin_edges[i]), 3),
                "bin_hi": round(float(bin_edges[i + 1]), 3),
                "bin_center": round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 3),
                "count": 0,
                "mean_predicted": None,
                "mean_observed": None,
                "gap": None,
            })
            continue

        avg_pred = float(predictions[mask].mean())
        avg_target = float(targets[mask].mean())
        gap = abs(avg_pred - avg_target)

        ece += (count / n) * gap
        mce = max(mce, gap)

        bin_data.append({
            "bin_lo": round(float(bin_edges[i]), 3),
            "bin_hi": round(float(bin_edges[i + 1]), 3),
            "bin_center": round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 3),
            "count": count,
            "mean_predicted": round(avg_pred, 4),
            "mean_observed": round(avg_target, 4),
            "gap": round(gap, 4),
        })

    return round(float(ece), 4), round(float(mce), 4), bin_data


def brier_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Brier score = mean((pred - target)^2).

    Proper scoring rule. Decomposes into reliability + resolution - uncertainty.
    Lower is better. Range: [0, 1].
    """
    return round(float(np.mean((predictions - targets) ** 2)), 4)


def brier_decomposition(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Murphy decomposition of Brier score.

    BS = Reliability - Resolution + Uncertainty

    Reliability: penalty for miscalibration (lower = better calibrated)
    Resolution:  ability to separate outcomes (higher = more informative)
    Uncertainty: inherent unpredictability of targets (constant for dataset)

    Ref: Murphy, "A new vector partition of the probability score",
    J Applied Meteorology 1973.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    n = len(predictions)
    t_bar = targets.mean()

    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (predictions >= bin_edges[i + 1])

        n_k = mask.sum()
        if n_k == 0:
            continue

        avg_pred = predictions[mask].mean()
        avg_target = targets[mask].mean()

        reliability += (n_k / n) * (avg_pred - avg_target) ** 2
        resolution += (n_k / n) * (avg_target - t_bar) ** 2

    uncertainty = t_bar * (1 - t_bar)

    return {
        "brier_score": round(float(reliability - resolution + uncertainty), 4),
        "reliability": round(float(reliability), 4),
        "resolution": round(float(resolution), 4),
        "uncertainty": round(float(uncertainty), 4),
    }


# ======================================================================
# Model inference
# ======================================================================


def get_model_predictions(
    checkpoint_path: str | None,
    data_split: str = "test",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load model and get predictions on specified split.

    Returns: (predictions, targets, sequences)
    """
    import torch
    from compass_net.compass_ml import CompassML
    from compass_net.data.loaders.load_kim2018 import load_kim2018_domains
    from compass_net.data.paired_loader import SingleTargetDataset
    from compass_net.data.embedding_cache import EmbeddingCache
    from compass_net.training.train_compass_ml import collate_single_target, _get_batch_embeddings
    from torch.utils.data import DataLoader

    # Load data
    data = load_kim2018_domains(KIM2018_PATH)

    if data_split == "val":
        sequences = data["val_sequences"]
        activities = data["val_activities"]
    else:
        sequences = data["test_sequences"]
        activities = data["test_activities"]

    # Quantile normalise
    ranks = sp_stats.rankdata(np.array(activities))
    targets = (ranks / len(ranks)).astype(np.float32)

    # Find checkpoint
    if checkpoint_path is None:
        # Search for best benchmark checkpoint
        benchmark_dir = Path("results/research/kim2018_benchmark")
        candidates = list(benchmark_dir.glob("config_*/seed_*/best_model.pt"))
        if not candidates:
            raise FileNotFoundError("No model checkpoints found. Run benchmark first.")
        checkpoint_path = str(candidates[0])
        logger.info("Using checkpoint: %s", checkpoint_path)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Infer model config from checkpoint path (architecture in dir name)
    ckpt_path_str = str(checkpoint_path).lower()
    use_rnafm = "rnafm" in ckpt_path_str
    use_rlpa = "rlpa" in ckpt_path_str

    model = CompassML(use_rnafm=use_rnafm, use_rloop_attention=use_rlpa)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Embedding cache
    cache = EmbeddingCache(RNAFM_CACHE_DIR) if use_rnafm else None

    # Build dataset
    ds = SingleTargetDataset(sequences, targets.tolist())
    loader = DataLoader(ds, batch_size=512, collate_fn=collate_single_target)

    # Inference
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            target_oh = batch["target_onehot"].to(device)
            crrna_emb = None
            if use_rnafm and cache:
                crrna_emb = _get_batch_embeddings(batch["crrna_spacer"], cache, device)
            output = model(target_onehot=target_oh, crrna_rnafm_emb=crrna_emb)
            all_preds.extend(output["efficiency"].squeeze(-1).cpu().tolist())

    predictions = np.array(all_preds)
    return predictions, targets, sequences


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description="Calibration evaluation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "experiment_config.json", "w") as f:
        json.dump({
            "experiment": "calibration_evaluation",
            "timestamp": datetime.now().isoformat(),
            "checkpoint": args.checkpoint,
            "n_bins": args.n_bins,
            "methods": ["raw", "temperature_scaling", "isotonic", "histogram_binning"],
        }, f, indent=2)

    # Get predictions
    logger.info("Loading model and computing predictions...")

    try:
        val_preds, val_targets, _ = get_model_predictions(args.checkpoint, "val")
        test_preds, test_targets, test_seqs = get_model_predictions(args.checkpoint, "test")
    except (FileNotFoundError, ImportError) as e:
        logger.error("Cannot load model: %s", e)
        logger.info("Run benchmark_kim2018_full.py first to generate checkpoints.")
        return

    logger.info("Val: %d samples, Test: %d samples", len(val_preds), len(test_preds))

    # Evaluate each calibration method
    results = {}

    # 1. Raw (uncalibrated)
    logger.info("\n=== Raw (uncalibrated) ===")
    ece, mce, bins = expected_calibration_error(test_preds, test_targets, args.n_bins)
    bs = brier_score(test_preds, test_targets)
    decomp = brier_decomposition(test_preds, test_targets, args.n_bins)
    rho = float(sp_stats.spearmanr(test_preds, test_targets).statistic)
    results["raw"] = {
        "ece": ece, "mce": mce, "brier": bs,
        "brier_decomp": decomp, "spearman_rho": round(rho, 4),
        "bins": bins,
    }
    logger.info("ECE=%.4f, MCE=%.4f, Brier=%.4f, ρ=%.4f", ece, mce, bs, rho)

    # 2. Temperature scaling (fit on val, evaluate on test)
    logger.info("\n=== Temperature Scaling ===")
    _, T_opt = temperature_scaling(val_preds, val_targets)
    # Apply to test
    def _logit(p, eps=1e-7):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    test_temp = _sigmoid(_logit(test_preds) / T_opt)

    ece, mce, bins = expected_calibration_error(test_temp, test_targets, args.n_bins)
    bs = brier_score(test_temp, test_targets)
    decomp = brier_decomposition(test_temp, test_targets, args.n_bins)
    rho_temp = float(sp_stats.spearmanr(test_temp, test_targets).statistic)
    results["temperature_scaling"] = {
        "ece": ece, "mce": mce, "brier": bs,
        "brier_decomp": decomp, "spearman_rho": round(rho_temp, 4),
        "temperature": round(T_opt, 3),
        "bins": bins,
    }
    logger.info("T*=%.3f, ECE=%.4f, MCE=%.4f, Brier=%.4f, ρ=%.4f",
                T_opt, ece, mce, bs, rho_temp)

    # 3. Isotonic regression
    logger.info("\n=== Isotonic Regression ===")
    test_iso = isotonic_regression(val_preds, val_targets, test_preds)
    ece, mce, bins = expected_calibration_error(test_iso, test_targets, args.n_bins)
    bs = brier_score(test_iso, test_targets)
    decomp = brier_decomposition(test_iso, test_targets, args.n_bins)
    rho_iso = float(sp_stats.spearmanr(test_iso, test_targets).statistic)
    results["isotonic"] = {
        "ece": ece, "mce": mce, "brier": bs,
        "brier_decomp": decomp, "spearman_rho": round(rho_iso, 4),
        "bins": bins,
    }
    logger.info("ECE=%.4f, MCE=%.4f, Brier=%.4f, ρ=%.4f", ece, mce, bs, rho_iso)

    # 4. Histogram binning
    logger.info("\n=== Histogram Binning ===")
    test_hist = histogram_binning(val_preds, val_targets, test_preds, n_bins=args.n_bins)
    ece, mce, bins = expected_calibration_error(test_hist, test_targets, args.n_bins)
    bs = brier_score(test_hist, test_targets)
    decomp = brier_decomposition(test_hist, test_targets, args.n_bins)
    rho_hist = float(sp_stats.spearmanr(test_hist, test_targets).statistic)
    results["histogram_binning"] = {
        "ece": ece, "mce": mce, "brier": bs,
        "brier_decomp": decomp, "spearman_rho": round(rho_hist, 4),
        "bins": bins,
    }
    logger.info("ECE=%.4f, MCE=%.4f, Brier=%.4f, ρ=%.4f", ece, mce, bs, rho_hist)

    # Save results
    with open(RESULTS_DIR / "calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save reliability diagram data
    rel_rows = []
    for method, r in results.items():
        for b in r.get("bins", []):
            if b["count"] and b["count"] > 0:
                rel_rows.append({"method": method, **b})
    if rel_rows:
        with open(RESULTS_DIR / "reliability_diagram.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rel_rows[0].keys())
            writer.writeheader()
            writer.writerows(rel_rows)

    # Format report
    lines = [
        "# Calibration Evaluation Report",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"Test set: Kim 2018 HT2+HT3 (N={len(test_preds)})",
        "",
        "| Method | ECE ↓ | MCE ↓ | Brier ↓ | Reliability ↓ | Resolution ↑ | ρ (preserved?) |",
        "|--------|-------|-------|---------|---------------|--------------|----------------|",
    ]

    for method, r in results.items():
        d = r.get("brier_decomp", {})
        lines.append(
            f"| {method} | {r['ece']:.4f} | {r['mce']:.4f} | {r['brier']:.4f} | "
            f"{d.get('reliability', '—'):.4f} | {d.get('resolution', '—'):.4f} | "
            f"{r['spearman_rho']:.4f} |"
        )

    lines.extend([
        "",
        "## Key Observations",
        "",
        f"- Raw ECE: {results['raw']['ece']:.4f}",
        f"- Best ECE: {min(r['ece'] for r in results.values()):.4f} "
        f"({min(results.items(), key=lambda x: x[1]['ece'])[0]})",
        f"- Spearman rho preserved across all methods: "
        f"{all(abs(r['spearman_rho'] - results['raw']['spearman_rho']) < 0.001 for r in results.values())}",
        "",
        "## Recommendation",
        "",
    ])

    # Find best method
    best_method = min(results.items(), key=lambda x: x[1]["ece"])
    raw_ece = results["raw"]["ece"]
    best_ece = best_method[1]["ece"]

    if raw_ece < 0.05:
        lines.append(f"Model is already well-calibrated (ECE={raw_ece:.4f} < 0.05).")
        lines.append("No post-hoc calibration needed.")
    elif best_ece < 0.05:
        lines.append(
            f"Apply {best_method[0]} (ECE: {raw_ece:.4f} → {best_ece:.4f})."
        )
        if best_method[0] == "temperature_scaling":
            lines.append(f"Temperature T* = {results['temperature_scaling']['temperature']:.3f}")
    else:
        lines.append(f"Best ECE = {best_ece:.4f} > 0.05. Model is not well-calibrated.")
        lines.append("Consider: larger validation set, Bayesian calibration, or ensemble methods.")

    report = "\n".join(lines)
    with open(RESULTS_DIR / "calibration_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("\n%s", report)


if __name__ == "__main__":
    main()
