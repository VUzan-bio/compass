"""Benchmark and ablation evaluation for Compass-ML.

Produces the ablation table comparing model configurations:
    - Heuristic (baseline, 0 params)
    - SeqCNN v1 (CNN only, ~65K params)
    - Compass-ML: CNN only (sanity check = v1)
    - Compass-ML: CNN + RNA-FM
    - Compass-ML: CNN + RNA-FM + RLPA
    - Compass-ML: CNN + RNA-FM + RLPA + multi-task

All configurations evaluated with 5-fold clustered CV.
Reports: mean +/- std of Spearman rho across folds.
Results averaged over 3 random seeds per configuration.

Usage:
    python -m compass_ml.evaluation.benchmark \\
        --data compass/data/kim2018/nbt4061_source_data.xlsx \\
        --output compass-net/results/ablation.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ..compass_ml import CompassML

logger = logging.getLogger(__name__)


# ======================================================================
# Model evaluation
# ======================================================================


def evaluate_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Comprehensive evaluation metrics.

    Returns: spearman_rho, pearson_r, mse, mae, top_k_precision.
    """
    rho, rho_p = spearmanr(predictions, targets)
    r, r_p = pearsonr(predictions, targets)
    mse = float(mean_squared_error(targets, predictions))
    mae = float(mean_absolute_error(targets, predictions))

    # Top-k precision: fraction of top-20% predicted that are truly top-20%
    k = max(1, len(targets) // 5)
    top_pred = set(np.argsort(predictions)[-k:])
    top_true = set(np.argsort(targets)[-k:])
    top_k_prec = len(top_pred & top_true) / k

    return {
        "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
        "pearson_r": float(r) if not np.isnan(r) else 0.0,
        "mse": mse,
        "mae": mae,
        "top_k_precision": top_k_prec,
    }


def predict_compass_ml(
    model: CompassML,
    target_onehots: torch.Tensor,
    crrna_embeddings: torch.Tensor | None = None,
    device: torch.device | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """Run inference on a batch of targets.

    Args:
        model: trained CompassML.
        target_onehots: (N, 4, 34) one-hot target DNA.
        crrna_embeddings: (N, 20, 640) RNA-FM embeddings or None.
        device: torch device.
        batch_size: inference batch size.

    Returns:
        (N,) predicted efficiencies.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_preds: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, len(target_onehots), batch_size):
            batch_targets = target_onehots[i:i + batch_size].to(device)
            batch_emb = None
            if crrna_embeddings is not None:
                batch_emb = crrna_embeddings[i:i + batch_size].to(device)

            out = model(
                target_onehot=batch_targets,
                crrna_rnafm_emb=batch_emb,
            )
            all_preds.append(out["efficiency"].squeeze(-1).cpu())

    return torch.cat(all_preds).numpy()


# ======================================================================
# Ablation table
# ======================================================================

ABLATION_CONFIGS = [
    {
        "name": "Compass-ML: CNN only",
        "use_rnafm": False,
        "use_rloop_attention": False,
        "multitask": False,
    },
    {
        "name": "Compass-ML: CNN + RNA-FM",
        "use_rnafm": True,
        "use_rloop_attention": False,
        "multitask": False,
    },
    {
        "name": "Compass-ML: CNN + RNA-FM + RLPA",
        "use_rnafm": True,
        "use_rloop_attention": True,
        "multitask": False,
    },
    {
        "name": "Compass-ML: CNN + RNA-FM + RLPA + MT",
        "use_rnafm": True,
        "use_rloop_attention": True,
        "multitask": True,
    },
]


def run_ablation(
    configs: list[dict] | None = None,
    train_fn=None,
    evaluate_fn=None,
    n_folds: int = 5,
    n_seeds: int = 3,
    output_path: str | None = None,
) -> list[dict]:
    """Run ablation study across model configurations.

    This is a framework -- provide train_fn and evaluate_fn callables
    that handle the actual data loading and training.

    Args:
        configs: list of model configs (default: ABLATION_CONFIGS).
        train_fn: callable(model, fold_data, seed) -> trained_model.
        evaluate_fn: callable(model, test_data) -> metrics_dict.
        n_folds: number of CV folds.
        n_seeds: number of random seeds per config.
        output_path: save results to JSON.

    Returns:
        List of result dicts with mean/std across folds and seeds.
    """
    if configs is None:
        configs = ABLATION_CONFIGS

    results: list[dict] = []

    for cfg in configs:
        name = cfg.pop("name", "unnamed")
        all_rhos: list[float] = []

        for seed in range(n_seeds):
            model = CompassML(**cfg)
            n_params = model.count_trainable_params()

            if train_fn and evaluate_fn:
                for fold in range(n_folds):
                    trained = train_fn(model, fold, seed)
                    metrics = evaluate_fn(trained, fold)
                    all_rhos.append(metrics.get("spearman_rho", 0.0))

        result = {
            "name": name,
            "params": n_params,
            "val_rho_mean": float(np.mean(all_rhos)) if all_rhos else None,
            "val_rho_std": float(np.std(all_rhos)) if all_rhos else None,
            "n_evaluations": len(all_rhos),
            "config": cfg,
        }
        results.append(result)
        cfg["name"] = name  # restore

        if all_rhos:
            logger.info(
                "%-40s | %6dK | rho = %.4f +/- %.4f",
                name, n_params // 1000,
                result["val_rho_mean"], result["val_rho_std"],
            )
        else:
            logger.info("%-40s | %6dK | (no evaluation)", name, n_params // 1000)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Ablation results saved to %s", output_path)

    return results


def format_ablation_table(results: list[dict]) -> str:
    """Format ablation results as a markdown table."""
    lines = [
        "| Model | Params | Val rho (5-fold) |",
        "|-------|--------|-----------------|",
    ]
    for r in results:
        name = r["name"]
        params = f"{r['params'] // 1000}K" if r["params"] else "0"
        if r["val_rho_mean"] is not None:
            rho = f"{r['val_rho_mean']:.4f} +/- {r['val_rho_std']:.4f}"
        else:
            rho = "pending"
        lines.append(f"| {name} | {params} | {rho} |")
    return "\n".join(lines)
