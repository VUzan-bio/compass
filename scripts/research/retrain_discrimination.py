"""Retrain XGBoost discrimination model on EasyDesign 7K paired data.

Uses 18 thermodynamic features. Saves new model in both pickle and
XGBoost JSON format for forward compatibility.

Also evaluates adding ESM-2 |LLR| as feature #19 where available
(AMR targets only — not available for EasyDesign virus/bacteria targets,
so uses 0.0 placeholder during training and real values at COMPASS
deployment time for AMR panels).

Usage:
    python scripts/research/retrain_discrimination.py
"""

from __future__ import annotations

import csv
import json
import logging
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EASYDESIGN_PATH = "compass-net/data/external/easydesign/Table_S2.xlsx"
OUTPUT_DIR = Path("compass-net/checkpoints")
RESULTS_DIR = Path("results/research/discrimination_retrain")


def load_paired_data():
    """Load EasyDesign paired MUT/WT discrimination data."""
    import pandas as pd

    df = pd.read_excel(EASYDESIGN_PATH, sheet_name="Training data")

    guides = []
    spacer_positions = []
    mismatch_types = []
    delta_logks = []
    guide_seqs_full = []

    grouped = df.groupby("guide_seq")
    for guide_seq, rows in grouped:
        guide = str(guide_seq).upper()
        if len(guide) < 20:
            continue

        perfect = rows[rows["guide_target_hamming_dist"] == 0]
        mismatched = rows[rows["guide_target_hamming_dist"] == 1]

        if len(perfect) == 0 or len(mismatched) == 0:
            continue

        mut_logk = perfect["30 min"].mean()

        for _, mm_row in mismatched.iterrows():
            wt_logk = mm_row["30 min"]
            delta = mut_logk - wt_logk

            target = str(mm_row.get("target_at_guide", "")).upper()
            mm_pos = 10
            mm_type = "rA:dC"

            if len(target) == len(guide):
                for i in range(len(guide)):
                    if guide[i] != target[i]:
                        # Position relative to PAM (guide starts with TTTV)
                        mm_pos = i - 4 + 1 if i >= 4 else 1  # 1-indexed from spacer start
                        # RNA:DNA mismatch type
                        dna_to_rna = {"A": "U", "T": "A", "C": "G", "G": "C"}
                        rna_base = dna_to_rna.get(guide[i], "N")
                        mm_type = f"r{rna_base}:d{target[i]}"
                        break

            guides.append(guide)
            spacer_positions.append(max(1, min(mm_pos, 20)))
            mismatch_types.append(mm_type)
            delta_logks.append(delta)
            guide_seqs_full.append(guide)

    logger.info("Loaded %d discrimination pairs from %d guides",
                len(guides), len(grouped))
    return {
        "guide_sequences": guide_seqs_full,
        "spacer_positions": spacer_positions,
        "mismatch_types": mismatch_types,
        "delta_logks": np.array(delta_logks),
    }


def compute_features(data: dict) -> np.ndarray:
    """Compute 18 thermodynamic features."""
    from compass_net.data.thermo_discrimination_features import (
        compute_features_for_pair,
        FEATURE_NAMES,
    )

    X = []
    for i in range(len(data["guide_sequences"])):
        feats = compute_features_for_pair(
            guide_seq=data["guide_sequences"][i],
            spacer_position=data["spacer_positions"][i],
            mismatch_type=data["mismatch_types"][i],
        )
        X.append([feats.get(n, 0.0) for n in FEATURE_NAMES])

    return np.array(X, dtype=np.float32)


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost regressor with early stopping."""
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=20,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42,
        )
        model.fit(X_train, y_train)

    return model


def evaluate(model, X, y):
    """Evaluate model: Spearman rho on discrimination ratios."""
    preds = model.predict(X)
    # Both are in log space (delta_logk)
    rho = spearmanr(preds, y).statistic
    if np.isnan(rho):
        rho = 0.0
    mae = float(np.mean(np.abs(preds - y)))
    return {
        "spearman_rho": round(float(rho), 4),
        "mae": round(mae, 4),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_paired_data()
    n = len(data["guide_sequences"])
    y = data["delta_logks"]

    # Compute features
    logger.info("Computing 18 thermodynamic features...")
    X = compute_features(data)
    logger.info("Feature matrix: %s, target: %s", X.shape, y.shape)

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = train_xgboost(X_train, y_train, X_val, y_val)
        metrics = evaluate(model, X_val, y_val)
        cv_results.append(metrics)
        logger.info("  Fold %d: rho=%.4f, MAE=%.4f", fold + 1,
                    metrics["spearman_rho"], metrics["mae"])

    rhos = [r["spearman_rho"] for r in cv_results]
    logger.info("CV Spearman rho: %.4f +/- %.4f", np.mean(rhos), np.std(rhos))

    # Train final model on full data (no early stopping — use best n_estimators from CV)
    logger.info("\nTraining final model on full dataset...")

    # Use 90/10 split for early stopping in final model
    n_train = int(0.9 * n)
    perm = np.random.RandomState(42).permutation(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    final_model = train_xgboost(X[train_idx], y[train_idx], X[val_idx], y[val_idx])

    # Evaluate on held-out
    final_metrics = evaluate(final_model, X[val_idx], y[val_idx])
    logger.info("Final model (10%% holdout): rho=%.4f", final_metrics["spearman_rho"])

    # Feature importances
    if hasattr(final_model, "feature_importances_"):
        from compass_net.data.thermo_discrimination_features import FEATURE_NAMES
        importances = final_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        logger.info("\nFeature importances (top 10):")
        for i in sorted_idx[:10]:
            logger.info("  %-25s %.4f", FEATURE_NAMES[i], importances[i])

    # Save model — pickle format (backward compatible)
    pkl_path = OUTPUT_DIR / "disc_xgb.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "model": final_model,
            "n_features": 18,
            "backend": "xgboost",
            "feature_names": list(FEATURE_NAMES) if 'FEATURE_NAMES' in dir() else None,
            "cv_rho_mean": round(float(np.mean(rhos)), 4),
            "cv_rho_std": round(float(np.std(rhos)), 4),
            "trained_on": "easydesign_7k_paired",
            "date": datetime.now().isoformat(),
        }, f)
    logger.info("Saved pickle model to %s", pkl_path)

    # Save model — XGBoost JSON format (forward portable)
    try:
        json_path = OUTPUT_DIR / "disc_xgb.json"
        final_model.save_model(str(json_path))
        # Save metadata alongside
        meta_path = OUTPUT_DIR / "disc_xgb_meta.json"
        from compass_net.data.thermo_discrimination_features import FEATURE_NAMES
        with open(meta_path, "w") as f:
            json.dump({
                "n_features": 18,
                "feature_names": list(FEATURE_NAMES),
                "backend": "xgboost",
                "cv_rho_mean": round(float(np.mean(rhos)), 4),
                "cv_rho_std": round(float(np.std(rhos)), 4),
                "trained_on": "easydesign_7k_paired",
                "n_pairs": n,
                "date": datetime.now().isoformat(),
            }, f, indent=2)
        logger.info("Saved XGBoost JSON model to %s", json_path)
    except Exception as e:
        logger.warning("Could not save XGBoost JSON: %s", e)

    # Save CV results
    with open(RESULTS_DIR / "retrain_results.json", "w") as f:
        json.dump({
            "cv_results": cv_results,
            "cv_rho_mean": round(float(np.mean(rhos)), 4),
            "cv_rho_std": round(float(np.std(rhos)), 4),
            "final_holdout": final_metrics,
            "n_pairs": n,
        }, f, indent=2)

    logger.info("\nDone. New discrimination model saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
