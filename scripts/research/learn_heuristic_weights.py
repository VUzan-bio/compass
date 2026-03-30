"""Learn optimal heuristic sub-score weights from EasyDesign data.

Currently the 6 heuristic sub-scores have hand-tuned weights:
  seed=0.35, gc=0.20, structure=0.20, homopolymer=0.10, offtarget=0.15

This script computes the 6 sub-scores for each EasyDesign guide and fits
Ridge regression to find optimal weights. Also tests L1 (Lasso) to see
if any sub-scores should be dropped entirely.

Usage:
    python scripts/research/learn_heuristic_weights.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EASYDESIGN_PATH = "compass-net/data/external/easydesign/Table_S2.xlsx"
RESULTS_DIR = Path("results/research/heuristic_weights")


def compute_gc_content(seq: str) -> float:
    gc = sum(1 for c in seq.upper() if c in "GC")
    return gc / len(seq) if seq else 0.0


def compute_homopolymer_max(seq: str) -> int:
    if not seq:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def compute_mfe_approx(seq: str) -> float:
    """Approximate MFE from GC content (when ViennaRNA not available)."""
    gc = compute_gc_content(seq)
    return -2.0 * gc  # rough approximation


def compute_heuristic_subscores(guide_seq: str, optimal_gc: float = 0.50) -> dict:
    """Compute the 6 heuristic sub-scores for a guide sequence.

    Mirrors the logic in compass/scoring/heuristic.py but standalone.
    """
    # Extract spacer (skip PAM)
    if len(guide_seq) >= 24:
        spacer = guide_seq[4:24]
    else:
        spacer = guide_seq[:20]

    # 1. Seed position score — for EasyDesign we don't have a specific
    #    SNP position, so use a neutral value based on spacer quality
    #    (EasyDesign guides are designed for general trans-cleavage, not
    #    mutation-specific discrimination). Use middle position as proxy.
    seed_score = 0.5  # neutral for non-discrimination data

    # 2. GC penalty
    gc = compute_gc_content(spacer)
    max_dev = 0.25
    gc_score = max(0.0, 1.0 - abs(gc - optimal_gc) / max_dev)

    # 3. Structure penalty (MFE)
    mfe = compute_mfe_approx(spacer)
    mfe_threshold = -2.0
    if mfe >= 0:
        structure_score = 1.0
    else:
        structure_score = max(0.0, 1.0 - mfe / mfe_threshold)

    # 4. Homopolymer penalty
    max_run = compute_homopolymer_max(spacer)
    homo_max = 4
    if max_run <= 1:
        homo_score = 1.0
    else:
        homo_score = max(0.0, 1.0 - (max_run - 1) / homo_max)

    # 5. Off-target score — we don't have off-target data for EasyDesign,
    #    so set to neutral (1.0 = no off-targets)
    offtarget_score = 1.0

    # 6. PAM activity weight
    pam = guide_seq[:4].upper() if len(guide_seq) >= 4 else "TTTN"
    if pam[:3] == "TTT" and pam[3] in "ACG":
        pam_score = 1.0
    elif pam[:3] == "TTT":
        pam_score = 0.8
    else:
        pam_score = 0.5

    return {
        "seed_position": seed_score,
        "gc": gc_score,
        "structure": structure_score,
        "homopolymer": homo_score,
        "offtarget": offtarget_score,
        "pam": pam_score,
        # Raw values for analysis
        "gc_raw": gc,
        "homo_raw": max_run,
        "mfe_raw": mfe,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load EasyDesign data
    df = pd.read_excel(EASYDESIGN_PATH, sheet_name="Training data")

    # Use only perfect-match guides (dist=0) for activity scoring
    perfect = df[df["guide_target_hamming_dist"] == 0].copy()
    logger.info("Perfect-match guides: %d", len(perfect))

    # Compute sub-scores
    features = []
    activities = []

    for _, row in perfect.iterrows():
        guide = str(row["guide_seq"]).upper()
        if len(guide) < 20:
            continue

        subscores = compute_heuristic_subscores(guide)
        features.append([
            subscores["gc"],
            subscores["structure"],
            subscores["homopolymer"],
            subscores["pam"],
            subscores["gc_raw"],
            subscores["homo_raw"],
        ])
        activities.append(float(row["30 min"]))

    X = np.array(features, dtype=np.float64)
    y = np.array(activities, dtype=np.float64)

    feature_names = ["gc_score", "structure_score", "homopolymer_score",
                     "pam_score", "gc_raw", "homo_raw"]

    logger.info("Feature matrix: %s, Target: %s", X.shape, y.shape)
    logger.info("Activity range: [%.3f, %.3f]", y.min(), y.max())

    # Individual correlations
    logger.info("\nIndividual feature correlations with activity:")
    for i, name in enumerate(feature_names):
        rho = spearmanr(X[:, i], y).statistic
        logger.info("  %-20s rho=%.4f", name, rho)

    # Use only the score features (not raw) for weight learning
    X_scores = X[:, :4]  # gc, structure, homopolymer, pam
    score_names = feature_names[:4]

    # Ridge regression to learn weights
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_scores)

    # Ridge (L2)
    ridge = Ridge(alpha=1.0)
    ridge_cv = cross_val_score(ridge, X_scaled, y, cv=5,
                                scoring="neg_mean_squared_error")
    ridge.fit(X_scaled, y)

    logger.info("\n=== Ridge Regression ===")
    logger.info("CV MSE: %.4f +/- %.4f", -ridge_cv.mean(), ridge_cv.std())

    # Get Spearman rho on full data
    ridge_preds = ridge.predict(X_scaled)
    ridge_rho = spearmanr(ridge_preds, y).statistic
    logger.info("Full-data Spearman rho: %.4f", ridge_rho)

    # Normalized weights (sum to 1)
    raw_weights = ridge.coef_ / scaler.scale_
    # Make all weights positive (flip sign if negative correlation helps)
    abs_weights = np.abs(raw_weights)
    norm_weights = abs_weights / abs_weights.sum()

    logger.info("\nLearned weights (Ridge):")
    for name, w, rw in zip(score_names, norm_weights, raw_weights):
        logger.info("  %-20s %.4f (raw coeff: %.4f)", name, w, rw)

    # Lasso (L1) — which features get zeroed out?
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_scaled, y)
    lasso_preds = lasso.predict(X_scaled)
    lasso_rho = spearmanr(lasso_preds, y).statistic

    logger.info("\n=== Lasso Regression ===")
    logger.info("Full-data Spearman rho: %.4f", lasso_rho)
    logger.info("Lasso coefficients (0 = dropped):")
    for name, c in zip(score_names, lasso.coef_):
        status = "KEPT" if abs(c) > 0.001 else "DROPPED"
        logger.info("  %-20s %.4f  [%s]", name, c, status)

    # Compare with current hand-tuned weights
    current_weights = {
        "gc": 0.20,
        "structure": 0.20,
        "homopolymer": 0.10,
        "pam": 0.15,  # implicit in multiplicative factor
    }

    # Also test: raw features (gc_raw, homo_raw) as direct regressors
    logger.info("\n=== Raw Features (not sub-scores) ===")
    X_raw = X[:, 4:]  # gc_raw, homo_raw
    raw_names = ["gc_raw", "homo_raw"]
    for i, name in enumerate(raw_names):
        rho = spearmanr(X_raw[:, i], y).statistic
        logger.info("  %-20s rho=%.4f", name, rho)

    # XGBoost on all 6 features
    try:
        from xgboost import XGBRegressor

        logger.info("\n=== XGBoost on all features ===")
        xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                           random_state=42, verbosity=0)
        xgb_cv = cross_val_score(xgb, X, y, cv=5, scoring="neg_mean_squared_error")
        xgb.fit(X, y)
        xgb_preds = xgb.predict(X)
        xgb_rho = spearmanr(xgb_preds, y).statistic
        logger.info("CV MSE: %.4f +/- %.4f", -xgb_cv.mean(), xgb_cv.std())
        logger.info("Full-data Spearman rho: %.4f", xgb_rho)

        logger.info("Feature importances:")
        for name, imp in zip(feature_names, xgb.feature_importances_):
            logger.info("  %-20s %.4f", name, imp)
    except ImportError:
        pass

    # Save results
    results = {
        "n_guides": len(features),
        "ridge_rho": round(float(ridge_rho), 4),
        "lasso_rho": round(float(lasso_rho), 4),
        "learned_weights": {name: round(float(w), 4) for name, w in zip(score_names, norm_weights)},
        "current_weights": current_weights,
        "individual_correlations": {
            name: round(float(spearmanr(X[:, i], y).statistic), 4)
            for i, name in enumerate(feature_names)
        },
    }

    with open(RESULTS_DIR / "learned_weights.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\nSaved to %s", RESULTS_DIR / "learned_weights.json")

    # Summary comparison
    logger.info("\n" + "=" * 60)
    logger.info("WEIGHT COMPARISON")
    logger.info("=" * 60)
    logger.info("%-20s %10s %10s", "Feature", "Current", "Learned")
    logger.info("-" * 45)
    for name in score_names:
        curr = current_weights.get(name.replace("_score", ""), 0)
        learned = dict(zip(score_names, norm_weights)).get(name, 0)
        logger.info("%-20s %10.3f %10.3f", name, curr, learned)


if __name__ == "__main__":
    main()
