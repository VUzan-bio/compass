"""Train discrimination prediction models with 5-fold stratified CV.

Trains both XGBoost and Neural models on EasyDesign paired data,
evaluates with stratified cross-validation (stratified by seed vs non-seed),
and compares against the heuristic baseline.

Outputs:
  - checkpoints/disc_xgb.pkl       — trained XGBoost model
  - checkpoints/disc_neural.pt     — trained neural model
  - checkpoints/disc_cv_results.json — CV metrics and feature importance

Usage:
    python compass-net/scripts/train_discrimination.py
    python compass-net/scripts/train_discrimination.py --folds 5 --no-neural

References:
  - Huang et al. (2024) iMeta — EasyDesign dataset
  - Zhang et al. (2024) NAR — R-loop energetics
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "compass-net" / "data"))
sys.path.insert(0, str(_PROJECT_ROOT / "compass-net"))

from extract_discrimination_pairs import extract_discrimination_pairs, pairs_to_dataframe
from thermo_discrimination_features import compute_features_batch, FEATURE_NAMES, compute_features_for_pair
from models.discrimination_model import FeatureDiscriminationModel, NeuralDiscriminationModel

logger = logging.getLogger(__name__)


def stratified_kfold_split(
    pairs: list,
    n_folds: int = 5,
    random_state: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Stratified K-fold by seed vs non-seed and guide identity.

    Ensures:
      1. Same guide never appears in both train and test (guide-level split)
      2. Seed/non-seed ratio preserved across folds
    """
    rng = np.random.RandomState(random_state)

    # Group pairs by guide
    guide_indices: dict[str, list[int]] = {}
    for i, p in enumerate(pairs):
        guide_indices.setdefault(p.guide_seq, []).append(i)

    # Classify guides by whether they have seed pairs
    guides_with_seed = []
    guides_without_seed = []
    for guide, idxs in guide_indices.items():
        has_seed = any(pairs[i].in_seed for i in idxs)
        if has_seed:
            guides_with_seed.append(guide)
        else:
            guides_without_seed.append(guide)

    rng.shuffle(guides_with_seed)
    rng.shuffle(guides_without_seed)

    # Distribute guides across folds
    fold_guides: list[list[str]] = [[] for _ in range(n_folds)]
    for i, g in enumerate(guides_with_seed):
        fold_guides[i % n_folds].append(g)
    for i, g in enumerate(guides_without_seed):
        fold_guides[i % n_folds].append(g)

    # Convert to index splits
    folds = []
    all_guides_set = set(g for gs in fold_guides for g in gs)
    for fold_idx in range(n_folds):
        test_guides = set(fold_guides[fold_idx])
        train_guides = all_guides_set - test_guides

        train_idxs = []
        test_idxs = []
        for guide, idxs in guide_indices.items():
            if guide in test_guides:
                test_idxs.extend(idxs)
            else:
                train_idxs.extend(idxs)

        folds.append((train_idxs, test_idxs))

    return folds


def compute_heuristic_baseline(pairs: list) -> dict:
    """Compute heuristic discrimination predictions for comparison.

    Uses the position_sensitivity × mismatch_destab model from
    compass/scoring/discrimination.py as the baseline.
    """
    from compass.candidates.synthetic_mismatch import (
        MISMATCH_DESTABILISATION,
        POSITION_SENSITIVITY_PROFILES,
        MismatchType,
        _predict_activity,
    )

    profile = POSITION_SENSITIVITY_PROFILES.get("LbCas12a", POSITION_SENSITIVITY_PROFILES["enAsCas12a"])

    y_true = []
    y_pred_heuristic = []

    for pair in pairs:
        pos = pair.spacer_position
        sensitivity = profile.get(pos, 0.05)

        # Look up mismatch destabilisation
        destab = 0.5
        for mt in MismatchType:
            if mt.value == pair.mismatch_type:
                destab = MISMATCH_DESTABILISATION.get(mt, 0.5)
                break

        # Seed floor
        if pos <= 8:
            destab = max(destab, 0.85)

        # Heuristic activity prediction
        wt_activity = _predict_activity([(pos, sensitivity, destab)], "LbCas12a")
        # MUT activity = 1.0 (perfect match)
        heuristic_ratio = 1.0 / max(wt_activity, 1e-6)

        y_true.append(pair.ratio_linear)
        y_pred_heuristic.append(heuristic_ratio)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred_heuristic)

    # Compute metrics in log space for fair comparison
    log_true = np.log10(np.clip(y_true, 1e-6, None))
    log_pred = np.log10(np.clip(y_pred, 1e-6, None))

    rmse = float(np.sqrt(np.mean((log_true - log_pred) ** 2)))
    corr = float(np.corrcoef(log_true, log_pred)[0, 1]) if len(log_true) > 2 else 0.0

    return {
        "rmse_log": rmse,
        "corr_log": corr,
        "n_samples": len(y_true),
        "median_pred_ratio": float(np.median(y_pred)),
        "median_true_ratio": float(np.median(y_true)),
    }


def run_cv(
    pairs: list,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    train_neural: bool = True,
) -> dict:
    """Run stratified cross-validation for both models."""
    folds = stratified_kfold_split(pairs, n_folds=n_folds)

    xgb_results = []
    neural_results = []

    for fold_idx, (train_idxs, test_idxs) in enumerate(folds):
        X_train, X_test = X[train_idxs], X[test_idxs]
        y_train, y_test = y[train_idxs], y[test_idxs]

        logger.info(
            "Fold %d/%d: train=%d, test=%d",
            fold_idx + 1, n_folds, len(train_idxs), len(test_idxs),
        )

        # XGBoost
        xgb_model = FeatureDiscriminationModel()
        xgb_model.fit(X_train, y_train, feature_names=FEATURE_NAMES, X_val=X_test, y_val=y_test)
        y_pred_xgb = xgb_model.predict(X_test)

        rmse_xgb = float(np.sqrt(np.mean((y_test - y_pred_xgb) ** 2)))
        corr_xgb = float(np.corrcoef(y_test, y_pred_xgb)[0, 1]) if len(y_test) > 2 else 0.0

        # Ratio-space metrics
        true_ratios = np.power(10, y_test)
        pred_ratios_xgb = np.power(10, y_pred_xgb)
        ratio_corr_xgb = float(np.corrcoef(true_ratios, pred_ratios_xgb)[0, 1]) if len(y_test) > 2 else 0.0

        xgb_results.append({
            "fold": fold_idx,
            "rmse": rmse_xgb,
            "corr": corr_xgb,
            "ratio_corr": ratio_corr_xgb,
            "n_test": len(test_idxs),
        })
        logger.info("  XGBoost: RMSE=%.4f, r=%.4f, ratio_r=%.4f", rmse_xgb, corr_xgb, ratio_corr_xgb)

        # Neural
        if train_neural:
            neural_model = NeuralDiscriminationModel(input_dim=X.shape[1])
            neural_model.fit(X_train, y_train, X_val=X_test, y_val=y_test, epochs=80, patience=12)
            y_pred_neural = neural_model.predict(X_test)

            rmse_neural = float(np.sqrt(np.mean((y_test - y_pred_neural) ** 2)))
            corr_neural = float(np.corrcoef(y_test, y_pred_neural)[0, 1]) if len(y_test) > 2 else 0.0

            pred_ratios_nn = np.power(10, y_pred_neural)
            ratio_corr_nn = float(np.corrcoef(true_ratios, pred_ratios_nn)[0, 1]) if len(y_test) > 2 else 0.0

            neural_results.append({
                "fold": fold_idx,
                "rmse": rmse_neural,
                "corr": corr_neural,
                "ratio_corr": ratio_corr_nn,
                "n_test": len(test_idxs),
            })
            logger.info("  Neural:  RMSE=%.4f, r=%.4f, ratio_r=%.4f", rmse_neural, corr_neural, ratio_corr_nn)

    # Aggregate
    def aggregate(results):
        if not results:
            return {}
        return {
            "mean_rmse": float(np.mean([r["rmse"] for r in results])),
            "std_rmse": float(np.std([r["rmse"] for r in results])),
            "mean_corr": float(np.mean([r["corr"] for r in results])),
            "std_corr": float(np.std([r["corr"] for r in results])),
            "mean_ratio_corr": float(np.mean([r["ratio_corr"] for r in results])),
            "per_fold": results,
        }

    return {
        "xgboost": aggregate(xgb_results),
        "neural": aggregate(neural_results) if train_neural else {},
    }


def train_final_models(
    X: np.ndarray,
    y: np.ndarray,
    checkpoint_dir: Path,
    train_neural: bool = True,
) -> dict:
    """Train final models on all data and save checkpoints."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # XGBoost on full data
    xgb_model = FeatureDiscriminationModel()
    xgb_metrics = xgb_model.fit(X, y, feature_names=FEATURE_NAMES)
    xgb_model.save(checkpoint_dir / "disc_xgb.pkl")

    result = {"xgboost": xgb_metrics}

    # Neural on full data
    if train_neural:
        # Use 10% as validation for early stopping
        n = len(X)
        perm = np.random.RandomState(42).permutation(n)
        split = int(0.9 * n)
        X_train, X_val = X[perm[:split]], X[perm[split:]]
        y_train, y_val = y[perm[:split]], y[perm[split:]]

        neural_model = NeuralDiscriminationModel(input_dim=X.shape[1])
        neural_metrics = neural_model.fit(X_train, y_train, X_val=X_val, y_val=y_val, epochs=100)
        neural_model.save(checkpoint_dir / "disc_neural.pt")
        result["neural"] = neural_metrics

    return result


def main():
    parser = argparse.ArgumentParser(description="Train discrimination prediction models")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--no-neural", action="store_true", help="Skip neural model training")
    parser.add_argument("--checkpoint-dir", type=str, default="compass-net/checkpoints",
                        help="Directory for model checkpoints")
    parser.add_argument("--skip-cv", action="store_true", help="Skip CV, only train final models")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    train_neural = not args.no_neural

    # ── Step 1: Extract pairs ──
    logger.info("=" * 60)
    logger.info("Step 1: Extracting discrimination pairs from EasyDesign")
    logger.info("=" * 60)
    t0 = time.time()
    pairs = extract_discrimination_pairs()
    logger.info("Extracted %d pairs in %.1fs", len(pairs), time.time() - t0)

    # ── Step 2: Compute features ──
    logger.info("=" * 60)
    logger.info("Step 2: Computing thermodynamic features")
    logger.info("=" * 60)
    X, y = compute_features_batch(pairs, cas_variant="LbCas12a")
    logger.info("Feature matrix: %s, Target: %s", X.shape, y.shape)

    # ── Step 3: Heuristic baseline ──
    logger.info("=" * 60)
    logger.info("Step 3: Computing heuristic baseline")
    logger.info("=" * 60)
    heuristic = compute_heuristic_baseline(pairs)
    logger.info("Heuristic baseline: RMSE_log=%.4f, r_log=%.4f", heuristic["rmse_log"], heuristic["corr_log"])

    # ── Step 4: Cross-validation ──
    cv_results = {}
    if not args.skip_cv:
        logger.info("=" * 60)
        logger.info("Step 4: %d-fold stratified cross-validation", args.folds)
        logger.info("=" * 60)
        t0 = time.time()
        cv_results = run_cv(pairs, X, y, n_folds=args.folds, train_neural=train_neural)
        cv_time = time.time() - t0
        logger.info("CV completed in %.1fs", cv_time)

        logger.info("\n" + "=" * 60)
        logger.info("CV RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info("Heuristic baseline:  RMSE=%.4f, r=%.4f", heuristic["rmse_log"], heuristic["corr_log"])
        if cv_results.get("xgboost"):
            xgb = cv_results["xgboost"]
            logger.info("XGBoost (5-fold):    RMSE=%.4f±%.4f, r=%.4f±%.4f",
                        xgb["mean_rmse"], xgb["std_rmse"], xgb["mean_corr"], xgb["std_corr"])
        if cv_results.get("neural"):
            nn = cv_results["neural"]
            logger.info("Neural (5-fold):     RMSE=%.4f±%.4f, r=%.4f±%.4f",
                        nn["mean_rmse"], nn["std_rmse"], nn["mean_corr"], nn["std_corr"])

    # ── Step 5: Train final models ──
    logger.info("=" * 60)
    logger.info("Step 5: Training final models on all data")
    logger.info("=" * 60)
    final_metrics = train_final_models(X, y, checkpoint_dir, train_neural=train_neural)

    # ── Step 6: Save results ──
    results = {
        "dataset": {
            "name": "EasyDesign",
            "n_pairs": len(pairs),
            "n_guides": len(set(p.guide_seq for p in pairs)),
            "n_seed_pairs": sum(1 for p in pairs if p.in_seed),
            "cas_variant": "LbCas12a",
        },
        "features": {
            "names": FEATURE_NAMES,
            "n_features": len(FEATURE_NAMES),
        },
        "heuristic_baseline": heuristic,
        "cv_results": cv_results,
        "final_models": final_metrics,
    }

    results_path = checkpoint_dir / "disc_cv_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved results to %s", results_path)

    # Feature importance summary
    if final_metrics.get("xgboost", {}).get("feature_importance"):
        fi = final_metrics["xgboost"]["feature_importance"]
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        logger.info("\nFeature importance (XGBoost, top 10):")
        for name, imp in sorted_fi[:10]:
            logger.info("  %25s: %.4f", name, imp)

    logger.info("\nDone. Models saved to %s/", checkpoint_dir)


if __name__ == "__main__":
    main()
