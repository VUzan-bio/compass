"""Temperature calibration and ensemble weight optimisation for Compass-ML.

Same approach as SeqCNN calibration (calibrate.py) but adapted for the
Compass-ML dual-branch architecture. The model's sigmoid output saturates
similarly, so we apply temperature scaling to spread predictions.

The ensemble combines calibrated Compass-ML scores with heuristic scores
using weight alpha, found by maximising Spearman rho on the validation set.

Usage:
    python -m compass.scoring.calibrate_compass_ml [--weights PATH] [--data PATH]

If no validation data is available yet, generates a default calibration
from the model's output distribution on Kim 2018 data (same data used
for SeqCNN, allows direct comparison).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "compass_ml_diagnostic.pt"
_DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "weights" / "compass_ml_calibration.json"


def _predict_batch_compass_ml(
    model,
    X: np.ndarray,
    device,
    use_rnafm: bool = False,
    batch_size: int = 512,
) -> np.ndarray:
    """Run Compass-ML inference on one-hot encoded sequences.

    Returns raw sigmoid outputs (not logits) since Compass-ML's
    architecture doesn't expose pre-sigmoid logits as cleanly as SeqCNN.
    We calibrate in probability space using Platt-style temperature scaling.

    When use_rnafm=True, passes zero embeddings (no cache available for
    calibration data). The model degrades gracefully — CNN branch dominates.
    """
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            rnafm_emb = None
            if use_rnafm:
                # Zero embeddings — no RNA-FM cache for calibration data
                rnafm_emb = torch.zeros(batch.shape[0], 20, 640, device=device)
            output = model(target_onehot=batch, crrna_rnafm_emb=rnafm_emb)
            preds.append(output["efficiency"].squeeze(-1).cpu().numpy())
    return np.concatenate(preds).flatten()


def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Inverse sigmoid: logit(p) = log(p / (1-p))."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def find_optimal_temperature(
    raw_scores: np.ndarray,
    y: np.ndarray,
) -> float:
    """Find temperature T that matches calibrated score distribution to target.

    Applies: calibrated = sigmoid(logit(raw) / T)

    Since Spearman rho is rank-invariant (monotonic transforms don't change it),
    we instead find T such that the calibrated score distribution matches the
    target activity distribution in terms of spread. This ensures Block 3
    thresholds (efficiency >= 0.3/0.4/0.6) remain meaningful.

    Objective: minimise MSE between calibrated quantiles and target quantiles
    at [10th, 25th, 50th, 75th, 90th] percentiles.
    """
    logits = _logit(raw_scores)
    quantile_points = [0.10, 0.25, 0.50, 0.75, 0.90]
    y_quantiles = np.quantile(y, quantile_points)

    def quantile_mse(T: float) -> float:
        calibrated = _sigmoid(logits / T)
        cal_quantiles = np.quantile(calibrated, quantile_points)
        return float(np.mean((cal_quantiles - y_quantiles) ** 2))

    result = minimize_scalar(quantile_mse, bounds=(0.1, 20.0), method="bounded")
    return float(result.x)


def find_optimal_alpha(
    heuristic_scores: np.ndarray,
    gn_calibrated: np.ndarray,
    y: np.ndarray,
) -> float:
    """Find alpha that maximises Spearman rho of the ensemble.

    combined = alpha * heuristic + (1 - alpha) * calibrated_gn
    """
    def neg_spearman(alpha: float) -> float:
        combined = alpha * heuristic_scores + (1 - alpha) * gn_calibrated
        rho, _ = spearmanr(combined, y)
        return -rho

    result = minimize_scalar(neg_spearman, bounds=(0.0, 1.0), method="bounded")
    return float(result.x)


def _compute_heuristic_scores(X: np.ndarray) -> np.ndarray:
    """Compute simplified heuristic scores from one-hot sequences.

    Same implementation as calibrate.py for consistency.
    """
    scores = []
    bases = "ACGT"
    for i in range(len(X)):
        seq_indices = X[i].argmax(axis=0)
        seq = "".join(bases[idx] for idx in seq_indices)
        spacer = seq[8:28]

        gc = (spacer.count("G") + spacer.count("C")) / len(spacer)
        gc_score = max(0, 1.0 - abs(gc - 0.5) * 4)

        max_run = 1
        run = 1
        for j in range(1, len(spacer)):
            if spacer[j] == spacer[j - 1]:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        homo_score = 1.0 if max_run < 4 else max(0, 1.0 - (max_run - 3) * 0.3)

        h_score = 0.5 * gc_score + 0.3 * homo_score + 0.2 * 0.5
        scores.append(max(0.0, min(1.0, h_score)))

    return np.array(scores, dtype=np.float32)


def calibrate_compass_ml(
    weights_path: str | Path = _DEFAULT_WEIGHTS,
    data_path: str | Path = "compass/data/kim2018/nbt4061_source_data.xlsx",
    output_path: str | Path = _DEFAULT_OUTPUT,
    use_rnafm: bool = False,
    use_rlpa: bool = True,
) -> dict:
    """Run full Compass-ML calibration: find optimal T and alpha, save to JSON.

    Args:
        weights_path: Path to compass_ml_best.pt checkpoint.
        data_path: Path to Kim 2018 source data (Excel).
        output_path: Where to save calibration JSON.
        use_rnafm: Whether to use RNA-FM embeddings (requires cache).
        use_rlpa: Whether model was trained with RLPA.
    """
    import sys
    import importlib
    import torch
    from compass.scoring.data_loader import load_kim2018

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import Compass-ML model
    compass_ml_dir = Path(__file__).resolve().parent.parent.parent / "compass-net"
    compass_ml_str = str(compass_ml_dir)
    if "compass_ml" not in sys.modules:
        if compass_ml_str not in sys.path:
            sys.path.insert(0, compass_ml_str)
        spec = importlib.util.spec_from_file_location(
            "compass_ml",
            str(compass_ml_dir / "__init__.py"),
            submodule_search_locations=[compass_ml_str],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["compass_ml"] = mod
        spec.loader.exec_module(mod)

    from compass_ml import CompassML

    # Load checkpoint first to detect architecture
    checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Auto-detect use_rnafm from checkpoint: if rnafm branch weights exist, enable it
    ckpt_has_rnafm = any(k.startswith("rnafm.") for k in state_dict)
    if ckpt_has_rnafm and not use_rnafm:
        logger.info("Checkpoint has RNA-FM weights — enabling use_rnafm=True")
        use_rnafm = True

    # Load model with correct architecture
    model = CompassML(
        use_rnafm=use_rnafm,
        use_rloop_attention=use_rlpa,
        multitask=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    val_rho_from_ckpt = checkpoint.get("val_rho") or checkpoint.get("best_val_rho")

    # Load Kim 2018 data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_kim2018(data_path)

    # Get raw predictions on validation set
    val_raw = _predict_batch_compass_ml(model, X_val, device, use_rnafm=use_rnafm)

    # Find optimal temperature
    optimal_T = find_optimal_temperature(val_raw, y_val)

    # Apply calibration
    val_logits = _logit(val_raw)
    gn_calibrated = _sigmoid(val_logits / optimal_T)

    # Compute heuristic scores
    heuristic_scores = _compute_heuristic_scores(X_val)

    # Find optimal ensemble weight
    optimal_alpha = find_optimal_alpha(heuristic_scores, gn_calibrated, y_val)

    # Compute all metrics
    ensemble = optimal_alpha * heuristic_scores + (1 - optimal_alpha) * gn_calibrated

    raw_rho, _ = spearmanr(val_raw, y_val)
    cal_rho, _ = spearmanr(gn_calibrated, y_val)
    heur_rho, _ = spearmanr(heuristic_scores, y_val)
    ens_rho, _ = spearmanr(ensemble, y_val)

    # Test set metrics
    test_raw = _predict_batch_compass_ml(model, X_test, device, use_rnafm=use_rnafm)
    test_logits = _logit(test_raw)
    test_calibrated = _sigmoid(test_logits / optimal_T)
    test_heuristic = _compute_heuristic_scores(X_test)
    test_ensemble = optimal_alpha * test_heuristic + (1 - optimal_alpha) * test_calibrated

    test_raw_rho, _ = spearmanr(test_raw, y_test)
    test_cal_rho, _ = spearmanr(test_calibrated, y_test)
    test_ens_rho, _ = spearmanr(test_ensemble, y_test)

    calibration = {
        "model": "compass_ml",
        "temperature": round(optimal_T, 4),
        "alpha": round(optimal_alpha, 4),
        "use_rnafm": use_rnafm,
        "use_rlpa": use_rlpa,
        "val_rho_gn_raw": round(float(raw_rho), 4),
        "val_rho_gn_calibrated": round(float(cal_rho), 4),
        "val_rho_heuristic": round(float(heur_rho), 4),
        "val_rho_ensemble": round(float(ens_rho), 4),
        "test_rho_gn_raw": round(float(test_raw_rho), 4),
        "test_rho_gn_calibrated": round(float(test_cal_rho), 4),
        "test_rho_ensemble": round(float(test_ens_rho), 4),
        "val_gn_raw_range": [round(float(val_raw.min()), 4), round(float(val_raw.max()), 4)],
        "val_gn_calibrated_range": [round(float(gn_calibrated.min()), 4), round(float(gn_calibrated.max()), 4)],
        "val_rho_from_checkpoint": round(float(val_rho_from_ckpt), 4) if val_rho_from_ckpt else None,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nCompass-ML Temperature calibration:")
    print(f"  Optimal T = {optimal_T:.2f}")
    print(f"  Raw GN:        range [{val_raw.min():.3f}, {val_raw.max():.3f}], val rho = {raw_rho:.4f}")
    print(f"  Calibrated GN: range [{gn_calibrated.min():.3f}, {gn_calibrated.max():.3f}], val rho = {cal_rho:.4f}")
    print(f"\nEnsemble calibration:")
    print(f"  Optimal alpha = {optimal_alpha:.3f}")
    print(f"  Heuristic alone: val rho = {heur_rho:.4f}")
    print(f"  GN calibrated:   val rho = {cal_rho:.4f}")
    print(f"  Ensemble:        val rho = {ens_rho:.4f}")
    print(f"\nTest set:")
    print(f"  GN raw:          test rho = {test_raw_rho:.4f}")
    print(f"  GN calibrated:   test rho = {test_cal_rho:.4f}")
    print(f"  Ensemble:        test rho = {test_ens_rho:.4f}")
    print(f"\nSaved to {output_path}")

    return calibration


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Calibrate Compass-ML scores")
    parser.add_argument("--weights", default=str(_DEFAULT_WEIGHTS))
    parser.add_argument("--data", default="compass/data/kim2018/nbt4061_source_data.xlsx")
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--rlpa", action="store_true", default=True)
    parser.add_argument("--no-rlpa", dest="rlpa", action="store_false")
    args = parser.parse_args()
    calibrate_compass_ml(
        weights_path=args.weights,
        data_path=args.data,
        output_path=args.output,
        use_rlpa=args.rlpa,
    )
