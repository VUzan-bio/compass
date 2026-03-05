"""Temperature calibration and ensemble weight optimisation for SeqCNN.

The CNN sigmoid saturates outputs to 0.9–1.0. Temperature scaling divides
logits by T > 1 before sigmoid, spreading the distribution. The ensemble
combines calibrated CNN scores with heuristic scores using weight α.

Both T and α are found by maximising Spearman ρ on the validation set.

Usage:
    python -m compass.scoring.calibrate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def _extract_logits(model, X: np.ndarray, device, batch_size: int = 512) -> np.ndarray:
    """Extract pre-sigmoid logits from a trained SeqCNN."""
    import torch

    model.eval()
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).to(device)
            h = model.multi_scale(batch)
            h = model.dilated(h)
            h = model.reduce(h)
            h = model.pool(h).squeeze(-1)
            # Forward through head layers EXCEPT the final Sigmoid
            head_modules = list(model.head.children())
            for layer in head_modules[:-1]:
                h = layer(h)
            logits_list.append(h.cpu().numpy())
    return np.concatenate(logits_list).flatten()


def find_optimal_temperature(
    logits: np.ndarray,
    y: np.ndarray,
) -> float:
    """Find temperature T that maximises Spearman ρ on validation set.

    Searches T ∈ [0.5, 10.0]. Higher T = more spread.
    """
    def neg_spearman(T):
        calibrated = 1.0 / (1.0 + np.exp(-logits / T))
        rho, _ = spearmanr(calibrated, y)
        return -rho

    result = minimize_scalar(neg_spearman, bounds=(0.5, 10.0), method="bounded")
    return float(result.x)


def find_optimal_alpha(
    heuristic_scores: np.ndarray,
    cnn_calibrated: np.ndarray,
    y: np.ndarray,
) -> float:
    """Find α that maximises Spearman ρ of the ensemble.

    combined = α × heuristic + (1 - α) × calibrated_cnn
    """
    def neg_spearman(alpha):
        combined = alpha * heuristic_scores + (1 - alpha) * cnn_calibrated
        rho, _ = spearmanr(combined, y)
        return -rho

    result = minimize_scalar(neg_spearman, bounds=(0.0, 1.0), method="bounded")
    return float(result.x)


def _compute_heuristic_scores(X: np.ndarray) -> np.ndarray:
    """Compute simplified heuristic scores from one-hot encoded sequences.

    Since the heuristic scorer needs CrRNACandidate objects, we compute
    a simplified version using GC content and homopolymer features.
    """
    scores = []
    bases = "ACGT"
    for i in range(len(X)):
        seq_indices = X[i].argmax(axis=0)  # (34,) indices 0-3
        seq = "".join(bases[idx] for idx in seq_indices)
        spacer = seq[8:28]  # PAM at 4:8, spacer at 8:28

        # GC content score
        gc = (spacer.count("G") + spacer.count("C")) / len(spacer)
        gc_score = max(0, 1.0 - abs(gc - 0.5) * 4)

        # Homopolymer score
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


def calibrate_and_save(
    cnn_weights: str = "compass/weights/seq_cnn_best.pt",
    data_path: str = "compass/data/kim2018/nbt4061_source_data.xlsx",
    output_path: str = "compass/weights/calibration.json",
) -> dict:
    """Run full calibration: find optimal T and α, save to JSON."""
    import torch
    from compass.scoring.data_loader import load_kim2018
    from compass.scoring.seq_cnn import SeqCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CNN
    model = SeqCNN()
    checkpoint = torch.load(cnn_weights, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load validation data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_kim2018(data_path)

    # Extract logits for validation set
    val_logits = _extract_logits(model, X_val, device)

    # Find optimal temperature
    optimal_T = find_optimal_temperature(val_logits, y_val)

    # Compute calibrated CNN scores
    cnn_calibrated = 1.0 / (1.0 + np.exp(-val_logits / optimal_T))

    # Compute heuristic scores for validation set
    heuristic_scores = _compute_heuristic_scores(X_val)

    # Find optimal ensemble weight
    optimal_alpha = find_optimal_alpha(heuristic_scores, cnn_calibrated, y_val)

    # Compute all metrics
    raw_scores = 1.0 / (1.0 + np.exp(-val_logits))
    ensemble = optimal_alpha * heuristic_scores + (1 - optimal_alpha) * cnn_calibrated

    raw_rho, _ = spearmanr(raw_scores, y_val)
    cal_rho, _ = spearmanr(cnn_calibrated, y_val)
    heur_rho, _ = spearmanr(heuristic_scores, y_val)
    ens_rho, _ = spearmanr(ensemble, y_val)

    # Also compute test set metrics
    test_logits = _extract_logits(model, X_test, device)
    test_calibrated = 1.0 / (1.0 + np.exp(-test_logits / optimal_T))
    test_heuristic = _compute_heuristic_scores(X_test)
    test_ensemble = optimal_alpha * test_heuristic + (1 - optimal_alpha) * test_calibrated

    test_raw_rho, _ = spearmanr(1.0 / (1.0 + np.exp(-test_logits)), y_test)
    test_cal_rho, _ = spearmanr(test_calibrated, y_test)
    test_ens_rho, _ = spearmanr(test_ensemble, y_test)

    calibration = {
        "temperature": round(optimal_T, 4),
        "alpha": round(optimal_alpha, 4),
        "val_rho_cnn_raw": round(float(raw_rho), 4),
        "val_rho_cnn_calibrated": round(float(cal_rho), 4),
        "val_rho_heuristic": round(float(heur_rho), 4),
        "val_rho_ensemble": round(float(ens_rho), 4),
        "test_rho_cnn_raw": round(float(test_raw_rho), 4),
        "test_rho_cnn_calibrated": round(float(test_cal_rho), 4),
        "test_rho_ensemble": round(float(test_ens_rho), 4),
        "val_cnn_raw_range": [round(float(raw_scores.min()), 4), round(float(raw_scores.max()), 4)],
        "val_cnn_calibrated_range": [round(float(cnn_calibrated.min()), 4), round(float(cnn_calibrated.max()), 4)],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nTemperature calibration:")
    print(f"  Optimal T = {optimal_T:.2f}")
    print(f"  Raw CNN:        range [{raw_scores.min():.3f}, {raw_scores.max():.3f}], val rho = {raw_rho:.4f}")
    print(f"  Calibrated CNN: range [{cnn_calibrated.min():.3f}, {cnn_calibrated.max():.3f}], val rho = {cal_rho:.4f}")
    print(f"\nEnsemble calibration:")
    print(f"  Optimal alpha = {optimal_alpha:.3f}")
    print(f"  Heuristic alone: val rho = {heur_rho:.4f}")
    print(f"  CNN calibrated:  val rho = {cal_rho:.4f}")
    print(f"  Ensemble:        val rho = {ens_rho:.4f}")
    print(f"\nTest set:")
    print(f"  CNN raw:         test rho = {test_raw_rho:.4f}")
    print(f"  CNN calibrated:  test rho = {test_cal_rho:.4f}")
    print(f"  Ensemble:        test rho = {test_ens_rho:.4f}")
    print(f"\nSaved to {output_path}")

    return calibration


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    calibrate_and_save()
