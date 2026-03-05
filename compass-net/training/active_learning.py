"""Active learning loop for Compass-ML domain adaptation.

Workflow:
    1. MC-dropout inference -> estimate uncertainty per candidate
    2. Select batch via UCB acquisition + sequence diversity
    3. Measure trans-cleavage on electrochemical platform
    4. Fine-tune Compass-ML on augmented dataset (Phase 3)
    5. Recompute Pareto frontier -> track improvement
    6. Repeat 3-4 cycles

Acquisition function: Upper Confidence Bound (UCB)
    score = mu + beta * sigma
    where beta=2.0 is the standard GP-UCB default (Srinivas et al., ICML 2010).

Batch diversity: greedy farthest-point selection in Hamming distance space
to avoid clustering selections in the same sequence region.

References:
    Srinivas et al., "GP Optimization in the Bandit Setting." ICML 2010.
    Desautels et al., "Parallelizing Exploration-Exploitation." JMLR 2014.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class UncertaintyEstimate:
    candidate_id: str
    target_label: str
    sequence: str
    mean_efficiency: float
    std_efficiency: float
    mean_discrimination: float
    std_discrimination: float
    acquisition_score: float


def mc_dropout_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    n_samples: int = 50,
    device: str | torch.device = "cuda",
    embedding_cache=None,
) -> list[UncertaintyEstimate]:
    """MC-dropout: keep dropout ON during inference, run N forward passes.

    Variance across passes = epistemic (model) uncertainty.
    High uncertainty = model hasn't seen similar inputs = maximum
    learning signal from experimental validation.

    Args:
        model: CompassML (must have dropout layers).
        loader: DataLoader yielding candidate batches.
        n_samples: number of stochastic forward passes.
        device: torch device.
        embedding_cache: RNA-FM embedding cache.

    Returns:
        List of UncertaintyEstimate sorted by acquisition score (descending).
    """
    model.train()  # keep dropout active

    results: list[UncertaintyEstimate] = []

    for batch in loader:
        target_onehot = batch["target_onehot"].to(device)
        batch_size = target_onehot.size(0)

        # Get crRNA embeddings if available
        crrna_emb = None
        if embedding_cache is not None and "crrna_spacer" in batch:
            from .train_compass_ml import _get_batch_embeddings
            crrna_emb = _get_batch_embeddings(
                batch["crrna_spacer"], embedding_cache, device,
            )

        # Collect N stochastic predictions
        eff_samples = torch.zeros(n_samples, batch_size)
        with torch.no_grad():
            for s in range(n_samples):
                out = model(
                    target_onehot=target_onehot,
                    crrna_rnafm_emb=crrna_emb,
                )
                eff_samples[s] = out["efficiency"].squeeze(-1).cpu()

        # Compute mean and std across samples
        means = eff_samples.mean(dim=0)
        stds = eff_samples.std(dim=0)

        for i in range(batch_size):
            results.append(UncertaintyEstimate(
                candidate_id=batch.get("candidate_id", [""] * batch_size)[i],
                target_label=batch.get("target_label", [""] * batch_size)[i],
                sequence=batch.get("sequence", [""] * batch_size)[i],
                mean_efficiency=means[i].item(),
                std_efficiency=stds[i].item(),
                mean_discrimination=0.0,
                std_discrimination=0.0,
                acquisition_score=0.0,  # computed below
            ))

    model.eval()
    return results


def compute_ucb_scores(
    estimates: list[UncertaintyEstimate],
    beta: float = 2.0,
) -> list[UncertaintyEstimate]:
    """Compute UCB acquisition scores.

    UCB = mu + beta * sigma (Srinivas et al., ICML 2010)

    Args:
        estimates: list from mc_dropout_inference.
        beta: exploration-exploitation tradeoff.
            beta=0: pure exploitation (pick highest predicted)
            beta->inf: pure exploration (pick most uncertain)
            beta=2.0: standard GP-UCB default.

    Returns:
        Same list with acquisition_score populated, sorted descending.
    """
    for est in estimates:
        est.acquisition_score = est.mean_efficiency + beta * est.std_efficiency

    return sorted(estimates, key=lambda x: x.acquisition_score, reverse=True)


def select_diverse_batch(
    estimates: list[UncertaintyEstimate],
    batch_size: int = 10,
    beta: float = 2.0,
    diversity_weight: float = 0.3,
) -> list[UncertaintyEstimate]:
    """Select a diverse batch of candidates for experimental validation.

    Uses greedy farthest-point selection to ensure sequence diversity,
    weighted against the UCB acquisition score.

    Algorithm:
        1. Compute UCB scores for all candidates
        2. Pick the top candidate by UCB
        3. For each remaining slot: pick the candidate that maximises
           acquisition_score + diversity_weight * min_hamming_to_selected

    This prevents the batch from clustering in one region of sequence
    space, which would waste experimental budget.

    Args:
        estimates: uncertainty estimates from MC-dropout.
        batch_size: number of candidates to select.
        beta: UCB beta parameter.
        diversity_weight: weight for sequence diversity term.

    Returns:
        Selected candidates (batch_size or fewer).
    """
    if len(estimates) <= batch_size:
        return estimates

    # Compute UCB
    estimates = compute_ucb_scores(estimates, beta)

    # Greedy farthest-point selection
    selected: list[UncertaintyEstimate] = [estimates[0]]
    remaining = list(estimates[1:])

    for _ in range(batch_size - 1):
        if not remaining:
            break

        best_score = -float("inf")
        best_idx = 0

        for i, cand in enumerate(remaining):
            # Minimum Hamming distance to any selected candidate
            min_dist = min(
                _hamming_distance(cand.sequence, s.sequence)
                for s in selected
            ) if selected else 0

            # Combined score: UCB + diversity bonus
            combined = cand.acquisition_score + diversity_weight * min_dist
            if combined > best_score:
                best_score = combined
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def _hamming_distance(s1: str, s2: str) -> int:
    """Hamming distance between two sequences of equal length."""
    if not s1 or not s2:
        return 0
    min_len = min(len(s1), len(s2))
    return sum(a != b for a, b in zip(s1[:min_len], s2[:min_len]))
