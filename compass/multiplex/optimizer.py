"""Multiplex panel optimization via simulated annealing.

Selects the best crRNA candidate for each target in a multiplexed
diagnostic panel, optimizing three objectives simultaneously:

  1. Efficiency: prefer candidates with highest predicted cleavage activity
  2. Discrimination: prefer candidates with highest MUT/WT ratio
  3. Cross-reactivity: avoid candidate pairs whose spacers are similar
     enough to cause off-target trans-cleavage within the same reaction

The optimization is NP-hard (combinatorial selection from N candidates
per M targets), solved here with simulated annealing. For a 14-plex
panel with ~20 candidates per target, the search space is ~20^14 ≈ 10^18.

Simulated annealing parameters are tuned for convergence within
10,000 iterations, which takes <1 second on any modern CPU.

Cross-reactivity scoring:
  Two crRNA spacers with ≤4 mismatches risk cross-activation in the
  same reaction. The cross-reactivity matrix C[i,j] estimates the
  probability that crRNA_i cleaves crRNA_j's amplicon, based on
  sequence similarity in the seed region (positions 1-8).

References:
  - Kirkpatrick et al., Science 1983 (simulated annealing)
  - Gootenberg et al., Science 2018 (SHERLOCKv2 multiplexing)
  - Li et al., Cell Rep Med 2022 (CRISPR multiplex design)

"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from compass.core.types import (
    CrRNACandidate,
    DetectionStrategy,
    MultiplexPanel,
    PanelMember,
    ScoredCandidate,
    Target,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Simulated annealing parameters."""

    max_iterations: int = 10_000
    initial_temperature: float = 1.0
    cooling_rate: float = 0.9995  # temperature *= cooling_rate each step
    min_temperature: float = 0.001

    # Objective weights (must sum to ~1.0)
    efficiency_weight: float = 0.4
    discrimination_weight: float = 0.3
    cross_reactivity_weight: float = 0.3

    # Cross-reactivity
    cross_reactivity_threshold: float = 0.3  # max allowed cross-reactivity score
    seed_region_end: int = 8  # positions 1-8 are seed region

    # Seed for reproducibility
    random_seed: Optional[int] = 42


def _spacer_similarity(seq_a: str, seq_b: str, seed_end: int = 8) -> float:
    """Compute sequence similarity between two spacers.

    Weighted: seed mismatches count 3× more than non-seed mismatches.
    Returns a cross-reactivity risk score in [0, 1] where:
      0 = completely different (no risk)
      1 = identical (maximum risk)
    """
    min_len = min(len(seq_a), len(seq_b))
    if min_len == 0:
        return 0.0

    weighted_matches = 0.0
    weighted_total = 0.0

    for i in range(min_len):
        pos = i + 1  # 1-indexed
        weight = 3.0 if pos <= seed_end else 1.0
        weighted_total += weight
        if seq_a[i].upper() == seq_b[i].upper():
            weighted_matches += weight

    return weighted_matches / weighted_total if weighted_total > 0 else 0.0


class MultiplexOptimizer:
    """Optimize candidate selection across a multiplex panel.

    Given a set of scored candidates per target, select one candidate
    per target that maximizes panel-level quality while minimizing
    cross-reactivity between selected crRNAs.

    Usage:
        optimizer = MultiplexOptimizer()
        panel = optimizer.optimize(
            targets=targets,
            candidates_by_target=candidates_per_target,
        )

    The optimizer handles targets with 0 candidates gracefully —
    they are reported as gaps in the panel.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None) -> None:
        self.config = config or OptimizationConfig()

    def optimize(
        self,
        targets: list[Target],
        candidates_by_target: dict[str, list[ScoredCandidate]],
    ) -> MultiplexPanel:
        """Run simulated annealing to find the optimal candidate selection.

        Args:
            targets: List of Target objects for the panel.
            candidates_by_target: {target_label: [ScoredCandidate, ...]}.
                Candidates should already be scored (heuristic + discrimination).

        Returns:
            MultiplexPanel with the selected candidates and quality matrices.
        """
        cfg = self.config

        if cfg.random_seed is not None:
            random.seed(cfg.random_seed)

        # Filter to targets with at least 1 candidate
        active_labels = [
            t.label for t in targets if candidates_by_target.get(t.label)
        ]
        empty_labels = [
            t.label for t in targets if not candidates_by_target.get(t.label)
        ]

        if empty_labels:
            logger.warning(
                "Targets with 0 candidates (will be panel gaps): %s",
                ", ".join(empty_labels),
            )

        if not active_labels:
            logger.error("No targets with candidates — empty panel")
            return MultiplexPanel(members=[])

        # --- Greedy initialization ---
        current = self._greedy_init(active_labels, candidates_by_target)
        current_score = self._panel_score(current, candidates_by_target, cfg)

        best = dict(current)
        best_score = current_score

        # --- Simulated annealing ---
        temperature = cfg.initial_temperature
        n_accepted = 0
        score_trace: list[float] = [best_score]

        for iteration in range(cfg.max_iterations):
            # Pick a random target and swap to a different candidate
            label = random.choice(active_labels)
            candidates = candidates_by_target[label]
            if len(candidates) <= 1:
                continue

            old_idx = current[label]
            new_idx = random.randint(0, len(candidates) - 1)
            if new_idx == old_idx:
                continue

            current[label] = new_idx
            new_score = self._panel_score(current, candidates_by_target, cfg)

            # Accept or reject
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-10)):
                current_score = new_score
                n_accepted += 1
                if current_score > best_score:
                    best = dict(current)
                    best_score = current_score
            else:
                current[label] = old_idx  # revert

            # Record best score every 100 iterations for convergence trace
            if (iteration + 1) % 100 == 0:
                score_trace.append(best_score)

            temperature *= cfg.cooling_rate

            if temperature < cfg.min_temperature:
                break

        logger.info(
            "SA optimization: %d iterations, %d accepted, "
            "best score=%.4f, final temp=%.6f",
            iteration + 1,
            n_accepted,
            best_score,
            temperature,
        )

        # --- Build panel ---
        target_map = {t.label: t for t in targets}
        members: list[PanelMember] = []

        for label in active_labels:
            idx = best[label]
            sc = candidates_by_target[label][idx]
            target = target_map[label]
            members.append(PanelMember(
                target=target,
                selected_candidate=sc,
            ))

        # Compute cross-reactivity matrix
        cross_matrix = self._cross_reactivity_matrix(
            [candidates_by_target[l][best[l]] for l in active_labels],
            cfg.seed_region_end,
        )

        panel = MultiplexPanel(
            members=members,
            cross_reactivity_matrix=cross_matrix,
            panel_score=best_score,
            optimizer_iterations=iteration + 1,
            optimizer_temperature=temperature,
            optimizer_score_trace=score_trace,
        )

        logger.info(
            "Panel assembled: %d/%d targets, worst cross-reactivity=%.3f",
            len(members),
            len(targets),
            panel.worst_cross_reactivity or 0.0,
        )

        return panel

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _greedy_init(
        self,
        labels: list[str],
        candidates: dict[str, list[ScoredCandidate]],
    ) -> dict[str, int]:
        """Initialize by picking the top-ranked candidate per target."""
        selection: dict[str, int] = {}
        for label in labels:
            cands = candidates[label]
            # Pick highest composite score
            best_idx = 0
            best_val = -1.0
            for i, sc in enumerate(cands):
                val = sc.composite_score
                if val > best_val:
                    best_val = val
                    best_idx = i
            selection[label] = best_idx
        return selection

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _panel_score(
        self,
        selection: dict[str, int],
        candidates: dict[str, list[ScoredCandidate]],
        cfg: OptimizationConfig,
    ) -> float:
        """Compute panel-level score for a candidate selection.

        Higher = better. Combines:
          1. Mean efficiency across selected candidates
          2. Mean discrimination ratio (log-scaled)
          3. Cross-reactivity penalty (inversely scaled)
        """
        selected: list[ScoredCandidate] = []
        for label, idx in selection.items():
            cands = candidates.get(label, [])
            if idx < len(cands):
                selected.append(cands[idx])

        if not selected:
            return 0.0

        # Efficiency component
        efficiencies = [sc.composite_score for sc in selected]
        mean_eff = sum(efficiencies) / len(efficiencies)

        # Discrimination component
        disc_scores = []
        for sc in selected:
            if sc.discrimination is not None:
                # Log-scale: ratio of 2 → 0.30, 5 → 0.70, 10 → 1.0, 20+ → 1.0
                ratio = sc.discrimination.ratio
                disc_scores.append(min(math.log2(max(ratio, 1.0)) / math.log2(20), 1.0))
            else:
                disc_scores.append(0.5)  # no data → neutral
        mean_disc = sum(disc_scores) / len(disc_scores)

        # Cross-reactivity penalty
        cross_matrix = self._cross_reactivity_matrix(
            selected, cfg.seed_region_end
        )
        worst_cross = 0.0
        n = len(cross_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                worst_cross = max(worst_cross, cross_matrix[i][j])

        cross_penalty = max(0.0, 1.0 - worst_cross / cfg.cross_reactivity_threshold)

        # Weighted sum
        score = (
            cfg.efficiency_weight * mean_eff
            + cfg.discrimination_weight * mean_disc
            + cfg.cross_reactivity_weight * cross_penalty
        )

        return score

    def _cross_reactivity_matrix(
        self,
        selected: list[ScoredCandidate],
        seed_end: int,
        amplicons: Optional[dict[str, str]] = None,
    ) -> list[list[float]]:
        """Compute pairwise cross-reactivity between selected candidates.

        Returns an NxN symmetric matrix where C[i,j] is the predicted
        off-target activity of crRNA_i on crRNA_j's amplicon.

        When amplicon sequences are available, uses the position-weighted
        mismatch model from cross_reactivity.py for biophysically accurate
        scoring. Falls back to seed-weighted spacer similarity otherwise.
        """
        n = len(selected)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                spacer_i = selected[i].candidate.spacer_seq
                spacer_j = selected[j].candidate.spacer_seq
                label_i = selected[i].candidate.target_label
                label_j = selected[j].candidate.target_label

                # Try position-weighted model with amplicons
                risk = None
                if amplicons:
                    amp_j = amplicons.get(label_j)
                    amp_i = amplicons.get(label_i)
                    if amp_j and amp_i:
                        try:
                            from compass.scoring.cross_reactivity import _best_off_target_score
                            ij = _best_off_target_score(spacer_i, amp_j)
                            ji = _best_off_target_score(spacer_j, amp_i)
                            risk = max(ij["activity"], ji["activity"])
                        except Exception:
                            pass

                if risk is None:
                    # Fallback: seed-weighted spacer similarity
                    sim = _spacer_similarity(spacer_i, spacer_j, seed_end)
                    risk = max(0.0, (sim - 0.5) / 0.5) if sim > 0.5 else 0.0

                matrix[i][j] = risk
                matrix[j][i] = risk

        return matrix
