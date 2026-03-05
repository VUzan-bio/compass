"""Level 1 — Rule-based heuristic scoring.

Based on feature importance analysis from Kim et al. (2018) and
empirical rules from Cas12a guide design literature.

Each sub-score is normalised to [0, 1] where 1 = optimal.
The composite score is a weighted sum.

Proximity-aware: for PROXIMITY candidates (PAM desert fallback),
the seed_position_score is replaced by a proximity_bonus that
rewards crRNAs closer to the mutation site. This is because
proximity candidates have no mutation inside the spacer — their
discrimination comes from allele-specific RPA primers, not from
crRNA mismatch position.

This is the baseline that works immediately without any training data.
"""

from __future__ import annotations

import math

from compass.core.constants import (
    GC_MAX,
    GC_MIN,
    GC_OPTIMAL_DEFAULT,
    GC_OPTIMAL_MTB,
    HEURISTIC_WEIGHTS,
    HOMOPOLYMER_MAX,
    MFE_THRESHOLD,
    SEED_REGION_END,
)
from compass.core.types import (
    CrRNACandidate,
    DetectionStrategy,
    HeuristicScore,
    OffTargetReport,
    ScoredCandidate,
)
from compass.scoring.base import Scorer


class HeuristicScorer(Scorer):
    """Rule-based crRNA scoring.

    Handles both DIRECT and PROXIMITY candidates:
    - DIRECT: seed_position_score based on mutation position in spacer
    - PROXIMITY: seed_position_score = 0, proximity_bonus based on
      distance to mutation (closer = higher bonus)

    Usage:
        scorer = HeuristicScorer()
        scored = scorer.score_batch(candidates, offtargets)
    """

    # Maximum proximity distance (bp) that gets any bonus.
    # Beyond this, proximity_bonus = 0.
    MAX_PROXIMITY_BONUS_DISTANCE: int = 100

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        organism: str = "default",
    ) -> None:
        self.weights = weights or HEURISTIC_WEIGHTS
        self.gc_optimal = GC_OPTIMAL_MTB if organism == "mtb" else GC_OPTIMAL_DEFAULT

    def score(
        self,
        candidate: CrRNACandidate,
        offtarget: OffTargetReport,
    ) -> ScoredCandidate:
        seed_score = self._score_seed_position(candidate.mutation_position_in_spacer)
        gc_penalty = self._score_gc(candidate.gc_content, self.gc_optimal)
        structure_penalty = self._score_structure(candidate.mfe)
        homo_penalty = self._score_homopolymer(candidate.homopolymer_max)
        ot_penalty = self._score_offtarget(offtarget)

        # Proximity bonus for PROXIMITY candidates
        prox_bonus = 0.0
        is_proximity = getattr(candidate, "detection_strategy", None) == DetectionStrategy.PROXIMITY
        if is_proximity:
            prox_bonus = self._score_proximity_distance(
                getattr(candidate, "proximity_distance", 100)
            )

        composite = (
            self.weights["seed_position"] * seed_score
            + self.weights["gc"] * gc_penalty
            + self.weights["structure"] * structure_penalty
            + self.weights["homopolymer"] * homo_penalty
            + self.weights["offtarget"] * ot_penalty
        )

        # For proximity candidates, replace seed contribution with proximity bonus
        # since seed_score is 0 (no mutation in spacer), this effectively adds
        # the proximity signal into the composite
        if is_proximity:
            composite += self.weights["seed_position"] * prox_bonus

        # Apply PAM activity penalty as multiplicative factor.
        # Canonical TTTV = 1.0 (no change). Non-canonical PAMs reduce the
        # composite proportionally to their relative activity from
        # Kleinstiver et al. 2019. This naturally ranks TTTV candidates
        # higher while still allowing expanded-PAM candidates to compete
        # when they have superior seed position or other features.
        pam_penalty = getattr(candidate, "pam_activity_weight", 1.0)
        composite *= pam_penalty

        # --- Split QC sub-scores ---
        # Activity QC: biophysical quality (will the crRNA cut well?)
        activity_qc = (
            0.30 * gc_penalty
            + 0.30 * structure_penalty
            + 0.15 * homo_penalty
            + 0.25 * ot_penalty
        )

        # Discrimination QC: SNP distinction ability
        mm_type_score = self._score_mismatch_type(candidate)
        flank_gc_score = self._score_flanking_gc(candidate)

        if is_proximity:
            # Proximity: discrimination comes from AS-RPA, not crRNA mismatch
            discrimination_qc = prox_bonus
        else:
            discrimination_qc = (
                0.50 * seed_score
                + 0.30 * mm_type_score
                + 0.20 * flank_gc_score
            )

        heuristic = HeuristicScore(
            seed_position_score=seed_score,
            gc_penalty=gc_penalty,
            structure_penalty=structure_penalty,
            homopolymer_penalty=homo_penalty,
            offtarget_penalty=ot_penalty,
            composite=composite,
            proximity_bonus=prox_bonus,
            pam_penalty=pam_penalty,
            activity_qc=round(activity_qc, 4),
            discrimination_qc=round(discrimination_qc, 4),
            mismatch_type_score=round(mm_type_score, 4),
            flanking_gc_score=round(flank_gc_score, 4),
        )

        return ScoredCandidate(
            candidate=candidate,
            offtarget=offtarget,
            heuristic=heuristic,
        )

    # ------------------------------------------------------------------
    # Sub-scores, each normalised to [0, 1]
    # ------------------------------------------------------------------

    @staticmethod
    def _score_seed_position(pos: int | None) -> float:
        """Closer to PAM = better discrimination. Linear decay from pos 1-8.

        Returns 0.0 for proximity candidates (pos=None) since the mutation
        is outside the spacer. Their score contribution comes from
        proximity_bonus instead.
        """
        if pos is None:
            return 0.0
        if pos > SEED_REGION_END:
            return 0.0
        return 1.0 - (pos - 1) / SEED_REGION_END

    @staticmethod
    def _score_gc(gc: float, optimal: float = 0.50) -> float:
        """Score GC content relative to optimum. Penalise deviation.

        For TB (optimal=0.55), the penalty is asymmetric: the scoring
        window extends further toward high GC (matching the genome's
        65.6% GC) than toward low GC.
        """
        max_deviation = max(abs(GC_MAX - optimal), abs(GC_MIN - optimal))
        deviation = abs(gc - optimal)
        return max(0.0, 1.0 - deviation / max_deviation)

    @staticmethod
    def _score_structure(mfe: float | None) -> float:
        """Less negative MFE = less secondary structure = better.

        MFE of 0 (no structure) → score 1.0
        MFE at threshold → score 0.0
        """
        if mfe is None:
            return 0.5  # no data → neutral score
        if mfe >= 0:
            return 1.0
        return max(0.0, 1.0 - mfe / MFE_THRESHOLD)

    @staticmethod
    def _score_homopolymer(max_run: int) -> float:
        """No homopolymers → 1.0. At max → 0.0."""
        if max_run <= 1:
            return 1.0
        return max(0.0, 1.0 - (max_run - 1) / HOMOPOLYMER_MAX)

    @staticmethod
    def _score_offtarget(report: OffTargetReport) -> float:
        """Clean (no risky hits) → 1.0. Exponential decay with hit count."""
        n = report.total_risky_hits
        if n == 0:
            return 1.0
        return math.exp(-0.5 * n)

    @staticmethod
    def _score_mismatch_type(candidate: CrRNACandidate) -> float:
        """Score the mismatch type at the mutation position.

        Transversions (purine↔pyrimidine) destabilise R-loops more than
        transitions (purine↔purine / pyrimidine↔pyrimidine), giving
        stronger SNP discrimination.

        Returns:
            1.0 for transversions (C↔A, G↔T, etc.)
            0.5 for transitions (A↔G, C↔T)
            0.3 when mismatch type is unknown
            0.0 for proximity candidates (no spacer mismatch)
        """
        pos = candidate.mutation_position_in_spacer
        if pos is None:
            return 0.0  # proximity — no mismatch in spacer

        ref_base = getattr(candidate, "ref_base_at_mutation", None)
        if not ref_base:
            return 0.3  # unknown

        # The spacer_seq is the MUT-matching spacer; ref_base is the WT base
        # at that position (in spacer orientation)
        purines = {"A", "G"}
        pyrimidines = {"C", "T"}
        ref = ref_base.upper()
        # MUT base at the mutation position in the spacer
        idx = pos - 1  # 0-indexed
        if idx < 0 or idx >= len(candidate.spacer_seq):
            return 0.3
        mut = candidate.spacer_seq[idx].upper()

        if ref == mut:
            return 0.0  # no mismatch (shouldn't happen for DIRECT)

        # Transversion: purine↔pyrimidine
        ref_is_purine = ref in purines
        mut_is_purine = mut in purines
        if ref_is_purine != mut_is_purine:
            return 1.0  # transversion — strong destabilisation
        return 0.5  # transition — weaker destabilisation

    @staticmethod
    def _score_flanking_gc(candidate: CrRNACandidate) -> float:
        """Score local GC content around the mismatch position.

        Lower local GC around the mismatch makes the R-loop more sensitive
        to mismatches (less stable duplex) → better discrimination.

        Uses a ±3 nt window around the mutation position.
        Returns 1.0 - local_gc (inverted: AT-rich flanks = better).
        Returns 0.5 (neutral) when position data unavailable.
        """
        pos = candidate.mutation_position_in_spacer
        if pos is None:
            return 0.5  # proximity — neutral

        spacer = candidate.spacer_seq.upper()
        idx = pos - 1  # 0-indexed
        window = 3
        start = max(0, idx - window)
        end = min(len(spacer), idx + window + 1)
        region = spacer[start:end]

        if not region:
            return 0.5

        gc_count = sum(1 for nt in region if nt in ("G", "C"))
        local_gc = gc_count / len(region)
        return round(1.0 - local_gc, 4)

    @classmethod
    def _score_proximity_distance(cls, distance: int) -> float:
        """Score for proximity candidates based on distance to mutation.

        Closer to mutation = higher score (better for AS-RPA design).
        Linear decay from 1.0 at distance=0 to 0.0 at MAX_PROXIMITY_BONUS_DISTANCE.

        Distance 0 means spacer edge is adjacent to mutation → score 1.0
        Distance 13 bp (typical nearest in PAM desert) → score ~0.87
        Distance 50 bp → score 0.5
        Distance 100+ bp → score 0.0
        """
        if distance <= 0:
            return 1.0
        if distance >= cls.MAX_PROXIMITY_BONUS_DISTANCE:
            return 0.0
        return 1.0 - (distance / cls.MAX_PROXIMITY_BONUS_DISTANCE)
