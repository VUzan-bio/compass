"""crRNA–primer co-selection validation.

Validates that a crRNA candidate and its assigned RPA primer pair are
physically and functionally compatible. This is the critical integration
point between the crRNA design (Modules 1-6) and primer design (Module 8).

Validation checks:
  1. crRNA target site within RPA amplicon boundaries (HARD)
  2. crRNA on non-displaced strand (SOFT — both strands available after
     first RPA cycle, but non-displaced strand gives faster kinetics)
  3. For PROXIMITY: at least one allele-specific primer present (HARD)
  4. Primer-dimer ΔG above threshold (HARD if below -6.0 kcal/mol)
  5. Amplicon length within cfDNA-compatible range (SOFT, 80-120 bp preferred)
  6. crRNA not overlapping primer binding sites (HARD)

RPA mechanism context:
  RPA uses recombinase-coated primers to invade dsDNA at 37°C without
  thermal denaturation. The displaced strand forms a D-loop, while the
  non-displaced strand remains base-paired. After the first amplification
  cycle, both strands are available as single-stranded intermediates.

  For Cas12a trans-cleavage detection, the crRNA must bind the amplicon
  product. Binding the non-displaced strand gives slightly faster kinetics
  in the first cycle (no strand displacement needed), but both work.

References:
  - Piepenburg et al., PLoS Biol 2006 (RPA mechanism)
  - Li et al., Cell Rep Med 2022 (CRISPR-RPA integration rules)
  - Kellner et al., Nature Protocols 2019 (SHERLOCK protocol)

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from compass.core.types import (
    CrRNACandidate,
    DetectionStrategy,
    RPAPrimerPair,
    ScoredCandidate,
    Strand,
)

logger = logging.getLogger(__name__)


class RejectionReason(str, Enum):
    """Why a crRNA-primer pair was rejected."""

    CRRNA_OUTSIDE_AMPLICON = "crRNA target site outside amplicon boundaries"
    NO_AS_PRIMER_FOR_PROXIMITY = "Proximity target requires allele-specific primer"
    PRIMER_DIMER_SEVERE = "Primer-dimer ΔG below threshold"
    CRRNA_OVERLAPS_PRIMER = "crRNA binding site overlaps primer binding region"
    AMPLICON_TOO_SHORT = "Amplicon too short for RPA"
    AMPLICON_TOO_LONG = "Amplicon too long for efficient RPA"


@dataclass
class CoselectionResult:
    """Result of crRNA-primer compatibility validation."""

    candidate_id: str
    compatible: bool = False
    rejection_reasons: list[RejectionReason] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Metrics
    amplicon_length: int = 0
    crrna_position_in_amplicon: Optional[int] = None  # bp from amplicon start
    strand_optimal: bool = True  # crRNA on non-displaced strand
    has_allele_specific: bool = False
    dimer_dg: Optional[float] = None

    # Composite compatibility score (0-1, for ranking alternative pairs)
    score: float = 0.0

    @property
    def summary(self) -> str:
        status = "PASS" if self.compatible else "FAIL"
        reasons = "; ".join(r.value for r in self.rejection_reasons)
        return f"{self.candidate_id}: {status} (amp={self.amplicon_length}bp) {reasons}"


class CoselectionValidator:
    """Validate crRNA–primer pair compatibility.

    Usage:
        validator = CoselectionValidator()
        result = validator.validate(scored_candidate, primer_pair)

        # Batch validation with multiple primer options:
        best_pair, result = validator.select_best_pair(
            scored_candidate, primer_pairs
        )
    """

    def __init__(
        self,
        amplicon_min: int = 80,
        amplicon_max: int = 120,  # Blood cfDNA hard cap
        dimer_dg_threshold: float = -6.0,
        primer_crrna_min_gap: int = 5,  # min bp between primer end and crRNA start
    ) -> None:
        self.amplicon_min = amplicon_min
        self.amplicon_max = amplicon_max
        self.dimer_dg_threshold = dimer_dg_threshold
        self.primer_crrna_min_gap = primer_crrna_min_gap

    def validate(
        self,
        candidate: CrRNACandidate,
        primer_pair: RPAPrimerPair,
    ) -> CoselectionResult:
        """Validate one crRNA-primer combination.

        Runs all checks and returns a CoselectionResult with pass/fail
        status, rejection reasons, and a compatibility score.
        """
        result = CoselectionResult(candidate_id=candidate.candidate_id)
        rejections: list[RejectionReason] = []
        warnings: list[str] = []

        # Amplicon boundaries
        amp_start = primer_pair.fwd.amplicon_start
        amp_end = primer_pair.rev.amplicon_end
        amp_len = amp_end - amp_start

        result.amplicon_length = amp_len

        # Check 1: crRNA within amplicon (HARD)
        crrna_start = candidate.genomic_start
        crrna_end = candidate.genomic_end

        if crrna_start < amp_start or crrna_end > amp_end:
            rejections.append(RejectionReason.CRRNA_OUTSIDE_AMPLICON)
        else:
            result.crrna_position_in_amplicon = crrna_start - amp_start

        # Check 2: Amplicon length (SOFT — warning only)
        if amp_len < self.amplicon_min:
            rejections.append(RejectionReason.AMPLICON_TOO_SHORT)
        elif amp_len > self.amplicon_max:
            # Soft constraint — RPA works up to ~300bp but less efficiently
            if amp_len > self.amplicon_max * 1.5:
                rejections.append(RejectionReason.AMPLICON_TOO_LONG)
            else:
                warnings.append(
                    f"Amplicon {amp_len}bp exceeds optimal RPA range "
                    f"({self.amplicon_min}-{self.amplicon_max}bp)"
                )

        # Check 3: For PROXIMITY, require allele-specific primer (HARD)
        if candidate.detection_strategy != DetectionStrategy.DIRECT:
            if not primer_pair.has_allele_specific_primer:
                rejections.append(RejectionReason.NO_AS_PRIMER_FOR_PROXIMITY)
            else:
                result.has_allele_specific = True

        # Check 4: Primer-dimer (HARD if severe)
        if primer_pair.dimer_dg is not None:
            result.dimer_dg = primer_pair.dimer_dg
            if primer_pair.dimer_dg < self.dimer_dg_threshold:
                rejections.append(RejectionReason.PRIMER_DIMER_SEVERE)

        # Check 5: crRNA doesn't overlap primer binding sites (HARD)
        fwd_end = primer_pair.fwd.amplicon_start + len(primer_pair.fwd.seq)
        rev_start = primer_pair.rev.amplicon_end - len(primer_pair.rev.seq)

        if crrna_start < fwd_end + self.primer_crrna_min_gap:
            if crrna_start >= amp_start:  # only if crRNA is in amplicon
                rejections.append(RejectionReason.CRRNA_OVERLAPS_PRIMER)
        if crrna_end > rev_start - self.primer_crrna_min_gap:
            if crrna_end <= amp_end:
                rejections.append(RejectionReason.CRRNA_OVERLAPS_PRIMER)

        # Check 6: Strand optimality (SOFT)
        # Non-displaced strand gives faster kinetics in cycle 1
        # For fwd primer: non-displaced = plus strand
        # For rev primer: non-displaced = minus strand
        # After cycle 1 both strands are available
        result.strand_optimal = True  # both strands work after first cycle
        if candidate.strand == Strand.MINUS:
            # Slightly less optimal but still functional
            warnings.append("crRNA on displaced strand (slower first-cycle kinetics)")
            result.strand_optimal = False

        # Compute compatibility score
        result.compatible = len(rejections) == 0
        result.rejection_reasons = rejections
        result.warnings = warnings
        result.score = self._compute_score(result, candidate, primer_pair)

        return result

    def validate_scored(
        self,
        scored: ScoredCandidate,
        primer_pair: RPAPrimerPair,
    ) -> CoselectionResult:
        """Convenience: validate from a ScoredCandidate."""
        return self.validate(scored.candidate, primer_pair)

    def select_best_pair(
        self,
        candidate: CrRNACandidate,
        primer_pairs: list[RPAPrimerPair],
    ) -> tuple[Optional[RPAPrimerPair], CoselectionResult]:
        """Select the best compatible primer pair for a candidate.

        Tests all primer pairs and returns the one with the highest
        compatibility score that passes all hard checks.

        Returns:
            (best_pair, best_result) — both None if no compatible pair found.
        """
        if not primer_pairs:
            return None, CoselectionResult(
                candidate_id=candidate.candidate_id,
                compatible=False,
                rejection_reasons=[RejectionReason.CRRNA_OUTSIDE_AMPLICON],
            )

        best_pair: Optional[RPAPrimerPair] = None
        best_result = CoselectionResult(
            candidate_id=candidate.candidate_id,
            compatible=False,
        )

        for pair in primer_pairs:
            result = self.validate(candidate, pair)
            if result.compatible and result.score > best_result.score:
                best_pair = pair
                best_result = result

        if best_pair is None:
            # Return the result from the first pair (for rejection reasons)
            best_result = self.validate(candidate, primer_pairs[0])

        return best_pair, best_result

    def validate_batch(
        self,
        candidates: list[CrRNACandidate],
        primer_pairs_map: dict[str, list[RPAPrimerPair]],
    ) -> dict[str, tuple[Optional[RPAPrimerPair], CoselectionResult]]:
        """Validate and select best primers for a batch of candidates.

        Args:
            candidates: List of crRNA candidates.
            primer_pairs_map: {candidate_id: [RPAPrimerPair, ...]}

        Returns:
            {candidate_id: (best_pair, result)}
        """
        results = {}
        for cand in candidates:
            pairs = primer_pairs_map.get(cand.candidate_id, [])
            pair, result = self.select_best_pair(cand, pairs)
            results[cand.candidate_id] = (pair, result)

            if result.compatible:
                logger.debug(
                    "Co-selection PASS: %s (amp=%dbp, score=%.2f)",
                    cand.candidate_id,
                    result.amplicon_length,
                    result.score,
                )
            else:
                logger.debug(
                    "Co-selection FAIL: %s — %s",
                    cand.candidate_id,
                    "; ".join(r.value for r in result.rejection_reasons),
                )

        n_pass = sum(1 for _, (_, r) in results.items() if r.compatible)
        logger.info(
            "Co-selection: %d/%d candidates have compatible primer pairs",
            n_pass,
            len(candidates),
        )

        return results

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        result: CoselectionResult,
        candidate: CrRNACandidate,
        pair: RPAPrimerPair,
    ) -> float:
        """Compute a 0-1 compatibility score for ranking alternative pairs.

        Factors:
          - Amplicon length (shorter within range = better, less non-target DNA)
          - Tm balance between primers (closer = better)
          - crRNA position in amplicon (central = slightly better)
          - Strand optimality bonus
          - AS-RPA bonus for proximity candidates
        """
        if not result.compatible:
            return 0.0

        score = 0.0

        # Amplicon length: prefer shorter (less background)
        # Normalized: 100bp → 1.0, 200bp → 0.5, 300bp → 0.0
        amp_norm = max(0.0, 1.0 - (result.amplicon_length - self.amplicon_min) / (
            self.amplicon_max * 1.5 - self.amplicon_min
        ))
        score += 0.25 * amp_norm

        # Tm balance between primers
        tm_diff = abs(pair.fwd.tm - pair.rev.tm)
        tm_score = max(0.0, 1.0 - tm_diff / 5.0)
        score += 0.25 * tm_score

        # crRNA position (prefer central)
        if result.crrna_position_in_amplicon is not None and result.amplicon_length > 0:
            rel_pos = result.crrna_position_in_amplicon / result.amplicon_length
            centrality = 1.0 - abs(rel_pos - 0.5) * 2
            score += 0.20 * centrality

        # Strand bonus
        if result.strand_optimal:
            score += 0.10

        # AS-RPA bonus for proximity
        if result.has_allele_specific:
            score += 0.20

        return min(score, 1.0)
