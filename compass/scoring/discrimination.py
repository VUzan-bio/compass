"""Heuristic discrimination scoring — B-JEPA Path B stand-in.

Predicts MUT/WT activity ratio using the same position-dependent
mismatch tolerance profiles as the synthetic mismatch module. This
gives every crRNA candidate a quantitative discrimination estimate
without requiring a trained ML model.

The scoring model:
  For a crRNA designed to match the MUTANT sequence:
    - vs MUT: perfect match → activity ≈ 1.0
    - vs WT:  N mismatches → activity = Π(1 - sensitivity[pos] × destab[type])
    - Discrimination ratio = activity_MUT / activity_WT (higher = better)

  For PROXIMITY candidates:
    - No crRNA-level mismatch → ratio ≈ 1.0 (discrimination via AS-RPA)
    - Conservative estimate assigned pending RPA-level validation

Cooperativity model (same as synthetic_mismatch._predict_activity):
  Closely spaced mismatches (within 4 positions) cooperatively
  destabilise the R-loop, making the combined effect super-multiplicative.
  This is the key biophysical mechanism that makes seed-region mutations
  so effective for SNP discrimination.

When the B-JEPA checkpoint becomes available, this scorer is replaced
by a single-line change: swapping HeuristicDiscriminationScorer for
JEPAScorer with mode=DISCRIMINATION. The interface is identical.

References:
  - Kim et al., Nature Methods 2020 — Cas12a mismatch profiling
  - Strohkendl et al., Molecular Cell 2018 — R-loop cooperativity
  - Chen et al., Science 2018 — mismatch-enhanced discrimination
  - Teng et al., Genome Biology 2019 — position-dependent tolerance

"""

from __future__ import annotations

import logging
from typing import Optional

from compass.candidates.synthetic_mismatch import (
    MISMATCH_DESTABILISATION,
    POSITION_SENSITIVITY_PROFILES,
    MismatchType,
    _DNA_TO_RNA_COMPLEMENT,
    _classify_mismatch,
    _predict_activity,
)
from compass.core.types import (
    CrRNACandidate,
    DetectionStrategy,
    DiscriminationScore,
    MismatchPair,
    OffTargetReport,
    ScoredCandidate,
    Target,
)
from compass.scoring.base import Scorer

logger = logging.getLogger(__name__)

# Minimum discrimination ratio for clinical diagnostic use
# Below this, the assay cannot reliably distinguish MUT from WT
DISCRIMINATION_THRESHOLD = 2.0

# Conservative ratio for proximity candidates (discrimination via RPA, not crRNA)
PROXIMITY_DEFAULT_RATIO = 0.9


class HeuristicDiscriminationScorer(Scorer):
    """Predict crRNA discrimination using biophysical mismatch models.

    Drop-in replacement for JEPAScorer in Path B (discrimination) mode.
    Uses the same position-dependent sensitivity profiles from empirical
    Cas12a mismatch tolerance data.

    Usage:
        scorer = HeuristicDiscriminationScorer(cas_variant="enAsCas12a")
        scored = scorer.score_with_pair(candidate, pair, offtarget)

    For pipeline integration:
        scorer = HeuristicDiscriminationScorer()
        for sc, pair in zip(scored_candidates, mismatch_pairs):
            sc = scorer.add_discrimination(sc, pair)
    """

    def __init__(
        self,
        cas_variant: str = "enAsCas12a",
        min_ratio: float = DISCRIMINATION_THRESHOLD,
        heuristic_fallback: Optional[Scorer] = None,
    ) -> None:
        self.cas_variant = cas_variant
        self.min_ratio = min_ratio
        self._fallback = heuristic_fallback

        self.profile = POSITION_SENSITIVITY_PROFILES.get(
            cas_variant,
            POSITION_SENSITIVITY_PROFILES["enAsCas12a"],
        )

    def score(
        self,
        candidate: CrRNACandidate,
        offtarget: OffTargetReport,
    ) -> ScoredCandidate:
        """Score a candidate (without discrimination — use score_with_pair)."""
        if self._fallback:
            return self._fallback.score(candidate, offtarget)
        from compass.scoring.heuristic import HeuristicScorer

        return HeuristicScorer().score(candidate, offtarget)

    def score_with_pair(
        self,
        candidate: CrRNACandidate,
        pair: MismatchPair,
        offtarget: OffTargetReport,
    ) -> ScoredCandidate:
        """Score a candidate WITH discrimination analysis.

        This is the primary entry point for discrimination scoring.
        Returns a ScoredCandidate with the discrimination field populated.
        """
        # Get base scored candidate
        scored = self.score(candidate, offtarget)

        # Add discrimination
        scored.discrimination = self.predict_discrimination(candidate, pair)

        return scored

    def predict_discrimination(
        self,
        candidate: CrRNACandidate,
        pair: MismatchPair,
    ) -> DiscriminationScore:
        """Predict MUT/WT activity ratio from mismatch profile.

        For DIRECT candidates:
          - Identify mismatch positions and types from the MismatchPair
          - Compute activity reduction using position sensitivity × mismatch
            destabilisation
          - Apply cooperativity penalty for nearby mismatches
          - Return ratio = activity_mut / activity_wt

        For PROXIMITY candidates:
          - No crRNA-level discrimination possible
          - Return conservative estimate pending AS-RPA validation
        """
        strategy = candidate.detection_strategy

        # PROXIMITY: no crRNA-level discrimination
        if strategy != DetectionStrategy.DIRECT:
            return DiscriminationScore(
                wt_activity=1.0,
                mut_activity=PROXIMITY_DEFAULT_RATIO,
                model_name="heuristic_proximity",
                is_measured=False,
                detection_strategy=strategy,
            )

        # DIRECT: compute from mismatch profile
        wt_activity = self._compute_activity_vs_target(
            candidate, pair, target="wt"
        )
        mut_activity = self._compute_activity_vs_target(
            candidate, pair, target="mut"
        )

        return DiscriminationScore(
            wt_activity=wt_activity,
            mut_activity=mut_activity,
            model_name="heuristic_discrimination",
            is_measured=False,
            detection_strategy=strategy,
        )

    def add_discrimination(
        self,
        scored: ScoredCandidate,
        pair: MismatchPair,
    ) -> ScoredCandidate:
        """Add discrimination score to an existing ScoredCandidate.

        Modifies the scored candidate in place and returns it.
        """
        scored.discrimination = self.predict_discrimination(
            scored.candidate, pair
        )
        return scored

    def add_discrimination_batch(
        self,
        scored_candidates: list[ScoredCandidate],
        pairs: list[MismatchPair],
    ) -> list[ScoredCandidate]:
        """Add discrimination scores to a batch of candidates.

        Pairs are matched by candidate_id. Candidates without a matching
        pair get no discrimination score (left as None).
        """
        pair_map = {p.candidate_id: p for p in pairs}

        for sc in scored_candidates:
            pair = pair_map.get(sc.candidate.candidate_id)
            if pair is not None:
                self.add_discrimination(sc, pair)

        n_scored = sum(
            1 for sc in scored_candidates if sc.discrimination is not None
        )
        logger.info(
            "Discrimination scoring: %d/%d candidates scored",
            n_scored,
            len(scored_candidates),
        )

        return scored_candidates

    # ------------------------------------------------------------------
    # Activity computation
    # ------------------------------------------------------------------

    def _compute_activity_vs_target(
        self,
        candidate: CrRNACandidate,
        pair: MismatchPair,
        target: str,  # "wt" or "mut"
    ) -> float:
        """Compute predicted Cas12a activity against WT or MUT target.

        The crRNA is designed to match MUT:
          vs MUT: 0 mismatches → activity ≈ 1.0
          vs WT:  N mismatches → reduced activity

        We enumerate all mismatch positions between the crRNA and the
        specified target, then use the multiplicative activity model
        with cooperativity corrections.
        """
        spacer_len = candidate.spacer_length

        if target == "mut":
            # crRNA matches MUT perfectly (by design)
            # Only have mismatches if there's a synthetic mismatch (not modelled here)
            return 1.0

        # target == "wt": compute mismatches between crRNA and WT
        wt_spacer = pair.wt_spacer
        mut_spacer = pair.mut_spacer  # = crRNA spacer (matches MUT)

        if not wt_spacer or not mut_spacer:
            return 0.5  # no data → neutral estimate

        if len(wt_spacer) != len(mut_spacer):
            return 0.5

        # Find mismatch positions
        mismatches: list[tuple[int, float, float]] = []

        # Seed-region destabilisation floor: R-loop collapse in positions
        # 1-8 is catastrophic regardless of mismatch type. Even wobble
        # pairs (rG:dT, rU:dG) severely disrupt R-loop propagation in
        # the seed because the PAM-proximal helix must form first.
        # Kim et al. 2017, Strohkendl et al. 2018.
        seed_end = 8
        seed_destab_floor = 0.85

        for i in range(len(wt_spacer)):
            if wt_spacer[i].upper() == mut_spacer[i].upper():
                continue

            pos = i + 1  # 1-indexed from PAM-proximal end

            # Position sensitivity
            sensitivity = self.profile.get(pos, 0.05)

            # Mismatch type: crRNA RNA base vs WT DNA base
            # crRNA base = RNA complement of MUT spacer base
            crna_rna = _DNA_TO_RNA_COMPLEMENT.get(
                mut_spacer[i].upper(), "N"
            )
            wt_dna = wt_spacer[i].upper()

            mm_type = _classify_mismatch(crna_rna, wt_dna)
            destab = MISMATCH_DESTABILISATION.get(mm_type, 0.5) if mm_type else 0.5

            # Seed floor: ensure minimum destabilisation in seed region
            if pos <= seed_end:
                destab = max(destab, seed_destab_floor)

            mismatches.append((pos, sensitivity, destab))

        if not mismatches:
            # No mismatches between WT and MUT spacer → no discrimination
            return 1.0

        return _predict_activity(mismatches, self.cas_variant)

    # ------------------------------------------------------------------
    # Batch analysis helpers
    # ------------------------------------------------------------------

    def analyze_panel_discrimination(
        self,
        scored_candidates: list[ScoredCandidate],
    ) -> dict[str, dict]:
        """Analyze discrimination across a panel of candidates.

        Returns a summary dict grouped by target label with:
          - best_ratio: highest discrimination ratio
          - mean_ratio: average across candidates
          - n_passing: candidates above threshold
          - strategy: detection strategy of best candidate
        """
        by_target: dict[str, list[ScoredCandidate]] = {}
        for sc in scored_candidates:
            label = sc.candidate.target_label
            by_target.setdefault(label, []).append(sc)

        summary = {}
        for label, candidates in by_target.items():
            ratios = []
            for sc in candidates:
                if sc.discrimination is not None:
                    ratios.append(sc.discrimination.ratio)

            if not ratios:
                summary[label] = {
                    "best_ratio": None,
                    "mean_ratio": None,
                    "n_passing": 0,
                    "n_total": len(candidates),
                    "strategy": "unknown",
                }
                continue

            best_idx = ratios.index(max(ratios))
            best_cand = candidates[best_idx]

            summary[label] = {
                "best_ratio": max(ratios),
                "mean_ratio": sum(ratios) / len(ratios),
                "n_passing": sum(1 for r in ratios if r >= self.min_ratio),
                "n_total": len(candidates),
                "strategy": str(best_cand.candidate.detection_strategy.value),
            }

        return summary


# ======================================================================
# PAM-disruption binary discrimination
# ======================================================================

# Canonical TTTV PAM consensus: positions 0-2 must be T, position 3 must be A/C/G
_PAM_CONSENSUS = [{"T"}, {"T"}, {"T"}, {"A", "C", "G"}]


def check_pam_disruption(
    candidate: CrRNACandidate,
    target: Target,
) -> dict:
    """Check whether the resistance SNP falls within the crRNA's PAM.

    If the SNP disrupts the PAM consensus for the wildtype allele, Cas12a
    physically cannot bind WT DNA at this locus — giving binary (infinite)
    discrimination without relying on mismatch intolerance.

    This is the strongest possible discrimination mechanism: all-or-nothing
    PAM recognition gating. However, it is extremely rare in GC-rich genomes
    like M. tuberculosis (65.6% GC) because TTTV PAMs require three
    consecutive thymines.

    Returns:
        dict with keys:
          - pam_disrupted: bool
          - pam_disruption_type: "wt_pam_broken" | "mut_pam_broken" | None
    """
    result = {"pam_disrupted": False, "pam_disruption_type": None}

    # Only meaningful for DIRECT candidates with known PAM
    if candidate.detection_strategy != DetectionStrategy.DIRECT:
        return result
    if not candidate.pam_seq or len(candidate.pam_seq) != 4:
        return result

    # Determine PAM genomic window.
    # Cas12a PAM is upstream (5') of the spacer on the target strand.
    # Plus strand: PAM is at [genomic_start - 4, genomic_start)
    # Minus strand: PAM is at [genomic_end, genomic_end + 4)
    from compass.core.types import Strand

    if candidate.strand == Strand.PLUS:
        pam_start = candidate.genomic_start - 4
    else:
        pam_start = candidate.genomic_end

    pam_end = pam_start + 4

    # Check if the SNP falls within the PAM window
    snp_pos = target.genomic_pos
    snp_footprint = target.mutation_footprint_bp

    # For codon mutations, check all positions in the footprint
    snp_positions = list(range(snp_pos, snp_pos + snp_footprint))
    overlap = [p for p in snp_positions if pam_start <= p < pam_end]

    if not overlap:
        return result

    # SNP falls in PAM — check if it breaks the consensus
    pam_seq = candidate.pam_seq.upper()
    ref_codon = target.ref_codon.upper()
    alt_codon = target.alt_codon.upper()

    # For each overlapping position, determine WT and MUT bases
    for pos in overlap:
        pam_offset = pos - pam_start  # 0-3 within PAM
        codon_offset = pos - snp_pos   # offset within codon

        if codon_offset >= len(ref_codon) or codon_offset >= len(alt_codon):
            continue

        # Get the WT and MUT bases at this PAM position
        # For minus strand, we need to complement since PAM is read 5'→3'
        wt_base = ref_codon[codon_offset].upper()
        mut_base = alt_codon[codon_offset].upper()

        if candidate.strand == Strand.MINUS:
            _comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
            wt_base = _comp.get(wt_base, wt_base)
            mut_base = _comp.get(mut_base, mut_base)

        # Check PAM consensus at this position
        consensus = _PAM_CONSENSUS[pam_offset] if pam_offset < 4 else set()
        wt_valid = wt_base in consensus
        mut_valid = mut_base in consensus

        if wt_valid != mut_valid:
            # One allele satisfies PAM, the other doesn't → binary discrimination
            result["pam_disrupted"] = True
            if not wt_valid:
                # WT breaks PAM → Cas12a can't bind WT → infinite discrimination
                result["pam_disruption_type"] = "wt_pam_broken"
            else:
                # MUT breaks PAM → Cas12a can't bind MUT → inverted discrimination
                result["pam_disruption_type"] = "mut_pam_broken"
            break

    return result
