"""WT/MUT mismatch pair generation for crRNA discrimination analysis.

For each crRNA candidate targeting the mutant allele, generates the
corresponding wild-type spacer. The mismatch position and type determine
predicted discrimination ratio — the core metric for diagnostic design.

Handles all mutation types:
  - AA substitution: single codon change → 1 nt mismatch in spacer
  - rRNA/promoter: single nt change → 1 nt mismatch
  - Insertion: WT lacks inserted bases → bulge mismatch
  - Deletion: MUT lacks deleted bases → gap mismatch
  - MNV: multiple nt changes → multiple mismatches
  - Proximity: both spacers identical (no crRNA-level mismatch)

The MismatchPair is consumed by:
  - HeuristicDiscriminationScorer → estimates MUT/WT activity ratio
  - SyntheticMismatchEnhancer → generates SM variants at optimal positions
  - JEPAScorer.score_discrimination() → ML-predicted discrimination
  - Active learning loop → experimental validation targets

References:
  - Kim et al., Nat Methods 2020 — Cas12a mismatch tolerance
  - Chen et al., Science 2018 — mismatch-based SNP discrimination
"""

from __future__ import annotations

import logging
from typing import Optional

from Bio.Seq import Seq

from compass.core.types import (
    CrRNACandidate,
    DetectionStrategy,
    MismatchPair,
    MutationCategory,
    Strand,
    Target,
)

logger = logging.getLogger(__name__)

# Transition map for approximating WT base from MUT base
_TRANSITION = {"A": "G", "G": "A", "T": "C", "C": "T"}


class MismatchGenerator:
    """Generate WT/MUT spacer pairs for discrimination analysis.

    For DIRECT candidates:
      - The spacer matches the MUT sequence (designed to detect mutant)
      - The WT spacer is derived by reverting the mutation in the spacer
      - Mismatch position(s) are recorded for scoring

    For PROXIMITY candidates:
      - Both spacers are identical (mutation is outside the spacer)
      - Detection relies on AS-RPA primers, not crRNA mismatch
      - MismatchPair is still generated for pipeline uniformity

    Usage:
        gen = MismatchGenerator()
        pair = gen.generate(candidate, target)
        pairs = gen.generate_batch(candidates, targets_map)
    """

    def generate(
        self,
        candidate: CrRNACandidate,
        target: Target,
    ) -> MismatchPair:
        """Generate a WT/MUT mismatch pair for one candidate.

        Args:
            candidate: The crRNA candidate (spacer matches MUT).
            target: The resolved genomic target with ref/alt codons.

        Returns:
            MismatchPair with WT/MUT spacers and mismatch metadata.
        """
        # PROXIMITY: no crRNA-level mismatch
        if candidate.is_proximity:
            return MismatchPair(
                candidate_id=candidate.candidate_id,
                wt_spacer=candidate.spacer_seq,
                mut_spacer=candidate.spacer_seq,
                mismatch_positions=[],
                mismatch_type="proximity",
                mutation_category=_infer_category(target),
                detection_strategy=candidate.detection_strategy,
            )

        mut_spacer = candidate.spacer_seq
        mm_pos = candidate.mutation_position_in_spacer
        ref_base = candidate.ref_base_at_mutation

        logger.debug(
            "M5.5 %s: spacer=%s mut_pos=%s ref_base=%s ref=%s alt=%s strand=%s",
            candidate.target_label,
            candidate.spacer_seq[:12] + "...",
            mm_pos,
            ref_base,
            target.ref_codon,
            target.alt_codon,
            candidate.strand,
        )

        if mm_pos is None or mm_pos < 1:
            # No known position — approximate with transition
            logger.warning(
                "M5.5 %s: mut_pos is %s — falling back to approximate (disc will be ~1×)",
                candidate.target_label,
                mm_pos,
            )
            return self._generate_approximate(candidate, target)

        # DIRECT: derive WT spacer by reverting the mutation
        # If ref_base_at_mutation is available (set by scanner from original
        # flanking), use it directly — no codon math needed.
        if ref_base and len(ref_base) == 1:
            idx = mm_pos - 1  # 0-indexed
            wt_list = list(mut_spacer)
            if 0 <= idx < len(wt_list) and wt_list[idx] != ref_base:
                mut_base = wt_list[idx]
                wt_list[idx] = ref_base
                wt_spacer = "".join(wt_list)
                positions = [mm_pos]
                mm_type = f"{mut_base}>{ref_base}"
            else:
                # ref_base == mut_base means something is wrong, fall through
                logger.warning(
                    "M5.5 %s: ref_base=%s == spacer[%d]=%s, falling through to codon derivation",
                    candidate.target_label, ref_base, idx,
                    wt_list[idx] if 0 <= idx < len(wt_list) else "OOB",
                )
                wt_spacer, positions, mm_type = self._derive_wt_spacer(
                    mut_spacer, mm_pos, target, candidate.strand,
                )
        else:
            wt_spacer, positions, mm_type = self._derive_wt_spacer(
                mut_spacer, mm_pos, target, candidate.strand,
            )

        if wt_spacer == mut_spacer:
            logger.warning(
                "M5.5 %s: WT == MUT after derivation! mm_type=%s positions=%s",
                candidate.target_label, mm_type, positions,
            )
        else:
            logger.debug(
                "M5.5 %s: OK mut=%s wt=%s mm_type=%s positions=%s",
                candidate.target_label, mut_spacer[:12], wt_spacer[:12], mm_type, positions,
            )

        return MismatchPair(
            candidate_id=candidate.candidate_id,
            wt_spacer=wt_spacer,
            mut_spacer=mut_spacer,
            mismatch_positions=positions,
            mismatch_type=mm_type,
            mutation_category=_infer_category(target),
            detection_strategy=candidate.detection_strategy,
        )

    def generate_batch(
        self,
        candidates: list[CrRNACandidate],
        targets_map: dict[str, Target],
    ) -> list[MismatchPair]:
        """Generate pairs for a batch of candidates.

        Args:
            candidates: List of crRNA candidates.
            targets_map: {target_label: Target} lookup.

        Returns:
            List of MismatchPairs (one per candidate).
        """
        pairs = []
        for cand in candidates:
            target = targets_map.get(cand.target_label)
            if target is None:
                logger.warning(
                    "No target for %s — skipping mismatch generation",
                    cand.candidate_id,
                )
                continue
            try:
                pair = self.generate(cand, target)
                pairs.append(pair)
            except Exception as e:
                logger.debug(
                    "Mismatch generation failed for %s: %s",
                    cand.candidate_id, e,
                )
        return pairs

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _derive_wt_spacer(
        self,
        mut_spacer: str,
        mm_pos: int,
        target: Target,
        strand: Strand,
    ) -> tuple[str, list[int], str]:
        """Derive WT spacer from MUT spacer using target ref/alt codons.

        For codon mutations (len 3): identify which codon position
        changed, then revert in the spacer.

        For single-nt mutations (rRNA/promoter): direct single-base revert.

        Returns:
            (wt_spacer, mismatch_positions, mismatch_type_str)
        """
        ref_codon = target.ref_codon.upper()
        alt_codon = target.alt_codon.upper()

        if ref_codon == "---" or alt_codon == "---":
            # Large deletion — presence/absence, use transition approx
            return self._approx_transition(mut_spacer, mm_pos)

        # Single-nt mutations (rRNA, promoter)
        if len(ref_codon) == 1 and len(alt_codon) == 1:
            wt_base = ref_codon
            mut_base = alt_codon
            if strand == Strand.MINUS:
                wt_base = str(Seq(wt_base).complement())
                mut_base = str(Seq(mut_base).complement())

            wt_list = list(mut_spacer)
            idx = mm_pos - 1
            if 0 <= idx < len(wt_list):
                wt_list[idx] = wt_base
            wt_spacer = "".join(wt_list)
            return wt_spacer, [mm_pos], f"{mut_base}>{wt_base}"

        # Codon mutations (3 nt)
        if len(ref_codon) == 3 and len(alt_codon) == 3:
            # Find which position(s) differ
            diffs = []
            for i in range(3):
                if ref_codon[i] != alt_codon[i]:
                    diffs.append(i)

            if not diffs:
                # Synonymous — shouldn't happen for resistance mutations
                return self._approx_transition(mut_spacer, mm_pos)

            # The mutation maps to codon positions starting at mm_pos
            # in the spacer. For plus strand, mm_pos is the first codon nt.
            # For minus strand, mm_pos maps to the reverse complement.
            wt_list = list(mut_spacer)
            positions = []

            for d in diffs:
                if strand == Strand.PLUS:
                    spacer_idx = (mm_pos - 1) + d
                else:
                    # Minus strand: codon is reverse-complemented
                    spacer_idx = (mm_pos - 1) + (2 - d)

                if 0 <= spacer_idx < len(wt_list):
                    # Revert to ref base (accounting for strand)
                    if strand == Strand.PLUS:
                        wt_list[spacer_idx] = ref_codon[d]
                    else:
                        wt_list[spacer_idx] = str(
                            Seq(ref_codon[d]).complement()
                        )
                    positions.append(spacer_idx + 1)  # 1-indexed

            wt_spacer = "".join(wt_list)
            mm_type = f"{alt_codon}>{ref_codon}"
            if len(diffs) == 1:
                d = diffs[0]
                mm_type = f"{alt_codon[d]}>{ref_codon[d]}"

            return wt_spacer, sorted(positions), mm_type

        # Fallback for unusual codon lengths
        return self._approx_transition(mut_spacer, mm_pos)

    def _generate_approximate(
        self,
        candidate: CrRNACandidate,
        target: Target,
    ) -> MismatchPair:
        """Approximate WT spacer when exact derivation isn't possible."""
        wt_spacer = candidate.spacer_seq  # identical = no discrimination
        return MismatchPair(
            candidate_id=candidate.candidate_id,
            wt_spacer=wt_spacer,
            mut_spacer=candidate.spacer_seq,
            mismatch_positions=[],
            mismatch_type="unknown",
            mutation_category=_infer_category(target),
            detection_strategy=candidate.detection_strategy,
        )

    @staticmethod
    def _approx_transition(
        spacer: str, mm_pos: int,
    ) -> tuple[str, list[int], str]:
        """Approximate WT by applying a transition at the mismatch position."""
        wt_list = list(spacer)
        idx = mm_pos - 1
        if 0 <= idx < len(wt_list):
            orig = wt_list[idx].upper()
            wt_list[idx] = _TRANSITION.get(orig, orig)
        wt_spacer = "".join(wt_list)
        return wt_spacer, [mm_pos], f"approx_transition"


def _infer_category(target: Target) -> MutationCategory:
    """Infer mutation category from target data."""
    mut = target.mutation
    if mut.category:
        return mut.category
    if mut.is_rrna:
        return MutationCategory.RRNA
    if mut.is_promoter:
        return MutationCategory.PROMOTER
    if target.ref_codon == "---" or target.alt_codon == "---":
        return MutationCategory.LARGE_DELETION
    return MutationCategory.AA_SUBSTITUTION
