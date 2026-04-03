"""Allele-specific RPA (AS-RPA) primer design for proximity detection.

For PAM-desert targets where no crRNA can directly overlap the mutation,
discrimination comes from allele-specific amplification:

  1. One primer has its 3'-terminal base matching the MUTANT allele
  2. A deliberate mismatch at position -2 or -3 from 3' end enhances
     selectivity by further destabilising WT extension
  3. The complementary primer is standard (flanking)
  4. The crRNA cuts the amplicon for signal — no allele discrimination

AS-RPA design rules (Ye et al., Biosens Bioelectron 2019):
  - Primer 30-35 nt, standard RPA length constraints
  - 3' terminal base = mutant allele (locked)
  - Position -2: deliberate mismatch (C→A or similar strong disruption)
  - Tm 60-65°C including mismatches (use nearest-neighbour correction)
  - Amplicon 80-120 bp encompassing the crRNA target site
    (blood cfDNA: fragments ~100–160 bp, hard cap 120 bp)

Bell et al. (Sci Adv 2025) — Asymmetric primer ratios:
  - AS primer at lower concentration (200 nM vs 480 nM standard)
  - Prevents Cas12a from degrading the template before amplification
  - 5 copies/μL LOD with 93% clinical sensitivity

References:
  - Ye et al., Biosens Bioelectron 2019 (AS-RPA method)
  - Bell et al., Sci Adv 2025 (one-pot asymmetric RPA-CRISPR)
  - Li et al., Cell Rep Med 2022 (RPA-CRISPR design rules)
  - Piepenburg et al., PLoS Biol 2006 (RPA mechanism)
"""

from __future__ import annotations

import logging
from typing import Optional

from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt

from compass.core.constants import (
    RPA_AMPLICON_MAX_GENOMIC,
    RPA_AMPLICON_MIN,
    RPA_AMPLICON_SOFT_PENALTY_START,
    RPA_PRIMER_LENGTH_MAX,
    RPA_PRIMER_LENGTH_MIN,
    RPA_TM_MAX,
    RPA_TM_MIN,
)
from compass.core.types import (
    CrRNACandidate,
    DetectionStrategy,
    RPAPrimer,
    RPAPrimerPair,
    Strand,
    Target,
)

logger = logging.getLogger(__name__)

# Mismatches that maximally disrupt WT extension at position -2
# Ordered by disruption strength (purine:purine > purine:pyrimidine > wobble)
_STRONG_MISMATCHES = {
    "A": ["C", "T"],  # A→C (strong), A→T (moderate)
    "T": ["G", "A"],  # T→G (strong), T→A (moderate)
    "G": ["T", "A"],  # G→T (strong), G→A (moderate)
    "C": ["A", "G"],  # C→A (strong), C→G (moderate)
}


class ASRPADesigner:
    """Design allele-specific RPA primers for proximity-mode detection.

    Usage:
        designer = ASRPADesigner()
        pairs = designer.design(
            candidate=scored_candidate.candidate,
            target=target,
            genome_seq=genome_string,
        )

    Each returned RPAPrimerPair has:
      - One allele-specific primer (is_allele_specific=True)
      - One standard flanking primer
      - detection_strategy = DetectionStrategy.PROXIMITY
    """

    def __init__(
        self,
        primer_len_min: int = RPA_PRIMER_LENGTH_MIN,
        primer_len_max: int = RPA_PRIMER_LENGTH_MAX,
        tm_min: float = RPA_TM_MIN,
        tm_max: float = RPA_TM_MAX,
        amplicon_min: int = RPA_AMPLICON_MIN,
        amplicon_max: int = RPA_AMPLICON_MAX_GENOMIC,
        tm_opt: float = 64.5,
        deliberate_mm_positions: tuple[int, ...] = (-2, -3),
    ) -> None:
        self.primer_len_min = primer_len_min
        self.primer_len_max = primer_len_max
        self.tm_min = tm_min
        self.tm_max = tm_max
        self.tm_opt = tm_opt
        self.amplicon_min = amplicon_min
        self.amplicon_max = amplicon_max
        self.deliberate_mm_positions = deliberate_mm_positions

    def design(
        self,
        candidate: CrRNACandidate,
        target: Target,
        genome_seq: str,
    ) -> list[RPAPrimerPair]:
        """Design AS-RPA primer pairs for a proximity candidate.

        The allele-specific primer has its 3' end at the mutation site.
        The standard primer flanks on the opposite side, ensuring the
        crRNA target site is within the amplicon.

        Args:
            candidate: PROXIMITY crRNA candidate.
            target: Resolved target with genomic coordinates.
            genome_seq: Full genome sequence string.

        Returns:
            Ranked list of AS-RPA primer pairs.
        """
        if candidate.detection_strategy == DetectionStrategy.DIRECT:
            logger.debug(
                "Skipping AS-RPA for direct candidate %s",
                candidate.candidate_id,
            )
            return []

        mutation_pos = target.genomic_pos
        ref_codon = target.ref_codon.upper()
        alt_codon = target.alt_codon.upper()

        # Determine mutation base and position for AS primer design.
        # ref_codon/alt_codon are in CODING orientation, but the genome
        # (and primers) are in plus-strand orientation. For minus-strand
        # genes, we must convert to genomic coordinates and complement bases.
        if len(ref_codon) == 1 and len(alt_codon) == 1:
            snp_genomic = mutation_pos
            ref_base = ref_codon
            alt_base = alt_codon
        elif len(ref_codon) == 3 and len(alt_codon) == 3:
            # Find which coding position differs
            snp_offset = None
            for i in range(3):
                if ref_codon[i] != alt_codon[i]:
                    snp_offset = i
                    break
            if snp_offset is None:
                logger.warning("No SNP found in codon for %s", target.label)
                return []

            # Detect minus-strand gene by comparing genome bases to ref codon.
            # genomic_pos points to the lowest coordinate of the 3-nt footprint.
            # Plus-strand gene: genome[pos:pos+3] == ref_codon
            # Minus-strand gene: genome[pos:pos+3] == RC(ref_codon)
            genome_codon = genome_seq[mutation_pos:mutation_pos + 3].upper()
            rc_ref = str(Seq(ref_codon).reverse_complement())

            if genome_codon == ref_codon:
                # Plus-strand gene: coding matches genome directly
                snp_genomic = mutation_pos + snp_offset
                ref_base = ref_codon[snp_offset]
                alt_base = alt_codon[snp_offset]
            elif genome_codon == rc_ref:
                # Minus-strand gene: reverse offset, complement bases
                snp_genomic = mutation_pos + (2 - snp_offset)
                ref_base = str(Seq(ref_codon[snp_offset]).complement())
                alt_base = str(Seq(alt_codon[snp_offset]).complement())
            else:
                # Fallback: use genome directly to determine ref base
                snp_genomic = mutation_pos + snp_offset
                ref_base = genome_seq[snp_genomic].upper() if snp_genomic < len(genome_seq) else ref_codon[snp_offset]
                alt_base = alt_codon[snp_offset]
                logger.debug(
                    "%s: genome codon %s doesn't match ref %s or RC %s, using genome ref_base=%s",
                    target.label, genome_codon, ref_codon, rc_ref, ref_base,
                )
        else:
            logger.warning("Unsupported codon format for AS-RPA: %s", target.label)
            return []

        pairs = []

        # Try forward AS primer (3' end at SNP, extends rightward through SNP)
        fwd_as_primers = self._design_as_primer(
            genome_seq, snp_genomic, alt_base, ref_base,
            direction="fwd",
        )

        # Try reverse AS primer (3' end at SNP, extends leftward through SNP)
        rev_as_primers = self._design_as_primer(
            genome_seq, snp_genomic, alt_base, ref_base,
            direction="rev",
        )

        # Pair each AS primer with a standard flanking primer on the other side
        crrna_start = candidate.genomic_start
        crrna_end = candidate.genomic_end

        for as_primer in fwd_as_primers:
            # Forward AS → need reverse flanking downstream of crRNA
            rev_flanks = self._design_flanking_primer(
                genome_seq, crrna_end, "rev",
                min_dist=10, max_dist=self.amplicon_max - 50,
            )
            for rev in rev_flanks:
                amp_len = rev.amplicon_end - as_primer.amplicon_start
                if not (self.amplicon_min <= amp_len <= self.amplicon_max):
                    continue
                # Verify crRNA is within amplicon
                if crrna_start >= as_primer.amplicon_start and crrna_end <= rev.amplicon_end:
                    pairs.append(RPAPrimerPair(
                        fwd=as_primer,
                        rev=rev,
                        detection_strategy=DetectionStrategy.PROXIMITY,
                    ))

        for as_primer in rev_as_primers:
            # Reverse AS → need forward flanking upstream of crRNA
            fwd_flanks = self._design_flanking_primer(
                genome_seq, crrna_start, "fwd",
                min_dist=10, max_dist=self.amplicon_max - 50,
            )
            for fwd in fwd_flanks:
                amp_len = as_primer.amplicon_end - fwd.amplicon_start
                if not (self.amplicon_min <= amp_len <= self.amplicon_max):
                    continue
                if crrna_start >= fwd.amplicon_start and crrna_end <= as_primer.amplicon_end:
                    pairs.append(RPAPrimerPair(
                        fwd=fwd,
                        rev=as_primer,
                        detection_strategy=DetectionStrategy.PROXIMITY,
                    ))

        # Rank by: AS primer quality × Tm balance
        pairs.sort(key=self._pair_score, reverse=True)

        logger.info(
            "AS-RPA for %s: %d pairs (SNP at %d, %s>%s)",
            target.label, len(pairs), snp_genomic, ref_base, alt_base,
        )

        return pairs[:10]  # cap output

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _design_as_primer(
        self,
        genome: str,
        snp_pos: int,
        mut_base: str,
        wt_base: str,
        direction: str,
    ) -> list[RPAPrimer]:
        """Design allele-specific primers anchored at the SNP.

        The 3'-terminal base matches the mutant allele.
        Position -2 or -3 gets a deliberate mismatch to enhance selectivity.
        """
        primers = []
        glen = len(genome)

        for length in range(self.primer_len_min, self.primer_len_max + 1):
            for mm_offset in self.deliberate_mm_positions:
                if direction == "fwd":
                    # Forward primer: 3' end AT snp_pos (inclusive)
                    # Primer covers [snp_pos - length + 1, snp_pos + 1)
                    start = snp_pos - length + 1
                    end = snp_pos + 1
                    if start < 0 or end > glen:
                        continue

                    seq = list(genome[start:end].upper())

                    # Set 3' terminal base to mutant
                    seq[-1] = mut_base.upper()

                    # Set deliberate mismatch at offset from 3' end
                    mm_idx = length + mm_offset  # e.g. -2 → length-2
                    if 0 <= mm_idx < length - 1:
                        original = seq[mm_idx]
                        replacements = _STRONG_MISMATCHES.get(original, [])
                        if replacements:
                            seq[mm_idx] = replacements[0]

                    primer_seq = "".join(seq)
                    amp_start = start
                    amp_end = start + self.amplicon_max  # placeholder

                else:
                    # Reverse primer: 3' end at snp_pos (on rev-comp strand)
                    # Covers [snp_pos, snp_pos + length) on genome, then RC
                    start = snp_pos
                    end = snp_pos + length
                    if start < 0 or end > glen:
                        continue

                    seq = list(genome[start:end].upper())

                    # 3' terminal base: set genomic position to mut_base so
                    # that after RC the primer 3' = complement(mut_base),
                    # which Watson-Crick pairs with the plus-strand mutant allele.
                    seq[0] = mut_base.upper()

                    # Deliberate mismatch near 3' end (position 0 is 3' after RC)
                    # mm_idx in genomic coords: after RC, idx 1 → pos -2 from 3'
                    mm_idx = -mm_offset - 1  # e.g. -2 → idx 1
                    if 0 < mm_idx < length:
                        original = seq[mm_idx]
                        replacements = _STRONG_MISMATCHES.get(original, [])
                        if replacements:
                            seq[mm_idx] = replacements[0]

                    primer_seq = str(Seq("".join(seq)).reverse_complement())
                    amp_start = start - self.amplicon_max  # placeholder
                    amp_end = end

                # Check basic sequence validity
                if not set(primer_seq.upper()).issubset({"A", "T", "G", "C"}):
                    continue

                # Compute Tm
                try:
                    tm = float(mt.Tm_NN(Seq(primer_seq), nn_table=mt.DNA_NN3))
                except Exception:
                    continue

                if not (self.tm_min <= tm <= self.tm_max):
                    continue

                primers.append(RPAPrimer(
                    seq=primer_seq,
                    tm=tm,
                    direction=direction,
                    amplicon_start=max(0, amp_start),
                    amplicon_end=min(glen, amp_end),
                    is_allele_specific=True,
                    allele_specific_position=abs(mm_offset),
                ))

        return primers[:20]  # cap per direction

    def _design_flanking_primer(
        self,
        genome: str,
        anchor_pos: int,
        direction: str,
        min_dist: int = 10,
        max_dist: int = 150,
    ) -> list[RPAPrimer]:
        """Design standard (non-allele-specific) flanking primer."""
        primers = []
        glen = len(genome)

        if direction == "fwd":
            # Search upstream of anchor
            for offset in range(min_dist, max_dist):
                start = anchor_pos - offset
                for length in range(self.primer_len_min, self.primer_len_max + 1):
                    end = start + length
                    if start < 0 or end > glen:
                        continue
                    seq = genome[start:end].upper()
                    if not set(seq).issubset({"A", "T", "G", "C"}):
                        continue
                    try:
                        tm = float(mt.Tm_NN(Seq(seq), nn_table=mt.DNA_NN3))
                    except Exception:
                        continue
                    if self.tm_min <= tm <= self.tm_max:
                        primers.append(RPAPrimer(
                            seq=seq,
                            tm=tm,
                            direction="fwd",
                            amplicon_start=start,
                            amplicon_end=start + self.amplicon_max,
                        ))
                        break  # one per offset
        else:
            # Search downstream of anchor
            for offset in range(min_dist, max_dist):
                end = anchor_pos + offset
                for length in range(self.primer_len_min, self.primer_len_max + 1):
                    start = end - length
                    if start < 0 or end > glen:
                        continue
                    seq = str(Seq(genome[start:end].upper()).reverse_complement())
                    if not set(seq).issubset({"A", "T", "G", "C"}):
                        continue
                    try:
                        tm = float(mt.Tm_NN(Seq(seq), nn_table=mt.DNA_NN3))
                    except Exception:
                        continue
                    if self.tm_min <= tm <= self.tm_max:
                        primers.append(RPAPrimer(
                            seq=seq,
                            tm=tm,
                            direction="rev",
                            amplicon_start=end - self.amplicon_max,
                            amplicon_end=end,
                        ))
                        break

        return primers[:50]  # wider pool to allow distant pairing

    def _pair_score(self, pair: RPAPrimerPair) -> float:
        """Score an AS-RPA pair. Higher = better.

        Includes cfDNA soft penalty: amplicons > soft_penalty_start receive
        a linear penalty reflecting reduced template capture probability
        on fragmented circulating DNA.
        """
        fwd_tm = 1.0 - abs(pair.fwd.tm - self.tm_opt) / 8.0
        rev_tm = 1.0 - abs(pair.rev.tm - self.tm_opt) / 8.0
        amp_range = max(self.amplicon_max - self.amplicon_min, 1)
        amp = 1.0 - (pair.amplicon_length - self.amplicon_min) / amp_range
        as_bonus = 0.3 if pair.has_allele_specific_primer else 0.0
        # cfDNA soft penalty: linear ramp above SOFT_PENALTY_START
        cfdna_penalty = 0.0
        if pair.amplicon_length > RPA_AMPLICON_SOFT_PENALTY_START:
            penalty_range = max(self.amplicon_max - RPA_AMPLICON_SOFT_PENALTY_START, 1)
            cfdna_penalty = 0.3 * (
                (pair.amplicon_length - RPA_AMPLICON_SOFT_PENALTY_START)
                / penalty_range
            )
        return fwd_tm + rev_tm + amp + as_bonus - cfdna_penalty
