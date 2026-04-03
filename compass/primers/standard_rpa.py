"""Standard RPA primer design for DIRECT detection candidates.

For DIRECT detection candidates the crRNA spacer overlaps the mutation site,
so discrimination comes from Cas12a mismatch intolerance — not from primers.
Primers simply need to amplify an 80–120 bp region that contains the crRNA
binding site.

Blood cfDNA constraint (BRIDGE project):
  Circulating free DNA fragments are ~100–160 bp (median ~140 bp).
  Amplicons >120 bp risk spanning fragment junctions → amplification failure.
  Soft penalty applied above 100 bp; hard reject above 120 bp.

Design rules:
  - Primer length 25–38 nt (widened for M.tb 65.6% GC genome)
  - Tm 57–72°C (RPA uses recombinase at 37°C, tolerant of Tm variation)
  - Amplicon 80–120 bp containing the full crRNA target site
  - Spacer binding site ≥15 bp from each primer 3' end
  - Both primers are standard/symmetrical — no allele-specificity needed

References:
  - Piepenburg et al., PLoS Biol 2006 (RPA mechanism)
  - Li et al., Cell Rep Med 2022 (RPA-CRISPR design rules)
  - Lo et al., Sci Transl Med 2010 (cfDNA fragment sizes)
  - Mouliere et al., Sci Transl Med 2018 (cfDNA size profiling)
"""

from __future__ import annotations

import logging
from typing import Optional

from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq

from compass.core.constants import (
    RPA_AMPLICON_MAX_CFDNA,
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
    Target,
)

logger = logging.getLogger(__name__)

_VALID_BASES = {"A", "T", "G", "C"}

# Maximum self-folding ΔG (kcal/mol) for primer acceptance.
# Primers more negative than this form stable hairpins that compete
# with recombinase binding and reduce RPA efficiency.
_SELF_FOLD_DG_THRESHOLD = -8.0


class StandardRPADesigner:
    """Design flanking RPA primers for DIRECT detection candidates."""

    def __init__(
        self,
        primer_len_min: int = RPA_PRIMER_LENGTH_MIN,
        primer_len_max: int = RPA_PRIMER_LENGTH_MAX,
        tm_min: float = RPA_TM_MIN,
        tm_max: float = RPA_TM_MAX,
        amplicon_min: int = RPA_AMPLICON_MIN,
        amplicon_max: int = RPA_AMPLICON_MAX_GENOMIC,
        tm_opt: float = 64.5,
        spacer_min_gap: int = 15,
    ) -> None:
        self.primer_len_min = primer_len_min
        self.primer_len_max = primer_len_max
        self.tm_min = tm_min
        self.tm_max = tm_max
        self.tm_opt = tm_opt
        self.amplicon_min = amplicon_min
        self.amplicon_max = amplicon_max
        self.spacer_min_gap = spacer_min_gap

    def design(
        self,
        candidate: CrRNACandidate,
        target: Target,
        genome_seq: str,
    ) -> list[RPAPrimerPair]:
        """Design standard flanking RPA primer pairs for a DIRECT candidate.

        Args:
            candidate: DIRECT crRNA candidate with genomic coordinates.
            target: Resolved target (unused for DIRECT — primers anchor on crRNA).
            genome_seq: Full genome sequence string.

        Returns:
            Ranked list of RPAPrimerPair (max 10).
        """
        if candidate.detection_strategy != DetectionStrategy.DIRECT:
            logger.debug(
                "Skipping standard RPA for non-direct candidate %s",
                candidate.candidate_id,
            )
            return []

        crrna_start = candidate.genomic_start
        crrna_end = candidate.genomic_end

        # Design forward primers upstream of crRNA
        fwd_primers = self._design_flanking_primer(
            genome_seq,
            anchor_pos=crrna_start - self.spacer_min_gap,
            direction="fwd",
        )

        # Design reverse primers downstream of crRNA
        rev_primers = self._design_flanking_primer(
            genome_seq,
            anchor_pos=crrna_end + self.spacer_min_gap,
            direction="rev",
        )

        if not fwd_primers or not rev_primers:
            logger.debug(
                "No flanking primers for %s (fwd=%d, rev=%d)",
                candidate.candidate_id,
                len(fwd_primers),
                len(rev_primers),
            )
            return []

        # Pair and filter
        pairs: list[RPAPrimerPair] = []
        for fwd in fwd_primers:
            for rev in rev_primers:
                amp_start = fwd.amplicon_start
                amp_end = rev.amplicon_end
                amp_len = amp_end - amp_start

                # Amplicon length check
                if amp_len < self.amplicon_min or amp_len > self.amplicon_max:
                    continue

                # crRNA must be entirely within amplicon
                if crrna_start < amp_start or crrna_end > amp_end:
                    continue

                # Self-folding check: reject primers with stable hairpins
                fwd_dg = self._self_fold_dg(fwd.seq)
                if fwd_dg is not None and fwd_dg < _SELF_FOLD_DG_THRESHOLD:
                    continue
                rev_dg = self._self_fold_dg(rev.seq)
                if rev_dg is not None and rev_dg < _SELF_FOLD_DG_THRESHOLD:
                    continue

                # Amplicon GC window check: extract amplicon and flag extremes
                amp_seq = genome_seq[amp_start:amp_end].upper() if amp_end <= len(genome_seq) else ""
                gc_extremes = self.amplicon_gc_extremes(amp_seq) if amp_seq else []
                # Skip if >3 extreme GC windows (likely un-amplifiable)
                if len(gc_extremes) > 3:
                    continue

                pair = RPAPrimerPair(
                    fwd=fwd,
                    rev=rev,
                    amplicon_seq=amp_seq,
                    detection_strategy=DetectionStrategy.DIRECT,
                )
                pairs.append(pair)

        # Rank by score
        pairs.sort(key=self._pair_score, reverse=True)
        return pairs[:10]

    def _design_flanking_primer(
        self,
        genome: str,
        anchor_pos: int,
        direction: str,
        min_dist: int = 10,
        max_dist: int = 150,
    ) -> list[RPAPrimer]:
        """Design standard flanking primers in one direction."""
        primers: list[RPAPrimer] = []
        glen = len(genome)

        if direction == "fwd":
            for offset in range(min_dist, max_dist):
                start = anchor_pos - offset
                for length in range(self.primer_len_min, self.primer_len_max + 1):
                    end = start + length
                    if start < 0 or end > glen:
                        continue
                    seq = genome[start:end].upper()
                    if not set(seq).issubset(_VALID_BASES):
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
            for offset in range(min_dist, max_dist):
                end = anchor_pos + offset
                for length in range(self.primer_len_min, self.primer_len_max + 1):
                    start = end - length
                    if start < 0 or end > glen:
                        continue
                    seq = str(Seq(genome[start:end].upper()).reverse_complement())
                    if not set(seq).issubset(_VALID_BASES):
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
                        break  # one per offset

        return primers[:20]

    @staticmethod
    def _self_fold_dg(seq: str) -> Optional[float]:
        """Estimate self-folding ΔG for a primer using ViennaRNA RNAfold.

        Returns ΔG in kcal/mol, or None if ViennaRNA is unavailable.
        More negative = more stable hairpin = worse for RPA.
        """
        import subprocess
        try:
            result = subprocess.run(
                ["RNAfold", "--noPS"],
                input=seq.upper(),
                capture_output=True,
                text=True,
                timeout=5,
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                mfe_str = lines[1].split("(")[-1].rstrip(")")
                return float(mfe_str.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        return None

    def _pair_score(self, pair: RPAPrimerPair) -> float:
        """Score a standard RPA pair. Higher = better.

        Includes cfDNA soft penalty: amplicons > 100 bp receive a linear
        penalty reflecting reduced template capture probability on
        fragmented circulating DNA.
        """
        fwd_tm = 1.0 - abs(pair.fwd.tm - self.tm_opt) / 8.0
        rev_tm = 1.0 - abs(pair.rev.tm - self.tm_opt) / 8.0
        amp_range = max(self.amplicon_max - self.amplicon_min, 1)
        amp = 1.0 - (pair.amplicon_length - self.amplicon_min) / amp_range
        # cfDNA soft penalty: linear ramp above SOFT_PENALTY_START
        cfdna_penalty = 0.0
        if pair.amplicon_length > RPA_AMPLICON_SOFT_PENALTY_START:
            penalty_range = max(self.amplicon_max - RPA_AMPLICON_SOFT_PENALTY_START, 1)
            cfdna_penalty = 0.3 * (
                (pair.amplicon_length - RPA_AMPLICON_SOFT_PENALTY_START)
                / penalty_range
            )
        return fwd_tm + rev_tm + amp - cfdna_penalty

    @staticmethod
    def amplicon_gc_extremes(
        amplicon_seq: str,
        window: int = 15,
        gc_max: float = 0.85,
        gc_min: float = 0.15,
    ) -> list[dict]:
        """Identify locally extreme GC regions in an amplicon.

        Sliding window of `window` nt across the amplicon. Regions with
        GC > gc_max or < gc_min are problematic:
          - High GC windows cause polymerase stalling and RPA dropout
          - Low GC windows cause unstable R-loop binding for Cas12a

        Returns a list of {start, end, gc, type} dicts for flagged windows.
        Empty list = no extreme regions.
        """
        seq = amplicon_seq.upper()
        if len(seq) < window:
            return []

        flagged = []
        for i in range(len(seq) - window + 1):
            w = seq[i:i + window]
            gc = sum(1 for nt in w if nt in ("G", "C")) / window
            if gc > gc_max:
                flagged.append({"start": i, "end": i + window, "gc": round(gc, 3), "type": "high"})
            elif gc < gc_min:
                flagged.append({"start": i, "end": i + window, "gc": round(gc, 3), "type": "low"})

        # Merge overlapping flagged windows of same type
        if not flagged:
            return []

        merged = [flagged[0]]
        for f in flagged[1:]:
            prev = merged[-1]
            if f["type"] == prev["type"] and f["start"] <= prev["end"]:
                prev["end"] = f["end"]
                prev["gc"] = max(prev["gc"], f["gc"]) if f["type"] == "high" else min(prev["gc"], f["gc"])
            else:
                merged.append(f)

        return merged
