"""Main pipeline orchestration — end-to-end assay design.

Wires modules 1-10 together. A single call to `run()` executes the full
design workflow from WHO catalogue mutations to complete assay specifications
(crRNA + RPA primers + multiplex panel).

Pipeline stages:
  Module 1: Target resolution (resolver) → Target objects with genomic coords
  Module 2: PAM scanning (scanner) → CrRNACandidate objects (direct + proximity)
  Module 3: Candidate filtering (filters) → biophysically acceptable candidates
  Module 4: Off-target screening (screener) → OffTargetReport per candidate
  Module 5: Heuristic scoring (heuristic) → ScoredCandidate with composite score
  Module 5.5: Mismatch pair generation → WT/MUT pairs for discrimination
  Module 6: Synthetic mismatch enhancement → SM variants for borderline cases
  Module 6.5: Discrimination scoring → MUT/WT activity ratio per candidate
  Module 7: Multiplex optimization → select best candidate per target
  Module 8: RPA primer design → standard + AS-RPA primer pairs
  Module 8.5: Co-selection validation → verify crRNA-primer compatibility
  Module 9: Panel assembly → MultiplexPanel with IS6110 control
  Module 10: Export → JSON, TSV, and structured outputs

Key design decisions:
  - Scanner is initialised with Cas12a variant only; spacer lengths come from
    the scanner's built-in config (multi-length by default). NEVER override
    scanner lengths from pipeline config — this caused PAM desert failures.
  - Filter is initialised with OrganismPreset. Thresholds come from the
    organism preset, NOT from pipeline config.
  - PROXIMITY candidates are routed through AS-RPA primer design instead of
    standard RPA. The co-selection validator ensures compatibility.
  - The IS6110 M.tb species identification channel is added as a hardcoded
    literature-validated crRNA (Ai et al. 2019). No pipeline processing needed.

"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from compass.candidates.filters import CandidateFilter, OrganismPreset
from compass.candidates.mismatch import MismatchGenerator
from compass.candidates.scanner import PAMScanner, ScanResult
from compass.candidates.synthetic_mismatch import (
    EnhancementConfig,
    EnhancementReport,
    enhance_from_scored_candidates,
)
from compass.core.config import PipelineConfig
from compass.core.constants import IS6110_PAM, IS6110_SPACER
from compass.core.types import (
    CrRNACandidate,
    DetectionStrategy,
    DiscriminationScore,
    HeuristicScore,
    MLScore,
    MismatchPair,
    MultiplexPanel,
    Mutation,
    OffTargetReport,
    PanelMember,
    PAMVariant,
    RPAPrimer,
    RPAPrimerPair,
    ScoredCandidate,
    Strand,
    Target,
)
from compass.multiplex.optimizer import MultiplexOptimizer, OptimizationConfig
from compass.offtarget.screener import OffTargetScreener
from compass.primers.coselection import CoselectionValidator
from compass.scoring.base import Scorer
from compass.scoring.discrimination import HeuristicDiscriminationScorer, check_pam_disruption
from compass.scoring.heuristic import HeuristicScorer
from compass.scoring.learned_discrimination import LearnedDiscriminationScorer
from compass.scoring.sequence_ml import SequenceMLScorer
from compass.targets.resolver import TargetResolver

logger = logging.getLogger(__name__)


# ======================================================================
# Organism preset mapping
# ======================================================================

_ORGANISM_PRESETS = {
    "mtb": OrganismPreset.MYCOBACTERIUM_TUBERCULOSIS,
    "ecoli": OrganismPreset.ESCHERICHIA_COLI,
    "saureus": OrganismPreset.STAPHYLOCOCCUS_AUREUS,
    "paeruginosa": OrganismPreset.PSEUDOMONAS_AERUGINOSA,
}


class COMPASSPipeline:
    """End-to-end crRNA design pipeline.

    Usage:
        config = PipelineConfig.from_yaml("configs/mdr_14plex.yaml")
        pipeline = COMPASSPipeline(config)
        results = pipeline.run(mutations)
        panel = pipeline.run_full(mutations)  # end-to-end with primers
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._output = Path(config.output_dir)
        self._output.mkdir(parents=True, exist_ok=True)

        # Module 1: Target resolver
        # Known codon numbering offsets for M.tb WHO catalogue
        # katG: WHO S315T = H37Rv position 315 (no offset for most)
        # pncA: WHO H57D = H37Rv position 57
        # These may vary by genome build; brute-force scan handles mismatches
        mtb_offsets = {"rpoB": [0, 81], "katG": [0], "pncA": [0], "gyrA": [0, 3]}
        self.resolver = TargetResolver(
            fasta=str(config.reference.genome_fasta),
            gff=str(config.reference.gff_annotation)
            if config.reference.gff_annotation
            else None,
            genbank=str(config.reference.genbank_annotation)
            if config.reference.genbank_annotation
            else None,
            known_offsets=mtb_offsets,
        )

        # Module 2: PAM scanner
        cas_variant = config.candidates.resolve_enzyme_id()
        self.scanner = PAMScanner(cas_variant=cas_variant)

        # Module 3: Candidate filter
        organism = _ORGANISM_PRESETS.get(
            config.organism, OrganismPreset.GENERIC_HIGH_GC
        )
        self.filter = CandidateFilter(organism=organism, check_structure=False)

        # Module 4: Off-target screener
        from compass.offtarget.screener import ScreeningDatabase

        ot_databases = []
        if config.reference.genome_index:
            ot_databases.append(ScreeningDatabase(
                name="mtb",
                index_path=config.reference.genome_index,
                category="mtb",
            ))
        if config.reference.human_index:
            ot_databases.append(ScreeningDatabase(
                name="human",
                index_path=config.reference.human_index,
                category="human",
            ))
        for ntm_idx in config.reference.ntm_indices:
            ot_databases.append(ScreeningDatabase(
                name=f"ntm_{ntm_idx.stem}",
                index_path=ntm_idx,
                category="cross_reactivity",
            ))
        self.screener = OffTargetScreener(databases=ot_databases)

        # Module 5: Heuristic scorer
        self.heuristic_scorer = HeuristicScorer()

        # Module 5 ML: scorer selection (Compass-ML or SeqCNN)
        if (
            config.scoring.scorer == "compass_ml"
            and config.scoring.compass_ml_weights
            and Path(config.scoring.compass_ml_weights).exists()
        ):
            from compass.scoring.compass_ml_scorer import CompassMlScorer
            self.ml_scorer = CompassMlScorer(
                weights_path=config.scoring.compass_ml_weights,
                heuristic_fallback=self.heuristic_scorer,
                rnafm_cache_dir=(
                    str(config.scoring.rnafm_cache_dir)
                    if config.scoring.rnafm_cache_dir else None
                ),
                use_rlpa=config.scoring.compass_ml_use_rlpa,
                use_rnafm=config.scoring.compass_ml_use_rnafm,
                collect_embeddings=True,
            )
            logger.info("Using Compass-ML scorer (RLPA=%s, RNA-FM=%s)",
                        config.scoring.compass_ml_use_rlpa,
                        config.scoring.compass_ml_use_rnafm)
        else:
            self.ml_scorer = SequenceMLScorer(
                model_path=config.scoring.ml_model_path,
                heuristic_fallback=self.heuristic_scorer,
            )

        # Module 5.5: Mismatch generator
        self.mismatch_gen = MismatchGenerator()

        # Module 6.5: Discrimination scorer (learned model preferred, heuristic fallback)
        disc_method = getattr(config.scoring, "discrimination_method", "auto")
        disc_model_path = getattr(config.scoring, "discrimination_model_path", None)

        if disc_method == "heuristic":
            self.disc_scorer = HeuristicDiscriminationScorer(
                cas_variant=cas_variant,
                min_ratio=config.scoring.discrimination_min_ratio,
            )
        else:
            # "auto" or "learned": try learned model, fallback to heuristic
            self.disc_scorer = LearnedDiscriminationScorer(
                model_path=disc_model_path,
                cas_variant=cas_variant,
                min_ratio=config.scoring.discrimination_min_ratio,
            )
            if hasattr(self.disc_scorer, '_model_loaded'):
                logger.info(
                    "Discrimination scorer: %s",
                    "learned model" if self.disc_scorer._model_loaded else "heuristic fallback",
                )

        # Module 7: Multiplex optimizer
        self.optimizer = MultiplexOptimizer(OptimizationConfig(
            max_iterations=config.multiplex.max_iterations,
            efficiency_weight=config.multiplex.efficiency_weight,
            discrimination_weight=config.multiplex.discrimination_weight,
            cross_reactivity_weight=config.multiplex.cross_reactivity_weight,
        ))

        # Module 8.5: Co-selection validator
        self.coselection = CoselectionValidator(
            amplicon_min=config.primers.amplicon_min,
            amplicon_max=config.primers.amplicon_max,
        )

        # Genome sequence (lazy-loaded for primer design)
        self._genome_seq: Optional[str] = None

        # Per-module statistics (populated by run_full)
        self._stats: list[dict[str, Any]] = []

        logger.info(
            "COMPASSPipeline initialised: organism=%s, cas=%s, output=%s",
            config.organism,
            cas_variant,
            self._output,
        )

    @property
    def last_stats(self) -> list[dict[str, Any]]:
        """Module statistics from the most recent run_full() call."""
        return list(self._stats)

    # ==================================================================
    # Public API — Modules 1-5 (basic pipeline, backward compatible)
    # ==================================================================

    def run(
        self,
        mutations: list[Mutation],
    ) -> dict[str, list[ScoredCandidate]]:
        """Run Modules 1-5: target → scan → filter → OT → score.

        Returns {target_label: [ScoredCandidate, ...]} sorted by rank.
        This is the backward-compatible entry point used by the existing
        scripts and CLI.
        """
        results: dict[str, list[ScoredCandidate]] = {}

        for mutation in mutations:
            label = mutation.label
            target_dir = self._output / label
            target_dir.mkdir(parents=True, exist_ok=True)

            # Module 1: Resolve target
            try:
                target = self.resolver.resolve(mutation)
            except Exception as e:
                logger.error("Failed to resolve %s: %s", label, e)
                results[label] = []
                continue

            if target is None:
                logger.warning("Resolver returned None for %s", label)
                results[label] = []
                continue

            # Module 2: Scan for candidates
            scan_result = self.scanner.scan_detailed(target)
            candidates = scan_result.all_candidates

            logger.info(
                "%s: %d direct + %d proximity candidates (PAM desert: %s)",
                label,
                len(scan_result.direct_candidates),
                len(scan_result.proximity_candidates),
                scan_result.pam_desert,
            )

            if not candidates:
                logger.warning("No candidates for %s after scanning", label)
                results[label] = []
                continue

            # Module 3: Filter candidates
            filtered = self.filter.filter_batch(candidates)

            if not filtered:
                logger.warning(
                    "All %d candidates filtered for %s", len(candidates), label
                )
                # Relax and try again with all candidates
                filtered = candidates

            # Module 4: Off-target screening
            ot_reports = self.screener.screen_batch(filtered)

            # Module 5: Heuristic scoring
            scored = self.heuristic_scorer.score_batch(filtered, ot_reports)

            # Save intermediate results
            self._save_scored(scored, target_dir, scan_result)

            results[label] = scored

        return results

    # ==================================================================
    # Public API — Full pipeline (Modules 1-9)
    # ==================================================================

    def run_full(
        self,
        mutations: list[Mutation],
        parameter_profile: Optional["ParameterProfile"] = None,
    ) -> MultiplexPanel:
        """Run complete end-to-end pipeline: Modules 1-9.

        Args:
            mutations: List of mutations to design against.
            parameter_profile: Optional sensitivity-specificity profile
                that overrides default optimizer thresholds. If provided,
                the optimizer weights and thresholds are adjusted accordingly.

        Returns a MultiplexPanel with crRNAs, primers, discrimination
        scores, and the IS6110 positive control.
        """
        self._stats = []
        pipeline_t0 = time.perf_counter_ns()

        logger.info(
            "=" * 70 + "\n  COMPASS FULL PIPELINE: %d targets\n" + "=" * 70,
            len(mutations),
        )

        # --- Module 1: Target resolution ---
        t0 = time.perf_counter_ns()
        targets: list[Target] = []
        target_map: dict[str, Target] = {}
        n_resolved = 0
        unique_genes: set[str] = set()
        unique_drugs: set[str] = set()
        for mut in mutations:
            try:
                t = self.resolver.resolve(mut)
                if t is not None:
                    targets.append(t)
                    target_map[t.label] = t
                    n_resolved += 1
                    unique_genes.add(mut.gene)
                    unique_drugs.add(mut.drug.value if hasattr(mut.drug, 'value') else str(mut.drug))
            except Exception as e:
                logger.error("Failed to resolve %s: %s", mut.label, e)

        self._stats.append({
            "module_id": "M1", "module_name": "Target Resolution",
            "duration_ms": (time.perf_counter_ns() - t0) // 1_000_000,
            "candidates_in": len(mutations),
            "candidates_out": n_resolved,
            "detail": f"{n_resolved} WHO catalogue mutations \u2192 genomic coordinates on H37Rv (NC_000962.3)",
            "breakdown": {
                "genes": len(unique_genes),
                "drug_classes": len(unique_drugs),
                "reference": "H37Rv NC_000962.3",
            },
        })

        # --- Module 2: PAM scanning ---
        t0 = time.perf_counter_ns()
        scan_results: dict[str, ScanResult] = {}
        total_direct = 0
        total_proximity = 0
        n_deserts = 0
        for target in targets:
            sr = self.scanner.scan_detailed(target)
            scan_results[target.label] = sr
            total_direct += len(sr.direct_candidates)
            total_proximity += len(sr.proximity_candidates)
            if sr.pam_desert:
                n_deserts += 1
            logger.info(
                "%s: %d direct + %d proximity candidates (PAM desert: %s)",
                target.label,
                len(sr.direct_candidates),
                len(sr.proximity_candidates),
                sr.pam_desert,
            )
        total_candidates_m2 = total_direct + total_proximity
        total_positions = sum(sr.positions_scanned for sr in scan_results.values())
        total_pam_hits = sum(sr.pam_hits for sr in scan_results.values())

        self._stats.append({
            "module_id": "M2", "module_name": "PAM Scanner",
            "duration_ms": (time.perf_counter_ns() - t0) // 1_000_000,
            "candidates_in": len(targets),
            "candidates_out": total_candidates_m2,
            "detail": f"{total_positions:,} positions scanned \u2192 {total_pam_hits:,} PAM sites \u2192 {total_candidates_m2:,} candidates",
            "breakdown": {
                "positions_scanned": total_positions,
                "pam_hits": total_pam_hits,
                "direct_hits": total_direct,
                "proximity_hits": total_proximity,
                "pam_deserts": n_deserts,
                "cas_variant": self.config.candidates.resolve_enzyme_id(),
            },
        })

        # --- Module 3: Candidate filtering ---
        logger.info("Module 3: Filtering candidates...")
        t0 = time.perf_counter_ns()
        total_before_filter = 0
        total_after_filter = 0
        filtered_by_target: dict[str, list] = {}

        # Cap candidates per target to avoid combinatorial explosion with expanded PAMs.
        # Sort by PAM activity weight (best PAM first) so canonical TTTV candidates are
        # always retained. Cap at 50 per target — more than enough for downstream selection.
        MAX_CANDIDATES_PER_TARGET = 50

        for label, sr in scan_results.items():
            candidates = sr.all_candidates
            total_before_filter += len(candidates)
            if not candidates:
                filtered_by_target[label] = []
                continue
            # Pre-sort by PAM activity weight (best first) and cap to avoid
            # combinatorial explosion with expanded PAMs (9 PAMs × 6 lengths)
            candidates = sorted(
                candidates,
                key=lambda c: getattr(c, "pam_activity_weight", 1.0),
                reverse=True,
            )[:MAX_CANDIDATES_PER_TARGET]
            filtered = self.filter.filter_batch(candidates)
            if not filtered:
                logger.warning("All %d candidates filtered for %s", len(candidates), label)
                filtered = candidates
            total_after_filter += len(filtered)
            filtered_by_target[label] = filtered

        n_rejected_m3 = total_before_filter - total_after_filter
        self._stats.append({
            "module_id": "M3", "module_name": "Candidate Filter",
            "duration_ms": (time.perf_counter_ns() - t0) // 1_000_000,
            "candidates_in": total_before_filter,
            "candidates_out": total_after_filter,
            "detail": f"{total_before_filter:,} \u2192 {total_after_filter:,} ({n_rejected_m3:,} removed: GC, homopolymer, Tm)",
            "breakdown": {},
        })

        # --- Module 4: Off-target screening ---
        logger.info("Module 4: Off-target screening (Bowtie2)...")
        t0 = time.perf_counter_ns()
        total_before_ot = 0
        total_after_ot = 0
        ot_by_target: dict[str, list] = {}
        for label, filtered in filtered_by_target.items():
            total_before_ot += len(filtered)
            ot_reports = self.screener.screen_batch(filtered)
            # Keep only clean candidates
            clean = [f for f, r in zip(filtered, ot_reports) if r.is_clean]
            ot_by_target[label] = (filtered, ot_reports)
            total_after_ot += sum(1 for r in ot_reports if r.is_clean)

        has_bt2 = self.screener.has_valid_databases
        ot_rejected = total_before_ot - total_after_ot
        if has_bt2:
            ot_detail = f"{total_before_ot:,} \u2192 {total_after_ot:,} ({ot_rejected} off-target hits, Bowtie2 \u22643 mismatches)"
            ot_method = "Bowtie2 FM-index"
        else:
            ot_detail = f"{total_before_ot:,} \u2192 {total_after_ot:,} (Bowtie2 index not found \u2014 screening skipped)"
            ot_method = "skipped (no index)"
        self._stats.append({
            "module_id": "M4", "module_name": "Off-Target Screen",
            "duration_ms": (time.perf_counter_ns() - t0) // 1_000_000,
            "candidates_in": total_before_ot,
            "candidates_out": total_after_ot,
            "detail": ot_detail,
            "breakdown": {"method": ot_method, "max_mismatches": 3},
        })

        # --- Module 5: Heuristic scoring ---
        logger.info("Module 5: Heuristic scoring...")
        t0 = time.perf_counter_ns()
        scored_by_target: dict[str, list[ScoredCandidate]] = {}
        total_scored = 0
        all_composites: list[float] = []
        for label, (filtered, ot_reports) in ot_by_target.items():
            scored = self.heuristic_scorer.score_batch(filtered, ot_reports)
            scored_by_target[label] = scored
            total_scored += len(scored)
            all_composites.extend(sc.heuristic.composite for sc in scored)

            # Save intermediate results
            target_dir = self._output / label
            target_dir.mkdir(parents=True, exist_ok=True)
            self._save_scored(scored, target_dir, scan_results.get(label, ScanResult(target_label=label)))

        score_min = min(all_composites) if all_composites else 0
        score_max = max(all_composites) if all_composites else 0
        score_mean = sum(all_composites) / len(all_composites) if all_composites else 0

        heuristic_dur = (time.perf_counter_ns() - t0) // 1_000_000

        # --- Module 5 CNN: SeqCNN scoring + calibration + ensemble ---
        cnn_available = self.ml_scorer.model is not None
        all_cnn_raw: list[float] = []
        all_cnn_cal: list[float] = []
        all_ensemble: list[float] = []
        if cnn_available:
            t0_ml = time.perf_counter_ns()
            ml_name = "compass_ml" if hasattr(self.ml_scorer, '_predict_single') else "seq_cnn"
            # Batch all candidates for efficient inference
            all_sc = []
            for scored_list in scored_by_target.values():
                all_sc.extend(scored_list)
            if hasattr(self.ml_scorer, '_encode_context'):
                # CompassMlScorer: batch encode + predict
                contexts = [self.ml_scorer._encode_context(sc.candidate) for sc in all_sc]
                rnafm_embs = [self.ml_scorer._get_rnafm_embedding(sc.candidate) for sc in all_sc]
                raw_preds = self.ml_scorer._predict_batch(contexts, rnafm_embs)

                # Collect 128-dim RLPA embeddings for UMAP visualization
                if (getattr(self.ml_scorer, 'collect_embeddings', False)
                        and self.ml_scorer._last_batch_embeddings is not None):
                    batch_embs = self.ml_scorer._last_batch_embeddings
                    for idx, sc in enumerate(all_sc):
                        cand = sc.candidate
                        self.ml_scorer._collected_embeddings.append({
                            "target_label": cand.target_label,
                            "spacer_seq": cand.spacer_seq,
                            "embedding": batch_embs[idx],  # numpy (128,)
                            "score": raw_preds[idx],
                            "gc_content": cand.gc_content,
                            "detection_strategy": cand.detection_strategy.value,
                            "drug": None,  # populated below
                            "selected": False,
                        })
                    # Attach drug info from target metadata
                    drug_by_label = {}
                    for t in targets:
                        drug_by_label[t.label] = t.mutation.drug.value if hasattr(t.mutation.drug, 'value') else str(t.mutation.drug)
                    for emb_entry in self.ml_scorer._collected_embeddings:
                        if emb_entry["drug"] is None:
                            emb_entry["drug"] = drug_by_label.get(emb_entry["target_label"])

                    # --- UMAP: broad PAM scan for dense background ---
                    # Scan ±5kb genome windows × 6 spacer lengths × 2 strands
                    # per target. Capped at 8K to stay within container RAM.
                    import re as _re
                    import random as _rng
                    from compass.candidates.scanner import _gc
                    from types import SimpleNamespace

                    collected_spacers = {e["spacer_seq"] for e in self.ml_scorer._collected_embeddings}
                    bg_spacers: list[tuple] = []
                    _RC = str.maketrans("ACGT", "TGCA")
                    UMAP_WINDOW = 5000
                    UMAP_MAX_BG = 8000  # cap to avoid OOM on small containers
                    sp_lengths = self.scanner.lengths  # (18..23)

                    # Build a single regex for all PAM patterns (fast)
                    _IUPAC = {"V": "[ACG]", "N": "[ACGT]", "K": "[GT]",
                              "Y": "[CT]", "R": "[AG]", "W": "[AT]",
                              "S": "[GC]", "M": "[AC]", "B": "[CGT]",
                              "D": "[AGT]", "H": "[ACT]"}
                    pam_alts = []
                    for pd in self.scanner.pams:
                        rx = "".join(_IUPAC.get(c, c) for c in pd.pattern)
                        pam_alts.append(rx)
                    pam_re = _re.compile("|".join(f"(?={p})" for p in pam_alts))

                    genome = self._genome_seq or self._load_genome_seq()

                    for target in targets:
                        drug = drug_by_label.get(target.label, "OTHER")
                        if genome and hasattr(target, "flanking_start"):
                            center = target.flanking_start + len(target.flanking_seq) // 2
                            region = genome[max(0, center - UMAP_WINDOW):
                                           min(len(genome), center + UMAP_WINDOW)]
                        else:
                            region = target.flanking_seq.upper()

                        for seq_strand in [region, region[::-1].translate(_RC)]:
                            slen = len(seq_strand)
                            for m in pam_re.finditer(seq_strand):
                                sp_start = m.start() + 4
                                pam4 = seq_strand[m.start():sp_start]
                                for sp_len in sp_lengths:
                                    sp_end = sp_start + sp_len
                                    if sp_end > slen:
                                        continue
                                    spacer = seq_strand[sp_start:sp_end]
                                    if spacer in collected_spacers:
                                        continue
                                    collected_spacers.add(spacer)
                                    bg_spacers.append((spacer, pam4, target.label, drug, _gc(spacer)))

                    # Downsample if too many to fit in container memory
                    if len(bg_spacers) > UMAP_MAX_BG:
                        _rng.seed(42)
                        bg_spacers = _rng.sample(bg_spacers, UMAP_MAX_BG)

                    if bg_spacers:
                        logger.info(
                            "UMAP: encoding %d background PAM spacers for dense embedding...",
                            len(bg_spacers),
                        )

                        CHUNK = 512  # small chunks to limit peak memory
                        for start in range(0, len(bg_spacers), CHUNK):
                            batch = bg_spacers[start:start + CHUNK]
                            objs = [SimpleNamespace(pam_seq=p, spacer_seq=s)
                                    for s, p, _, _, _ in batch]
                            ctx = [self.ml_scorer._encode_context(c) for c in objs]
                            rfm = [self.ml_scorer._get_rnafm_embedding(c) for c in objs]
                            chunk_preds = self.ml_scorer._predict_batch(ctx, rfm)
                            del ctx, rfm  # free immediately

                            if self.ml_scorer._last_batch_embeddings is not None:
                                chunk_embs = self.ml_scorer._last_batch_embeddings
                                for idx in range(len(batch)):
                                    sp, pam, tl, dr, gc = bg_spacers[start + idx]
                                    self.ml_scorer._collected_embeddings.append({
                                        "target_label": tl,
                                        "spacer_seq": sp,
                                        "embedding": chunk_embs[idx],
                                        "score": chunk_preds[idx],
                                        "gc_content": gc,
                                        "detection_strategy": "direct",
                                        "drug": dr,
                                        "selected": False,
                                    })
                        logger.info(
                            "UMAP: total embeddings = %d (scored: %d + background: %d)",
                            len(self.ml_scorer._collected_embeddings),
                            len(all_sc), len(bg_spacers),
                        )
            else:
                # SeqCNN: individual predictions
                raw_preds = [self.ml_scorer._predict(sc.candidate) for sc in all_sc]
            for sc, raw_pred in zip(all_sc, raw_preds):
                sc.cnn_score = round(raw_pred, 4)
                sc.ml_scores = [MLScore(model_name=ml_name, predicted_efficiency=raw_pred)]
                all_cnn_raw.append(raw_pred)
                cal_pred = self.ml_scorer.calibrated_score(raw_pred)
                sc.cnn_calibrated = round(cal_pred, 4)
                all_cnn_cal.append(cal_pred)
                ens = self.ml_scorer.ensemble_score(sc.heuristic.composite, cal_pred)
                sc.ensemble_score = round(ens, 4)
                all_ensemble.append(ens)

            ml_dur = (time.perf_counter_ns() - t0_ml) // 1_000_000
            ml_rho = self.ml_scorer.validation_rho or 0.0

        # Emit combined M5 stats
        if cnn_available and all_cnn_cal:
            cal_min = min(all_cnn_cal)
            cal_max = max(all_cnn_cal)
            cal_mean = sum(all_cnn_cal) / len(all_cnn_cal)
            ens_min = min(all_ensemble)
            ens_max = max(all_ensemble)
            ens_mean = sum(all_ensemble) / len(all_ensemble)
            cal_T_val = self.ml_scorer.temperature
            cal_alpha_val = self.ml_scorer.alpha
            ens_rho = self.ml_scorer.calibration_meta.get("val_rho_ensemble", ml_rho)
            self._stats.append({
                "module_id": "M5", "module_name": "Scoring",
                "duration_ms": heuristic_dur + ml_dur,
                "candidates_in": total_scored,
                "candidates_out": total_scored,
                "detail": (
                    f"{total_scored:,} candidates scored \u2014 "
                    f"Heuristic ({score_min:.3f}\u2013{score_max:.3f}) \u00b7 "
                    f"SeqCNN calibrated T={cal_T_val:.1f} ({cal_min:.3f}\u2013{cal_max:.3f}) \u00b7 "
                    f"Ensemble \u03b1={cal_alpha_val:.2f} ({ens_min:.3f}\u2013{ens_max:.3f})"
                ),
                "breakdown": {
                    "heuristic_range": [round(score_min, 3), round(score_max, 3)],
                    "heuristic_mean": round(score_mean, 3),
                    "cnn_calibrated_range": [round(cal_min, 3), round(cal_max, 3)],
                    "cnn_calibrated_mean": round(cal_mean, 3),
                    "ensemble_range": [round(ens_min, 3), round(ens_max, 3)],
                    "ensemble_mean": round(ens_mean, 3),
                    "temperature": round(cal_T_val, 2),
                    "alpha": round(cal_alpha_val, 4),
                    "model": "seq_cnn",
                    "val_rho": round(ml_rho, 4),
                    "val_rho_ensemble": round(ens_rho, 4),
                },
            })
        else:
            self._stats.append({
                "module_id": "M5", "module_name": "Heuristic Scoring",
                "duration_ms": heuristic_dur,
                "candidates_in": total_scored,
                "candidates_out": total_scored,
                "detail": f"{total_scored:,} candidates scored (range {score_min:.3f}\u2013{score_max:.3f}, mean {score_mean:.3f})",
                "breakdown": {
                    "score_range": [round(score_min, 3), round(score_max, 3)],
                    "score_mean": round(score_mean, 3),
                },
            })

        # --- Module 5.5: Mismatch pair generation ---
        t0 = time.perf_counter_ns()
        logger.info("Module 5.5: Generating mismatch pairs...")
        pairs_by_target: dict[str, list[MismatchPair]] = {}
        total_pairs = 0
        n_direct_pairs = 0
        n_proximity_pairs = 0
        for label, scored_list in scored_by_target.items():
            target = target_map.get(label)
            if target is None:
                continue
            candidates = [sc.candidate for sc in scored_list]
            pairs = self.mismatch_gen.generate_batch(
                candidates, {label: target}
            )
            pairs_by_target[label] = pairs
            total_pairs += len(pairs)
            for p in pairs:
                if p.detection_strategy == DetectionStrategy.DIRECT:
                    n_direct_pairs += 1
                else:
                    n_proximity_pairs += 1
            # Copy wt_spacer back to candidates for downstream serialization
            pair_map = {p.candidate_id: p.wt_spacer for p in pairs}
            for sc in scored_list:
                wt = pair_map.get(sc.candidate.candidate_id)
                if wt:
                    sc.candidate.wt_spacer_seq = wt

        self._stats.append({
            "module_id": "M5.5", "module_name": "Mismatch Pairs",
            "duration_ms": (time.perf_counter_ns() - t0) // 1_000_000,
            "candidates_in": total_scored,
            "candidates_out": total_pairs,
            "detail": f"{total_pairs:,} MUT/WT spacer pairs generated ({n_direct_pairs} direct, {n_proximity_pairs} proximity)",
            "breakdown": {"direct_pairs": n_direct_pairs, "proximity_pairs": n_proximity_pairs},
        })

        # --- Module 6: Synthetic mismatch enhancement ---
        t0 = time.perf_counter_ns()
        logger.info("Module 6: Synthetic mismatch enhancement...")
        sm_config = EnhancementConfig(
            cas_variant=self.config.candidates.resolve_enzyme_id(),
            allow_double_synthetic=self.config.synthetic_mismatch.allow_double_sm,
            min_activity_vs_mut=self.config.synthetic_mismatch.min_activity_vs_mut,
            search_radius=6,
        )

        enhancement_reports: dict[str, list[EnhancementReport]] = {}
        n_sm_evaluated = 0
        n_sm_enhanced = 0
        for label, scored_list in scored_by_target.items():
            pairs = pairs_by_target.get(label, [])
            if scored_list and pairs:
                reports = enhance_from_scored_candidates(
                    scored_list, pairs, sm_config
                )
                enhancement_reports[label] = reports
                n_sm_evaluated += len(reports)
                n_enhanced = sum(1 for r in reports if r.enhancement_possible)
                n_sm_enhanced += n_enhanced
                if n_enhanced > 0:
                    logger.info(
                        "  %s: %d/%d candidates SM-enhanced",
                        label, n_enhanced, len(reports),
                    )

        self._stats.append({
            "module_id": "M6", "module_name": "SM Enhancement",
            "duration_ms": (time.perf_counter_ns() - t0) // 1_000_000,
            "candidates_in": n_sm_evaluated,
            "candidates_out": n_sm_enhanced,
            "detail": f"{n_sm_evaluated} candidates evaluated, {n_sm_enhanced} enhanced (seed positions 2\u20136)",
            "breakdown": {"strategy": "Single + double mismatches at positions 2\u20136"},
        })

        # --- Module 6.5: Discrimination scoring ---
        t0 = time.perf_counter_ns()
        logger.info("Module 6.5: Discrimination scoring...")
        for label, scored_list in scored_by_target.items():
            pairs = pairs_by_target.get(label, [])
            if pairs:
                self.disc_scorer.add_discrimination_batch(scored_list, pairs)

        # Log discrimination summary
        all_scored = [sc for scs in scored_by_target.values() for sc in scs]
        disc_summary = self.disc_scorer.analyze_panel_discrimination(all_scored)
        n_above_2x = 0
        n_above_3x = 0
        n_above_10x = 0
        for label, info in disc_summary.items():
            if info["best_ratio"] is not None:
                logger.info(
                    "  %s: best ratio=%.1f, %d/%d passing (strategy=%s)",
                    label, info["best_ratio"], info["n_passing"],
                    info["n_total"], info["strategy"],
                )
        for sc in all_scored:
            if sc.discrimination and sc.discrimination.wt_activity > 0:
                ratio = sc.discrimination.mut_activity / sc.discrimination.wt_activity
                if ratio >= 10:
                    n_above_10x += 1
                if ratio >= 3:
                    n_above_3x += 1
                if ratio >= 2:
                    n_above_2x += 1

        self._stats.append({
            "module_id": "M6.5", "module_name": "Discrimination",
            "duration_ms": (time.perf_counter_ns() - t0) // 1_000_000,
            "candidates_in": len(all_scored),
            "candidates_out": n_above_2x,
            "detail": f"{len(all_scored):,} \u2192 {n_above_2x} above 2\u00d7 threshold ({n_above_3x} diagnostic-grade \u22653\u00d7)",
            "breakdown": {"above_10x": n_above_10x, "above_3x": n_above_3x, "above_2x": n_above_2x},
        })

        # --- Module 7: Multiplex optimization ---
        t0 = time.perf_counter_ns()
        logger.info("Module 7: Multiplex panel optimization...")
        pool_size = sum(len(sl) for sl in scored_by_target.values())

        # Apply parameter profile overrides to optimizer config if provided
        if parameter_profile is not None:
            self.optimizer.config.efficiency_weight = parameter_profile.efficiency_weight
            self.optimizer.config.discrimination_weight = parameter_profile.discrimination_weight
            self.optimizer.config.cross_reactivity_weight = parameter_profile.cross_reactivity_weight
            self.optimizer.config.cross_reactivity_threshold = parameter_profile.cross_reactivity_max
            logger.info(
                "  Using profile '%s': eff_w=%.2f, disc_w=%.2f, xr_w=%.2f",
                parameter_profile.name,
                parameter_profile.efficiency_weight,
                parameter_profile.discrimination_weight,
                parameter_profile.cross_reactivity_weight,
            )

        panel = self.optimizer.optimize(targets, scored_by_target)

        self._stats.append({
            "module_id": "M7", "module_name": "Multiplex Optimization",
            "duration_ms": (time.perf_counter_ns() - t0) // 1_000_000,
            "candidates_in": pool_size,
            "candidates_out": panel.plex,
            "detail": f"{pool_size:,} \u2192 {panel.plex} selected (simulated annealing, {self.config.multiplex.max_iterations:,} iterations)",
            "breakdown": {"algorithm": "Simulated annealing", "iterations": self.config.multiplex.max_iterations},
        })

        # --- Attach SM enhancement data to selected panel members ---
        n_sm_attached = 0
        for member in panel.members:
            enh_reports = enhancement_reports.get(member.label, [])
            cid = member.selected_candidate.candidate.candidate_id
            for r in enh_reports:
                if r.candidate_id == cid and r.enhancement_possible and r.best_variant:
                    v = r.best_variant
                    member.has_sm = True
                    member.sm_enhanced_spacer = v.enhanced_spacer_seq
                    if v.synthetic_mismatches:
                        sm = v.synthetic_mismatches[0]
                        member.sm_position = sm.position
                        member.sm_original_base = sm.original_rna_base
                        member.sm_replacement_base = sm.synthetic_rna_base
                    member.sm_discrimination_score = r.best_discrimination_score
                    member.sm_improvement_factor = r.improvement_factor
                    n_sm_attached += 1
                    break
        if n_sm_attached:
            logger.info("SM enhancement: %d/%d panel members enhanced", n_sm_attached, panel.plex)

        # --- PAM-disruption binary discrimination flag ---
        n_pam_disrupted = 0
        for member in panel.members:
            pam_result = check_pam_disruption(
                member.selected_candidate.candidate,
                member.target,
            )
            # Store on the ScoredCandidate for downstream serialization
            member.selected_candidate.pam_disrupted = pam_result["pam_disrupted"]
            member.selected_candidate.pam_disruption_type = pam_result["pam_disruption_type"]
            if pam_result["pam_disrupted"]:
                n_pam_disrupted += 1
                logger.info(
                    "  PAM-disrupted: %s (%s)",
                    member.label, pam_result["pam_disruption_type"],
                )
        if n_pam_disrupted:
            logger.info("PAM disruption: %d/%d panel members", n_pam_disrupted, panel.plex)

        # --- Module 8: RPA primer design ---
        t0 = time.perf_counter_ns()
        logger.info("Module 8: RPA primer design...")
        genome_seq = self._load_genome_seq()
        n_with_primers = 0
        n_standard = 0
        n_asrpa = 0

        if genome_seq:
            from compass.primers.as_rpa import ASRPADesigner
            from compass.primers.standard_rpa import StandardRPADesigner

            primer_kwargs = dict(
                primer_len_min=self.config.primers.primer_length_min,
                primer_len_max=self.config.primers.primer_length_max,
                tm_min=self.config.primers.tm_min,
                tm_max=self.config.primers.tm_max,
                amplicon_min=self.config.primers.amplicon_min,
                amplicon_max=self.config.primers.amplicon_max,
            )
            as_rpa = ASRPADesigner(**primer_kwargs)
            std_rpa = StandardRPADesigner(**primer_kwargs)

            for member in panel.members:
                target = target_map.get(member.target.label)
                if target is None:
                    continue

                candidate = member.selected_candidate.candidate

                # Design primers — strategy-based dispatch
                if candidate.detection_strategy == DetectionStrategy.DIRECT:
                    primer_pairs = std_rpa.design(
                        candidate=candidate,
                        target=target,
                        genome_seq=genome_seq,
                    )
                else:
                    primer_pairs = as_rpa.design(
                        candidate=candidate,
                        target=target,
                        genome_seq=genome_seq,
                    )

                if primer_pairs:
                    # Module 8.5: Co-selection validation
                    best_pair, cosel_result = self.coselection.select_best_pair(
                        candidate, primer_pairs
                    )
                    if best_pair is not None:
                        member.primers = best_pair
                        logger.info(
                            "  %s: primers OK (amp=%dbp, score=%.2f)",
                            member.label,
                            cosel_result.amplicon_length,
                            cosel_result.score,
                        )
                    else:
                        member.primers = primer_pairs[0]
                        logger.warning(
                            "  %s: primer co-selection failed, using best available",
                            member.label,
                        )
                    n_with_primers += 1
                    if candidate.detection_strategy == DetectionStrategy.DIRECT:
                        n_standard += 1
                    else:
                        n_asrpa += 1
                else:
                    logger.warning(
                        "  %s: no primer pairs designed", member.label
                    )

        self._stats.append({
            "module_id": "M8", "module_name": "RPA Primer Design",
            "duration_ms": (time.perf_counter_ns() - t0) // 1_000_000,
            "candidates_in": panel.plex,
            "candidates_out": n_with_primers,
            "detail": f"{n_with_primers}/{panel.plex} primer pairs designed ({n_standard} standard, {n_asrpa} AS-RPA)",
            "breakdown": {"standard_rpa": n_standard, "allele_specific_rpa": n_asrpa, "tm_range": "57\u201372\u00b0C", "amplicon_range": "80\u2013250 bp"},
        })

        # --- Module 8.5a: AS-RPA thermodynamic discrimination ---
        t0 = time.perf_counter_ns()
        logger.info("Module 8.5: Post-primer analysis (AS-RPA discrimination + primer dimer check)...")
        n_asrpa_disc = 0
        try:
            from compass.primers.asrpa_discrimination import compute_asrpa_discrimination

            for member in panel.members:
                if member.selected_candidate.candidate.detection_strategy != DetectionStrategy.PROXIMITY:
                    continue
                if member.primers is None:
                    continue

                # Find the allele-specific primer
                as_primer = None
                if member.primers.fwd.is_allele_specific:
                    as_primer = member.primers.fwd
                elif member.primers.rev.is_allele_specific:
                    as_primer = member.primers.rev

                if as_primer is None:
                    logger.debug("  %s: no allele-specific primer found, skipping AS-RPA disc", member.label)
                    continue

                # The 3' terminal base of the AS primer is the mutant allele base
                primer_3p = as_primer.seq[-1].upper()

                # Determine the WT template base from the target mutation
                target = target_map.get(member.target.label)
                if target is None:
                    continue

                # For AS-RPA: primer 3' = mutant base, WT template base creates the mismatch
                # The ref codon vs alt codon tells us the WT base at this position
                ref_codon = target.ref_codon.upper() if target.ref_codon else ""
                alt_codon = target.alt_codon.upper() if target.alt_codon else ""

                # Determine WT base on the template strand opposite the primer 3' end
                # The primer matches the mutant allele; the WT template is the complement
                # of the ref base on the same strand as the primer
                wt_template_base = None
                if len(ref_codon) == 1 and len(alt_codon) == 1:
                    # Single-base: ref_base is in coding orientation
                    if as_primer.direction == "fwd":
                        # Forward primer reads plus strand; template is minus strand
                        # WT template base = complement of ref base
                        from Bio.Seq import Seq as _Seq
                        wt_template_base = str(_Seq(ref_codon).complement())
                    else:
                        wt_template_base = ref_codon
                elif len(ref_codon) == 3 and len(alt_codon) == 3:
                    # Find the differing position in the codon
                    for ci in range(3):
                        if ref_codon[ci] != alt_codon[ci]:
                            ref_base = ref_codon[ci]
                            break
                    else:
                        continue
                    # Determine strand orientation from genome
                    if genome_seq:
                        from Bio.Seq import Seq as _Seq
                        gcodon = genome_seq[target.genomic_pos:target.genomic_pos + 3].upper()
                        rc_ref = str(_Seq(ref_codon).reverse_complement())
                        if gcodon == ref_codon:
                            # Plus-strand gene
                            if as_primer.direction == "fwd":
                                wt_template_base = str(_Seq(ref_base).complement())
                            else:
                                wt_template_base = ref_base
                        elif gcodon == rc_ref:
                            # Minus-strand gene: coding strand IS the minus strand.
                            # ref_base is in coding orientation (= on minus strand).
                            # comp_ref = ref base on plus strand (genomic).
                            comp_ref = str(_Seq(ref_base).complement())
                            if as_primer.direction == "fwd":
                                # Fwd primer hybridizes to minus strand (coding)
                                # WT template = ref on minus strand = ref_base
                                wt_template_base = ref_base
                            else:
                                # Rev primer hybridizes to plus strand
                                # WT template = ref on plus strand = comp_ref
                                wt_template_base = comp_ref
                        else:
                            wt_template_base = str(_Seq(ref_base).complement()) if as_primer.direction == "fwd" else ref_base

                if wt_template_base is None or len(wt_template_base) != 1:
                    logger.debug("  %s: could not determine WT template base (ref=%s, alt=%s)", member.label, ref_codon, alt_codon)
                    continue

                # Check for deliberate penultimate mismatch
                has_pen_mm = as_primer.allele_specific_position is not None and as_primer.allele_specific_position <= 3

                try:
                    disc_result = compute_asrpa_discrimination(
                        primer_3prime_base=primer_3p,
                        wt_template_base=wt_template_base,
                        has_penultimate_mm=has_pen_mm,
                    )
                    member.asrpa_discrimination = disc_result
                    n_asrpa_disc += 1
                    logger.info(
                        "  %s: AS-RPA disc %s → ratio=%.1f× (%s)",
                        member.label,
                        disc_result["terminal_mismatch"],
                        disc_result["disc_ratio"],
                        disc_result["block_class"],
                    )
                except (ValueError, KeyError) as e:
                    logger.debug("  %s: AS-RPA disc failed: %s", member.label, e)

        except ImportError:
            logger.debug("asrpa_discrimination module not available")

        if n_asrpa_disc:
            logger.info("AS-RPA discrimination: %d/%d proximity targets scored", n_asrpa_disc, n_asrpa)

        # --- Module 8.5b: Multiplex primer dimer analysis ---
        t0_dimer = time.perf_counter_ns()
        n_dimer_flagged = 0
        try:
            from compass.multiplex.primer_dimer import analyse_panel_dimers

            primer_entries = []
            for member in panel.members:
                if member.primers is not None:
                    primer_entries.append({
                        "target": member.label,
                        "fwd": member.primers.fwd.seq,
                        "rev": member.primers.rev.seq,
                    })

            if len(primer_entries) >= 2:
                dimer_report = analyse_panel_dimers(primer_entries)

                # Populate the panel's primer_dimer_matrix (3'-anchored ΔG)
                panel.primer_dimer_matrix = dimer_report.dg_matrix_3prime.tolist()
                panel.primer_dimer_labels = dimer_report.oligo_labels

                # Store report summary
                panel.primer_dimer_report = {
                    "panel_dimer_score": dimer_report.panel_dimer_score,
                    "high_risk_pairs": dimer_report.high_risk_pairs,
                    "flagged_pairs": dimer_report.flagged_pairs,
                    "internal_dimers": dimer_report.internal_dimers,
                    "recommendations": dimer_report.recommendations,
                    "dg_matrix_full": dimer_report.dg_matrix_full.tolist(),
                }
                n_dimer_flagged = len(dimer_report.high_risk_pairs) + len(dimer_report.flagged_pairs)

                logger.info(
                    "Primer dimer analysis: %d oligos, score=%.4f, %d high-risk, %d moderate-risk",
                    len(dimer_report.oligo_labels),
                    dimer_report.panel_dimer_score,
                    len(dimer_report.high_risk_pairs),
                    len(dimer_report.flagged_pairs),
                )
        except ImportError:
            logger.debug("primer_dimer module not available")
        except Exception as e:
            logger.warning("Primer dimer analysis failed: %s", e)

        self._stats.append({
            "module_id": "M8.5", "module_name": "Post-Primer Analysis",
            "duration_ms": ((time.perf_counter_ns() - t0) + (time.perf_counter_ns() - t0_dimer)) // 1_000_000,
            "candidates_in": n_with_primers,
            "candidates_out": n_with_primers,
            "detail": f"AS-RPA disc: {n_asrpa_disc} scored | Dimer check: {n_dimer_flagged} flagged pairs",
            "breakdown": {"asrpa_scored": n_asrpa_disc, "dimer_flagged": n_dimer_flagged},
        })

        # --- Module 9: Panel assembly + IS6110 control ---
        t0 = time.perf_counter_ns()
        logger.info("Module 9: Panel assembly + IS6110 control...")
        pre_is6110_plex = panel.plex
        if self.config.multiplex.include_is6110:
            panel = self._add_is6110_control(panel)

            # Design primers for IS6110 (added after Module 8, needs its own step)
            is6110_member = panel.members[-1]  # just appended
            is6110_cand = is6110_member.selected_candidate.candidate
            if is6110_cand.candidate_id.startswith("IS6110"):
                is6110_pairs = []
                if genome_seq:
                    is6110_pairs = std_rpa.design(
                        candidate=is6110_cand,
                        target=is6110_member.target,
                        genome_seq=genome_seq,
                    )
                if is6110_pairs:
                    is6110_member.primers = is6110_pairs[0]
                    logger.info(
                        "  IS6110: primers OK (amp=%dbp)",
                        is6110_pairs[0].amplicon_length,
                    )
                else:
                    # Hard fallback: published IS6110 primers (Ai et al. 2019)
                    from compass.core.constants import (
                        IS6110_FWD_PRIMER,
                        IS6110_REV_PRIMER,
                        IS6110_AMPLICON_LENGTH,
                    )
                    from Bio.SeqUtils import MeltingTemp as _mt
                    from Bio.Seq import Seq as _Seq

                    fwd_tm = float(_mt.Tm_NN(_Seq(IS6110_FWD_PRIMER), nn_table=_mt.DNA_NN3))
                    rev_tm = float(_mt.Tm_NN(_Seq(IS6110_REV_PRIMER), nn_table=_mt.DNA_NN3))
                    mid = is6110_cand.genomic_start

                    is6110_member.primers = RPAPrimerPair(
                        fwd=RPAPrimer(
                            seq=IS6110_FWD_PRIMER,
                            tm=fwd_tm,
                            direction="fwd",
                            amplicon_start=mid - IS6110_AMPLICON_LENGTH // 2,
                            amplicon_end=mid + IS6110_AMPLICON_LENGTH // 2,
                        ),
                        rev=RPAPrimer(
                            seq=IS6110_REV_PRIMER,
                            tm=rev_tm,
                            direction="rev",
                            amplicon_start=mid - IS6110_AMPLICON_LENGTH // 2,
                            amplicon_end=mid + IS6110_AMPLICON_LENGTH // 2,
                        ),
                        detection_strategy=DetectionStrategy.DIRECT,
                    )
                    logger.info(
                        "  IS6110: using published primers (Ai et al. 2019, amp=%dbp)",
                        IS6110_AMPLICON_LENGTH,
                    )

        n_direct = len(panel.direct_members)
        n_prox = len(panel.proximity_members)
        self._stats.append({
            "module_id": "M9", "module_name": "Panel Assembly",
            "duration_ms": (time.perf_counter_ns() - t0) // 1_000_000,
            "candidates_in": pre_is6110_plex,
            "candidates_out": panel.plex,
            "detail": f"{pre_is6110_plex} candidates + IS6110 species control \u2192 final {panel.plex}-channel panel",
            "breakdown": {"direct_channels": n_direct, "proximity_channels": n_prox, "species_control": "IS6110"},
        })

        # --- Module 9.5: Top-K alternatives (Block 3) ---
        from compass.optimisation.top_k import collect_top_k
        top_k = parameter_profile.top_k if parameter_profile else 5
        top_k_results = collect_top_k(
            members=panel.members,
            candidates_by_target=scored_by_target,
            k=top_k,
        )
        self._top_k_results = top_k_results
        self._scored_by_target = scored_by_target

        # --- Module 9.6: Diagnostic metrics (Block 3) ---
        from compass.optimisation.metrics import compute_diagnostic_metrics
        eff_thresh = parameter_profile.efficiency_threshold if parameter_profile else 0.4
        disc_thresh = parameter_profile.discrimination_threshold if parameter_profile else 3.0
        self._diagnostic_metrics = compute_diagnostic_metrics(
            members=panel.members,
            candidates_by_target=scored_by_target,
            efficiency_threshold=eff_thresh,
            discrimination_threshold=disc_thresh,
        )
        logger.info(
            "  Diagnostics: sensitivity=%.3f, specificity=%.3f, coverage=%.3f",
            self._diagnostic_metrics.sensitivity,
            self._diagnostic_metrics.specificity,
            self._diagnostic_metrics.drug_class_coverage,
        )

        # --- Module 10: Export ---
        t0 = time.perf_counter_ns()
        self._export_panel(panel, scored_by_target, enhancement_reports)

        self._stats.append({
            "module_id": "M10", "module_name": "Export",
            "duration_ms": (time.perf_counter_ns() - t0) // 1_000_000,
            "candidates_in": panel.plex,
            "candidates_out": panel.plex,
            "detail": "JSON + TSV + FASTA structured output",
            "breakdown": {"formats": ["JSON", "TSV", "CSV", "FASTA"]},
        })

        # --- Module 11: UMAP embedding visualization (optional) ---
        if (hasattr(self.ml_scorer, 'get_collected_embeddings')
                and self.ml_scorer.get_collected_embeddings()):
            t0 = time.perf_counter_ns()
            try:
                from compass.viz.umap_panel import compute_panel_umap
                embeddings = self.ml_scorer.get_collected_embeddings()
                panel_labels = [m.target.label for m in panel.members]

                # Mark selected candidates by spacer_seq (not target_label,
                # which would match ALL candidates for that target)
                panel_spacers = set()
                for m in panel.members:
                    c = m.selected_candidate.candidate
                    panel_spacers.add(c.spacer_seq)
                for e in embeddings:
                    e["selected"] = e["spacer_seq"] in panel_spacers

                umap_path = self._output / "umap_embeddings.json"
                umap_result = compute_panel_umap(embeddings, panel_labels, umap_path)
                n_total = umap_result.get("n_total", 0)
                n_selected = umap_result.get("n_selected", 0)
                method = umap_result.get("stats", {}).get("method", "UMAP")

                self._stats.append({
                    "module_id": "M11", "module_name": "Embedding UMAP",
                    "duration_ms": (time.perf_counter_ns() - t0) // 1_000_000,
                    "candidates_in": n_total,
                    "candidates_out": n_selected,
                    "detail": f"{n_total:,} embeddings → 2D {method} ({n_selected} panel members highlighted)",
                    "breakdown": {"method": method, "n_total": n_total, "n_selected": n_selected},
                })
                logger.info(
                    "Module 11: UMAP computed — %d candidates → 2D (%d selected)",
                    n_total, n_selected,
                )
            except Exception as e:
                logger.warning("UMAP computation failed (non-fatal): %s", e)
            finally:
                self.ml_scorer.clear_embeddings()

        # Summary
        n_complete = panel.complete_members
        total_ms = (time.perf_counter_ns() - pipeline_t0) // 1_000_000
        logger.info(
            "\n" + "=" * 70 + "\n"
            "  PANEL COMPLETE: %d/%d targets with primers\n"
            "  Direct: %d | Proximity: %d | IS6110: %s\n"
            "  Panel score: %.4f | Total time: %dms\n"
            + "=" * 70,
            n_complete,
            panel.plex,
            n_direct,
            n_prox,
            "YES" if self.config.multiplex.include_is6110 else "NO",
            panel.panel_score or 0.0,
            total_ms,
        )

        return panel

    # ==================================================================
    # IS6110 MTB-positive control
    # ==================================================================

    def _add_is6110_control(self, panel: MultiplexPanel) -> MultiplexPanel:
        """Add the IS6110 species identification channel.

        IS6110 is a multi-copy insertion element specific to the
        M. tuberculosis complex. It serves as a positive control for
        both species confirmation and DNA extraction quality.

        The crRNA is literature-validated (Ai et al. 2019) and does not
        need pipeline-level design. 6-16 copies per genome ensures
        high sensitivity.
        """
        # Build a minimal Target for IS6110
        # Use IS6110 copy 1 coordinates (889021-890375, + strand in H37Rv)
        _IS6110_COPY1_START = 889021
        _IS6110_COPY1_MID = 889698  # midpoint of copy 1

        is6110_mutation = Mutation(
            gene="IS6110",
            position=0,
            ref_aa="N",
            alt_aa="N",
            notes="MTB species ID control (6-16 copies/genome)",
        )
        is6110_target = Target(
            mutation=is6110_mutation,
            genomic_pos=_IS6110_COPY1_MID,
            ref_codon="NNN",
            alt_codon="NNN",
            flanking_seq="N" * 100,
            flanking_start=_IS6110_COPY1_MID - 50,
        )

        # Build the hardcoded crRNA candidate
        # Spacer targets conserved region within IS6110 (Ai et al. 2019)
        # Use copy 1 midpoint as anchor for primer design
        is6110_candidate = CrRNACandidate(
            candidate_id="IS6110_ctrl_001",
            target_label="IS6110_DETECTION",
            spacer_seq=IS6110_SPACER,
            pam_seq=IS6110_PAM,
            pam_variant=PAMVariant.TTTV,
            strand=Strand.PLUS,
            genomic_start=_IS6110_COPY1_MID,
            genomic_end=_IS6110_COPY1_MID + 20,
            gc_content=sum(1 for b in IS6110_SPACER if b in "GC") / len(IS6110_SPACER),
            homopolymer_max=2,
            pam_activity_weight=1.0,
            detection_strategy=DetectionStrategy.DIRECT,
        )

        # Minimal scored candidate
        is6110_scored = ScoredCandidate(
            candidate=is6110_candidate,
            offtarget=OffTargetReport(
                candidate_id="IS6110_ctrl_001", is_clean=True
            ),
            heuristic=HeuristicScore(
                seed_position_score=1.0,
                gc_penalty=0.8,
                structure_penalty=0.9,
                homopolymer_penalty=1.0,
                offtarget_penalty=1.0,
                composite=0.95,
            ),
            discrimination=DiscriminationScore(
                wt_activity=0.0,
                mut_activity=1.0,
                model_name="literature_validated",
                is_measured=True,
            ),
        )

        is6110_member = PanelMember(
            target=is6110_target,
            selected_candidate=is6110_scored,
            channel="IS6110_MTB_ID",
        )

        # Add to panel
        panel.members.append(is6110_member)
        return panel

    # ==================================================================
    # Genome loading
    # ==================================================================

    def _load_genome_seq(self) -> Optional[str]:
        """Lazy-load the full genome sequence for primer design."""
        if self._genome_seq is not None:
            return self._genome_seq

        fasta_path = self.config.reference.genome_fasta
        if not Path(fasta_path).exists():
            logger.warning("Genome FASTA not found: %s", fasta_path)
            return None

        try:
            from Bio import SeqIO

            record = next(SeqIO.parse(str(fasta_path), "fasta"))
            self._genome_seq = str(record.seq).upper()
            logger.info(
                "Loaded genome: %s (%d bp)",
                record.id,
                len(self._genome_seq),
            )
            return self._genome_seq
        except Exception as e:
            logger.error("Failed to load genome: %s", e)
            return None

    # ==================================================================
    # Save / export
    # ==================================================================

    def _save_scored(
        self,
        scored: list[ScoredCandidate],
        target_dir: Path,
        scan_result: ScanResult,
    ) -> None:
        """Save intermediate scoring results for one target."""
        try:
            data = []
            for sc in scored:
                c = sc.candidate
                entry = {
                    "candidate_id": c.candidate_id,
                    "target_label": c.target_label,
                    "spacer_seq": c.spacer_seq,
                    "pam_seq": c.pam_seq,
                    "pam_variant": c.pam_variant.value,
                    "strand": c.strand.value,
                    "genomic_start": c.genomic_start,
                    "genomic_end": c.genomic_end,
                    "mutation_position_in_spacer": c.mutation_position_in_spacer,
                    "ref_base_at_mutation": c.ref_base_at_mutation,
                    "gc_content": round(c.gc_content, 3),
                    "detection_strategy": c.detection_strategy.value,
                    "proximity_distance": c.proximity_distance,
                    "heuristic_composite": round(sc.heuristic.composite, 4),
                    "rank": sc.rank,
                }
                data.append(entry)

            with open(target_dir / "scored_candidates.json", "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.debug("Failed to save scored candidates: %s", e)

    def _export_panel(
        self,
        panel: MultiplexPanel,
        scored_by_target: dict[str, list[ScoredCandidate]],
        enhancement_reports: dict[str, list[EnhancementReport]],
    ) -> None:
        """Export final panel results in multiple formats."""
        # JSON report
        report = {
            "pipeline": "COMPASS",
            "organism": self.config.organism,
            "plex": panel.plex,
            "panel_score": panel.panel_score,
            "targets": [],
        }

        for member in panel.members:
            c = member.selected_candidate.candidate
            disc = member.selected_candidate.discrimination

            entry = {
                "target": member.label,
                "drug": str(member.target.mutation.drug.value)
                if hasattr(member.target.mutation, "drug")
                else "N/A",
                "detection_strategy": c.detection_strategy.value,
                "spacer_seq": c.spacer_seq,
                "pam_seq": c.pam_seq,
                "pam_variant": c.pam_variant.value,
                "strand": c.strand.value,
                "heuristic_score": round(
                    member.selected_candidate.heuristic.composite, 4
                ),
                "discrimination_ratio": round(disc.ratio, 2) if disc else None,
                "discrimination_passes": disc.passes_threshold if disc else None,
                "discrimination": {
                    "ratio": round(disc.ratio, 2),
                    "mut_activity": round(disc.mut_activity, 4),
                    "wt_activity": round(disc.wt_activity, 4),
                    "model_name": disc.model_name,
                } if disc else None,
                "has_primers": member.primers is not None,
                "is_complete": member.is_complete,
            }

            if member.primers is not None:
                entry["fwd_primer"] = member.primers.fwd.seq
                entry["rev_primer"] = member.primers.rev.seq
                entry["amplicon_length"] = member.primers.amplicon_length
                entry["has_as_rpa"] = member.primers.has_allele_specific_primer

            if member.asrpa_discrimination is not None:
                entry["asrpa_discrimination"] = member.asrpa_discrimination

            # Enhancement info
            enh_reports = enhancement_reports.get(member.label, [])
            best_enh = None
            for r in enh_reports:
                if r.enhancement_possible and r.best_variant:
                    if best_enh is None or r.best_discrimination_score > best_enh.discrimination_score:
                        best_enh = r.best_variant
            if best_enh:
                entry["sm_enhanced_spacer"] = best_enh.enhanced_spacer_seq
                entry["sm_discrimination"] = round(
                    best_enh.discrimination_score, 2
                )
                entry["sm_improvement"] = round(
                    best_enh.discrimination_score
                    / max(best_enh.predicted_activity_vs_wt, 0.01),
                    1,
                )

            report["targets"].append(entry)

        # Save JSON
        json_path = self._output / "full_panel_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Panel report: %s", json_path)

        # Save TSV
        tsv_path = self._output / "full_panel_report.tsv"
        with open(tsv_path, "w") as f:
            headers = [
                "target",
                "drug",
                "strategy",
                "spacer",
                "pam",
                "score",
                "disc_ratio",
                "disc_pass",
                "has_primers",
                "amplicon_bp",
                "as_rpa",
            ]
            f.write("\t".join(headers) + "\n")

            for entry in report["targets"]:
                row = [
                    entry["target"],
                    entry.get("drug", ""),
                    entry["detection_strategy"],
                    entry["spacer_seq"],
                    entry["pam_seq"],
                    str(entry["heuristic_score"]),
                    str(entry.get("discrimination_ratio", "")),
                    str(entry.get("discrimination_passes", "")),
                    str(entry.get("has_primers", False)),
                    str(entry.get("amplicon_length", "")),
                    str(entry.get("has_as_rpa", "")),
                ]
                f.write("\t".join(row) + "\n")

        logger.info("Panel TSV: %s", tsv_path)

        # Save panel summary (backward compatible)
        summary_path = self._output / "panel_summary.json"
        summary = {
            "plex": panel.plex,
            "complete": panel.complete_members,
            "direct": len(panel.direct_members),
            "proximity": len(panel.proximity_members),
            "panel_score": panel.panel_score,
            "targets": {
                m.label: {
                    "n_candidates": len(scored_by_target.get(m.label, [])),
                    "strategy": m.selected_candidate.candidate.detection_strategy.value,
                    "score": m.selected_candidate.heuristic.composite,
                }
                for m in panel.members
            },
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
