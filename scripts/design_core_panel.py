#!/usr/bin/env python3
"""Design a 14-plex MDR-TB diagnostic panel with COMPASS.

Defines the WHO-critical resistance mutations for multidrug-resistant
tuberculosis and runs the complete COMPASS pipeline (Modules 1-9) to
produce crRNA + RPA primer specifications for each target.

Mutation selection based on:
  - WHO 2023 Catalogue of mutations in M. tuberculosis
  - Clinical frequency data from TBProfiler (Phelan et al., Genome Med 2019)
  - Drug resistance testing guidelines (WHO 2022)

Panel coverage:
  RIF:  rpoB S531L, H526Y, D516V          (covers ~95% of RIF-R)
  INH:  katG S315T, fabG1 c.-15C>T         (covers ~85% of INH-R)
  EMB:  embB M306V, M306I                  (covers ~60% of EMB-R)
  PZA:  pncA H57D, D49N                    (covers ~30% of PZA-R)
  FQ:   gyrA D94G, A90V                    (covers ~70% of FQ-R)
  AG:   rrs A1401G, C1402T, eis c.-14C>T   (covers ~85% of AG-R)
  MTB:  IS6110 (species ID control)

Usage:
    python scripts/design_core_panel.py \\
        -r data/references/H37Rv.fasta \\
        -g data/references/H37Rv.gff3 \\
        -o results/mdr_14plex_full
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure compass is importable from any working directory
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("design_core_panel")


from compass.core.config import (
    CandidateConfig,
    MultiplexConfig,
    PipelineConfig,
    PrimerConfig,
    ReferenceConfig,
    ScoringConfig,
    SyntheticMismatchConfig,
)
from compass.core.types import Drug, Mutation, MutationCategory


# ======================================================================
# WHO-critical 14-plex MDR-TB panel
# ======================================================================

def define_mdr_panel() -> list[Mutation]:
    """Define the 14 WHO-critical resistance mutations.

    Each mutation includes:
      - Gene/position/ref/alt from WHO 2023 Catalogue
      - Drug association and WHO confidence grading
      - Clinical frequency estimates
      - Mutation category for resolver dispatch
    """
    mutations = [
        # ── Rifampicin (RIF) — first-line ──
        Mutation(
            gene="rpoB", position=531, ref_aa="S", alt_aa="L",
            drug=Drug.RIFAMPICIN, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.42,
            notes="Most common RIF mutation (40-70% of RIF-R). "
                  "PAM desert in RRDR — requires proximity detection.",
        ),
        Mutation(
            gene="rpoB", position=526, ref_aa="H", alt_aa="Y",
            drug=Drug.RIFAMPICIN, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.15,
            notes="Second most common RIF mutation (10-20% of RIF-R).",
        ),
        Mutation(
            gene="rpoB", position=516, ref_aa="D", alt_aa="V",
            drug=Drug.RIFAMPICIN, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.08,
            notes="Third most common RIF mutation (~8% of RIF-R).",
        ),

        # ── Isoniazid (INH) — first-line ──
        Mutation(
            gene="katG", position=315, ref_aa="S", alt_aa="T",
            drug=Drug.ISONIAZID, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.65,
            notes="Most common INH mutation (50-90% of INH-R). "
                  "katG is on minus strand (Rv1908c).",
        ),
        Mutation(
            gene="fabG1", position=-15, ref_aa="C", alt_aa="T",
            nucleotide_change="c.-15C>T",
            drug=Drug.ISONIAZID, who_confidence="assoc w resistance",
            category=MutationCategory.PROMOTER,
            clinical_frequency=0.25,
            notes="fabG1 (mabA) promoter mutation (15-35% of INH-R). "
                  "Upregulates InhA, reducing INH efficacy.",
        ),

        # ── Ethambutol (EMB) — first-line ──
        Mutation(
            gene="embB", position=306, ref_aa="M", alt_aa="V",
            drug=Drug.ETHAMBUTOL, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.30,
            notes="Most common EMB mutation. M306V and M306I together "
                  "account for ~60% of EMB-R.",
        ),
        Mutation(
            gene="embB", position=306, ref_aa="M", alt_aa="I",
            drug=Drug.ETHAMBUTOL, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.25,
            notes="Second embB 306 variant.",
        ),

        # ── Pyrazinamide (PZA) — first-line ──
        Mutation(
            gene="pncA", position=57, ref_aa="H", alt_aa="D",
            drug=Drug.PYRAZINAMIDE, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.08,
            notes="pncA H57D — minus strand gene (Rv2043c). "
                  "PZA-R is genetically diverse; no single mutation dominates.",
        ),
        Mutation(
            gene="pncA", position=49, ref_aa="D", alt_aa="N",
            drug=Drug.PYRAZINAMIDE, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.05,
            notes="pncA D49N — common PZA-R hotspot.",
        ),

        # ── Fluoroquinolones (FQ) — Group A ──
        Mutation(
            gene="gyrA", position=94, ref_aa="D", alt_aa="G",
            drug=Drug.FLUOROQUINOLONE, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.35,
            notes="Most common FQ mutation (30-40% of FQ-R). "
                  "QRDR position 94.",
        ),
        Mutation(
            gene="gyrA", position=90, ref_aa="A", alt_aa="V",
            drug=Drug.FLUOROQUINOLONE, who_confidence="assoc w resistance",
            category=MutationCategory.AA_SUBSTITUTION,
            clinical_frequency=0.20,
            notes="Second most common FQ mutation (15-25% of FQ-R).",
        ),

        # ── Aminoglycosides (AG) — injectable ──
        Mutation(
            gene="rrs", position=1401, ref_aa="A", alt_aa="G",
            drug=Drug.AMINOGLYCOSIDE, who_confidence="assoc w resistance",
            category=MutationCategory.RRNA,
            clinical_frequency=0.60,
            notes="rrs A1401G — most common AG mutation (AMK, KAN, CAP). "
                  "rRNA gene, nucleotide-level annotation.",
        ),
        Mutation(
            gene="rrs", position=1402, ref_aa="C", alt_aa="T",
            drug=Drug.AMINOGLYCOSIDE, who_confidence="assoc w resistance",
            category=MutationCategory.RRNA,
            clinical_frequency=0.05,
            notes="rrs C1402T — less common AG mutation.",
        ),
        Mutation(
            gene="eis", position=-14, ref_aa="C", alt_aa="T",
            nucleotide_change="c.-14C>T",
            drug=Drug.AMINOGLYCOSIDE, who_confidence="assoc w resistance",
            category=MutationCategory.PROMOTER,
            clinical_frequency=0.10,
            notes="eis promoter mutation — KAN/AMK low-level resistance. "
                  "Upregulates EIS acetyltransferase.",
        ),
    ]

    return mutations


# ======================================================================
# Pipeline execution
# ======================================================================

def run_panel(reference: str, gff: str, output_dir: str) -> None:
    """Run the full COMPASS pipeline on the 14-plex MDR-TB panel."""
    t0 = time.time()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Build config
    config = PipelineConfig(
        name="mdr_14plex",
        output_dir=output,
        organism="mtb",
        reference=ReferenceConfig(
            genome_fasta=Path(reference),
            gff_annotation=Path(gff),
        ),
        candidates=CandidateConfig(
            use_enascas12a=True,
            cas_variant="enAsCas12a",
        ),
        synthetic_mismatch=SyntheticMismatchConfig(
            enabled=True,
            allow_double_sm=True,
            min_activity_vs_mut=0.25,
        ),
        scoring=ScoringConfig(
            use_heuristic=True,
            use_discrimination=True,
            discrimination_min_ratio=2.0,
            scorer="compass_ml",
            compass_ml_weights=Path("compass/weights/compass_ml_diagnostic.pt"),
            rnafm_cache_dir=Path("E:/compass-net-data/cache/rnafm"),
        ),
        multiplex=MultiplexConfig(
            max_plex=14,
            include_is6110=True,
            max_iterations=10_000,
        ),
        primers=PrimerConfig(
            enable_allele_specific=True,
        ),
    )

    # Define mutations
    mutations = define_mdr_panel()
    log.info("Panel: %d mutations defined", len(mutations))
    for m in mutations:
        log.info("  %s %s (%s)", m.gene, m.label, m.drug.value)

    # Run pipeline
    from compass.pipeline.runner import COMPASSPipeline

    pipeline = COMPASSPipeline(config)
    panel = pipeline.run_full(mutations)

    # Save additional panel summary
    summary = {
        "pipeline": "COMPASS v0.2.0",
        "panel_name": "MDR-TB 14-plex",
        "organism": "Mycobacterium tuberculosis H37Rv",
        "reference": reference,
        "n_targets": len(mutations),
        "n_panel_members": panel.plex,
        "n_complete": panel.complete_members,
        "n_direct": len(panel.direct_members),
        "n_proximity": len(panel.proximity_members),
        "panel_score": panel.panel_score,
        "elapsed_seconds": round(time.time() - t0, 1),
        "drug_coverage": {},
    }

    # Drug coverage
    for member in panel.members:
        drug = str(member.target.mutation.drug.value)
        if drug not in summary["drug_coverage"]:
            summary["drug_coverage"][drug] = []
        summary["drug_coverage"][drug].append({
            "target": member.label,
            "strategy": member.selected_candidate.candidate.detection_strategy.value,
            "score": round(member.selected_candidate.heuristic.composite, 3),
            "has_primers": member.is_complete,
        })

    summary_path = output / "panel_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Panel summary: %s", summary_path)

    # TSV summary — derived from panel, no second pipeline run
    tsv_path = output / "panel_summary.tsv"
    panel_by_label = {m.label: m for m in panel.members}

    with open(tsv_path, "w") as f:
        headers = [
            "target", "drug", "gene", "mutation", "strategy",
            "score", "disc_ratio", "has_primers", "complete",
        ]
        f.write("\t".join(headers) + "\n")

        for mut in mutations:
            label = mut.label
            member = panel_by_label.get(label)

            if member is not None:
                sc = member.selected_candidate
                disc = sc.discrimination
                f.write("\t".join([
                    label,
                    str(mut.drug.value),
                    mut.gene,
                    f"{mut.ref_aa}{mut.position}{mut.alt_aa}",
                    sc.candidate.detection_strategy.value,
                    f"{sc.heuristic.composite:.3f}",
                    f"{disc.ratio:.1f}" if disc else "N/A",
                    str(member.is_complete),
                    str(member.is_complete),
                ]) + "\n")
            else:
                f.write("\t".join([
                    label,
                    str(mut.drug.value),
                    mut.gene,
                    f"{mut.ref_aa}{mut.position}{mut.alt_aa}",
                    "unresolved",
                    "0.000",
                    "N/A",
                    "False",
                    "False",
                ]) + "\n")

    log.info("Panel TSV: %s", tsv_path)

    elapsed = time.time() - t0
    log.info(
        "\nPipeline complete in %.1fs. Output: %s/",
        elapsed, output,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Design 14-plex MDR-TB diagnostic panel",
    )
    parser.add_argument("-r", "--reference", required=True, help="H37Rv FASTA")
    parser.add_argument("-g", "--gff", required=True, help="H37Rv GFF3")
    parser.add_argument(
        "-o", "--output", default="results/mdr_14plex_full",
        help="Output directory",
    )
    args = parser.parse_args()

    run_panel(args.reference, args.gff, args.output)


if __name__ == "__main__":
    main()
