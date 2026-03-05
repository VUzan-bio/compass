"""COMPASS command-line interface.

Usage:
    compass run -c configs/mdr_14plex.yaml              # Modules 1-5 (basic)
    compass run-full -c configs/mdr_14plex.yaml          # Modules 1-10 (end-to-end)
    compass design -r H37Rv.fasta -g H37Rv.gff3          # 14-plex MDR-TB panel
    compass info                                          # Pipeline version + capabilities
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compass.cli")


def cmd_run(args: argparse.Namespace) -> None:
    """Run Modules 1-5 (basic pipeline)."""
    from compass.core.config import PipelineConfig
    from compass.pipeline.runner import COMPASSPipeline

    config = PipelineConfig.from_yaml(args.config)
    if args.output:
        config.output_dir = Path(args.output)

    pipeline = COMPASSPipeline(config)
    mutations = _load_mutations(args)
    results = pipeline.run(mutations)

    total = sum(len(v) for v in results.values())
    logger.info("Pipeline complete: %d targets, %d total candidates", len(results), total)


def cmd_run_full(args: argparse.Namespace) -> None:
    """Run Modules 1-10 (full end-to-end pipeline)."""
    from compass.core.config import PipelineConfig
    from compass.pipeline.runner import COMPASSPipeline

    config = PipelineConfig.from_yaml(args.config)
    if args.output:
        config.output_dir = Path(args.output)

    pipeline = COMPASSPipeline(config)
    mutations = _load_mutations(args)

    t0 = time.time()
    panel = pipeline.run_full(mutations)
    elapsed = time.time() - t0

    logger.info(
        "Full pipeline complete in %.1fs: %d/%d targets with primers "
        "(direct=%d, proximity=%d, score=%.4f)",
        elapsed, panel.complete_members, panel.plex,
        len(panel.direct_members), len(panel.proximity_members),
        panel.panel_score or 0.0,
    )


def cmd_design(args: argparse.Namespace) -> None:
    """Design the 14-plex MDR-TB panel (convenience wrapper)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from scripts.design_core_panel import run_panel
    run_panel(args.reference, args.gff, args.output)


def cmd_info(args: argparse.Namespace) -> None:
    """Print pipeline version and capabilities."""
    from compass import __version__
    info = {
        "name": "COMPASS",
        "full_name": "Guide RNA Automated Resistance Diagnostics",
        "version": __version__,
        "modules": [
            "1. Target resolution (WHO catalogue → genomic coordinates)",
            "2. PAM scanning (multi-PAM, multi-length, proximity fallback)",
            "3. Candidate filtering (organism-aware biophysical thresholds)",
            "4. Off-target screening (Bowtie2 + heuristic fallback)",
            "5. Heuristic scoring (Kim et al. 2018 feature-weighted)",
            "5.5 Mismatch pair generation (WT/MUT spacer derivation)",
            "6. Synthetic mismatch enhancement (2-5× → 10-100× discrimination)",
            "6.5 Discrimination scoring (B-JEPA / heuristic)",
            "7. Multiplex optimization (simulated annealing)",
            "8. RPA primer design (standard + allele-specific AS-RPA)",
            "8.5 Co-selection validation (crRNA-primer compatibility)",
            "9. Panel assembly (+ IS6110 MTB species ID control)",
            "10. Export (JSON, TSV, structured reports)",
        ],
        "cas12a_variants": ["AsCas12a", "enAsCas12a", "LbCas12a", "FnCas12a", "Cas12a_ultra"],
        "scoring_backends": ["heuristic", "seq_cnn", "jepa_efficiency", "jepa_discrimination"],
        "organisms_tested": ["M. tuberculosis (H37Rv)"],
    }
    print(json.dumps(info, indent=2))


def _load_mutations(args: argparse.Namespace) -> list:
    """Load mutations from a panel definition file or use default MDR-TB panel."""
    if hasattr(args, "panel") and args.panel:
        from compass.targets.who_parser import WHOCatalogueParser
        parser = WHOCatalogueParser()
        return parser.parse(args.panel)

    # Default: use the MDR-TB 14-plex panel
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from scripts.design_core_panel import define_mdr_panel
    return define_mdr_panel()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="compass",
        description="COMPASS — CRISPR-Cas12a diagnostic assay design pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Basic pipeline (Modules 1-5)")
    p_run.add_argument("-c", "--config", required=True, help="YAML config file")
    p_run.add_argument("-o", "--output", help="Override output directory")
    p_run.add_argument("-p", "--panel", help="Mutation panel file (TSV/CSV)")
    p_run.set_defaults(func=cmd_run)

    # run-full
    p_full = sub.add_parser("run-full", help="Full pipeline (Modules 1-10)")
    p_full.add_argument("-c", "--config", required=True, help="YAML config file")
    p_full.add_argument("-o", "--output", help="Override output directory")
    p_full.add_argument("-p", "--panel", help="Mutation panel file (TSV/CSV)")
    p_full.set_defaults(func=cmd_run_full)

    # design
    p_design = sub.add_parser("design", help="Design 14-plex MDR-TB panel")
    p_design.add_argument("-r", "--reference", required=True, help="H37Rv FASTA")
    p_design.add_argument("-g", "--gff", required=True, help="H37Rv GFF3")
    p_design.add_argument("-o", "--output", default="results/mdr_14plex_full")
    p_design.set_defaults(func=cmd_design)

    # info
    p_info = sub.add_parser("info", help="Pipeline version and capabilities")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
