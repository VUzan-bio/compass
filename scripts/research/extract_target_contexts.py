"""Extract ±250bp wildtype and mutant genomic contexts for all 42 SNP targets.

These contexts are the input for EVO-2 LLR computation (weeks 3-4).

Usage:
    python scripts/research/extract_target_contexts.py

Output:
    data/card/target_contexts.json — JSON with wildtype_context, mutant_context,
                                      mutation_pos for each target
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/card")
CONTEXT_FLANK = 250  # ±250 bp = 500 bp total context for EVO-2

# Organism → reference FASTA path
REFERENCE_PATHS = {
    "mtb": Path("data/references/H37Rv.fasta"),
    "ecoli": Path("data/references/ecoli_K12.fasta"),
    "saureus": Path("data/references/saureus_NCTC8325.fasta"),
    "ngonorrhoeae": Path("data/references/ngono_FA1090.fasta"),
}

# Organism → GFF3 annotation path
GFF_PATHS = {
    "mtb": Path("data/references/H37Rv.gff3"),
    "ecoli": Path("data/references/ecoli_K12.gff3"),
    "saureus": Path("data/references/saureus_NCTC8325.gff3"),
    "ngonorrhoeae": Path("data/references/ngono_FA1090.gff3"),
}


def load_genome(fasta_path: Path) -> str:
    """Load genome sequence from FASTA, concatenating all contigs."""
    seqs = []
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                seqs.append(line.strip().upper())
    return "".join(seqs)


def extract_contexts() -> list[dict]:
    """Extract wildtype and mutant contexts for each SNP target.

    Uses the TargetResolver to find the exact genomic position of each
    mutation, then extracts ±250bp flanking context.
    """
    from compass.core.types import Drug, Mutation, MutationCategory
    from compass.targets.resolver import TargetResolver

    # Load target metadata
    import csv
    meta_path = OUTPUT_DIR / "amr_target_metadata.csv"
    if not meta_path.exists():
        logger.error("Run setup_card_data.py first to generate %s", meta_path)
        return []

    with open(meta_path) as f:
        targets = list(csv.DictReader(f))

    results = []
    genomes: dict[str, str] = {}
    resolvers: dict[str, TargetResolver] = {}

    for t in targets:
        org_id = t["organism_id"]
        gene = t["gene"]
        mutation_str = t["mutation"]
        label = t["label"]

        # Load genome and resolver lazily
        if org_id not in genomes:
            fasta = REFERENCE_PATHS.get(org_id)
            gff = GFF_PATHS.get(org_id)
            if not fasta or not fasta.exists():
                logger.warning("Reference genome not found for %s, skipping", org_id)
                continue
            genomes[org_id] = load_genome(fasta)
            if gff and gff.exists():
                resolvers[org_id] = TargetResolver(
                    fasta=fasta,
                    gff=gff,
                )
            else:
                logger.warning("GFF not found for %s, skipping", org_id)
                continue

        genome = genomes[org_id]
        resolver = resolvers.get(org_id)
        if resolver is None:
            continue

        # Parse mutation string to extract ref/alt/position
        # Formats: S531L (AA), C-15T (promoter/rRNA), A1401G (rRNA)
        ref_aa, position, alt_aa = _parse_mutation(mutation_str)
        if ref_aa is None:
            logger.warning("Could not parse mutation %s for %s, skipping", mutation_str, label)
            continue

        # Build Mutation object
        drug_map = {
            "RIF": Drug.RIFAMPICIN, "INH": Drug.ISONIAZID, "EMB": Drug.ETHAMBUTOL,
            "PZA": Drug.PYRAZINAMIDE, "FQ": Drug.FLUOROQUINOLONE,
            "AG": Drug.AMINOGLYCOSIDE, "CIP": Drug.CIPROFLOXACIN,
            "AMP": Drug.AMPICILLIN, "ESC": Drug.CEFTRIAXONE,
            "AZM": Drug.AZITHROMYCIN, "FUS": Drug.OTHER,
            "TMP": Drug.TRIMETHOPRIM_SULFAMETHOXAZOLE, "DAP": Drug.DAPTOMYCIN,
            "SUL": Drug.TRIMETHOPRIM_SULFAMETHOXAZOLE, "CAR": Drug.CARBAPENEM,
        }
        drug = drug_map.get(t["drug_class"], Drug.OTHER)

        try:
            # Determine mutation category
            if alt_aa == "del":
                category = MutationCategory.DELETION
                alt_aa = "-"  # Pydantic expects max 3 chars
            elif position < 0:
                category = MutationCategory.PROMOTER
            elif gene.startswith("rrs") or gene.startswith("23S"):
                category = MutationCategory.RRNA
            elif len(ref_aa) == 1 and ref_aa in "ACGT":
                category = MutationCategory.NUCLEOTIDE_SNP
            else:
                category = MutationCategory.AA_SUBSTITUTION

            mutation = Mutation(
                gene=gene,
                ref_aa=ref_aa,
                position=position,
                alt_aa=alt_aa,
                drug=drug,
                category=category,
            )
            target = resolver.resolve(mutation)
        except Exception as e:
            logger.warning("Failed to resolve %s: %s", label, e)
            target = None

        if target is None:
            logger.warning("Resolver returned None for %s", label)
            # For targets that can't be resolved, use gene-level context
            results.append({
                "label": label,
                "organism_id": org_id,
                "gene": gene,
                "mutation": mutation_str,
                "resolved": False,
                "wildtype_context": None,
                "mutant_context": None,
                "mutation_pos": None,
                "genomic_pos": None,
            })
            continue

        # Extract context
        gpos = target.genomic_pos
        start = max(0, gpos - CONTEXT_FLANK)
        end = min(len(genome), gpos + CONTEXT_FLANK + 1)
        wt_context = genome[start:end]
        mutation_pos_in_context = gpos - start

        # Build mutant context by substituting the codon/nucleotide
        mut_context = list(wt_context)
        # For SNPs, replace the single nucleotide or codon
        if len(target.alt_codon) == 1:
            # Single nucleotide (rRNA, promoter)
            mut_context[mutation_pos_in_context] = target.alt_codon
        elif len(target.alt_codon) == 3:
            # Codon substitution (AA change)
            codon_start = mutation_pos_in_context - (mutation_pos_in_context % 3)
            for i, nt in enumerate(target.alt_codon):
                if codon_start + i < len(mut_context):
                    mut_context[codon_start + i] = nt
        mut_context = "".join(mut_context)

        results.append({
            "label": label,
            "organism_id": org_id,
            "gene": gene,
            "mutation": mutation_str,
            "resolved": True,
            "wildtype_context": wt_context,
            "mutant_context": mut_context,
            "mutation_pos": mutation_pos_in_context,
            "genomic_pos": gpos,
            "context_start": start,
            "context_end": end,
            "context_length": len(wt_context),
            "gc_content": round(
                (wt_context.count("G") + wt_context.count("C")) / len(wt_context), 4
            ),
        })
        logger.info(
            "  %s: pos=%d, context=%d bp, GC=%.1f%%",
            label, gpos, len(wt_context),
            results[-1]["gc_content"] * 100,
        )

    return results


def _parse_mutation(mut: str) -> tuple:
    """Parse mutation string into (ref, position, alt).

    Handles:
      S531L  → ("S", 531, "L")    AA substitution
      C-15T  → ("C", -15, "T")    Promoter/rRNA
      A1401G → ("A", 1401, "G")   rRNA
      A-35del → ("A", -35, "del") Deletion
    """
    import re

    # Deletion: A-35del
    m = re.match(r"^([A-Z])(-?\d+)(del)$", mut)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)

    # Standard: ref + position + alt
    m = re.match(r"^([A-Z])(-?\d+)([A-Z])$", mut)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)

    return None, None, None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Extracting Target Contexts for EVO-2 LLR ===")
    contexts = extract_contexts()

    resolved = [c for c in contexts if c.get("resolved")]
    unresolved = [c for c in contexts if not c.get("resolved")]

    logger.info("")
    logger.info("Resolved: %d / %d targets", len(resolved), len(contexts))
    if unresolved:
        logger.info("Unresolved: %s", [c["label"] for c in unresolved])

    # Save
    out_path = OUTPUT_DIR / "target_contexts.json"
    with open(out_path, "w") as f:
        json.dump(contexts, f, indent=2)
    logger.info("Saved to %s", out_path)

    # Summary by organism
    by_org: dict[str, list] = {}
    for c in resolved:
        by_org.setdefault(c["organism_id"], []).append(c)
    for org, ctx_list in sorted(by_org.items()):
        gc_vals = [c["gc_content"] for c in ctx_list]
        logger.info(
            "  %s: %d targets, GC range %.1f%%–%.1f%%",
            org, len(ctx_list),
            min(gc_vals) * 100, max(gc_vals) * 100,
        )


if __name__ == "__main__":
    main()
