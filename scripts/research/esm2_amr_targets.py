"""ESM-2 protein-level log-likelihood ratio on AMR targets.

Parallel experiment to EVO-2 DNA-level LLR (evo2_amr_targets.py).
Computes protein-level LLR for the 28 amino acid substitution targets
(excludes 14 promoter/rRNA targets where ESM-2 is not applicable).

Key hypothesis: if protein LLR correlates with prevalence for AA
substitutions but NOT for promoter/rRNA mutations (where ESM-2 LLR ≡ 0),
this supports a multi-scale genomic representation story:
  - DNA-level (EVO-2): captures all mutation types
  - Protein-level (ESM-2): captures AA substitutions only

Usage:
    # Full run (requires ESM-2 model):
    python scripts/research/esm2_amr_targets.py

    # Analysis only (uses cached LLR values):
    python scripts/research/esm2_amr_targets.py --analysis-only

Output:
    results/research/esm2_llr/
        llr_results.csv         — per-target ESM-2 LLR values
        correlation_analysis.csv — Spearman rho per organism and pooled
        comparison_evo2.csv     — DNA vs protein LLR comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/research/esm2_llr")
EVO2_RESULTS_DIR = Path("results/research/evo2_llr")
METADATA_PATH = Path("data/card/amr_target_metadata.csv")

# Protein sequences for target genes (UniProt reference sequences)
# These are loaded dynamically from GFF + genome in extract step;
# hardcoded subset for genes with known AA substitution targets.
# Format: (organism_id, gene, uniprot_id_or_locus_tag)
PROTEIN_TARGETS = {
    # MTB
    "mtb_rpoB": ("mtb", "rpoB", "Rv0667"),
    "mtb_katG": ("mtb", "katG", "Rv1908c"),
    "mtb_embB": ("mtb", "embB", "Rv3795"),
    "mtb_pncA": ("mtb", "pncA", "Rv2043c"),
    "mtb_gyrA": ("mtb", "gyrA", "Rv0006"),
    # E. coli
    "ecoli_gyrA": ("ecoli", "gyrA", "b2231"),
    "ecoli_parC": ("ecoli", "parC", "b3019"),
    # S. aureus
    "saureus_gyrA": ("saureus", "gyrA", "SAOUHSC_00006"),
    "saureus_grlA": ("saureus", "grlA", "SAOUHSC_00472"),
    "saureus_rpoB": ("saureus", "rpoB", "SAOUHSC_00538"),
    "saureus_fusA": ("saureus", "fusA", "SAOUHSC_00541"),
    "saureus_dfrB": ("saureus", "dfrB", "SAOUHSC_00730"),
    "saureus_mprF": ("saureus", "mprF", "SAOUHSC_01359"),
    # N. gonorrhoeae
    "ngono_penA": ("ngonorrhoeae", "penA", "NGO0628"),
    "ngono_gyrA": ("ngonorrhoeae", "gyrA", "NGO1218"),
    "ngono_parC": ("ngonorrhoeae", "parC", "NGO1525"),
    "ngono_folP": ("ngonorrhoeae", "folP", "NGO1649"),
}

# AA substitution targets only (28 of 42)
# Excludes: rrs_A1401G, rrs_C1402T, eis_C-14T, fabG1_C-15T (promoter/rRNA)
#           ampC_C-42T (promoter), 23S_rRNA_C2611T, 23S_rRNA_A2059G (rRNA)
#           mtrR_A-35del (deletion)
AA_TARGETS = [
    # MTB
    ("mtb", "rpoB", "S531L", "RIF"),
    ("mtb", "rpoB", "H526Y", "RIF"),
    ("mtb", "rpoB", "D516V", "RIF"),
    ("mtb", "katG", "S315T", "INH"),
    ("mtb", "embB", "M306V", "EMB"),
    ("mtb", "embB", "M306I", "EMB"),
    ("mtb", "pncA", "H57D", "PZA"),
    ("mtb", "pncA", "D49N", "PZA"),
    ("mtb", "gyrA", "D94G", "FQ"),
    ("mtb", "gyrA", "A90V", "FQ"),
    # E. coli
    ("ecoli", "gyrA", "S83L", "CIP"),
    ("ecoli", "gyrA", "D87N", "CIP"),
    ("ecoli", "parC", "S80I", "CIP"),
    ("ecoli", "parC", "E84V", "CIP"),
    # S. aureus
    ("saureus", "gyrA", "S84L", "CIP"),
    ("saureus", "grlA", "S80F", "CIP"),
    ("saureus", "grlA", "S80Y", "CIP"),
    ("saureus", "rpoB", "H481N", "RIF"),
    ("saureus", "rpoB", "S464P", "RIF"),
    ("saureus", "fusA", "L461K", "FUS"),
    ("saureus", "dfrB", "F99Y", "TMP"),
    ("saureus", "mprF", "S295L", "DAP"),
    # N. gonorrhoeae
    ("ngonorrhoeae", "penA", "A501V", "ESC"),
    ("ngonorrhoeae", "penA", "A501T", "ESC"),
    ("ngonorrhoeae", "penA", "G545S", "ESC"),
    ("ngonorrhoeae", "penA", "I312M", "ESC"),
    ("ngonorrhoeae", "penA", "V316T", "ESC"),
    ("ngonorrhoeae", "penA", "T483S", "ESC"),
    ("ngonorrhoeae", "gyrA", "S91F", "CIP"),
    ("ngonorrhoeae", "gyrA", "D95A", "CIP"),
    ("ngonorrhoeae", "gyrA", "D95G", "CIP"),
    ("ngonorrhoeae", "parC", "D86N", "CIP"),
    ("ngonorrhoeae", "parC", "S87R", "CIP"),
    ("ngonorrhoeae", "folP", "R228S", "SUL"),
]


def load_metadata() -> dict[str, dict]:
    """Load target metadata (prevalence, drug class, etc.)."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found at {METADATA_PATH}. "
            "Run: python scripts/research/setup_card_data.py"
        )
    with open(METADATA_PATH) as f:
        rows = list(csv.DictReader(f))
    return {r["label"]: r for r in rows}


def _parse_aa_mutation(mut: str) -> tuple[str, int, str]:
    """Parse AA substitution string like S531L -> (S, 531, L)."""
    import re
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", mut)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    raise ValueError(f"Cannot parse AA mutation: {mut}")


def compute_esm2_llr_single(
    protein_sequence: str,
    position: int,
    ref_aa: str,
    alt_aa: str,
    model=None,
    alphabet=None,
    batch_converter=None,
) -> float:
    """Compute ESM-2 log-likelihood ratio at a single position.

    LLR = log P(alt_aa | context) - log P(ref_aa | context)

    Uses masked marginal scoring: mask the target position and compute
    log-probabilities for ref vs alt amino acid.

    Args:
        protein_sequence: full protein sequence (wildtype)
        position: 1-indexed position of the mutation
        ref_aa: reference amino acid (single letter)
        alt_aa: alternate amino acid (single letter)
        model: pre-loaded ESM-2 model (if None, loads fresh)
        alphabet: ESM-2 alphabet
        batch_converter: ESM-2 batch converter

    Returns:
        Log-likelihood ratio (scalar float).
    """
    import torch

    if model is None:
        model, alphabet, batch_converter = _load_esm2()

    # Verify ref AA matches sequence
    pos_idx = position - 1  # Convert to 0-indexed
    if pos_idx < 0 or pos_idx >= len(protein_sequence):
        raise ValueError(
            f"Position {position} out of range for protein of length {len(protein_sequence)}"
        )
    if protein_sequence[pos_idx] != ref_aa:
        logger.warning(
            "Ref AA mismatch at position %d: expected %s, found %s",
            position, ref_aa, protein_sequence[pos_idx],
        )

    # Mask the target position
    masked_seq = protein_sequence[:pos_idx] + "<mask>" + protein_sequence[pos_idx + 1:]

    # Tokenize
    data = [("protein", masked_seq)]
    _, _, tokens = batch_converter(data)

    device = next(model.parameters()).device
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[], return_contacts=False)
        logits = results["logits"]  # (1, seq_len, vocab_size)

    # Get log-probabilities at the masked position
    # Token position = pos_idx + 1 (due to <cls> token at position 0)
    token_pos = pos_idx + 1
    log_probs = torch.nn.functional.log_softmax(logits[0, token_pos], dim=-1)

    # Get token indices for ref and alt AAs
    ref_idx = alphabet.get_idx(ref_aa)
    alt_idx = alphabet.get_idx(alt_aa)

    llr = log_probs[alt_idx].item() - log_probs[ref_idx].item()
    return llr


def _load_esm2():
    """Load ESM-2 model (650M parameter version)."""
    import torch
    try:
        import esm
    except ImportError:
        raise ImportError(
            "ESM-2 not installed. Install via: pip install fair-esm"
        )

    logger.info("Loading ESM-2 (esm2_t33_650M_UR50D)...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    logger.info("ESM-2 loaded on %s", device)

    return model, alphabet, batch_converter


def compute_all_targets(
    protein_sequences: dict[str, str],
) -> list[dict]:
    """Compute ESM-2 LLR for all AA substitution targets.

    Args:
        protein_sequences: dict mapping "organism_gene" -> protein sequence

    Returns:
        List of result dicts.
    """
    model, alphabet, batch_converter = _load_esm2()

    results = []
    for org_id, gene, mutation, drug in AA_TARGETS:
        label = f"{gene}_{mutation}"
        ref_aa, position, alt_aa = _parse_aa_mutation(mutation)

        key = f"{org_id}_{gene}"
        protein_seq = protein_sequences.get(key)
        if protein_seq is None:
            logger.warning("No protein sequence for %s, skipping %s", key, label)
            results.append({
                "label": label,
                "organism_id": org_id,
                "gene": gene,
                "mutation": mutation,
                "drug_class": drug,
                "esm2_llr": None,
                "abs_esm2_llr": None,
                "ref_aa": ref_aa,
                "position": position,
                "alt_aa": alt_aa,
                "ref_log_prob": None,
                "alt_log_prob": None,
            })
            continue

        try:
            llr = compute_esm2_llr_single(
                protein_seq, position, ref_aa, alt_aa,
                model=model, alphabet=alphabet, batch_converter=batch_converter,
            )

            # Also get individual log-probs for analysis
            import torch
            pos_idx = position - 1
            masked_seq = protein_seq[:pos_idx] + "<mask>" + protein_seq[pos_idx + 1:]
            data = [("protein", masked_seq)]
            _, _, tokens = batch_converter(data)
            device = next(model.parameters()).device
            tokens = tokens.to(device)
            with torch.no_grad():
                logits = model(tokens, repr_layers=[], return_contacts=False)["logits"]
            log_probs = torch.nn.functional.log_softmax(logits[0, pos_idx + 1], dim=-1)
            ref_lp = log_probs[alphabet.get_idx(ref_aa)].item()
            alt_lp = log_probs[alphabet.get_idx(alt_aa)].item()

            results.append({
                "label": label,
                "organism_id": org_id,
                "gene": gene,
                "mutation": mutation,
                "drug_class": drug,
                "esm2_llr": round(llr, 6),
                "abs_esm2_llr": round(abs(llr), 6),
                "ref_aa": ref_aa,
                "position": position,
                "alt_aa": alt_aa,
                "ref_log_prob": round(ref_lp, 6),
                "alt_log_prob": round(alt_lp, 6),
            })
            logger.info(
                "  %s: LLR=%.4f (ref_lp=%.3f, alt_lp=%.3f)",
                label, llr, ref_lp, alt_lp,
            )
        except Exception as e:
            logger.warning("Failed for %s: %s", label, e)
            results.append({
                "label": label,
                "organism_id": org_id,
                "gene": gene,
                "mutation": mutation,
                "drug_class": drug,
                "esm2_llr": None,
                "abs_esm2_llr": None,
                "ref_aa": ref_aa,
                "position": position,
                "alt_aa": alt_aa,
                "ref_log_prob": None,
                "alt_log_prob": None,
            })

    return results


def correlation_analysis(
    llr_results: list[dict],
    metadata: dict[str, dict],
) -> list[dict]:
    """Correlate |ESM-2 LLR| with clinical prevalence."""
    from scipy import stats as sp_stats

    analysis = []

    # Merge with metadata
    merged = []
    for r in llr_results:
        if r["esm2_llr"] is None:
            continue
        meta = metadata.get(r["label"], {})
        prev = meta.get("prevalence_pct")
        merged.append({
            **r,
            "prevalence_pct": float(prev) if prev else None,
        })

    # Per-organism
    by_org: dict[str, list] = {}
    for m in merged:
        by_org.setdefault(m["organism_id"], []).append(m)

    for org_id, targets in sorted(by_org.items()):
        valid = [(t["abs_esm2_llr"], t["prevalence_pct"])
                 for t in targets if t["prevalence_pct"] is not None]

        logger.info("\n=== %s (N=%d, %d with prevalence) ===",
                    org_id, len(targets), len(valid))

        if len(valid) >= 5:
            x, y = zip(*valid)
            rho, p = sp_stats.spearmanr(x, y)
            logger.info("  rho(|ESM2_LLR|, prevalence) = %.3f (p=%.4f)", rho, p)
            analysis.append({
                "organism": org_id,
                "variable": "prevalence",
                "n": len(valid),
                "spearman_rho": round(rho, 4),
                "p_value": round(p, 4),
                "model": "esm2",
            })
        else:
            logger.info("  Too few targets (N=%d, need >=5)", len(valid))

    # Pooled
    all_valid = [(m["abs_esm2_llr"], m["prevalence_pct"])
                 for m in merged if m["prevalence_pct"] is not None]
    if len(all_valid) >= 10:
        x, y = zip(*all_valid)
        rho, p = sp_stats.spearmanr(x, y)
        logger.info("\n=== POOLED (N=%d) ===", len(all_valid))
        logger.info("  rho(|ESM2_LLR|, prevalence) = %.3f (p=%.4f)", rho, p)
        analysis.append({
            "organism": "pooled",
            "variable": "prevalence",
            "n": len(all_valid),
            "spearman_rho": round(rho, 4),
            "p_value": round(p, 4),
            "model": "esm2",
        })

    return analysis


def compare_with_evo2(
    esm2_results: list[dict],
    metadata: dict[str, dict],
) -> list[dict]:
    """Compare ESM-2 protein LLR with EVO-2 DNA LLR for shared targets.

    Key question: do DNA and protein LLR capture different information?
    """
    from scipy import stats as sp_stats

    # Load EVO-2 results if available
    evo2_path = EVO2_RESULTS_DIR / "llr_results.csv"
    if not evo2_path.exists():
        logger.info("No EVO-2 results found at %s, skipping comparison", evo2_path)
        return []

    with open(evo2_path) as f:
        evo2_rows = {r["label"]: float(r["evo2_llr"]) for r in csv.DictReader(f)}

    comparison = []
    for r in esm2_results:
        if r["esm2_llr"] is None:
            continue
        evo2_llr = evo2_rows.get(r["label"])
        if evo2_llr is None:
            continue
        meta = metadata.get(r["label"], {})
        prev = meta.get("prevalence_pct")
        comparison.append({
            "label": r["label"],
            "organism_id": r["organism_id"],
            "gene": r["gene"],
            "mutation": r["mutation"],
            "esm2_llr": r["esm2_llr"],
            "abs_esm2_llr": r["abs_esm2_llr"],
            "evo2_llr": round(evo2_llr, 6),
            "abs_evo2_llr": round(abs(evo2_llr), 6),
            "prevalence_pct": float(prev) if prev else None,
        })

    if len(comparison) >= 5:
        esm2_vals = [c["abs_esm2_llr"] for c in comparison]
        evo2_vals = [c["abs_evo2_llr"] for c in comparison]
        rho, p = sp_stats.spearmanr(esm2_vals, evo2_vals)
        logger.info("\n=== ESM-2 vs EVO-2 comparison (N=%d) ===", len(comparison))
        logger.info("  rho(|ESM2|, |EVO2|) = %.3f (p=%.4f)", rho, p)
        logger.info("  If low correlation → complementary information (good for features)")

        # Compare which predicts prevalence better
        valid = [c for c in comparison if c["prevalence_pct"] is not None]
        if len(valid) >= 5:
            esm2_vs_prev = sp_stats.spearmanr(
                [c["abs_esm2_llr"] for c in valid],
                [c["prevalence_pct"] for c in valid],
            )
            evo2_vs_prev = sp_stats.spearmanr(
                [c["abs_evo2_llr"] for c in valid],
                [c["prevalence_pct"] for c in valid],
            )
            logger.info("  ESM-2 vs prevalence: rho=%.3f (p=%.4f)", *esm2_vs_prev)
            logger.info("  EVO-2 vs prevalence: rho=%.3f (p=%.4f)", *evo2_vs_prev)

    return comparison


def load_protein_sequences() -> dict[str, str]:
    """Load protein sequences from GFF + reference genomes.

    Returns dict mapping "organism_gene" -> protein sequence.
    Falls back to placeholder sequences if references not available.
    """
    sequences = {}

    # Try loading from extracted CDS sequences
    cds_path = Path("data/card/protein_sequences.json")
    if cds_path.exists():
        with open(cds_path) as f:
            sequences = json.load(f)
        logger.info("Loaded %d protein sequences from %s", len(sequences), cds_path)
        return sequences

    # Try extracting from GFF + genome
    logger.info("Protein sequences not cached, attempting extraction from references...")
    try:
        from compass.targets.resolver import TargetResolver

        ref_paths = {
            "mtb": ("data/references/H37Rv.fasta", "data/references/H37Rv.gff3"),
            "ecoli": ("data/references/ecoli_K12.fasta", "data/references/ecoli_K12.gff3"),
            "saureus": ("data/references/saureus_NCTC8325.fasta", "data/references/saureus_NCTC8325.gff3"),
            "ngonorrhoeae": ("data/references/ngono_FA1090.fasta", "data/references/ngono_FA1090.gff3"),
        }

        for key, (org_id, gene, locus_tag) in PROTEIN_TARGETS.items():
            fasta, gff = ref_paths.get(org_id, (None, None))
            if fasta and Path(fasta).exists() and gff and Path(gff).exists():
                try:
                    resolver = TargetResolver(
                        genome_fasta=Path(fasta),
                        gff_path=Path(gff),
                    )
                    protein_seq = resolver.get_protein_sequence(gene)
                    if protein_seq:
                        sequences[key] = protein_seq
                except Exception as e:
                    logger.warning("Failed to extract protein for %s: %s", key, e)

        if sequences:
            # Cache for next time
            Path("data/card").mkdir(parents=True, exist_ok=True)
            with open(cds_path, "w") as f:
                json.dump(sequences, f, indent=2)
            logger.info("Cached %d protein sequences to %s", len(sequences), cds_path)

    except ImportError:
        logger.warning("TargetResolver not available, protein sequences must be provided manually")

    return sequences


def main():
    parser = argparse.ArgumentParser(description="ESM-2 LLR on AMR targets")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Skip LLR computation, use cached results")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = {
        "experiment": "esm2_amr_llr",
        "timestamp": datetime.now().isoformat(),
        "model": "esm2_t33_650M_UR50D",
        "n_aa_targets": len(AA_TARGETS),
        "analysis_only": args.analysis_only,
    }
    with open(RESULTS_DIR / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    metadata = load_metadata()

    if args.analysis_only:
        cache_path = RESULTS_DIR / "llr_results.csv"
        if not cache_path.exists():
            raise FileNotFoundError(f"No cached results at {cache_path}")
        with open(cache_path) as f:
            llr_results = list(csv.DictReader(f))
        for r in llr_results:
            if r.get("esm2_llr") and r["esm2_llr"] != "":
                r["esm2_llr"] = float(r["esm2_llr"])
                r["abs_esm2_llr"] = abs(r["esm2_llr"])
            else:
                r["esm2_llr"] = None
                r["abs_esm2_llr"] = None
    else:
        protein_sequences = load_protein_sequences()
        if not protein_sequences:
            logger.error(
                "No protein sequences available. "
                "Ensure reference genomes are downloaded and GFF files are present."
            )
            return

        logger.info("Computing ESM-2 LLR for %d AA substitution targets...", len(AA_TARGETS))
        llr_results = compute_all_targets(protein_sequences)

        # Save raw results
        out_path = RESULTS_DIR / "llr_results.csv"
        valid_results = [r for r in llr_results if r.get("esm2_llr") is not None]
        if valid_results:
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=valid_results[0].keys())
                writer.writeheader()
                writer.writerows(llr_results)
            logger.info("Saved ESM-2 LLR results to %s", out_path)

    # Correlation analysis
    analysis = correlation_analysis(llr_results, metadata)

    if analysis:
        out_path = RESULTS_DIR / "correlation_analysis.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=analysis[0].keys())
            writer.writeheader()
            writer.writerows(analysis)
        logger.info("Saved correlation analysis to %s", out_path)

    # Compare with EVO-2
    comparison = compare_with_evo2(llr_results, metadata)
    if comparison:
        out_path = RESULTS_DIR / "comparison_evo2.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=comparison[0].keys())
            writer.writeheader()
            writer.writerows(comparison)
        logger.info("Saved EVO-2 comparison to %s", out_path)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    computed = [r for r in llr_results if r.get("esm2_llr") is not None]
    logger.info("Computed ESM-2 LLR for %d / %d AA targets", len(computed), len(AA_TARGETS))

    if computed:
        llrs = [r["esm2_llr"] for r in computed]
        logger.info("  LLR range: [%.3f, %.3f]", min(llrs), max(llrs))
        logger.info("  |LLR| mean: %.3f", np.mean([abs(l) for l in llrs]))
        neg = sum(1 for l in llrs if l < 0)
        logger.info("  LLR < 0 (wildtype preferred): %d/%d (%.0f%%)",
                    neg, len(llrs), 100 * neg / len(llrs))


if __name__ == "__main__":
    main()
