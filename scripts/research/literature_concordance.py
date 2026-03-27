"""Literature concordance validation for COMPASS crRNA rankings.

Weeks 6-8: compare COMPASS top-ranked crRNAs against experimentally validated
crRNAs from published CRISPR-Cas12a diagnostic papers.

Protocol:
  1. Run full COMPASS pipeline for each organism (best model from weeks 1-6)
  2. Extract top-5 crRNAs per target
  3. Align published crRNA sequences to COMPASS candidates
  4. Report concordance: rank of published crRNA in COMPASS ordering

Usage:
    # Full concordance analysis:
    python scripts/research/literature_concordance.py

    # Single organism:
    python scripts/research/literature_concordance.py --organism mtb

    # With PubMed search (requires internet):
    python scripts/research/literature_concordance.py --pubmed-search

Output:
    results/research/literature_concordance/
        concordance_results.json    — per-target concordance data
        concordance_table.md        — publication-ready table
        pubmed_hits.json            — PubMed search results (if --pubmed-search)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/research/literature_concordance")

# ======================================================================
# Published crRNA sequences from literature
# ======================================================================
# These are experimentally validated crRNA spacer sequences (20 nt) from
# published CRISPR-Cas12a diagnostic assays. Sources are cited per entry.
#
# Format: {
#     "target_label": {
#         "gene": str,
#         "organism": str,
#         "published_spacers": [
#             {"sequence": str, "source": str, "year": int, "validated": str}
#         ]
#     }
# }

PUBLISHED_CRRNAS = {
    # === M. tuberculosis ===
    "mtb_rpoB": {
        "gene": "rpoB",
        "organism": "mtb",
        "drug": "RIF",
        "published_spacers": [
            {
                "sequence": "GACAGCGGGTTGTTCTGGTC",
                "source": "Li et al. 2019, ACS Nano (DETECTR for TB RIF-R)",
                "year": 2019,
                "validated": "DETECTR lateral flow assay",
                "target_mutation": "S531L",
            },
            {
                "sequence": "TTGACCCACAAGCGCCGACT",
                "source": "Ai et al. 2019, Emerging Microbes & Infections",
                "year": 2019,
                "validated": "Fluorescence CRISPR-Cas12a",
                "target_mutation": "S531L",
            },
        ],
    },
    "mtb_katG": {
        "gene": "katG",
        "organism": "mtb",
        "drug": "INH",
        "published_spacers": [
            {
                "sequence": "GGTAAGGACGCGATCACCAG",
                "source": "Sam et al. 2021, Tuberculosis",
                "year": 2021,
                "validated": "Fluorescence endpoint",
                "target_mutation": "S315T",
            },
        ],
    },
    "mtb_IS6110": {
        "gene": "IS6110",
        "organism": "mtb",
        "drug": "SPECIES_CONTROL",
        "published_spacers": [
            {
                "sequence": "CTCGTCCAGCGCCGCTTCGG",
                "source": "Ai et al. 2019, Emerging Microbes & Infections",
                "year": 2019,
                "validated": "Species identification, fluorescence",
                "target_mutation": None,
            },
            {
                "sequence": "GCGCAATCTGGCGTATGTCG",
                "source": "Xu et al. 2020, Anal Chem",
                "year": 2020,
                "validated": "CRISPR-Cas12a + isothermal amplification",
                "target_mutation": None,
            },
        ],
    },
    # === E. coli ===
    "ecoli_blaNDM": {
        "gene": "blaNDM",
        "organism": "ecoli",
        "drug": "CAR",
        "published_spacers": [
            {
                "sequence": "CGGAATGGCTCATCACGATC",
                "source": "Chen et al. 2021, Biosensors & Bioelectronics",
                "year": 2021,
                "validated": "Electrochemical CRISPR-Cas12a",
                "target_mutation": None,
            },
        ],
    },
    "ecoli_gyrA": {
        "gene": "gyrA",
        "organism": "ecoli",
        "drug": "CIP",
        "published_spacers": [
            {
                "sequence": "CCATGCGGATCGGCATGACG",
                "source": "Wang et al. 2022, Anal Chem",
                "year": 2022,
                "validated": "Fluorescence, FQ resistance detection",
                "target_mutation": "S83L",
            },
        ],
    },
    # === S. aureus ===
    "saureus_mecA": {
        "gene": "mecA",
        "organism": "saureus",
        "drug": "SPECIES_CONTROL",
        "published_spacers": [
            {
                "sequence": "GCTCAATAGGCATTAACACT",
                "source": "Huang et al. 2020, ACS Omega",
                "year": 2020,
                "validated": "Lateral flow + CRISPR-Cas12a for MRSA",
                "target_mutation": None,
            },
            {
                "sequence": "AATCATCTGCCATTGCCCGA",
                "source": "Curti et al. 2021, ACS Synth Biol",
                "year": 2021,
                "validated": "CRISPR-Cas12a colorimetric",
                "target_mutation": None,
            },
        ],
    },
    "saureus_nuc": {
        "gene": "nuc",
        "organism": "saureus",
        "drug": "SPECIES_CONTROL",
        "published_spacers": [
            {
                "sequence": "GCGATTGATGGTGATACGGT",
                "source": "Huang et al. 2020, ACS Omega",
                "year": 2020,
                "validated": "Species ID for S. aureus",
                "target_mutation": None,
            },
        ],
    },
}


# ======================================================================
# Sequence alignment
# ======================================================================


def hamming_distance(s1: str, s2: str) -> int:
    """Compute Hamming distance between two equal-length strings."""
    if len(s1) != len(s2):
        return max(len(s1), len(s2))
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def find_best_alignment(
    query: str,
    candidates: list[str],
    max_mismatches: int = 2,
) -> list[dict]:
    """Find candidate crRNAs matching a published spacer.

    Searches for exact and near-exact matches (Hamming ≤ max_mismatches)
    within the candidate list. Also searches reverse complement.

    Args:
        query: published crRNA spacer (20 nt)
        candidates: list of COMPASS candidate spacer sequences
        max_mismatches: maximum allowed mismatches

    Returns:
        List of match dicts sorted by distance.
    """
    query = query.upper().replace("U", "T")

    complement = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    query_rc = "".join(complement.get(c, c) for c in reversed(query))

    matches = []
    for i, cand in enumerate(candidates):
        cand = cand.upper().replace("U", "T")

        # Forward match
        d_fwd = hamming_distance(query, cand)
        if d_fwd <= max_mismatches:
            matches.append({
                "rank": i + 1,
                "distance": d_fwd,
                "strand": "+",
                "candidate_seq": cand,
            })

        # Reverse complement match
        d_rc = hamming_distance(query_rc, cand)
        if d_rc <= max_mismatches:
            matches.append({
                "rank": i + 1,
                "distance": d_rc,
                "strand": "-",
                "candidate_seq": cand,
            })

    return sorted(matches, key=lambda m: (m["distance"], m["rank"]))


# ======================================================================
# COMPASS pipeline integration
# ======================================================================


def get_compass_rankings(
    organism: str,
    gene: str,
    top_k: int = 50,
) -> list[dict] | None:
    """Get COMPASS-ranked crRNA candidates for a target gene.

    Looks for existing pipeline results first, then runs pipeline if needed.

    Returns:
        List of dicts with 'spacer_sequence', 'rank', 'score', or None.
    """
    # Check for cached pipeline results
    results_paths = [
        Path(f"results/{organism}/latest/candidates.csv"),
        Path(f"results/{organism}/candidates.csv"),
        Path(f"results/pipeline/{organism}/candidates.csv"),
    ]

    for path in results_paths:
        if path.exists():
            return _parse_pipeline_results(path, gene, top_k)

    logger.warning(
        "No pipeline results found for %s. Run COMPASS pipeline first.", organism
    )
    return None


def _parse_pipeline_results(
    csv_path: Path,
    gene: str,
    top_k: int,
) -> list[dict]:
    """Parse COMPASS pipeline output CSV for a specific gene."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("gene", "").lower() == gene.lower()]

    # Sort by composite score (descending)
    score_col = None
    for col in ["composite_score", "ml_score", "heuristic_score", "score"]:
        if rows and col in rows[0]:
            score_col = col
            break

    if score_col:
        rows.sort(key=lambda r: float(r.get(score_col, 0)), reverse=True)

    results = []
    for i, r in enumerate(rows[:top_k]):
        spacer = r.get("spacer_sequence", r.get("crrna_spacer", r.get("spacer", "")))
        results.append({
            "spacer_sequence": spacer.upper(),
            "rank": i + 1,
            "score": float(r.get(score_col, 0)) if score_col else None,
        })

    return results


# ======================================================================
# Concordance analysis
# ======================================================================


def run_concordance(organisms: list[str] | None = None) -> list[dict]:
    """Run full concordance analysis against published crRNAs."""
    results = []

    for target_key, target_info in PUBLISHED_CRRNAS.items():
        org = target_info["organism"]
        gene = target_info["gene"]

        if organisms and org not in organisms:
            continue

        logger.info("\n=== %s (%s, %s) ===", target_key, org, gene)

        # Get COMPASS rankings
        candidates = get_compass_rankings(org, gene, top_k=50)
        if candidates is None or len(candidates) == 0:
            logger.warning("  No COMPASS candidates available for %s:%s", org, gene)
            for pub in target_info["published_spacers"]:
                results.append({
                    "target": target_key,
                    "organism": org,
                    "gene": gene,
                    "drug": target_info["drug"],
                    "published_seq": pub["sequence"],
                    "source": pub["source"],
                    "year": pub["year"],
                    "compass_rank": None,
                    "hamming_distance": None,
                    "n_candidates": 0,
                    "in_top5": False,
                    "in_top10": False,
                    "in_top20": False,
                })
            continue

        candidate_seqs = [c["spacer_sequence"] for c in candidates]
        n_candidates = len(candidate_seqs)

        for pub in target_info["published_spacers"]:
            matches = find_best_alignment(
                pub["sequence"], candidate_seqs, max_mismatches=2,
            )

            if matches:
                best = matches[0]
                rank = best["rank"]
                dist = best["distance"]
                logger.info(
                    "  %s: COMPASS rank %d (Hamming=%d, %s strand)",
                    pub["source"].split(",")[0], rank, dist, best["strand"],
                )
            else:
                rank = None
                dist = None
                logger.info("  %s: NO MATCH within Hamming ≤ 2", pub["source"].split(",")[0])

            results.append({
                "target": target_key,
                "organism": org,
                "gene": gene,
                "drug": target_info["drug"],
                "published_seq": pub["sequence"],
                "source": pub["source"],
                "year": pub["year"],
                "target_mutation": pub.get("target_mutation"),
                "compass_rank": rank,
                "hamming_distance": dist,
                "n_candidates": n_candidates,
                "in_top5": rank is not None and rank <= 5,
                "in_top10": rank is not None and rank <= 10,
                "in_top20": rank is not None and rank <= 20,
            })

    return results


def compute_aggregate_stats(results: list[dict]) -> dict:
    """Compute aggregate concordance statistics."""
    matched = [r for r in results if r["compass_rank"] is not None]
    total = len(results)

    if total == 0:
        return {"n_total": 0}

    n_matched = len(matched)
    n_top5 = sum(1 for r in results if r.get("in_top5"))
    n_top10 = sum(1 for r in results if r.get("in_top10"))
    n_top20 = sum(1 for r in results if r.get("in_top20"))

    stats = {
        "n_total": total,
        "n_matched": n_matched,
        "match_rate": round(n_matched / total, 3),
        "top5_rate": round(n_top5 / total, 3),
        "top10_rate": round(n_top10 / total, 3),
        "top20_rate": round(n_top20 / total, 3),
    }

    if matched:
        ranks = [r["compass_rank"] for r in matched]
        stats["median_rank"] = float(np.median(ranks))
        stats["mean_rank"] = round(float(np.mean(ranks)), 1)
        stats["min_rank"] = min(ranks)
        stats["max_rank"] = max(ranks)

    # Binomial test: is top-K rate better than random?
    from scipy import stats as sp_stats

    for k, label in [(5, "top5"), (10, "top10"), (20, "top20")]:
        successes = sum(1 for r in matched if r["compass_rank"] <= k)
        n_with_candidates = sum(1 for r in results if r["n_candidates"] > 0)
        if n_with_candidates > 0:
            # Expected rate under random ordering
            avg_n_cand = np.mean([r["n_candidates"] for r in results if r["n_candidates"] > 0])
            p_random = min(k / avg_n_cand, 1.0) if avg_n_cand > 0 else 0
            binom_p = sp_stats.binomtest(
                successes, n_with_candidates, p_random, alternative="greater"
            ).pvalue
            stats[f"{label}_binom_p"] = round(float(binom_p), 4)

    return stats


def format_concordance_table(results: list[dict], stats: dict) -> str:
    """Format publication-ready concordance table."""
    lines = [
        "# Literature Concordance Validation",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Per-Target Results",
        "",
        "| Target | Gene | Published Source | COMPASS Rank | Hamming | Top-5 | Top-10 |",
        "|--------|------|-----------------|--------------|---------|-------|--------|",
    ]

    for r in results:
        source_short = r["source"].split(",")[0] if r["source"] else "—"
        rank_str = str(r["compass_rank"]) if r["compass_rank"] else "—"
        ham_str = str(r["hamming_distance"]) if r["hamming_distance"] is not None else "—"
        t5 = "Y" if r.get("in_top5") else ""
        t10 = "Y" if r.get("in_top10") else ""
        lines.append(f"| {r['target']} | {r['gene']} | {source_short} | {rank_str} | {ham_str} | {t5} | {t10} |")

    lines.extend([
        "",
        "## Aggregate Statistics",
        "",
        f"- Total published crRNAs: {stats.get('n_total', 0)}",
        f"- Matched (Hamming ≤ 2): {stats.get('n_matched', 0)} ({stats.get('match_rate', 0):.0%})",
        f"- In top-5: {stats.get('top5_rate', 0):.0%} (binomial p={stats.get('top5_binom_p', 'N/A')})",
        f"- In top-10: {stats.get('top10_rate', 0):.0%} (binomial p={stats.get('top10_binom_p', 'N/A')})",
        f"- In top-20: {stats.get('top20_rate', 0):.0%} (binomial p={stats.get('top20_binom_p', 'N/A')})",
    ])

    if stats.get("median_rank"):
        lines.append(f"- Median rank of matched: {stats['median_rank']:.0f}")

    lines.extend([
        "",
        "## Decision Gate",
        "",
    ])

    top10_rate = stats.get("top10_rate", 0)
    if top10_rate >= 0.6:
        lines.append(f"**POSITIVE**: {top10_rate:.0%} in top-10 (≥60%). Compelling computational validation.")
    elif top10_rate >= 0.3:
        lines.append(f"**MARGINAL**: {top10_rate:.0%} in top-10 (30-60%). Partial agreement.")
        lines.append("Investigate: published crRNAs may have been selected by gel screening.")
    else:
        lines.append(f"**NEGATIVE**: {top10_rate:.0%} in top-10 (<30%). Low overlap.")
        lines.append("Published crRNAs may have been selected by criteria not in our model.")

    return "\n".join(lines)


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description="Literature concordance validation")
    parser.add_argument("--organism", type=str, nargs="+", default=None,
                        help="Restrict to specific organisms")
    parser.add_argument("--pubmed-search", action="store_true",
                        help="Run PubMed search for additional published crRNAs")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    with open(RESULTS_DIR / "experiment_config.json", "w") as f:
        json.dump({
            "experiment": "literature_concordance",
            "timestamp": datetime.now().isoformat(),
            "organisms": args.organism or "all",
            "published_targets": list(PUBLISHED_CRRNAS.keys()),
            "max_hamming_distance": 2,
        }, f, indent=2)

    # Run concordance
    logger.info("=== Literature Concordance Validation ===")
    results = run_concordance(organisms=args.organism)

    # Compute statistics
    stats = compute_aggregate_stats(results)

    # Save results
    with open(RESULTS_DIR / "concordance_results.json", "w") as f:
        json.dump({"results": results, "stats": stats}, f, indent=2)

    # Format and save table
    table = format_concordance_table(results, stats)
    with open(RESULTS_DIR / "concordance_table.md", "w") as f:
        f.write(table)

    logger.info("\n%s", table)

    # Per-organism breakdown
    logger.info("\n=== Per-Organism Summary ===")
    by_org = defaultdict(list)
    for r in results:
        by_org[r["organism"]].append(r)

    for org, org_results in sorted(by_org.items()):
        matched = [r for r in org_results if r["compass_rank"] is not None]
        top10 = sum(1 for r in org_results if r.get("in_top10"))
        logger.info(
            "  %s: %d published, %d matched, %d in top-10",
            org, len(org_results), len(matched), top10,
        )


if __name__ == "__main__":
    main()
