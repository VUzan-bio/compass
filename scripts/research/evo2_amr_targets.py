"""EVO-2 log-likelihood ratio experiment on 42 AMR targets.

Weeks 3-4 experiment: correlate delta_LLR with clinical prevalence and
predicted discrimination. Novel result — nobody has used EVO-2 for AMR
target prioritization.

Usage:
    # Full run (requires GPU with EVO-2 7B):
    python scripts/research/evo2_amr_targets.py

    # Analysis only (uses cached LLR values):
    python scripts/research/evo2_amr_targets.py --analysis-only

Output:
    results/research/evo2_llr/
        llr_results.csv         — per-target LLR values
        correlation_analysis.csv — Spearman rho per organism and pooled
        figures/                 — publication-ready figures
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

RESULTS_DIR = Path("results/research/evo2_llr")
CONTEXTS_PATH = Path("data/card/target_contexts.json")
METADATA_PATH = Path("data/card/amr_target_metadata.csv")


def load_contexts() -> list[dict]:
    """Load pre-extracted target contexts from Week 0 setup."""
    if not CONTEXTS_PATH.exists():
        raise FileNotFoundError(
            f"Target contexts not found at {CONTEXTS_PATH}. "
            "Run: python scripts/research/extract_target_contexts.py"
        )
    with open(CONTEXTS_PATH) as f:
        contexts = json.load(f)
    resolved = [c for c in contexts if c.get("resolved")]
    logger.info("Loaded %d resolved target contexts", len(resolved))
    return resolved


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


def compute_llr_all(contexts: list[dict], window: int = 2) -> list[dict]:
    """Compute EVO-2 LLR for all targets."""
    from compass_net.features.evo2_llr import compute_evo2_llr_batch

    targets = [
        {
            "mutant_context": c["mutant_context"],
            "wildtype_context": c["wildtype_context"],
            "mutation_pos": c["mutation_pos"],
        }
        for c in contexts
    ]

    logger.info("Computing EVO-2 LLR for %d targets (window=%d)...", len(targets), window)
    llr_values = compute_evo2_llr_batch(targets, window=window)

    results = []
    for ctx, llr in zip(contexts, llr_values):
        results.append({
            "label": ctx["label"],
            "organism_id": ctx["organism_id"],
            "gene": ctx["gene"],
            "mutation": ctx["mutation"],
            "genomic_pos": ctx.get("genomic_pos"),
            "gc_content": ctx.get("gc_content"),
            "evo2_llr": llr,
            "abs_llr": abs(llr),
        })
        logger.info("  %s: LLR=%.4f (|LLR|=%.4f, GC=%.1f%%)",
                     ctx["label"], llr, abs(llr),
                     ctx.get("gc_content", 0) * 100)

    return results


def compute_llr_windows(contexts: list[dict]) -> dict[int, list[dict]]:
    """Compute LLR at multiple window sizes for sensitivity analysis."""
    results_by_window = {}
    for w in [1, 2, 5, 10]:
        logger.info("--- Window size: %d ---", w)
        results_by_window[w] = compute_llr_all(contexts, window=w)
    return results_by_window


def correlation_analysis(
    llr_results: list[dict],
    metadata: dict[str, dict],
) -> list[dict]:
    """Compute Spearman correlations between |LLR| and clinical variables.

    Reports:
    - Per-organism: rho(|LLR|, prevalence), rho(|LLR|, WHO tier)
    - Pooled across organisms
    - Partial correlation controlling for GC content
    - Permutation p-values (10,000 permutations)
    """
    from scipy import stats

    analysis_rows = []

    # Merge LLR with metadata
    merged = []
    for r in llr_results:
        meta = metadata.get(r["label"], {})
        prev = meta.get("prevalence_pct")
        tier = meta.get("who_confidence_tier")
        merged.append({
            **r,
            "prevalence_pct": float(prev) if prev else None,
            "who_tier": int(tier) if tier else None,
        })

    # --- Per-organism analysis ---
    by_org: dict[str, list] = {}
    for m in merged:
        by_org.setdefault(m["organism_id"], []).append(m)

    for org_id, targets in sorted(by_org.items()):
        abs_llr = [t["abs_llr"] for t in targets]
        prev = [t["prevalence_pct"] for t in targets if t["prevalence_pct"] is not None]
        gc = [t["gc_content"] for t in targets if t["gc_content"] is not None]

        n = len(targets)
        logger.info("\n=== %s (N=%d) ===", org_id, n)

        # Correlation with prevalence
        valid_prev = [(t["abs_llr"], t["prevalence_pct"]) for t in targets
                      if t["prevalence_pct"] is not None]
        if len(valid_prev) >= 5:
            x, y = zip(*valid_prev)
            rho, p = stats.spearmanr(x, y)
            # Permutation test
            perm_p = _permutation_test(x, y, n_perms=10000)
            logger.info("  rho(|LLR|, prevalence) = %.3f (parametric p=%.4f, perm p=%.4f)",
                        rho, p, perm_p)
            analysis_rows.append({
                "organism": org_id,
                "variable": "prevalence",
                "n": len(valid_prev),
                "spearman_rho": round(rho, 4),
                "p_parametric": round(p, 4),
                "p_permutation": round(perm_p, 4),
            })

            # Partial correlation controlling for GC
            valid_gc = [(t["abs_llr"], t["prevalence_pct"], t["gc_content"])
                        for t in targets
                        if t["prevalence_pct"] is not None and t["gc_content"] is not None]
            if len(valid_gc) >= 5:
                x, y, z = zip(*valid_gc)
                partial_rho = _partial_spearman(x, y, z)
                logger.info("  partial rho(|LLR|, prevalence | GC) = %.3f", partial_rho)
                analysis_rows.append({
                    "organism": org_id,
                    "variable": "prevalence_gc_partial",
                    "n": len(valid_gc),
                    "spearman_rho": round(partial_rho, 4),
                    "p_parametric": None,
                    "p_permutation": None,
                })
        else:
            logger.info("  Too few targets with prevalence data (N=%d, need >=5)", len(valid_prev))

        # Correlation with WHO tier
        valid_tier = [(t["abs_llr"], t["who_tier"]) for t in targets
                      if t["who_tier"] is not None]
        if len(valid_tier) >= 5:
            x, y = zip(*valid_tier)
            rho, p = stats.spearmanr(x, y)
            logger.info("  rho(|LLR|, WHO_tier) = %.3f (p=%.4f)", rho, p)
            analysis_rows.append({
                "organism": org_id,
                "variable": "who_tier",
                "n": len(valid_tier),
                "spearman_rho": round(rho, 4),
                "p_parametric": round(p, 4),
                "p_permutation": None,
            })

    # --- Pooled analysis ---
    all_valid = [(m["abs_llr"], m["prevalence_pct"], m["gc_content"], m["organism_id"])
                 for m in merged
                 if m["prevalence_pct"] is not None and m["gc_content"] is not None]
    if len(all_valid) >= 10:
        x = [v[0] for v in all_valid]
        y = [v[1] for v in all_valid]
        gc = [v[2] for v in all_valid]

        rho, p = stats.spearmanr(x, y)
        perm_p = _permutation_test(x, y, n_perms=10000)
        partial_rho = _partial_spearman(x, y, gc)

        logger.info("\n=== POOLED (N=%d) ===", len(all_valid))
        logger.info("  rho(|LLR|, prevalence) = %.3f (perm p=%.4f)", rho, perm_p)
        logger.info("  partial rho (| GC) = %.3f", partial_rho)

        analysis_rows.append({
            "organism": "pooled",
            "variable": "prevalence",
            "n": len(all_valid),
            "spearman_rho": round(rho, 4),
            "p_parametric": round(p, 4),
            "p_permutation": round(perm_p, 4),
        })
        analysis_rows.append({
            "organism": "pooled",
            "variable": "prevalence_gc_partial",
            "n": len(all_valid),
            "spearman_rho": round(partial_rho, 4),
            "p_parametric": None,
            "p_permutation": None,
        })

    # BH correction across organism-level tests
    org_pvals = [r["p_permutation"] for r in analysis_rows
                 if r["variable"] == "prevalence" and r["organism"] != "pooled"
                 and r["p_permutation"] is not None]
    if org_pvals:
        bh_corrected = _bh_correction(org_pvals)
        logger.info("\nBH-corrected p-values: %s", [round(p, 4) for p in bh_corrected])

    return analysis_rows


def _permutation_test(x, y, n_perms: int = 10000) -> float:
    """Permutation test for Spearman rho."""
    from scipy import stats
    rng = np.random.default_rng(42)
    observed = stats.spearmanr(x, y).statistic
    x_arr = np.array(x)
    y_arr = np.array(y)
    count = 0
    for _ in range(n_perms):
        perm_y = rng.permutation(y_arr)
        perm_rho = stats.spearmanr(x_arr, perm_y).statistic
        if abs(perm_rho) >= abs(observed):
            count += 1
    return count / n_perms


def _partial_spearman(x, y, z) -> float:
    """Partial Spearman correlation between x and y controlling for z.

    Uses rank-based partial correlation: partial out the effect of z
    on both x and y rankings, then correlate residuals.
    """
    from scipy import stats
    x_rank = stats.rankdata(x)
    y_rank = stats.rankdata(y)
    z_rank = stats.rankdata(z)
    # Residualize x and y ranks on z ranks
    x_res = x_rank - np.polyval(np.polyfit(z_rank, x_rank, 1), z_rank)
    y_res = y_rank - np.polyval(np.polyfit(z_rank, y_rank, 1), z_rank)
    return float(np.corrcoef(x_res, y_res)[0, 1])


def _bh_correction(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_p = np.array(pvals)[sorted_idx]
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[sorted_idx[i]] = sorted_p[i]
        else:
            adjusted[sorted_idx[i]] = min(
                sorted_p[i] * n / (i + 1),
                adjusted[sorted_idx[i + 1]]
            )
    return adjusted.tolist()


def main():
    parser = argparse.ArgumentParser(description="EVO-2 LLR on AMR targets")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Skip LLR computation, use cached results")
    parser.add_argument("--window", type=int, default=2,
                        help="LLR window size (default: 2)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = {
        "experiment": "evo2_amr_llr",
        "timestamp": datetime.now().isoformat(),
        "window": args.window,
        "n_permutations": 10000,
        "analysis_only": args.analysis_only,
    }
    with open(RESULTS_DIR / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    metadata = load_metadata()

    if args.analysis_only:
        # Load cached LLR results
        cache_path = RESULTS_DIR / "llr_results.csv"
        if not cache_path.exists():
            raise FileNotFoundError(f"No cached results at {cache_path}")
        with open(cache_path) as f:
            llr_results = list(csv.DictReader(f))
        for r in llr_results:
            r["abs_llr"] = abs(float(r["evo2_llr"]))
            r["evo2_llr"] = float(r["evo2_llr"])
            r["gc_content"] = float(r["gc_content"]) if r.get("gc_content") else None
    else:
        contexts = load_contexts()
        llr_results = compute_llr_all(contexts, window=args.window)

        # Save raw results
        out_path = RESULTS_DIR / "llr_results.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=llr_results[0].keys())
            writer.writeheader()
            writer.writerows(llr_results)
        logger.info("Saved LLR results to %s", out_path)

    # Run correlation analysis
    analysis = correlation_analysis(llr_results, metadata)

    # Save analysis
    if analysis:
        out_path = RESULTS_DIR / "correlation_analysis.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=analysis[0].keys())
            writer.writeheader()
            writer.writerows(analysis)
        logger.info("Saved correlation analysis to %s", out_path)

    # Decision gate summary
    logger.info("\n" + "=" * 60)
    logger.info("DECISION GATE SUMMARY")
    logger.info("=" * 60)
    significant = [r for r in analysis
                   if r["variable"] == "prevalence"
                   and r["organism"] != "pooled"
                   and r["p_permutation"] is not None
                   and r["p_permutation"] < 0.05
                   and abs(r["spearman_rho"]) > 0.5]
    if len(significant) >= 2:
        logger.info("POSITIVE: |rho| > 0.5 with p < 0.05 in %d organisms", len(significant))
        logger.info("  → Add EVO-2 LLR as scalar feature to Compass-ML")
        for r in significant:
            logger.info("    %s: rho=%.3f, p=%.4f", r["organism"], r["spearman_rho"], r["p_permutation"])
    else:
        logger.info("NEGATIVE: Insufficient significant correlations (%d/4)", len(significant))
        logger.info("  → EVO-2 LLR does not predict AMR target quality")
        logger.info("  → Report as negative result, try window sensitivity analysis")


if __name__ == "__main__":
    main()
