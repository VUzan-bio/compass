"""
Compare PAM coverage across Cas12a variants on the COMPASS panel.

For each variant with a scanner_variant mapping, runs the ACTUAL PAM
scanner against the H37Rv genome around each mutation site.
Results are real computation, not mocked.

If the genome is not available (e.g., Railway deployment), returns
profile summaries without coverage data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from compass.nuclease.profile_loader import NucleaseProfile

logger = logging.getLogger(__name__)


def compare_pam_coverage(
    variant_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run PAM scanning for each variant on the MDR-TB panel.

    Returns per-variant coverage dict with real PAM scan results.
    Gracefully handles missing genome (returns coverage=null).
    """
    if variant_ids is None:
        variant_ids = NucleaseProfile.available()

    results: Dict[str, Any] = {}

    # Try to resolve targets from the panel
    targets = _resolve_panel_targets()
    if targets is None:
        logger.warning("Cannot resolve panel targets — genome not available")
        for vid in variant_ids:
            profile = NucleaseProfile.load(vid)
            results[vid] = {
                "variant_id": vid,
                "display_name": profile.display_name,
                "coverage": None,
                "error": "Genome database not available — PAM scan requires local H37Rv genome",
            }
        return results

    for vid in variant_ids:
        profile = NucleaseProfile.load(vid)
        scanner_key = profile.scanner_variant

        if scanner_key is None:
            # No scanner mapping (e.g., AmCas12a with non-standard PAM)
            results[vid] = {
                "variant_id": vid,
                "display_name": profile.display_name,
                "coverage": None,
                "error": f"No scanner configuration for {profile.display_name} — PAM set not mappable to IUPAC patterns",
            }
            continue

        try:
            coverage = _scan_variant(scanner_key, targets)
            results[vid] = {
                "variant_id": vid,
                "display_name": profile.display_name,
                **coverage,
            }
        except Exception as e:
            logger.error("PAM scan failed for %s: %s", vid, e)
            results[vid] = {
                "variant_id": vid,
                "display_name": profile.display_name,
                "coverage": None,
                "error": str(e),
            }

    return results


def _resolve_panel_targets() -> Optional[List[dict]]:
    """Resolve the MDR-TB panel mutations to Target objects.

    Returns list of {label, drug, target} dicts, or None if genome unavailable.
    """
    try:
        from compass.panels.mdr_tb import define_mdr_panel
        from compass.targets.resolver import TargetResolver

        resolver = TargetResolver()
        mutations = define_mdr_panel()
        targets = []

        for mut in mutations:
            try:
                target = resolver.resolve(mut)
                targets.append({
                    "label": f"{mut.gene}_{mut.ref_aa}{mut.position}{mut.alt_aa}",
                    "drug": mut.drug.value if hasattr(mut.drug, 'value') else str(mut.drug),
                    "target": target,
                })
            except Exception as e:
                logger.warning("Failed to resolve %s_%s%d%s: %s",
                               mut.gene, mut.ref_aa, mut.position, mut.alt_aa, e)

        return targets if targets else None

    except Exception as e:
        logger.warning("Target resolution failed: %s", e)
        return None


def _scan_variant(scanner_key: str, targets: List[dict]) -> dict:
    """Run PAM scanner with a specific variant config on all targets."""
    from compass.candidates.scanner import PAMScanner, CONFIGS

    if scanner_key not in CONFIGS:
        raise ValueError(f"Unknown scanner variant: {scanner_key}")

    scanner = PAMScanner(cas_variant=scanner_key)

    per_target = []
    targets_with_pam = 0
    total_direct = 0
    total_proximity = 0

    for t in targets:
        target_obj = t["target"]
        try:
            result = scanner.scan_detailed(target_obj)
            n_direct = len(result.direct_candidates)
            n_proximity = len(result.proximity_candidates)
            n_total = n_direct + n_proximity
            total_direct += n_direct
            total_proximity += n_proximity

            if n_total > 0:
                targets_with_pam += 1

            # Best PAM from direct candidates
            best_pam = None
            if result.direct_candidates:
                best_pam = result.direct_candidates[0].pam_seq

            per_target.append({
                "target": t["label"],
                "drug": t["drug"],
                "n_direct": n_direct,
                "n_proximity": n_proximity,
                "n_total": n_total,
                "best_pam": best_pam,
                "has_direct": n_direct > 0,
                "pam_desert": result.pam_desert,
            })
        except Exception as e:
            logger.warning("Scan failed for %s: %s", t["label"], e)
            per_target.append({
                "target": t["label"],
                "drug": t["drug"],
                "n_direct": 0,
                "n_proximity": 0,
                "n_total": 0,
                "best_pam": None,
                "has_direct": False,
                "pam_desert": True,
                "error": str(e),
            })

    pam_desert_targets = [t["target"] for t in per_target if t["pam_desert"]]

    return {
        "targets_with_pam": targets_with_pam,
        "targets_total": len(targets),
        "total_direct_candidates": total_direct,
        "total_proximity_candidates": total_proximity,
        "pam_desert_targets": pam_desert_targets,
        "avg_direct_per_target": round(total_direct / max(len(targets), 1), 1),
        "per_target": per_target,
    }
