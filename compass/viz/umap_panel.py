"""UMAP visualization of candidate embeddings with panel selection highlighted.

Computes 2D projection of all screened candidate embeddings (~34K × 128-dim)
and marks the selected panel members. Falls back to PCA when umap-learn is
not installed.

Output: JSON with x, y coordinates per candidate plus metadata (score, drug,
strategy, selected flag). Spacer sequences stripped from unselected candidates
to keep file size manageable (~2-5MB for 34K points).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def compute_panel_umap(
    embeddings: list[dict],
    panel_members: list[str],
    output_path: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.05,
    metric: str = "cosine",
    random_state: int = 42,
) -> dict:
    """Compute UMAP on all candidate embeddings, mark panel members.

    Args:
        embeddings: list of dicts from scorer.get_collected_embeddings().
            Each dict has: target_label, spacer_seq, embedding (np.array 128),
            score, gc_content, detection_strategy, drug, selected.
        panel_members: target_labels selected for the panel.
        output_path: path for the output JSON.

    Returns:
        dict with 'points', 'n_total', 'n_selected', 'stats'.
    """
    if not embeddings:
        return {"points": [], "n_total": 0, "n_selected": 0, "stats": {}}

    # Stack embeddings into matrix
    emb_matrix = np.array([e["embedding"] for e in embeddings])  # (N, 128)
    n_total = len(emb_matrix)
    panel_set = set(panel_members)

    logger.info(
        "Computing %s on %d candidates (128-dim) ...",
        "UMAP" if HAS_UMAP else "PCA",
        n_total,
    )

    # PCA pre-reduction: strip noise dimensions before UMAP
    if emb_matrix.shape[1] > 30:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=30, random_state=random_state)
        emb_matrix = pca.fit_transform(emb_matrix)
        logger.info("PCA pre-reduction: 128-dim → 30-dim (%.1f%% variance retained)",
                     pca.explained_variance_ratio_.sum() * 100)

    if HAS_UMAP:
        effective_neighbors = min(n_neighbors, n_total - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            low_memory=True,
        )
        coords = reducer.fit_transform(emb_matrix)  # (N, 2)
        method = "UMAP"
        method_params = {
            "n_neighbors": effective_neighbors,
            "min_dist": min_dist,
            "metric": metric,
        }
    else:
        # PCA fallback via SVD
        centered = emb_matrix - emb_matrix.mean(axis=0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        coords = U[:, :2] * S[:2]
        var_total = (S ** 2).sum()
        method = "PCA"
        method_params = {
            "variance_explained": [
                round(float(S[0] ** 2 / var_total), 4),
                round(float(S[1] ** 2 / var_total), 4),
            ],
        }

    # Build output points (strip spacer_seq from unselected for size)
    points = []
    for i, e in enumerate(embeddings):
        # Use pre-set "selected" flag from pipeline (matched by spacer_seq)
        is_selected = e.get("selected", False)
        pt = {
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "target_label": e["target_label"],
            "score": round(float(e["score"]), 4) if e.get("score") is not None else None,
            "gc_content": round(float(e["gc_content"]), 3) if e.get("gc_content") is not None else None,
            "drug": e.get("drug"),
            "detection_strategy": e.get("detection_strategy", "direct"),
            "selected": is_selected,
        }
        if is_selected:
            pt["spacer_seq"] = e.get("spacer_seq", "")
        points.append(pt)

    # Stats: panel spread (avg pairwise distance of selected candidates)
    sel_indices = [i for i, e in enumerate(embeddings) if e.get("selected", False)]
    sel_coords = coords[sel_indices] if sel_indices else np.empty((0, 2))

    panel_spread = 0.0
    if len(sel_coords) > 1:
        from scipy.spatial.distance import pdist
        panel_spread = float(np.mean(pdist(sel_coords)))

    # Coverage: convex hull of selected / convex hull of all
    coverage = None
    try:
        if len(sel_coords) >= 3 and n_total >= 3:
            from scipy.spatial import ConvexHull
            hull_all = ConvexHull(coords)
            hull_sel = ConvexHull(sel_coords)
            coverage = round(hull_sel.volume / hull_all.volume, 4)
    except Exception:
        pass

    result = {
        "points": points,
        "n_total": n_total,
        "n_selected": len(sel_indices),
        "stats": {
            "method": method,
            "panel_spread": round(panel_spread, 4),
            "coverage": coverage,
            **method_params,
        },
    }

    # Save to disk
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "%s complete: %d points, %d selected, %.1fMB JSON",
        method, n_total, len(sel_indices), size_mb,
    )

    return result
