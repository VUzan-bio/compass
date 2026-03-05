"""Simple JSON-file backed ablation result store."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "research" / "ablation_results.json"


def _store_path() -> Path:
    return _DEFAULT_PATH


def load_ablation_rows() -> list[dict[str, Any]]:
    """Load all ablation rows from disk."""
    path = _store_path()
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("rows", []) if isinstance(data, dict) else data


def save_ablation_rows(rows: list[dict[str, Any]]) -> None:
    """Save ablation rows to disk."""
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"rows": rows}, f, indent=2)


def add_ablation_row(row: dict[str, Any]) -> dict[str, Any]:
    """Add a new row and return the updated list."""
    rows = load_ablation_rows()
    row["id"] = max((r.get("id", 0) for r in rows), default=0) + 1
    rows.append(row)
    save_ablation_rows(rows)
    return row
