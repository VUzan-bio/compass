"""Saved panel CRUD endpoints."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/panels", tags=["panels"])

PANELS_DIR = Path("results/panels")


class PanelCreate(BaseModel):
    name: str
    description: str = ""
    mutations: list[dict] = Field(default_factory=list)


class PanelResponse(BaseModel):
    panel_id: str
    name: str
    description: str
    mutations: list[dict]
    created_at: str


def _panels_path() -> Path:
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    return PANELS_DIR / "panels.json"


def _load_panels() -> list[dict[str, Any]]:
    path = _panels_path()
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save_panels(panels: list[dict[str, Any]]) -> None:
    with open(_panels_path(), "w") as f:
        json.dump(panels, f, indent=2)


@router.get("", response_model=list[PanelResponse])
async def list_panels() -> list[PanelResponse]:
    panels = _load_panels()
    return [PanelResponse(**p) for p in panels]


@router.post("", response_model=PanelResponse, status_code=201)
async def create_panel(req: PanelCreate) -> PanelResponse:
    panels = _load_panels()
    panel = {
        "panel_id": uuid.uuid4().hex[:12],
        "name": req.name,
        "description": req.description,
        "mutations": req.mutations,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    panels.append(panel)
    _save_panels(panels)
    return PanelResponse(**panel)


@router.get("/{panel_id}", response_model=PanelResponse)
async def get_panel(panel_id: str) -> PanelResponse:
    panels = _load_panels()
    for p in panels:
        if p["panel_id"] == panel_id:
            return PanelResponse(**p)
    raise HTTPException(404, f"Panel {panel_id} not found")


@router.delete("/{panel_id}", status_code=204, response_model=None)
async def delete_panel(panel_id: str) -> None:
    panels = _load_panels()
    filtered = [p for p in panels if p["panel_id"] != panel_id]
    if len(filtered) == len(panels):
        raise HTTPException(404, f"Panel {panel_id} not found")
    _save_panels(filtered)
