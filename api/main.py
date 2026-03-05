"""COMPASS Platform — FastAPI application factory.

Single entry point: uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from api.state import AppState

# ── Security configuration from environment ──
API_KEY = os.environ.get("COMPASS_API_KEY", "")
RATE_LIMIT = int(os.environ.get("COMPASS_RATE_LIMIT", "120"))  # requests per minute


class SafeJSONResponse(JSONResponse):
    """JSONResponse that converts inf/nan to null instead of crashing."""

    def _sanitize(self, obj: Any) -> Any:
        if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
            return None
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        return obj

    def render(self, content: Any) -> bytes:
        return json.dumps(
            self._sanitize(content),
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global state — initialized in lifespan
_state: AppState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize AppState on startup, clean up on shutdown."""
    global _state

    _state = AppState(results_dir="results/api")
    logger.info("COMPASS Platform starting — results at %s", _state.results_dir)

    # Wire state into route modules
    from api.routes import figures, pipeline, research, results
    from api import ws

    pipeline.init(_state)
    results.init(_state)
    figures.init(_state)
    research.init(_state)
    ws.init(_state)

    yield

    _state.shutdown()
    logger.info("COMPASS Platform shutdown complete")


app = FastAPI(
    title="COMPASS Platform",
    description="Guide RNA Automated Resistance Diagnostics — CRISPR-Cas12a diagnostic panel design",
    version="0.2.0",
    lifespan=lifespan,
    default_response_class=SafeJSONResponse,
)

# CORS — restrict to known origins
_cors_origins = os.environ.get("CORS_ORIGINS", "").split(",")
_cors_origins = [o.strip() for o in _cors_origins if o.strip()]
if not _cors_origins:
    _cors_origins = [
        "https://compass-design.app",
        "https://www.compass-design.app",
        "https://compass-production.up.railway.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type", "Authorization"],
)


# ── API key authentication middleware ──
_AUTH_SKIP_PATHS = {"/api/health", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def check_api_key(request: Request, call_next: object) -> Response:
    """Require X-API-Key header when COMPASS_API_KEY is set."""
    # Skip auth if no API key configured (local development)
    if not API_KEY:
        return await call_next(request)  # type: ignore[operator]

    # Skip auth for health check, docs, static files, and websocket
    path = request.url.path
    if path in _AUTH_SKIP_PATHS or not path.startswith("/api"):
        return await call_next(request)  # type: ignore[operator]

    provided = request.headers.get("X-API-Key", "")
    if provided != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"},
        )

    return await call_next(request)  # type: ignore[operator]


# ── Rate limiter middleware ──
_rate_store: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit(request: Request, call_next: object) -> Response:
    """Simple in-memory per-IP rate limiter."""
    # Skip rate limiting for static files and websocket
    if not request.url.path.startswith("/api"):
        return await call_next(request)  # type: ignore[operator]

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Prune entries older than 60 seconds
    _rate_store[client_ip] = [
        t for t in _rate_store[client_ip] if now - t < 60
    ]

    if len(_rate_store[client_ip]) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again in a minute."},
        )

    _rate_store[client_ip].append(now)
    return await call_next(request)  # type: ignore[operator]

# Include routers
from api.routes import figures, optimisation, panels, pipeline, research, results, scoring, validation
from api import ws

app.include_router(pipeline.router)
app.include_router(results.router)
app.include_router(panels.router)
app.include_router(figures.router)
app.include_router(scoring.router)
app.include_router(research.router)
app.include_router(validation.router)
app.include_router(optimisation.router)
app.include_router(ws.router)

@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": "0.2.0",
        "pipeline": "COMPASS",
    }


@app.get("/")
async def root():
    return JSONResponse({"status": "ok", "app": "COMPASS Platform", "docs": "/docs"})


# Serve frontend static files if built.
STATIC_DIR = Path("compass-ui/dist")
if STATIC_DIR.exists():
    from starlette.requests import Request
    from starlette.responses import FileResponse, Response

    @app.middleware("http")
    async def spa_middleware(request: Request, call_next: object) -> Response:
        """Serve API routes normally; fall back to static/SPA for everything else."""
        if request.url.path.startswith("/api") or request.url.path.startswith("/ws"):
            return await call_next(request)  # type: ignore[operator]

        # Try to serve the exact static file (with path traversal protection)
        file_path = (STATIC_DIR / request.url.path.lstrip("/")).resolve()
        if file_path.is_relative_to(STATIC_DIR.resolve()) and file_path.is_file():
            return FileResponse(str(file_path))

        # SPA fallback — serve index.html for all other routes
        return FileResponse(str(STATIC_DIR / "index.html"))
