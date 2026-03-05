"""Integration test — full backend API flow.

Tests the submit → poll → results → export → figures cycle.
Run with: pytest tests/integration/test_api.py -v
"""

from __future__ import annotations

import asyncio
import time

import pytest
import httpx

BASE_URL = "http://localhost:8000"

TEST_MUTATIONS = [
    {"gene": "rpoB", "ref_aa": "S", "position": 450, "alt_aa": "L", "drug": "RIF"},
    {"gene": "katG", "ref_aa": "S", "position": 315, "alt_aa": "T", "drug": "INH"},
    {"gene": "gyrA", "ref_aa": "D", "position": 94, "alt_aa": "G", "drug": "FQ"},
]


@pytest.fixture(scope="module")
def client():
    """Synchronous client for testing against running server."""
    with httpx.Client(base_url=BASE_URL, timeout=180.0) as c:
        yield c


def test_health(client: httpx.Client) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.2.0"


def test_scoring_models(client: httpx.Client) -> None:
    resp = client.get("/api/scoring/models")
    assert resp.status_code == 200
    models = resp.json()
    names = [m["name"] for m in models]
    assert "heuristic" in names
    assert "jepa" in names


def test_full_pipeline_flow(client: httpx.Client) -> None:
    """End-to-end: submit → poll → results → export → figures."""

    # 1. Submit pipeline run
    resp = client.post("/api/pipeline/run", json={
        "name": "Integration Test",
        "mode": "full",
        "mutations": TEST_MUTATIONS,
    })
    assert resp.status_code == 202
    job = resp.json()
    job_id = job["job_id"]
    assert job["status"] == "pending" or job["status"] == "running"
    assert job["n_mutations"] == 3

    # 2. Poll until completed (timeout 120s)
    start = time.time()
    while time.time() - start < 120:
        resp = client.get(f"/api/pipeline/jobs/{job_id}")
        assert resp.status_code == 200
        status = resp.json()
        if status["status"] == "completed":
            break
        if status["status"] == "failed":
            pytest.fail(f"Pipeline failed: {status.get('error', 'unknown')}")
        time.sleep(2)
    else:
        pytest.fail("Pipeline timed out after 120s")

    assert status["progress"] == 1.0

    # 3. Get results
    resp = client.get(f"/api/results/{job_id}")
    assert resp.status_code == 200
    results = resp.json()
    assert results["panel"]["plex"] >= 3
    assert len(results["targets"]) >= 3

    for target in results["targets"]:
        assert target["gene"]
        if target["selected_candidate"]:
            assert target["selected_candidate"]["spacer_seq"]
            assert target["selected_candidate"]["composite_score"] > 0

    # 4. Export TSV
    resp = client.get(f"/api/results/{job_id}/export?format=tsv")
    assert resp.status_code == 200
    lines = resp.text.strip().split("\n")
    assert len(lines) >= 4  # header + 3 data rows

    # 5. Export FASTA
    resp = client.get(f"/api/results/{job_id}/export?format=fasta")
    assert resp.status_code == 200
    assert resp.text.startswith(">")

    # 6. Figures
    resp = client.get(f"/api/figures/{job_id}/discrimination")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    # Check PNG magic bytes
    assert resp.content[:4] == b"\x89PNG"

    resp = client.get(f"/api/figures/{job_id}/ranking")
    assert resp.status_code == 200
    assert resp.content[:4] == b"\x89PNG"

    resp = client.get(f"/api/figures/{job_id}/dashboard")
    assert resp.status_code == 200
    assert resp.content[:4] == b"\x89PNG"


def test_list_jobs(client: httpx.Client) -> None:
    resp = client.get("/api/pipeline/jobs")
    assert resp.status_code == 200
    jobs = resp.json()
    assert len(jobs) >= 1


def test_panels_crud(client: httpx.Client) -> None:
    # Create
    resp = client.post("/api/panels", json={
        "name": "Test Panel",
        "description": "Integration test panel",
        "mutations": TEST_MUTATIONS,
    })
    assert resp.status_code == 201
    panel_id = resp.json()["panel_id"]

    # List
    resp = client.get("/api/panels")
    assert resp.status_code == 200
    assert any(p["panel_id"] == panel_id for p in resp.json())

    # Get
    resp = client.get(f"/api/panels/{panel_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Test Panel"

    # Delete
    resp = client.delete(f"/api/panels/{panel_id}")
    assert resp.status_code == 204

    # Verify deleted
    resp = client.get(f"/api/panels/{panel_id}")
    assert resp.status_code == 404


def test_job_not_found(client: httpx.Client) -> None:
    resp = client.get("/api/pipeline/jobs/nonexistent")
    assert resp.status_code == 404
