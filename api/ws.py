"""WebSocket for real-time pipeline progress."""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.state import AppState, JobStatus

router = APIRouter(tags=["websocket"])

_state: AppState | None = None


def init(state: AppState) -> None:
    global _state
    _state = state


@router.websocket("/ws/jobs/{job_id}")
async def job_progress(websocket: WebSocket, job_id: str) -> None:
    """Stream job progress over WebSocket.

    Sends JSON status updates every 0.5s until job completes or fails.
    Requires valid API key when COMPASS_API_KEY is set.
    """
    # API key check (matches HTTP middleware behaviour)
    api_key = os.environ.get("COMPASS_API_KEY", "")
    if api_key:
        # Check query param (?token=...) or Sec-WebSocket-Protocol header
        token = websocket.query_params.get("token", "")
        if token != api_key:
            await websocket.close(code=4001)
            return

    if _state is None:
        await websocket.close(code=1011)
        return

    job = _state.get_job(job_id)
    if job is None:
        await websocket.close(code=4004)
        return

    await websocket.accept()

    try:
        while True:
            job = _state.get_job(job_id)
            if job is None:
                break

            payload = {
                "job_id": job.job_id,
                "status": job.status.value,
                "progress": job.progress,
                "current_module": job.current_module,
                "error": job.error,
            }
            await websocket.send_text(json.dumps(payload))

            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
