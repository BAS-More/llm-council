"""Async job layer for the Council — Sprint 45.3 behaviour, reconstructed durably.

WHY THIS EXISTS
    The original async endpoints were edited directly on the Hetzner box and never
    committed to GitHub, so the next atomic-redeploy (clones fresh from GitHub) wiped
    them — the council-mcp wrapper then 404'd on /api/council/decide_async. This module
    re-implements that behaviour IN THE REPO so it survives every redeploy.

WHAT IT DOES
    Wraps run_full_council_roles() in a fire-and-forget background task with a job/poll
    API, so each HTTP hop returns in well under a second. That keeps every call under
    Cloudflare's ~100s edge timeout (a full ultrathink run is ~170s).

        start(prompt[, role_betas]) -> {"job_id", "status"}        (returns immediately)
        result(job_id)             -> {"status": pending|done|error, ...}

BULLETPROOFING ("doesn't break no matter what")
    - Exception-isolated: a failing run is recorded as {status:error} on the job; the
      exception never escapes to crash the server.
    - Timeout-bounded: a hung provider can't pin a job forever (COUNCIL_RUN_TIMEOUT).
    - Concurrency-capped: a flood can't exhaust API budget / memory (MAX_INFLIGHT).
    - TTL-expired: finished jobs are lazily GC'd (JOB_TTL); no background timer to crash.
    - Unknown / expired job_id -> clean {status:error}, never an exception.
    The result() shape matches exactly what council-mcp's shape_result() reads:
    {status, stage1:[...], stage2, stage3:{response}, metadata:{chairman}}.

    NOTE: the job store is in-process. Run the backend single-worker
    (uvicorn ... --workers 1). With >1 worker a poll may hit a different worker and get
    a clean "job not found" (degraded, never a crash).
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, Optional

from .council_roles import run_full_council_roles

_log = logging.getLogger("uvicorn.error")

# --- Tunables (conservative defaults; raising them is a protected-path change → review) ---
COUNCIL_RUN_TIMEOUT = 600   # hard ceiling for one run (ultrathink ~170s; 600 = wide safety margin)
MAX_INFLIGHT = 3            # bulkhead: max concurrent runs (bounds API spend + memory)
JOB_TTL = 3600             # seconds a job is retained after creation (1h)

_JOBS: Dict[str, Dict[str, Any]] = {}
_inflight = 0
_lock = asyncio.Lock()


def _gc(now: float) -> None:
    """Lazily drop jobs older than JOB_TTL. Snapshot iteration is safe against concurrent
    updates; worst case a just-expired job returns a clean 'not found'. Never raises."""
    try:
        stale = [jid for jid, j in list(_JOBS.items())
                 if now - j.get("created", now) > JOB_TTL]
        for jid in stale:
            _JOBS.pop(jid, None)
    except Exception:
        pass


async def _run(job_id: str, prompt: str, role_betas: Optional[dict]) -> None:
    """Background worker. FULLY exception-isolated — a job failure is recorded on the job,
    never propagated. Always releases its in-flight slot."""
    global _inflight
    try:
        stage1, stage2, stage3, metadata = await asyncio.wait_for(
            run_full_council_roles(prompt, role_betas), timeout=COUNCIL_RUN_TIMEOUT
        )
        metadata = dict(metadata or {})
        # council-mcp reads metadata["chairman"]; the roles path doesn't set it, so derive
        # it from the Stage-3 result (the model actually used, incl. any fallback).
        metadata.setdefault("chairman", (stage3 or {}).get("model"))
        metadata.setdefault("models_used", [r.get("model") for r in (stage1 or [])])
        job = _JOBS.get(job_id)
        if job is not None:
            job.update(status="done", stage1=stage1, stage2=stage2,
                       stage3=stage3, metadata=metadata)
    except asyncio.TimeoutError:
        job = _JOBS.get(job_id)
        if job is not None:
            job.update(status="error", error=f"council run exceeded {COUNCIL_RUN_TIMEOUT}s")
    except Exception:  # noqa: BLE001 — deliberate catch-all: a job must never crash the server
        # Log full detail server-side; return a GENERIC message to the caller so exception
        # text never reaches the HTTP response (avoids py/stack-trace-exposure / info leak).
        _log.exception("council job %s failed", job_id)
        job = _JOBS.get(job_id)
        if job is not None:
            job.update(status="error", error="internal council error")
    finally:
        async with _lock:
            _inflight = max(0, _inflight - 1)


async def start(prompt: str, role_betas: Optional[dict] = None) -> Dict[str, Any]:
    """Create a job and launch its background run. Returns {job_id, status} immediately.
    Validates input and enforces the concurrency cap; both rejections come back as a
    terminal job (status:error) rather than raising, so callers always get a job_id."""
    global _inflight
    now = time.monotonic()
    _gc(now)

    if not isinstance(prompt, str) or not prompt.strip():
        jid = uuid.uuid4().hex
        _JOBS[jid] = {"status": "error", "error": "prompt must be a non-empty string", "created": now}
        return {"job_id": jid, "status": "error"}

    async with _lock:
        if _inflight >= MAX_INFLIGHT:
            jid = uuid.uuid4().hex
            _JOBS[jid] = {"status": "error",
                          "error": f"council busy ({_inflight}/{MAX_INFLIGHT} in flight) — retry shortly",
                          "created": now}
            return {"job_id": jid, "status": "error"}
        _inflight += 1

    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {"status": "pending", "created": now}
    # Keep a reference to the task so it isn't garbage-collected mid-run.
    _JOBS[job_id]["_task"] = asyncio.create_task(_run(job_id, prompt, role_betas))
    return {"job_id": job_id, "status": "pending"}


def result(job_id: str) -> Dict[str, Any]:
    """Return the council-mcp-shaped result for a job_id. Unknown/expired -> clean error.
    Pure read (plus lazy GC); never raises."""
    _gc(time.monotonic())
    job = _JOBS.get(job_id)
    if job is None:
        return {"status": "error", "error": "job not found or expired"}
    status = job.get("status", "pending")
    if status == "pending":
        return {"status": "pending"}
    if status == "error":
        return {"status": "error", "error": job.get("error", "unknown council error")}
    return {
        "status": "done",
        "stage1": job.get("stage1", []),
        "stage2": job.get("stage2", []),
        "stage3": job.get("stage3", {}),
        "metadata": job.get("metadata", {}),
    }
