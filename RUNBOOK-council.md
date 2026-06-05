# LLM Council — Runbook (async + roles)

The Council exposes a job/poll async API consumed by **council-mcp**:

- `POST /api/council/decide_async`  body `{"prompt", "roles"?}` → `{"job_id", "status"}` (returns ~instantly)
- `GET  /api/council/decide_result/{job_id}` → `{"status": "pending" | "done" | "error", ...}`
  - `done` shape: `{status, stage1[], stage2, stage3:{response}, metadata:{chairman,...}}` — this is exactly what council-mcp's `shape_result()` reads. **Do not change these keys without updating council-mcp.**
- `POST /api/council/decide` (sync) still works for legacy/explicit-`models` callers.

The async path runs `run_full_council_roles`: 7 specialised roles (first-principles, expansionist,
outsider, risk-manager, investigator, foresight, steel-manner) → per-role Stage-2 rankings + an
independent fabrication auditor → an **Opus 4.8 ultrathink** Chairman (retry → sonnet-4-6 fallback).

## Hard operational rules

1. **Single worker.** The job store (`council_async._JOBS`) is in-process. Run uvicorn with
   **one** worker (it already defaults to 1; the Dockerfile pins `--workers 1`). With >1 worker a
   poll may hit a different worker and get a clean "job not found" — degraded, never a crash, but
   the run is effectively lost. If you ever need horizontal scale, move the store to Redis first
   (propose-and-wait change).

2. **Every model the Council uses MUST be registered in `providers.py`.** A role or chairman model
   that isn't routable silently fails (returns `None`). `tests/test_council_async.py` enforces this
   invariant — keep it green. (This is the bug class that took the Council down.)

3. **Image changes need a rebuild, not just recreate** (source is COPY'd into the image). After any
   `backend/*.py` change: `docker compose build llm-council` then `up -d --force-recreate`.

4. **It must survive redeploys.** All Council code lives in `BAS-More/llm-council` (this repo).
   On-box edits do **not** persist — the atomic-redeploy clones fresh from GitHub. Never patch the
   box directly. See FREEZE.md.

## Resilience guarantees (why it "doesn't break")

- A failing/raising/timed-out run is recorded as `{status:error}` on that one job — never crashes the server.
- Runs are timeout-bounded (`COUNCIL_RUN_TIMEOUT=600s`) and concurrency-capped (`MAX_INFLIGHT=3`).
- Unknown/expired `job_id` → clean `{status:error}`.
- The async endpoints are registered **defensively** in `main.py`: if `council_async` fails to import
  for any reason, the rest of the API (sync `/decide`, `/`, conversations) still boots — async just
  degrades to "unavailable".

## Health gate

`python3 scripts/council_health_gate.py` — run after every deploy.
- default (smoke): free, verifies the async routes are registered (catches the 404 regression).
- `--full`: one real round-trip (~170s + Opus spend) for a complete confirmation.
Exit 0 = healthy; exit 2 = BLOCK the deploy.

## Changing anything health-affecting

Protected paths (see `.github/CODEOWNERS`) are **propose-and-wait**: open a PR, get Avi's approval,
re-run the health gate, rebuild + re-freeze the image (FREEZE.md). No autonomous agent may merge them.
