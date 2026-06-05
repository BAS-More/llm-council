#!/usr/bin/env python3
"""council-mcp — exposes the LLM Council as Model Context Protocol tools.

Wraps the async Council HTTP API (POST /api/council/decide_async +
GET /api/council/decide_result/{job_id}) behind MCP so any client (Claude
Code/Desktop, the AVI-OS output-gate) can call the Council the same way it
calls total-recall, claude-vault, avios-context-mcp, etc.

WHY two tools + long-poll (not one blocking call): a Council run is ~170s and
this server sits behind Cloudflare (~100s hard edge). Every tool call must
return well under that ceiling, so:
    council_start(prompt[, roles])  -> {job_id}            (~1s)
    council_result(job_id, wait=90) -> {pending | done...} (long-polls <=90s)
The model calls start once, then result 1-3 times. No hop ever nears 100s.
This is the same job/poll pattern that made the Council itself Cloudflare-proof,
applied one layer up.

Transport : streamable-HTTP (remote / cloud-hosted, co-located with the Council).
Auth      : Bearer token (COUNCIL_MCP_TOKEN), FAIL-CLOSED. The Council triggers
            Opus 4.8 ultrathink (real API spend) on every call, so this MUST NOT
            be left open like the Council's UA-only front door.

Run  : uvicorn council_mcp_server:app --host 0.0.0.0 --port 8002
Env  : COUNCIL_BASE_URL      (default http://llm-council:8001 — compose service name)
       COUNCIL_MCP_TOKEN     (REQUIRED; server denies every request if unset)
       COUNCIL_POLL_INTERVAL (default 3 seconds)
MCP endpoint : /mcp        Health : /healthz (unauthenticated)
"""
import hmac
import os

# --- config (env-driven; read at import) ---
COUNCIL_BASE_URL = os.environ.get("COUNCIL_BASE_URL", "http://llm-council:8001").rstrip("/")
COUNCIL_MCP_TOKEN = os.environ.get("COUNCIL_MCP_TOKEN", "")
POLL_INTERVAL = int(os.environ.get("COUNCIL_POLL_INTERVAL", "3"))
MAX_WAIT = 90  # Cloudflare-safe ceiling for a single council_result call (< ~100s edge)
START_ENDPOINT = "/api/council/decide_async"
RESULT_ENDPOINT = "/api/council/decide_result/"


# ======================================================================
# Pure logic (stdlib only) — unit-tested in tests/test_council_mcp_server.py.
# Kept dependency-free so the module imports for tests without httpx/mcp.
# ======================================================================
def clamp_wait(wait, lo=0, hi=MAX_WAIT):
    """Clamp a requested long-poll wait into [lo, hi]; hi stays under Cloudflare's ~100s edge."""
    try:
        w = int(wait)
    except (TypeError, ValueError):
        w = hi
    return max(lo, min(hi, w))


def check_bearer(authorization, expected):
    """Validate an 'Authorization: Bearer <token>' header against the configured token.
    FAIL-CLOSED: no configured token => deny. Returns (ok: bool, reason: str)."""
    if not expected:
        return (False, "server has no COUNCIL_MCP_TOKEN configured (fail-closed: deny all)")
    if not authorization:
        return (False, "missing Authorization header")
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return (False, "malformed Authorization header (expected 'Bearer <token>')")
    if not hmac.compare_digest(parts[1].strip(), str(expected)):
        return (False, "invalid token")
    return (True, "ok")


def build_start_payload(prompt, roles=None):
    """Body for POST /decide_async."""
    payload = {"prompt": prompt}
    if roles:
        payload["roles"] = roles
    return payload


def shape_result(status, body):
    """Normalise a /decide_result response into a compact tool result."""
    body = body or {}
    if status == "pending":
        return {"status": "pending",
                "hint": "not finished yet — call council_result again with the same job_id"}
    if status == "error":
        return {"status": "error", "error": body.get("error", "unknown council error")}
    s3 = body.get("stage3", {})
    synthesis = s3.get("response", "") if isinstance(s3, dict) else str(s3)
    meta = body.get("metadata", {}) or {}
    return {
        "status": "done",
        "synthesis": synthesis,
        "chairman": meta.get("chairman"),
        "roles_responded": len(body.get("stage1", []) or []),
        "stage1": body.get("stage1"),
        "stage2": body.get("stage2"),
        "metadata": meta,
    }


def poll_until(job_id, wait, getter, sleeper, interval=POLL_INTERVAL):
    """Long-poll a job to completion or until `wait` seconds elapse.
    getter(job_id) -> (status:str, body:dict); sleeper(seconds) -> None.
    Returns a shaped result; {status:pending} if still running at the deadline.
    Pure + synchronous so it can be unit-tested with fakes; council_result mirrors it."""
    waited = 0
    status, body = getter(job_id)
    if status != "pending":
        return shape_result(status, body)
    while waited < wait:
        sleeper(interval)
        waited += interval
        status, body = getter(job_id)
        if status != "pending":
            return shape_result(status, body)
    return shape_result("pending", {})


# ======================================================================
# Runtime wiring (needs httpx + mcp + starlette) — import-guarded so the module
# still imports for unit tests when those server deps aren't installed.
# ======================================================================
try:
    import asyncio
    import httpx
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    _RUNTIME = True
except ImportError:  # unit-test / lint environment without server deps
    _RUNTIME = False


# Hosts the MCP streamable-HTTP transport will accept (DNS-rebinding protection). This server
# sits behind Cloudflare -> tunnel -> Caddy, which forwards Host: <public host>; without
# allow-listing it, the transport rejects with 421 "Invalid Host header". Env-configurable.
_PUBLIC_HOST = os.environ.get("COUNCIL_MCP_PUBLIC_HOST", "council-mcp.aidev.com.au")
_ALLOWED_HOSTS = [_PUBLIC_HOST, _PUBLIC_HOST + ":443",
                  "localhost", "localhost:8002", "127.0.0.1", "127.0.0.1:8002"]
_ALLOWED_ORIGINS = ["https://" + _PUBLIC_HOST]


if _RUNTIME:
    mcp = FastMCP("council", transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_ALLOWED_HOSTS,
        allowed_origins=_ALLOWED_ORIGINS,
    ))

    async def _result_getter(job_id):
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{COUNCIL_BASE_URL}{RESULT_ENDPOINT}{job_id}")
            r.raise_for_status()
            body = r.json()
        return body.get("status", "pending"), body

    @mcp.tool()
    async def council_start(prompt: str, roles: dict | None = None) -> dict:
        """Start an LLM Council decision and return a job_id immediately.

        The Council runs 7 specialised roles (First-Principles, Expansionist, Outsider,
        Risk-Manager, Investigator, Foresight, Steel-Manner) plus an Opus 4.8 ultrathink
        Chairman, and takes ~170s. This returns at once; then call
        council_result(job_id, wait=90) to retrieve the synthesis.

        Args:
            prompt: the question / draft to put to the Council.
            roles:  optional per-role intensity overrides, e.g. {"risk-manager": {"beta": 0.0}}.
        """
        payload = build_start_payload(prompt, roles)
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(f"{COUNCIL_BASE_URL}{START_ENDPOINT}", json=payload)
            r.raise_for_status()
            data = r.json()
        return {"job_id": data.get("job_id"), "status": "pending",
                "hint": "call council_result(job_id, wait=90) to poll (~170s total run)"}

    @mcp.tool()
    async def council_result(job_id: str, wait: int = MAX_WAIT) -> dict:
        """Retrieve a Council decision started with council_start.

        Long-polls up to `wait` seconds (clamped to <=90s so the call stays under
        Cloudflare's edge). Returns {status:"pending"} (call again with the same job_id)
        or {status:"done", synthesis, chairman, roles_responded, stage1, stage2, metadata}.
        """
        w = clamp_wait(wait)
        status, body = await _result_getter(job_id)
        if status != "pending":
            return shape_result(status, body)
        waited = 0
        while waited < w:
            await asyncio.sleep(POLL_INTERVAL)
            waited += POLL_INTERVAL
            status, body = await _result_getter(job_id)
            if status != "pending":
                return shape_result(status, body)
        return shape_result("pending", {})

    # --- Bearer auth (fail-closed) wrapping the streamable-HTTP app ---
    class BearerAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.url.path in ("/healthz", "/health"):
                return await call_next(request)
            ok, reason = check_bearer(request.headers.get("authorization"), COUNCIL_MCP_TOKEN)
            if not ok:
                return JSONResponse({"error": "unauthorized", "reason": reason}, status_code=401)
            return await call_next(request)

    async def _healthz(request):
        return JSONResponse({"status": "ok", "service": "council-mcp",
                             "council": COUNCIL_BASE_URL,
                             "auth": "configured" if COUNCIL_MCP_TOKEN else "MISSING-TOKEN"})

    app = mcp.streamable_http_app()
    app.add_middleware(BearerAuthMiddleware)
    app.router.routes.append(Route("/healthz", _healthz, methods=["GET"]))


if __name__ == "__main__":
    if not _RUNTIME:
        raise SystemExit("council-mcp runtime deps missing — pip install -r requirements.txt")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8002")))
