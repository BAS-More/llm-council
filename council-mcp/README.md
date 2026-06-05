# council-mcp

Exposes the **LLM Council** as Model Context Protocol tools, so any MCP client (Claude
Code/Desktop, the AVI-OS output-gate) can call the Council like `total-recall`, `claude-vault`,
or `avios-context-mcp`.

## Why this shape

A Council run is **~170s** and this server sits behind **Cloudflare (~100s hard edge)**. A single
blocking tool call would re-introduce the exact timeout the async Council was built to defeat. So
the server mirrors that job/poll pattern one layer up — **every tool call returns in well under
100s**:

| Tool | Returns | Timing |
|------|---------|--------|
| `council_start(prompt, [roles])` | `{job_id, status:"pending"}` | ~1s |
| `council_result(job_id, wait=90)` | long-polls ≤90s → `{status:"pending"}` or `{status:"done", synthesis, chairman, roles_responded, stage1, stage2, metadata}` | ≤90s |

The model calls `council_start` once, then `council_result` 1–3 times (the `wait=90` long-poll means
~2 calls cover a 170s run). No hop ever nears Cloudflare's edge or an MCP client's tool timeout.

- **Transport:** streamable-HTTP (remote, cloud-hosted, co-located with the Council). MCP endpoint `/mcp`.
- **Auth:** `Authorization: Bearer <COUNCIL_MCP_TOKEN>`, **fail-closed** (no token configured ⇒ every
  request denied). The Council triggers **Opus 4.8 ultrathink** (real API spend) on every call — this
  must not be open like the Council's UA-only front door.
- **Health:** `GET /healthz` (unauthenticated) → `{"status":"ok", "auth":"configured|MISSING-TOKEN"}`.

## Files
- `council_mcp_server.py` — the server (pure logic + import-guarded MCP/httpx/starlette wiring).
- `requirements.txt` · `Dockerfile` · `docker-compose.snippet.yml`
- Tests: `../../tests/test_council_mcp_server.py` (pure logic; no server deps needed).

## Deploy

### Claude's part (done in repo)
Server + auth + tests are written and unit-tested. The container builds from this folder.

### Avi's part (infra — when you're ready)
1. **Vault token:** generate a strong secret and add it to `/opt/ecosystem/.env`:
   ```
   COUNCIL_MCP_TOKEN=<paste once — your 401 scar; Ctrl+V can double on hidden inputs>
   ```
2. **Copy** this `council-mcp/` folder to the box at `/opt/repos/llm-council/council-mcp/`.
3. **Compose:** append `docker-compose.snippet.yml` to `/opt/ecosystem/docker-compose.yml`, then:
   ```
   cd /opt/ecosystem && docker compose build council-mcp && docker compose up -d council-mcp
   ```
4. **Cloudflare/DNS:** point `council-mcp.aidev.com.au` → this host `:8002`.
5. **Verify:** `curl https://council-mcp.aidev.com.au/healthz` → `{"status":"ok","auth":"configured"}`.

## Connect a client (Claude Code)
```
claude mcp add --transport http council \
  https://council-mcp.aidev.com.au/mcp \
  --header "Authorization: Bearer <COUNCIL_MCP_TOKEN>"
```
Then `council_start` / `council_result` appear as tools.

## Local smoke (without Cloudflare)
```
pip install -r requirements.txt
COUNCIL_BASE_URL=http://localhost:8001 COUNCIL_MCP_TOKEN=dev \
  uvicorn council_mcp_server:app --port 8002
curl localhost:8002/healthz
```
