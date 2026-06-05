# Connecting to the Council MCP

How to wire the **LLM Council** into an MCP client (Claude Code / Desktop) so `council_start` and
`council_result` appear as tools. The server itself is already deployed (see `README.md`); this is the
**client** side.

| | |
|---|---|
| **Endpoint** | `https://council-mcp.aidev.com.au/mcp` (streamable-HTTP) |
| **Auth** | `Authorization: Bearer <COUNCIL_MCP_TOKEN>` — **fail-closed** (no/!wrong token ⇒ 401) |
| **Token** | lives in the vault at `council-mcp/token` (`vault get council-mcp token`) |
| **Tools** | `council_start(prompt[,roles])` → `{job_id}` · `council_result(job_id, wait≤90)` → poll |

> ⚠️ The token gates **paid Opus-4.8-ultrathink** runs on every call. Treat it as a secret — never commit
> it, never paste it where it'll be echoed.

---

## Option A — project scope (this repo's `.mcp.json`)

This repo ships a project-scoped [`.mcp.json`](../../.mcp.json) that auto-registers the Council MCP whenever
you open the Brain x2 project. It reads the token from the **`COUNCIL_MCP_TOKEN` environment variable** (no
secret is committed):

```jsonc
{ "mcpServers": { "council": {
  "type": "http",
  "url": "https://council-mcp.aidev.com.au/mcp",
  "headers": { "Authorization": "Bearer ${COUNCIL_MCP_TOKEN}" }
}}}
```

**Before launching Claude Code**, export the token (PowerShell):

```powershell
$env:COUNCIL_MCP_TOKEN = & "C:\Dev\tools\vault\vault.bat" get council-mcp token
# then launch / relaunch Claude Code from this same shell so it inherits the var
```

Claude Code will prompt to approve the project MCP server on first open. Approve it, and the tools load.

## Option B — user scope (global, every project)

To have the Council available in **every** project (not just this repo), add it at **user scope** once.
This is Claude Code Desktop's only "MCP setup" path — there is no in-app UI; the `claude mcp` CLI edits the
same `.claude.json` the Desktop app reads.

```powershell
# remove any project-local copy first (run from the dir where you previously added it), to avoid a duplicate
claude mcp remove council

# add globally (one line — paste on its own; multi-line paste can inject terminal escape sequences)
$tok = & "C:\Dev\tools\vault\vault.bat" get council-mcp token; claude mcp add --scope user --transport http council "https://council-mcp.aidev.com.au/mcp" --header "Authorization: Bearer $tok"
```

Then **fully quit and reopen** Claude Code Desktop (quit the process / tray icon — not just a new chat);
MCP servers attach at startup, so an already-open session won't pick it up.

Verify:

```powershell
claude mcp list      # should show: council
```

User-scope entries live in the top-level `mcpServers` of `~\.claude-primary\.claude.json` (this install
uses the `.claude-primary` config dir, not the default `~\.claude.json`).

---

## Using it

1. `council_start("<your question>")` → returns a `job_id` immediately.
2. `council_result("<job_id>", wait=90)` → call 1–3 times until `status: "done"`; a full run is ~170–280s
   (7 specialised roles + Opus-4.8 ultrathink chairman + the fabrication-audit gate), so each `wait=90`
   poll returns `pending` until it lands.

The result carries `synthesis`, `chairman`, `roles_responded`, `stage1`, `stage2`, and
`metadata.fabrication_audit` (the independent auditor's flag list).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| **401 unauthorized** | Missing/blank/wrong token. Check `COUNCIL_MCP_TOKEN` is set in the launching shell, or re-add with the correct `vault get council-mcp token`. |
| **421 Invalid Host header** | The server's DNS-rebinding allow-list doesn't include the host. This is server-side (env `COUNCIL_MCP_PUBLIC_HOST`) and already covers `council-mcp.aidev.com.au`. |
| **403 / "error code: 1010"** | Cloudflare blocking a non-browser User-Agent. Only affects raw scripts (e.g. Python `urllib`) — use a browser UA. Real MCP clients are fine. |
| **Tools don't appear** | MCP servers load at **session start** — fully relaunch the client; and confirm scope (a user-scope server won't appear if a project-scope one shadows it, or vice-versa). |
| **`council_result` always pending** | Normal early on (run is minutes). If it never completes, the chairman retries Opus once then falls back to sonnet — check the Council itself: `https://council.aidev.com.au/`. |

## Rotating the token

```powershell
# 1) new token in vault
$new = py -3.13 -c "import secrets; print(secrets.token_urlsafe(32))"
& "C:\Dev\tools\vault\vault.bat" set council-mcp token $new
# 2) update the server-side .env on Hetzner (COUNCIL_MCP_TOKEN) + recreate the container
#    ssh root@178.104.86.210 ... docker compose up -d --force-recreate council-mcp
# 3) re-point clients: re-export COUNCIL_MCP_TOKEN (Option A) or claude mcp remove/add (Option B)
```
