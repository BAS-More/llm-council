# Council FREEZE procedure — "no changes without approval, survives any redeploy"

The Council was lost once because (a) its code was edited only on the Hetzner box and never
committed, so an atomic-redeploy (which clones fresh from GitHub) deleted it, and (b) nothing
gated the deploy on the Council still working. This procedure makes both impossible.

The code half is now in the repo (durable). The steps below are the **operational freeze** — they
require GitHub-admin / Hetzner access and are therefore **Avi's to run** (propose-and-wait, never an
agent's). Do them once after this PR merges.

## 1. Build the approved image and pin it immutably
On the Hetzner box, after merging this PR and letting the deploy build the new image:
```bash
# capture the new image's content digest
docker image inspect ecosystem-llm-council:latest --format '{{.Id}}'
# tag it immutably (date or digest)
docker tag ecosystem-llm-council:latest ecosystem-llm-council:frozen-YYYY-MM-DD
```
(For reference, the pre-fix image was sha256:efe0e294a0f5… — the new one will differ.)

## 2. Take the Council OUT of the auto-rebuild path
In `BAS-More/infrastructure` → `hetzner/docker-compose.yml`, for **both** `llm-council` and
`council-mcp` services, replace the `build:` block with the pinned image:
```yaml
  llm-council:
    image: ecosystem-llm-council:frozen-YYYY-MM-DD   # pinned; do NOT rebuild without approval
    # build: ...   <-- removed so atomic-redeploy can't rebuild it from GitHub
```
And add `llm-council` + `council-mcp` to the **skip-list** in
`/opt/repos/infrastructure/hetzner/atomic-redeploy.sh` so they are not re-cloned/rebuilt on routine
deploys. Result: routine redeploys leave the Council untouched; changing it is a deliberate act.

## 3. Pin the base image by digest (reproducible rebuilds)
At freeze time:
```bash
docker pull python:3.12-slim
docker inspect --format '{{index .RepoDigests 0}}' python:3.12-slim
```
Put that `python@sha256:…` in the `Dockerfile` `FROM` line (a TODO marks the spot).

## 4. Branch protection + ownership (no silent code changes)
On `BAS-More/llm-council`:
- Protect `master`: require a pull request, **1 approval**, and **require review from Code Owners**.
- Restrict who can push to `master`.
- Set the real owner in `.github/CODEOWNERS` (currently `@OWNER` placeholder).
Net: nothing reaches the deploy source without Avi's approval.

## 5. Gate every deploy on health
Wire `scripts/council_health_gate.py` into the deploy (run after `up -d`); abort on exit 2.
Run `--full` once manually after this PR's first deploy to confirm the real Opus round-trip.

## 6. Keep agents off it
- Jules allowlist already excludes `llm-council` — keep it that way.
- The Council's protected paths are **propose-and-wait** for every agent (self-healing, best-practice,
  PR-health/autofix). An agent may open a PR; only Avi merges. See agent-governance.

## To make an approved change later (the only sanctioned path)
1. PR against `master` (CODEOWNERS review) → 2. green tests + `council_health_gate.py --full`
→ 3. Avi approves + merges → 4. rebuild image, re-tag frozen-NEW-DATE, re-pin compose (steps 1–2)
→ 5. deploy → 6. re-run the health gate.
