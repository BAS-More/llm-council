#!/usr/bin/env python3
"""Pre/post-deploy HEALTH GATE for the LLM Council — the "can-I-deploy" check.

Exit 0 = healthy (deploy may proceed).  Exit 2 = UNHEALTHY → BLOCK the deploy.
Exit 3 = bad usage.  Designed to be wired into atomic-redeploy AFTER `up -d` so a
broken Council can never silently go (or stay) live — and to be run by hand for a
full confirmation. Stdlib only (the council container has no `curl`); run with the
host's python3 against the container's published port or the internal service.

MODES
  (default) smoke  — FREE, no LLM spend. Asserts: backend up; the async endpoints are
                     REGISTERED (this is exactly the regression that wiped the Council —
                     POST /api/council/decide_async returned 404 — caught for $0).
  --full           — ALSO runs ONE real async council round-trip (~170s + Opus spend).
                     Use for a manual post-deploy confirmation, not on every deploy.

USAGE
  python3 council_health_gate.py [--base http://localhost:8001] [--full] [--timeout 300]
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def _req(method, url, body=None, timeout=20):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, (e.read().decode() if e.fp else "")
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def smoke(base):
    """Free checks — no LLM call. Returns a list of failure strings (empty = healthy)."""
    fails = []

    st, _ = _req("GET", f"{base}/")
    if st != 200:
        fails.append(f"GET / -> {st} (expected 200; backend down?)")

    # POST-only route: GET should be 405 (registered) — 404 means the async route is MISSING.
    st, _ = _req("GET", f"{base}/api/council/decide_async")
    if st == 404:
        fails.append("POST /api/council/decide_async MISSING (404) — async Council not wired")
    elif st is None:
        fails.append("/api/council/decide_async unreachable")

    # GET route: an unknown job_id returns 200 with {status:error,'job not found'}.
    # 404 here means the result route itself is missing.
    st, body = _req("GET", f"{base}/api/council/decide_result/healthprobe")
    if st == 404:
        fails.append("GET /api/council/decide_result/{job_id} MISSING (404)")
    elif st != 200:
        fails.append(f"decide_result -> {st} (expected 200 with a status field)")
    return fails


def full(base, timeout):
    """One real async round-trip. Returns a list of failure strings (empty = healthy)."""
    fails = []
    st, body = _req("POST", f"{base}/api/council/decide_async",
                    {"prompt": "Health gate probe — reply in one short sentence."})
    if st != 200:
        return [f"decide_async POST -> {st}: {body[:200]}"]
    try:
        job_id = json.loads(body)["job_id"]
    except Exception:  # noqa: BLE001
        return [f"decide_async returned no job_id: {body[:200]}"]

    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        st, body = _req("GET", f"{base}/api/council/decide_result/{job_id}")
        if st != 200:
            return [f"decide_result -> {st}: {body[:200]}"]
        try:
            last = json.loads(body)
        except Exception:  # noqa: BLE001
            return [f"decide_result not JSON: {body[:200]}"]
        if last.get("status") != "pending":
            break
        time.sleep(3)

    if last.get("status") != "done":
        return [f"council did not complete: {json.dumps(last)[:300]}"]
    if not (last.get("stage3") or {}).get("response"):
        fails.append("done but empty synthesis (stage3.response)")
    if not (last.get("metadata") or {}).get("chairman"):
        fails.append("done but no chairman in metadata")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8001")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()
    base = args.base.rstrip("/")

    fails = smoke(base)
    if not fails and args.full:
        fails += full(base, args.timeout)

    if fails:
        print("COUNCIL HEALTH GATE: FAIL — BLOCK DEPLOY")
        for f in fails:
            print(f"  - {f}")
        sys.exit(2)
    print(f"COUNCIL HEALTH GATE: PASS ({'smoke+full' if args.full else 'smoke'})")
    sys.exit(0)


if __name__ == "__main__":
    main()
