#!/usr/bin/env python3
"""Tests for scripts/council-mcp/council_mcp_server.py — pure logic (no server deps).

Covers the Cloudflare-safety + auth + result-shaping logic the MCP tools rely on:
clamp_wait, check_bearer (fail-closed), build_start_payload, shape_result, poll_until.
The httpx/mcp/starlette runtime wiring is import-guarded, so this runs without them.

Run: python tests/test_council_mcp_server.py   (or python -m pytest tests/)
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel):
    spec = importlib.util.spec_from_file_location("_council_mcp_uut", os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mod = _load("scripts/council-mcp/council_mcp_server.py")


def test_clamp_wait_caps_below_cloudflare():
    assert mod.clamp_wait(1000) == 90          # never exceeds the ~100s edge ceiling
    assert mod.clamp_wait(45) == 45
    assert mod.clamp_wait(-5) == 0
    assert mod.clamp_wait("not a number") == 90


def test_check_bearer_fail_closed_without_token():
    ok, reason = mod.check_bearer("Bearer anything", "")
    assert ok is False and "fail-closed" in reason


def test_check_bearer_missing_header():
    assert mod.check_bearer(None, "secret")[0] is False


def test_check_bearer_malformed():
    assert mod.check_bearer("secret", "secret")[0] is False        # no 'Bearer ' scheme
    assert mod.check_bearer("Basic Zm9v", "secret")[0] is False


def test_check_bearer_wrong_and_right_token():
    assert mod.check_bearer("Bearer nope", "secret")[0] is False
    assert mod.check_bearer("Bearer secret", "secret")[0] is True
    assert mod.check_bearer("bearer secret", "secret")[0] is True  # scheme case-insensitive


def test_build_start_payload():
    assert mod.build_start_payload("Q") == {"prompt": "Q"}
    p = mod.build_start_payload("Q", {"risk-manager": {"beta": 0.0}})
    assert p["roles"] == {"risk-manager": {"beta": 0.0}}


def test_shape_result_pending_and_error():
    assert mod.shape_result("pending", {})["status"] == "pending"
    err = mod.shape_result("error", {"error": "boom"})
    assert err["status"] == "error" and err["error"] == "boom"


def test_shape_result_done_extracts_synthesis():
    body = {"stage1": [{"model": "a"}, {"model": "b"}],
            "stage3": {"response": "Chairman synthesis."},
            "metadata": {"chairman": "anthropic/claude-opus-4-8"}}
    out = mod.shape_result("done", body)
    assert out["status"] == "done"
    assert out["synthesis"] == "Chairman synthesis."
    assert out["chairman"] == "anthropic/claude-opus-4-8"
    assert out["roles_responded"] == 2


def test_poll_until_returns_when_done():
    seq = [("pending", {}), ("pending", {}),
           ("done", {"stage3": {"response": "ok"}, "stage1": [{"m": 1}], "metadata": {}})]
    calls = {"n": 0, "slept": 0}

    def getter(_jid):
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return seq[i]

    def sleeper(_s):
        calls["slept"] += 1

    out = mod.poll_until("job", wait=90, getter=getter, sleeper=sleeper, interval=3)
    assert out["status"] == "done" and out["synthesis"] == "ok"
    assert calls["slept"] >= 1                  # it polled at least once


def test_poll_until_times_out_pending():
    out = mod.poll_until("job", wait=6, getter=lambda _j: ("pending", {}),
                         sleeper=lambda _s: None, interval=3)
    assert out["status"] == "pending"


def _run():
    tests = sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v))
    passed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa
            print(f"ERROR {name}: {e!r}")
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(_run())
