"""Tests for the reconstructed async Council + role wiring.

Runs fully offline (run_full_council_roles is monkeypatched — no LLM/API calls) and
with ZERO test deps: `python tests/test_council_async.py` runs them via the built-in
runner below, and pytest can also collect the test_* functions if it's installed.

Covers: the council-mcp contract shape, resilience (error isolation, timeout,
concurrency cap, unknown/expired job), and the provider-routability invariants whose
absence is exactly what broke production (a role/chairman model not registered in
providers.py silently fails).
"""
import asyncio
import inspect

import backend.council_async as ca


def _reset():
    ca._JOBS.clear()
    ca._inflight = 0


class _MP:
    """Minimal monkeypatch shim so tests run without pytest. pytest injects its own
    `monkeypatch` fixture (same .setattr API, auto-undo); the standalone runner injects
    this and calls undoall() after each test."""
    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, val):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def undoall(self):
        for obj, name, old in reversed(self._undo):
            setattr(obj, name, old)
        self._undo = []


def _ok_run(_prompt, _betas):
    """Stand-in run_full_council_roles return: (stage1, stage2, stage3, metadata)."""
    async def _run(_p, _b):
        return (
            [{"role": "first-principles", "model": "groq/qwen3-32b", "response": "a"}],
            [{"role": "first-principles", "model": "groq/qwen3-32b", "ranking": "FINAL RANKING:\n1. Response A"}],
            {"model": "anthropic/claude-opus-4-8", "response": "the synthesis"},
            {"label_to_model": {}, "aggregate_rankings": [], "fabrication_audit": "FABRICATION FLAGS: none"},
        )
    return _run


async def _start_and_finish(prompt):
    started = await ca.start(prompt)
    task = ca._JOBS.get(started["job_id"], {}).get("_task")
    if task is not None:
        await task
    return started["job_id"], started


# ---------------------------------------------------------------- happy path + contract

def test_done_result_matches_council_mcp_contract(monkeypatch):
    _reset()
    monkeypatch.setattr(ca, "run_full_council_roles", _ok_run(None, None))
    jid, started = asyncio.run(_start_and_finish("hello"))
    assert started["status"] == "pending"
    res = ca.result(jid)
    assert res["status"] == "done"
    assert isinstance(res["stage1"], list) and len(res["stage1"]) == 1
    assert res["stage3"]["response"] == "the synthesis"                 # shape_result reads stage3.response
    assert res["metadata"]["chairman"] == "anthropic/claude-opus-4-8"   # shape_result reads metadata.chairman
    assert res["metadata"]["models_used"] == ["groq/qwen3-32b"]


# ---------------------------------------------------------------- resilience

def test_unknown_job_is_clean_error():
    _reset()
    res = ca.result("does-not-exist")
    assert res["status"] == "error" and "not found" in res["error"]


def test_empty_prompt_rejected_without_crash():
    _reset()
    started = asyncio.run(ca.start("   "))
    assert started["status"] == "error"
    assert ca.result(started["job_id"])["status"] == "error"


def test_run_exception_is_isolated(monkeypatch):
    _reset()
    async def boom(_p, _b):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(ca, "run_full_council_roles", boom)
    jid, _ = asyncio.run(_start_and_finish("x"))
    res = ca.result(jid)
    assert res["status"] == "error" and "kaboom" in res["error"]


def test_run_timeout_is_bounded(monkeypatch):
    _reset()
    async def slow(_p, _b):
        await asyncio.sleep(0.5)
        return ([], [], {"response": ""}, {})
    monkeypatch.setattr(ca, "COUNCIL_RUN_TIMEOUT", 0.01)
    monkeypatch.setattr(ca, "run_full_council_roles", slow)
    jid, _ = asyncio.run(_start_and_finish("x"))
    res = ca.result(jid)
    assert res["status"] == "error" and "exceeded" in res["error"]


def test_concurrency_cap_returns_busy():
    _reset()
    ca._inflight = ca.MAX_INFLIGHT
    started = asyncio.run(ca.start("x"))
    assert started["status"] == "error"
    assert "busy" in ca.result(started["job_id"])["error"]


# ---------------------------------------------------------------- provider-routability invariants
# (the class of bug that broke prod: a model used by the council but not registered in providers.py)

def test_all_role_models_are_registered_in_providers():
    _reset()
    from backend.roles import ROLE_DEFINITIONS
    from backend.providers import PROVIDERS
    for name, rd in ROLE_DEFINITIONS.items():
        assert rd["model"] in PROVIDERS, f"role '{name}' model '{rd['model']}' not routable"


def test_chairman_is_opus48_and_routable():
    _reset()
    from backend.roles import CHAIRMAN_MODEL
    from backend.providers import PROVIDERS
    assert CHAIRMAN_MODEL == "anthropic/claude-opus-4-8"
    assert CHAIRMAN_MODEL in PROVIDERS, "chairman model not registered in providers.py"


def test_auditor_and_fallback_routable():
    _reset()
    from backend import council_roles as cr
    from backend.providers import PROVIDERS
    assert cr.AUDITOR_MODEL in PROVIDERS
    assert cr.CHAIRMAN_FALLBACK_MODEL in PROVIDERS


def test_opus48_has_ultrathink_config():
    _reset()
    from backend.providers import PROVIDERS
    p = PROVIDERS["anthropic/claude-opus-4-8"]
    assert p["type"] == "anthropic"
    assert p.get("thinking") is True
    assert p["max_tokens"] > p["thinking_budget"]  # Anthropic API requires this


# ---------------------------------------------------------------- wiring

def test_async_routes_registered():
    _reset()
    import backend.main as m
    paths = {r.path for r in m.app.routes}
    assert "/api/council/decide" in paths
    assert "/api/council/decide_async" in paths
    assert "/api/council/decide_result/{job_id}" in paths


if __name__ == "__main__":
    import sys
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    failures = []
    for name, fn in tests:
        _reset()
        mp = _MP()
        try:
            if "monkeypatch" in inspect.signature(fn).parameters:
                fn(mp)
            else:
                fn()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            failures.append((name, e))
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        finally:
            mp.undoall()
            _reset()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
