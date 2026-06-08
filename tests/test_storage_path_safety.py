"""Path-traversal safety tests for backend.storage (CodeQL py/path-injection #2/#3/#4).

The conversation_id reaches storage straight from the URL path param, so a malicious
id must never be able to escape DATA_DIR. Runs fully offline with zero test deps:
`python tests/test_storage_path_safety.py` uses the built-in runner below, and pytest
can also collect the test_* functions.
"""
from pathlib import Path

import backend.storage as storage
from backend.config import DATA_DIR

VALID_ID = "12345678-1234-4234-8234-123456789abc"

# Each of these, if used unsanitized in os.path.join(DATA_DIR, f"{id}.json"), would
# read/write outside DATA_DIR or otherwise abuse the filesystem.
MALICIOUS_IDS = [
    "../../etc/passwd",
    "..\\..\\..\\Windows\\System32\\config\\SAM",
    "/etc/passwd",
    "....//....//secret",
    "abc/../../../etc/passwd",
    "subdir/id",
    "id with spaces",
    "id;rm -rf /",
    "..%2f..%2fetc%2fpasswd",
    "\x00",
    "a" * 500,
    "",
    "not-a-uuid",
]


def test_valid_uuid_maps_directly_inside_data_dir():
    p = Path(storage.get_conversation_path(VALID_ID)).resolve()
    assert p.parent == Path(DATA_DIR).resolve(), p
    assert p.name == f"{VALID_ID}.json"


def test_get_conversation_path_rejects_malicious_ids():
    for bad in MALICIOUS_IDS:
        try:
            storage.get_conversation_path(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for id {bad!r}")


def test_get_conversation_returns_none_for_malicious_id():
    # Must fail closed (no traversal, no exception bubbling) -> endpoint sees 404.
    for bad in MALICIOUS_IDS:
        assert storage.get_conversation(bad) is None, bad


def test_non_string_id_is_rejected():
    for bad in (None, 123, ["x"], {"a": 1}):
        try:
            storage.get_conversation_path(bad)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for non-string id {bad!r}")


if __name__ == "__main__":
    import sys

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
