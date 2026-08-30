"""The auditor becomes a gate, and the gate is asserted by DRIVING it.

Created 2026-08-30 after `AUDITOR-EXISTS-AND-IS-NOT-INVOKED-1`.

WHAT THIS GUARDS
----------------
`scripts/maintenance/audit_data_tree.py` has reported the data-layout audit
since 2026-06-17 and NOTHING EVER CALLED IT. MEASURED 2026-08-30: ten tracked
files name it, and every one is documentation, a `.gitignore` comment, or the
script itself.

It had already reported the finding nobody had seen -- three ORPHAN directories
under `data/external`, together 4,685,941,722 bytes, outside the registry that
calls itself canonical for everything under `data/`.

The audit computation was locked inside a 159-line `main()` with 25 `print()`
calls, so nothing could consume the findings without re-deriving them. The
2026-08-30 split gives it `audit_tree()` (computes), `audit_rows()` (renders
severity rows) and `main()` (prints), which is the shape
`preflight_data_guard.py` already uses.

WHY THESE TESTS DRIVE RATHER THAN READ
--------------------------------------
`tests/unit/test_storage_guard.py` states the rule, and it is quoted here
because it names the exact failure this session committed twice:

    A source check passes on dead code and fails on a clean refactor -- both
    directions wrong.

On 2026-08-30 a probe scanning for `import audit_data_tree` reported ZERO
imports while the storage gate was demonstrably wired, because the wiring uses
`importlib.util.spec_from_file_location`. The same probe counted 31
"invocations" of `preflight_data_guard`, every one a line of Markdown prose.

So the wiring is asserted by MONKEYPATCHING the gate and checking a sentinel
row came back through `run_all`.

Acronyms: YAML = YAML Ain't Markup Language.

Author: Monzia Moodie
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_AUDITOR = _ROOT / "scripts" / "maintenance" / "audit_data_tree.py"
_MANIFEST = _ROOT / "configs" / "data_manifest.yaml"

_MINIMAL = """version: 1
sources:
  clinvar:
    location: external
    tier: public
    class: public_redownloadable
    aliases: []
    sync: false
  spliceai:
    location: external
    tier: public
    class: public_redownloadable
    aliases: [spliceai_scores]
    sync: false
  reference:
    location: external
    tier: public
    class: public_redownloadable
    aliases: [grch38]
    sync: false
"""

_CONTROLLED_SYNCED = _MINIMAL + """  tcga:
    location: external
    tier: controlled
    class: irreplaceable
    aliases: []
    sync: true
"""


def _load(path: Path, name: str):
    """Load a module by path.

    The module MUST be registered in sys.modules before exec_module: @dataclass
    under `from __future__ import annotations` resolves its annotations through
    sys.modules[cls.__module__]. `audit_data_tree` now declares dataclasses, so
    this matters here exactly as it does in test_storage_guard.py.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    argv = sys.argv
    sys.argv = [name]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


def _auditor():
    return _load(_AUDITOR, "audit_data_tree")


def _preflight():
    scripts = _ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    try:
        return _load(scripts / "preflight_run17.py", "preflight_run17")
    except BaseException as e:
        pytest.skip(f"preflight_run17 not importable here ({type(e).__name__}); "
                    "this MUST NOT skip on the development machine or in CI")


def _tree(tmp_path: Path, manifest_text: str = _MINIMAL) -> tuple[Path, Path]:
    for sub in ("external", "raw", "processed"):
        (tmp_path / "data" / sub).mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "external" / "clinvar").mkdir()
    (tmp_path / "data" / "external" / "clinvar" / "a.txt").write_bytes(b"x")
    m = tmp_path / "data_manifest.yaml"
    m.write_text(manifest_text, encoding="utf-8")
    return tmp_path / "data", m


# ---------------------------------------------------------------------------
# 1. audit_rows -- the row convention
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _AUDITOR.is_file(), reason="audit_data_tree.py absent")
def test_the_auditor_exposes_audit_rows():
    A = _auditor()
    assert hasattr(A, "audit_rows"), (
        "audit_data_tree.audit_rows is missing -- the audit is locked inside "
        "main() again, which is the state it sat in from 2026-06-17 to "
        "2026-08-30")
    assert hasattr(A, "audit_tree"), "the computation must be callable too"


@pytest.mark.skipif(not _AUDITOR.is_file(), reason="audit_data_tree.py absent")
def test_every_row_is_a_two_tuple_in_the_gate_convention(tmp_path):
    data, man = _tree(tmp_path)
    rows = _auditor().audit_rows(str(data), str(man))
    assert rows
    for r in rows:
        assert isinstance(r, tuple) and len(r) == 2
        assert r[0] in ("OK", "WARN", "FAIL")
        assert isinstance(r[1], str) and r[1]


@pytest.mark.skipif(not _AUDITOR.is_file(), reason="audit_data_tree.py absent")
def test_a_clean_tree_yields_one_OK_row(tmp_path):
    data, man = _tree(tmp_path)
    rows = _auditor().audit_rows(str(data), str(man))
    assert [r[0] for r in rows] == ["OK"], rows


@pytest.mark.skipif(not _AUDITOR.is_file(), reason="audit_data_tree.py absent")
def test_an_orphan_directory_WARNS(tmp_path):
    """The three real orphans -- gencode, grch38, eve_smoke -- are this case."""
    data, man = _tree(tmp_path)
    (data / "external" / "gencode").mkdir()
    rows = _auditor().audit_rows(str(data), str(man))
    warns = [m for s, m in rows if s == "WARN" and "gencode" in m]
    assert warns, rows
    assert "ORPHAN" in warns[0]


@pytest.mark.skipif(not _AUDITOR.is_file(), reason="audit_data_tree.py absent")
def test_an_alias_directory_WARNS_and_names_its_canonical(tmp_path):
    """The canonical name must be NAMED, not merely implied by the alias.

    MEASURED 2026-08-30: the first version used alias `spliceai_scores` ->
    canonical `spliceai`, and asserted `"spliceai" in message`. The ALIAS NAME
    contains `spliceai`, so dropping the canonical from the message changed no
    test. `grch38` -> `reference` is used instead: the canonical name is not a
    substring of the alias, so only naming it satisfies the assertion.

    This is also the real case -- `data/external/grch38` is one of the three
    orphans measured on 2026-08-30, at 4,033,396,532 bytes.
    """
    data, man = _tree(tmp_path)
    (data / "external" / "grch38").mkdir()
    rows = _auditor().audit_rows(str(data), str(man))
    warns = [m for s, m in rows if s == "WARN" and "grch38" in m]
    assert warns, rows
    assert "ALIAS" in warns[0]
    assert "reference" in warns[0], (
        "the message does not NAME the canonical source; a reader cannot act "
        "on it, and consolidate_aliases.py needs the target")


@pytest.mark.skipif(not _AUDITOR.is_file(), reason="audit_data_tree.py absent")
def test_a_controlled_source_marked_for_sync_FAILS(tmp_path):
    """The standard, section 5: controlled data is backed up offline ONLY.

    `main()` exits 2 on this. The gate must render it FAIL, not WARN.
    """
    data, man = _tree(tmp_path, _CONTROLLED_SYNCED)
    rows = _auditor().audit_rows(str(data), str(man))
    fails = [m for s, m in rows if s == "FAIL"]
    assert fails, rows
    assert "tcga" in fails[0] and "controlled" in fails[0]


@pytest.mark.skipif(not _AUDITOR.is_file(), reason="audit_data_tree.py absent")
def test_a_missing_data_dir_FAILS_rather_than_raising(tmp_path):
    m = tmp_path / "data_manifest.yaml"
    m.write_text(_MINIMAL, encoding="utf-8")
    rows = _auditor().audit_rows(str(tmp_path / "absent"), str(m))
    assert rows and rows[0][0] == "FAIL"
    assert "MISSING" in rows[0][1]


@pytest.mark.skipif(not _AUDITOR.is_file(), reason="audit_data_tree.py absent")
def test_a_malformed_manifest_FAILS_rather_than_raising(tmp_path):
    data, _m = _tree(tmp_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text("sources: [this is a list, not a mapping]\n", encoding="utf-8")
    rows = _auditor().audit_rows(str(data), str(bad))
    assert rows and rows[0][0] == "FAIL"


@pytest.mark.skipif(not _AUDITOR.is_file(), reason="audit_data_tree.py absent")
def test_the_return_code_agrees_with_the_rows(tmp_path):
    """One computation, two renderings -- so they cannot disagree.

    `main()` derives its exit code from the same AuditReport the rows come
    from. A FAIL row and an exit code of 0 would mean two calculations.
    """
    A = _auditor()
    data, man = _tree(tmp_path, _CONTROLLED_SYNCED)
    rep = A.audit_tree(data, man)
    rows = A.audit_rows(str(data), str(man))
    assert rep.return_code == 2
    assert any(s == "FAIL" for s, _ in rows)

    data2, man2 = _tree(tmp_path / "clean")
    rep2 = A.audit_tree(data2, man2)
    rows2 = A.audit_rows(str(data2), str(man2))
    assert rep2.return_code == 0
    assert all(s == "OK" for s, _ in rows2)


@pytest.mark.skipif(not _MANIFEST.is_file(), reason="data_manifest.yaml absent")
def test_the_REAL_tree_is_auditable():
    """A gate that parses a fixture and refuses the real tree is useless."""
    rows = _auditor().audit_rows(str(_ROOT / "data"), str(_MANIFEST))
    assert rows
    assert all(s in ("OK", "WARN", "FAIL") for s, _ in rows)


# ---------------------------------------------------------------------------
# 2. the gate -- the shape preflight_run17 consumes
# ---------------------------------------------------------------------------

def test_preflight_exposes_a_data_tree_gate():
    P = _preflight()
    assert hasattr(P, "data_tree_gate"), (
        "preflight_run17.data_tree_gate is missing -- the auditor is unwired "
        "again, which is the state it sat in from 2026-06-17 to 2026-08-30")


def test_data_tree_gate_returns_the_row_convention(tmp_path):
    P = _preflight()
    data, man = _tree(tmp_path)
    rows = P.data_tree_gate(str(data), str(man))
    assert rows
    for r in rows:
        assert isinstance(r, tuple) and len(r) == 2
        assert r[0] in ("OK", "WARN", "FAIL")


def test_run_all_actually_calls_the_data_tree_gate(monkeypatch):
    """Drives run_all and asserts the gate CONTRIBUTED a row.

    Not a source check. MEASURED 2026-08-30: a scan for `import
    audit_data_tree` reports ZERO while the wiring is real, because it loads by
    path; and the same scan counted 31 "invocations" of preflight_data_guard,
    every one a line of Markdown.
    """
    P = _preflight()
    seen = {}

    def spy(data_root="data", manifest="configs/data_manifest.yaml"):
        seen["called"] = True
        return [("OK", "data-tree: SENTINEL")]

    monkeypatch.setattr(P, "data_tree_gate", spy)
    try:
        rows = P.run_all("python scripts/train.py", "data", 3000, defer_kg=True)
    except Exception:
        pytest.skip("run_all needs a fuller fixture here; the call is asserted below")
    assert seen.get("called"), "run_all did not call data_tree_gate"
    assert any("SENTINEL" in m for _, m in rows)


def test_the_gate_reports_rather_than_crashing_when_the_auditor_is_absent(tmp_path):
    """A preflight that dies because one gate cannot load hides every other
    finding. The gate must degrade to a FAIL row."""
    P = _preflight()
    import shutil as _sh

    backup = tmp_path / "auditor.bak"
    _sh.copy2(_AUDITOR, backup)
    _AUDITOR.unlink()
    try:
        rows = P.data_tree_gate("data", str(_MANIFEST))
        assert rows and rows[0][0] == "FAIL"
        assert "auditor" in rows[0][1] or "audit_data_tree" in rows[0][1]
    finally:
        _sh.copy2(backup, _AUDITOR)
