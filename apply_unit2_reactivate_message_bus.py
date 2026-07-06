#!/usr/bin/env python3
"""
apply_unit2_reactivate_message_bus.py -- Unit 2 edit-script (D12 reactivation).

Transforms test_message_bus.py from a quarantined custom-harness file into a
proper, collected pytest module that is SAFE to run non-interactively:

  T1  remove the sys.path.insert(_HERE) hack (full-package imports don't need it)
  T2  remove MODULE-LEVEL sys.modules stubbing (the D12 pollution: it leaked a
      MagicMock torch into the whole pytest collection -> broke scipy in 12 files)
  T3  add `import pytest`
  T4  add two autouse fixtures: _isolated_optional_deps (teardown-safe per-test
      sys.modules stubs, replacing T2) and _no_interactive_input (input() guardrail)
  T5  rename every `def _test_*` -> `def test_*`
  T6  remove the custom harness (trackers + _run + TESTS list + main + __main__)
  T7  remove unused imports left by T6 (json, traceback, unittest)
  T8  repoint bare `from orchestrator import Orchestrator` -> full package path
  T9  remove the 5 orphaned `_check_drift` patch clauses (Unit 1 deleted the method)
  T10 add per-test `_require_approval` control to the Group-4 emits test (approach A)

ABORTS (raises) on any drifted/unexpected anchor; never silently no-ops.
Idempotent: re-running on an already-transformed file makes no further change.
"""
from __future__ import annotations
import argparse, ast, io, re, sys


def _read(p): return io.open(p, "r", encoding="utf-8", newline="\n").read()
def _write(p, s): io.open(p, "w", encoding="utf-8", newline="\n").write(s)


def _require(cond, msg):
    if not cond:
        raise SystemExit(f"ABORT (anchor drift): {msg}")


def transform(src: str) -> tuple[str, dict]:
    report = {}
    already = "_isolated_optional_deps" in src and "def _test_" not in src

    # ---- counts BEFORE (dynamic; not hardcoded to 35) ----
    n_test_before = len(re.findall(r"^def _test_", src, flags=re.M))
    n_checkdrift_before = src.count('"_check_drift"')
    n_orch_before = src.count("from orchestrator import Orchestrator")
    report["before"] = dict(def_test=n_test_before, check_drift=n_checkdrift_before, orch=n_orch_before)

    s = src

    # ---------- T1: remove sys.path hack ----------
    # Dash/whitespace-tolerant: the real file's comment uses an em-dash (U+2014).
    # Match the comment-bar + "Path setup" line + bar + the 3 code lines as a unit.
    t1_re = re.compile(
        r"# -{5,}\n"
        r"# Path setup [\u2012-\u2015\-]+ allow running from project root or agent_layer/[ \t]*\n"
        r"# -{5,}\n"
        r"_HERE = Path\(__file__\)\.parent\n"
        r"if str\(_HERE\) not in sys\.path:\n"
        r"    sys\.path\.insert\(0, str\(_HERE\)\)\n"
    )
    if t1_re.search(s):
        s = t1_re.sub("", s, count=1)
    else:
        _require("_HERE = Path(__file__).parent" not in s, "T1 sys.path hack not found and not already removed")

    # ---------- T2: remove module-level stub invocation ----------
    t2_re = re.compile(
        r'if "config" not in sys\.modules:\n'
        r'    sys\.modules\["config"\] = _make_config_stub\(\)\n'
        r'\n'
        r'#[^\n]*\n'                       # the "Stub heavy optional dependencies" comment (any text)
        r'for _mod in \("ewc_utils", "shap", "torch", "feedparser", "requests"\):\n'
        r'    if _mod not in sys\.modules:\n'
        r'        sys\.modules\[_mod\] = MagicMock\(\)\n'
    )
    t2_new = (
        "# NOTE (D12 / INCIDENT_2026-05-26): the module-level `config` stub and the\n"
        "# heavy-optional-dependency stubs formerly injected here are now applied\n"
        "# PER-TEST via the autouse `_isolated_optional_deps` fixture below, which is\n"
        "# teardown-safe. Injecting a MagicMock `torch` at import time leaked into the\n"
        "# whole pytest collection and broke scipy's array-api import in 12 files.\n"
    )
    if t2_re.search(s):
        s = t2_re.sub(lambda _m: t2_new, s, count=1)
    else:
        _require("def _isolated_optional_deps(" in s, "T2 module-level stub block not found and fixture not present")

    # ---------- T3: import pytest ----------
    if "import pytest" not in s:
        anchor = "from unittest.mock import MagicMock, patch\n"
        _require(anchor in s, "T3 unittest.mock import anchor missing")
        s = s.replace(anchor, anchor + "\nimport pytest\n", 1)

    # ---------- T4: add autouse fixtures (after _make_bus) ----------
    if "def _isolated_optional_deps(" not in s:
        anchor = "def _make_bus(state: SharedState) -> MessageBus:\n    return MessageBus(state)\n"
        _require(anchor in s, "T4 _make_bus anchor missing")
        fixtures = (
            anchor
            + '''

@pytest.fixture(autouse=True)
def _isolated_optional_deps(monkeypatch):
    """D12 fix. Replaces the former MODULE-LEVEL sys.modules mutation with
    per-test, teardown-safe stubs. monkeypatch.setitem restores the real modules
    (and real torch) after each test, so nothing leaks into the collection of the
    12 downstream files that import scipy. Reproduces the standalone-passing env
    per test: the Group-4 agent lazily imports `requests` and binds this mock; the
    interpretability/literature tests mock their dep-using methods, so mock
    torch/shap/feedparser are never actually exercised."""
    monkeypatch.setitem(sys.modules, "config", _make_config_stub())
    for _mod in ("ewc_utils", "shap", "torch", "feedparser", "requests"):
        monkeypatch.setitem(sys.modules, _mod, MagicMock())


@pytest.fixture(autouse=True)
def _no_interactive_input(monkeypatch):
    """Guardrail: no test may EVER block on input(). Any unmocked approval prompt
    hard-fails loudly instead of hanging the run (CI has no TTY; a human lost
    hours to exactly this silent prompt)."""
    def _boom(*_a, **_k):
        raise AssertionError(
            "input() called during a test -- an approval gate was not mocked"
        )
    monkeypatch.setattr("builtins.input", _boom)
'''
        )
        s = s.replace(anchor, fixtures, 1)

    # ---------- T9: remove 5 _check_drift patch clauses (before rename; agent var names differ) ----------
    # Site A (compound, keep _run_training + _require_approval)
    siteA_old = (
        '        with patch.object(agent, "_check_drift", return_value=False), patch.object(\n'
        '            agent, "_run_training", return_value="/tmp/checkpoints/model.pt"\n'
        '        ), patch.object(agent, "_require_approval", return_value=True):\n'
    )
    siteA_new = (
        "        with patch.object(\n"
        '            agent, "_run_training", return_value="/tmp/checkpoints/model.pt"\n'
        '        ), patch.object(agent, "_require_approval", return_value=True):\n'
    )
    # Sites B/C (single with -> drop with, dedent body: `result = agent.run(dry_run=False)`)
    siteBC_old = (
        '        with patch.object(agent, "_check_drift", return_value=False):\n'
        "            result = agent.run(dry_run=False)\n"
    )
    siteBC_new = "        result = agent.run(dry_run=False)\n"
    # Site D (single with -> drop with, dedent body: `agent.run(dry_run=False)`)
    siteD_old = (
        '        with patch.object(agent, "_check_drift", return_value=False):\n'
        "            agent.run(dry_run=False)\n"
    )
    siteD_new = "        agent.run(dry_run=False)\n"
    # Site E (compound with train_agent, keep _run_training + _require_approval)
    siteE_old = (
        "        with patch.object(\n"
        '            train_agent, "_check_drift", return_value=False\n'
        "        ), patch.object(\n"
        '            train_agent, "_run_training", return_value="/tmp/checkpoints/chain_test.pt"\n'
        "        ), patch.object(\n"
        '            train_agent, "_require_approval", return_value=True\n'
        "        ):\n"
    )
    siteE_new = (
        "        with patch.object(\n"
        '            train_agent, "_run_training", return_value="/tmp/checkpoints/chain_test.pt"\n'
        "        ), patch.object(\n"
        '            train_agent, "_require_approval", return_value=True\n'
        "        ):\n"
    )
    if '"_check_drift"' in s:
        _require(siteA_old in s, "T9 site A anchor drift")
        s = s.replace(siteA_old, siteA_new, 1)
        cBC = s.count(siteBC_old)
        _require(cBC == 2, f"T9 sites B/C expected 2 occurrences, found {cBC}")
        s = s.replace(siteBC_old, siteBC_new)  # both B and C
        _require(siteD_old in s, "T9 site D anchor drift")
        s = s.replace(siteD_old, siteD_new, 1)
        _require(siteE_old in s, "T9 site E anchor drift")
        s = s.replace(siteE_old, siteE_new, 1)

    # ---------- T10: Group-4 emits test approval control (first ftplib-with = emits) ----------
    if 'patch.object(\n            DataFreshnessAgent, "_require_approval"' not in s:
        ftplib_with = (
            "        with patch(\n"
            '            "genomic_variant_classifier.agent_layer.agents.data_freshness_agent.ftplib"\n'
            "        ) as mock_ftp:\n"
        )
        c = s.count(ftplib_with)
        _require(c == 2, f"T10 expected exactly 2 ftplib-with blocks (emits, dry_run), found {c}")
        ftplib_with_new = (
            "        with patch(\n"
            '            "genomic_variant_classifier.agent_layer.agents.data_freshness_agent.ftplib"\n'
            "        ) as mock_ftp, patch.object(\n"
            '            DataFreshnessAgent, "_require_approval", return_value=True\n'
            "        ):\n"
        )
        s = s.replace(ftplib_with, ftplib_with_new, 1)  # first == emits (precedes dry_run)

    # ---------- T5: rename _test_ -> test_ (defs AND any references) ----------
    s = re.sub(r"\bdef _test_", "def test_", s)

    # ---------- T6: remove the custom harness (Runner comment -> EOF) ----------
    if "TESTS = [" in s or "def main(" in s:
        m = re.search(r"\n# =+\n# Runner\n# =+\n", s)
        _require(m is not None, "T6 Runner section header not found")
        s = s[: m.start()] + "\n"
    # remove tracker block (_PASS/_FAIL/_ERROR/_results/_run) if present
    tracker_re = re.compile(
        r'\n_PASS = "PASS"\n_FAIL = "FAIL"\n_ERROR = "ERROR"\n_results:[^\n]*\n\n\n'
        r"def _run\(group: str, name: str, fn\):\n(?:.*\n)*?            traceback\.print_exc\(\)\n",
        flags=re.M,
    )
    if "def _run(" in s:
        s2 = tracker_re.sub("\n", s)
        _require(s2 != s, "T6 tracker/_run block present but regex failed to remove it")
        s = s2

    # ---------- T7: drop imports made unused by T6 ----------
    for imp in ("import json\n", "import traceback\n", "import unittest\n"):
        if imp in s:
            s = s.replace(imp, "", 1)

    # ---------- T8: repoint bare orchestrator import ----------
    s = s.replace(
        "from orchestrator import Orchestrator",
        "from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator",
    )

    # ---------- POST-CONDITIONS ----------
    _require(re.search(r"^def _test_", s, flags=re.M) is None, "post: a def _test_ remains")
    n_test_after = len(re.findall(r"^def test_", s, flags=re.M))
    n_testalready_before = len(re.findall(r"^def test_", src, flags=re.M))
    expect_after = n_test_before + n_testalready_before
    _require(n_test_after == expect_after,
             f"post: def test_ ({n_test_after}) != expected ({expect_after})")
    _require('"_check_drift"' not in s, "post: a _check_drift reference remains")
    _require("from orchestrator import Orchestrator" not in s, "post: a bare orchestrator import remains")
    _require("def _isolated_optional_deps(" in s, "post: isolation fixture missing")
    _require("def _no_interactive_input(" in s, "post: input-guardrail fixture missing")
    _require("TESTS = [" not in s, "post: TESTS list remains")
    _require("def main(" not in s, "post: main() remains")
    _require("def _run(" not in s, "post: _run harness remains")
    _require('DataFreshnessAgent, "_require_approval"' in s, "post: emits-test approval control missing")
    ast.parse(s)  # must be valid Python

    report["after"] = dict(def_test=n_test_after)
    report["already_applied_input"] = already
    return s, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--in-place", action="store_true")
    a = ap.parse_args()
    src = _read(a.path)
    out, report = transform(src)
    changed = out != src
    if a.in_place and changed:
        _write(a.path, out)
    print(f"REPORT {report}  changed={changed}")
    return out


if __name__ == "__main__":
    main()
