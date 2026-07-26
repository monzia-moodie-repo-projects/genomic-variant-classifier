"""Tests for the Run-16 preflight's feature-contract check.

WHY THIS FILE EXISTS
--------------------
`scripts/preflight_run16_inputs.py` had five checks. Four were tested by
`tests/unit/test_preflight_run16_inputs.py`; the fifth, the feature-count check,
was not tested at all. It pinned the literal 81 while the contract advanced to
95, so from some point after the 88 bump until 2026-07-20 the gate returned exit
2 on every invocation, on a clean tree. The single untested check was the single
check that drifted.

The repair (2026-07-20) replaced the literal with two properties that cannot go
stale: the fail-loud contract invariant, and membership by NAME. These tests
assert OUTCOMES against constructed contracts rather than reading the source,
which is the only technique that has reliably caught defects in this project.

Placement: tests/unit/test_preflight_run16_feature_contract.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"


def _load_gate(features, count):
    """Import the gate against a synthetic variant_ensemble carrying a chosen contract.

    The real module imports xgboost, lightgbm, catboost and torch at module level,
    which makes it unusable as a test fixture and impossible to vary. Injecting a
    stub into sys.modules BEFORE import lets every branch be exercised in
    milliseconds with no ML dependency, and lets us build contracts that must not
    exist in production (duplicates, missing features).
    """
    for name in [m for m in sys.modules
                 if "variant_ensemble" in m or m == "preflight_run16_inputs"]:
        del sys.modules[name]

    pkg = types.ModuleType("genomic_variant_classifier")
    pkg.__path__ = []
    models = types.ModuleType("genomic_variant_classifier.models")
    models.__path__ = []
    ve = types.ModuleType("genomic_variant_classifier.models.variant_ensemble")
    ve.TABULAR_FEATURES = features
    ve.EXPECTED_TABULAR_FEATURE_COUNT = count
    sys.modules["genomic_variant_classifier"] = pkg
    sys.modules["genomic_variant_classifier.models"] = models
    sys.modules["genomic_variant_classifier.models.variant_ensemble"] = ve

    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    import preflight_run16_inputs as pf  # noqa: E402
    return pf


@pytest.fixture(autouse=True)
def _restore_modules():
    """Leave sys.modules as we found it; a stub leaking into a later test that
    genuinely imports variant_ensemble would be a silent cross-test failure."""
    saved = dict(sys.modules)
    yield
    # Materialise the list BEFORE mutating, and use the loop variable, not the
    # comprehension variable -- which does not leak in Python 3. The first draft
    # of this fixture wrote `del sys.modules[m]` and raised NameError in teardown
    # while all eleven tests still reported PASSED. Recorded because it is the
    # same class of defect the suite exists to catch.
    for name in [mod for mod in sys.modules if mod not in saved]:
        del sys.modules[name]
    sys.modules.update(saved)


_MINIMAL = ("esm2_llr", "maxentscan_delta", "gene_constraint_oe", "af_raw")


def test_consistent_contract_passes():
    pf = _load_gate(_MINIMAL, len(_MINIMAL))
    ok, msg = pf.check_feature_contract()
    assert ok is True, msg
    assert pf.aggregate([(ok, msg)]) == 0


def test_stale_count_fails():
    """The exact defect: the constant lags the list."""
    pf = _load_gate(_MINIMAL, 81)
    ok, msg = pf.check_feature_contract()
    assert ok is False
    assert "81" in msg and str(len(_MINIMAL)) in msg
    assert pf.aggregate([(ok, msg)]) == 2


def test_duplicate_name_fails_even_when_count_matches():
    """A duplicate lets the count agree while a feature is silently lost.

    The pre-2026-07-20 count-only gate PASSED this. That is why the repair
    checks names and uniqueness, not an integer.
    """
    dup = list(_MINIMAL) + ["af_raw"]
    pf = _load_gate(dup, len(dup))
    ok, msg = pf.check_feature_contract()
    assert ok is False
    assert "duplicate" in msg and "af_raw" in msg


@pytest.mark.parametrize("dropped", ["esm2_llr", "maxentscan_delta", "gene_constraint_oe"])
def test_each_required_feature_is_actually_required(dropped):
    """Each name is load-bearing. A check that only ever sees the happy path
    cannot tell whether it is checking anything at all."""
    reduced = tuple(f for f in _MINIMAL if f != dropped)
    pf = _load_gate(reduced, len(reduced))
    ok, msg = pf.check_feature_contract()
    assert ok is False
    assert dropped in msg


def test_multiple_problems_are_all_reported():
    """The gate must not stop at the first problem; an operator fixing one
    failure and re-running should not discover a second on the next pass."""
    broken = list(_MINIMAL[:-1]) + ["esm2_llr"]      # duplicate AND missing af_raw path
    pf = _load_gate(broken, 999)                      # AND a count mismatch
    ok, msg = pf.check_feature_contract()
    assert ok is False
    assert "999" in msg and "duplicate" in msg


def test_import_failure_is_environment_not_failure():
    """Cannot-import must map to exit 3 (environment), never 2 (data failure).
    Conflating them would send an operator hunting a cohort problem that does
    not exist."""
    for name in [m for m in sys.modules
                 if "variant_ensemble" in m or m == "preflight_run16_inputs"]:
        del sys.modules[name]
    sys.modules["genomic_variant_classifier.models.variant_ensemble"] = None
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    import preflight_run16_inputs as pf  # noqa: E402
    ok, msg = pf.check_feature_contract()
    assert ok is None
    assert "ENV" in msg
    assert pf.aggregate([(ok, msg)]) == 3


def test_deprecated_alias_delegates():
    """check_feature_count() was public. It is retained and must agree."""
    pf = _load_gate(_MINIMAL, len(_MINIMAL))
    assert pf.check_feature_count() == pf.check_feature_contract()


def test_no_stale_literal_remains_in_the_gate():
    """A source-level backstop. Weak on its own -- every other test here asserts
    an outcome -- but it pins the specific regression: if anyone re-introduces a
    hard-coded expected count, this fires."""
    src = (_SCRIPTS / "preflight_run16_inputs.py").read_text(encoding="utf-8")
    assert "EXPECTED_COUNT = 81" not in src
    assert "REQUIRED_RUN16_FEATURES" in src


def test_required_features_are_registered_in_the_REAL_contract():
    """The one test that talks to production. Parsed by abstract syntax tree
    rather than imported, so it costs nothing and needs no ML stack."""
    import ast
    src = (_REPO / "src/genomic_variant_classifier/models/variant_ensemble.py").read_text(
        encoding="utf-8")
    real_features = None
    real_count = None
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", None) == "TABULAR_FEATURES":
                    real_features = ast.literal_eval(node.value)
                elif getattr(t, "id", None) == "EXPECTED_TABULAR_FEATURE_COUNT":
                    real_count = ast.literal_eval(node.value)
    assert real_features is not None, "TABULAR_FEATURES not a module-level literal"
    assert real_count == len(real_features), (
        f"contract broken in production: constant={real_count}, list={len(real_features)}")
    assert len(set(real_features)) == len(real_features), "duplicate names in production"

    pf = _load_gate(real_features, real_count)
    ok, msg = pf.check_feature_contract()
    assert ok is True, msg
