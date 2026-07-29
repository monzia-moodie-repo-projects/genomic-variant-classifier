"""test_evaluator_phase5.py -- Monzia Moodie

Phase 5 regression locks:
  (A) evaluator.py imports cleanly WITHOUT scikit-learn (lazy loader); sklearn loads on first evaluate.
  (B) F1 is a first-class metric: a field on EvaluationReport, computed at the SAME 0.5 threshold as
      MCC, printed in the summary, and present in the compare_models comparison DataFrame.

IMPORTANT -- test isolation:
  The "imports without sklearn" check MUST run in a SUBPROCESS. Blocking sklearn by mutating the
  parent interpreter's sys.modules / builtins.__import__ pollutes module identity for every test that
  runs afterwards: re-imported sklearn classes (e.g. StandardScaler) become NEW objects, so any model
  holding the old class identity fails to pickle with
  "Can't pickle <StandardScaler>: it's not the same object as sklearn...". That corrupts unrelated
  model-serialization tests later in the suite. Running the check in a child process keeps the parent's
  module graph pristine.
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest


EVAL_MOD = "genomic_variant_classifier.evaluation.evaluator"


def _repo_src_on_path() -> str:
    """Return a PYTHONPATH value that lets a subprocess import the package (mirrors the test env)."""
    # The installed/edited package is importable in this process; reuse the same sys.path for the child.
    return os.pathsep.join(p for p in sys.path if p)


def test_module_imports_without_sklearn():
    """Runs in a SUBPROCESS so blocking sklearn cannot pollute this interpreter's sys.modules."""
    code = textwrap.dedent(f"""
        import builtins, importlib
        real = builtins.__import__
        def blk(n, *a, **k):
            if n == "sklearn" or n.startswith("sklearn."):
                raise ModuleNotFoundError("No module named 'sklearn' (blocked for test)")
            return real(n, *a, **k)
        builtins.__import__ = blk
        m = importlib.import_module({EVAL_MOD!r})
        assert hasattr(m, "ClinicalEvaluator"), "ClinicalEvaluator missing"
        assert hasattr(m, "EvaluationReport"), "EvaluationReport missing"
        assert hasattr(m, "_ensure_sklearn"), "_ensure_sklearn missing"
        assert "f1" in m.EvaluationReport.__dataclass_fields__, "F1 field missing"
        print("PHASE5_IMPORT_OK")
    """)
    env = {**os.environ, "PYTHONPATH": _repo_src_on_path()}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert "PHASE5_IMPORT_OK" in r.stdout, (
        "evaluator must import without sklearn.\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )


def test_f1_is_report_field():
    import importlib
    mod = importlib.import_module(EVAL_MOD)
    assert "f1" in mod.EvaluationReport.__dataclass_fields__, "F1 must be a first-class report field"


def test_f1_computed_at_half_threshold():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    import importlib
    mod = importlib.import_module(EVAL_MOD)
    rng = np.random.default_rng(42)
    n = 300
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 + (y - 0.5) * 0.5 + rng.normal(0, 0.2, n), 1e-3, 1 - 1e-3)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rep = mod.ClinicalEvaluator().evaluate(y, p, model_name="t")
    expected = round(sklearn_metrics.f1_score(y, (p >= 0.5).astype(int)), 5)
    assert rep.f1 == expected, "F1 must use the same 0.5 threshold as MCC"


def test_f1_in_compare_models():
    pytest.importorskip("sklearn.metrics")
    import importlib
    mod = importlib.import_module(EVAL_MOD)
    rng = np.random.default_rng(7)
    n = 200
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 + (y - 0.5) * 0.5 + rng.normal(0, 0.2, n), 1e-3, 1 - 1e-3)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        comparison = mod.compare_models(y, {"m": p}, n_bootstrap=20,
                                        output_csv=os.devnull)
    # UPDATED 2026-07-28 (CI-q). `compare_models` now returns a ModelComparison
    # rather than a bare frame, because comparison-level facts -- the population
    # relation, whether the ranking was refused and why -- describe the
    # COMPARISON and cannot be carried by per-model rows without inviting a
    # reader to believe they could differ between rows.
    #
    # The property under test is unchanged: the table must carry the F1 column.
    assert "f1" in comparison.table.columns, (
        "compare_models output must carry the f1 column")
    assert comparison.comparison_rankable is True
    assert comparison.table["rank"].tolist() == [1]
