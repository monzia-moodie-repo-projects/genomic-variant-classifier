"""Package 1, set 1a transition dispositions: every affected, reachable operation is correct or explicitly refused.
Author: Monzia Moodie

Owner ruling of 2026-11-07 (decision 38f15c7a). Each refusal test makes the forbidden downstream effects observable
-- model-library imports, explanation construction, file creation -- and asserts both the refusal and their absence.
The amended acceptance matrix (PACKAGE1_ACCEPTANCE_MATRIX_2026-11-07_rev2) lists every entrypoint, its state after
1a, 1b and 1c, its refusal point and the test below that demonstrates it.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.containment import ContainmentError
from genomic_variant_classifier.quarantine_policy import HISTORICAL_EXECUTION_DENIED
from genomic_variant_classifier.transition_policy import (
    EXPLANATION_TARGET_MIGRATION_REQUIRED,
    PREDICTION_CONTRACT_TRANSITION_DENIED,
)

REPO = Path(__file__).resolve().parents[2]


def test_the_transition_denials_are_exactly_the_reviewed_three_and_carry_reason_codes():
    assert dict(PREDICTION_CONTRACT_TRANSITION_DENIED).keys() == {
        "scripts/calibrate_thresholds.py", "scripts/run9_ablations.py",
        "scripts/export_phase2_predictions_from_saved_models.py"}
    assert PREDICTION_CONTRACT_TRANSITION_DENIED["scripts/export_phase2_predictions_from_saved_models.py"].startswith(
        "SCORE_RECONSTRUCTION_REFUSED (disposition decided in package 1, set 1c)")
    assert PREDICTION_CONTRACT_TRANSITION_DENIED["scripts/calibrate_thresholds.py"].startswith(
        "THRESHOLD_CONSUMER_MIGRATION_REQUIRED (enabled by package 1, set 1c)")
    assert PREDICTION_CONTRACT_TRANSITION_DENIED["scripts/run9_ablations.py"].startswith(
        "ABLATION_WRITER_MIGRATION_REQUIRED (enabled by package 1, set 1b)")
    with pytest.raises(TypeError):
        PREDICTION_CONTRACT_TRANSITION_DENIED["scripts/x.py"] = "y"          # read-only policy


def test_the_transition_policy_is_separate_from_the_quarantine_policy():
    """A transition has nothing to do with the quarantine: the quarantine policy (and the digest admission
    decisions bind to) is unchanged by it."""
    assert not set(PREDICTION_CONTRACT_TRANSITION_DENIED) & set(HISTORICAL_EXECUTION_DENIED)
    from genomic_variant_classifier import model_admission
    source = Path(model_admission.__file__).read_text(encoding="utf-8")
    assert "transition_policy" not in source


@pytest.mark.parametrize("script", sorted(PREDICTION_CONTRACT_TRANSITION_DENIED))
def test_the_guard_is_the_first_statement_after_future_imports(script):
    tree = ast.parse((REPO / script).read_text(encoding="utf-8"))
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    body = [n for n in body if not (isinstance(n, ast.ImportFrom) and n.module == "__future__")]
    first = [ast.unparse(n) for n in body[:3]]
    assert first[0].startswith("from genomic_variant_classifier.containment import ")
    assert "require_execution_permitted" in first[0]
    assert first[1] == ("from genomic_variant_classifier.transition_policy import "
                        "PREDICTION_CONTRACT_TRANSITION_DENIED")
    assert first[2] == f"require_execution_permitted({script!r}, PREDICTION_CONTRACT_TRANSITION_DENIED)"


@pytest.mark.parametrize("script", sorted(PREDICTION_CONTRACT_TRANSITION_DENIED))
def test_running_the_script_refuses_before_any_model_library_data_or_output(script, tmp_path):
    wrapper = (
        "import runpy, sys\n"
        "from genomic_variant_classifier.containment import ContainmentError\n"
        "try:\n"
        "    runpy.run_path(sys.argv[1], run_name='__main__')\n"
        "except ContainmentError as e:\n"
        "    heavy = [m for m in ('joblib', 'sklearn', 'torch', 'shap', 'xgboost', 'lightgbm') if m in sys.modules]\n"
        "    print('REFUSED', heavy, str(e))\n"
        "    sys.exit(0)\n"
        "print('NOT REFUSED'); sys.exit(1)\n")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-c", wrapper, str(REPO / script), "--help"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.startswith("REFUSED [] Execution denied: " + script), r.stdout
    assert list(tmp_path.iterdir()) == [], "the refused script created files"


def test_the_threshold_helpers_cannot_be_imported_around_the_refusal():
    sys.modules.pop("calibrate_thresholds", None)
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        with pytest.raises(ContainmentError, match="THRESHOLD_CONSUMER_MIGRATION_REQUIRED"):
            import calibrate_thresholds  # noqa: F401
    finally:
        sys.path.remove(str(REPO / "scripts"))
    assert "calibrate_thresholds" not in sys.modules


def test_the_shap_writer_refuses_before_importing_shap_or_writing(tmp_path, monkeypatch):
    """The writer explained ensemble.meta_learner (the logistic research candidate). It must refuse before SHAP is
    imported, before the stacker is touched, and before any file appears."""
    from genomic_variant_classifier.evaluation.prediction_artifacts import RunArtifactWriter

    monkeypatch.setitem(sys.modules, "shap", None)        # importing shap would raise ImportError, not refuse

    class _Ensemble:
        @property
        def meta_learner(self):
            raise AssertionError("the logistic stacker was touched")

    writer = RunArtifactWriter.__new__(RunArtifactWriter)
    writer.output_dir = tmp_path
    with pytest.raises(ContainmentError) as refused:
        writer.save_shap_values(_Ensemble(), {"lightgbm": np.zeros(3)}, pd.DataFrame({"variant_id": list("abc")}))
    assert str(refused.value) == EXPLANATION_TARGET_MIGRATION_REQUIRED
    assert list(tmp_path.iterdir()) == []


def test_the_ablation_caller_propagates_a_refusal_through_its_broad_handler():
    """run9_ablations wrapped save_shap_values in `except Exception: logger.error(...)`, which would have turned the
    refusal into a log line and a run that looked complete. The ContainmentError clause must come first and
    re-raise. (The script is itself refused at import in set 1a; this guards the code set 1b re-enables.)"""
    tree = ast.parse((REPO / "scripts/run9_ablations.py").read_text(encoding="utf-8"))
    tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try)
             and "save_shap_values" in ast.unparse(n.body[0]) if n.body]
    assert len(tries) == 1, len(tries)
    handlers = tries[0].handlers
    assert ast.unparse(handlers[0].type) == "ContainmentError"
    assert isinstance(handlers[0].body[0], ast.Raise) and handlers[0].body[0].exc is None
    assert ast.unparse(handlers[1].type) == "Exception"


@pytest.mark.parametrize("record", [
    {"pathogenicity_score": 0.9, "classification": "Pathogenic", "confidence": "high", "variant_id": "1:1:A:T"},
    "schema_2_0_0_with_new_fields",
])
def test_a_3_0_0_reader_refuses_other_schemas_instead_of_guessing(record):
    from pydantic import ValidationError

    from genomic_variant_classifier.api.schemas import VariantPrediction
    from tests.unit._prediction_support import FixedProba, blend_pipeline

    good = blend_pipeline({"lightgbm": FixedProba(0.6)}).predict_single({"chrom": "1", "pos": 1, "ref": "A", "alt": "T"})
    VariantPrediction(**good)                                    # the positive control: 3.0.0 is understood
    candidate = dict(good, prediction_schema_version="2.0.0") if isinstance(record, str) else record
    with pytest.raises(ValidationError):
        VariantPrediction(**candidate)
