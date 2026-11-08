"""Package 1 transition dispositions: every affected, reachable operation is correct or explicitly refused.
Author: Monzia Moodie

Set 1b (2026-11-08) removed two set-1a refusals TOGETHER with their migrations: scripts/run9_ablations.py now runs on one
bound capture per split and publishes a completion manifest, and RunArtifactWriter.save_shap_values is retired in
favour of the selected predictor's member-score attributions (a feature-level explanation is a standing refusal).

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
from genomic_variant_classifier.prediction_contract import ContractError
from genomic_variant_classifier.transition_policy import PREDICTION_CONTRACT_TRANSITION_DENIED

REPO = Path(__file__).resolve().parents[2]


def test_the_transition_denials_are_exactly_the_two_left_for_set_1c_and_carry_reason_codes():
    assert dict(PREDICTION_CONTRACT_TRANSITION_DENIED).keys() == {
        "scripts/calibrate_thresholds.py", "scripts/export_phase2_predictions_from_saved_models.py"}
    assert PREDICTION_CONTRACT_TRANSITION_DENIED["scripts/export_phase2_predictions_from_saved_models.py"].startswith(
        "SCORE_RECONSTRUCTION_REFUSED (disposition decided in package 1, set 1c)")
    assert PREDICTION_CONTRACT_TRANSITION_DENIED["scripts/calibrate_thresholds.py"].startswith(
        "THRESHOLD_CONSUMER_MIGRATION_REQUIRED (enabled by package 1, set 1c)")
    import genomic_variant_classifier.transition_policy as policy
    assert not hasattr(policy, "EXPLANATION_TARGET_MIGRATION_REQUIRED"), "the 1b migration removes the constant"
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


def test_the_retired_shap_writer_refuses_before_importing_shap_or_writing(tmp_path, monkeypatch):
    """save_shap_values explained ensemble.meta_learner (the logistic research candidate). Set 1b retires it: it refuses
    with the standing feature-level reason before SHAP is imported, before any ensemble is touched, and before any file
    appears; the selected predictor's explanation is save_member_score_attributions."""
    from genomic_variant_classifier.evaluation.blend_attribution import FEATURE_LEVEL_EXPLANATION_UNAVAILABLE
    from genomic_variant_classifier.evaluation.prediction_artifacts import RunArtifactWriter

    monkeypatch.setitem(sys.modules, "shap", None)        # importing shap would raise ImportError, not refuse

    class _Ensemble:
        @property
        def meta_learner(self):
            raise AssertionError("the logistic stacker was touched")

    writer = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    with pytest.raises(ContractError) as refused:
        writer.save_shap_values(_Ensemble(), {"lightgbm": np.zeros(3)}, pd.DataFrame({"variant_id": list("abc")}))
    assert FEATURE_LEVEL_EXPLANATION_UNAVAILABLE in str(refused.value)
    assert "retired" in str(refused.value)
    assert list(tmp_path.iterdir()) == []


def test_the_ablation_harness_is_enabled_and_its_deliverables_are_never_swallowed():
    """Set 1b removes run9_ablations' transition guard together with its migration. Its requested deliverables run
    OUTSIDE any handler: a failure propagates, no completion manifest is written, and no broad `except Exception`
    remains in main() to turn a failure into a log line and a run that looks complete (owner ruling 2026-11-07)."""
    source = (REPO / "scripts/run9_ablations.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "require_execution_permitted" not in source
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    handlers = [h for t in ast.walk(main) if isinstance(t, ast.Try) for h in t.handlers]
    assert handlers == [], [ast.unparse(h.type) for h in handlers if h.type is not None]
    calls = {ast.unparse(n.func) for n in ast.walk(main) if isinstance(n, ast.Call)}
    for needed in ("writer.save_member_score_attributions", "writer.save_permutation_importance",
                   "ensemble.score_split", "compose_oof_identity", "writer.complete", "preflight"):
        assert needed in calls, needed
    assert "writer.save_shap_values" not in calls


def test_the_ablation_harness_help_runs_without_a_refusal(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, str(REPO / "scripts/run9_ablations.py"), "--help"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--skip-attributions" in r.stdout and "Execution denied" not in r.stdout + r.stderr
    assert list(tmp_path.iterdir()) == []


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
