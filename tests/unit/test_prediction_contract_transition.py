"""Package 1 transition dispositions: every affected, reachable operation is correct or explicitly refused.
Author: Monzia Moodie

Set 1b (2026-11-08) removed two set-1a refusals TOGETHER with their migrations: scripts/run9_ablations.py now runs on one
bound capture per split and publishes a completion manifest, and RunArtifactWriter.save_shap_values is retired in
favour of the selected predictor's member-score attributions (a feature-level explanation is a standing refusal).

Set 1c (2026-11-08) removes the last two and RETIRES the transition policy module: scripts/calibrate_thresholds.py is
migrated onto a completed run's bound predictions (evaluation/decision_policy.py), and
scripts/export_phase2_predictions_from_saved_models.py -- whose disposition was never a migration -- is denied as a
historical execution (quarantine_policy.HISTORICAL_EXECUTION_DENIED, tested in test_containment_completion.py).

Owner ruling of 2026-11-07 (decision 38f15c7a). Each refusal test makes the forbidden downstream effects observable
-- model-library imports, explanation construction, file creation -- and asserts both the refusal and their absence.
The amended acceptance matrix (PACKAGE1_ACCEPTANCE_MATRIX_2026-11-07_rev2) lists every entrypoint, its state after
1a, 1b and 1c, its refusal point and the test below that demonstrates it.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.containment import ContainmentError
from genomic_variant_classifier.prediction_contract import ContractError

REPO = Path(__file__).resolve().parents[2]


def test_the_transition_policy_is_retired_and_nothing_imports_it():
    """Package 1 closes in set 1c: no transition refusal remains, so the module and every reference to it go. A
    refusal that outlives its migration is a refusal nobody can lift deliberately."""
    assert importlib.util.find_spec("genomic_variant_classifier.transition_policy") is None
    offenders = []
    for root in ("src", "scripts"):
        for path in sorted((REPO / root).rglob("*.py")):
            if "transition_policy" in path.read_text(encoding="utf-8", errors="replace"):
                offenders.append(str(path.relative_to(REPO)))
    assert offenders == []


def test_the_migrated_threshold_workflow_runs_its_help_without_a_refusal_or_a_model_library(tmp_path):
    wrapper = (
        "import runpy, sys\n"
        "sys.argv = [sys.argv[1], '--help']\n"
        "try:\n"
        "    runpy.run_path(sys.argv[0], run_name='__main__')\n"
        "except SystemExit as e:\n"
        "    heavy = [m for m in ('joblib', 'sklearn', 'torch', 'shap', 'xgboost', 'lightgbm') if m in sys.modules]\n"
        "    print('EXIT', e.code, heavy)\n")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-c", wrapper, str(REPO / "scripts/calibrate_thresholds.py")], cwd=tmp_path,
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--run-dir" in r.stdout and r.stdout.rstrip().endswith("EXIT 0 []"), r.stdout
    assert "Execution denied" not in r.stdout + r.stderr
    assert list(tmp_path.iterdir()) == []


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
