"""Package 1, set 1c: the remaining consumers of the selected score read BOUND evidence, or refuse.

Author: Monzia Moodie

Frozen acceptance matrix (docs/package1/PACKAGE1_ACCEPTANCE_MATRIX_2026-11-07.md, revision 2), the set-1c rows of G12:
the threshold workflow (scripts/calibrate_thresholds.py), the export metadata (G7.2), the validators, the forensic
consumers (G2.4) and the remaining transition refusals. Inputs are REAL completed runs written by the production writer
from genuinely fitted ensembles (tests/unit/_run_artifact_support.py); every negative test asserts the refusal AND the
absence of what the refusal must prevent.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.evaluation.decision_policy import (
    BAND_ORDER,
    DecisionPolicy,
    DecisionPolicyRefused,
    build_decision_policy,
    load_decision_policy,
    write_decision_policy,
)
from genomic_variant_classifier.evaluation.prediction_artifacts import (
    COMPLETION_MANIFEST,
    SPLIT_PREDICTION_IDENTITY_COLUMNS,
    IncompleteRunError,
    load_completed_split_predictions,
)
from genomic_variant_classifier.prediction_contract import ContractError
from tests.unit._run_artifact_support import completed_run

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def run_with_val(tmp_path_factory):
    return completed_run(tmp_path_factory.mktemp("c1") / "run" / "full", seed=5, n_val=400, with_scaler=True)


def _copy_run(src: Path, dst: Path) -> Path:
    import shutil
    shutil.copytree(src, dst)
    return dst


def _rewrite_listed(run_dir: Path, name: str, data: bytes) -> None:
    """Replace a listed artefact AND its manifest entry consistently, so verify_completed_run passes and only the
    reader's own cross-checks can catch the change."""
    (run_dir / name).write_bytes(data)
    manifest = json.loads((run_dir / COMPLETION_MANIFEST).read_text(encoding="utf-8"))
    manifest["artefacts"][name].update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    (run_dir / COMPLETION_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------------
# The bound reader of a completed run's split predictions
# ---------------------------------------------------------------------------------------------------------------------

def test_the_bound_reader_returns_the_written_predictions_with_their_identities(run_with_val):
    bound = load_completed_split_predictions(run_with_val, "val")
    manifest = json.loads((run_with_val / COMPLETION_MANIFEST).read_text(encoding="utf-8"))
    on_disk = pd.read_parquet(run_with_val / "val_predictions.parquet")
    pd.testing.assert_frame_equal(bound.frame, on_disk)
    assert list(bound.frame.columns[:5]) == list(SPLIT_PREDICTION_IDENTITY_COLUMNS)
    ident = bound.identity()
    assert ident["combiner_sha256"] == manifest["predictor"]["combiner_sha256"]
    assert ident["result_block_sha256"] == manifest["prediction_blocks"]["val"]["result_block_sha256"]
    assert ident["predictions_sha256"] == manifest["artefacts"]["val_predictions.parquet"]["sha256"]
    assert ident["completion_manifest_sha256"] == hashlib.sha256(
        (run_with_val / COMPLETION_MANIFEST).read_bytes()).hexdigest()
    assert bound.scores.dtype == np.float64 and np.array_equal(bound.scores, on_disk["selected_score"].to_numpy())


def test_the_bound_reader_refuses_an_incomplete_run_or_an_absent_split(tmp_path, run_with_val):
    run = _copy_run(run_with_val, tmp_path / "r")
    (run / COMPLETION_MANIFEST).unlink()
    with pytest.raises(IncompleteRunError, match="not complete evidence"):
        load_completed_split_predictions(run, "val")
    with pytest.raises(IncompleteRunError, match="holds no holdout predictions"):
        load_completed_split_predictions(run_with_val, "holdout")
    with pytest.raises(ContractError, match="unknown split role"):
        load_completed_split_predictions(run_with_val, "validation")


def test_changed_prediction_bytes_are_refused(tmp_path, run_with_val):
    run = _copy_run(run_with_val, tmp_path / "r")
    with open(run / "val_predictions.parquet", "ab") as fh:
        fh.write(b"\x00")
    with pytest.raises(IncompleteRunError, match="changed since completion"):
        load_completed_split_predictions(run, "val")


@pytest.mark.parametrize("field", ["split_sha256", "rows_sha256", "n_rows", "result_block_sha256"])
def test_a_block_descriptor_that_disagrees_with_the_manifest_is_refused(tmp_path, run_with_val, field):
    run = _copy_run(run_with_val, tmp_path / "r")
    blocks = json.loads((run / "val_prediction_blocks.json").read_text(encoding="utf-8"))
    blocks[field] = blocks[field] + 1 if field == "n_rows" else "0" * 64
    _rewrite_listed(run, "val_prediction_blocks.json", json.dumps(blocks).encode("utf-8"))
    with pytest.raises(IncompleteRunError, match=f"disagree on {field}"):
        load_completed_split_predictions(run, "val")


def test_relabelled_or_reordered_predictions_are_refused(tmp_path, run_with_val):
    import io

    run = _copy_run(run_with_val, tmp_path / "r")
    frame = pd.read_parquet(run / "val_predictions.parquet")
    flipped = frame.copy()
    flipped.loc[0, "label"] = 1 - flipped.loc[0, "label"]
    buf = io.BytesIO()
    flipped.to_parquet(buf, index=False)
    _rewrite_listed(run, "val_predictions.parquet", buf.getvalue())
    with pytest.raises(IncompleteRunError, match="recorded label digest"):
        load_completed_split_predictions(run, "val")
    cols = list(frame.columns)
    cols[5], cols[6] = cols[6], cols[5]                                   # swap the two member columns
    buf = io.BytesIO()
    frame[cols].to_parquet(buf, index=False)
    _rewrite_listed(run, "val_predictions.parquet", buf.getvalue())
    with pytest.raises(IncompleteRunError, match="followed by the predictor's members"):
        load_completed_split_predictions(run, "val")


# ---------------------------------------------------------------------------------------------------------------------
# The decision policy (replaces calibrate_thresholds' private search)
# ---------------------------------------------------------------------------------------------------------------------

class _Bound:
    """A stand-in for BoundSplitPredictions over constructed arrays (selection tests need exact control)."""

    def __init__(self, scores, labels, *, combiner="a" * 64, role="val"):
        self.scores = np.asarray(scores, dtype=np.float64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.role = role
        self.predictor = {"combiner_sha256": combiner, "members": ["m"], "kind": "convex_probability_blend"}
        self._combiner = combiner

    def identity(self):
        return {"combiner_sha256": self._combiner, "predictions_sha256": "b" * 64, "role": self.role,
                "n_rows": len(self.scores)}


def _separated(seed: int = 3):
    """Negatives in [0, 0.3], positives in [0.7, 1], three errors on each side: the high floors are reachable and the
    two sides' regions do not meet."""
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.uniform(0.0, 0.3, 100), rng.uniform(0.7, 1.0, 100),
                        rng.uniform(0.8, 0.9, 3), rng.uniform(0.1, 0.2, 3)])
    y = np.concatenate([np.zeros(100), np.ones(100), np.zeros(3), np.ones(3)]).astype(np.int64)
    return p, y


def _reference_point(p, y, floor):
    """Independent brute force of Objective A on the positive side: the largest true-positive count among thresholds
    `p >= u` (u an observed score) whose positive predictive value is at least the floor; ties to fewer false
    positives. Returns the threshold."""
    best = None
    for u in np.unique(p):
        calls = p >= u
        tp, fp = int(y[calls].sum()), int((1 - y[calls]).sum())
        if tp / (tp + fp) >= floor and (best is None or (tp, -fp) > (best[1], -best[2])):
            best = (float(u), tp, fp)
    return best[0]


def test_the_policy_selects_objective_a_exactly_on_both_sides():
    p, y = _separated()
    record = build_decision_policy(_Bound(p, y), positive_high=0.95, positive_moderate=0.95, negative_high=0.95,
                                   negative_moderate=0.95)
    points = record["operating_points"]
    assert points["positive_high"]["threshold"] == _reference_point(p, y, 0.95)
    assert points["negative_high"]["threshold"] == _reference_point(1.0 - p, 1 - y, 0.95)
    assert {points[k]["operator"] for k in points} == {">="}
    for name in points:                                   # the stated metrics are the counts of the stated rule
        point, counts = points[name], points[name]["outcome"]["metrics"]["counts"]
        values = p if point["score_space"] == "selected_score" else 1.0 - p
        target = y if point["score_space"] == "selected_score" else 1 - y
        calls = values >= point["threshold"]
        assert counts["true_positive"] == int(target[calls].sum())
        assert counts["false_positive"] == int((1 - target[calls]).sum())
    assert float.fromhex(points["positive_high"]["threshold_binary64_hex"]) == points["positive_high"]["threshold"]


def test_the_policy_is_on_the_served_stage_and_a_monotone_transform_cannot_change_its_calls():
    """Why no calibration stage: thresholds chosen on f(score) for a strictly increasing f make exactly the calls the
    thresholds chosen on the score make."""
    p, y = _separated()
    raw = DecisionPolicy(build_decision_policy(_Bound(p, y), positive_moderate=0.95, negative_moderate=0.95), "x")
    logit = np.log(p) - np.log1p(-p)
    temperature = 1.0 / (1.0 + np.exp(-logit / 1.7))                  # temperature scaling, T = 1.7
    platt = 1.0 / (1.0 + np.exp(-(2.3 * logit - 0.4)))                 # Platt scaling, positive slope
    assert np.array_equal(np.argsort(temperature, kind="stable"), np.argsort(p, kind="stable"))
    for transformed in (temperature, platt):
        other = DecisionPolicy(build_decision_policy(_Bound(transformed, y), positive_moderate=0.95,
                                                     negative_moderate=0.95), "x")
        assert np.array_equal(raw.classify(p), other.classify(transformed))
    assert raw.record["score"]["transform"] == "identity"
    assert raw.record["score"]["stage"] == "selected_convex_blend_positive_probability"


def test_an_unreachable_floor_refuses_with_every_outcome_and_no_fallback():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=300)
    y = (rng.uniform(size=300) < 0.5).astype(np.int64)                # no signal: PPV 0.99 is unreachable
    y[np.argmax(p)] = 0                                                # and the top score is a negative
    with pytest.raises(DecisionPolicyRefused, match="no_candidate_satisfies_positive_predictive_value_floor") as e:
        build_decision_policy(_Bound(p, y), positive_high=0.99, positive_moderate=0.99)
    assert set(e.value.outcomes) == {"positive_high", "positive_moderate", "negative_high", "negative_moderate"}
    assert e.value.outcomes["positive_high"]["outcome"]["status"] != "ok"
    assert e.value.outcomes["positive_high"]["outcome"]["selection"]["status"] == "no_feasible_candidate"


def test_overlapping_regions_are_refused_rather_than_clamped(run_with_val):
    """The old script clamped crossing boundaries silently. A real completed run (AUROC about 0.9) with moderate floors
    of 0.80: the positive tail and the negative tail meet, and the policy refuses instead of choosing a side."""
    bound = load_completed_split_predictions(run_with_val, "val")
    with pytest.raises(DecisionPolicyRefused, match="bands overlap") as e:
        build_decision_policy(bound)
    pos, neg = e.value.outcomes["positive_moderate"], e.value.outcomes["negative_moderate"]
    both = (bound.scores >= pos["threshold"]) & ((1.0 - bound.scores) >= neg["threshold"])
    assert both.any(), "the refusal must describe a real overlap"


@pytest.mark.parametrize("bad", [{"positive_high": 0.8, "positive_moderate": 0.9}, {"negative_high": 0.0},
                                 {"positive_moderate": float("nan")}, {"negative_moderate": True}])
def test_inconsistent_or_invalid_floors_are_refused(bad):
    p, y = _separated()
    with pytest.raises(ContractError):
        build_decision_policy(_Bound(p, y), **bad)


def test_the_record_round_trips_verifies_and_refuses_edits(tmp_path):
    p, y = _separated()
    record = build_decision_policy(_Bound(p, y), positive_moderate=0.95, negative_moderate=0.95)
    path = tmp_path / "policy.json"
    sha = write_decision_policy(record, path)
    policy = load_decision_policy(path)
    assert policy.file_sha256 == sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert policy.record_sha256 == record["record_sha256"]
    bands = policy.classify(p)
    assert set(bands) <= set(BAND_ORDER)
    hi = policy.point("positive_high")
    assert np.array_equal(bands == "positive_high_ppv", p >= hi["threshold"])
    lo = policy.point("negative_high")
    assert np.array_equal(bands == "negative_high_npv", (1.0 - p) >= lo["threshold"])
    policy.require_predictor("a" * 64)
    with pytest.raises(ContractError, match="chosen for predictor"):
        policy.require_predictor("c" * 64)
    edited = json.loads(path.read_text(encoding="utf-8"))
    edited["operating_points"]["positive_high"]["threshold"] = 0.5
    (tmp_path / "edited.json").write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ContractError, match="record_sha256"):
        load_decision_policy(tmp_path / "edited.json")
    with pytest.raises(ContractError, match="finite and in"):
        policy.classify(np.array([0.5, np.nan]))


def test_the_record_names_its_blockers_and_is_not_a_clinical_classification():
    p, y = _separated()
    record = build_decision_policy(_Bound(p, y), positive_moderate=0.95, negative_moderate=0.95)
    assert "threshold_selected_and_evaluated_on_same_population" in record["certification_blockers"]
    assert "post_selection_validation_not_implemented" in record["certification_blockers"]
    text = json.dumps(record)
    for tier in ("Likely pathogenic", "Uncertain significance", "Likely benign"):
        assert tier not in text


# ---------------------------------------------------------------------------------------------------------------------
# scripts/calibrate_thresholds.py: migrated (transition refusal removed) and exercised end to end
# ---------------------------------------------------------------------------------------------------------------------

def _calibrate(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("calibrate_thresholds", None)
    import calibrate_thresholds
    return calibrate_thresholds


def test_the_threshold_cli_writes_a_bound_policy_from_a_completed_run(tmp_path, monkeypatch):
    run = completed_run(tmp_path / "run" / "full", seed=7, n_val=400, leaky=True)
    module = _calibrate(monkeypatch)
    out = tmp_path / "policy.json"
    assert module.main(["--run-dir", str(run), "--output", str(out), "--positive-moderate-floor", "0.99",
                        "--positive-high-floor", "0.99", "--negative-moderate-floor", "0.99",
                        "--negative-high-floor", "0.99"]) == 0
    policy = load_decision_policy(out)
    bound = load_completed_split_predictions(run, "val")
    assert policy.record["fitted_on"]["predictions_sha256"] == bound.predictions_sha256
    assert policy.combiner_sha256 == bound.predictor["combiner_sha256"]


def test_the_threshold_cli_refuses_without_writing(tmp_path, monkeypatch, run_with_val):
    module = _calibrate(monkeypatch)
    out = tmp_path / "policy.json"
    assert module.main(["--run-dir", str(run_with_val), "--output", str(out)]) == 1          # overlap
    assert module.main(["--run-dir", str(tmp_path / "missing"), "--output", str(out)]) == 2  # not a completed run
    assert not out.exists()
    source = (REPO / "scripts/calibrate_thresholds.py").read_text(encoding="utf-8")
    assert "meta_learner" not in source and "require_execution_permitted" not in source


# ---------------------------------------------------------------------------------------------------------------------
# Export metadata bound to the selected predictor (gate G7.2)
# ---------------------------------------------------------------------------------------------------------------------

def _export(monkeypatch, argv):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("export_model", None)
    import export_model
    monkeypatch.setattr(sys, "argv", ["export_model.py", *argv])
    return export_model


def _admit_for_test(monkeypatch, tmp_path, run_dir):
    """Register the run's orchestrator (its fitted columns are the synthetic f0..f4; its roster is the manifest's
    predictor) and allow THIS load with an explicit test authority. Gate C10 refuses every real load; that production
    behaviour is asserted separately below."""
    from genomic_variant_classifier import model_admission
    from tests.unit._admission_support import AllowForTests, register

    manifest = json.loads((run_dir / COMPLETION_MANIFEST).read_text(encoding="utf-8"))
    registry = register(tmp_path / "registry.v1.json", run_dir / "models" / "ensemble.joblib",
                        feature_names=tuple(f"f{i}" for i in range(5)), roster=list(manifest["predictor"]["members"]))
    monkeypatch.setattr(model_admission, "DEFAULT_AUTHORITY", AllowForTests())
    monkeypatch.setattr(model_admission, "repository_registry_path", lambda: registry)
    return registry


def test_the_evidence_is_recomputed_from_bound_predictions_for_one_predictor(run_with_val):
    from sklearn.metrics import roc_auc_score

    from genomic_variant_classifier.evaluation.prediction_artifacts import selected_predictor_evaluation_evidence

    manifest = json.loads((run_with_val / COMPLETION_MANIFEST).read_text(encoding="utf-8"))
    combiner = manifest["predictor"]["combiner_sha256"]
    evidence = selected_predictor_evaluation_evidence(run_with_val, combiner)
    assert evidence["status"] == "bound" and evidence["combiner_sha256"] == combiner
    assert set(evidence["splits"]) == {"val", "test"}
    for role in ("val", "test"):
        frame = pd.read_parquet(run_with_val / f"{role}_predictions.parquet")
        assert evidence["splits"][role]["auroc"] == pytest.approx(roc_auc_score(frame["label"], frame["selected_score"]),
                                                                  abs=1e-12)
        assert evidence["splits"][role]["predictions_sha256"] == manifest["artefacts"][
            f"{role}_predictions.parquet"]["sha256"]
    with pytest.raises(ContractError, match="not evidence about this predictor"):
        selected_predictor_evaluation_evidence(run_with_val, "0" * 64)


def test_the_pipeline_metadata_carries_only_bound_evidence(run_with_val, tmp_path):
    from genomic_variant_classifier.api.pipeline import SERVING_SCOPE, InferencePipeline, PipelineMetadata
    from genomic_variant_classifier.evaluation.prediction_artifacts import selected_predictor_evaluation_evidence
    from tests.unit._run_artifact_support import fit_fast, synthetic_split

    X, y, meta = synthetic_split(300, seed=5, role="train")
    ens = fit_fast(X, y, meta, tmp_path / "models")
    spec = ens.convex_blend_spec(SERVING_SCOPE)
    with pytest.raises(ContractError, match="not this pipeline's"):
        InferencePipeline.from_variant_ensemble(ens, evaluation_evidence={"status": "bound",
                                                                          "combiner_sha256": "0" * 64})
    with pytest.raises(ContractError, match="state its reason"):
        InferencePipeline.from_variant_ensemble(ens, evaluation_evidence={"status": "unavailable", "reason": " "})
    with pytest.raises(ContractError, match="differ from the features this pipeline renders"):
        InferencePipeline.from_variant_ensemble(ens, feature_names=["not", "these"])
    pipe = InferencePipeline.from_variant_ensemble(ens)
    assert pipe.metadata.evaluation_evidence["status"] == "unavailable"
    assert pipe.metadata.fit_input_rows == ens.fit_input_rows_ == 300
    assert pipe.metadata.feature_names == [str(f) for f in pipe._declared_features()]
    for name in ("val_auroc", "n_train"):
        with pytest.raises(ContractError, match="PIPEMETA-1"):
            getattr(pipe.metadata, name)
    legacy = PipelineMetadata()
    legacy.__dict__["val_auroc"] = 0.9847                               # what an older pickle restores
    with pytest.raises(ContractError, match="PIPEMETA-1"):
        legacy.val_auroc
    assert spec.digest == pipe.combiner.digest


def test_export_refuses_an_incomplete_run_and_writes_nothing(tmp_path, monkeypatch, run_with_val, caplog):
    import logging

    run = _copy_run(run_with_val, tmp_path / "r")
    (run / COMPLETION_MANIFEST).unlink()
    out = tmp_path / "pipeline.joblib"
    module = _export(monkeypatch, ["export", "--input", str(run), "--output", str(out), "--skip-smoke-test"])
    monkeypatch.setattr(module, "_load_ensemble_from_run",
                        lambda *a, **k: pytest.fail("an ensemble was opened from an incomplete run"))
    with caplog.at_level(logging.ERROR, logger="export_model"):
        assert module.main() == 1
    assert "not complete evidence" in caplog.text and "deployment candidate" in caplog.text
    assert not out.exists()
    source = (REPO / "scripts/export_model.py").read_text(encoding="utf-8")
    assert "metrics.json" not in source.split('"""', 2)[2] and "val_auroc" not in source.split('"""', 2)[2]


def test_export_is_refused_by_gate_c10_under_the_default_authority(tmp_path, monkeypatch, run_with_val):
    from genomic_variant_classifier.containment import ContainmentError

    out = tmp_path / "pipeline.joblib"
    module = _export(monkeypatch, ["export", "--input", str(run_with_val), "--output", str(out),
                                   "--skip-smoke-test"])
    with pytest.raises(ContainmentError):
        module.main()
    assert not out.exists()


def test_export_writes_a_pipeline_whose_metadata_is_bound_to_its_predictor(tmp_path, monkeypatch, run_with_val):
    from genomic_variant_classifier.api.pipeline import InferencePipeline
    from tests.unit._admission_support import AllowForTests, register

    _admit_for_test(monkeypatch, tmp_path, run_with_val)
    out = tmp_path / "pipeline.joblib"
    module = _export(monkeypatch, ["export", "--input", str(run_with_val), "--output", str(out),
                                   "--skip-smoke-test"])
    assert module.main() == 0
    manifest = json.loads((run_with_val / COMPLETION_MANIFEST).read_text(encoding="utf-8"))
    import joblib as _joblib
    probe = _joblib.load(out)                                     # test-only read of the artefact this test wrote
    registry = register(tmp_path / "serving.v1.json", out, feature_names=probe._declared_features(),
                        roster=list(probe.trained_models), predictor_contract_sha256=probe.predictor_contract_sha256())
    pipe = InferencePipeline.load(out, consumer="test", registry_path=registry, authority=AllowForTests())
    evidence = pipe.metadata.evaluation_evidence
    assert evidence["status"] == "bound"
    assert evidence["combiner_sha256"] == pipe.combiner.digest == manifest["predictor"]["combiner_sha256"]
    assert evidence["completion_manifest_sha256"] == hashlib.sha256(
        (run_with_val / COMPLETION_MANIFEST).read_bytes()).hexdigest()
    assert pipe.metadata.fit_input_rows == 300
    assert pipe.metadata.feature_names == [f"f{i}" for i in range(5)] and pipe.scaler is not None


def test_export_refuses_a_scaler_the_run_did_not_record(tmp_path, monkeypatch, run_with_val):
    run = _copy_run(run_with_val, tmp_path / "r")
    _admit_for_test(monkeypatch, tmp_path, run)
    manifest = json.loads((run / COMPLETION_MANIFEST).read_text(encoding="utf-8"))
    import joblib as _joblib
    from sklearn.preprocessing import StandardScaler
    _joblib.dump(StandardScaler().fit(np.eye(5)), run / "scaler.joblib")       # replaced after completion
    out = tmp_path / "pipeline.joblib"
    module = _export(monkeypatch, ["export", "--input", str(run), "--output", str(out), "--skip-smoke-test"])
    assert module.main() == 1                                                   # verify_completed_run refuses
    assert not out.exists()
    assert manifest["artefacts"]["scaler.joblib"]["origin"] == "external"


# ---------------------------------------------------------------------------------------------------------------------
# Conformal calibration on the selected score of a completed run
# ---------------------------------------------------------------------------------------------------------------------

def test_conformal_calibration_reads_a_completed_run_and_records_its_identity(tmp_path, run_with_val):
    from genomic_variant_classifier.conformal import calibrate as CAL

    res = CAL.calibrate_completed_run(run_with_val, "test")
    bound = load_completed_split_predictions(run_with_val, "test")
    assert res.substrate == {"bound": True, **bound.identity()}
    assert res.n_total == len(bound.frame)
    with pytest.raises(ValueError, match="selected score"):
        CAL.calibrate_completed_run(run_with_val, "test", CAL.CalibrationConfig(score_col="xgboost_prob"))
    broken = _copy_run(run_with_val, tmp_path / "r")
    (broken / COMPLETION_MANIFEST).unlink()
    with pytest.raises(IncompleteRunError):
        CAL.calibrate_completed_run(broken, "test")


def test_the_conformal_command_runs_on_a_completed_run_and_names_it(tmp_path, monkeypatch, run_with_val):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("run_conformal_calibration", None)
    import run_conformal_calibration as R
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--run-dir", str(run_with_val), "--role", "test", "--outdir", str(out)])
    assert R.main() == 0
    summary = json.loads((out / "conformal_calibration_summary.json").read_text(encoding="utf-8"))
    assert summary["substrate"]["bound"] is True
    assert summary["substrate"]["combiner_sha256"] == load_completed_split_predictions(
        run_with_val, "test").predictor["combiner_sha256"]
    assert "bound: run" in (out / "conformal_calibration_report.md").read_text(encoding="utf-8")
    missing = tmp_path / "missing"
    missing.mkdir()
    monkeypatch.setattr(sys, "argv", ["x", "--run-dir", str(missing), "--outdir", str(tmp_path / "o2")])
    assert R.main() == 2
    assert not (tmp_path / "o2").exists()


def test_the_conformal_script_uses_the_finite_sample_order_statistic(monkeypatch):
    """The old `np.quantile(scores, ceil((n+1)(1-alpha))/n)` interpolated between the k-th and (k+1)-th order
    statistics (not the conformal threshold) and, when k > n, returned the LARGEST score instead of +inf -- which
    leaves a new point whose score exceeds every calibration score out of its set (anti-conservative)."""
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("conformal_prediction", None)
    import conformal_prediction as CP

    rng = np.random.default_rng(1)
    y = (rng.uniform(size=37) < 0.4).astype(int)
    p = rng.uniform(size=37)
    scores = np.where(y == 1, 1.0 - p, p)
    k = int(np.ceil((37 + 1) * (1 - 0.1)))
    ordered = np.sort(scores)
    old = float(np.quantile(scores, min(1.0, k / 37)))
    assert ordered[k - 1] < old < ordered[k]                     # the old value sat strictly between order statistics
    assert CP.calibrate(y, p, 0.1) == float(ordered[k - 1])
    old_small_alpha = float(np.quantile(scores, min(1.0, np.ceil(38 * 0.99) / 37)))
    assert old_small_alpha == ordered[-1]                        # k > n: the old form returned the largest score
    assert CP.calibrate(y, p, 0.01) == np.inf                    # the guarantee needs every label in the set
    contains_pathogenic, contains_benign = CP.prediction_set(np.array([0.1, 0.5, 0.95]), 0.2)
    assert contains_pathogenic.tolist() == [False, False, True] and contains_benign.tolist() == [True, False, False]
    assert CP.empirical_coverage(y, p, CP.calibrate(y, p, 0.1)) >= 0.9
    with pytest.raises(ValueError):
        CP.calibrate(y, np.where(np.arange(37) == 0, np.nan, p), 0.1)


def test_no_consumer_indexes_the_one_dimensional_pipeline_score_as_a_matrix():
    """InferencePipeline.predict_proba returns a 1-D array since set 1a. Every module that holds an InferencePipeline
    reads it as one; `[:, 1]` on it raises IndexError (continual_trainer carried one behind its fail-closed guard)."""
    import ast as _ast

    offenders = []
    for path in sorted([*(REPO / "src").rglob("*.py"), *(REPO / "scripts").rglob("*.py")]):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "InferencePipeline" not in text:
            continue
        for node in _ast.walk(_ast.parse(text)):
            if (isinstance(node, _ast.Subscript) and isinstance(node.value, _ast.Call)
                    and isinstance(node.value.func, _ast.Attribute) and node.value.func.attr == "predict_proba"
                    and isinstance(node.slice, _ast.Tuple)):
                receiver = _ast.unparse(node.value.func.value)
                if "pipe" in receiver.lower():
                    offenders.append(f"{path.relative_to(REPO)}:{node.lineno} {_ast.unparse(node)}")
    assert offenders == []


# ---------------------------------------------------------------------------------------------------------------------
# Superseded patchers and the retired alignment check (G2.4): explicit refusals before reading or writing
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("script,word", [("scripts/apply_d1_fixes.py", "SUPERSEDED"),
                                         ("scripts/fix_oof_index.py", "SUPERSEDED"),
                                         ("scripts/forensics/verify_oof_alignment.py", "RETIRED")])
def test_superseded_and_retired_scripts_refuse_before_touching_anything(script, word, tmp_path, monkeypatch):
    import os
    import subprocess

    targets = [REPO / "scripts/run_phase2_eval.py", REPO / "src/genomic_variant_classifier/data/real_data_prep.py"]
    before = [hashlib.sha256(t.read_bytes()).hexdigest() for t in targets]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, str(REPO / script)], cwd=tmp_path, env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 2, r.stdout + r.stderr
    assert word in r.stdout + r.stderr and "_train_row_idx" in r.stdout + r.stderr
    assert list(tmp_path.iterdir()) == []
    assert [hashlib.sha256(t.read_bytes()).hexdigest() for t in targets] == before


# ---------------------------------------------------------------------------------------------------------------------
# Observability: the selected predictor's weights, not a never-written "meta-learner" file
# ---------------------------------------------------------------------------------------------------------------------

def _observability(name: str):
    import importlib.util as _util
    spec = _util.spec_from_file_location(f"{name}_observability_1c", REPO / "scripts" / f"{name}_observability.py")
    module = _util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["run14", "run15", "run17"])
def test_observability_reads_the_selected_blend_from_the_verified_manifest(name, tmp_path, run_with_val):
    module = _observability(name)
    manifest = json.loads((run_with_val / COMPLETION_MANIFEST).read_text(encoding="utf-8"))
    info = module.blend_weights(run_with_val)
    assert info["available"] is True and info["combiner_sha256"] == manifest["predictor"]["combiner_sha256"]
    assert list(info["weights_binary64_hex"].values()) == manifest["predictor"]["weights_binary64_hex"]
    assert all(float(w).hex() == h for w, h in zip(info["weights"].values(), info["weights_binary64_hex"].values()))
    incomplete = _copy_run(run_with_val, tmp_path / "r")
    (incomplete / COMPLETION_MANIFEST).unlink()
    missing = module.blend_weights(incomplete)
    assert missing["available"] is False and "not complete evidence" in missing["reason"]
    source = (REPO / "scripts" / f"{name}_observability.py").read_text(encoding="utf-8")
    assert 'outputs_dir / "meta_learner_coefs.json"' not in source          # the never-written file is not read
    assert "(meta-learner contribution per base model)" not in source         # and the conflation is gone


def test_the_run17_observability_script_is_still_generated_from_run15(tmp_path):
    import subprocess

    out = tmp_path / "gen17.py"
    r = subprocess.run([sys.executable, str(REPO / "scripts/gen_run17_observability.py"),
                        str(REPO / "scripts/run15_observability.py"), str(out)], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    assert out.read_bytes() == (REPO / "scripts/run17_observability.py").read_bytes()


# ---------------------------------------------------------------------------------------------------------------------
# validate_external: no import-time tiers; an explicit policy for THIS predictor, or none
# ---------------------------------------------------------------------------------------------------------------------

def _served_pipeline(tmp_path, monkeypatch):
    """A saved, registered, synthetic-member pipeline whose admission is allowed for this test only."""
    from genomic_variant_classifier import model_admission
    from tests.unit._admission_support import AllowForTests, register
    from tests.unit._prediction_support import FixedProba, blend_pipeline

    pipe = blend_pipeline({"lightgbm": FixedProba(0.7), "xgboost": FixedProba(0.4)})
    path = tmp_path / "pipeline.joblib"
    pipe.save(path)
    registry = register(tmp_path / "registry.v1.json", path, feature_names=pipe._declared_features(),
                        roster=list(pipe.trained_models), predictor_contract_sha256=pipe.predictor_contract_sha256())
    monkeypatch.setattr(model_admission, "DEFAULT_AUTHORITY", AllowForTests())
    monkeypatch.setattr(model_admission, "repository_registry_path", lambda: registry)
    return pipe, path


def _cohort(tmp_path, n=40):
    rng = np.random.default_rng(2)
    cohort = pd.DataFrame({"chrom": ["1"] * n, "pos": np.arange(1, n + 1), "ref": ["A"] * n, "alt": ["G"] * n,
                           "label": (rng.uniform(size=n) < 0.5).astype(int)})
    cohort.loc[0, "label"], cohort.loc[1, "label"] = 0, 1
    path = tmp_path / "cohort.parquet"
    cohort.to_parquet(path, index=False)
    return path


def _validate_external(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("validate_external", None)
    import validate_external
    return validate_external


def test_external_validation_applies_no_threshold_without_a_policy(tmp_path, monkeypatch):
    pipe, path = _served_pipeline(tmp_path, monkeypatch)
    module = _validate_external(monkeypatch)
    out = tmp_path / "out"
    assert module.main(["--pipeline", str(path), "--cohort", str(_cohort(tmp_path)), "--output-dir", str(out)]) == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["decision_policy"]["status"] == "unavailable"
    assert metrics["predictor"]["combiner_sha256"] == pipe.combiner.digest
    assert "pipeline_val_auroc" not in metrics
    predictions = pd.read_parquet(out / "predictions.parquet")
    assert "classification" not in predictions and "decision_band" not in predictions
    roc = pd.read_parquet(out / "roc_curve.parquet")
    assert len(roc) == len(roc.dropna(subset=["fpr", "tpr"]))
    pr = pd.read_parquet(out / "pr_curve.parquet")
    assert pr["threshold"].isna().sum() == 1 and np.isnan(pr["threshold"].iloc[-1])


def test_external_validation_applies_a_policy_only_for_its_own_predictor(tmp_path, monkeypatch):
    pipe, path = _served_pipeline(tmp_path, monkeypatch)
    p, y = _separated()
    own = tmp_path / "own.json"
    write_decision_policy(build_decision_policy(_Bound(p, y, combiner=pipe.combiner.digest), positive_moderate=0.95,
                                                negative_moderate=0.95), own)
    other = tmp_path / "other.json"
    write_decision_policy(build_decision_policy(_Bound(p, y), positive_moderate=0.95, negative_moderate=0.95), other)
    module = _validate_external(monkeypatch)
    cohort = _cohort(tmp_path)
    refused_out = tmp_path / "refused"
    with pytest.raises(ContractError, match="chosen for predictor"):
        module.main(["--pipeline", str(path), "--cohort", str(cohort), "--output-dir", str(refused_out),
                     "--decision-policy", str(other)])
    assert not any(refused_out.iterdir())
    out = tmp_path / "out"
    assert module.main(["--pipeline", str(path), "--cohort", str(cohort), "--output-dir", str(out),
                        "--decision-policy", str(own)]) == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["decision_policy"]["status"] == "applied"
    assert metrics["decision_policy"]["record_sha256"] == load_decision_policy(own).record_sha256
    predictions = pd.read_parquet(out / "predictions.parquet")
    expected = load_decision_policy(own).classify(predictions["score"].to_numpy(dtype=np.float64))
    assert predictions["decision_band"].tolist() == list(expected)
    point = metrics["decision_policy"]["operating_points"]["positive_high"]
    calls = predictions["score"].to_numpy() >= load_decision_policy(own).point("positive_high")["threshold"]
    assert point["counts"]["true_positive"] == int((calls & (predictions["label"].to_numpy() == 1)).sum())


# ---------------------------------------------------------------------------------------------------------------------
# validate_clinvar_temporal: overlap MEASURED (never assumed); undefined rates refused, not 0.0
# ---------------------------------------------------------------------------------------------------------------------

def _temporal(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("validate_clinvar_temporal", None)
    import validate_clinvar_temporal
    return validate_clinvar_temporal


def test_training_overlap_is_measured_and_removed_or_reported_unmeasured(tmp_path, monkeypatch):
    module = _temporal(monkeypatch)
    splits = tmp_path / "splits"
    splits.mkdir()
    pd.DataFrame({"chrom": ["1", "1", "2"], "pos": [10, 20, 30], "ref": ["A", "C", "G"],
                  "alt": ["G", "T", "A"]}).to_parquet(splits / "meta_train.parquet", index=False)
    new = pd.DataFrame({"chrom": ["1", "1", "3", None], "pos": [10, 21, 30, 5], "ref": ["A", "C", "G", "A"],
                        "alt": ["G", "T", "A", "T"], "label": [1, 0, 1, 0]})
    kept, record = module.measure_training_overlap(new, splits)
    assert record["status"] == "measured" and record["n_overlapping_removed"] == 1
    assert record["n_without_a_complete_locus_kept"] == 1 and record["n_training_identities"] == 3
    assert record["training_metadata_sha256"] == hashlib.sha256((splits / "meta_train.parquet").read_bytes()).hexdigest()
    assert kept["pos"].tolist() == [21, 30, 5]
    same, unmeasured = module.measure_training_overlap(new, tmp_path / "absent")
    assert unmeasured["status"] == "unmeasured" and "does NOT exclude training variants" in unmeasured["reason"]
    assert same is new
    import ast as _ast
    source = (REPO / "scripts/validate_clinvar_temporal.py").read_text(encoding="utf-8")
    assert "skipping variant_id dedup" not in source                     # the false claim is no longer emitted
    assert not [n for n in _ast.walk(_ast.parse(source)) if isinstance(n, _ast.Attribute) and n.attr == "val_auroc"]


def test_fixed_reporting_thresholds_refuse_undefined_rates(monkeypatch):
    module = _temporal(monkeypatch)
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.4, 0.6, 0.2])
    above_all = module._threshold_metrics(y, p, 0.9)
    assert above_all["metrics"]["positive_predictive_value"]["status"] == "undefined"
    assert above_all["metrics"]["positive_predictive_value"]["value"] is None
    assert above_all["counts"] == {"true_positive": 0, "false_positive": 0, "false_negative": 2, "true_negative": 2}
    half = module._threshold_metrics(y, p, 0.5)
    assert half["metrics"]["sensitivity"]["value"] == 0.5


# ---------------------------------------------------------------------------------------------------------------------
# validate_lovd_enriched: a model input is never fabricated silently
# ---------------------------------------------------------------------------------------------------------------------

def _lovd_enriched(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("validate_lovd_enriched", None)
    import validate_lovd_enriched
    return validate_lovd_enriched


def test_the_lovd_consequence_guess_is_refused_by_default_and_honest_when_allowed(tmp_path, monkeypatch):
    module = _lovd_enriched(monkeypatch)
    assert module._infer_consequence("1:10:A:G") == "missense_variant"
    assert module._infer_consequence("1:10:ATTT:A") == "inframe_deletion"          # was "frameshift_variant"
    assert module._infer_consequence("1:10:A:ATTTGGC") == "inframe_insertion"
    assert module._infer_consequence("1:10:AT:A") == "frameshift_variant"
    assert module._infer_consequence("1:10:?:A") == "unknown_consequence"           # was "frameshift_variant"
    assert module._infer_consequence("malformed") == "unknown_consequence"          # was "missense_variant"
    cohort = tmp_path / "lovd.parquet"
    pd.DataFrame({"variant_id": ["1:10:A:G", "1:11:AT:A"], "chrom": ["1", "1"], "pos": [10, 11], "ref": ["A", "AT"],
                  "alt": ["G", "A"], "gene_symbol": ["G1", "G1"], "label": [1, 0]}).to_parquet(cohort, index=False)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--cohort", str(cohort), "--output-dir", str(out),
                                      "--model", str(tmp_path / "never_loaded.joblib")])
    assert module.main() == 2
    assert not out.exists()
    source = (REPO / "scripts/validate_lovd_enriched.py").read_text(encoding="utf-8")
    assert 'Path("G:/My Drive' not in source


# ---------------------------------------------------------------------------------------------------------------------
# build_lovd_index: the label mapping, the client identity, the calibration kernel
# ---------------------------------------------------------------------------------------------------------------------

def _lovd_index(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("build_lovd_index", None)
    import build_lovd_index
    return build_lovd_index


def test_every_lovd_term_maps_to_its_own_class_and_nothing_is_guessed(monkeypatch):
    """Measured 2026-11-08 on the shipped code: "not pathogenic" and "probably not pathogenic" were labelled 1
    (pathogenic) and "Class 10" was labelled 0 -- substring matching with the pathogenic terms checked first."""
    module = _lovd_index(monkeypatch)
    for term in module._PATH_TERMS:
        assert module._lovd_label(term) == 1, term
        assert module._lovd_label(term.upper() + "  (recessive)") == 1, term
    for term in module._BENIGN_TERMS:
        assert module._lovd_label(term) == 0, term
    for unlabelled in ("VUS", "uncertain significance", "Class 10", "class 3", "+?", "-?", "pathogenicity unknown",
                       "not pathogenic, but", ""):
        assert module._lovd_label(unlabelled) is None, unlabelled


def test_the_lovd_client_names_itself_and_uses_the_calibration_kernel(monkeypatch):
    module = _lovd_index(monkeypatch)
    source = (REPO / "scripts/build_lovd_index.py").read_text(encoding="utf-8")
    assert "Mozilla" not in source and "Chrome/" not in source
    assert module._CLIENT_USER_AGENT.startswith("genomic-variant-classifier/")
    assert "calibration_curve(" not in source and "expected_calibration_error(" in source
    assert "except Exception:\n        pass" not in source
