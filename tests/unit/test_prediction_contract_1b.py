"""Package 1, set 1b: original out-of-fold identity, the completion manifest, and writers that consume bound captures.

Author: Monzia Moodie

Frozen acceptance matrix (docs/package1/PACKAGE1_ACCEPTANCE_MATRIX_2026-11-07.md, revision 2): gates G2 (original row
identity through training artifacts), G8 (publication) and the set-1b rows of G12 (writers consume bound predictions and
name their target). Every fixture is a GENUINELY fitted ensemble (tests/unit/_run_artifact_support.py); every negative
test asserts the refusal AND the absence of what a refusal must prevent (a file, a manifest, a library import).
"""
from __future__ import annotations

import io
import json
import math
import pickle
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from genomic_variant_classifier.evaluation.oof_identity import (
    OOF_IDENTITY_COLUMNS,
    OOFIdentity,
    compose_oof_identity,
)
from genomic_variant_classifier.evaluation.prediction_artifacts import (
    COMPLETION_MANIFEST,
    IncompleteRunError,
    RunArtifactWriter,
    calibration_points,
    verify_completed_run,
)
from genomic_variant_classifier.evaluation.serialization import (
    NonFiniteArtifactValue,
    UnserializableArtifactValue,
)
from genomic_variant_classifier.evaluation.split_identity import (
    OOF_UNIT_CONTEXT_SHA256,
    SPLIT_UNIT_CONTEXT_SHA256,
    split_identity,
    split_observations,
)
from genomic_variant_classifier.inert_pickle import InertObject, InertPickleError, read_inert, read_inert_bytes
from genomic_variant_classifier.models.variant_ensemble import (
    HISTORICAL_OOF_INDEX_SIDECAR_SEMANTICS,
    OOF_INDEX_SIDECAR_SEMANTICS,
    SELECTED_ENSEMBLE_ROW,
    read_member_oof_checkpoint,
)
from genomic_variant_classifier.prediction_contract import ContractError, canonical_digest
from tests.unit._run_artifact_support import (
    FAST_MEMBERS,
    completed_run,
    fit_fast,
    install_fast_roster,
    synthetic_split,
)

REPO = Path(__file__).resolve().parents[2]
EPS = np.finfo(np.float64).eps


# ---------------------------------------------------------------------------------------------------------------------
# Shared fitted fixtures (module scope: fitted once, never mutated by a test)
# ---------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_frames():
    return synthetic_split(300, seed=11, role="train")


@pytest.fixture(scope="module")
def fitted(tmp_path_factory, train_frames):
    X, y, meta = train_frames
    model_dir = tmp_path_factory.mktemp("fitted") / "models"
    ens = fit_fast(X, y, meta, model_dir)
    return ens, model_dir


@pytest.fixture(scope="module")
def captures(fitted):
    ens, _ = fitted
    out = {}
    for role, seed, n in (("test", 21, 180), ("val", 31, 140)):
        X, y, meta = synthetic_split(n, seed=seed, role=role)
        ident, vids = split_identity(X, y, meta, role)
        rows = split_observations(ident, vids, unit_context_sha256=SPLIT_UNIT_CONTEXT_SHA256)
        out[role] = (ens.score_split(X, None, rows, split_sha256=ident.prediction_split_sha256), X, y, meta, ident)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The inert reader (G2.3 derivation tool): data out, no code run
# ---------------------------------------------------------------------------------------------------------------------

_SIDE_EFFECTS: list = []


def _side_effect(*args):
    _SIDE_EFFECTS.append(args)
    return "constructed"


class _Hostile:
    def __reduce__(self):
        return (_side_effect, ("ran",))


def test_the_inert_reader_returns_plain_data_and_read_only_arrays(tmp_path):
    payload = {"idx": np.arange(7, dtype=np.int64), "w": np.array([0.25, 0.75]), "flags": np.array([True, False]),
               "fortran": np.asfortranarray(np.arange(6.0).reshape(2, 3)), "names": ["a", "b"],
               "nested": {"t": (1, 2.5, None)}, "u8": np.arange(3, dtype=np.uint8)}
    path = tmp_path / "x.joblib"
    joblib.dump(payload, path)
    got = read_inert(path)
    for key in ("idx", "w", "flags", "fortran", "u8"):
        assert isinstance(got[key], np.ndarray) and got[key].dtype == payload[key].dtype
        assert np.array_equal(got[key], payload[key]) and not got[key].flags.writeable
    assert got["names"] == ["a", "b"] and got["nested"] == {"t": (1, 2.5, None)}


def test_the_inert_reader_never_calls_what_the_stream_names(tmp_path):
    """A pickle whose REDUCE names a callable: pickle.loads runs it; the inert reader records it and runs nothing."""
    data = pickle.dumps({"x": _Hostile()}, protocol=4)
    _SIDE_EFFECTS.clear()
    assert pickle.loads(data)["x"] == "constructed" and _SIDE_EFFECTS == [("ran",)]   # the positive control
    _SIDE_EFFECTS.clear()
    got = read_inert_bytes(data)["x"]
    assert _SIDE_EFFECTS == []
    assert isinstance(got, InertObject) and got.name == "_side_effect" and got.args == ("ran",)


def test_the_inert_reader_never_imports_the_module_a_class_lives_in(tmp_path):
    mod = types.ModuleType("gvc_vanishing_module_1b")
    exec("class Thing:\n    def __init__(self):\n        self.v = 3\n", mod.__dict__)
    mod.Thing.__module__ = "gvc_vanishing_module_1b"
    sys.modules["gvc_vanishing_module_1b"] = mod
    try:
        data = pickle.dumps(mod.Thing(), protocol=4)
    finally:
        del sys.modules["gvc_vanishing_module_1b"]
    with pytest.raises(ModuleNotFoundError):
        pickle.loads(data)                                          # the positive control: a real load needs the module
    got = read_inert_bytes(data)
    assert isinstance(got, InertObject) and got.qualname == "gvc_vanishing_module_1b.Thing" and got.state == {"v": 3}
    assert "gvc_vanishing_module_1b" not in sys.modules


def test_an_object_array_stays_an_inert_record(tmp_path):
    path = tmp_path / "o.joblib"
    joblib.dump({"o": np.array(["a", None, 3], dtype=object), "after": np.arange(2)}, path)
    got = read_inert(path)
    assert not isinstance(got["o"], np.ndarray)
    assert np.array_equal(got["after"], np.arange(2))               # the stream stayed aligned after the nested pickle


@pytest.mark.parametrize("case", ["compressed", "trailing", "persistent", "structured"])
def test_the_inert_reader_refuses_what_it_cannot_read_inertly(tmp_path, case):
    path = tmp_path / "x.joblib"
    if case == "compressed":
        joblib.dump({"a": np.arange(3)}, path, compress=3)
        with pytest.raises(InertPickleError, match="compressed"):
            read_inert(path)
    elif case == "trailing":
        with pytest.raises(InertPickleError, match="trailing"):
            read_inert_bytes(pickle.dumps({"a": 1}, protocol=4) + b"\x00")
    elif case == "persistent":
        buf = io.BytesIO()

        class _P(pickle.Pickler):
            def persistent_id(self, obj):
                return "ext" if obj == "external" else None

        _P(buf, protocol=4).dump({"a": "external"})
        with pytest.raises(InertPickleError, match="persistent id"):
            read_inert_bytes(buf.getvalue())
    else:
        joblib.dump({"s": np.zeros(2, dtype=[("a", "<i4"), ("b", "<f8")])}, path)
        with pytest.raises(InertPickleError):
            read_inert(path)


def test_the_inert_reader_recovers_a_real_orchestrator_without_a_model_load(fitted, tmp_path):
    ens, _ = fitted
    path = tmp_path / "ensemble.joblib"
    ens.save(path)
    got = read_inert(path)
    assert np.array_equal(got["oof_fit_indices_"], ens.oof_fit_indices_)
    assert np.array_equal(got["oof_predictions_"], ens.oof_predictions_)
    assert np.array_equal(got["oof_fold_"], ens.oof_fold_)
    assert got["oof_model_names_"] == list(ens.oof_model_names_)
    assert isinstance(got["meta_learner"], InertObject)             # a model is never materialised


# ---------------------------------------------------------------------------------------------------------------------
# Folds and the per-member sidecar (G2, matrix set-1b list)
# ---------------------------------------------------------------------------------------------------------------------

def test_every_fit_row_has_exactly_one_recorded_fold_and_gene_groups_are_disjoint(fitted, train_frames):
    ens, _ = fitted
    _, y, meta = train_frames
    folds = ens.oof_fold_
    assert folds.shape == (len(ens.oof_fit_indices_),) and folds.min() == 0 and folds.max() == 2
    genes = meta["gene_symbol"].to_numpy()[ens.oof_fit_indices_]
    for g in set(genes.tolist()):
        assert len(set(folds[genes == g].tolist())) == 1, f"{g} spans folds"
    assert ens.oof_cv_scheme_["kind"] == "GroupKFold"
    assert ens.fit_input_rows_ == len(y)
    assert np.array_equal(ens.oof_labels_, y.to_numpy()[ens.oof_fit_indices_])


def test_without_gene_groups_the_recorded_folds_are_the_declared_stratified_splits(train_frames):
    X, y, meta = train_frames
    ens = fit_fast(X, y, meta, None, gene_groups=False)
    scheme = ens.oof_cv_scheme_
    assert scheme == {"kind": "StratifiedKFold", "n_splits": 3, "shuffle": True, "random_state": 42,
                      "stratified_on": "label"}
    expected = np.empty(len(ens.oof_labels_), dtype=np.int64)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for k, (_tr, va) in enumerate(splitter.split(np.zeros(len(expected)), ens.oof_labels_)):
        expected[va] = k
    assert np.array_equal(ens.oof_fold_, expected)


def test_the_member_sidecar_is_aligned_with_its_oof_vector(fitted):
    ens, model_dir = fitted
    for j, name in enumerate(FAST_MEMBERS):
        ckpt = read_member_oof_checkpoint(model_dir, name)
        assert ckpt["semantics"] == OOF_INDEX_SIDECAR_SEMANTICS
        assert np.array_equal(ckpt["fit_input_positions"], ens.oof_fit_indices_)
        assert np.array_equal(ckpt["folds"], ens.oof_fold_)
        assert np.array_equal(ckpt["oof"], ens.oof_predictions_[:, j])


def test_a_historical_sidecar_is_refused_with_its_meaning(fitted, tmp_path):
    _, model_dir = fitted
    name = FAST_MEMBERS[0]
    for suffix in ("_oof.npy", "_oof_indices.npy"):
        (tmp_path / f"{name}{suffix}").write_bytes((model_dir / f"{name}{suffix}").read_bytes())
    meta = json.loads((model_dir / f"{name}_meta.json").read_text())
    for key in ("oof_indices_semantics", "oof_folds_file", "oof_cv_scheme"):
        meta.pop(key)
    (tmp_path / f"{name}_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ContractError, match=HISTORICAL_OOF_INDEX_SIDECAR_SEMANTICS):
        read_member_oof_checkpoint(tmp_path, name)


# ---------------------------------------------------------------------------------------------------------------------
# Split identity (G1 on training artifacts)
# ---------------------------------------------------------------------------------------------------------------------

def test_the_split_dataset_digest_binds_every_part_of_the_split(train_frames):
    X, y, meta = train_frames
    base, _ = split_identity(X, y, meta, "train")
    X2 = X.copy()
    X2.iloc[5, 2] = X2.iloc[5, 2] + 1e-12
    y2 = y.copy()
    y2.iloc[0] = 1 - y2.iloc[0]
    meta2 = meta.copy()
    meta2.loc[3, "alt"] = "T"
    variants = [split_identity(X2, y, meta, "train")[0], split_identity(X, y2, meta, "train")[0],
                split_identity(X, y, meta2, "train")[0], split_identity(X, y, meta, "test")[0]]
    assert len({base.dataset_sha256, *(v.dataset_sha256 for v in variants)}) == 5


def test_unit_contexts_separate_predictions_of_the_same_source_row(train_frames):
    X, y, meta = train_frames
    ident, vids = split_identity(X, y, meta, "train")
    oof = split_observations(ident, vids, unit_context_sha256=OOF_UNIT_CONTEXT_SHA256, positions=[7, 2])
    held = split_observations(ident, vids, unit_context_sha256=SPLIT_UNIT_CONTEXT_SHA256, positions=[7, 2])
    assert [r.source_row_key for r in oof.rows] == ["train_split_position:7", "train_split_position:2"]
    assert [r.source_observation_id for r in oof.rows] == [r.source_observation_id for r in held.rows]
    assert not {r.prediction_unit_id for r in oof.rows} & {r.prediction_unit_id for r in held.rows}


def test_split_identity_refuses_ambiguous_inputs(train_frames):
    X, y, meta = train_frames
    with pytest.raises(ContractError, match="integer labels"):
        split_identity(X, y.astype(bool), meta, "train")
    with pytest.raises(ContractError, match="locus"):
        split_identity(X, y, meta.drop(columns=["alt"]), "train")
    with pytest.raises(ContractError, match="role"):
        split_identity(X, y, meta, "training")
    gappy = meta.copy()
    gappy["pos"] = gappy["pos"].astype(float)
    gappy.loc[4, "pos"] = np.nan
    _, vids = split_identity(X, y, gappy, "train")
    assert vids[4] == "unidentified_variant" and vids[5] == f"1:{int(meta.loc[5, 'pos'])}:A:G"


# ---------------------------------------------------------------------------------------------------------------------
# Composing the original OOF identity (G2.1, G2.2)
# ---------------------------------------------------------------------------------------------------------------------

def test_the_oof_table_carries_verified_training_split_identity(fitted, train_frames):
    ens, _ = fitted
    X, y, meta = train_frames
    ident, vids = split_identity(X, y, meta, "train")
    oof = compose_oof_identity(ens, train=ident, vids=vids, labels=y, meta_train=meta)
    assert isinstance(oof, OOFIdentity)
    assert list(oof.frame.columns) == list(OOF_IDENTITY_COLUMNS) + [f"{m}_prob" for m in FAST_MEMBERS]
    assert not any("blend" in c or c == "ensemble_prob" for c in oof.frame.columns)
    pos = oof.frame["train_split_position"].to_numpy()
    assert np.array_equal(pos, ens.oof_fit_indices_)
    assert np.array_equal(oof.frame["label"].to_numpy(), y.to_numpy()[pos])
    assert oof.frame["variant_id"].tolist() == meta["variant_id"].to_numpy()[pos].tolist()
    assert oof.evidence["labels_verified_against_fit_labels"] == len(pos)
    stripped = {k: v for k, v in oof.evidence.items() if k != "identity_sha256"}
    assert oof.evidence["identity_sha256"] == canonical_digest("gvc.oof-identity-evidence.v1", stripped)


def test_a_subsample_is_composed_through_and_verified(train_frames):
    X, y, meta = train_frames
    sub = np.random.RandomState(3).choice(len(y), size=220, replace=False)      # deliberately NOT sorted
    ens = fit_fast(X.iloc[sub].reset_index(drop=True), y.iloc[sub].reset_index(drop=True),
                   meta.iloc[sub].reset_index(drop=True), None)
    ident, vids = split_identity(X, y, meta, "train")
    oof = compose_oof_identity(ens, train=ident, vids=vids, labels=y, meta_train=meta, subsample_positions=sub)
    pos = oof.frame["train_split_position"].to_numpy()
    assert np.array_equal(pos, sub[ens.oof_fit_indices_])
    assert np.array_equal(oof.frame["label"].to_numpy(), y.to_numpy()[pos])
    assert oof.evidence["subsample"]["applied"] is True
    with pytest.raises(ContractError, match="subsample positions are required"):
        compose_oof_identity(ens, train=ident, vids=vids, labels=y, meta_train=meta)
    wrong = np.random.RandomState(4).permutation(sub)
    with pytest.raises(ContractError, match="identity binding FAILED"):
        compose_oof_identity(ens, train=ident, vids=vids, labels=y, meta_train=meta, subsample_positions=wrong)


def test_an_ensemble_without_recorded_oof_identity_is_refused_not_guessed(fitted, train_frames):
    ens, _ = fitted
    X, y, meta = train_frames
    ident, vids = split_identity(X, y, meta, "train")
    legacy = types.SimpleNamespace(**{k: getattr(ens, k) for k in (
        "oof_predictions_", "oof_fit_indices_", "oof_model_names_", "fit_input_rows_", "oof_labels_",
        "oof_cv_scheme_")}, oof_fold_=None)
    with pytest.raises(ContractError, match="never a sequential index"):
        compose_oof_identity(legacy, train=ident, vids=vids, labels=y, meta_train=meta)


def test_the_writer_refuses_a_bare_oof_frame(tmp_path):
    writer = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    frame = pd.DataFrame({"variant_id": ["v"], "fold": [0], "label": [1], "xgboost_prob": [0.5]})
    with pytest.raises(ContractError, match="OOFIdentity"):
        writer.save_oof_predictions(frame)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------------------------------------------------
# One bound capture per split (G4.4 offline; G7 writers consume it)
# ---------------------------------------------------------------------------------------------------------------------

def test_the_split_capture_is_the_selected_predictor_bound_to_its_member_blocks(fitted, captures):
    ens, _ = fitted
    scored, X, y, _meta, ident = captures["test"]
    assert np.array_equal(scored.scores(), ens.predict_proba(X)[:, 1])
    assert scored.result.binding.ordered_input_sha256 == canonical_digest(
        "gvc.combiner-input-blocks.v1", [b.digest for b in scored.members])
    assert {b.rows.digest for b in scored.members} == {scored.result.rows.digest}
    assert {b.binding.split_sha256 for b in scored.members} == {ident.prediction_split_sha256}
    pd.testing.assert_frame_equal(ens.evaluate_scored(scored, y), ens.evaluate(X, None, y))
    assert SELECTED_ENSEMBLE_ROW in ens.evaluate_scored(scored, y).index


def test_the_split_capture_refuses_rows_that_do_not_match(fitted, captures):
    ens, _ = fitted
    scored, X, _y, _meta, ident = captures["test"]
    shorter = split_observations(ident, [r.variant_identity for r in scored.result.rows.rows],
                                 unit_context_sha256=SPLIT_UNIT_CONTEXT_SHA256, positions=list(range(10)))
    with pytest.raises(ContractError, match="one prediction unit per split row"):
        ens.score_split(X, None, shorter, split_sha256=ident.prediction_split_sha256)


class _Untouchable:
    """A picklable member that must never be called."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        raise AssertionError("a member was called")


def test_a_sequence_member_without_windows_is_refused_before_any_member_runs(fitted, captures):
    """The probe is a VALID declared blend (weights, order and identities consistent), so the refusal it meets is the
    sequence-capability one, not a weight or order refusal that would make this test pass for the wrong reason."""
    import copy

    ens, _ = fitted
    probe = copy.copy(ens)
    names = ["cnn_1d", *ens.trained_models_]
    probe.trained_models_ = {name: _Untouchable() for name in names}
    probe.feature_names_ = names
    probe.blend_weights_ = np.array([0.25, 0.25, 0.5])
    probe.member_artifact_sha256_ = {}
    probe._require_declared_blend()                      # the positive control: the declared blend itself is valid
    scored, X, _y, _meta, ident = captures["test"]
    with pytest.raises(ValueError, match="received None"):
        probe.score_split(X, None, scored.result.rows, split_sha256=ident.prediction_split_sha256)


# ---------------------------------------------------------------------------------------------------------------------
# Writers of predictions, calibration and explanation
# ---------------------------------------------------------------------------------------------------------------------

def test_bare_arrays_are_refused_by_the_test_prediction_writer(tmp_path):
    writer = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    n = 4
    with pytest.raises(ContractError, match="bare arrays"):
        writer.save_test_predictions(np.zeros(n), np.zeros(n), {"x": np.zeros(n)}, pd.DataFrame(index=range(n)))
    with pytest.raises(ContractError, match="bare arrays"):
        writer.save_test_predictions(y_test=np.zeros(n), proba=np.zeros(n), base_probs={}, meta=pd.DataFrame())
    assert list(tmp_path.iterdir()) == []


def test_split_predictions_come_from_one_capture(tmp_path, captures):
    scored, _X, y, meta, _ident = captures["test"]
    writer = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    writer.save_split_predictions(scored, y, meta, role="test")
    df = pd.read_parquet(tmp_path / "test_predictions.parquet")
    assert df["source_observation_id"].tolist() == [r.source_observation_id for r in scored.result.rows.rows]
    assert np.array_equal(df["selected_score"].to_numpy(), scored.scores())
    for name, block in zip(scored.member_names, scored.members):
        assert np.array_equal(df[f"{name}_prob"].to_numpy(), block.positive_probability())
    blocks = json.loads((tmp_path / "test_prediction_blocks.json").read_text())
    assert blocks["result_block_sha256"] == scored.result.digest


def test_calibration_points_reproduce_scikit_learn_and_keep_the_counts(captures):
    scored, _X, y, _meta, _ident = captures["test"]
    p = scored.scores()
    pts = calibration_points(p, y.to_numpy(), n_bins=10)
    true, pred = calibration_curve(y.to_numpy(), p, n_bins=10, strategy="quantile")
    nonempty = pts[pts["n"] > 0]
    assert nonempty["fraction_positive"].tolist() == true.tolist()
    assert nonempty["mean_predicted"].tolist() == pred.tolist()
    assert int(pts["n"].sum()) == len(p) and int(pts["n_positive"].sum()) == int(y.sum())
    tied = calibration_points(np.repeat([0.2, 0.8], [50, 10]), np.r_[np.zeros(50, int), np.ones(10, int)])
    empty = tied[tied["n"] == 0]
    assert len(empty) and empty["mean_predicted"].isna().all()       # empty bins are kept, never dropped silently


def test_the_calibration_writer_separates_points_from_consequence_performance(tmp_path, captures):
    scored, _X, y, meta, _ident = captures["test"]
    p = scored.scores()
    true, pred = calibration_curve(y.to_numpy(), p, n_bins=10, strategy="quantile")
    report = types.SimpleNamespace(calibration_frac_pos=true.tolist(), calibration_mean_pred=pred.tolist(),
                                   consequence_breakdown=[{"consequence": "missense", "n_total": 10,
                                                           "n_pathogenic": 4, "auroc": 0.8, "auprc": 0.7,
                                                           "prevalence": 0.4}])
    writer = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    writer.save_calibration(scored, y, report=report, meta=meta)
    points = pd.read_parquet(tmp_path / "calibration_points.parquet")
    perf = pd.read_parquet(tmp_path / "consequence_performance.parquet")
    assert {"global", "consequence:missense_variant", "consequence:stop_gained"} == set(points["scope"])
    assert (points.loc[points["scope"] == "global", "n"] >= 0).all()
    assert list(perf.columns) == ["consequence", "n_total", "n_pathogenic", "auroc", "auprc", "prevalence"]
    assert perf.loc[0, "auroc"] == 0.8 and "mean_pred" not in perf.columns and "frac_pos" not in perf.columns
    other = RunArtifactWriter(run_id="r", ablation="b", output_dir=tmp_path / "other")
    bad = types.SimpleNamespace(**{**vars(report), "calibration_mean_pred": [v + 1e-9 for v in pred.tolist()]})
    with pytest.raises(ContractError, match="other scores"):
        other.save_calibration(scored, y, report=bad, meta=meta)
    assert not (tmp_path / "other" / "calibration_points.parquet").exists()


def test_member_score_attributions_close_on_the_published_score(tmp_path, captures):
    scored, *_ = captures["test"]
    background, *_ = captures["val"]
    writer = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    writer.save_member_score_attributions(scored, background=background)
    frame = pd.read_parquet(tmp_path / "member_score_attributions.parquet")
    desc = json.loads((tmp_path / "member_score_attributions.json").read_text())
    phi = frame[[f"phi_{m}" for m in scored.member_names]].to_numpy()
    base = float.fromhex(desc["base_value_binary64_hex"])
    recomputed = [math.fsum([q, -base, *(-v for v in row)]) for q, row in zip(scored.scores().tolist(), phi.tolist())]
    assert recomputed == frame["closure_residual"].tolist()
    assert max(abs(r) for r in recomputed) <= 8 * EPS
    mu = np.array([float.fromhex(h) for h in desc["background"]["member_means_binary64_hex"]])
    w = np.array(scored.combiner.weights)
    assert np.array_equal(phi, (scored.member_matrix() - mu) * w)
    assert desc["feature_level"]["status"] == "unavailable"
    assert desc["predictor"]["combiner_sha256"] == scored.combiner.digest


def test_a_background_from_another_predictor_is_refused(tmp_path, captures, train_frames):
    scored, X, y, meta, ident = captures["test"]
    Xt, yt, mt = train_frames
    other = fit_fast(Xt.iloc[:200], yt.iloc[:200], mt.iloc[:200], None)
    alien = other.score_split(X, None, scored.result.rows, split_sha256=ident.prediction_split_sha256)
    writer = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    with pytest.raises(ContractError, match="different predictor"):
        writer.save_member_score_attributions(scored, background=alien)
    assert not (tmp_path / "member_score_attributions.parquet").exists()


# ---------------------------------------------------------------------------------------------------------------------
# Completion manifest (G8.1, G8.2)
# ---------------------------------------------------------------------------------------------------------------------

def _write_minimal(writer, captures):
    scored, _X, y, meta, _ident = captures["test"]
    writer.save_manifest(git_sha="abc", versions={"numpy": np.__version__}, config={"n": np.int64(3)})
    writer.save_split_predictions(scored, y, meta, role="test")


def test_completion_is_published_last_and_verifies_every_byte(tmp_path, captures):
    writer = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    _write_minimal(writer, captures)
    assert not (tmp_path / COMPLETION_MANIFEST).exists()
    writer.complete(["manifest.json", "test_predictions.parquet"], deliverables={"x": {"status": "not_requested"}})
    manifest = verify_completed_run(tmp_path)
    assert set(manifest["artefacts"]) == {"manifest.json", "test_predictions.parquet", "test_prediction_blocks.json"}
    assert manifest["predictor"]["combiner_sha256"] == captures["test"][0].combiner.digest
    assert manifest["prediction_blocks"]["test"]["result_block_sha256"] == captures["test"][0].result.digest
    assert json.loads((tmp_path / "manifest.json").read_text())["config"]["n"] == 3     # strict: an int, not "3"
    with pytest.raises(ContractError, match="sealed"):
        writer.save_json("late.json", {})
    with open(tmp_path / "test_predictions.parquet", "ab") as fh:
        fh.write(b"\x00")
    with pytest.raises(IncompleteRunError, match="changed since completion"):
        verify_completed_run(tmp_path)


def test_a_file_changed_after_it_was_written_prevents_completion(tmp_path, captures):
    writer = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    _write_minimal(writer, captures)
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")       # another process overwrote it
    with pytest.raises(IncompleteRunError, match="changed after this writer wrote it"):
        writer.complete(["manifest.json", "test_predictions.parquet"])
    assert not (tmp_path / COMPLETION_MANIFEST).exists()


def test_a_missing_required_output_or_capture_prevents_completion(tmp_path, captures):
    writer = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    writer.save_json("metrics.json", {"auroc": 0.9})
    with pytest.raises(IncompleteRunError, match="no bound prediction capture"):
        writer.complete(["metrics.json"])
    _write_minimal(writer, captures)
    with pytest.raises(IncompleteRunError, match="not produced by this run"):
        writer.complete(["metrics.json", "eval_report.json"])
    assert not (tmp_path / COMPLETION_MANIFEST).exists()


def test_fault_injection_before_completion_leaves_an_inspectable_unverifiable_run(tmp_path, captures, monkeypatch):
    """G8.2: a failure between the outputs and the manifest leaves every file readable and the run refused."""
    writer = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    _write_minimal(writer, captures)

    def _boom(*a, **k):
        raise OSError("disk full (injected)")

    monkeypatch.setattr(writer, "save_eval_report", _boom)
    with pytest.raises(OSError, match="injected"):
        writer.save_eval_report({"x": 1})        # the run's driver never reaches complete() after this
    assert (tmp_path / "test_predictions.parquet").exists() and not (tmp_path / COMPLETION_MANIFEST).exists()
    assert len(pd.read_parquet(tmp_path / "test_predictions.parquet")) == len(captures["test"][2])
    with pytest.raises(IncompleteRunError, match="not complete evidence"):
        verify_completed_run(tmp_path)


def test_a_rerun_retires_the_previous_completion_before_writing_anything(tmp_path, captures):
    first = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    _write_minimal(first, captures)
    first.complete(["test_predictions.parquet"])
    before = (tmp_path / COMPLETION_MANIFEST).read_bytes()
    second = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    assert not (tmp_path / COMPLETION_MANIFEST).exists()
    assert (tmp_path / second.retired_manifest).read_bytes() == before       # renamed, never deleted
    with pytest.raises(IncompleteRunError):
        verify_completed_run(tmp_path)


def test_a_retirement_never_overwrites_an_earlier_retirement(tmp_path, monkeypatch):
    """Two retirements in one directory with the SAME timestamp -- a coarse clock repeats it (Python 3.12 on Windows
    advances in steps of about 15.6 ms) -- keep BOTH earlier manifests, byte for byte."""
    from datetime import datetime as real_datetime, timezone

    import genomic_variant_classifier.evaluation.prediction_artifacts as pa

    fixed = real_datetime(2026, 11, 8, 0, 0, 0, 123456, tzinfo=timezone.utc)

    class _RepeatingClock:
        @staticmethod
        def now(tz=None):
            return fixed

    monkeypatch.setattr(pa, "datetime", _RepeatingClock)
    first, second = b'{"run": 1}\n', b'{"run": 2}\n'
    (tmp_path / COMPLETION_MANIFEST).write_bytes(first)
    one = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    (tmp_path / COMPLETION_MANIFEST).write_bytes(second)
    two = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    assert one.retired_manifest == "completion_manifest.superseded-20261108T000000123456Z.json"
    assert two.retired_manifest == "completion_manifest.superseded-20261108T000000123456Z-2.json"
    assert (tmp_path / one.retired_manifest).read_bytes() == first
    assert (tmp_path / two.retired_manifest).read_bytes() == second
    assert not (tmp_path / COMPLETION_MANIFEST).exists()


def test_strict_manifest_json_refuses_unrepresentable_values(tmp_path):
    writer = RunArtifactWriter(run_id="r", ablation="a", output_dir=tmp_path)
    with pytest.raises(UnserializableArtifactValue):
        writer.save_manifest("sha", {}, {"x": object()})
    with pytest.raises(NonFiniteArtifactValue):
        writer.save_manifest("sha", {}, {"x": float("nan")})
    assert not (tmp_path / "manifest.json").exists()


def test_the_ablation_aggregate_admits_only_completed_runs_atomically(tmp_path, captures):
    master = tmp_path / "ablation_results.parquet"
    writer = RunArtifactWriter(run_id="r", ablation="full", output_dir=tmp_path / "full")
    _write_minimal(writer, captures)
    with pytest.raises(IncompleteRunError):
        writer.append_ablation_row(master, {"auroc": 0.9})
    assert not master.exists()
    writer.complete(["test_predictions.parquet"])
    writer.append_ablation_row(master, {"auroc": 0.9})
    writer.append_ablation_row(master, {"auroc": 0.91})
    df = pd.read_parquet(master)
    assert len(df) == 1 and df.loc[0, "auroc"] == 0.91
    assert df.loc[0, "completion_manifest_sha256"] == writer.completion_manifest_sha256
    assert not list(tmp_path.glob(".ablation_results.parquet.*.tmp"))
    with pytest.raises(ContractError, match="not this writer's"):
        writer.append_ablation_row(master, {"ablation": "other", "auroc": 0.5})


# ---------------------------------------------------------------------------------------------------------------------
# The migrated drivers, executed (not inspected)
# ---------------------------------------------------------------------------------------------------------------------

def _splits_dir(root: Path) -> Path:
    sd = root / "splits"
    sd.mkdir(parents=True)
    for role, seed, n in (("train", 41, 420), ("val", 42, 150), ("test", 43, 200)):
        X, y, meta = synthetic_split(n, seed=seed, role=role)
        X.to_parquet(sd / f"X_{role}.parquet", index=False)
        y.to_frame("label").to_parquet(sd / f"y_{role}.parquet", index=False)
        meta.to_parquet(sd / f"meta_{role}.parquet", index=False)
    return sd


def _run9(monkeypatch, argv):
    install_fast_roster(monkeypatch)
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("run9_ablations", None)
    import run9_ablations
    monkeypatch.setattr(sys, "argv", ["run9_ablations.py", *argv])
    return run9_ablations


def test_the_ablation_harness_runs_to_a_verified_completion(tmp_path, monkeypatch):
    sd = _splits_dir(tmp_path)
    out = tmp_path / "run" / "full"
    module = _run9(monkeypatch, ["--splits-dir", str(sd), "--run-id", "t", "--output-dir", str(out), "--n-folds",
                                 "3", "--max-train", "300", "--skip-permutation"])
    assert module.main() == 0
    manifest = verify_completed_run(out)
    assert manifest["deliverables"]["permutation_importance"] == {"status": "not_requested"}
    assert manifest["deliverables"]["member_score_attributions"]["status"] == "complete"
    oof = pd.read_parquet(out / "oof_predictions.parquet")
    y = pd.read_parquet(sd / "y_train.parquet")["label"].to_numpy()
    assert np.array_equal(oof["label"].to_numpy(), y[oof["train_split_position"].to_numpy()])
    assert manifest["oof_identity"]["subsample"]["n"] == 300
    agg = pd.read_parquet(tmp_path / "run" / "ablation_results.parquet")
    digest = __import__("hashlib").sha256((out / COMPLETION_MANIFEST).read_bytes()).hexdigest()
    assert agg.loc[0, "completion_manifest_sha256"] == digest


def test_a_failed_deliverable_leaves_the_ablation_incomplete_and_unaggregated(tmp_path, monkeypatch):
    sd = _splits_dir(tmp_path)
    out = tmp_path / "run" / "full"
    module = _run9(monkeypatch, ["--splits-dir", str(sd), "--run-id", "t", "--output-dir", str(out), "--n-folds", "3"])

    def _boom(self, *a, **k):
        raise RuntimeError("permutation importance failed (injected)")

    monkeypatch.setattr(RunArtifactWriter, "save_permutation_importance", _boom)
    with pytest.raises(RuntimeError, match="injected"):
        module.main()
    assert (out / "test_predictions.parquet").exists() and not (out / COMPLETION_MANIFEST).exists()
    assert not (tmp_path / "run" / "ablation_results.parquet").exists()


def test_the_ablation_preflight_refuses_before_any_fit(tmp_path, monkeypatch):
    sd = _splits_dir(tmp_path)
    (sd / "meta_train.parquet").unlink()
    out = tmp_path / "run" / "full"
    module = _run9(monkeypatch, ["--splits-dir", str(sd), "--run-id", "t", "--output-dir", str(out)])
    assert module.main() == 2
    assert not out.exists()


def _phase2(monkeypatch, tmp_path, argv):
    from sklearn.preprocessing import StandardScaler

    from genomic_variant_classifier.data import real_data_prep

    frames = {role: synthetic_split(n, seed=seed, role=role)
              for role, seed, n in (("train", 51, 420), ("val", 52, 150), ("test", 53, 200))}

    def fake_run(self, clinvar_path, gnomad_path=None, **kwargs):
        d = Path(self.config.output_dir)
        d.mkdir(parents=True, exist_ok=True)
        for role, (X, y, meta) in frames.items():
            X.to_parquet(d / f"X_{role}.parquet", index=False)
            y.to_frame("label").to_parquet(d / f"y_{role}.parquet", index=False)
            meta.to_parquet(d / f"meta_{role}.parquet", index=False)
        self.scaler = StandardScaler().fit(frames["train"][0])
        (Xtr, ytr, _), (Xva, yva, mva), (Xte, yte, mte) = frames["train"], frames["val"], frames["test"]
        return Xtr.copy(), Xva.copy(), Xte.copy(), ytr.copy(), yva.copy(), yte.copy(), mva.copy(), mte.copy()

    install_fast_roster(monkeypatch)
    monkeypatch.setattr(real_data_prep.DataPrepPipeline, "run", fake_run)
    clinvar = tmp_path / "clinvar.parquet"
    clinvar.write_bytes(b"placeholder: data preparation is replaced in this test")
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("run_phase2_eval", None)
    import run_phase2_eval
    monkeypatch.setattr(sys, "argv", ["run_phase2_eval.py", "--clinvar", str(clinvar), "--seq-windows", "",
                                      "--skip-cnn", "--allow-dead-features", "--n-folds", "3", *argv])
    return run_phase2_eval, frames


def test_the_mainline_max_train_path_composes_identity_and_completes(tmp_path, monkeypatch):
    """--max-train used to raise AttributeError (`seq_tr.iloc` on a WindowAttachment) and fed the GNN blocks the full
    meta_train; it now subsamples every row-aligned input once and the OOF identity composes through it."""
    out = tmp_path / "rpe" / "full"
    module, frames = _phase2(monkeypatch, tmp_path, ["--max-train", "300", "--output", str(out)])
    assert module.main() in (0, 1)
    manifest = verify_completed_run(out)
    assert manifest["oof_identity"]["subsample"]["applied"] is True
    oof = pd.read_parquet(out / "oof_predictions.parquet")
    y = frames["train"][1].to_numpy()
    assert np.array_equal(oof["label"].to_numpy(), y[oof["train_split_position"].to_numpy()])
    assert {"test_predictions.parquet", "member_score_attributions.parquet", "splits/meta_train.parquet",
            "models/ensemble.joblib"} <= set(manifest["artefacts"])


def test_a_failed_identity_binding_blocks_mainline_publication(tmp_path, monkeypatch):
    """G2.2: no warning-and-continue. The predictions stay inspectable; the run exits 2 without a manifest."""
    out = tmp_path / "rpe" / "full"
    module, _ = _phase2(monkeypatch, tmp_path, ["--output", str(out)])
    from genomic_variant_classifier.evaluation import oof_identity

    def _refuse(*a, **k):
        raise ContractError("identity binding FAILED (injected)")

    monkeypatch.setattr(oof_identity, "compose_oof_identity", _refuse)
    assert module.main() == 2
    assert (out / "test_predictions.parquet").exists()
    assert not (out / COMPLETION_MANIFEST).exists() and not (out / "oof_predictions.parquet").exists()


# ---------------------------------------------------------------------------------------------------------------------
# Consumers of the OOF table: completed runs only (matrix set-1b list)
# ---------------------------------------------------------------------------------------------------------------------

def test_the_insights_detector_reads_only_completed_runs(tmp_path):
    from genomic_variant_classifier.evaluation import model_insights_detector as D

    done = completed_run(tmp_path / "outputs" / "run1" / "full", seed=5)
    stray = tmp_path / "outputs" / "run2" / "full"
    stray.mkdir(parents=True)
    pd.read_parquet(done / "oof_predictions.parquet").to_parquet(stray / "oof_predictions.parquet")
    found, incomplete = D.discover_runs(str(tmp_path / "outputs"))
    assert found == done and [i["run_dir"] for i in incomplete] == [str(stray)]
    oof, manifest = D.load_completed_oof(found)
    result = D.analyze(oof, manifest["oof_identity"]["members"])
    assert {m.model for m in result["metrics"]} == set(FAST_MEMBERS)
    assert result["gene_disjoint"] is True


def test_folds_that_share_genes_are_flagged_from_a_real_run(tmp_path):
    from genomic_variant_classifier.evaluation import model_insights_detector as D

    run = completed_run(tmp_path / "outputs" / "r" / "full", seed=6, gene_groups=False)
    oof, manifest = D.load_completed_oof(run)
    result = D.analyze(oof, manifest["oof_identity"]["members"])
    assert result["gene_disjoint"] is False and result["flags"][0].startswith("GENE_DISJOINT_VIOLATION")


# ---------------------------------------------------------------------------------------------------------------------
# The separately identified derived record of a HISTORICAL run (G2.3) and its code provenance (G7.3)
# ---------------------------------------------------------------------------------------------------------------------

def _legacy_run(root: Path, *, n: int = 240, tamper: bool = False) -> Path:
    """A pre-set-1b run directory with Run 15's layout and its defects: `_train_row_idx` = 0..n-1 and fold-concatenated
    sidecars. The orchestrator is a real joblib dict holding a real EnsembleConfig, as Run 15's does."""
    from sklearn.model_selection import train_test_split

    from genomic_variant_classifier.models.variant_ensemble import EnsembleConfig

    rng = np.random.default_rng(9)
    y = (rng.random(n) < 0.3).astype(np.int64)
    fit_idx, _cal = train_test_split(np.arange(n), test_size=0.15, stratify=y, random_state=42)
    names = ["m_a", "m_b"]
    oof = np.column_stack([np.clip(0.7 * y[fit_idx] + 0.3 * rng.random(len(fit_idx)), 0, 1), rng.random(len(fit_idx))])
    models = root / "models"
    (root / "splits").mkdir(parents=True)
    models.mkdir()
    stored = np.random.default_rng(1).permutation(fit_idx) if tamper else fit_idx
    joblib.dump({"format_version": 2, "config": EnsembleConfig(n_folds=3, model_dir=None),
                 "blend_weights_": np.array([0.6, 0.4]), "oof_predictions_": oof, "oof_fit_indices_": stored,
                 "oof_model_names_": names, "saved_model_paths": {}, "models_dir_name": "ensemble_models"},
                models / "ensemble.joblib")
    (models / "ensemble.manifest.json").write_text(json.dumps({"libraries": {"scikit-learn": "1.8.0"}}))
    frame = pd.DataFrame(oof, columns=names)
    frame.insert(0, "_train_row_idx", np.arange(len(fit_idx), dtype=np.int64))
    frame.to_parquet(root / "oof_predictions.parquet", index=False)
    pd.DataFrame({"label": y}).to_parquet(root / "splits" / "y_train.parquet", index=False)
    pd.DataFrame({"variant_id": [f"v{i}" for i in range(n)], "gene_symbol": [f"G{i % 9}" for i in range(n)]}
                 ).to_parquet(root / "splits" / "meta_train.parquet", index=False)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    concatenated = np.concatenate([va for _tr, va in splitter.split(np.zeros(len(fit_idx)), y[fit_idx])])
    for name in names:
        np.save(models / f"{name}_oof_indices.npy", concatenated)
    return root


def test_a_historical_oof_identity_is_derived_from_the_artifact_and_verified(tmp_path):
    from genomic_variant_classifier.evaluation.historical_oof_identity import derive_legacy_oof_identity

    run = _legacy_run(tmp_path / "run")
    oof_before = (run / "oof_predictions.parquet").read_bytes()
    record = derive_legacy_oof_identity(run, mapping_out=tmp_path / "derived" / "mapping.parquet")
    assert record["stored_equals_reconstruction"] is True
    assert record["stored_oof_fit_indices_int64_sha256"] == record["reconstructed_fit_indices_int64_sha256"]
    assert record["members"]["m_a"]["auroc_through_derived_mapping"] > 0.95
    assert record["members"]["m_a"]["auroc_joined_naively_as_training_row"] < 0.75
    assert record["members"]["m_a"]["legacy_sidecar"]["equals_reconstructed_fold_concatenation"] is True
    assert record["stored_blend_weights"]["n"] == 2
    mapping = pd.read_parquet(tmp_path / "derived" / "mapping.parquet")
    y = pd.read_parquet(run / "splits" / "y_train.parquet")["label"].to_numpy()
    assert np.array_equal(mapping["label"].to_numpy(), y[mapping["train_split_position"].to_numpy()])
    assert (run / "oof_predictions.parquet").read_bytes() == oof_before       # the original is never rewritten
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    assert record["record_sha256"] == canonical_digest("gvc.historical-oof-identity-record.v1", body)


def test_a_historical_mapping_that_does_not_reproduce_is_refused(tmp_path):
    from genomic_variant_classifier.evaluation.historical_oof_identity import derive_legacy_oof_identity

    run = _legacy_run(tmp_path / "run", tamper=True)
    with pytest.raises(ContractError, match="differ from the reconstructed legacy carve"):
        derive_legacy_oof_identity(run, mapping_out=tmp_path / "derived" / "mapping.parquet")
    assert not (tmp_path / "derived").exists()


def test_the_code_provenance_is_read_from_the_commit_not_assumed(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("derive_run15_oof_identity", None)
    import derive_run15_oof_identity as D

    # The user's own git configuration (hooks, signing, line-end conversion) never reaches this repository: both
    # configuration files are an EMPTY FILE of this test (the same on every system; the null device's name is not).
    empty = tmp_path / "empty.gitconfig"
    empty.write_bytes(b"")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty))
    repo = tmp_path / "repo"
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
           "-C", str(repo)]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for path, facts in D._FACTS.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(facts.values()) + "\n", encoding="utf-8")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "--no-verify", "-m", "historical"], check=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    record = D.code_provenance(repo, head)
    assert record["commit"] == head and set(record["files"]) == set(D._FACTS)
    (repo / D._ENSEMBLE).write_text("blend = something_else\n", encoding="utf-8")
    subprocess.run([*git, "commit", "-q", "--no-verify", "-am", "changed"], check=True)
    with pytest.raises(RuntimeError, match="the attribution is not proven"):
        D.code_provenance(repo, "HEAD")
