"""Containment at the serving, loading and fitted-model boundaries (completes the 2026-09-22 quarantine).

Owner rulings: "Quarantined information cannot enter through producers, defaults, caches, fitted models,
stacking, or serving." Before this, api/pipeline.py zero-filled every declared column missing from the
input, so a 95-feature model was served with the four quarantined features fabricated as 0.0.

Author: Monzia Moodie
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from genomic_variant_classifier.api import pipeline as P
from genomic_variant_classifier.containment import ContainmentError
from genomic_variant_classifier.models import variant_ensemble as ve
from genomic_variant_classifier.quarantine_policy import QUARANTINED_FEATURES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _admission_support import AllowForTests, register  # noqa: E402

Q0 = QUARANTINED_FEATURES[0]


def _decoy(tmp_path):
    """A DIFFERENT registered artifact: the registry exists and holds a record, yet binds nothing loaded."""
    p = tmp_path / "decoy.bin"
    p.write_bytes(b"some other artifact")
    return p


def _raw(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"chrom": [str(1 + i % 22) for i in range(n)], "pos": rng.integers(1, 10**6, n),
                         "ref": ["A"] * n, "alt": ["G"] * n, "consequence": ["missense_variant"] * n,
                         "allele_freq": rng.uniform(0, 0.01, n)})


def _usable_features(k=5):
    X = ve.engineer_features(_raw())
    return [c for c in X.columns if not X[c].isna().any()][:k]


def _fitted(columns):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(40, len(columns))), columns=columns)
    y = np.array([0, 1] * 20)
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression().fit(scaler.transform(X), y)
    meta = LogisticRegression().fit(lr.predict_proba(scaler.transform(X))[:, 1:], y)
    return scaler, lr, meta


def _legacy(columns):
    """An old pickle: attributes restored WITHOUT __init__, exactly as joblib.load does."""
    scaler, lr, meta = _fitted(columns)
    obj = P.InferencePipeline.__new__(P.InferencePipeline)
    obj.__dict__.update(trained_models={"logistic_regression": lr}, meta_learner=meta, scaler=scaler,
                        metadata=P.PipelineMetadata(feature_names=list(columns)), gnn_scorer=None, preprocessor_=None)
    return obj


class TestConstructionAndLoading:
    def test_construction_refuses_a_quarantine_era_model(self):
        scaler, lr, meta = _fitted(_usable_features(3) + [Q0])
        with pytest.raises(ContainmentError, match="suspended"):
            P.InferencePipeline({"logistic_regression": lr}, meta, scaler=scaler)

    # The ONE admission route (model_admission): registry binding + typed decision, BEFORE deserialization.
    def _no_open(self, monkeypatch):
        opened = []
        monkeypatch.setattr(joblib, "load", lambda *a, **k: opened.append(a) or pytest.fail("deserialized"))
        return opened

    def test_an_unregistered_artifact_is_refused_before_opening(self, tmp_path, monkeypatch):
        path = tmp_path / "legacy.joblib"
        joblib.dump(_legacy(_usable_features(3) + [Q0]), path)
        registry = register(tmp_path / "registry.v1.json", _decoy(tmp_path), feature_names=["x"], roster=["m"])
        opened = self._no_open(monkeypatch)
        with pytest.raises(ContainmentError, match="Unbound artifact"):
            P.InferencePipeline.load(path, consumer="test", registry_path=registry, authority=AllowForTests())
        assert opened == []

    def test_a_record_naming_quarantined_features_is_refused_before_opening(self, tmp_path, monkeypatch):
        path = tmp_path / "m.joblib"
        joblib.dump(_legacy(_usable_features(3)), path)
        registry = register(tmp_path / "registry.v1.json", path, feature_names=_usable_features(3) + [Q0],
                            roster=["logistic_regression"])
        opened = self._no_open(monkeypatch)
        with pytest.raises(ContainmentError, match="suspended"):
            P.InferencePipeline.load(path, consumer="test", registry_path=registry, authority=AllowForTests())
        assert opened == []

    def test_positive_control_a_registered_allowed_pipeline_round_trips_exactly(self, tmp_path):
        cols = _usable_features(5)
        scaler, lr, meta = _fitted(cols)
        pipe = P.InferencePipeline({"logistic_regression": lr}, meta, scaler=scaler)
        before = pipe.predict_proba(_raw())
        path = tmp_path / "ok.joblib"
        pipe.save(path)
        registry = register(tmp_path / "registry.v1.json", path, feature_names=cols, roster=["logistic_regression"])
        back = P.InferencePipeline.load(path, consumer="test", registry_path=registry, authority=AllowForTests())
        np.testing.assert_array_equal(back.predict_proba(_raw()), before)

    def test_the_default_authority_refuses_even_a_registered_pipeline(self, tmp_path, monkeypatch):
        cols = _usable_features(5)
        scaler, lr, meta = _fitted(cols)
        path = tmp_path / "ok.joblib"
        P.InferencePipeline({"logistic_regression": lr}, meta, scaler=scaler).save(path)
        registry = register(tmp_path / "registry.v1.json", path, feature_names=cols, roster=["logistic_regression"])
        opened = self._no_open(monkeypatch)
        with pytest.raises(ContainmentError, match="C10"):
            P.InferencePipeline.load(path, consumer="test", registry_path=registry)
        assert opened == []

    def test_a_record_whose_features_differ_from_the_pipeline_is_refused(self, tmp_path):
        cols = _usable_features(5)
        scaler, lr, meta = _fitted(cols)
        path = tmp_path / "ok.joblib"
        P.InferencePipeline({"logistic_regression": lr}, meta, scaler=scaler).save(path)
        registry = register(tmp_path / "registry.v1.json", path, feature_names=list(reversed(cols)),
                            roster=["logistic_regression"])      # same names, different order
        with pytest.raises(ContainmentError, match="features differ from its registry record"):
            P.InferencePipeline.load(path, consumer="test", registry_path=registry, authority=AllowForTests())

    def test_a_record_whose_roster_differs_without_a_declared_projection_is_refused(self, tmp_path):
        cols = _usable_features(5)
        scaler, lr, meta = _fitted(cols)
        path = tmp_path / "ok.joblib"
        P.InferencePipeline({"logistic_regression": lr}, meta, scaler=scaler).save(path)
        registry = register(tmp_path / "registry.v1.json", path, feature_names=cols,
                            roster=["logistic_regression", "catboost"])
        with pytest.raises(ContainmentError, match="roster does not match"):
            P.InferencePipeline.load(path, consumer="test", registry_path=registry, authority=AllowForTests())

    def test_changed_bytes_after_registration_are_refused_before_deserializing(self, tmp_path, monkeypatch):
        cols = _usable_features(5)
        scaler, lr, meta = _fitted(cols)
        path = tmp_path / "t.joblib"
        P.InferencePipeline({"logistic_regression": lr}, meta, scaler=scaler).save(path)
        registry = register(tmp_path / "registry.v1.json", path, feature_names=cols, roster=["logistic_regression"])
        with path.open("ab") as fh:
            fh.write(b"\x00")
        opened = self._no_open(monkeypatch)
        with pytest.raises(ContainmentError, match="Unbound artifact"):
            P.InferencePipeline.load(path, consumer="test", registry_path=registry, authority=AllowForTests())
        assert opened == []

    def test_save_refuses_a_quarantine_era_pipeline(self, tmp_path):
        with pytest.raises(ContainmentError):
            _legacy(_usable_features(3) + [Q0]).save(tmp_path / "x.joblib")
        assert not (tmp_path / "x.joblib").exists()


class TestServingMatrix:
    @pytest.mark.parametrize("method", ["predict_proba", "predict_proba_with_uncertainty"])
    def test_a_quarantine_era_legacy_object_cannot_predict(self, method):
        with pytest.raises(ContainmentError, match="suspended"):
            getattr(_legacy(_usable_features(3) + [Q0]), method)(_raw())

    @pytest.mark.parametrize("method", ["predict_proba", "predict_proba_with_uncertainty"])
    def test_a_missing_feature_is_refused_never_zero_filled(self, method):
        scaler, lr, meta = _fitted(_usable_features(3) + ["made_up_feature"])
        pipe = P.InferencePipeline({"logistic_regression": lr}, meta, scaler=scaler)
        with pytest.raises(ContainmentError, match="never zero-filled"):
            getattr(pipe, method)(_raw())

    @pytest.mark.parametrize("method", ["predict_proba", "predict_proba_with_uncertainty"])
    def test_an_admissible_pipeline_predicts(self, method):
        scaler, lr, meta = _fitted(_usable_features(5))
        out = getattr(P.InferencePipeline({"logistic_regression": lr}, meta, scaler=scaler), method)(_raw())
        proba = out if method == "predict_proba" else out["proba"]
        assert proba.shape == (40,) and np.all((proba >= 0) & (proba <= 1))


class TestFittedModelBoundary:
    def _light(self, members=("logistic_regression",)):
        ens = ve.VariantEnsemble(ve.EnsembleConfig(n_folds=3))
        ens.base_estimators = {k: v for k, v in ens.base_estimators.items() if k in members}
        assert set(ens.base_estimators) == set(members), sorted(ens.base_estimators)
        return ens

    def _data(self):
        # 24 genes of 8-15 rows with random labels: large enough for the gene-disjoint calibration carve
        # (the same shape tests/unit/test_isolation_prerequisite.py uses).
        rng = np.random.default_rng(0)
        genes, labels = [], []
        for g in range(24):
            for _ in range(int(rng.integers(8, 16))):
                genes.append(f"G{g}"); labels.append(int(rng.random() < 0.5))
        X = pd.DataFrame(rng.normal(size=(len(genes), 3)), columns=["a", "b", "c"])
        return X, pd.Series(labels), pd.Series(genes)

    def test_fit_refuses_a_cached_frame_with_a_quarantined_column(self):
        X, y, g = self._data()
        with pytest.raises(ContainmentError, match="suspended"):
            self._light().fit(X.assign(**{Q0: 50.0}), None, y, gene_symbol=g)

    def test_fit_refuses_an_anonymous_array(self):
        X, y, g = self._data()
        with pytest.raises(ContainmentError, match="named columns"):
            self._light().fit(X.to_numpy(), None, y, gene_symbol=g)

    def test_predict_requires_the_fitted_names_in_the_fitted_order(self):
        X, y, g = self._data()
        ens = self._light().fit(X, None, y, gene_symbol=g)
        assert ens.tabular_columns_ == ("a", "b", "c")
        assert ens.feature_names_ == ["logistic_regression"]     # the pre-existing stacking-order list, untouched
        assert ens.predict_proba(X).shape == (len(X), 2)
        with pytest.raises(ContainmentError, match="differ from the bound contract"):
            ens.predict_proba(X[["c", "a", "b"]])
        with pytest.raises(ContainmentError, match="suspended"):
            ens.predict_proba(X.assign(**{Q0: 0.0}))

    def _saved(self, tmp_path, members=("logistic_regression",)):
        X, y, g = self._data()
        ens = self._light(members).fit(X, None, y, gene_symbol=g)
        path = tmp_path / "ens.joblib"
        ens.save(path)
        return X, ens, path

    def _load(self, path, registry):
        return ve.VariantEnsemble.load(path, consumer="test", registry_path=registry, authority=AllowForTests())

    def test_positive_control_a_registered_ensemble_round_trips_exactly(self, tmp_path):
        X, ens, path = self._saved(tmp_path)
        registry = register(tmp_path / "registry.v1.json", path, feature_names=("a", "b", "c"),
                            roster=list(ens.trained_models_))
        back = self._load(path, registry)
        assert back.tabular_columns_ == ("a", "b", "c") and list(back.trained_models_) == list(ens.trained_models_)
        np.testing.assert_array_equal(back.predict_proba(X), ens.predict_proba(X))

    def test_a_quarantined_bound_column_is_refused_even_when_registered(self, tmp_path):
        X, ens, path = self._saved(tmp_path)
        orch = joblib.load(path)
        orch["tabular_columns_"] = ("a", "b", Q0)
        joblib.dump(orch, path)
        registry = register(tmp_path / "registry.v1.json", path, feature_names=("a", "b", "c"),
                            roster=list(ens.trained_models_))
        with pytest.raises(ContainmentError, match="suspended"):
            self._load(path, registry)

    def test_one_tampered_member_refuses_the_whole_ensemble(self, tmp_path):
        X, ens, path = self._saved(tmp_path)
        registry = register(tmp_path / "registry.v1.json", path, feature_names=("a", "b", "c"),
                            roster=list(ens.trained_models_))
        member = next((path.parent / "ens_models").iterdir())
        with member.open("ab") as fh:
            fh.write(b"\x00")
        with pytest.raises(ContainmentError, match="differ from the admitted manifest"):
            self._load(path, registry)

    def test_a_missing_member_refuses_the_whole_ensemble(self, tmp_path):
        X, ens, path = self._saved(tmp_path)
        registry = register(tmp_path / "registry.v1.json", path, feature_names=("a", "b", "c"),
                            roster=list(ens.trained_models_))
        next((path.parent / "ens_models").iterdir()).unlink()
        with pytest.raises(FileNotFoundError):
            self._load(path, registry)

    def test_members_out_of_meta_learner_order_are_refused(self, tmp_path):
        # TWO members: reversing a one-member order is a no-op and could never reach this guard.
        X, ens, path = self._saved(tmp_path, members=("logistic_regression", "random_forest"))
        orch = joblib.load(path)
        reordered = list(reversed(orch["feature_names_"]))
        assert reordered != list(orch["feature_names_"]), "the fixture must actually change the order"
        orch["feature_names_"] = reordered
        joblib.dump(orch, path)
        registry = register(tmp_path / "registry.v1.json", path, feature_names=("a", "b", "c"),
                            roster=list(ens.trained_models_))
        with pytest.raises(ContainmentError, match="not in the order the meta-learner was fitted on"):
            self._load(path, registry)

    def test_save_is_all_or_nothing(self, tmp_path):
        X, y, g = self._data()
        ens = self._light().fit(X, None, y, gene_symbol=g)
        ens.trained_models_[next(iter(ens.trained_models_))] = lambda: None     # unpicklable member
        path = tmp_path / "ens.joblib"
        with pytest.raises(RuntimeError, match="Ensemble NOT saved"):
            ens.save(path)
        assert not path.exists()


class _Records(logging.Handler):
    """Collects records on the LOGGER UNDER TEST. The application's startup replaces every ROOT handler
    when python-json-logger is installed (as in CI), which removed pytest's caplog handler; a handler
    on api.main's own logger is unaffected (CI run #882, reproduced 2026-09-23)."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_the_api_server_refuses_to_serve_a_quarantined_artifact(tmp_path, monkeypatch):
    from genomic_variant_classifier.api import main
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"any bytes")
    monkeypatch.setattr(main, "MODEL_PATH", artifact)
    def refuse(path, loader):
        raise ContainmentError("Scientific contract suspended: ['alphafold_plddt']")
    monkeypatch.setattr(main, "load_pipeline_with_identity", refuse)
    async def run():
        async with main.lifespan(main.app):
            return main._PIPELINE
    records = _Records()
    root_handlers, root_level = list(logging.root.handlers), logging.root.level
    main.logger.addHandler(records)
    try:
        assert asyncio.run(run()) is None
    finally:
        main.logger.removeHandler(records)
        # Startup rewrote the ROOT logger; restore it so no later test inherits that change.
        logging.root.handlers[:] = root_handlers
        logging.root.setLevel(root_level)
    refusals = [r for r in records.records if "Refusing to serve" in r.getMessage()]
    assert len(refusals) == 1, [r.getMessage() for r in records.records]
    assert refusals[0].levelno == logging.ERROR and refusals[0].exc_info is not None
    assert refusals[0].exc_info[0].__name__ == "ContainmentError"
