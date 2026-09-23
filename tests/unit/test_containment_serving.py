"""Containment at the serving, loading and fitted-model boundaries (completes the 2026-09-22 quarantine).

Owner rulings: "Quarantined information cannot enter through producers, defaults, caches, fitted models,
stacking, or serving." Before this, api/pipeline.py zero-filled every declared column missing from the
input, so a 95-feature model was served with the four quarantined features fabricated as 0.0.

Author: Monzia Moodie
"""
from __future__ import annotations

import asyncio
import json
import logging

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

Q0 = QUARANTINED_FEATURES[0]


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

    def test_a_legacy_artifact_is_refused_immediately_after_loading(self, tmp_path):
        path = tmp_path / "legacy.joblib"
        joblib.dump(_legacy(_usable_features(3) + [Q0]), path)
        with pytest.raises(ContainmentError, match="suspended"):
            P.InferencePipeline.load(path)

    def test_a_manifest_naming_quarantined_features_is_refused_before_opening(self, tmp_path, monkeypatch):
        path = tmp_path / "m.joblib"
        joblib.dump(_legacy(_usable_features(3)), path)
        path.with_suffix(".manifest.json").write_text(json.dumps(
            {"artifact_sha256": "0" * 64, "feature_names": _usable_features(3) + [Q0]}))
        opened = []
        monkeypatch.setattr(joblib, "load", lambda *a, **k: opened.append(a) or pytest.fail("opened"))
        with pytest.raises(ContainmentError, match="suspended"):
            P.InferencePipeline.load(path)
        assert opened == []

    def test_save_records_digest_and_features_and_load_round_trips(self, tmp_path):
        cols = _usable_features(5)
        scaler, lr, meta = _fitted(cols)
        pipe = P.InferencePipeline({"logistic_regression": lr}, meta, scaler=scaler)
        before = pipe.predict_proba(_raw())
        path = tmp_path / "ok.joblib"
        pipe.save(path)
        man = json.loads(path.with_suffix(".manifest.json").read_text())
        assert man["feature_names"] == cols and len(man["artifact_sha256"]) == 64
        np.testing.assert_array_equal(P.InferencePipeline.load(path).predict_proba(_raw()), before)

    def test_changed_bytes_after_save_are_refused_before_deserializing(self, tmp_path):
        scaler, lr, meta = _fitted(_usable_features(5))
        path = tmp_path / "t.joblib"
        P.InferencePipeline({"logistic_regression": lr}, meta, scaler=scaler).save(path)
        with path.open("ab") as fh:
            fh.write(b"\x00")
        with pytest.raises(ContainmentError, match="differ from the admitted manifest"):
            P.InferencePipeline.load(path)

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
    def _light(self):
        ens = ve.VariantEnsemble(ve.EnsembleConfig(n_folds=3))
        ens.base_estimators = {k: v for k, v in ens.base_estimators.items() if k == "logistic_regression"}
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

    def test_save_and_load_keep_the_binding_and_load_refuses_a_quarantined_one(self, tmp_path):
        X, y, g = self._data()
        ens = self._light().fit(X, None, y, gene_symbol=g)
        path = tmp_path / "ens.joblib"
        ens.save(path)
        back = ve.VariantEnsemble.load(path)
        assert back.tabular_columns_ == ("a", "b", "c")
        np.testing.assert_array_equal(back.predict_proba(X), ens.predict_proba(X))
        orch = joblib.load(path)
        orch["tabular_columns_"] = ("a", "b", Q0)
        joblib.dump(orch, path)
        with pytest.raises(ContainmentError, match="suspended"):
            ve.VariantEnsemble.load(path)


def test_the_api_server_refuses_to_serve_a_quarantined_artifact(tmp_path, monkeypatch, caplog):
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
    with caplog.at_level(logging.ERROR):
        assert asyncio.run(run()) is None
    assert any("Refusing to serve" in r.getMessage() for r in caplog.records)
