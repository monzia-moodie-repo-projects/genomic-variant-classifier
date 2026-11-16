"""Stage 1D-1c.1C: projected members inside VariantEnsemble and InferencePipeline (owner ruling aa999b96, decisions A
and B; revised section 15 points 3-9 at the ensemble boundary). MEMBER-PROJECTION-SERVING-DRIFT-1.

Author: Monzia Moodie
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from genomic_variant_classifier.api.pipeline import InferencePipeline
from genomic_variant_classifier.models.feature_semantics import AF_FAMILY, FeatureSemantics as FS, roster_without_family
from genomic_variant_classifier.models.member import ProjectedMember, ProjectionProtocol as PP, is_projected
from genomic_variant_classifier.models.model_preprocessing import TabularModelPreprocessor
from genomic_variant_classifier.models.variant_ensemble import EnsembleConfig, VariantEnsemble, engineer_features
from genomic_variant_classifier.prediction_contract import ContractError, canonical_digest
from tests.unit._fit_contexts import EPOCH1_TEST_FIT, epoch2_test_fit
from tests.unit._prediction_support import FixedProba, blend_pipeline


def _raw(n: int = 500, seed: int = 0, constraint_missing: bool = True) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    af = np.where(rng.random(n) < 0.3, np.nan, rng.beta(0.3, 30, n))
    raw = pd.DataFrame({"allele_freq": af, "ref": rng.choice(list("ACGT"), n), "alt": rng.choice(["A", "C", "GT"], n),
                        "consequence": rng.choice(["missense_variant", "synonymous_variant", "stop_gained"], n),
                        "cadd_phred": rng.normal(15, 5, n), "revel_score": rng.random(n), "loeuf": rng.random(n) * 2,
                        "gene_constraint_oe": np.where((rng.random(n) < 0.2) & constraint_missing, np.nan, rng.random(n))})
    y = pd.Series(((np.nan_to_num(af, nan=0.5) < 1e-3) & (rng.random(n) < 0.7)).astype(int))
    genes = pd.Series([f"G{i % 37}" for i in range(n)])
    return raw, y, genes


def _ens(names) -> VariantEnsemble:
    ens = VariantEnsemble(EnsembleConfig(n_jobs=1))
    ens.base_estimators = {k: v for k, v in ens.base_estimators.items() if k in names}
    if "random_forest" in ens.base_estimators:
        ens.base_estimators["random_forest"].set_params(n_estimators=25)
    return ens


def _fit(ens, X, y, genes, ctx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ens.fit(X, None, y, gene_symbol=genes, authorization=ctx)


class AFProbe(ClassifierMixin, BaseEstimator):
    """Returns the projected AF (column 0 of the rendered matrix) as its probability: the fold statistic made visible."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.asarray(X)[:, 0].astype(float)
        return np.column_stack([1 - p, p])


def test_fit_requires_an_explicit_fit_context():
    raw, y, genes = _raw()
    X = engineer_features(raw)
    with pytest.raises(ContractError, match="authorization=FitContext"):
        _ens(["logistic_regression"]).fit(X, None, y)
    with pytest.raises(ContractError, match="authorization=FitContext"):
        _ens(["logistic_regression"]).fit(X, None, y, authorization="strict")


def test_the_matrix_epoch_must_equal_the_context_epoch():
    raw, y, genes = _raw()
    with pytest.raises(ContractError, match="epochs are never mixed"):
        _fit(_ens(["logistic_regression"]), engineer_features(raw, semantics=FS.V2_OBSERVED_AF), y, genes, EPOCH1_TEST_FIT)
    with pytest.raises(ContractError, match="epochs are never mixed"):
        _fit(_ens(["logistic_regression"]), engineer_features(raw), y, genes, epoch2_test_fit(PP.CAPABILITY_AWARE))


def test_a_legacy_fit_is_unchanged_and_records_its_context():
    # Epoch 1 passes a missing gene_constraint_oe to the members as NaN, and a finite-requiring member refuses loudly --
    # exactly as on main 1fba4cb8 (measured with the same fixture: the same RuntimeError, caused by "Input X contains
    # NaN"). The legacy path is reproduced, not repaired.
    raw, y, genes = _raw()
    with pytest.raises(RuntimeError, match="logistic_regression") as exc:
        _fit(_ens(["logistic_regression"]), engineer_features(raw), y, genes, EPOCH1_TEST_FIT)
    assert "Input X contains NaN" in str(exc.value.__cause__)
    raw, y, genes = _raw(constraint_missing=False)
    ens = _fit(_ens(["logistic_regression"]), engineer_features(raw), y, genes, EPOCH1_TEST_FIT)
    assert not is_projected(ens.trained_models_["logistic_regression"])
    assert type(ens.preprocessor_) is TabularModelPreprocessor and ens.member_projections_ == {}
    assert ens.fit_context_["protocol"] == "engineering_qualification" and ens.feature_semantics_ == "gvc.features/1"


def test_out_of_fold_projections_inside_the_ensemble_are_fold_local_exactly():
    """AF-OOF-PROJECTION-LEAKAGE-1 through VariantEnsemble.fit (calibration carve + gene-disjoint folds): every OOF
    surrogate is the median of the observed AF of THAT fold's training rows, never of rows it is evaluated on."""
    raw, y, genes = _raw(n=600, seed=3)
    X = engineer_features(raw, semantics=FS.V2_OBSERVED_AF)
    ens = VariantEnsemble(EnsembleConfig(n_jobs=1))
    ens.base_estimators = {"logistic_regression": AFProbe()}
    _fit(ens, X, y, genes, epoch2_test_fit(PP.COMMON_FOLD_MEDIAN))
    af_fit = X["af_raw"].to_numpy()[ens.oof_fit_indices_]
    oof, folds = ens.oof_predictions_[:, 0], ens.oof_fold_
    checked = 0
    for k in np.unique(folds):
        train_af = af_fit[folds != k]
        m_k = np.median(train_af[~np.isnan(train_af)])
        rows = (folds == k) & np.isnan(af_fit)
        assert np.array_equal(oof[rows], np.full(rows.sum(), m_k))
        checked += int(rows.sum())
    observed = ~np.isnan(af_fit)
    assert np.array_equal(oof[observed], af_fit[observed]) and checked > 0
    final = ens.trained_models_["logistic_regression"].projection_
    assert float.fromhex(final.statistics_["allele_frequency"]["median_binary64_hex"]) == np.median(af_fit[observed])


def test_a_capability_aware_fit_projects_each_member_and_serving_executes_exactly_those_projections():
    raw, y, genes = _raw(seed=4)
    X = engineer_features(raw, semantics=FS.V2_OBSERVED_AF)
    ens = _fit(_ens(["random_forest", "logistic_regression"]), X, y, genes, epoch2_test_fit(PP.CAPABILITY_AWARE))
    assert ens.preprocessor_ is None and ens.feature_semantics_ == "gvc.features/2"
    modes = {n: d["projection"]["mode"] for n, d in ens.member_projections_.items()}
    assert modes == {"random_forest": "native_nan", "logistic_regression": "fold_median"}
    assert type(ens.trained_models_["random_forest"]).__name__ == "_IsotonicCalibrator"   # calibrator stays outermost
    pipe = InferencePipeline.from_variant_ensemble(ens)
    scored = pipe.score(raw.head(40))
    assert scored.preprocessing_status == "fitted_member_projections"
    np.testing.assert_array_equal(scored.scores(), ens.predict_proba(X.head(40))[:, 1])      # bitwise parity
    # a different fit (other rows -> other fitted statistics) is a different predictor
    ens2 = _fit(_ens(["random_forest", "logistic_regression"]), X.iloc[50:].reset_index(drop=True),
                y.iloc[50:].reset_index(drop=True), genes.iloc[50:].reset_index(drop=True),
                epoch2_test_fit(PP.CAPABILITY_AWARE))
    assert InferencePipeline.from_variant_ensemble(ens2).predictor_contract_sha256() != pipe.predictor_contract_sha256()


def test_an_epoch_one_pipeline_cannot_render_epoch_two_input_and_keeps_its_identity():
    raw, y, genes = _raw(seed=5)
    legacy = blend_pipeline({"m": FixedProba([0.2])})
    assert legacy._declared_features()[2] == "af_is_absent"
    expected = canonical_digest("gvc.predictor-contract.v1", {
        "combiner_sha256": legacy.combiner.digest, "declared_features": legacy._declared_features(),
        "rendering": legacy.preprocessing_identity(), "class_order": [0, 1], "positive_class": 1})
    assert legacy.predictor_contract_sha256() == expected
    ens = _fit(_ens(["logistic_regression"]), engineer_features(raw, semantics=FS.V2_OBSERVED_AF), y, genes,
               epoch2_test_fit(PP.COMMON_FOLD_MEDIAN))
    v1 = engineer_features(raw)
    with pytest.raises(Exception):
        ens.predict_proba(v1)                                  # the epoch-1 roster is not the fitted roster


def test_an_availability_blinded_predictor_is_never_served():
    raw, y, genes = _raw(seed=6)
    X = engineer_features(raw, semantics=FS.V2_OBSERVED_AF)
    ens = _fit(_ens(["logistic_regression"]), X, y, genes, epoch2_test_fit(PP.AVAILABILITY_BLINDED))
    assert ens.member_projections_["logistic_regression"]["projection"]["mode"] == "availability_blinded_draw"
    with pytest.raises(ContractError, match="research only"):
        InferencePipeline.from_variant_ensemble(ens)


def test_the_af_free_arm_fits_without_the_family():
    raw, y, genes = _raw(seed=7)
    X = engineer_features(raw, semantics=FS.V2_OBSERVED_AF)
    X = X[list(roster_without_family(tuple(X.columns), AF_FAMILY))]
    ens = _fit(_ens(["logistic_regression"]), X, y, genes, epoch2_test_fit(PP.COMMON_FOLD_MEDIAN))
    assert ens.member_projections_["logistic_regression"]["projection"]["families"] == []


def test_the_saved_orchestrator_records_the_context_semantics_and_projections(tmp_path):
    raw, y, genes = _raw(seed=8)
    X = engineer_features(raw, semantics=FS.V2_OBSERVED_AF)
    ens = _fit(_ens(["logistic_regression"]), X, y, genes, epoch2_test_fit(PP.CAPABILITY_AWARE))
    ens.save(tmp_path / "ensemble.joblib")
    orchestrator = joblib.load(tmp_path / "ensemble.joblib")          # a file this test just wrote
    assert orchestrator["fit_context_"] == ens.fit_context_ and orchestrator["feature_semantics_"] == "gvc.features/2"
    assert orchestrator["member_projections_"] == ens.member_projections_ and orchestrator["preprocessor_"] is None
    member = joblib.load(tmp_path / "ensemble_models" / "logistic_regression.joblib")
    assert type(member) is ProjectedMember
    assert member.projection_.identity_ == ens.trained_models_["logistic_regression"].projection_.identity_
