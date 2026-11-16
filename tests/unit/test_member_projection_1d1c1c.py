"""Stage 1D-1c.1C: the member projection (model_preprocessing schema version 2; owner ruling aa999b96; audit R1-R2, R7).

Author: Monzia Moodie
"""
from __future__ import annotations

import io
import pickle

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import KFold, cross_val_predict

from genomic_variant_classifier.models.feature_semantics import AF_FAMILY, FeatureSemantics as FS, derive_family
from genomic_variant_classifier.models.member import MemberRendering, ProjectedMember
from genomic_variant_classifier.models.model_preprocessing import (
    MemberProjection,
    ProjectionMode as PM,
    UndeclaredMissingnessError,
    blinded_surrogates,
    row_content_keys,
)

V2 = FS.V2_OBSERVED_AF.value
MODEL = AF_FAMILY.model_name_map()


def canonical(af, **other) -> pd.DataFrame:
    fam = derive_family(AF_FAMILY, np.asarray(af, dtype=float))
    d = {m: fam[m] for m in AF_FAMILY.members}
    if not other:
        other = {"other": np.arange(len(af), dtype=float)}
    d.update({k: np.asarray(v, dtype=float) for k, v in other.items()})
    return pd.DataFrame(d)


def proj(X, mode: PM) -> MemberProjection:
    return MemberProjection(tuple(X.columns), mode.value, V2)


def test_the_fold_median_is_the_median_of_observed_fit_rows_only():
    X = canonical([0.0, 0.1, np.nan, 1.0])
    p = proj(X, PM.FOLD_MEDIAN).fit(X.iloc[:3])
    assert float.fromhex(p.statistics_["allele_frequency"]["median_binary64_hex"]) == 0.05
    out = p.transform(X)
    assert out.loc[2, "af_model_raw"] == 0.05 and out.loc[3, "af_model_raw"] == 1.0
    assert out.loc[2, "af_model_log10"] == np.log10(0.05 + 1e-8)
    assert (out.loc[2, ["af_model_is_zero", "af_model_is_ultra_rare", "af_model_is_rare", "af_model_is_common"]]
            .tolist() == [0.0, 0.0, 0.0, 1.0])
    assert np.isnan(X.loc[2, "af_raw"]), "the canonical evidence is never mutated"


def test_an_imputed_zero_is_never_an_observed_zero():
    # AF-PROJECTION-OBSERVATION-CONFLATION-1: every observed value is zero, so the fitted median is zero.
    X = canonical([0.0, 0.0, np.nan])
    out = proj(X, PM.FOLD_MEDIAN).fit(X).transform(X)
    assert out.loc[2, "af_model_is_zero"] == 1.0          # the SURROGATE is zero ...
    assert np.isnan(X.loc[2, "af_is_observed_zero"])       # ... the evidence still says "not observed"
    assert "af_is_observed_zero" not in out.columns and "af_raw" not in out.columns


def test_no_indicator_is_emitted_and_the_width_is_unchanged():
    X = canonical([0.1, np.nan, 0.2])
    for mode in PM:
        out = proj(X, mode).fit(X).transform(X)
        assert out.shape == X.shape
        assert not any("missing" in c or "indicator" in c for c in out.columns)
        assert tuple(out.columns) == tuple(MODEL.get(c, c) for c in X.columns)


def test_native_mode_passes_nan_in_all_six_and_values_otherwise():
    X = canonical([0.1, np.nan, 0.0])
    out = proj(X, PM.NATIVE_NAN).fit(X).transform(X)
    assert out.loc[1, list(MODEL.values())].isna().all()
    assert out.loc[0, "af_model_raw"] == 0.1 and out.loc[2, "af_model_is_zero"] == 1.0


@pytest.mark.parametrize("reported", [0.0, 1e-12, 1e-4, 9.9e-4, 0.01, 1.0])
def test_observed_values_pass_through_exactly(reported):
    X = canonical([reported, np.nan])
    out = proj(X, PM.FOLD_MEDIAN).fit(X).transform(X)
    assert out.loc[0, "af_model_raw"] == reported


def test_the_fitted_statistic_does_not_move_with_the_rows_transformed():
    p = proj(canonical([1e-5, 5e-5, np.nan]), PM.FOLD_MEDIAN).fit(canonical([1e-5, 5e-5, np.nan]))
    before = p.identity_
    p.transform(canonical([0.8, 0.95, np.nan]))
    assert p.identity_ == before and float.fromhex(p.statistics_["allele_frequency"]["median_binary64_hex"]) == 3e-5


def test_every_family_member_is_checked_exactly_against_its_registered_derivation():
    X = canonical([0.1, np.nan])
    X.loc[0, "af_is_ultra_rare"] = 1.0
    with pytest.raises(ValueError, match="registered derivation"):
        proj(X, PM.FOLD_MEDIAN).fit(X)
    Y = canonical([0.1, np.nan])
    Y.loc[1, "af_is_rare"] = 0.0                     # a flag claimed where the frequency was not observed
    with pytest.raises(ValueError, match="registered derivation"):
        proj(Y, PM.NATIVE_NAN).fit(Y)


def test_a_partial_family_reordered_roster_and_anonymous_array_are_refused():
    X = canonical([0.1, np.nan])
    with pytest.raises(ValueError, match="partial"):
        p = MemberProjection(tuple(c for c in X.columns if c != "af_is_rare"), PM.FOLD_MEDIAN.value, V2)
        p.fit(X.drop(columns="af_is_rare"))
    p = proj(X, PM.FOLD_MEDIAN).fit(X)
    with pytest.raises(ValueError, match="order"):
        p.transform(X[list(reversed(X.columns))])
    with pytest.raises(TypeError):
        p.transform(X.to_numpy())


def test_epoch_one_columns_are_refused_by_an_epoch_two_projection():
    X = canonical([0.1, np.nan]).rename(columns={"af_is_observed_zero": "af_is_absent"})
    with pytest.raises(ValueError):
        MemberProjection(tuple(X.columns), PM.FOLD_MEDIAN.value, V2).fit(X)
    # An epoch-1 name with no family member beside it: only the epoch check can see it (the family checks see nothing).
    lone = pd.DataFrame({"af_is_absent": [0.0, 1.0], "other": [1.0, 2.0]})
    with pytest.raises(ValueError, match="belong to gvc.features/1"):
        MemberProjection(tuple(lone.columns), PM.NATIVE_NAN.value, V2).fit(lone)


def test_undeclared_missingness_outside_the_family_fails_closed():
    X = canonical([0.1, np.nan], other=[1.0, np.nan])
    for mode in PM:
        with pytest.raises(UndeclaredMissingnessError):
            proj(X, mode).fit(X)


def test_declared_gene_constraint_oe_is_a_fold_median_for_finite_members_and_nan_for_native():
    X = canonical([0.1, 0.2, np.nan], gene_constraint_oe=[0.2, np.nan, 0.6])
    fin = proj(X, PM.FOLD_MEDIAN).fit(X).transform(X)
    assert fin.loc[1, "gene_constraint_oe"] == 0.4
    nat = proj(X, PM.NATIVE_NAN).fit(X).transform(X)
    assert np.isnan(nat.loc[1, "gene_constraint_oe"])


def test_gene_is_constrained_missingness_is_refused_for_finite_members_until_ruled():
    X = canonical([0.1, 0.2, np.nan], gene_is_constrained=[1.0, np.nan, 0.0])
    for mode in (PM.FOLD_MEDIAN, PM.AVAILABILITY_BLINDED_DRAW):
        with pytest.raises(UndeclaredMissingnessError, match="no finite policy"):
            proj(X, mode).fit(X)
    assert np.isnan(proj(X, PM.NATIVE_NAN).fit(X).transform(X).loc[1, "gene_is_constrained"])


def test_an_all_missing_root_is_refused_for_a_finite_mode_and_accepted_natively():
    X = canonical([np.nan, np.nan])
    for mode in (PM.FOLD_MEDIAN, PM.AVAILABILITY_BLINDED_DRAW):
        with pytest.raises(UndeclaredMissingnessError, match="entirely unobserved"):
            proj(X, mode).fit(X)
    assert proj(X, PM.NATIVE_NAN).fit(X).transform(X)[list(MODEL.values())].isna().all().all()


def test_the_identity_binds_the_fitted_statistic_and_mode_and_survives_pickling():
    a = proj(canonical([0.1, 0.2, np.nan]), PM.FOLD_MEDIAN).fit(canonical([0.1, 0.2, np.nan]))
    b = proj(canonical([0.1, 0.3, np.nan]), PM.FOLD_MEDIAN).fit(canonical([0.1, 0.3, np.nan]))
    c = proj(canonical([0.1, 0.2, np.nan]), PM.NATIVE_NAN).fit(canonical([0.1, 0.2, np.nan]))
    assert len({a.identity_, b.identity_, c.identity_}) == 3
    buf = io.BytesIO(); joblib.dump(a, buf); buf.seek(0); r = joblib.load(buf)
    assert r.identity_ == a.identity_
    X = canonical([0.1, np.nan, 0.5])
    np.testing.assert_array_equal(r.transform(X).to_numpy(), a.transform(X).to_numpy())


def test_a_tampered_fitted_statistic_is_refused_at_transform():
    X = canonical([0.1, 0.2, np.nan])
    p = proj(X, PM.FOLD_MEDIAN).fit(X)
    p.statistics_["allele_frequency"]["median_binary64_hex"] = (0.5).hex()
    with pytest.raises(RuntimeError, match="identity"):
        p.transform(X)


def test_the_identity_binds_the_code_of_every_formula_and_a_code_change_refuses_to_serve(monkeypatch):
    from genomic_variant_classifier.models import model_preprocessing as MP
    X = canonical([0.1, 0.2, np.nan])
    p = proj(X, PM.FOLD_MEDIAN).fit(X)
    sem = "src/genomic_variant_classifier/models/feature_semantics.py"
    expected = {"src/genomic_variant_classifier/models/model_preprocessing.py::MemberProjection",
                f"{sem}::derive_family", f"{sem}::DerivationFamily"} | {
                f"{sem}::{fn}" for fn in AF_FAMILY.implementation_names().values()}
    code = MP.projection_code_identity(p.families_)
    assert set(code) == expected and all(len(v) == 64 for v in code.values())
    # the same projection, once the code of one formula is different, is a different projection: refused, not served
    real = MP._closure
    monkeypatch.setattr(MP, "_closure", lambda s: ("0" * 64) if s.endswith("::formula_af_is_rare") else real(s))
    with pytest.raises(RuntimeError, match="identity"):
        p.transform(X)


def test_clone_gives_an_unfitted_projection_with_the_same_parameters():
    X = canonical([0.1, np.nan])
    p = proj(X, PM.FOLD_MEDIAN).fit(X)
    q = clone(p)
    assert q.get_params() == p.get_params() and not hasattr(q, "identity_")


# ------------------------------------------------------------------ the availability-blinded draw (audit R1)
def test_blinded_draws_come_from_the_observed_values_and_are_batch_independent():
    rng = np.random.default_rng(11)
    n = 4000
    af = np.where(rng.random(n) < 0.3, np.nan, rng.beta(0.3, 30, n))
    X = canonical(af, other=rng.normal(size=n), other2=rng.integers(0, 50, n))
    p = proj(X, PM.AVAILABILITY_BLINDED_DRAW).fit(X)
    out = p.transform(X)
    miss = np.isnan(af)
    assert np.isin(out["af_model_raw"].to_numpy()[miss], af[~miss]).all()
    assert np.isfinite(out.to_numpy()).all()
    perm = rng.permutation(n)
    again = p.transform(X.iloc[perm])
    np.testing.assert_array_equal(again["af_model_raw"].to_numpy(), out["af_model_raw"].to_numpy()[perm])
    # no point mass: the most frequent surrogate value is a small share of the unobserved rows
    _, counts = np.unique(out["af_model_raw"].to_numpy()[miss], return_counts=True)
    assert counts.max() / miss.sum() < 0.01
    assert p.blinded_distinct_key_fraction_ > 0.99


def test_blinded_draws_are_coherent_and_a_median_creates_the_point_mass_they_remove():
    rng = np.random.default_rng(12)
    n = 3000
    af = np.where(rng.random(n) < 0.3, np.nan, rng.beta(0.3, 30, n))
    X = canonical(af, other=rng.normal(size=n))
    med = proj(X, PM.FOLD_MEDIAN).fit(X).transform(X)["af_model_raw"].to_numpy()
    m = np.median(af[~np.isnan(af)])
    assert ((med == m) == np.isnan(af)).mean() > 0.999          # the median IS the missingness indicator here
    blind = proj(X, PM.AVAILABILITY_BLINDED_DRAW).fit(X).transform(X)
    fam = derive_family(AF_FAMILY, blind["af_model_raw"].to_numpy())
    for c, mname in MODEL.items():
        np.testing.assert_array_equal(blind[mname].to_numpy(), fam[c])


def test_the_blinded_draw_needs_a_non_family_feature_to_key_rows():
    X = canonical([0.1, np.nan]).drop(columns="other")
    with pytest.raises(ValueError, match="non-family"):
        proj(X, PM.AVAILABILITY_BLINDED_DRAW).fit(X)


def test_row_keys_canonicalise_negative_zero_and_nan():
    a = row_content_keys(np.array([[0.0, np.nan]]), 7)
    b = row_content_keys(np.array([[-0.0, float("nan")]]), 7)
    assert a.tolist() == b.tolist()
    assert row_content_keys(np.array([[1.0, 2.0]]), 7).tolist() != row_content_keys(np.array([[1.0, 2.0]]), 8).tolist()
    sorted_obs = np.array([0.1, 0.2, 0.3])
    assert set(blinded_surrogates(sorted_obs, row_content_keys(np.arange(30.0).reshape(-1, 1), 1)).tolist()) <= {0.1, 0.2, 0.3}


# ------------------------------------------------------------------ fold-local OOF, exactly (AF-OOF-PROJECTION-LEAKAGE-1)
class AFProbe(ClassifierMixin, BaseEstimator):
    """Exposes the projected AF (column 0 of the rendered ndarray) as its probability: no learning at all."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.asarray(X)[:, 0].astype(float)
        return np.column_stack([1 - p, p])


def test_out_of_fold_medians_are_fold_local_exactly():
    af = [.1, .2, np.nan, np.nan, .7, .9, np.nan, np.nan]
    X = canonical(af)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    member = ProjectedMember(proj(X, PM.FOLD_MEDIAN), AFProbe(), MemberRendering.NDARRAY.value, "probe")
    p = cross_val_predict(member, X, y, cv=KFold(2, shuffle=False), method="predict_proba", n_jobs=1)[:, 1]
    assert np.isclose(p[2], .8) and np.isclose(p[3], .8)       # fold A's validation rows: fold B's 0.7 and 0.9
    assert np.isclose(p[6], .15) and np.isclose(p[7], .15)     # fold B's validation rows: fold A's 0.1 and 0.2
    final = clone(member).fit(X, y)
    assert float.fromhex(final.projection_.statistics_["allele_frequency"]["median_binary64_hex"]) == np.median([.1, .2, .7, .9])
