"""Which libraries in the roster actually enforce column order -- MEASURED, not assumed.

Added 2026-07-13.

WHY THIS FILE EXISTS
--------------------
`VariantEnsemble.fit` uses a 3-way dispatch:

    cnn_1d    -> X_seq            (one-hot DNA sequence)
    catboost  -> X_tab            (DataFrame; needs names for categorical resolution)
    else      -> X_tab.values     (raw ndarray -- column names DISCARDED)

The `.values` in that third branch looks like a wart. It is not. It is LOAD-BEARING, and
without this file the next person to read that dispatch will "clean it up" by passing
DataFrames everywhere -- and will silently corrupt every LightGBM prediction in the model
that classifies variant pathogenicity.

THE MEASUREMENT (2026-07-13)
----------------------------
Fit each library on a DataFrame, then predict with THE SAME DATA in the WRONG COLUMN ORDER:

    library                       result
    ----------------------------  --------------------------------------------------
    scikit-learn (RandomForest)   ValueError                          SAFE (refuses)
    XGBoost                       ValueError                          SAFE (refuses)
    CatBoost                      reordered by name, identical output SAFE (corrects)
    LightGBM                      *** SILENTLY WRONG ***              maps POSITIONALLY
                                  max delta 0.8553 in predicted probability,
                                  no error, no warning, even under -W error

LightGBM is the sole outlier in the roster. Its `feature_names_in_` attribute is DECORATIVE:
it is populated, it is reported, and it is never enforced.

Two consequences, and this file gates both:

  1. LightGBM must keep receiving an ndarray, so that column order is positional BY
     CONSTRUCTION rather than protected by a name-check that LightGBM silently ignores.
     The order is then guaranteed upstream by `engineer_features` -- the single source of
     truth (since 2026-07-11) -- which emits TABULAR_FEATURES in a fixed order behind the
     EXPECTED_TABULAR_FEATURE_COUNT fail-loud guard.

  2. The spurious "X does not have valid feature names" UserWarning suppressed in
     pyproject.toml is safe to suppress ONLY BECAUSE of (1). If LightGBM ever starts
     receiving a DataFrame, that filter would begin hiding a real signal. The dispatch test
     below is the premise the filter rests on.

THIS FILE IS ALSO A LIBRARY-UPGRADE TRIPWIRE. Every assertion records the behaviour of a
SPECIFIC library version. If a future LightGBM release fixes the positional-mapping defect,
`test_lightgbm_does_NOT_enforce_column_order` FAILS -- deliberately -- and tells us the
constraint can be revisited. A pinned assumption that is never re-derived becomes a lie on a
schedule (docs/ROADMAP.md, section 7, failure pattern (a)); this converts the assumption into
something that re-derives itself on every run.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from genomic_variant_classifier.models.variant_ensemble import (
    EnsembleConfig,
    VariantEnsemble,
)

COLS = ["sift_score", "cadd_phred", "allele_freq"]
SHUFFLED = ["allele_freq", "sift_score", "cadd_phred"]   # same data, wrong order


def _xy(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=COLS)
    y = (X["sift_score"] + X["cadd_phred"] > 0).astype(int)
    return X, y


def _predict_shuffled(model):
    """fit(DataFrame) -> predict(DataFrame with columns in the WRONG order).

    Returns (outcome, detail) where outcome is one of:
        "refused"   the library raised -- the mistake is impossible
        "corrected" the library reordered by name -- output identical
        "silent"    the library accepted it and returned DIFFERENT numbers
    """
    X, y = _xy()
    model.fit(X, y)
    p_ok = model.predict_proba(X)[:, 1]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            p_bad = model.predict_proba(X[SHUFFLED])[:, 1]
    except Exception as exc:
        return "refused", type(exc).__name__

    delta = float(np.abs(p_ok - p_bad).max())
    if np.allclose(p_ok, p_bad):
        return "corrected", delta
    return "silent", delta


# ---------------------------------------------------------------------------
# The outlier. This is the whole reason the dispatch discards column names.
# ---------------------------------------------------------------------------
def test_lightgbm_does_NOT_enforce_column_order():
    """LightGBM maps columns POSITIONALLY and lies about it.

    IF THIS TEST FAILS, THAT IS GOOD NEWS -- it means a LightGBM upgrade has fixed the
    defect, and the `.values` constraint plus the pyproject.toml warning filter can be
    revisited. Do not simply delete the test; re-measure, then update the contract.
    """
    lgb = pytest.importorskip("lightgbm")

    outcome, detail = _predict_shuffled(
        lgb.LGBMClassifier(n_estimators=20, verbose=-1, random_state=0)
    )

    assert outcome == "silent", (
        f"LightGBM's column-order behaviour has CHANGED (now: {outcome!r}, detail={detail}). "
        f"As of lightgbm 4.6.0 it accepted mis-ordered columns and returned silently wrong "
        f"predictions (max delta 0.8553). If it now refuses or corrects them, the ndarray "
        f"dispatch in VariantEnsemble.fit and the LGBMClassifier warning filter in "
        f"pyproject.toml were both justified by that defect and can be reconsidered. "
        f"RE-MEASURE before changing either."
    )
    assert detail > 0.1, (
        f"expected a large divergence from mis-ordered columns; got max delta {detail}"
    )


def test_lightgbm_invents_feature_names_from_a_bare_ndarray():
    """The root cause of the 11 spurious UserWarnings, pinned.

    scikit-learn leaves `feature_names_in_` UNSET when fitted on an ndarray. LightGBM
    fabricates 'Column_0', 'Column_1', ... -- so scikit-learn's predict-time check believes
    the model "was fitted with feature names" and warns when handed a nameless array. That
    warning is filtered in pyproject.toml. This test pins the premise of the filter.
    """
    lgb = pytest.importorskip("lightgbm")
    X, y = _xy()

    m = lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(X.values, y)
    names = list(getattr(m, "feature_names_in_", []))
    assert names == ["Column_0", "Column_1", "Column_2"], (
        f"LightGBM no longer fabricates synthetic feature names from an ndarray "
        f"(got {names!r}). The 'X does not have valid feature names' filter in "
        f"pyproject.toml exists solely because of this behaviour -- REMOVE THE FILTER if "
        f"this has changed, rather than leaving it to hide a future real warning."
    )

    rf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X.values, y)
    assert not hasattr(rf, "feature_names_in_"), (
        "scikit-learn is expected NOT to set feature_names_in_ from an ndarray; the "
        "asymmetry with LightGBM is the entire cause of the spurious warning"
    )


# ---------------------------------------------------------------------------
# The safe libraries. Asserted so that a REGRESSION in any of them is caught too.
# ---------------------------------------------------------------------------
def test_sklearn_refuses_mis_ordered_columns():
    outcome, detail = _predict_shuffled(
        RandomForestClassifier(n_estimators=20, random_state=0)
    )
    assert outcome == "refused", (
        f"scikit-learn must REFUSE mis-ordered columns (expected ValueError); got "
        f"{outcome!r} / {detail!r}"
    )


def test_xgboost_refuses_mis_ordered_columns():
    xgb = pytest.importorskip("xgboost")
    outcome, detail = _predict_shuffled(
        xgb.XGBClassifier(n_estimators=20, verbosity=0, random_state=0)
    )
    assert outcome == "refused", (
        f"XGBoost must REFUSE mis-ordered columns (expected ValueError); got "
        f"{outcome!r} / {detail!r}"
    )


def test_catboost_corrects_mis_ordered_columns_by_name():
    """CatBoost is the ONE model VariantEnsemble hands a DataFrame -- and that is safe.

    It reorders by name, so the DataFrame dispatch (needed for categorical-feature
    resolution) carries no column-order hazard. If this ever regresses to 'silent',
    CatBoost's dispatch becomes as dangerous as LightGBM's would be, and must change.
    """
    cb = pytest.importorskip("catboost")
    outcome, detail = _predict_shuffled(
        cb.CatBoostClassifier(iterations=20, verbose=0, random_seed=0)
    )
    assert outcome == "corrected", (
        f"CatBoost must reorder mis-ordered columns BY NAME (identical predictions); got "
        f"{outcome!r} / {detail!r}. VariantEnsemble dispatches catboost a DataFrame in fit, "
        f"predict_proba and evaluate. If CatBoost now maps positionally, that dispatch is "
        f"unsafe and must be changed to .values like the rest of the roster."
    )


# ---------------------------------------------------------------------------
# The dispatch itself: the premise everything above rests on.
# ---------------------------------------------------------------------------
def test_the_ensemble_dispatch_still_hands_lightgbm_an_ndarray():
    """LightGBM must NEVER be handed a DataFrame by VariantEnsemble.

    Read the dispatch out of the source rather than trusting a comment: catboost (and only
    catboost) is exempted from `.values`. cnn_1d takes X_seq. Everything else -- lightgbm
    included -- gets the raw ndarray.
    """
    import inspect

    src = inspect.getsource(VariantEnsemble.fit)

    assert 'name == "catboost"' in src, (
        "the fit() dispatch no longer special-cases catboost by name; re-read the dispatch "
        "and re-derive this contract from the code"
    )
    assert "X_tab_fit.values" in src, (
        "fit() no longer passes `X_tab_fit.values` to the non-catboost branch. If LightGBM "
        "is now receiving a DataFrame, its predictions are POSITIONALLY MAPPED and silently "
        "wrong whenever column order differs -- see test_lightgbm_does_NOT_enforce_column_"
        "order. Revert, or prove LightGBM has been fixed."
    )
    assert 'name == "lightgbm"' not in src, (
        "lightgbm has been given its own branch in the fit() dispatch. The ONLY safe input "
        "for it is the raw ndarray -- if that branch hands it a DataFrame, its predictions "
        "are positionally mapped and silently wrong on any column-order change."
    )


def test_lightgbm_is_in_the_roster_at_all():
    """Guard against this whole file quietly becoming vacuous."""
    pytest.importorskip("lightgbm")
    roster = VariantEnsemble(EnsembleConfig()).base_estimators
    assert "lightgbm" in roster, (
        "lightgbm is absent from the base-model roster -- if it was removed deliberately, "
        "the .values constraint and the pyproject.toml warning filter can both be revisited"
    )
