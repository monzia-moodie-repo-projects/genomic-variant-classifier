"""Confidence-Based Performance Estimation -- tests. RUN IN THE ISOLATED DRIFT ENVIRONMENT.

Added 2026-07-13 (roadmap 6.19).

WHY THIS DIRECTORY IS NOT `tests/`
----------------------------------
`nannyml` requires lightgbm<4.6; the ensemble TRAINS on lightgbm 4.6.0 (a base model). So
nannyml CANNOT be installed in the training environment, and these tests cannot run there.

The obvious move -- put them in `tests/` behind a module-level `pytest.importorskip("nannyml")`
-- is WRONG, and specifically it would break a guard built earlier today:

    A module-level importorskip collapses N tests into ONE skip entry. That makes the
    COLLECTED count environment-dependent (Continuous Integration would collect fewer items
    than a machine with nannyml). The suite-size ratchet (roadmap 6.14) asserts an EXACT
    collected count, and it only works because the environments now collect identically --
    a property that was only achieved this morning by closing the dependency gap (6.17).
    Re-introducing a conditional module-level skip would silently re-break it.

So: a SEPARATE TOP-LEVEL DIRECTORY, outside `testpaths = ["tests"]` in pyproject.toml.
`pytest tests/` will never see this. Enforced by CONFIGURATION, not by a comment someone has
to remember.

HOW TO RUN THESE
----------------
    python -m venv .venv-drift
    .venv-drift/Scripts/pip install -r requirements-drift.txt
    .venv-drift/Scripts/pip install -e . --no-deps
    .venv-drift/Scripts/pip install pytest
    .venv-drift/Scripts/python -m pytest tests_drift/ -v

Continuous Integration runs exactly this, in its own job.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.monitoring.performance_estimator import (
    COL_PRED,
    COL_PROBA,
    COL_TRUE,
    NannyMLUnavailableError,
    PerformanceEstimate,
    build_analysis_frame,
    build_reference_frame,
    estimate_performance,
)


def _synthetic(n: int, shift: float = 0.0, seed: int = 0):
    """A classifier whose inputs drift. `shift` moves the feature distribution."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(shift, 1.0, n)
    f2 = rng.normal(0.0, 1.0, n)
    logit = 1.2 * f1 + 0.8 * f2
    proba = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < proba).astype(int)
    features = pd.DataFrame({"cadd_phred": f1, "sift_score": f2})
    return features, proba, y


# ---------------------------------------------------------------------------
# 1. The environment itself. If these fail, nothing below means anything.
# ---------------------------------------------------------------------------
def test_nannyml_actually_imports_here():
    """The whole point of the isolated environment. Import is not a formality."""
    nml = pytest.importorskip("nannyml")
    assert hasattr(nml, "CBPE"), "nannyml imported but has no CBPE -- wrong package?"


def test_pyspark_is_NOT_installed_in_this_environment():
    """THE ISOLATION INVARIANT. If pyspark appears here, nannyml dies.

    nannyml does not DEPEND on pyspark -- but it OPTIONALLY IMPORTS it. pyspark 3.5.x's
    pandas module calls `np.NaN`, which NumPy REMOVED in 2.0. So on any machine where pyspark
    is importable, `import nannyml` raises AttributeError from inside pyspark.

    That is why nannyml never imported on the developer's laptop, and it is not fixed by any
    nannyml version. requirements-drift.txt therefore installs NO pyspark, deliberately.
    """
    import importlib.metadata as md

    with pytest.raises(md.PackageNotFoundError):
        md.version("pyspark")


def test_lightgbm_here_is_BELOW_4_6_and_training_is_NOT():
    """The reason this environment exists at all, asserted rather than remembered.

    nannyml requires lightgbm>=3.3,<4.6 (verified in BOTH 0.13.0 and 0.13.1 METADATA on
    2026-07-13). The ensemble trains on lightgbm 4.6.0. Downgrading a BASE MODEL to satisfy a
    MONITORING library would change the science to suit the instrument. So the environments
    are separate, and this is the assertion that keeps them that way.
    """
    import importlib.metadata as md
    from packaging.version import Version

    here = Version(md.version("lightgbm"))
    assert here < Version("4.6"), (
        f"lightgbm in the DRIFT environment is {here}, but nannyml requires <4.6. If this has "
        f"changed, nannyml has relaxed its cap -- RE-MEASURE and consider collapsing the two "
        f"environments back into one."
    )


def test_this_environment_MUST_NOT_be_able_to_unpickle_a_trained_model():
    """THE HARD BOUNDARY (roadmap 6.19).

    This environment holds LightGBM 4.5.0 and XGBoost 2.1.4 -- NOT the 4.6.0 / 3.2.0 the
    ensemble is trained with. Deserialising a trained booster here would load a 4.6.0 artifact
    into a 4.5.0 runtime: either a warning nobody reads, or silently wrong deserialisation.
    That is root pattern (d) -- a green result from a mutated environment.

    Confidence-Based Performance Estimation consumes PREDICTED PROBABILITIES, not the model,
    so the boundary costs nothing. This test pins it: the drift path must not even be able to
    reach the training-only libraries.
    """
    import importlib.metadata as md

    for forbidden, why in [
        ("torch", "the neural base models (tabular_nn, cnn_1d, mc_dropout, deep_ensemble, kan)"),
        ("torch_geometric", "the graph-neural-network branch"),
    ]:
        with pytest.raises(md.PackageNotFoundError):
            md.version(forbidden)


# ---------------------------------------------------------------------------
# 2. The frame builders. Fail at the door, not deep inside a third-party library.
# ---------------------------------------------------------------------------
def test_reference_frame_carries_labels_and_analysis_frame_does_not():
    feats, proba, y = _synthetic(500)
    ref = build_reference_frame(y_true=y, y_pred_proba=proba, features=feats)
    ana = build_analysis_frame(y_pred_proba=proba, features=feats)

    assert {COL_TRUE, COL_PROBA, COL_PRED}.issubset(ref.columns)
    assert COL_TRUE not in ana.columns, (
        "the analysis frame must NOT carry labels -- the entire point of CBPE is to estimate "
        "performance where the labels do not exist yet"
    )
    assert set(ref[COL_PRED].unique()) <= {0, 1}


def test_probabilities_outside_0_1_are_REJECTED():
    """Raw scores or logits produce confident nonsense. Refuse them at the door."""
    _, _, y = _synthetic(100)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build_reference_frame(y_true=y, y_pred_proba=np.full(100, 3.7))


def test_mismatched_lengths_are_REJECTED():
    """A length mismatch here silently mis-calibrates EVERY downstream estimate."""
    with pytest.raises(ValueError, match="same length"):
        build_reference_frame(y_true=np.zeros(100), y_pred_proba=np.zeros(50))


def test_passing_labels_in_the_ANALYSIS_frame_is_REJECTED():
    """If you have labels, MEASURE the metric. An estimate is strictly worse than a measurement."""
    feats, proba, y = _synthetic(500)
    ref = build_reference_frame(y_true=y, y_pred_proba=proba, features=feats)
    bad = build_analysis_frame(y_pred_proba=proba, features=feats)
    bad[COL_TRUE] = y  # the mistake

    with pytest.raises(ValueError, match="CONTAINS y_true"):
        estimate_performance(ref, bad)


# ---------------------------------------------------------------------------
# 3. THE CAPABILITY. Estimate performance on data with NO labels.
# ---------------------------------------------------------------------------
def test_cbpe_estimates_performance_on_UNLABELLED_data():
    """The capability requirements.in has claimed since May and which never existed."""
    ref_feats, ref_proba, ref_y = _synthetic(4000, shift=0.0, seed=0)
    ana_feats, ana_proba, _ana_y = _synthetic(4000, shift=0.0, seed=1)

    reference = build_reference_frame(ref_y, ref_proba, ref_feats)
    analysis = build_analysis_frame(ana_proba, ana_feats)   # NO labels

    est = estimate_performance(reference, analysis, metrics=("roc_auc", "accuracy"))

    assert isinstance(est, PerformanceEstimate)
    assert est.n_reference == 4000 and est.n_analysis == 4000
    assert set(est.analysis_summary) == {"roc_auc", "accuracy"}

    auc = est.analysis_summary["roc_auc"]
    assert 0.0 <= auc <= 1.0, f"estimated ROC AUC out of range: {auc}"
    assert auc > 0.7, (
        f"CBPE estimated ROC AUC {auc:.3f} on a well-separated synthetic problem. The "
        f"estimator is calibrated against the reference period, so a low estimate here means "
        f"the wiring is wrong, not the model."
    )
    assert not est.estimates.empty
    assert set(est.estimates["period"]) == {"analysis"}


def test_cbpe_returns_confidence_bounds_that_bracket_the_estimate():
    """An estimate without bounds is a number pretending to be a measurement."""
    ref_feats, ref_proba, ref_y = _synthetic(4000, seed=0)
    ana_feats, ana_proba, _ = _synthetic(4000, seed=1)

    est = estimate_performance(
        build_reference_frame(ref_y, ref_proba, ref_feats),
        build_analysis_frame(ana_proba, ana_feats),
        metrics=("roc_auc",),
    )
    rows = est.estimates
    assert (rows["lower"] <= rows["estimate"]).all(), "lower bound above the estimate"
    assert (rows["estimate"] <= rows["upper"]).all(), "estimate above the upper bound"


def test_a_broken_model_produces_a_LOW_estimate_and_the_alert_fires():
    """The test that matters: CBPE must NOTICE when the model is failing on real data.

    Feed the analysis period probabilities that are pure noise -- a model that has stopped
    discriminating. CBPE has no labels. It must still say so.
    """
    ref_feats, ref_proba, ref_y = _synthetic(4000, seed=0)
    reference = build_reference_frame(ref_y, ref_proba, ref_feats)

    rng = np.random.default_rng(7)
    ana_feats = pd.DataFrame(
        {"cadd_phred": rng.normal(size=4000), "sift_score": rng.normal(size=4000)}
    )
    noise = rng.uniform(0.45, 0.55, size=4000)     # a model that has stopped deciding
    analysis = build_analysis_frame(noise, ana_feats)

    est = estimate_performance(reference, analysis, metrics=("roc_auc",))
    auc = est.analysis_summary["roc_auc"]

    assert auc < 0.65, (
        f"CBPE estimated ROC AUC {auc:.3f} for a model emitting pure noise. It should collapse "
        f"toward 0.5. If it does not, the estimator is not actually reading the analysis "
        f"probabilities -- and a drift monitor that cannot see a dead model is worse than none."
    )


# ---------------------------------------------------------------------------
# 4. Fail-loud. This module must NEVER degrade into a logger.warning.
# ---------------------------------------------------------------------------
def test_missing_nannyml_RAISES_and_names_the_remediation(monkeypatch):
    """It must not `try/except ImportError -> logger.warning`.

    That is exactly what `_export_evidently` did: it logged "Evidently AI not installed. Run:
    pip install evidently" while Evidently WAS installed and the real fault was that the API
    had been deleted. The code misdiagnosed its own failure and the drift report was silently
    never produced, for months.
    """
    import builtins

    import genomic_variant_classifier.monitoring.performance_estimator as pe

    real_import = builtins.__import__

    def _no_nannyml(name, *args, **kwargs):
        if name == "nannyml" or name.startswith("nannyml."):
            raise ImportError("simulated: nannyml absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_nannyml)

    with pytest.raises(NannyMLUnavailableError) as exc:
        pe._import_nannyml()

    msg = str(exc.value)
    assert "requirements-drift.txt" in msg, "the error must name the remediation, not just the fault"
    assert "lightgbm" in msg, "the error must explain WHY it lives in a separate environment"
