"""Contract tests for the X_seq refusal (roadmap 6.28, Part 3, ff97c34..ea8d6e8).

`VariantEnsemble.fit` accepted `X_seq: pd.DataFrame` as a REQUIRED positional argument until
2026-07-19. Callers with no sequence windows could not say so, so they manufactured a value
instead. `scripts/train.py:523-525` documented the workaround in a comment -- "X_seq_train /
X_seq_test remain valid two-column placeholder DataFrames; with cnn_1d removed they satisfy the
seq-aware signatures but are unused" -- and three tests built `pd.Series(["A" * 101] * n)` for
the same reason, one of them annotated "# inert: cnn_1d is excluded below".

Part 3 (ff97c34) made None a legal value and added the refusal this file tests. On
2026-07-20 that refusal became `_require_sequence_windows`, which takes a MAPPING of
every sequence parameter rather than one argument -- because the predecessor checked
X_seq alone while fit() has two, so X_seq_cal_ext went unchecked and a run could fit
cnn_1d on real sequence while calibrating it on fabricated sequence, silently.

LINE NUMBERS ARE DELIBERATELY NOT QUOTED HERE. The three this paragraph used to cite
(1902, 2248, 2338) were all stale within a day: the module docstring rewrite at 84c6c54
shifted the file by +23 lines. Locate by name.

WHY A REFUSAL AND NOT A DEFAULT
--------------------------------
Optional does not mean tolerated. If cnn_1d is active and X_seq is None there is no honest way
to continue: fabricating a placeholder is precisely what roadmap 6.28 recorded as training the
sequence model on invented data and reporting a number. The refusal raises, names the model,
and states the remedy -- BEFORE any estimator is fitted, so a misconfigured run costs a second
rather than hours of paid compute.

WHAT THIS FILE GUARDS THAT NOTHING ELSE DOES
---------------------------------------------
Before it, `_require_x_seq` had no direct test. It was exercised only incidentally, by tests
that pass X_seq=None with a roster that happens to exclude cnn_1d -- which is the SILENT branch.
Nothing asserted that the loud branch is reachable at all. A guard whose firing path is never
executed is indistinguishable from a guard that has been disarmed, and this project has now
recorded three of those: the SpliceAI silent zero (9ba3127), rekey_seq_windows_v2's gate, and
train.py's `_POLY_WIN` content detector, which went unconditionally true when PLACEHOLDER_BASE
changed from "A" to "N" while the full suite stayed green.

Every positive test below is therefore paired with a NEGATIVE CONTROL that must stay silent.
A test that only ever sees the raise cannot tell a correct guard from one that raises always.

NO MODEL IS EVER FITTED HERE. The refusal fires before training by design, so exercising it
through fit() with a roster containing cnn_1d costs nothing; the predict_proba and evaluate
tests populate `trained_models_` directly rather than training a real convolutional neural
network to reach a guard that runs before training. Feature matrices use the real
TABULAR_FEATURES columns because `_assert_no_dead_features` (:2334) runs BEFORE the refusal
and would otherwise raise first -- a test that passes on the wrong exception proves nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.models.variant_ensemble import (
    SEQUENCE_MODELS,
    TABULAR_FEATURES,
    EnsembleConfig,
    VariantEnsemble,
)

# Fast tabular models only. cnn_1d is added per-test where the sequence branch is the subject.
FAST = {"logistic_regression"}


def _tab(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """A feature matrix `_assert_no_dead_features` accepts.

    Real TABULAR_FEATURES column names, random values so nothing is constant. The census at
    variant_ensemble.py:2334 runs BEFORE the refusal, so a matrix it rejects would raise the
    wrong exception and every `pytest.raises` below would pass for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.random((n, len(TABULAR_FEATURES))), columns=list(TABULAR_FEATURES))


def _labels(n: int = 40) -> pd.Series:
    return pd.Series([0] * (n // 2) + [1] * (n - n // 2))


def _seq(n: int = 200):
    """A real attachment -- what production passes, carrying its own provenance.

    Deliberately NOT a pd.Series, and since 2026-07-20 deliberately not a bare DataFrame
    either. On a Series `oh_alt - oh_ref` is identically zero: four of thirteen channels
    dead, eight duplicated. A bare frame cannot say whether its windows came from the
    reference genome or were invented to fill a column, so the gate refuses it.

    Built through the real tier-1 path -- windows plus an `ok` column, giving provenance
    "rows+ok" -- rather than by constructing WindowAttachment directly, so the fixture
    exercises the same resolution the production cohort does.

    n defaults to 200 because EnsembleConfig.seq_min_usable_rows is 100. A 40-row fixture
    would be refused for coverage, and a test that passes on the wrong refusal proves
    nothing -- the same trap _tab() documents about _assert_no_dead_features.
    """
    from genomic_variant_classifier.data.seq_window_join import attach_delta_windows
    return attach_delta_windows(pd.DataFrame({
        "fasta_seq_ref": ["ACGT" * 8] * n,
        "fasta_seq_alt": ["ACGA" * 8] * n,
        "ok": [True] * n,
    }))


def _ens(**kw) -> VariantEnsemble:
    return VariantEnsemble(EnsembleConfig(n_folds=2, **kw))


# ---------------------------------------------------------------------------
# The roster constant
# ---------------------------------------------------------------------------

def test_sequence_models_names_cnn1d_and_only_real_models():
    """SEQUENCE_MODELS exists so a second sequence model needs one edit, not three.

    Asserted against the ensemble's own default roster rather than a hardcoded list, so that
    renaming a model cannot leave this constant pointing at a name nothing builds.
    """
    assert "cnn_1d" in SEQUENCE_MODELS
    known = set(_ens().base_estimators)
    assert SEQUENCE_MODELS <= known, (
        f"SEQUENCE_MODELS names {sorted(SEQUENCE_MODELS - known)}, which the default roster "
        f"does not build. The constant has drifted from the models it describes."
    )


# ---------------------------------------------------------------------------
# _require_x_seq directly -- the full truth table
# ---------------------------------------------------------------------------

def test_require_x_seq_refuses_when_a_sequence_model_is_active_and_x_seq_is_none():
    with pytest.raises(ValueError) as exc:
        _ens()._require_sequence_windows(
            {"X_seq": None}, {"cnn_1d": object(), "xgboost": object()}, "fit")
    assert "cnn_1d" in str(exc.value)


def test_require_x_seq_is_silent_when_no_sequence_model_is_active():
    """NEGATIVE CONTROL. X_seq=None is the normal case for a tabular-only roster.

    Asserted explicitly rather than by the absence of an exception: a test body that is a
    bare call is indistinguishable from a stub, and this project has recorded three guards
    whose firing path was never executed.
    """
    got = _ens()._require_sequence_windows(
        {"X_seq": None}, {"xgboost": object(), "logistic_regression": object()}, "fit")
    assert got == {"X_seq": None}, (
        "the guard must pass every input through unchanged, not signal through a sentinel"
    )


def test_require_x_seq_is_silent_when_the_sequence_is_supplied():
    """NEGATIVE CONTROL. cnn_1d active WITH windows is the whole point of the branch."""
    got = _ens()._require_sequence_windows(
        {"X_seq": _seq()}, {"cnn_1d": object()}, "fit")
    assert isinstance(got["X_seq"], pd.DataFrame), (
        "a verified attachment must satisfy the guard, and the guard must resolve it to "
        "the 2-column frame the dispatch sites consume"
    )


def test_require_x_seq_is_silent_on_an_empty_roster():
    """NEGATIVE CONTROL. No models means nothing to refuse on behalf of."""
    got = _ens()._require_sequence_windows({"X_seq": None}, {}, "fit")
    assert got == {"X_seq": None}, (
        "an empty roster has no sequence model to refuse on behalf of"
    )


def test_the_refusal_names_the_model_the_flag_and_the_remedy():
    """A guard that fires without saying what to do sends the reader to the source.

    Every token below is load-bearing: the model that needs sequence, the launcher flag that
    removes it, the in-process equivalent, how to build the input it wants, and the
    fact that nothing has been trained yet -- so the reader knows the run can simply be
    relaunched.
    """
    with pytest.raises(ValueError) as exc:
        _ens()._require_sequence_windows({"X_seq": None}, {"cnn_1d": object()}, "fit")
    msg = str(exc.value)
    for token in ("cnn_1d", "--skip-cnn", "base_estimators.pop", "attach_delta_windows",
                  "not its `.windows`", "no compute was spent"):
        assert token in msg, f"the refusal message does not mention {token!r}:\n{msg}"


# ---------------------------------------------------------------------------
# Through the public API
# ---------------------------------------------------------------------------

def test_fit_refuses_before_training_anything(tmp_path):
    """The refusal must precede training, or it saves nothing worth saving.

    Asserted by state, not by timing: if any estimator had been fitted, trained_models_ would
    be non-empty. On a 4.4-million-row cohort the difference between refusing here and
    refusing after the first base model is hours of paid compute.
    """
    ens = _ens(model_dir=tmp_path / "models")
    ens.base_estimators = {k: v for k, v in ens.base_estimators.items()
                           if k in FAST or k == "cnn_1d"}
    assert "cnn_1d" in ens.base_estimators, "fixture precondition: cnn_1d must be active"

    with pytest.raises(ValueError, match="cnn_1d"):
        ens.fit(_tab(), None, _labels())

    assert not getattr(ens, "trained_models_", {}), (
        "the refusal fired only AFTER a model was trained; it must run before any fit"
    )
    assert not list((tmp_path / "models").glob("*")), (
        "the refusal fired only after artifacts were written to disk"
    )


def test_fit_accepts_none_for_a_tabular_only_roster(tmp_path):
    """NEGATIVE CONTROL, and the reason Part 3 exists.

    A tabular-only run passes None and trains normally. Before ff97c34 this call was
    impossible: the signature required a value, so callers fabricated one.
    """
    ens = _ens(model_dir=tmp_path / "models")
    ens.base_estimators = {k: v for k, v in ens.base_estimators.items() if k in FAST}
    ens.fit(_tab(n=60), None, _labels(n=60))
    assert ens.trained_models_, "a tabular-only roster with X_seq=None should train normally"


def test_predict_proba_refuses_when_a_sequence_model_is_trained_and_x_seq_is_none(tmp_path):
    """Prediction is refused on the same terms as training.

    trained_models_ is populated directly rather than by fitting a real convolutional neural
    network: the guard runs before any inference, so training one to reach it would test the
    fixture rather than the guard.
    """
    ens = _ens(model_dir=tmp_path / "models")
    ens.trained_models_ = {"cnn_1d": object()}
    with pytest.raises(ValueError, match="cnn_1d"):
        ens.predict_proba(_tab(), None)


def test_evaluate_refuses_when_a_sequence_model_is_trained_and_x_seq_is_none(tmp_path):
    """NEGATIVE-CONTROLLED BELOW: the same call with a tabular-only roster must not raise here.

    evaluate() is the path that produces the numbers a reader would quote, so an unnoticed
    None here is the one that reaches a results table.
    """
    ens = _ens(model_dir=tmp_path / "models")
    ens.trained_models_ = {"cnn_1d": object()}
    with pytest.raises(ValueError, match="cnn_1d"):
        ens.evaluate(_tab(), None, _labels())


def test_predict_proba_and_evaluate_are_silent_for_a_tabular_only_trained_roster(tmp_path):
    """NEGATIVE CONTROL for both API paths at once.

    Neither should reach the refusal; both fail LATER and for an unrelated reason, because the
    placeholder objects in trained_models_ are not real estimators. That is the point: the
    assertion is that the failure is NOT the sequence refusal.
    """
    ens = _ens(model_dir=tmp_path / "models")
    ens.trained_models_ = {"logistic_regression": object()}
    for call in (lambda: ens.predict_proba(_tab(), None),
                 lambda: ens.evaluate(_tab(), None, _labels())):
        with pytest.raises(Exception) as exc:
            call()
        assert "cnn_1d" not in str(exc.value), (
            "a tabular-only roster triggered the sequence refusal; the guard fires too broadly"
        )
