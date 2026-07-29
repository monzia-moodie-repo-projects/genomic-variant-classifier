"""tests/unit/test_tabular_nn_mc_dropout.py

Regression tests for TabularNNClassifier._predict_proba_single_pass()
and its integration with MCDropoutWrapper.

Background
----------
A2 anomaly (RUN_15_PLAN.md B.O3 / C.2): TabularNNClassifier did not
expose _predict_proba_single_pass(), causing MCDropoutWrapper to fall
back to the degenerate-uncertainty path in mc_dropout.py L238-241
(returning zero epistemic and aleatoric uncertainty). Per
scripts/write_session_docs.py L67-70, this "effectively reduces to a
single TabularNN" and warns 3 times during fit.

Fix: implement _predict_proba_single_pass() with selective dropout
activation (BatchNorm stays in eval; only Dropout layers get .train()).

Test classes:
  1. Contract           - hasattr / shape / probability validity (3 tests)
  2. Stochasticity      - dropout actually fires (3 tests)
  3. SideEffects        - no state leak, BatchNorm preserved (2 tests)
  4. Integration        - MCDropoutWrapper end-to-end (2 tests)
  5. ScientificProperties - theoretical claims (5 tests)
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from genomic_variant_classifier.models.mc_dropout import (
    MCDropoutWrapper,
    _decompose_uncertainty,
)
from genomic_variant_classifier.models.variant_ensemble import TabularNNClassifier

MC_DROPOUT_LOGGER = "genomic_variant_classifier.models.mc_dropout"


def _make_fitted_tabular_nn(n_samples=150, n_features=8, seed=42, epochs=3,
                            hidden_dims=(16, 8), dropout=0.3):
    """Build and fit a small TabularNN for testing.

    Tiny architecture + few epochs keeps runtime under a few seconds;
    tests assert structural / mechanical / theoretical properties rather
    than predictive accuracy.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    clf = TabularNNClassifier(
        hidden_dims=hidden_dims,
        dropout=dropout,
        epochs=epochs,
        batch_size=32,
        random_state=seed,
    )
    clf.fit(X, y)
    return clf, X, y


# ---------------------------------------------------------------------------
# 1. Contract: A2 regression guard for MCDropoutWrapper L216 hasattr check
# ---------------------------------------------------------------------------
class TestPredictProbaSinglePassContract:
    """A2 regression guard: method must satisfy MCDropoutWrapper L216 contract."""

    def test_method_exists(self):
        clf, _, _ = _make_fitted_tabular_nn()
        assert hasattr(clf, "_predict_proba_single_pass"),             "TabularNNClassifier must expose _predict_proba_single_pass for MCDropoutWrapper"

    def test_output_shape_is_n_by_two(self):
        clf, X, _ = _make_fitted_tabular_nn()
        proba = clf._predict_proba_single_pass(X[:50], seed=123)
        assert proba.shape == (50, 2), f"expected (50, 2) got {proba.shape}"

    def test_output_probabilities_valid(self):
        clf, X, _ = _make_fitted_tabular_nn()
        proba = clf._predict_proba_single_pass(X[:30], seed=7)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Stochasticity: verify dropout actually fires
# ---------------------------------------------------------------------------
class TestPredictProbaSinglePassStochasticity:
    """Verify dropout actually fires (not silently disabled)."""

    def test_same_seed_deterministic(self):
        clf, X, _ = _make_fitted_tabular_nn()
        a = clf._predict_proba_single_pass(X[:40], seed=999)
        b = clf._predict_proba_single_pass(X[:40], seed=999)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_stochastic(self):
        clf, X, _ = _make_fitted_tabular_nn()
        a = clf._predict_proba_single_pass(X[:40], seed=1)
        b = clf._predict_proba_single_pass(X[:40], seed=2)
        assert not np.array_equal(a, b),             "Same output with different seeds: dropout is NOT active (A2 regression)"

    def test_variance_across_seeds_nonzero(self):
        clf, X, _ = _make_fitted_tabular_nn()
        passes = np.stack([
            clf._predict_proba_single_pass(X[:25], seed=s)[:, 1]
            for s in range(20)
        ])
        per_sample_var = passes.var(axis=0)
        assert (per_sample_var > 0).any(),             "All-zero variance across 20 dropout passes; dropout is not stochastic"


# ---------------------------------------------------------------------------
# 3. SideEffects: no state leak, BatchNorm preserved
# ---------------------------------------------------------------------------
class TestPredictProbaSinglePassSideEffects:
    """No dropout state leak; BatchNorm preserved."""

    def test_no_leak_to_predict_proba(self):
        clf, X, _ = _make_fitted_tabular_nn()
        _ = clf._predict_proba_single_pass(X[:20], seed=42)
        a = clf.predict_proba(X[:20])
        b = clf.predict_proba(X[:20])
        np.testing.assert_array_equal(a, b)

    def test_batchnorm_not_corrupted_on_single_row(self):
        clf, X, _ = _make_fitted_tabular_nn()
        proba_single = clf._predict_proba_single_pass(X[:1], seed=42)
        assert np.all(np.isfinite(proba_single)),             "Single-row input produced NaN; BatchNorm leaked into train mode (A2 regression)"


# ---------------------------------------------------------------------------
# 4. Integration with MCDropoutWrapper
# ---------------------------------------------------------------------------
class TestMCDropoutWrapperIntegration:
    """End-to-end: MCDropoutWrapper(TabularNN) produces non-zero epistemic."""

    def test_no_missing_method_warning(self, caplog):
        _, X, y = _make_fitted_tabular_nn(n_samples=100)
        wrapper = MCDropoutWrapper(
            base_estimator=TabularNNClassifier(
                hidden_dims=(8,), dropout=0.3, epochs=2,
                batch_size=32, random_state=0,
            ),
            n_passes=5,
            random_state=0,
        )
        # Scope caplog to the mc_dropout module logger explicitly so propagation
        # settings don't cause the test to miss the warning.
        with caplog.at_level(logging.WARNING, logger=MC_DROPOUT_LOGGER):
            wrapper.fit(X, y)
        warning_msgs = [r.getMessage() for r in caplog.records]
        assert not any("does not expose _predict_proba_single_pass" in m for m in warning_msgs),             f"MCDropoutWrapper warned about missing method (A2 regression): {warning_msgs}"

    def test_epistemic_uncertainty_nonzero(self):
        _, X, y = _make_fitted_tabular_nn(n_samples=100)
        wrapper = MCDropoutWrapper(
            base_estimator=TabularNNClassifier(
                hidden_dims=(16, 8), dropout=0.3, epochs=3,
                batch_size=32, random_state=0,
            ),
            n_passes=10,
            random_state=0,
        )
        wrapper.fit(X, y)
        _mean, epi, _alea = wrapper.predict_with_uncertainty(X[:20])
        assert (epi > 0).any(),             "All epistemic uncertainties are zero; MC-dropout fallback path was hit (A2)"


# ---------------------------------------------------------------------------
# 5. Scientific properties: theoretical claims of MC-dropout
# ---------------------------------------------------------------------------
class TestPredictProbaSinglePassScientificProperties:
    """Scientific properties validating MC-dropout theoretical claims."""

    def test_predict_proba_auroc_floor_on_linearly_separable(self):
        """Sanity floor: on a linearly separable problem with 15 epochs,
        in-sample AUROC must exceed 0.85. Catches silent regressions
        where predict_proba returns constant or uninformative values."""
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(123)
        X = rng.standard_normal((300, 8)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        clf = TabularNNClassifier(
            hidden_dims=(32, 16), dropout=0.3, epochs=15,
            batch_size=32, random_state=0,
        ).fit(X, y)
        auroc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
        assert auroc > 0.85,             f"Linearly separable AUROC dropped to {auroc:.3f} (floor 0.85). predict_proba may be silently broken."

    def test_mean_of_k_passes_approximates_deterministic_predict_proba(self):
        """MC-dropout theoretical claim: E[stochastic_pass] approximately
        equals deterministic_pass for moderate dropout rates."""
        clf, X, _ = _make_fitted_tabular_nn(n_samples=200, epochs=5)
        K = 50
        deterministic = clf.predict_proba(X[:30])[:, 1]
        stochastic_mean = np.mean([
            clf._predict_proba_single_pass(X[:30], seed=s)[:, 1]
            for s in range(K)
        ], axis=0)
        np.testing.assert_allclose(
            stochastic_mean, deterministic, atol=0.1,
            err_msg="Mean of K MC-dropout passes diverged from deterministic predict_proba. "
                    "New method may be using a different network state than predict_proba.")

    def test_aleatoric_bounded_by_binary_entropy(self):
        """Information-theoretic bound: aleatoric cannot exceed log(2)."""
        clf, X, _ = _make_fitted_tabular_nn(n_samples=200)
        passes = np.stack([
            clf._predict_proba_single_pass(X[:40], seed=s)[:, 1]
            for s in range(20)
        ])
        _, _, alea = _decompose_uncertainty(passes)
        bound = np.log(2.0)
        assert alea.max() <= bound + 1e-9,             f"Aleatoric exceeded binary entropy bound log(2)={bound:.4f}: max={alea.max():.6f}"

    def test_aleatoric_higher_near_decision_boundary(self):
        """Calibration property: aleatoric uncertainty must peak near p=0.5."""
        # FIFTY EPOCHS, NOT FIVE (2026-07-29), and the precondition is now
        # ASSERTED rather than skipped.
        #
        # This test never ran. At five epochs the model has learned almost
        # nothing -- every prediction sat between 0.28 and 0.73 -- so no row was
        # confident enough to reach an extreme band, the precondition failed, and
        # the test skipped silently. One "s" in a 3,711-test run, since it was
        # written.
        #
        # MEASURED 2026-07-29 on this corpus with seed 42:
        #
        #      5 epochs  range [0.283, 0.731]  boundary 252  extreme  0  SKIPS
        #     10 epochs  range [0.226, 0.771]  boundary 193  extreme  0  SKIPS
        #     25 epochs  range [0.066, 0.840]  boundary  81  extreme  3  spans
        #     50 epochs  range [0.025, 0.919]  boundary  25  extreme 58  SPANS
        #
        # Twenty-five epochs is the CHEAPEST span but leaves only three extreme
        # rows -- an average over three samples, which would flicker back to
        # skipping on any small change. Fifty is healthy on both sides.
        #
        # A DESIGNED CORPUS WAS TRIED AND WAS WORSE: forcing half the rows close
        # to the separating plane needed THIRTY epochs rather than twenty-five,
        # because it adds ambiguous rows without adding confident ones. The
        # problem was never the data.
        clf, X, _ = _make_fitted_tabular_nn(n_samples=300, epochs=50)
        passes = np.stack([
            clf._predict_proba_single_pass(X, seed=s)[:, 1]
            for s in range(20)
        ])
        _, _, alea = _decompose_uncertainty(passes)
        mean_p = passes.mean(axis=0)
        near_boundary = (mean_p > 0.4) & (mean_p < 0.6)
        near_extreme = (mean_p < 0.1) | (mean_p > 0.9)

        # ASSERTED, NOT SKIPPED. A precondition that silently skips is a guard
        # reporting success while checking nothing -- the same defect as the
        # empty parameter set closed in d8d04ab, but invisible. If the corpus
        # ever stops spanning both regions, that is a FAILURE requiring the
        # fixture to be re-measured, not a reason to stop testing the property.
        assert near_boundary.any() and near_extreme.any(), (
            f"the corpus no longer spans both regions: {int(near_boundary.sum())} "
            f"near the boundary, {int(near_extreme.sum())} at the extremes, over "
            f"range [{mean_p.min():.3f}, {mean_p.max():.3f}]. Re-measure the "
            "training budget rather than restoring the skip.")
        assert alea[near_boundary].mean() > alea[near_extreme].mean(),             f"Aleatoric should peak near p=0.5 (binary entropy max). "             f"Got mean(boundary)={alea[near_boundary].mean():.4f} vs "             f"mean(extreme)={alea[near_extreme].mean():.4f}"

    def test_higher_dropout_rate_yields_higher_epistemic_uncertainty(self):
        """Empirical claim: dropout=0.5 yields more epistemic variance than dropout=0.1.

        Uses 5 epochs (was 3) to allow both models to diverge enough that the
        dropout-rate-driven mask-diversity difference shows up clearly in pass
        variance.
        """
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 8)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        clf_low = TabularNNClassifier(
            hidden_dims=(16, 8), dropout=0.1, epochs=5,
            batch_size=32, random_state=0,
        ).fit(X, y)
        clf_high = TabularNNClassifier(
            hidden_dims=(16, 8), dropout=0.5, epochs=5,
            batch_size=32, random_state=0,
        ).fit(X, y)
        epi_low = np.stack([
            clf_low._predict_proba_single_pass(X[:50], seed=s)[:, 1]
            for s in range(20)
        ]).var(axis=0).mean()
        epi_high = np.stack([
            clf_high._predict_proba_single_pass(X[:50], seed=s)[:, 1]
            for s in range(20)
        ]).var(axis=0).mean()
        assert epi_high > epi_low,             f"Higher dropout (0.5) should yield more epistemic variance than lower (0.1). "             f"Got epi_low={epi_low:.6f}, epi_high={epi_high:.6f}"
