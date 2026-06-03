"""Regression test for the float32 NaN-entropy bug in _decompose_uncertainty.

INCIDENT_2026-06-01_mc-dropout-nan-entropy: with eps=1e-8 the clip upper bound
(1-eps) rounds to 1.0 in float32, so fully-confident passes produced log(0)=NaN
in the aleatoric term. This test fails on the pre-patch code and passes after.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from genomic_variant_classifier.models.mc_dropout import _decompose_uncertainty


def _confident_float32_stack() -> np.ndarray:
    # Includes p=1.0 and p=0.0 (the cases that triggered the bug) plus mid values,
    # in float32 — the dtype the neural passes actually emit.
    return np.array(
        [[1.0, 0.0, 0.5, 0.9999, 1.0],
         [1.0, 0.0, 0.5, 0.0001, 1.0]],
        dtype=np.float32,
    )


def test_aleatoric_is_finite_on_confident_float32_passes():
    mean_prob, epistemic, aleatoric = _decompose_uncertainty(_confident_float32_stack())
    assert np.isfinite(mean_prob).all(), "mean_prob has non-finite values"
    assert np.isfinite(epistemic).all(), "epistemic has non-finite values"
    assert np.isfinite(aleatoric).all(), "aleatoric (entropy) has NaN/inf — float32 eps bug"


def test_no_numpy_warnings_emitted():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any divide-by-zero / invalid-value warning -> failure
        _decompose_uncertainty(_confident_float32_stack())


def test_entropy_is_nonnegative_and_bounded():
    # Binary entropy in nats is in [0, ln 2]; allow a hair over for fp slack.
    _, _, aleatoric = _decompose_uncertainty(_confident_float32_stack())
    assert (aleatoric >= -1e-9).all()
    assert (aleatoric <= np.log(2) + 1e-6).all()


def test_certain_passes_have_near_zero_entropy():
    # Column 0 is p=1.0 in every pass -> entropy ~ 0 (well under the eps-driven ceiling).
    _, _, aleatoric = _decompose_uncertainty(_confident_float32_stack())
    assert aleatoric[0] < 1e-3
