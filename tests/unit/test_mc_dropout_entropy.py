"""Float32 draws: no NaN, no NumPy warnings, exact zeros for certain passes (INCIDENT_2026-06-01).
Author: Monzia Moodie

INCIDENT_2026-06-01_mc-dropout-nan-entropy: with eps=1e-8 the clip upper bound (1-eps) rounded to 1.0 in float32, so
fully confident passes produced log(0)=NaN. Rewritten 2026-11-07 for the schema-2 decomposition, which never clips:
the entropy is exact (0 log 0 := 0), the precision of the producer is recorded, and the computation is float64.
"""
from __future__ import annotations

import warnings

import numpy as np

from genomic_variant_classifier.models.disagreement import draw_decomposition


def _confident_float32_stack() -> np.ndarray:
    # p=1.0 and p=0.0 (the cases that triggered the bug) plus mid values, in float32 as the neural passes emit them.
    return np.array([[1.0, 0.0, 0.5, 0.9999, 1.0],
                     [1.0, 0.0, 0.5, 0.0001, 1.0]], dtype=np.float32)


def test_every_component_is_finite_on_confident_float32_passes():
    r = draw_decomposition(_confident_float32_stack())
    for key in ("mixture_mean", "expected_draw_entropy_nats", "between_draw_spread_variance_probability_squared"):
        assert np.isfinite(r[key]).all(), key
    assert r["producer_precision"] == "float32"


def test_no_numpy_warnings_emitted():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        draw_decomposition(_confident_float32_stack())


def test_entropy_is_within_zero_and_log_two():
    e = draw_decomposition(_confident_float32_stack())["expected_draw_entropy_nats"]
    assert (e >= 0.0).all() and (e <= np.log(2.0)).all()


def test_certain_passes_have_exactly_zero_entropy_and_agree():
    r = draw_decomposition(_confident_float32_stack())
    assert r["expected_draw_entropy_nats"][0] == 0.0 and r["expected_draw_entropy_nats"][1] == 0.0
    assert r["disagreement_status"][:3] == ["exact_agreement"] * 3
    # Column 3 disagrees strongly (0.9999 vs 0.0001) and is resolved, not zero.
    assert r["disagreement_status"][3] == "resolved" and r["disagreement_nats"][3] > 0.5
