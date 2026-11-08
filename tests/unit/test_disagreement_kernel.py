"""The near/far Bernoulli Kullback-Leibler kernel and its row-local status (package 1, gate G5.3).
Author: Monzia Moodie

Each numerical case below was a measured failure mode of an earlier kernel (AUDIT_OF_RULING_2026-11-07.md and the
owner's review bundle revision 4): cancellation near p == q, an inexact complement for p < 1/2, (p - q)/q rounding to
-1, underflow of d*d, and a rounded mixture mean landing on an endpoint.
"""
from __future__ import annotations

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

from genomic_variant_classifier.models.disagreement import bernoulli_kl, disagreement_with_status  # noqa: E402
from genomic_variant_classifier.prediction_contract import ContractError  # noqa: E402


def _oracle_jsd(values, dps=120):
    with mp.workdps(dps):
        v = [mp.mpf(float(x)) for x in values]
        q = mp.fsum(v) / len(v)

        def kl(p):
            if p == q:
                return mp.mpf(0)
            a = p * mp.log(p / q) if p > 0 else mp.mpf(0)
            b = (1 - p) * (mp.log1p(-p) - mp.log1p(-q)) if p < 1 else mp.mpf(0)
            return a + b
        return float(mp.fsum(kl(p) for p in v) / len(v))


def test_one_and_its_predecessor_is_resolved_not_an_endpoint():
    p = np.array([[1.0], [1.0 - 2.0 ** -53]])
    out = disagreement_with_status(p)
    assert out["status"] == ["resolved"] and out["complemented_rows"] == [True]
    assert out["values"][0] == pytest.approx(3.8477e-17, rel=1e-4)
    assert out["values"][0] == pytest.approx(_oracle_jsd([1.0, 1.0 - 2.0 ** -53]), rel=1e-12)


def test_a_subnormal_result_is_unresolved_with_a_reason():
    out = disagreement_with_status(np.array([[5e-308], [1e-307]]))
    assert out["status"] == ["unresolved"] and out["values"] == [None]
    assert out["reason"] == ["below_qualified_normal_domain"]


def test_a_mixture_mean_that_rounds_to_an_endpoint_is_unresolved():
    out = disagreement_with_status(np.array([[0.0], [5e-324]]))
    assert out["status"] == ["unresolved"] and out["reason"] == ["rounded_mixture_endpoint"]
    assert out["mixture_predictive_entropy_nats"] == [None]


def test_exact_agreement_is_an_exact_zero():
    out = disagreement_with_status(np.array([[0.25, 1.0], [0.25, 1.0], [0.25, 1.0]]))
    assert out["values"] == [0.0, 0.0] and out["status"] == ["exact_agreement"] * 2


def test_agreement_with_a_high_precision_oracle_across_scales():
    rng = np.random.default_rng(20261107)
    cases = []
    for scale in (1e-300, 1e-30, 1e-8, 1e-3, 0.3, 0.5, 0.9, 1 - 1e-9):
        for _ in range(12):
            base = scale * rng.uniform(0.5, 1.0)
            jitter = rng.choice([1e-15, 1e-9, 1e-3, 0.3]) * base
            col = np.clip([base, base + jitter * rng.uniform(-1, 1), base], 0.0, 1.0)   # test data only
            cases.append(col)
    stack = np.array(cases).T
    out = disagreement_with_status(stack)
    for i in range(stack.shape[1]):
        if out["status"][i] != "resolved":
            continue
        assert out["values"][i] == pytest.approx(_oracle_jsd(stack[:, i]), rel=1e-11), stack[:, i]


@pytest.mark.parametrize("p, q", [(1.5, 0.2), (0.2, -0.1), (np.nan, 0.2)])
def test_out_of_domain_input_is_refused(p, q):
    with pytest.raises(ContractError):
        bernoulli_kl(np.array([p]), np.array([q]))


def test_string_and_integer_input_is_refused():
    with pytest.raises(ContractError):
        bernoulli_kl(np.array([1]), np.array([0.5]))
    with pytest.raises(ContractError):
        disagreement_with_status(np.array([["0.1"], ["0.2"]]))
