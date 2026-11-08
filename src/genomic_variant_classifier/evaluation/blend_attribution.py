"""src/genomic_variant_classifier/evaluation/blend_attribution.py

Author: Monzia Moodie

Exact member-score accounting of the SELECTED predictor (package 1, set 1b; owner decision ccffb8cf).

The selected predictor is the stored convex blend q_i = RN(sum_j w_j p_ij) over member probabilities p_ij. Relative to
a declared background with member means mu_j, every prediction decomposes as

    sum_j w_j p_ij  =  b  +  sum_j phi_ij,      b = sum_j w_j mu_j,      phi_ij = w_j (p_ij - mu_j)

exactly in real arithmetic. phi_ij is the contribution of member j's score to row i's deviation from the background
expectation, in probability units. For a model linear in its inputs with an independent background this is also the
interventional Shapley value of each input, so no sampling, kernel or surrogate is involved. Explanation-as-consumer:
the decomposition is computed FROM the bound capture of the selected predictor (its member blocks, its weights, its
result block), never by re-predicting, and its CLOSURE is recorded against the score actually published:

    r_i = q_i - (b + sum_j phi_ij)      computed exactly (math.fsum) and rounded once

Floating-point arithmetic makes r_i nonzero at the level of a few units in the last place; it is reported, never
hidden or absorbed.

What is NOT provided: a feature-level explanation of the blend. That would require every member's own explanation to
share one definition, one background and one unit; tree members, linear members, neural members and the sequence
model do not (owner decision ccffb8cf). It is reported as unavailable with that reason.
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from genomic_variant_classifier.models.combiner import exact_blend
from genomic_variant_classifier.prediction_contract import ContractError, canonical_digest

logger = logging.getLogger(__name__)

__all__ = ["FEATURE_LEVEL_EXPLANATION_UNAVAILABLE", "MemberScoreAttribution", "member_score_attribution"]

FEATURE_LEVEL_EXPLANATION_UNAVAILABLE = (
    "FEATURE_LEVEL_EXPLANATION_UNAVAILABLE (owner decision ccffb8cf): a feature-level explanation of the selected "
    "convex blend is defined only when every member's explanation shares one definition, one background and one unit. "
    "The fitted members do not: tree members expose split or path attributions, linear members coefficients, neural "
    "members gradients, and cnn_1d consumes a sequence input with no tabular features at all. The member-score "
    "accounting phi_ij = w_j (p_ij - mu_j) with a recorded closure residual is the explanation of the selected "
    "predictor that is defined (RunArtifactWriter.save_member_score_attributions).")


@dataclass(frozen=True, eq=False)
class MemberScoreAttribution:
    frame: pd.DataFrame
    descriptor: dict
    __hash__ = None  # type: ignore[assignment]


def _capture_parts(scored: Any) -> tuple[Any, tuple[str, ...], np.ndarray, np.ndarray]:
    for attr in ("result", "members", "combiner"):
        if not hasattr(scored, attr):
            raise ContractError("a bound split capture (ScoredSplit) is required; bare arrays are refused")
    names = tuple(scored.combiner.member_names)
    if len(scored.members) != len(names):
        raise ContractError("the capture's member blocks do not match its combiner")
    matrix = np.column_stack([np.asarray(b.positive_probability(), dtype=np.float64) for b in scored.members])
    scores = np.asarray(scored.result.positive_probability(), dtype=np.float64)
    return scored.combiner, names, matrix, scores


def member_score_attribution(scored: Any, background: Any) -> MemberScoreAttribution:
    """phi_ij = w_j (p_ij - mu_j) for every explained row, with mu from the declared background capture."""
    spec, names, P, q = _capture_parts(scored)
    bspec, bnames, B, _bq = _capture_parts(background)
    if bspec.digest != spec.digest:
        raise ContractError("the background was captured from a different predictor (combiner digests differ)")
    for explained, ref in zip(scored.members, background.members):
        if explained.binding.producer_sha256 != ref.binding.producer_sha256:
            raise ContractError("background member blocks come from different fitted members")
    weights = tuple(spec.weights)
    n_bg = B.shape[0]
    # mu_j: the correctly rounded exact sum, divided once (two roundings, recorded).
    mu = np.array([math.fsum(B[:, j].tolist()) / n_bg for j in range(B.shape[1])], dtype=np.float64)
    base = float(exact_blend(mu.reshape(1, -1), weights)[0])
    W = np.asarray(weights, dtype=np.float64)
    phi = (P - mu) * W                                 # binary64 subtraction, then binary64 multiplication
    residual = np.empty(len(q), dtype=np.float64)
    for i, (qi, row) in enumerate(zip(q.tolist(), phi.tolist())):
        residual[i] = math.fsum([qi, -base, *(-v for v in row)])
    abs_r = np.abs(residual)
    frame = pd.DataFrame({
        "source_observation_id": [r.source_observation_id for r in scored.result.rows.rows],
        "prediction_unit_id": [r.prediction_unit_id for r in scored.result.rows.rows],
        "variant_identity": [r.variant_identity for r in scored.result.rows.rows],
        "selected_score": q,
    })
    for j, name in enumerate(names):
        frame[f"phi_{name}"] = phi[:, j]
    frame["closure_residual"] = residual
    descriptor = {
        "schema": "gvc.member-score-attribution.v1",
        "definition": "phi_ij = w_j * (p_ij - mu_j); b = sum_j w_j * mu_j; closure r_i = q_i - (b + sum_j phi_ij)",
        "arithmetic": {"mu": "math.fsum(background column) / n_background (binary64)",
                       "base_value": "exact_binary64_products_sum_rne_v1 over (mu, w)",
                       "phi": "binary64 (p - mu), then binary64 product with w",
                       "closure_residual": "math.fsum of the binary64 terms (correctly rounded exact value)"},
        "units": "probability (the selected predictor's score scale)",
        "predictor": {"combiner_sha256": spec.digest, "kernel": spec.kernel, "members": list(names),
                      "weights_binary64_hex": [w.hex() for w in weights],
                      "prediction_stages": [m.prediction_stage_sha256 for m in spec.members]},
        "explained": {"result_block_sha256": scored.result.digest, "rows_sha256": scored.result.rows.digest,
                      "n_rows": int(len(q)), "split_sha256": getattr(scored, "split_sha256", None),
                      "member_block_sha256": [b.digest for b in scored.members]},
        "background": {"result_block_sha256": background.result.digest,
                       "rows_sha256": background.result.rows.digest, "n_rows": int(n_bg),
                       "split_sha256": getattr(background, "split_sha256", None),
                       "member_block_sha256": [b.digest for b in background.members],
                       "member_means_binary64_hex": [float(m).hex() for m in mu]},
        "base_value": base, "base_value_binary64_hex": base.hex(),
        "closure": {"max_abs_residual": float(abs_r.max()), "rows_nonzero": int((residual != 0).sum()),
                    "mean_abs_residual": math.fsum(abs_r.tolist()) / len(abs_r),
                    "residual_float64_sha256": hashlib.sha256(residual.astype("<f8").tobytes()).hexdigest()},
        "feature_level": {"status": "unavailable", "reason": FEATURE_LEVEL_EXPLANATION_UNAVAILABLE},
    }
    descriptor["descriptor_sha256"] = canonical_digest("gvc.member-score-attribution-descriptor.v1", descriptor)
    logger.info("Member-score attribution: %d rows x %d members; base value %.6g; max |closure residual| %.3g.",
                len(q), len(names), base, float(abs_r.max()))
    return MemberScoreAttribution(frame=frame, descriptor=descriptor)
