"""Classify an engineered feature column into a correctness STATE.

HARNESS-NULL-1
==============
The correctness harness computed its silent-zero verdict as

    zero_rate = float((s.fillna(0) == 0).mean())

which makes NaN identical to zero inside the diagnostic. Measured 2026-08-09:
`gene_constraint_oe`, whose 200 values were ALL missing on the reference slice,
was reported as

    feature 'gene_constraint_oe' is 100% zero (>= 95%) and non-binary
    - probable silent-zero connector (connector-dead class)

Every value was NaN. Not one was zero.

That is the exact conflation this whole line of work exists to end -- an
absence reported as a measurement -- sitting inside the instrument built to
detect it. And it is about to get worse: once the declared missing-value policy
lands, more features will legitimately carry NaN, and every one of them would
be misreported as a dead connector.

WHY THIS IS A SEPARATE MODULE
==============================
There are already two definitions of "constant" and "near-constant" in this
repository: `feature_health.col_health` (the shared health-semantics authority,
DEFAULT_NEAR_CONSTANT_FRAC = 0.999) and the harness's own inline rule. Adding a
third would be the drift this session keeps repairing.

So the layering is:

    feature_health.py       COMPUTES observations   (what the column looks like)
    this module             APPLIES policy          (what that state means)
    MissingValuePolicy      DECLARES permission     (whether it is admissible)

`classify_feature_state` therefore takes a column and returns a state plus the
EVIDENCE it was derived from. It decides nothing about admissibility -- that is
the allowlist's job and the declared-missingness policy's job.

THE STATES, AND WHY EACH EXISTS
================================
    ALL_MISSING            every value NaN. Distinct from ALL_ZERO_OBSERVED,
                           which is what the old rule called it.
    MOSTLY_MISSING         at or above the missing-rate threshold.
    ALL_ZERO_OBSERVED      every OBSERVED value is 0.0 -- the genuine dead
                           connector, now separable from the missing case.
    MOSTLY_ZERO_OBSERVED   at or above the zero-rate threshold among observed.
    CONSTANT               one distinct observed value that is not zero.
    NEAR_CONSTANT          one value dominates the observed distribution.
    HEALTHY                none of the above.

Zero rate is computed AMONG OBSERVED VALUES ONLY. A column that is 90 per cent
missing and whose ten observed values are all zero is 100 per cent zero
*observed* and 90 per cent missing -- two findings, not one, and a single number
cannot carry both.

THE BINARY EXEMPTION
====================
The harness exempts columns whose values are exactly {0, 1}, because an
indicator that is mostly zero is doing its job. That exemption is preserved and
is applied to the OBSERVED values: a mask column with NaN in it is not a
well-formed indicator, and quietly treating it as one would hide a defect.

An all-zero column is NOT an indicator -- it never takes the other value -- and
falls through to the zero-rate rule, as it did before.

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Kept aligned with feature_health.DEFAULT_NEAR_CONSTANT_FRAC by
# `test_near_constant_threshold_matches_feature_health`, which imports the
# authority rather than restating its value here.
DEFAULT_NEAR_CONSTANT_FRAC = 0.999
DEFAULT_ZERO_RATE_THRESHOLD = 0.95
DEFAULT_MISSING_RATE_THRESHOLD = 0.95

_BINARY_SETS = ({0, 1}, {0.0, 1.0}, {0, 1.0}, {0.0, 1})


class FeatureState(str, Enum):
    ALL_MISSING = "all_missing"
    MOSTLY_MISSING = "mostly_missing"
    ALL_ZERO_OBSERVED = "all_zero_observed"
    MOSTLY_ZERO_OBSERVED = "mostly_zero_observed"
    CONSTANT = "constant"
    NEAR_CONSTANT = "near_constant"
    HEALTHY = "healthy"


#: The SILENT-ZERO class -- the connector-dead finding stage 5 has always made.
#: This set is deliberately EXACTLY the prior scope of that audit.
#:
#: CONSTANT and NEAR_CONSTANT are NOT here. Measured 2026-08-10: including them
#: added a stage-5 failure that had never existed, because the previous audit
#: had no constancy rule -- only a zero-rate rule. The suite gate caught it as
#: an unexpected failure and the whole edit was rolled back.
#:
#: HARNESS-NULL-1's mandate is to change how stage 5 DESCRIBES what it flags,
#: not WHICH features it flags. Widening an audit while repairing it makes the
#: repair unattributable: a new finding could be the widening or a genuine
#: regression, and nothing distinguishes them.
#:
#: Constancy is measured by feature_health.col_health and adjudicated by the
#: feature-vitality census, which own that question.
SILENT_ZERO_STATES = frozenset({
    FeatureState.ALL_ZERO_OBSERVED,
    FeatureState.MOSTLY_ZERO_OBSERVED,
})

#: Available to consumers that DO adjudicate constancy. Stage 5 does not.
CONSTANCY_STATES = frozenset({
    FeatureState.CONSTANT,
    FeatureState.NEAR_CONSTANT,
})

#: Missingness is adjudicated by the declared missing-value policy, never by a
#: silent-zero audit. Reporting a legitimately-missing feature as a dead
#: connector is the defect this module exists to end.
MISSINGNESS_STATES = frozenset({
    FeatureState.ALL_MISSING,
    FeatureState.MOSTLY_MISSING,
})


@dataclass(frozen=True)
class FeatureStateEvidence:
    """The observations the verdict was derived from. Frozen: a state without
    its evidence is an assertion, and this project has repeatedly found that
    a number reported without its basis cannot be audited."""
    n: int
    n_missing: int
    n_observed: int
    missing_rate: float
    zero_rate_observed: float
    n_distinct_observed: int
    modal_fraction_observed: float
    binary_observed: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def classify_feature_state(
    series,
    *,
    zero_rate_threshold: float = DEFAULT_ZERO_RATE_THRESHOLD,
    missing_rate_threshold: float = DEFAULT_MISSING_RATE_THRESHOLD,
    near_constant_frac: float = DEFAULT_NEAR_CONSTANT_FRAC,
):
    """Return (FeatureState, FeatureStateEvidence).

    Non-numeric columns are not classified here; the caller skips them, exactly
    as the harness already does.
    """
    s = pd.Series(series)
    n = int(len(s))
    if n == 0:
        return FeatureState.ALL_MISSING, FeatureStateEvidence(
            0, 0, 0, 0.0, 0.0, 0, 0.0, False)

    n_missing = int(s.isna().sum())
    missing_rate = n_missing / n
    observed = s.dropna()
    n_observed = int(len(observed))

    if n_observed == 0:
        # DISTINCT from ALL_ZERO_OBSERVED. The old rule reported this as
        # "100% zero", which named a connector defect for what is an absence.
        return FeatureState.ALL_MISSING, FeatureStateEvidence(
            n, n_missing, 0, missing_rate, float("nan"), 0, float("nan"), False)

    counts = observed.value_counts(dropna=True)
    n_distinct = int(len(counts))
    modal_fraction = float(counts.iloc[0] / n_observed)
    zero_rate_observed = float((observed == 0).mean())
    try:
        binary = set(np.unique(observed.to_numpy())) in _BINARY_SETS
    except (TypeError, ValueError):
        binary = False

    evidence = FeatureStateEvidence(
        n=n, n_missing=n_missing, n_observed=n_observed,
        missing_rate=missing_rate, zero_rate_observed=zero_rate_observed,
        n_distinct_observed=n_distinct, modal_fraction_observed=modal_fraction,
        binary_observed=binary)

    # MISSINGNESS IS ADJUDICATED FIRST, and separately. A mostly-missing column
    # is a missingness finding; whether its handful of observed values happen to
    # be zero is a second question the caller may still ask via the evidence.
    if missing_rate >= missing_rate_threshold:
        return FeatureState.MOSTLY_MISSING, evidence

    # A well-formed indicator takes BOTH values and is doing its job when mostly
    # zero. An all-zero column never takes the other value and is not exempt.
    if binary:
        return FeatureState.HEALTHY, evidence

    if zero_rate_observed >= 1.0:
        return FeatureState.ALL_ZERO_OBSERVED, evidence
    if zero_rate_observed >= zero_rate_threshold:
        return FeatureState.MOSTLY_ZERO_OBSERVED, evidence
    if n_distinct <= 1:
        return FeatureState.CONSTANT, evidence
    if modal_fraction >= near_constant_frac:
        return FeatureState.NEAR_CONSTANT, evidence
    return FeatureState.HEALTHY, evidence


def describe(name: str, state: FeatureState, ev: FeatureStateEvidence) -> str:
    """A message that names the OBSERVED basis, so no reader can mistake a
    missing column for a zero-filled one."""
    if state is FeatureState.ALL_MISSING:
        return ("feature {!r} is ENTIRELY MISSING ({} of {} rows NaN). This is an "
                "absence, NOT a zero-filled connector: whether it is admissible "
                "is decided by the declared missing-value policy.".format(
                    name, ev.n_missing, ev.n))
    if state is FeatureState.MOSTLY_MISSING:
        return ("feature {!r} is {:.0%} MISSING ({} of {} rows). Adjudicate "
                "against the declared missing-value policy, not the "
                "silent-zero allowlist.".format(name, ev.missing_rate,
                                                ev.n_missing, ev.n))
    if state is FeatureState.ALL_ZERO_OBSERVED:
        return ("feature {!r} is 100% zero across all {} OBSERVED value(s) and "
                "non-binary -- probable silent-zero connector "
                "(connector-dead class).".format(name, ev.n_observed))
    if state is FeatureState.MOSTLY_ZERO_OBSERVED:
        return ("feature {!r} is {:.0%} zero among {} OBSERVED value(s) and "
                "non-binary -- probable silent-zero connector "
                "(connector-dead class).".format(
                    name, ev.zero_rate_observed, ev.n_observed))
    if state is FeatureState.CONSTANT:
        return ("feature {!r} has ONE distinct observed value across {} "
                "observation(s).".format(name, ev.n_observed))
    if state is FeatureState.NEAR_CONSTANT:
        return ("feature {!r} is near-constant: one value covers {:.4%} of {} "
                "observation(s).".format(
                    name, ev.modal_fraction_observed, ev.n_observed))
    return "feature {!r} is healthy.".format(name)
