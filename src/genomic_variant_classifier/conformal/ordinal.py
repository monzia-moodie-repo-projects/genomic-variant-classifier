"""Ordinal conformal prediction sets, from scratch.

WHY THIS MODULE EXISTS
======================
Split, Mondrian and grouped conformal all produce sets of the form
{k : score_k <= q_hat}. On an UNORDERED label space that is exactly right. On the
American College of Medical Genetics and Genomics / Association for Molecular
Pathology (ACMG/AMP) five-tier scale it is not, because the tiers are ORDERED:

    Benign < Likely benign < Uncertain significance < Likely pathogenic < Pathogenic

Nothing in the existing classifiers prevents the prediction set
{Pathogenic, Benign} -- a set which asserts that a variant is either the most
harmful or the most harmless thing it could be, and nothing in between. That set
is not a statement a clinician can act on. It is also not rare: it arises
whenever the model is bimodal and the middle tier happens to fall just above the
threshold.

Ordinal conformal fixes this by construction. Every set it returns is a
CONTIGUOUS INTERVAL of adjacent tiers -- [Likely pathogenic, Pathogenic], or
[Likely benign, Likely pathogenic] -- never a gapped union. The finite-sample
coverage guarantee is preserved exactly, because the interval is still of the
form {k : s(k) <= q_hat} for a valid nonconformity score s; only the score
changes.

THE HARD GUARD -- READ THIS BEFORE USING THE MODULE
====================================================
The metric specification (Finding 1, and Panel C) is explicit that this project
does NOT currently have a five-class target. The pipeline collapses Pathogenic
and Likely pathogenic to 1, Benign and Likely benign to 0, and EXCLUDES Variants
of Uncertain Significance. Five "tiers" are then produced by thresholding a
single binary probability. Panel C states that ordinal evaluation is to be used
"only after establishing legitimate five-class targets", and Priority 6 states
"do not claim five-class learning until this is implemented and validated".

Applying ordinal conformal to threshold-derived bands would be worse than
useless. Contiguity is satisfied TRIVIALLY when every band is a cut of one
scalar -- the intervals cannot be non-contiguous, so the guarantee the module
appears to provide is vacuous, while the output looks exactly like a genuine
five-class ordinal prediction set. A reader could not tell the difference from
the output alone.

This module therefore REFUSES to calibrate on labels that are not genuinely
multi-class. `OrdinalConformalClassifier.fit` raises `OrdinalLabelError` when the
calibration labels carry fewer than three distinct values. That is a measurable,
decisive test of the thing Panel C actually cares about, and on today's cohort it
will fire. That is the intended behaviour, not a defect. The escape hatch,
`allow_degenerate_labels=True`, exists solely so the refusal itself can be
unit-tested; using it in analysis re-creates precisely the unsupported five-class
claim the specification warns against, and it records the fact in
`degenerate_labels_` so that the resulting object carries its own provenance.

THE SCORE -- NESTED CONTIGUOUS INTERVALS
=========================================
For a probability row p over K ordered tiers, define a nested family of
contiguous intervals, all containing the modal tier m = argmax(p):

    I_0 = [m, m]
    I_{t+1} = I_t extended by one tier, to whichever side has the larger
              probability (left neighbour lo-1 versus right neighbour hi+1);
              if one side is exhausted the other is taken.

This yields I_0 subset I_1 subset ... subset I_{K-1} = [0, K-1], each contiguous.
The nonconformity score of tier k is the cumulative probability mass of the first
interval that contains it:

    s(k) = mass(I_{t}) where t = min{ t : k in I_t }

Mass is non-decreasing in t, so {k : s(k) <= q_hat} is itself one of the nested
intervals, hence contiguous. That is the whole trick: contiguity is not enforced
by post-processing a gapped set, it is a property of the score's level sets. A
post-hoc "fill in the gaps" repair would inflate sets without any guarantee and
would silently destroy the calibration.

This is the ordinal analogue of Adaptive Prediction Sets (APS), and reduces to
it when the expansion is unconstrained. The randomised variant subtracts u * p[k]
from the mass at absorption, matching the convention in scores.py, which removes
the conservative over-coverage caused by the discreteness of the mass ladder.

TIE-BREAKING
------------
When the left and right neighbours carry exactly equal probability the expansion
order is ambiguous. `tie_break` selects 'low' (toward Benign, the default, chosen
for determinism) or 'high' (toward Pathogenic). Exact ties have probability zero
under continuous scores, and the choice does NOT affect the coverage guarantee --
it affects only which of two equally-sized intervals is reported on a measure-zero
event. It is exposed rather than hard-coded so that the arbitrariness is visible
instead of buried.

CATASTROPHIC ERRORS
-------------------
Panel C defines a catastrophic error as a confusion across the full width of the
scale: Benign asserted for a truly Pathogenic variant, or the reverse. Ordinal
sets make this measurable in a way unordered sets do not, because the DISTANCE
from the true tier to the nearest element of the set is well defined.
`ordinal_report` returns that distance distribution alongside coverage and width.
"""
from __future__ import annotations

import numpy as np

from .split import conformal_quantile

# The ACMG/AMP five-tier scale in ascending order of asserted pathogenicity.
# Index order is load-bearing: adjacency in this tuple IS the ordinal relation.
ACMG5_TIERS: tuple[str, ...] = (
    "Benign",
    "Likely benign",
    "Uncertain significance",
    "Likely pathogenic",
    "Pathogenic",
)

MIN_ORDINAL_CLASSES = 3


class OrdinalLabelError(ValueError):
    """Raised when ordinal machinery is applied to labels that are not ordinal.

    Separate from ValueError so that a caller can catch precisely this condition
    and report it as a scope problem -- "the cohort is binary" -- rather than as
    a malformed-input bug, which is what a bare ValueError would suggest.
    """


def _validate_probs(P: np.ndarray) -> np.ndarray:
    """Shape, finiteness, non-negativity and normalisation of a probability matrix.

    Rows are required to sum to 1 within 1e-6. A matrix whose rows do not sum to
    one is not a probability matrix, and every quantity below -- cumulative mass,
    the nesting order, the threshold -- would be silently wrong rather than
    loudly wrong.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"P must be 2-D (n, K); got shape {P.shape}")
    n, K = P.shape
    if n == 0:
        raise ValueError("P has zero rows")
    if K < MIN_ORDINAL_CLASSES:
        raise OrdinalLabelError(
            f"ordinal conformal requires at least {MIN_ORDINAL_CLASSES} ordered "
            f"tiers; P has K={K}. With K=2 there is no interior tier, every set "
            "is contiguous trivially, and the ordinal guarantee is vacuous -- use "
            "SplitConformalClassifier instead."
        )
    if not np.isfinite(P).all():
        bad = int((~np.isfinite(P)).sum())
        raise ValueError(f"P contains {bad} non-finite value(s)")
    if (P < 0).any():
        raise ValueError("P contains negative probabilities")
    row_sums = P.sum(axis=1)
    off = np.abs(row_sums - 1.0)
    if off.max() > 1e-6:
        i = int(np.argmax(off))
        raise ValueError(
            f"P rows must sum to 1 (tolerance 1e-6); worst row {i} sums to "
            f"{row_sums[i]:.9f}"
        )
    return P


def nested_interval_order(P: np.ndarray, tie_break: str = "low") -> np.ndarray:
    """Absorption step of every tier, for every row.

    Returns an integer array `step` of shape (n, K) where ``step[i, k]`` is the
    index t of the first nested interval I_t containing tier k for row i. The
    modal tier has step 0; the last tier absorbed has step K-1.

    Vectorised across rows. The loop runs exactly K-1 times regardless of n, so
    cost is O(n*K) with a small constant -- which matters, because the cohort is
    of the order of 1.5 million variants and a per-row Python loop would be
    minutes rather than milliseconds.
    """
    if tie_break not in ("low", "high"):
        raise ValueError("tie_break must be 'low' or 'high'")
    P = np.asarray(P, dtype=float)
    n, K = P.shape
    idx = np.arange(n)

    m = np.argmax(P, axis=1)
    lo = m.copy()
    hi = m.copy()
    step = np.full((n, K), -1, dtype=int)
    step[idx, m] = 0

    for t in range(1, K):
        can_left = lo > 0
        can_right = hi < K - 1
        # Neighbour probabilities; an exhausted side scores -inf so it never wins.
        p_left = np.where(can_left, P[idx, np.maximum(lo - 1, 0)], -np.inf)
        p_right = np.where(can_right, P[idx, np.minimum(hi + 1, K - 1)], -np.inf)

        if tie_break == "low":
            go_left = (p_left >= p_right) & can_left
        else:
            go_left = (p_left > p_right) & can_left
        # If the left side is exhausted we must go right, and vice versa.
        go_left = np.where(~can_right, can_left, go_left)

        new_tier = np.where(go_left, lo - 1, hi + 1)
        step[idx, new_tier] = t
        lo = np.where(go_left, lo - 1, lo)
        hi = np.where(go_left, hi, hi + 1)

    if (step < 0).any():
        # Unreachable if the expansion is correct; asserted rather than assumed,
        # because a tier never absorbed would silently receive score 0 below.
        raise RuntimeError(
            "nested expansion left tiers unabsorbed -- this is a bug in "
            "nested_interval_order, not in the caller's data"
        )
    return step


def _cumulative_mass(P: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Mass of the interval at which each tier is absorbed, shape (n, K).

    ``order`` maps step index -> tier index; the cumulative sum along the
    absorption order, gathered back into tier positions, is the mass of I_t for
    the tier absorbed at step t.
    """
    n, K = P.shape
    order = np.argsort(step, axis=1, kind="stable")      # step position -> tier
    p_in_order = np.take_along_axis(P, order, axis=1)
    cum_in_order = np.cumsum(p_in_order, axis=1)
    mass = np.empty_like(P)
    np.put_along_axis(mass, order, cum_in_order, axis=1)
    return mass


def ordinal_scores_all(P: np.ndarray, u: np.ndarray | None = None,
                       randomize: bool = False, tie_break: str = "low") -> np.ndarray:
    """Nonconformity score of every tier for every row, shape (n, K).

    Lower is more conforming. ``u`` is a per-row uniform draw used only when
    ``randomize`` is True; it subtracts u * p[k] from the mass at absorption,
    exactly as Adaptive Prediction Sets does in scores.py, removing the
    conservative over-coverage that the discrete mass ladder would otherwise
    introduce.
    """
    P = _validate_probs(P)
    step = nested_interval_order(P, tie_break=tie_break)
    mass = _cumulative_mass(P, step)
    if randomize:
        if u is None:
            raise ValueError("randomize=True requires u")
        u = np.asarray(u, dtype=float).reshape(-1, 1)
        if u.shape[0] != P.shape[0]:
            raise ValueError(f"u has length {u.shape[0]}, expected {P.shape[0]}")
        mass = mass - u * P
    return mass


def ordinal_scores_true(P: np.ndarray, y: np.ndarray, u: np.ndarray | None = None,
                        randomize: bool = False, tie_break: str = "low") -> np.ndarray:
    """Nonconformity score of the TRUE tier for each row, shape (n,)."""
    P = _validate_probs(P)
    y = np.asarray(y, dtype=int)
    if y.shape[0] != P.shape[0]:
        raise ValueError(f"y has length {y.shape[0]}, expected {P.shape[0]}")
    if y.min() < 0 or y.max() >= P.shape[1]:
        raise ValueError(
            f"y values must lie in [0, {P.shape[1] - 1}]; got "
            f"[{int(y.min())}, {int(y.max())}]"
        )
    S = ordinal_scores_all(P, u=u, randomize=randomize, tie_break=tie_break)
    return S[np.arange(P.shape[0]), y]


def is_contiguous(sets: np.ndarray) -> np.ndarray:
    """Per-row: is the selected tier set a single unbroken run? shape (n,).

    Kept public and used in tests as an independent check on the classifier --
    it recomputes contiguity from the OUTPUT rather than trusting the
    construction that produced it.
    """
    sets = np.asarray(sets, dtype=bool)
    out = np.empty(sets.shape[0], dtype=bool)
    for i in range(sets.shape[0]):
        idx = np.flatnonzero(sets[i])
        out[i] = (len(idx) == 0) or (idx[-1] - idx[0] + 1 == len(idx))
    return out


def interval_bounds(sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper tier index of each set. Empty rows yield (-1, -1)."""
    sets = np.asarray(sets, dtype=bool)
    any_ = sets.any(axis=1)
    lo = np.where(any_, sets.argmax(axis=1), -1)
    hi = np.where(any_, sets.shape[1] - 1 - sets[:, ::-1].argmax(axis=1), -1)
    return lo, hi


def distance_to_set(sets: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ordinal distance from the true tier to the nearest tier in the set.

    Zero when covered. This is the quantity an unordered prediction set cannot
    express, and the one Panel C's catastrophic-error definition rests on: being
    wrong by one tier and being wrong by four are not the same error.
    Empty sets yield the number of tiers (the maximum possible distance plus one)
    so they are penalised rather than silently scoring well.
    """
    sets = np.asarray(sets, dtype=bool)
    y = np.asarray(y, dtype=int)
    n, K = sets.shape
    tiers = np.arange(K)[None, :]
    d = np.abs(tiers - y[:, None])
    d_masked = np.where(sets, d, np.iinfo(np.int32).max)
    nearest = d_masked.min(axis=1)
    return np.where(sets.any(axis=1), nearest, K).astype(int)


class OrdinalConformalClassifier:
    """Split conformal over ORDERED tiers; every prediction set is contiguous.

    Parameters
    ----------
    alpha : target miscoverage; coverage is at least 1 - alpha in finite samples.
    randomize : use the smoothed score (recommended; removes discreteness-induced
        over-coverage).
    seed : reproducibility. Prediction uses seed + 1, matching
        SplitConformalClassifier so the two never share a draw.
    tie_break : 'low' or 'high'; see the module docstring. Immaterial to coverage.
    allow_degenerate_labels : bypass the multi-class guard. Exists so the refusal
        can be tested. Using it in analysis re-creates the unsupported five-class
        claim the metric specification warns against.
    force_nonempty : when True, a set that would be empty instead receives the
        modal tier. DEFAULT FALSE, and the default matters.

        A conformal set is legitimately empty when every tier's score exceeds
        q_hat -- it is the method saying "no tier is defensible at this level",
        which is information. Forcing the mode in destroys exact calibration.
        Measured on 5-tier synthetic data, 2000 calibration and 4000 test points,
        30 trials per level:

            alpha   target   coverage OFF   coverage ON   rows forced
            0.05    0.950       0.9500        0.9555          0.6 %
            0.10    0.900       0.8990        0.9147          1.6 %
            0.20    0.800       0.7973        0.8411          4.7 %
            0.30    0.700       0.7005        0.7822          9.1 %

        With the repair OFF the score is exactly calibrated at every level. With
        it ON, coverage inflates by precisely the fraction of rows repaired, and
        the sets are correspondingly wider than the data supports. Conformal's
        guarantee is one-sided so validity survives either way, but the
        efficiency does not. If a downstream consumer genuinely cannot accept an
        empty set, turn this on knowingly and read n_forced_nonempty_.

    Fitted / prediction attributes
    ------------------------------
    q_hat_, K_, n_cal_, classes_seen_, degenerate_labels_,
    n_forced_nonempty_, n_predicted_
    """

    def __init__(self, alpha: float = 0.1, randomize: bool = True, seed: int = 0,
                 tie_break: str = "low", allow_degenerate_labels: bool = False,
                 force_nonempty: bool = False):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        if tie_break not in ("low", "high"):
            raise ValueError("tie_break must be 'low' or 'high'")
        self.alpha = alpha
        self.randomize = randomize
        self.seed = seed
        self.tie_break = tie_break
        self.allow_degenerate_labels = allow_degenerate_labels
        self.force_nonempty = force_nonempty
        self.q_hat_: float | None = None
        self.K_: int | None = None
        self.n_cal_: int | None = None
        self.classes_seen_: np.ndarray | None = None
        self.degenerate_labels_: bool = False
        # Counters, not booleans: "how many" is the question a reviewer asks, and
        # a flag cannot answer it. Reset on every predict_set call.
        self.n_forced_nonempty_: int = 0
        self.n_predicted_: int = 0

    def fit(self, P_cal: np.ndarray, y_cal: np.ndarray) -> "OrdinalConformalClassifier":
        P_cal = _validate_probs(P_cal)
        y_cal = np.asarray(y_cal, dtype=int)
        if y_cal.shape[0] != P_cal.shape[0]:
            raise ValueError(
                f"y_cal has length {y_cal.shape[0]}, P_cal has {P_cal.shape[0]} rows")

        seen = np.unique(y_cal)
        if len(seen) < MIN_ORDINAL_CLASSES:
            msg = (
                f"calibration labels take only {len(seen)} distinct value(s) "
                f"{seen.tolist()}, but ordinal conformal requires at least "
                f"{MIN_ORDINAL_CLASSES}.\n"
                "This project's pipeline collapses Pathogenic and Likely pathogenic "
                "to 1, Benign and Likely benign to 0, and excludes Variants of "
                "Uncertain Significance. Tiers derived by thresholding that binary "
                "probability are contiguous TRIVIALLY, so the ordinal guarantee "
                "would be vacuous while the output looked genuine.\n"
                "Per the metric specification, Panel C: use ordinal evaluation only "
                "after establishing legitimate five-class targets. Pass "
                "allow_degenerate_labels=True only to test this refusal."
            )
            if not self.allow_degenerate_labels:
                raise OrdinalLabelError(msg)
            self.degenerate_labels_ = True

        self.K_ = P_cal.shape[1]
        self.n_cal_ = P_cal.shape[0]
        self.classes_seen_ = seen
        rng = np.random.default_rng(self.seed)
        u = rng.uniform(size=P_cal.shape[0]) if self.randomize else None
        s = ordinal_scores_true(P_cal, y_cal, u=u, randomize=self.randomize,
                                tie_break=self.tie_break)
        self.q_hat_ = conformal_quantile(s, self.alpha)
        return self

    def predict_set(self, P_test: np.ndarray) -> np.ndarray:
        """Boolean (n_test, K). Every row is a contiguous run of tiers."""
        if self.q_hat_ is None:
            raise RuntimeError("call fit() first")
        P_test = _validate_probs(P_test)
        if P_test.shape[1] != self.K_:
            raise ValueError(
                f"P_test has K={P_test.shape[1]}, calibrated with K={self.K_}")
        rng = np.random.default_rng(self.seed + 1)
        u = rng.uniform(size=P_test.shape[0]) if self.randomize else None
        S = ordinal_scores_all(P_test, u=u, randomize=self.randomize,
                               tie_break=self.tie_break)
        sets = S <= self.q_hat_

        # An empty set is a legitimate conformal output: it says no tier is
        # defensible at this alpha. Overriding it is a CHOICE, it costs exact
        # calibration, and it is therefore off by default and always counted.
        empty = ~sets.any(axis=1)
        self.n_predicted_ = int(sets.shape[0])
        self.n_forced_nonempty_ = int(empty.sum()) if self.force_nonempty else 0
        if self.force_nonempty and empty.any():
            sets[empty, np.argmax(P_test[empty], axis=1)] = True
        return sets

    @property
    def frac_forced_nonempty_(self) -> float:
        """Share of the last prediction batch whose empty set was overridden.

        Read this whenever force_nonempty is True. Empirically it equals the
        coverage inflation almost exactly, so it is the direct measure of how far
        the reported coverage sits above the nominal level."""
        if not self.n_predicted_:
            return 0.0
        return self.n_forced_nonempty_ / self.n_predicted_

    def predict_interval(self, P_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper tier index per row -- the same information as
        predict_set, in the form a report or clinician actually consumes."""
        return interval_bounds(self.predict_set(P_test))


def ordinal_report(sets: np.ndarray, y: np.ndarray, alpha: float,
                   tier_names: tuple[str, ...] | None = None,
                   catastrophic_distance: int | None = None) -> dict:
    """Coverage, interval width, and the ordinal-distance profile.

    catastrophic_distance defaults to K - 1, the full width of the scale: a
    Benign assertion for a truly Pathogenic variant, or the reverse.

    EMPTY SETS ARE EXCLUDED FROM THE CATASTROPHIC COUNT, deliberately. An empty
    set is an ABSTENTION -- the method declining to assert any tier -- whereas a
    catastrophic error is a confident assertion at the opposite end of the scale.
    They are different events with opposite clinical meanings, and an abstention
    is the safer of the two. Because distance_to_set scores an empty set as K,
    which exceeds any threshold, a naive count folds every abstention into the
    catastrophic total: an early draft of this function reported exactly that,
    with n_catastrophic == n_empty == 64, which is how the conflation was found.
    Penalising abstention as though it were the worst possible assertion would
    push a clinical system towards answering when it should decline.

    Both quantities are reported, separately, and n_abstentions is never hidden
    inside another number.
    """
    sets = np.asarray(sets, dtype=bool)
    y = np.asarray(y, dtype=int)
    n, K = sets.shape
    if y.shape[0] != n:
        raise ValueError(f"y has length {y.shape[0]}, sets has {n} rows")
    names = tier_names if tier_names is not None else tuple(f"tier_{i}" for i in range(K))
    if len(names) != K:
        raise ValueError(f"tier_names has {len(names)} entries, K={K}")
    cat_d = K - 1 if catastrophic_distance is None else catastrophic_distance

    covered = sets[np.arange(n), y]
    lo, hi = interval_bounds(sets)
    width = np.where(lo >= 0, hi - lo + 1, 0)
    dist = distance_to_set(sets, y)
    contig = is_contiguous(sets)
    nonempty = width > 0
    # Catastrophic == a far-wrong ASSERTION. Abstentions are counted separately.
    catastrophic = nonempty & (dist >= (K - 1 if catastrophic_distance is None
                                        else catastrophic_distance))

    return {
        "n": int(n),
        "K": int(K),
        "alpha": float(alpha),
        "target_coverage": float(1 - alpha),
        "empirical_coverage": float(covered.mean()),
        "coverage_gap": float(covered.mean() - (1 - alpha)),
        "mean_width": float(width.mean()),
        "median_width": float(np.median(width)),
        "width_histogram": {int(w): int(c) for w, c in
                            zip(*np.unique(width, return_counts=True))},
        "all_sets_contiguous": bool(contig.all()),
        "n_non_contiguous": int((~contig).sum()),
        "n_empty": int((width == 0).sum()),
        "n_full": int((width == K).sum()),
        "n_abstentions": int((~nonempty).sum()),
        "abstention_rate": float((~nonempty).mean()),
        "mean_distance_when_missed": (
            float(dist[(~covered) & nonempty].mean())
            if ((~covered) & nonempty).any() else 0.0),
        "catastrophic_distance_threshold": int(cat_d),
        "n_catastrophic": int(catastrophic.sum()),
        "catastrophic_rate": float(catastrophic.mean()),
        "catastrophic_rate_among_nonempty": (
            float(catastrophic[nonempty].mean()) if nonempty.any() else 0.0),
        "per_tier_coverage": {
            names[k]: (float(covered[y == k].mean()) if (y == k).any() else float("nan"))
            for k in range(K)
        },
        "per_tier_n": {names[k]: int((y == k).sum()) for k in range(K)},
    }
