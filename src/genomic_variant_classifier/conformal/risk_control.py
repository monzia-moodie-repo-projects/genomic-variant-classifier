"""Risk-controlling prediction sets: bound a CLINICAL risk, not just miscoverage.

Every other module in this package controls COVERAGE -- the probability that the
true label lands in the set. That is the right guarantee for a general classifier
and the wrong one for a clinical screen, where the quantity that matters is the
rate at which a genuinely pathogenic variant is NOT flagged. Coverage says
nothing about which errors happen; risk control names the error and bounds it.

    project_metrics.txt:909   Selective prediction   AURC and pathogenic FN rate after deferral
    project_metrics.txt:912   Five-tier safety       Catastrophic error rate

THE GUARANTEE
-------------
Given a nested family of set predictors indexed by a parameter lambda, a risk
R(lambda) that is NON-INCREASING in lambda, a target level alpha and a confidence
level delta, this module returns lambda_hat with

    P( R(lambda_hat) <= alpha ) >= 1 - delta

following Bates, Angelopoulos, Lei, Malik and Jordan (2021). The construction is

    lambda_hat = inf { lambda : Rplus(lambda') < alpha for every lambda' >= lambda }

where Rplus is a pointwise (1 - delta) upper confidence bound on the risk. The
"for every lambda' >= lambda" is not decoration: taking the infimum over the
point where the bound merely first dips below alpha would give a guarantee that
holds pointwise and not uniformly, which is a different and weaker statement.

MONOTONICITY IS A PRECONDITION, NOT AN ASSUMPTION
-------------------------------------------------
If R(lambda) is not non-increasing, the guarantee is void -- not approximate,
void. This module therefore checks it and RAISES, in the same spirit as
calibrate.py's alignment gate, which refuses to calibrate rather than calibrate
on a broken join. A small tolerance is allowed for Monte Carlo noise in an
estimated risk curve and it is a declared parameter, not a hidden constant.

THE BOUNDS, AND WHY THERE ARE THREE
-----------------------------------
hoeffding_upper_bound
    Closed form, valid for any loss bounded in [0, 1], and auditable by eye:
    r_hat + sqrt(log(1/delta) / (2n)). Loose, and the loosest bound is the one
    worth keeping when a reader needs to check the arithmetic without trusting
    an implementation.

clopper_pearson_upper_bound
    For a BINARY loss, which is what a false-negative rate is. Inverts the
    binomial tail rather than approximating it: it is the smallest R with
    P(Bin(n, R) <= k) <= delta. Because the binomial is discrete its coverage is
    AT LEAST 1 - delta and generally above it, so it is CONSERVATIVE. (Corrected
    2026-11-24: this paragraph said "neither conservative nor anticonservative".
    The exact coverage measured at delta = 0.10 was 0.907 to 1.000 over n in
    {20, 50, 200} and R in {0.01, 0.05, 0.10, 0.20}; tests/unit/
    test_conformal_reliability_1r.py recomputes it.) It is valid only for a whole
    number of binary events: inside control_risk a risk whose r_hat * n is not an
    integer is refused (NonBinaryRiskError), where it used to be rounded to the
    nearest count -- 1.05 events became 1, an anticonservative bound.

hoeffding_bentkus_upper_bound
    The bound Bates et al. actually recommend for a general [0, 1] loss: the
    tighter of a Hoeffding-style relative-entropy bound and a Bentkus binomial
    bound. Used as the default because it is valid beyond binary losses and is
    substantially tighter than plain Hoeffding at the sample sizes this project
    works at.

NO EXTERNAL DEPENDENCY, per this package's contract. The binomial tail is
computed in log space from `math.lgamma`, which is in the standard library, so
nothing here requires SciPy.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "RiskControlError",
    "NonMonotoneRiskError",
    "hoeffding_upper_bound",
    "clopper_pearson_upper_bound",
    "hoeffding_bentkus_upper_bound",
    "binomial_cdf",
    "false_negative_risk",
    "abstention_rate",
    "risk_curve",
    "control_risk",
    "risk_control_report",
    "NonBinaryRiskError",
    "predicted_class_deferral",
    "truth_class_routing",
]

_BOUNDS = ("hoeffding", "clopper_pearson", "hoeffding_bentkus")


class RiskControlError(RuntimeError):
    """Raised when a risk-control guarantee cannot be established."""


class NonMonotoneRiskError(RiskControlError):
    """Raised when the risk curve is not non-increasing in lambda.

    The Risk-Controlling Prediction Sets guarantee assumes a nested family, and a
    nested family has a non-increasing risk. Where that fails the guarantee is
    VOID rather than approximate, so this is a refusal and not a warning.
    """


class NonBinaryRiskError(RiskControlError, ValueError):
    """Raised when the Clopper-Pearson bound is asked to bound a risk that is not
    a whole number of binary events (r_hat * n not an integer). The bound inverts
    a binomial tail, so it has no meaning for a graded loss."""


# --------------------------------------------------------------------------- #
# The binomial tail, from scratch in log space.
# --------------------------------------------------------------------------- #
def _log_binom_pmf(k_max: int, n: int, p: float) -> np.ndarray:
    """log P(Bin(n, p) == k) for every k from 0 to k_max, as one array.

    BY RECURRENCE, NOT BY GAMMA FUNCTION:

        log_pmf[0] = n * log(1 - p)
        log_pmf[i] = log_pmf[i-1] + log((n - i + 1) / i) + log(p) - log(1 - p)

    which is a single cumulative sum over a numpy array. The first draft used
    `np.vectorize(math.lgamma)`, and numpy's own documentation says np.vectorize
    "is provided primarily for convenience, not for performance" and is
    "essentially a for loop". Every tail evaluation therefore made 2(k+1)
    Python-level calls, every bisection made about fifty tail evaluations, and
    the end-to-end simulation made 8,400 bisections.

    MEASURED 2026-07-30: the recurrence is 9x faster at n=200, 19x at n=1,000
    and 31x at n=5,000, and agrees with the gamma-function form to 1.2e-11 in
    log space over 6,410 comparisons, including both degenerate values of p.

    Everything stays in log space, so a tail of 1e-300 is representable rather
    than rounding to zero on the way.
    """
    if p <= 0.0:
        v = np.full(k_max + 1, -np.inf)
        v[0] = 0.0
        return v
    if p >= 1.0:
        v = np.full(k_max + 1, -np.inf)
        if k_max >= n:
            v[n] = 0.0
        return v
    base = n * math.log1p(-p)
    if k_max <= 0:
        return np.array([base])
    i = np.arange(1.0, k_max + 1.0)
    step = np.log(n - i + 1.0) - np.log(i) + math.log(p) - math.log1p(-p)
    return np.concatenate(([base], base + np.cumsum(step)))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(Bin(n, p) <= k), exact to floating point, no SciPy.

    Summed in log space and exponentiated once via the log-sum-exp trick, so a
    tail of 1e-300 is representable rather than rounding to zero.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    logs = _log_binom_pmf(k, n, p)
    finite = logs[np.isfinite(logs)]
    if finite.size == 0:
        return 0.0
    m = float(finite.max())
    return float(min(1.0, math.exp(m) * float(np.sum(np.exp(finite - m)))))


# --------------------------------------------------------------------------- #
# Upper confidence bounds on a risk.
# --------------------------------------------------------------------------- #
def _check_inputs(r_hat: float, n: int, delta: float) -> None:
    if not (0.0 <= r_hat <= 1.0):
        raise ValueError(f"r_hat must lie in [0, 1]; got {r_hat}")
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must lie in (0, 1); got {delta}")


def hoeffding_upper_bound(r_hat: float, n: int, delta: float) -> float:
    """r_hat + sqrt(log(1/delta) / (2n)), clipped to [0, 1].

    Valid for any loss bounded in [0, 1]. Deliberately the loosest of the three:
    a reader can check it with a calculator, which is worth keeping when the
    other two require trusting an implementation of a binomial tail.
    """
    _check_inputs(r_hat, n, delta)
    return float(min(1.0, r_hat + math.sqrt(math.log(1.0 / delta) / (2.0 * n))))


def clopper_pearson_upper_bound(k: int, n: int, delta: float) -> float:
    """The exact one-sided upper limit for a BINARY loss: the smallest R with
    P(Bin(n, R) <= k) <= delta.

    Found by bisection on R. The binomial cumulative distribution function is
    non-increasing in R for fixed k, so the bisection is well posed and converges
    to machine precision in about fifty steps.

    EXACT rather than approximate: for a false-negative rate -- which is a count
    of binary events -- this is the bound to use, and it is materially tighter
    than Hoeffding at the sample sizes a gene-disjoint calibration split gives.
    """
    if k < 0 or n <= 0 or k > n:
        raise ValueError(f"need 0 <= k <= n and n > 0; got k={k}, n={n}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must lie in (0, 1); got {delta}")
    if k == n:
        return 1.0
    lo, hi = float(k) / n, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if binomial_cdf(k, n, mid) <= delta:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-15:
            break
    return float(hi)


def _kl_bernoulli(a: float, b: float) -> float:
    """Relative entropy between Bernoulli(a) and Bernoulli(b), in nats."""
    if b <= 0.0:
        return math.inf if a > 0.0 else 0.0
    if b >= 1.0:
        return math.inf if a < 1.0 else 0.0
    total = 0.0
    if a > 0.0:
        total += a * math.log(a / b)
    if a < 1.0:
        total += (1.0 - a) * math.log((1.0 - a) / (1.0 - b))
    return total


def hoeffding_bentkus_upper_bound(r_hat: float, n: int, delta: float) -> float:
    """The tighter of a relative-entropy Hoeffding bound and a Bentkus binomial
    bound: the bound Bates et al. (2021) recommend for a general [0, 1] loss.

        Rplus = inf { R > r_hat : max(g_hoeffding(R), g_bentkus(R)) <= delta }

    with g_hoeffding(R) = exp(-n * kl(r_hat, R)) and
    g_bentkus(R) = e * P(Bin(n, R) <= ceil(n * r_hat)).

    Both tail functions are non-increasing in R above r_hat, so their maximum is
    too and bisection is well posed. Valid for any loss in [0, 1], which is why
    it is the default: a risk that is not a simple count -- a weighted or graded
    error rate -- still gets a valid bound.
    """
    _check_inputs(r_hat, n, delta)
    if r_hat >= 1.0:
        return 1.0
    k = int(math.ceil(n * r_hat))

    def tail(R: float) -> float:
        g_h = math.exp(-n * _kl_bernoulli(r_hat, R))
        g_b = math.e * binomial_cdf(k, n, R)
        return max(g_h, g_b)

    lo, hi = r_hat, 1.0
    if tail(hi) > delta:
        return 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tail(mid) <= delta:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-15:
            break
    return float(hi)


def _upper_bound(name: str, r_hat: float, n: int, delta: float) -> float:
    if name == "hoeffding":
        return hoeffding_upper_bound(r_hat, n, delta)
    if name == "clopper_pearson":
        events = r_hat * n
        k = round(events)
        # A count k / n carried in floating point reproduces k to within a few
        # units in the last place; 1e-9 * n is far wider than that and far
        # narrower than any graded loss worth bounding.
        if abs(events - k) > 1e-9 * max(1, n):
            raise NonBinaryRiskError(
                f"clopper_pearson bounds a count of binary events, but r_hat * n = "
                f"{events!r} is not an integer (r_hat={r_hat!r}, n={n}). Use "
                "'hoeffding_bentkus' for a graded loss; rounding to a count would "
                "under-state the risk.")
        return clopper_pearson_upper_bound(int(k), n, delta)
    if name == "hoeffding_bentkus":
        return hoeffding_bentkus_upper_bound(r_hat, n, delta)
    raise ValueError(f"unknown bound {name!r}; expected one of {_BOUNDS}")


# --------------------------------------------------------------------------- #
# The risks this project actually cares about.
# --------------------------------------------------------------------------- #
def false_negative_risk(sets: np.ndarray, y: np.ndarray,
                        positive_class: int = 1) -> float:
    """The fraction of TRULY POSITIVE rows whose prediction set omits the
    positive class -- a pathogenic variant the screen did not flag.

    Computed over the positive rows only, which is the point: a marginal error
    rate on a cohort that is 90 per cent benign is dominated by the benign rows
    and can look excellent while missing most of the pathogenic ones.

    Returns NaN when the cohort contains no positive row. A rate over an empty
    denominator is not zero, and reporting zero would read as perfect
    sensitivity on a cohort that never tested it.
    """
    sets = np.asarray(sets, dtype=bool)
    y = np.asarray(y, dtype=int)
    if sets.ndim != 2:
        raise ValueError("sets must be (n, K) boolean")
    if len(sets) != len(y):
        raise ValueError(f"length mismatch: sets {len(sets)} vs y {len(y)}")
    if not (0 <= positive_class < sets.shape[1]):
        raise ValueError(
            f"positive_class {positive_class} outside [0, {sets.shape[1]})")
    positive = y == positive_class
    if not positive.any():
        return float("nan")
    return float(np.mean(~sets[positive, positive_class]))


def abstention_rate(sets: np.ndarray) -> float:
    """The fraction of rows whose prediction set is EMPTY.

    Reported beside every risk because the two trade off directly: a procedure
    can drive any error rate to zero by abstaining on everything, and a risk
    figure quoted without its abstention rate does not say whether that is what
    happened.
    """
    sets = np.asarray(sets, dtype=bool)
    return float(np.mean(sets.sum(axis=1) == 0))


# --------------------------------------------------------------------------- #
# The search.
# --------------------------------------------------------------------------- #
def risk_curve(lambdas, risk_fn) -> np.ndarray:
    """Evaluate `risk_fn` at each lambda, in the given order."""
    return np.asarray([float(risk_fn(float(l))) for l in lambdas], dtype=float)


def _assert_non_increasing(lambdas: np.ndarray, risks: np.ndarray,
                           tolerance: float) -> None:
    rises = np.diff(risks)
    bad = np.flatnonzero(rises > tolerance)
    if bad.size:
        i = int(bad[0])
        raise NonMonotoneRiskError(
            f"MONOTONICITY GATE FAILED: the risk RISES from {risks[i]:.6f} to "
            f"{risks[i + 1]:.6f} between lambda={lambdas[i]:g} and "
            f"lambda={lambdas[i + 1]:g}, by {rises[i]:.3e} against a tolerance "
            f"of {tolerance:.3e} ({bad.size} such step(s) in total). The "
            "risk-controlling guarantee assumes a NESTED family, whose risk is "
            "non-increasing; where that fails the guarantee is void rather than "
            "approximate. Refusing to return a threshold.")


def control_risk(lambdas, risks, n: int, alpha: float, delta: float,
                 bound: str = "hoeffding_bentkus",
                 monotone_tolerance: float = 0.0) -> dict:
    """Choose the smallest lambda whose risk is bounded by alpha with confidence
    1 - delta, and report every quantity the choice rests on.

    `lambdas` must be ASCENDING and `risks` non-increasing along it. The returned
    dictionary carries `lambda_hat=None` when no lambda controls the risk, which
    is a legitimate answer -- the honest one when the calibration set is too
    small or the target too tight -- and not an error.

    THE SUFFIX RULE. lambda_hat is the smallest lambda from which the bound stays
    below alpha for EVERY larger lambda, not merely the first lambda where it
    dips below. The distinction matters whenever the bound is non-monotone
    through sampling noise, and it is what makes the guarantee uniform rather
    than pointwise.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    risks = np.asarray(risks, dtype=float)
    if lambdas.ndim != 1 or risks.ndim != 1:
        raise ValueError("lambdas and risks must be one-dimensional")
    if lambdas.size != risks.size:
        raise ValueError(
            f"length mismatch: {lambdas.size} lambdas against {risks.size} risks")
    if lambdas.size == 0:
        raise ValueError("no lambdas to search")
    if np.any(np.diff(lambdas) <= 0):
        raise ValueError("lambdas must be strictly ascending")
    if not np.isfinite(risks).all():
        raise ValueError("the risk curve contains a non-finite value")
    if np.any(risks < 0.0) or np.any(risks > 1.0):
        raise ValueError("risks must lie in [0, 1]")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}")
    if bound not in _BOUNDS:
        raise ValueError(f"unknown bound {bound!r}; expected one of {_BOUNDS}")
    if monotone_tolerance < 0.0:
        raise ValueError("monotone_tolerance must be non-negative")

    _assert_non_increasing(lambdas, risks, monotone_tolerance)

    upper = np.asarray([_upper_bound(bound, float(r), n, delta) for r in risks],
                       dtype=float)
    below = upper < alpha
    # suffix_all[i] is True when every entry from i onward is below alpha.
    suffix_all = np.cumprod(below[::-1])[::-1].astype(bool)
    idx = int(np.argmax(suffix_all)) if suffix_all.any() else None

    return {
        "bound": bound,
        "n": int(n),
        "alpha": float(alpha),
        "delta": float(delta),
        "lambdas": lambdas,
        "risks": risks,
        "upper_bounds": upper,
        "controls": below,
        "lambda_hat": (float(lambdas[idx]) if idx is not None else None),
        "index_hat": idx,
        "risk_at_lambda_hat": (float(risks[idx]) if idx is not None else None),
        "upper_at_lambda_hat": (float(upper[idx]) if idx is not None else None),
        "guarantee": ("P(risk <= alpha) >= 1 - delta"
                      if idx is not None else "no lambda controls the risk"),
    }


def risk_control_report(lambdas, risks, n: int, alpha: float, delta: float,
                        bound: str = "hoeffding_bentkus",
                        monotone_tolerance: float = 0.0,
                        abstentions=None) -> dict:
    """`control_risk` plus the abstention rate at each lambda, when supplied.

    A risk figure without its abstention rate does not say whether the procedure
    controlled the error or simply declined to answer, so the two are reported
    together or the caller is told the second is missing.
    """
    out = control_risk(lambdas, risks, n, alpha, delta, bound=bound,
                       monotone_tolerance=monotone_tolerance)
    if abstentions is None:
        out["abstention"] = None
        out["abstention_note"] = (
            "not supplied; a risk controlled by abstaining is not a risk "
            "controlled, and this report cannot tell which happened")
        return out
    abstentions = np.asarray(abstentions, dtype=float)
    if abstentions.shape != out["lambdas"].shape:
        raise ValueError(
            f"abstentions has shape {abstentions.shape}, expected "
            f"{out['lambdas'].shape}")
    out["abstention"] = abstentions
    i = out["index_hat"]
    out["abstention_at_lambda_hat"] = (float(abstentions[i])
                                       if i is not None else None)
    out["abstention_note"] = None
    return out


# --------------------------------------------------------------------------- #
# Fixed-quota predicted-class deferral (2026-11-24) -- DESCRIPTIVE, NOT RISK CONTROL
# --------------------------------------------------------------------------- #
# A review budget is a workload policy, not an error bound: nothing below gives a
# guarantee on any risk. It is the operational baseline the threshold-aware
# selective-prediction study (BioData Mining, 2026) reduces to: with a fixed quota
# per PREDICTED class, deferring by its threshold-rescaled entropy and deferring by
# distance to the threshold choose the same rows, so the simpler rule is the one
# kept. Thresholds and quotas are declared inputs, never fitted on the rows they
# route.
def _scores_and_threshold(scores, threshold) -> np.ndarray:
    s = np.asarray(scores)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if s.dtype.kind not in "fiu":
        raise ValueError(f"scores must be real numbers; got dtype {s.dtype}")
    s = s.astype(float)
    if not np.isfinite(s).all() or np.any(s < 0.0) or np.any(s > 1.0):
        raise ValueError("scores must be finite and lie in [0, 1]")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.floating)) or not (
            math.isfinite(float(threshold)) and 0.0 < float(threshold) < 1.0):
        raise ValueError(f"threshold must be a declared number strictly inside (0, 1); got {threshold!r}")
    return s


def _quota(count: int, fraction) -> int:
    """floor(count * fraction) in EXACT arithmetic on the fraction as declared.

    The float product is not: 100 * 0.29 is 28.999999999999996 and floors to 28.
    A float is read through its shortest decimal representation (str), which is
    the number the caller wrote; a fractions.Fraction is taken as it is.
    """
    from fractions import Fraction
    if isinstance(fraction, bool):
        raise ValueError("a deferral fraction must be a number, not a boolean")
    if isinstance(fraction, Fraction):
        f = fraction
    elif isinstance(fraction, (int, float, np.floating, np.integer)) and math.isfinite(float(fraction)):
        f = Fraction(str(float(fraction))) if isinstance(fraction, (float, np.floating)) else Fraction(int(fraction))
    else:
        raise ValueError(f"a deferral fraction must be a finite number; got {fraction!r}")
    if not (0 <= f <= 1):
        raise ValueError(f"a deferral fraction must lie in [0, 1]; got {fraction!r}")
    return math.floor(count * f)


def predicted_class_deferral(scores, threshold: float, *, negative_fraction,
                             positive_fraction) -> np.ndarray:
    """Defer, within each PREDICTED class, the rows nearest the declared threshold.

    A row is predicted positive when score >= threshold. In each predicted class
    floor(count * fraction) rows are deferred (True); the rest are accepted.

    THE ORDERING IS EXACT. Rows predicted negative are taken in DESCENDING score,
    rows predicted positive in ASCENDING score, ties by ascending row index. It is
    not |score - threshold|: that subtraction rounds distinct scores to one value
    (measured: 1e-20, 2e-20 and 3e-20 against 0.5 all give 0.5), after which the
    tie-break defers the row FARTHEST from the threshold. Both 2026-11-21 research
    packages did exactly that.
    """
    s = _scores_and_threshold(scores, threshold)
    t = float(threshold)
    defer = np.zeros(s.size, dtype=bool)
    neg = np.flatnonzero(s < t)
    pos = np.flatnonzero(s >= t)
    k_neg = _quota(neg.size, negative_fraction)
    k_pos = _quota(pos.size, positive_fraction)
    if k_neg:
        defer[neg[np.lexsort((neg, -s[neg]))][:k_neg]] = True
    if k_pos:
        defer[pos[np.lexsort((pos, s[pos]))][:k_pos]] = True
    return defer


def truth_class_routing(labels, scores, deferred, threshold: float) -> dict:
    """What routing did to each TRUE class -- the denominators kept apart.

    For true class c with n_c rows: acceptance_rate = P(accepted | Y = c);
    wrong_automatic_per_class = P(accepted and predicted != c | Y = c) -- for the
    pathogenic class this is the share of all pathogenic rows automatically called
    benign; wrong_rate_among_accepted = P(predicted != c | Y = c, accepted), None
    when nothing of that class was accepted (a rate over an empty denominator is
    not zero). A class with no rows is reported with status 'no_observations' and
    no rates. Descriptive only: no bound on any of these is implied.
    """
    s = _scores_and_threshold(scores, threshold)
    y = np.asarray(labels)
    d = np.asarray(deferred)
    if y.shape != s.shape or d.shape != s.shape:
        raise ValueError("labels, scores and deferred must be aligned one-dimensional arrays")
    if y.dtype.kind not in "iu" or np.any((y != 0) & (y != 1)):
        raise ValueError("labels must be integers 0 or 1; define the evaluation population upstream")
    if d.dtype != np.bool_:
        raise ValueError(f"deferred must be a boolean mask; got dtype {d.dtype}")
    predicted = (s >= float(threshold)).astype(int)
    accepted = ~d
    out = {"threshold": float(threshold), "n_rows": int(s.size),
           "accepted_fraction": float(accepted.mean()),
           "interpretation": "descriptive_routing_not_risk_control", "classes": {}}
    for c in (0, 1):
        m = y == c
        n_c = int(m.sum())
        if n_c == 0:
            out["classes"][c] = {"status": "no_observations", "n_total": 0, "n_accepted": 0,
                                 "n_deferred": 0, "n_wrong_automatic": 0, "acceptance_rate": None,
                                 "wrong_automatic_per_class": None,
                                 "wrong_rate_among_accepted": None}
            continue
        acc = accepted & m
        wrong = acc & (predicted != c)
        n_acc = int(acc.sum())
        out["classes"][c] = {
            "status": "observed", "n_total": n_c, "n_accepted": n_acc,
            "n_deferred": n_c - n_acc, "n_wrong_automatic": int(wrong.sum()),
            "acceptance_rate": n_acc / n_c,
            "wrong_automatic_per_class": int(wrong.sum()) / n_c,
            "wrong_rate_among_accepted": (int(wrong.sum()) / n_acc) if n_acc else None,
        }
    return out
