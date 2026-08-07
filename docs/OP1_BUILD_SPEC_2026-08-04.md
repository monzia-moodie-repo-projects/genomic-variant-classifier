# OP-1 BUILD SPECIFICATION — the typed, population-bearing operating-point subsystem

**Author: Monzia Moodie**
**Written 2026-08-04, against `HEAD = origin/main = d4b4259`.**
**Prerequisites, all landed: POP-1a `1577f0b`, POP-1b `00e180c`, REG-1 `90ed465`, OP-0 `d4b4259`.**

Evidence:
`docs/measurements/` — REG-1 baseline and R2 mutation reports.
`op1_defect_reverification_r2_2026-08-04.txt` — the register re-verified at `d4b4259`.
`op1_defect_reverification_2026-08-04.txt` — **preserved**: the same register read by a defective instrument.

---

## 0. One correction to the adopted decision document, from measurement

The two-axis example table gives `D12 | legacy tie behavior documented`. **That
half is wrong**, and it descends from my own false closure rather than from the
code.

D12 concerns `_find_operating_point` at `if diff < best_diff` (line 1527).
**OP-0 touched `_find_high_ppv_point` only** — the rename and the docstring — and
never modified the sibling. The R2 verifier measured that function and found
**zero comment lines in it**. Nothing documents the tie-break.

The baseline verifier reported D12 CLOSED because `/tie|ties/` matched
`validate_probabili**ties**`, a function call at line 1497. The hedge at the head
of that section — *"may now be documented by OP-0"* — was the right instinct.

**Corrected row:** `D12 | LEGACY_OPEN, undocumented strict tie-break | structured,
persisted tie-resolution policy required`.

The two-axis model itself is adopted unchanged; D8 demonstrates it cleanly on its
own terms.

---

## 1. The two-axis defect state, adopted

`open` versus `closed` is insufficient, because a legacy ambiguity can be
characterized while the target obligation remains. That distinction prevents
**"documented" from being mistaken for "architecturally solved."**

```python
class DefectState(str, Enum):
    LEGACY_OPEN = "legacy_open"
    LEGACY_CHARACTERIZED = "legacy_characterized"
    TARGET_REQUIRED = "target_required"
    TARGET_IMPLEMENTED = "target_implemented"
    NOT_APPLICABLE = "not_applicable"
    INDETERMINATE = "indeterminate"
```

| defect | legacy state | OP-1 obligation |
|---|---|---|
| D1 | LEGACY_OPEN — thousand-point grid | exact sweep over achievable thresholds |
| D2-D5 | LEGACY_OPEN — 5 `else 0.0` fabrications | refuse, never fabricate |
| D6 | LEGACY_OPEN — 12 `round(...)` at construction | store computed values; round only for display |
| D7 | LEGACY_CHARACTERIZED and corrected (OP-0) | none |
| D8 | LEGACY_CHARACTERIZED — Objective B named | implement Objective A separately |
| D9 | LEGACY_OPEN — O(k·n) | one sort plus cumulative sums, O(n log n) |
| D10 | LEGACY_OPEN — invariant work in the sweep | hoisted by construction |
| D11 | LEGACY_OPEN — two sweep strategies | one sweep, both selectors |
| D12 | **LEGACY_OPEN** — undocumented strict tie-break | structured, persisted tie policy |

---

## 2. Why this is a rewrite and not a patch

**Four defects collapse into one design decision.** A single sort with cumulative
sums yields exact counts at every achievable threshold in O(n log n):

- **D1** disappears — the achievable thresholds *are* the unique score values, so
  there is no grid to miss them.
- **D9** disappears — 2.25 × 10¹² element operations at 1.5 million variants
  becomes ~3.1 × 10⁷.
- **D10** disappears — the positive and negative totals are computed once,
  outside any loop, because there is no loop.
- **D11** disappears — one sweep serves both selectors.

| n | grid O(1000·n) | unique O(k·n) | sort + cumulative sum |
|---:|---:|---:|---:|
| 1,000 | 1,000,000 | 1,000,000 | 9,965 |
| 100,000 | 100,000,000 | 10,000,000,000 | 1,660,964 |
| 1,500,000 | 1,500,000,000 | **2,250,000,000,000** | **30,774,796** |

Patching four defects separately would leave the architecture that produced them.

---

## 3. The typed outcome — `Optional[OperatingPoint]` is the defect

The legacy type is eleven scalars and cannot represent refusal status, refusal
reason, population identity, certification, threshold-policy provenance, target
versus achieved values, or candidate-domain provenance. **`None` currently
carries every kind of unavailability**, which `evaluate_registered`'s own
docstring says is exactly wrong: *"an absent key and a refused metric are
different facts and a caller cannot tell them apart."*

**Do not add optional fields to the legacy dataclass.** That produces an
increasingly ambiguous compatibility object.

```python
@dataclass(frozen=True)
class OperatingPointOutcome:
    status: MetricStatus
    reason: str | None
    metrics: OperatingPointMetrics | None
    population: EvaluationPopulation
    policy: OperatingPointSelectionPolicy


@dataclass(frozen=True)
class OperatingPointMetrics:
    threshold: float
    counts: ConfusionCounts

    sensitivity: MetricResult
    specificity: MetricResult
    positive_predictive_value: MetricResult
    negative_predictive_value: MetricResult
    f1: MetricResult
    matthews_correlation_coefficient: MetricResult

    flagged_fraction: MetricResult
    prevalence: MetricResult          # POPULATION statistic -- see section 5

    population_scope: str
    population_fingerprint: str
    n_observations: int

    certification_eligible: bool
    certification_blockers: tuple[OperatingPointCertificationBlocker, ...]
```

Every metric is a `MetricResult`, so **D2-D5 close by construction**: a quantity
that cannot be computed refuses with a status and a reason rather than returning
`0.0`. And `f1` can no longer inherit a fabrication, because there is no
fabrication to inherit.

**D6 closes** by storing computed values and rounding only at display. A stored
`f1` must be reproducible from stored `ppv` and `sensitivity`.

---

## 4. Decision 1 — applicability is intrinsic; provenance is contextual

`PPV = TP/(TP+FP)` is undefined when `TP+FP = 0`, and **that depends only on the
counts**. Passing `ThresholdParameters` into the predicate would make a
mathematical fact depend on execution history.

```python
def ppv_applicability_from_counts(counts: ConfusionCounts) -> Applicability:
    """Intrinsic mathematical applicability only."""
    if counts.n_predicted_positive == 0:
        return Applicability(
            applicable=False,
            status=MetricStatus.INSUFFICIENT_SUPPORT,
            reason="empty_predicted_positive_set")
    return APPLICABLE
```

Threshold provenance attaches in a second stage, behind a `reports_threshold`
flag, and **raises `RegistryInvariantError` when required provenance is absent**
rather than silently omitting it. That is REG-1's *reject, don't shadow*
discipline applied one layer down.

---

## 5. Decision 3 — prevalence is a population statistic

```
sensitivity, specificity, PPV, NPV, F1, MCC   threshold-derived
flagged_fraction                              decision statistic
prevalence                                    POPULATION statistic
```

Prevalence is `#{Y=1} / N_label_eligible`. It does not depend on the threshold,
predicted-positive membership, the policy, or the sweep — and POP-1a and POP-1b
made that distinction load-bearing by giving the population an explicit identity.

Canonical prevalence comes from `registry.compute(by_name("prevalence"), ctx)`.
**Excluded from count-level Oracle C.** The count-derived equality is retained as
an *audit* oracle only, and only after proving `counts.n == population.n`.

> This avoids inventing two prevalences that agree until a population bug makes
> them diverge.

---

## 6. Decision 2 — two oracles, not one

**Oracle C1, intrinsic count semantics:** status, reason, value with exact NaN
semantics, intrinsic applicability metadata.

**Oracle C2, full result identity at the canonical threshold:** the entire
`MetricResult` contract including all metadata, support counts, population scope,
population fingerprint, certification eligibility and blockers, and the
serialized form.

The register measured **four `INSUFFICIENT_SUPPORT` and two `UNDEFINED`** among
the confusion-derived metrics. Comparing reason strings alone would erase a
scientifically meaningful distinction — and a single all-or-nothing oracle would
conflate count mathematics, threshold provenance, population support and
certification policy. **The split says which one drifted.**

---

## 7. Decision 6 — stable codes, centralized prose

```python
class OperatingPointCertificationBlocker(str, Enum):
    SAME_POPULATION_SELECTION_AND_EVALUATION = (
        "threshold_selected_and_evaluated_on_same_population")
    POST_SELECTION_VALIDATION_NOT_IMPLEMENTED = (
        "post_selection_validation_not_implemented")
```

Artifacts persist **codes**; reports render prose from a central mapping. Never
serialize mutable prose as the primary scientific identifier — the same rule that
made POP-1b's population scope an enum member rather than a sentence.

A vocabulary-completeness gate asserts every code has prose, in the manner of the
existing registry-vocabulary tests.

---

## 8. Objective A, and the tie-break that closes D12

```
Objective A:  max sensitivity(t)  subject to  PPV(t) >= floor
Objective B:  the most permissive t before the first floor violation
```

OP-0 froze B and named it. OP-1 implements A for `at_high_ppv`, with B preserved
as a separately named policy.

**The tie-break must be declared and persisted**, not inherited from iteration
order. The legacy rule resolves ties to the lower, more liberal threshold purely
because `np.linspace` ascends and the comparison is strict — defensible,
undocumented, and invisible in the result. OP-1 states its rule in the policy
object and records it in the outcome.

---

## 9. Acceptance

**A1** the non-monotone fixture selects `t = 0.7` at sensitivity 1.0 and
positive predictive value 2/3 — the candidate Objective B cannot reach.
**A2** a cohort with no positives refuses with a status and a reason, not `None`.
**A3** no `MetricResult` in any outcome carries a fabricated `0.0`.
**A4** a stored `f1` is reproducible from stored `ppv` and `sensitivity`.
**A5** every outcome carries population scope and fingerprint, and they match the
evaluation population.
**A6** Oracle C1 holds for every confusion-derived metric.
**A7** Oracle C2 holds at the canonical threshold, serialized form included.
**A8** prevalence in the composite is the registry's, not count-derived.
**A9** the tie policy is asserted from the persisted outcome, not inferred.
**A10** exactness: the selected threshold is a value achievable on the data.
**A11** the legacy characterization tests still pass unchanged — OP-1 adds a
selector, it does not silently retarget the old one.

---

## 10. Sequence

1. `ConfusionCounts` and the exact sweep, with its own tests. **No selector yet.**
2. The typed outcome and metrics, with refusal semantics.
3. Count-level applicability and the two oracles.
4. `select_operating_point` with the policy object; Objective A and B both named.
5. Shadow comparison against the frozen legacy selector.
6. Cut over the three report fields; schema bump.

**OPCOV-1 is the standing caution.** `_find_high_ppv_point` was exercised by
nothing before OP-0 and the operating points by seven lines across four files, so
a shadow comparison that agrees proves agreement **on the cases exercised** —
which are few. Steps 1 and 2 must carry their own tests rather than lean on it.
