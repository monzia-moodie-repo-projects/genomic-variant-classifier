# OP-1 BUILD SPECIFICATION — the operating-point subsystem

**Companion to `HANDOFF_2026-07-31_op1-operating-point.md`. Read that first for
repository state and verified interface facts; this document is the build.**

**This specification is COMPLETE. Every interface it names was measured out of
the repository on 2026-07-30 or 2026-07-31. Nothing here needs re-litigating.**

---

## THE SHAPE OF THE COMMIT

One commit. Five internally gated layers. Non-authoritative — the legacy
selectors stay live and no report field changes.

```
Layer A   formula centralization        steps A-D
Layer B   exact candidate sweep         step  E
Layer C   deterministic policy          steps F-G
Layer D   typed composite + projection  steps H-J
Layer E   shadow comparison             step  K
```

---

## STEP A — freeze `compute()`

Capture every branch of `compute()` (registry.py REPO 1540–1631) as a frozen
oracle BEFORE touching it: for a battery of contexts, record
`(status, value, reason, dict(metadata))` for all 24 registered metrics.

**Gate:** the frozen record exists and is reproducible.

---

## STEP B — extract `_finalize_metric_result` in `registry.py`

**Extract the SMALLEST CONTIGUOUS TAIL.** This is mechanically cutting and
pasting existing lines, not redesigning them.

`compute()` RETAINS: the missing-input gate, applicability evaluation, and
function dispatch — **in that order; inputs before applicability.**

The finalizer OWNS: metadata assembly, protected-key validation, result status
construction, certification metadata.

The finalizer must know NOTHING about `MetricDescriptor.function`,
`_missing_inputs`, threshold selection, or `ConfusionCounts`. It receives only:
metric identity, context, an already-decided applicability verdict, an optional
scalar value, and already-approved metric metadata.

```python
def _finalize_metric_result(
    *,
    metric_name: str,
    ctx: MetricContext,
    applicability: Applicability,
    value: float | None,
    additional_metadata: Mapping | None = None,
) -> MetricResult:
```

**Do not commit even that signature until the extraction point is inspected** —
mirror the data actually present there.

**PRESERVE EXACTLY:**

- the derived protected set —
  `{METRIC_NAME, CERTIFICATION_ELIGIBLE, CERTIFICATION_BLOCKED_BY} | set(ctx.support())`
  — **never hand-listed**;
- `RegistryInvariantError` on collision, with the existing message;
- merge order `meta = {**dict(verdict.metadata), **meta}`;
- `CERTIFICATION_BLOCKED_BY` set only when not eligible;
- the four non-OK reason strings verbatim: `"required_inputs_missing"`,
  `"metric_computation_failed"`, `"applicable_metric_returned_non_finite"`, and
  whatever the verdict supplies.

**`_certification_eligibility(d, ctx)` never reads `d`** — it may be called as
`(ctx)` if that preserves the interface, or left as-is.

### Gates for step B

**ORACLE A.** All 24 registered metrics, all statuses, all metadata, **ZERO
movement** before versus after.

`test_missing_inputs_precede_applicability` — a context violating BOTH conditions
must yield the landed status and reason unchanged.

**The synthetic-key test:** monkeypatch `MetricContext.support()` to return an
extra key; assert it becomes protected **without editing the protected-key
helper.** This tests the architectural reason the code states, not the current
key list.

**Registry result-construction guard:** `compute()` calls the finalizer and does
not directly instantiate `MetricResult` outside approved branches.

---

## STEP C — `metrics.py`: `ConfusionCounts`, applicability, formulas, specifications

### C.1 `ConfusionCounts`

```python
@dataclass(frozen=True)
class ConfusionCounts:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
```

Integer-only, **Boolean rejected** (`isinstance(v, bool)` first), non-negative,
derived margins only, **no separately supplied totals.**

Derived properties: `n`, `n_actual_positive` (tp+fn), `n_actual_negative`
(tn+fp), `n_predicted_positive` (tp+fp), `n_predicted_negative` (tn+fn).

### C.2 `confusion_counts_at_threshold`

```python
def confusion_counts_at_threshold(y, prob, *, threshold, operator) -> ConfusionCounts
```

**The ONE place arrays become counts.** Preserves `_margins_at_threshold`'s
fail-closed checks: `is_probability`, `_require_finite_probabilities`, `_clean`,
size check, `apply_decision_threshold`, `_confusion_counts`.

**`_margins_at_threshold` may be removed** — no external caller, and the guard
does not patch it. If removed, add a structural test that it no longer exists.

### C.3 Count-level applicability — **reason strings must match EXACTLY**

| Function | Condition | Reason |
|---|---|---|
| `sensitivity_applicability_from_counts` | `n_actual_positive == 0` | must match `_requires_class_support(positive=True)` → `f"{label}_class_support_required"` |
| `specificity_applicability_from_counts` | `n_actual_negative == 0` | must match `_requires_class_support(positive=False)` |
| `positive_predictive_value_applicability_from_counts` | `n_predicted_positive == 0` | must match `_requires_flagged_margin(flagged=True)` → `f"empty_{side}_set"` |
| `negative_predictive_value_applicability_from_counts` | `n_predicted_negative == 0` | must match `_requires_flagged_margin(flagged=False)` |
| `f1_applicability_from_counts` | `2*tp + fp + fn == 0` | **`"zero_f1_denominator"`** |
| `matthews_correlation_coefficient_applicability_from_counts` | any margin zero | **`"zero_confusion_margin"`** |
| `prevalence_applicability_from_counts` | `n == 0` | must match `_requires_reference_labels_only` → `"empty_cohort"` |
| `flagged_fraction_applicability_from_counts` | `n == 0` | defined whenever `n > 0` |

**READ THE EXACT `label` AND `side` SUBSTITUTIONS from `registry.py` before
freezing these.** `_requires_class_support` and `_requires_flagged_margin` build
their reasons with f-strings and I did not capture the substituted values.

**Why this matters — `registry.py` lines 882–888, verbatim:**

> *"A single shared reason would let the legacy compatibility projection
> substitute the Matthews value for an F1 undefined for an entirely different
> cause. THE SUBSTITUTION MUST BE AUTHORISED BY METRIC IDENTITY AND BY THE EXACT
> UNDEFINED REASON."*

**The reason string is an authorisation token consumed by `legacy_projection`.**
A mismatch causes the compatibility layer to refuse a substitution it should
make, or make one it should not.

Also preserve the applicability METADATA. `_requires_nondegenerate_confusion`
attaches `{"n_predicted_positive": ...}`.

### C.4 Count-level formulas

```python
sensitivity_from_counts                      tp / (tp + fn)      NaN if 0
specificity_from_counts                      tn / (tn + fp)      NaN if 0
positive_predictive_value_from_counts        tp / (tp + fp)      NaN if 0
negative_predictive_value_from_counts        tn / (tn + fn)      NaN if 0
f1_from_counts                               2tp / (2tp+fp+fn)   NaN if 0
matthews_correlation_coefficient_from_counts (tp*tn - fp*fn) / sqrt(product of margins)
prevalence_from_counts                       (tp + fn) / n       NaN if n == 0
flagged_fraction_from_counts                 (tp + fp) / n       NaN if n == 0
```

Each **retains a defensive NaN guard** even though applicability prevents the
state from dispatching under the canonical path.

### C.5 `CountMetricSpecification` — the closed specification

```python
@dataclass(frozen=True)
class CountMetricSpecification:
    metric_name: str
    applicability: Callable[[ConfusionCounts], Applicability]
    compute_value: Callable[[ConfusionCounts], float]

COUNT_METRIC_SPECIFICATIONS = (
    SENSITIVITY_FROM_COUNTS, SPECIFICITY_FROM_COUNTS,
    POSITIVE_PREDICTIVE_VALUE_FROM_COUNTS, NEGATIVE_PREDICTIVE_VALUE_FROM_COUNTS,
    F1_FROM_COUNTS, MCC_FROM_COUNTS,
    PREVALENCE_FROM_COUNTS, FLAGGED_FRACTION_FROM_COUNTS,
)
```

**Why:** it makes unconstructible the pairing of `name="positive_predictive_value"`
with the negative predicate. **Acceptance invariant:** every `metric_name` in the
tuple is unique.

### C.6 The vectorised candidate table — **Oracle D's subject**

The sweep has up to *n*+1 candidates; calling scalar Python primitives per
candidate defeats the sorted sweep, and duplicating the formulas in
`operating_point.py` is what the whole design prevents.

```python
@dataclass(frozen=True)
class ConfusionCountArrays:
    true_positive:  NDArray[np.int64]
    false_positive: NDArray[np.int64]
    false_negative: NDArray[np.int64]
    true_negative:  NDArray[np.int64]

def metric_arrays_from_confusion_count_arrays(counts) -> SweepMetricArrays
```

**ONE vectorised builder, in `metrics.py`, beside the scalar authorities.** Use
`np.divide(..., out=np.full(shape, np.nan), where=denominator != 0)` so undefined
stays NaN.

---

## STEP D — refactor the six public kernels

```
apply_decision_threshold → confusion_counts_at_threshold → *_from_counts
```

Six kernels: `sensitivity`, `specificity`, `positive_predictive_value`,
`negative_predictive_value`, `f1_at_threshold`,
`matthews_correlation_coefficient`. **The last two currently inline the six
steps** and must adopt the shared path.

**Also route `registry._requires_nondegenerate_confusion` through
`confusion_counts_at_threshold`** — it restates `_confusion_counts` at lines
870–875 (defect D6).

**Optional refinement, NOT required:** rename generic descriptor predicates to
one-per-metric adapters. **Do not force it if it creates descriptor churn.**

### Gates for step D

**ORACLE B** — every denominator regime; bit-identical finite values;
NaN-for-NaN; same exception category; same fail-closed behaviour; no silent
filtering. Cover: all-unique scores, all-tied, mixed/single-class, predict-none,
predict-all, perfect, inverse, zero PPV / NPV / F1 / Matthews denominators, both
operators, exact threshold edges and `np.nextafter` neighbours, empty inputs.

**ORACLE C** — at threshold 0.5, operator `>=`, for each of the seven metrics,
across all **thirteen** confusion regimes:

```python
A = compute(by_name(name), ctx)
B = _finalize_metric_result(metric_name=name, ctx=ctx,
        applicability=spec.applicability(counts_at_0_5),
        value=spec.compute_value(counts_at_0_5) if applicable else None)

assert A.status == B.status
assert A.reason == B.reason
assert (A.value == B.value) or (isnan(A.value) and isnan(B.value))
assert A.certification_eligible == B.certification_eligible
assert dict(A.metadata) == dict(B.metadata)      # COMPLETE, no excepted field
```

**Compare typed fields FIRST, serialisation SEPARATELY** — a serialiser that
normalises or omits `None` would hide a divergence.

**Public threshold-kernel guard:** `apply_decision_threshold`, the count
constructor, and the corresponding `*_from_counts` each invoked ONCE.

---

## STEP E — the exact sorted sweep, in `operating_point.py`

Sort descending once; group equal scores; move one COMPLETE tie group at a time;
cumulative true/false positives; derive false negatives and true negatives from
class totals; emit candidates only after complete tie groups; include explicit
predict-none and predict-all boundaries where the rule requires them.

**NEVER SPLIT TIES.** Under `>=`, a threshold equal to a tied score includes the
entire tied group.

```python
@dataclass(frozen=True)
class ThresholdCandidateDomain:
    kind: CandidateDomainKind          # DISTINCT_SCORE_BOUNDARIES is production
    n_candidates: int
    n_distinct_scores: int
    operator: ThresholdOperator
    tie_policy: str
    includes_predict_none: bool
    includes_predict_all: bool
```

`FIXED_GRID` survives in the vocabulary **only** so the shadow adapter can
describe the old selector.

The sweep contains **numeric arrays, not one `MetricResult` per candidate.**

**ORACLE D:** for every candidate,
`vectorised.<rate>[i] == <rate>_from_counts(counts.at(i))`, **including NaN
equivalence.** Plus: exhaustive comparison against every distinct threshold;
tie groups never split; **total-count conservation at every candidate**; random
permutation invariance; monotone true/false-positive counts; no non-finite
scores; no silent clipping.

---

## STEP F — the capability-aware policy schema

Carry the FULL vocabulary now; implement only the deterministic subset.

```python
class PolicyCapability(str, Enum):
    IMPLEMENTED = "implemented"
    RESERVED    = "reserved"

@dataclass(frozen=True)
class StatisticalConstraint:
    metric: OperatingMetric
    operator: ConstraintOperator          # GREATER_THAN_OR_EQUAL
    target: float
    confidence_level: float | None        # RESERVED for OP-3
    bound_method: str | None              # RESERVED for OP-3
    capability: PolicyCapability

@dataclass(frozen=True)
class PracticalEquivalencePolicy:
    mode: EquivalenceMode                 # ABSOLUTE | RELATIVE
    tolerance: float                      # reject negative and non-finite

@dataclass(frozen=True)
class OperatingPointSelectionPolicy:
    constraint: StatisticalConstraint
    objective: OperatingObjective
    practical_equivalence: PracticalEquivalencePolicy
    robustness: tuple[RobustnessCriterion, ...]      # RESERVED
    burden: BurdenCriterion                          # FLAGGED_FRACTION
    deterministic_resolution: DeterministicResolutionPolicy
    policy_name: str
    policy_version: str
```

Equivalence, defined precisely:

```
ABSOLUTE   U(t) >= U_max - epsilon
RELATIVE   U(t) >= U_max - epsilon * |U_max|
```

**Absolute is the clearest first implementation for metrics on [0,1].**

```python
def validate_policy_capabilities(policy) -> None:
    """Runs BEFORE the sweep. Refuses an unexecutable policy before sorting
    1.7 million rows."""
    if policy.constraint.bound_method is not None:
        raise UnsupportedPolicyCapabilityError(
            "confidence-bound constraints are reserved for OP-3")
```

**NEVER silently downgrade a requested lower-confidence-bound constraint to an
empirical one.**

---

## STEP G — the deterministic hierarchy

```
1. Hard EMPIRICAL constraint       candidate admissible iff constraint metric
                                   is APPLICABLE and satisfies the target
2. Maximise the primary objective
3. Retain within the explicit numerical-equivalence tolerance
4. Minimise FLAGGED FRACTION
5. Deterministic equivalence resolution
```

**Undefined handling:**

```
constraint or objective UNDEFINED for a candidate  -> candidate INADMISSIBLE
all candidates inadmissible                        -> typed refusal
a NON-ESSENTIAL derived metric undefined at the
    selected candidate                             -> point still selected,
                                                      that field UNDEFINED
```

```python
class TieResolutionReason(str, Enum):
    UNIQUE_OBJECTIVE_MAXIMUM            = "unique_objective_maximum"
    PRACTICAL_EQUIVALENCE_APPLIED       = "practical_equivalence_applied"
    LOWER_BURDEN_SELECTED               = "lower_burden_selected"
    DETERMINISTIC_EQUIVALENCE_RESOLUTION = "deterministic_equivalence_resolution"
```

**Step 5 is a deterministic representative of an equivalence class, NOT a claim
of clinical superiority.** Threshold magnitude appears nowhere else.

### The three report policies

```
at_sensitivity_90   constraint sensitivity >= 0.90, maximise PPV
at_sensitivity_95   same at 0.95
at_high_ppv         constraint PPV >= declared floor, maximise sensitivity
```

**`"high"` is not reproducible — the PPV floor must be explicit.** The legacy
default is `min_ppv=0.80`.

A fourth policy — maximise sensitivity subject to a specificity floor — **has no
legacy counterpart.** If built, its outputs are a **declared addition**, not a
movement.

---

## STEPS H, I — the composite

```python
@dataclass(frozen=True)
class OperatingPointMetrics:
    policy: OperatingPointSelectionPolicy
    selection: ThresholdSelection
    threshold_parameters: ThresholdParameters      # source=CALIBRATED, UNROUNDED
    population: EvaluationPopulation
    counts: ConfusionCounts

    sensitivity: MetricResult
    specificity: MetricResult
    positive_predictive_value: MetricResult
    negative_predictive_value: MetricResult
    f1: MetricResult
    prevalence: MetricResult
    flagged_fraction: MetricResult

    certification_eligible: bool
    certification_blockers: tuple[str, ...]

@dataclass(frozen=True)
class OperatingPointOutcome:
    status: MetricStatus
    metrics: OperatingPointMetrics | None
    reason: str | None
    metadata: Mapping
```

Enforced: `status == OK` ⟺ metrics present and reason absent.

**Contained results are built through `_finalize_metric_result` via
`CountMetricSpecification`. They carry NO execution-source field** — the parent
establishes provenance, and this is what keeps Oracle C exact.

**Composite certification in OP-1:**

```
certification_eligible = False
certification_blockers = (
    "threshold_selected_and_evaluated_on_same_population",
    "post_selection_validation_not_implemented",
)
```

**Contained scalar certification and composite certification are different
claims.** Do not imply that selected sensitivity at 90 per cent is certified
merely because the sensitivity scalar would be certifiable at a prespecified
threshold.

**Do NOT clone a descriptor with 0.5 replaced.** That creates an instance absent
from the validated registry.

---

## STEP J — extend `legacy_projection.py`

**It already owns all six mechanisms** — see handoff §3.14. Add operating-point
field policies there. **Do not add a second policy table in
`operating_point.py`.**

The projection is a **pure adapter**: it must not recompute counts, rates,
thresholds or objective values, and must not call thresholds, public kernels,
count formulas or selector logic.

**READ THE 409 LINES FIRST** — they are in
`C:\Users\monzi\Downloads\op1_final_2026-07-31.txt`, section A.

---

## STEP K — the shadow comparison

Classify every field comparison:

```
EQUAL
EXPECTED_EXACT_SWEEP_IMPROVEMENT      the exact sweep found a superior feasible point
EXPECTED_UNDEFINEDNESS_CORRECTION     a legacy 0.0 is a canonical UNDEFINED
UNEXPECTED_MOVEMENT                   must be EMPTY
```

**Do not require thresholds to match when decision sets match:**

```python
def same_decision_partition(first, second) -> bool:
    return first.counts == second.counts
```

Two numerically different thresholds can be decision-equivalent.

**Selection dominance:** for each legacy policy, the new candidate satisfies the
same constraint, the new objective is not worse, and any difference is
attributable to exact enumeration, explicit tolerance, burden minimisation or
undefinedness semantics.

**Create a FROZEN MOVEMENT SET from actual measurements before OP-2.**

**Evaluator shadow guard:** the legacy selector remains authoritative; the new
selector runs only in shadow; no report field is assigned from the new bundle.
**Invert in OP-2.**

**Operating-point source guard:** inside `operating_point.py`, reject direct
division involving TP/FP/FN/TN, per-metric calls to `apply_decision_threshold`,
scikit-learn confusion metrics, and inline denominator guards. **Allow**
cumulative count construction and the vectorised candidate table.

---

## THE ONE THING TO READ BEFORE WRITING

**`_requires_class_support` and `_requires_flagged_margin` build their reason
strings with f-strings — `f"{label}_class_support_required"` and
`f"empty_{side}_set"` — and I did NOT capture the substituted values.**

```powershell
$RepoRoot = "C:\Projects\genomic-variant-classifier"
$r  = "$RepoRoot\src\genomic_variant_classifier\evaluation\registry.py"
$rl = @(Get-Content -Path $r)

foreach ($fn in @("_requires_class_support", "_requires_flagged_margin",
                  "_requires_interior_specificity")) {
  $d = @(Select-String -Path $r -Pattern ("^def " + $fn + "\("))
  if ($d.Count -gt 0) {
    $s = [int]$d[0].LineNumber
    Write-Output ("---- " + $fn + " ----")
    for ($i = $s - 1; $i -lt ($s + 39) -and $i -lt $rl.Count; $i++) { "{0,4}| {1}" -f ($i + 1), $rl[$i] }
  }
}
Write-Output "---- every descriptor's applicability= line ----"
Select-String -Path $r -Pattern "applicability=" |
  ForEach-Object { "{0,5}| {1}" -f $_.LineNumber, $_.Line.Trim() }
```

**Acceptance criterion 19 cannot be met without those exact strings.**

---

*Written 2026-07-31. Companion to `HANDOFF_2026-07-31_op1-operating-point.md`.*
