# POP-1 BUILD SPECIFICATION — explicit label-eligible evaluation populations

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Written 2026-08-01, against `HEAD = origin/main = 960f807`, tree clean, ratchet 4121.**
**Prerequisite to REG-1 and OP-1, per the decisions of 2026-08-01.**

---

## 1. What POP-1 does

`ClinicalEvaluator.evaluate` currently builds `EvaluationPopulation.full(n, …)`
at `evaluator.py:942` and hands **unprojected, source-length arrays** to every
consumer. When a reference label is withheld — carried as NaN by
`CanonicalVariantTable`, and first-class in this project — the registry's kernels
narrow the row set silently inside `metrics.clean_arrays`, and the result is
reported with the wider population's size and fingerprint.

Demonstrated end to end on 2026-08-01:

```
y = [1, 1, 0, nan]   p = [0.9, 0.1, 0.2, 0.8]
positive_predictive_value
  value 1.0   status ok   reason None
  CERTIFICATION_ELIGIBLE True
  N_OBSERVATIONS 4                          <- computed over THREE rows
  POPULATION_FINGERPRINT sha256:9ff577fc…   <- attests to a set it did not describe
```

POP-1 constructs the label-eligible population explicitly and projects every
consumer through it, so that one row set is described by one fingerprint.

---

## 2. The read this specification rests on

`ClinicalEvaluator.evaluate` spans `evaluator.py:790-1132` and was read in full on
2026-08-01, together with `_interval_fields` (1158-1252),
`_consequence_breakdown` (1404-1465), `_gene_error_analysis` (1467-1520) and
every call site. Nothing below is inferred from a summary.

### 2.1 The complete consumer map

| line | consumer | arrays it receives |
|---:|---|---|
| 814-816 | `y`, `p`, `n` bound from the parameters | attempted |
| 817-818 | `n_pos = int(y.sum())`, `n_neg = n - n_pos` | **attempted, and `y` may hold NaN** |
| 846-847 | `validate_reference_labels`, `validate_probabilities` | attempted |
| 863 | `validate_ranking_scores` | attempted `scores` |
| 883 | `ranking_values = p` | attempted |
| 932 | `population.n_source != n` guard on a supplied population | attempted `n` |
| 942 | `EvaluationPopulation.full(n, scope="attempted_cohort", …)` | — |
| 958 | `MetricContext(y_true=y, y_prob=p, y_score=ranking_values, population=…)` | attempted |
| 968 | `resolve_gene_clusters(meta)` | `meta`, source-aligned |
| 981-982 | `_interval_fields("auroc"/"auprc", y, p, …, cluster)` | attempted + `cluster.values` |
| 985 | `roc_curve(y, ranking_values)` | attempted |
| 994 | `precision_recall_curve(y, ranking_values)` | attempted |
| 1003 | `calibration_curve(y, p, …)` | attempted |
| 1016-1018 | the three operating points | attempted |
| 1024-1025 | `_consequence_breakdown`, `_gene_error_analysis` | attempted **+ `meta`** |
| 1080, 1097 | `n_expected=n`, `n_rows=n` | attempted `n` |
| 1105 | `n_samples=n, n_pathogenic=n_pos, n_benign=n_neg` | attempted |

**Fourteen distinct consumers.** A projection that reaches thirteen produces a
fingerprint that is actively wrong rather than merely over-broad.

### 2.2 The defence that already exists, and its exact limit

`tests/unit/test_evaluation_population.py:410-433` establishes that
`MetricContext` **refuses arrays whose length does not match its population**:

```python
with pytest.raises(ValueError, match="ALREADY\s+PROJECTED|already\s+projected"):
    MetricContext(y_true=y, y_score=y, population=eligible)
```

This is why the divergence is reachable today and not caught: the population at
942 is `full`, so `population.n == n == len(y)` and the check passes. **Under
POP-1 that same check becomes a guarantee** — once the population is restricted,
passing an unprojected array to the registry raises immediately.

The limit is exact and must be stated: **only the registry path is defended.**
`_interval_fields`, the three operating points, both breakdowns, `roc_curve`,
`precision_recall_curve` and `calibration_curve` take bare arrays with no such
check. A missed projection there fails silently. Those are the consumers the test
battery in section 6 exists to pin.

### 2.3 The vocabulary is already settled by the suite

22 `restrict` call sites across `tests/`, using `scope="label_eligible"` and
`reason="reference_label_withheld"`, with the `mask.all()` guard that `restrict`
requires because it refuses a no-op narrowing. POP-1 adopts these exactly.

`tests/unit/test_prediction_input_contract.py:340-345` already asserts that
`metrics.select_finite_reference_labels` no longer exists, in source as well as
by attribute. The retirement is enforced; POP-1 is the other half.

### 2.4 The registry snapshot is already frozen with this idiom applied

`tests/unit/test_registry_vocabulary_completion.py:526-541` builds a
`label_restricted` cohort — 100 finite labels plus 20 NaN — and constructs its
context through **precisely the POP-1 idiom**, then
`test_existing_registry_results_do_not_move` (544+) compares eight fields per
result against a snapshot frozen before any of this was written, with *"no
carve-outs and no expected-change list."*

**The registry's behaviour under POP-1 is therefore already pinned.** POP-1 makes
production agree with what the suite asserts.

---

## 3. Four decisions this read surfaced

None was visible before `evaluate` was read in full. Recommendations are marked;
none is acted on without authorisation.

### 3.1 `meta` and `cluster.values` must be projected, or they misalign

`_consequence_breakdown` computes `mask = (consequence_coarse == cat).values` from
`meta` and then indexes `y[mask]` and `p[mask]`. `_gene_error_analysis` assigns
`meta["_fp"] = ((preds == 1) & (y == 0))`. `_interval_fields` passes
`clusters=cluster.values` alongside `y` and `p` into `bootstrap_metric`.

If `y` and `p` are projected to 100 rows and `meta` stays at 120, these are
either a broadcasting error or, worse, a silent misalignment.

**Recommendation.** Project both:

```python
meta_eval = None if meta is None else meta.iloc[population.indices].reset_index(drop=True)
cluster_values_eval = None if cluster.values is None else population.take(cluster.values)
```

`meta.iloc[population.indices]` is correct because `population.indices` are
absolute positions into the original source frame — `population.py:49-53` states
this explicitly. It is **not** `population.take(meta)`, which requires a numpy
array. Whether `resolve_gene_clusters` runs on `meta` or `meta_eval` is a real
choice: running it on `meta_eval` is simpler and keeps one row set, but changes
cluster resolution input on withheld-label cohorts.

### 3.2 `n_pos` is computed on an array that may contain NaN

`n_pos = int(y.sum())` at line 817, before any validation. With a NaN present,
`y.sum()` is NaN and `int(NaN)` raises `ValueError`. **This is a latent defect
independent of POP-1** — reachable today on any withheld-label cohort passed as
a float array.

I have not confirmed it by execution, and it may be masked if `y_true` arrives as
an integer or object dtype. It must be probed before POP-1 touches these lines,
because POP-1 moves them.

### 3.3 The report's `n_samples` becomes ambiguous

Line 1105 sets `n_samples=n, n_pathogenic=n_pos, n_benign=n_neg`. After POP-1
there are two defensible values, and the decision document already calls for
exposing both.

**Recommendation.** Keep `n_samples` as the **label-eligible** count, since every
metric in the report now describes that population, and add the attempted-cohort
figures as new fields — the decision document names `n_source`,
`n_label_eligible`, `n_reference_label_withheld`, population scope and parent
fingerprint. Neither `n_label_eligible` nor `n_reference_label_withheld` appears
anywhere in the repository today (measured: zero occurrences), so both are new
schema surface and the schema version must move.

### 3.4 `compare_models` shares one population object by construction

`evaluator.py:1803` builds `shared_population` once and hands the **same object**
to every model, and the comment at 926-930 states that intra-call sameness is
*"proved by construction rather than inferred from equal fingerprints."*

If `evaluate` restricts internally, each call produces a **new** restricted object.
Membership stays identical — `y_true` is shared across models, only `proba`
varies — so `same_membership_as` holds, but object identity does not.

**Recommendation.** Restrict once in `compare_models` and pass the already-
restricted population down, preserving the by-construction guarantee. This
requires the guard at 932 to compare against `population.n_source`, which
`restrict` preserves, rather than `population.n`. **Whether any code asserts
identity rather than membership must be checked in `model_comparison.py` before
this is written.**

---

## 4. The behaviour change POP-1 makes, stated plainly

This is the largest consequence and it must not be discovered later.

**Today**, a withheld label makes `label_check.ok` false, which sets
`ranking_usable` and `probability_usable` false at lines 901-902, which withholds
the receiver operating characteristic curve, the precision-recall curve and the
calibration curve, and marks the flat scalars absent with
`WITHHELD_BY_INPUT_GATE`. The three operating points return `None`. The registry
computes anyway, over a silently narrowed set.

**After POP-1**, the labels handed to every consumer are finite by construction,
so `label_check` passes and all of it is computed — over the declared
label-eligible population, with a fingerprint that describes exactly those rows.

That is the intended outcome: compute on a declared population rather than
withhold everything because part of the cohort is unlabelled. But it flips
withheld-label cohorts from *"nearly everything absent"* to *"everything present,
over a narrower and explicitly named population."* It is a large, deliberate
change in report content and must be declared in the commit message, exercised by
the tests in section 6, and reflected in the schema version.

---

## 5. Implementation

Ordered against the read line numbers.

```python
# after line 815, before n is used for anything
y = np.asarray(y_true)
p = np.asarray(y_proba)
n_source = len(y)

# --- the population is now built HERE, before the gates ---------------------
if population is not None:
    if population.n_source != n_source:
        raise ValueError(...)                      # unchanged text
    if source_id is not None and population.source_id != source_id:
        raise ValueError(...)                      # unchanged text
    attempted = population
else:
    attempted = EvaluationPopulation.full(
        n_source, scope="attempted_cohort", source_id=source_id)

label_mask = np.isfinite(np.asarray(y, dtype=float))
population = attempted if label_mask.all() else attempted.restrict(
    label_mask, scope="label_eligible", reason="reference_label_withheld")

# --- every array is projected exactly once ---------------------------------
y = population.take(y)
p = population.take(p)
n = population.n
scores_eval = None if scores is None else population.take(np.asarray(scores))
meta_eval = None if meta is None else meta.iloc[population.indices].reset_index(drop=True)

assert len(y) == n and len(p) == n
assert population.n <= attempted.n
assert np.isfinite(np.asarray(y, dtype=float)).all()
```

Every consumer from section 2.1 then reads `y`, `p`, `n`, `scores_eval`,
`meta_eval` — the projected names. Because the projection rebinds `y` and `p`
themselves, no call site below line 815 changes textually, which keeps the diff
small and reviewable; the assertions above are what prevent that from being a
silent trap.

`n_pos` and `n_neg` move to after the projection, where `y` is finite by
construction, resolving 3.2 as a side effect.

---

## 6. Acceptance criteria and test battery

**A1 — the withheld-label cohort.** `y = [1, 1, 0, nan]`, `p = [0.9, 0.1, 0.2, 0.8]`.
Assert `positive_predictive_value` has `N_OBSERVATIONS == 3`,
`population_scope == "label_eligible"`, and a fingerprint differing from the
attempted population's.

**A2 — every consumer sees the same rows.** On the same cohort, assert the
registry, both intervals, all three operating points, all three curves and both
breakdowns each describe three rows. This is the criterion that catches a missed
projection, and it must name each consumer explicitly rather than sampling.

**A3 — all-finite inputs are byte-identical.** A fully labelled cohort must
produce a report equal field-by-field to the pre-POP-1 output, and
`population is attempted` must hold, with `restriction_reason is None` and
`len(lineage()) == 1`. Pinned by the existing pattern at
`test_evaluation_population.py:391-404`.

**A4 — the lineage records the narrowing.** `n_excluded_from_parent == 1`,
`restriction_reason == "reference_label_withheld"`, lineage scopes
`["attempted_cohort", "label_eligible"]`, and `"reference_label_withheld"` in
`describe()`.

**A5 — no kernel narrows further.** After projection, assert that
`clean_arrays`'s mask is all-true for every registry kernel invocation on the
cohort — the population is now the only narrowing operation, which is the
2026-07-27 ruling.

**A6 — the existing registry snapshot still passes.**
`test_existing_registry_results_do_not_move` must remain green unchanged. Since
its `label_restricted` fixture already applies this idiom, POP-1 should move
nothing there at all.

**A7 — `compare_models` keeps its guarantee.** Two models over one cohort with
withheld labels must yield populations that are `same_membership_as` each other,
and identical by object identity if 3.4 is implemented as recommended.

**A8 — sabotage.** An unprojected array passed to `MetricContext` alongside a
restricted population must raise. Already pinned at
`test_evaluation_population.py:427`; assert it survives.

**A9 — the ratchet moves in the same commit**, computed by a real collection,
never typed. Current value 4121.

**A10 — the full suite is green** with the skip set unchanged.

---

## 7. Sequence

```
POP-1   explicit label-eligible population wiring          <- this document
REG-1   protected metadata ownership on every result path
OP-1    operating-point subsystem, non-authoritative
OP-2    authority switch
OP-3    post-selection statistical validation
```

## 8. Before any code is written

Three probes, none of which is a guess:

1. **Does `int(y.sum())` raise on a NaN-bearing float array?** (3.2)
2. **Does anything in `model_comparison.py` assert population *identity* rather
   than membership?** (3.4)
3. **Does `resolve_gene_clusters` require the full frame**, or is `meta_eval`
   safe as its input? (3.1)

Each is a small, bounded read or execution, and each decides a line of the
implementation above.
