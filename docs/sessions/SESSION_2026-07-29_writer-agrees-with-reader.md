# SESSION 2026-07-29 — the writer agrees with the reader (CI-p)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `f6d2442`, ratchet 3682
**Roadmap position:** CI-p — **the last open defect on the register**
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. The defect

`MetricResult.to_dict` emitted the raw `NaN` that a non-OK result is REQUIRED to
carry in memory, and `dump_strict_json` refuses a non-finite number by design.
Every refused result was therefore unpersistable through `to_dict` alone.

`from_dict` had always documented the opposite:

> Round-trip from to_dict(). NaN does not survive strict JSON, so a null value is
> read back as NaN rather than rejected.

**The reader was right the whole time; only the writer disagreed with it.**

Measured before the change:

    to_dict()['value']  : nan
    strict JSON         : REFUSED (NonFiniteArtifactValue)
    from_dict(None)     : nan          <- the reader already expected null

After:

    refused   value=None   PERSISTS   round-trips faithfully: True
    ok        value=0.87   PERSISTS   round-trips faithfully: True

## 2. What the fix does NOT change

`test_metric_result_relocation` pins that a non-OK result must carry `NaN` in
memory, and it still does. **`NaN` is a perfectly good in-memory sentinel; it is
only in an ARTIFACT that it becomes an absent estimate wearing a number's
clothes.** The internal representation is untouched.

The rule is STATUS-AWARE, not a blanket non-finite sweep. Absence is authorised
by the status, never inferred from the value: an OK result whose value is somehow
non-finite is a defect, and nulling it would disguise that defect as a legitimate
absence, so it is left for the strict writer to refuse.

## 3. Two layers met

`evaluator.py:430` raised `TypeError: must be real number, not NoneType`.

Commit 3a added a normalisation at the REPORT layer *because* the source emitted
a raw NaN. CI-p fixed the SOURCE, so `to_dict` now emits `null` itself — and 3a's
line, written when the value was always a float, met a `None`.

The patch is now redundant but not harmful. Removing it would make the report
layer depend on the source layer having already run, so it is KEPT and made
tolerant: a value that is already absent stays absent.

## 4. The claimed blast radius never existed

CI-p's original text named five Family B call sites in
`representation_geometry.py` and `clustering_metrics.py` as its constraint.
Measured 2026-07-28: **no Family B type is persistence-reachable.** Only two
`dump_strict_json` call sites exist in the package and neither references
`GeometrySummary`, `PartitionAgreementPanel`, `ConfounderComparison` or
`ConfounderGate`. The item was carried for months with a blast radius that did
not exist.

## 5. Verification

Regression `FAILED` list byte-identical at 40. Legacy report oracle moves only
`schema_version`.

**Sabotage: six mutations, six detected, zero undetected, zero anchor misses.**

| break | detected |
|---|---|
| B1 the writer reverts to emitting raw NaN | yes (18 tests) |
| B2 the rule becomes value-based, not status-based | yes |
| B3 an OK result is nulled too | yes |
| B4 the finiteness helper always says finite | yes (17 tests) |
| B5 the reader stops restoring the sentinel | yes |
| B6 the report layer stops tolerating None | yes |

## 6. A process lesson, recorded

Before starting, I searched the repository for existing work on this asymmetry
and found none — which is why CI-p was safe to build. That search was prompted by
the previous investigation, where I spent SIX malformed probes chasing a
scikit-learn warning that `tests/unit/test_sklearn_parallel_warning_contract.py`
had already resolved, with a scoped filter and three structural tests, before the
session began. Its name appears in plain sight in every full-suite run.

**Investigating a finding without first checking whether the repository already
contains its resolution** is now a named hazard, alongside the malformed probes.

## 7. Files

    src/genomic_variant_classifier/evaluation/capabilities.py   status-aware null
    src/genomic_variant_classifier/evaluation/evaluator.py      tolerant report layer
    tests/unit/test_metric_result_serialisation_contract.py     NEW, 27 tests
    tests/unit/test_carried_item_register.py                    CI-p discharged
    docs/CARRIED_ITEMS.md                                       CI-p discharged

Ratchet 3682 -> 3709 (+27), measured.

---

*Written 2026-07-29.*
