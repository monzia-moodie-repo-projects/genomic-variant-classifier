# SESSION 2026-07-28 — the typed report surface and schema version 3 (commit 3a)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `15ad3f0`, ratchet 3419
**Roadmap position:** Tier 1 item 6, commit 3a of 3a/3b
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. Why commit 3 was split

Commit 3 was to make the registry the only computation path and turn the report's
flat fields into projections. It was split because schema introduction and
computational retirement have different failure modes: a schema defect corrupts
artifacts, a retirement defect corrupts numbers. Landing them together would
leave any regression with two plausible causes.

    3a   the typed surface exists          acceptance: NOTHING moves
    3b   the report becomes a projection   acceptance: exactly four declared
                                                       field-cohort movements

3a retires nothing. `ClinicalEvaluator.evaluate` still computes the Matthews
correlation coefficient, F1 and the calibration errors itself, still emits schema
version 2, and still leaves `metric_results` empty.

---

## 2. The oracle, frozen before anything was written

Ten cohorts, 48 report fields, **480 field values**, captured on the untouched
2b-3 tree and committed as `tests/fixtures/report_snapshot_2b3.json`. Bootstrap
disabled so the capture is deterministic and interval fields are not compared
against random draws.

**Result: 480 values compared, ZERO movements.** No declared movement set at all
— a commit that adds a surface without touching a computation has no business
changing a number. Exactly one field was added (`metric_results`), none removed,
and `evaluate()` still emits version 2.

### 2.1 The oracle had to be repaired before it could be trusted

The first five cohorts could not distinguish a four-decimal from a five-decimal
prevalence contract: every one of their sample sizes divides cleanly, so both
roundings agree. Two cohorts were added whose prevalence separates the two **by
construction** — 333/700 and 401/900 — because a fixture that separates by design
does not stop separating when someone edits a sample size.

**And an error of mine, recorded because the failure mode is the one this project
keeps finding.** I twice ran a separation check comparing `round(frozen, 5)`
against `frozen`, where `frozen` was already rounded to four decimals, making the
comparison a tautology, and read its silence as evidence that the oracle was
blind. It was not; my check was. Measured correctly, **four of ten cohorts detect
a rounding-contract change.**

---

## 3. The per-field rounding contract, extracted rather than imposed

The ruling was explicit: do not choose a new resolution, extract the one the
landed code already embodies. Measured:

    prevalence           4 decimals
    auroc                5
    auprc                5
    mcc                  5
    f1                   5
    brier_score          5
    calibration_ece      5
    calibration_mce      5

`prevalence` at four is the trap. It became a registered metric in commit 2b-2,
so it is precisely the field where a plausible global `round(x, 5)` rule silently
disagrees with the landed contract. I had asserted a global five-decimal rule
twice, from a single line read eight days earlier. Extraction found it; assertion
would not have.

---

## 4. What was built

**`metric_results`** on `EvaluationReport`, with schema-aware validation in both
directions: a version-3 report requires a non-empty mapping, a version-1 or -2
report requires an empty one. A version asserts what a file contains, and a
version that can be true of two different contents is not evidence.

**Two constructors.** `from_metric_results(...)` builds version 3.
`from_serialized_v2(...)` reads a historical artifact and leaves the mapping
EMPTY — never synthesised. An `OK` result manufactured from a bare float would
assert a population scope, a support count, an applicability verdict, a threshold
provenance and a certification eligibility that the artifact never recorded. That
is fabrication, not recovery, and downstream it would be indistinguishable from
provenance that was genuinely measured. `from_serialized(...)` dispatches on the
artifact's own recorded version rather than on what the reader hopes to find.

**Direct construction remains available.** Making the fields `init=False` would
break every existing caller and every historical deserialisation path, so
consistency is enforced in `__post_init__` rather than by removing the door.

**`result_kind` is serialised but not stored.** Commit 2b-2 ruled it lives on the
descriptor and never in result metadata. But an artifact that cannot say what
kind of quantity it recorded is not self-describing, so it is written from the
descriptor at serialisation time and VERIFIED on read. A disagreement is raised
as a version conflict, never resolved by preferring today's registry — the
artifact is the evidence, the registry only the interpreter.

**`to_serializable()` exists because `asdict` cannot do this.** `asdict` walks the
dataclass and bypasses `to_dict()` entirely, so anything added there would never
reach a file. `save_report` is routed through it.

**The undefined reasons were split** — `zero_confusion_margin` for the Matthews
coefficient, `zero_f1_denominator` for F1 — so 3b's compatibility substitution can
be authorised by metric identity AND exact reason. Under a shared reason, an F1
undefined for an unrelated cause would receive the Matthews substitution. The one
test asserting the old shared reason now asserts the distinct ones; a test
accepting either would have defeated the policy it guards.

---

## 5. Three defects in landed code, surfaced by the build

### 5.1 The writer and reader implemented opposite halves of one contract

`MetricResult.from_dict` documents that *"NaN does not survive strict JSON, so a
null value is read back as NaN"*. `MetricResult.to_dict` emits raw `NaN`. And
`dump_strict_json` refuses NaN by design — *"an absent estimate wearing a
number's clothes"*.

**Every refused result was unpersistable.** Fixed at the report layer, where
`serialize_metric_results` writes `null` for a non-finite value; the status and
reason carry the meaning, and the null is only the absence of a number, which is
exactly what a refusal is.

Not fixed globally: `MetricResult.to_dict` has **five other call sites** in
`representation_geometry.py` and `clustering_metrics.py`, Family B probes that
legitimately produce non-finite results and are outside this commit's subject.
Recorded as carried item (p) with that blast radius measured.

### 5.2 Deserialisation dropped every enum-typed flat field

JSON flattens `auroc_ci_status`, `auprc_ci_status` and the two resampling units
to strings, and the report correctly refuses a bare string — *"would silently
miss every branch below."* Without restoration, every round trip crashed.

The dangerous repair would have been relaxing the report's type check, which is
the guard that stops an interval status being misread. Instead the reader
restores the enums and raises on an unrecognised member rather than coercing.

### 5.3 One of mine

My first test helper invented the twenty interval-provenance fields, which are
cross-validated by `_validate_ci_fields`. Inventing them failed, correctly. The
helper now uses a real configuration taken from the oracle — the one a genuine
evaluation with `n_bootstrap=0` produces. A fixture that fabricates a state the
code never emits tests nothing.

---

## 6. Verification

### 6.1 Regression

The 38 modules touching the evaluation stack produce a BYTE-IDENTICAL `FAILED`
list: 40, all sandbox dependency gaps. No test was lost.

The nine warnings in the affected suite are scikit-learn's own
`UndefinedMetricWarning` and `UserWarning`, raised by the LEGACY computation on
the degenerate cohorts the fixtures deliberately include. They are the evidence
justifying the canonical UNDEFINED semantics, not noise, and 3b retires the code
that raises them.

### 6.2 Sabotage matrix

Twelve breaks, **twelve detected, zero undetected**.

| break | detected |
|---|---|
| B1 a version-2 artifact synthesises typed results from bare floats | yes |
| B2 a result_kind conflict is overwritten instead of raised | yes |
| B3 NaN is written into the artifact again | yes |
| B4 a version-3 report may carry an empty typed mapping | yes |
| B5 a historical report may carry a populated mapping | yes |
| B6 a bare float is accepted as a typed result | yes |
| B7 enum-typed flat fields are no longer restored on read | yes |
| B8 from_metric_results accepts an empty mapping | yes |
| B9 save_report reverts to raw asdict, dropping result_kind | yes |
| B10 an unsupported schema version is accepted | yes |
| B11 the report oracle is regenerated on the current tree | yes |
| B12 one frozen report value is tampered with | yes |

### 6.3 THE FIRST RUN LEFT B11 UNDETECTED

A real gap. I wrote the staleness test as though it carried the decisive schema
assertion from commit 2b-3, and it did not.

The 2b-3 pattern does not transfer. There the fixture was captured under registry
schema 1 against a current 2, so "the recorded version must differ from the
current one" worked. Here the oracle was captured under report schema 2 and
`evaluate()` still emits 2 throughout 3a, so an inequality assertion would fail
for an entirely legitimate fixture.

The invariant that does hold is that the oracle must **predate the typed
emission**: `report_schema_at_capture < EVALUATION_REPORT_SCHEMA_VERSION_TYPED`.
A fixture regenerated once `evaluate()` emits version 3 — which is exactly what
3b makes it do — records 3 and is caught. The field and cohort counts were
unchanged by the break, so nothing else noticed.

---

## 7. Files

    src/genomic_variant_classifier/evaluation/evaluator.py   typed surface, schema, serialisation
    src/genomic_variant_classifier/evaluation/registry.py    undefined reasons split
    tests/fixtures/report_snapshot_2b3.json                  NEW, the frozen oracle
    tests/unit/test_typed_report_surface.py                  NEW, 26 tests
    tests/unit/test_registry_vocabulary_completion.py        distinct-reason assertion

Ratchet 3419 -> 3445 (+26), measured by `pytest --collect-only`.

---

## 8. Next

Commit 3b: `evaluation/legacy_projection.py` with a declarative, reason-sensitive
policy table; the projection invariant comparing against
`project_legacy_fields(self.metric_results)` through a NaN-aware comparison with
no tolerance; retirement of the evaluator's local computation at lines 481-482
and 511; the narrowed abstract-syntax-tree guard (carried item (o)); counting
wrappers proving each kernel runs exactly once, that the projection invokes no
kernel, that report construction performs no threshold comparison, and that the
expected and maximum calibration errors reuse ONE `CalibrationBins`; and the four
declared field-cohort movements.

---

*Written 2026-07-28.*
