# SESSION 2026-07-28 — one serialiser, not two (CI-u-1)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `857500d`, ratchet 3627
**Roadmap position:** CI-u-1, the first stage of a new item found while investigating CI-p
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. Two writers, one report, different artifacts

`RunArtifactWriter.save_eval_report` serialised through `asdict`, which walks the
dataclass and BYPASSES `EvaluationReport.to_serializable`. Commit 3a introduced
that method precisely because `asdict` cannot carry `result_kind` and does not
normalise a refused result's non-finite value — and this writer was never
updated.

Measured 2026-07-28 on one healthy report: the encodings differ in
`metric_results`, where `asdict` omits `result_kind` entirely and leaves `status`
as a `MetricStatus` enum object rather than its value.

**The writer's own comment claimed otherwise:**

> "This writer and `ClinicalEvaluator.save_report` now produce byte-identical
> encodings of the same report, which they previously did not guarantee."

That was true when written on 2026-07-26 and became false the moment
`to_serializable` existed. **A claim in a comment is not a check**, and this one
stopped being true silently, two days later, with nothing to notice.

After unification the two writers produce **byte-identical output at 15,509
bytes**, both carrying `result_kind`, with `status` serialised as a string.

## 2. What the investigation corrected

This began as CI-p, and two measurements redirected it.

**Family B is NOT persistence-reachable.** CI-p claimed a blast radius of five
call sites in `representation_geometry.py` and `clustering_metrics.py`. Measured:
no path carries `GeometrySummary`, `PartitionAgreementPanel`,
`ConfounderComparison` or `ConfounderGate` output into strict JSON. Only two
`dump_strict_json` call sites exist in the package, and neither references any
Family B type. **The constraint I had been designing around does not exist.**
CI-p is rescoped rather than closed: the writer/reader asymmetry itself remains.

**A larger defect sits underneath.** On a single-class cohort the FLAT `auroc`
and `tpr_curve[0]` are `NaN`, and `dump_strict_json` refuses them — correctly,
since a non-finite number in an evidence artifact is "an absent estimate wearing
a number's clothes". But the consequence is that **a scientifically valid
evaluation over a degenerate cohort produces an artifact that cannot be written
at all**. Absence has no representation on the flat surface, so the whole file is
rejected rather than the one field being recorded as absent.

That is CI-u, recorded as a new item and staged:

    u-1  unify the writers                        THIS COMMIT
    u-2  explicit absence representation          next
    u-3  schema version and read path

## 3. Verification

Regression `FAILED` list byte-identical at 40. The frozen report oracle moves
only `schema_version`, commit 3b-2's declared field.

**Sabotage: four mutations, four detected, zero undetected.**

| break | detected |
|---|---|
| B1 the writer reverts to `asdict` | yes |
| B2 `to_serializable` stops adding `result_kind` | yes |
| B3 `to_serializable` stops normalising non-finite values | yes |
| B4 the typed surface is withheld from the artifact | yes |

B1 is the exact silent divergence that occurred between 26 and 28 July, and it
now fails two tests.

### 3.1 A reading error of mine

I first reported three failures in the artifacts suite as though they included
one of mine. Run against the baseline, **all three are pre-existing** — missing
`pyarrow` in the sandbox, part of the standing forty-failure surface. My initial
count combined two suites and attributed a shared total to the wrong cause. The
two new tests pass in isolation and add nothing to the failure set.

## 4. Files

    src/genomic_variant_classifier/evaluation/prediction_artifacts.py  one serialiser
    tests/unit/test_prediction_artifacts.py                            11 -> 13
    tests/unit/test_carried_item_register.py                           CI-u predicate
    docs/CARRIED_ITEMS.md                                              CI-u added, CI-p rescoped

Ratchet 3627 -> 3630 (+3), measured.

---

*Written 2026-07-28.*
