# MEASUREMENT 2026-08-08 — METRICORIGIN-1: a metric's origin is part of the metric

**Author: Monzia Moodie**
**Base commit: `764147d`. Method: repository probes only; no run was executed
and no figure recomputed.**

The BASELINE-1 census (`MEASUREMENT_2026-08-08_baseline1-provenance-census.md`)
established **which** runs Commit C can seal. This census establishes **what a
sealed record must carry**, by reading what the sealable artefact actually
contains rather than by designing a type and hoping the data fits.

**Result: Run 14's own manifest holds four figures a careless reader would call
"Run 14's area under the receiver operating characteristic curve", spanning
0.9975 to 0.9985 — and distinguishes them only by a suffix in their key names.**
`EvaluationEvidence.metrics` is a flat `Mapping[str, float]`. Loading this
manifest into it would place a computed test-set figure and a log-scraped
out-of-fold figure side by side with nothing in the type to tell them apart.
That is precisely the mechanism by which `0.9847` came to be published as a
holdout metric.

---

## 1. The four figures

| value | key | what it is |
|---|---|---|
| `0.9975` | `metrics.json` → `auroc` | computed on the test split |
| `0.9975` | manifest → `stacker_metrics_test.auroc` | the same computation, agreeing exactly |
| `0.9984` | manifest → `lr_stacker_auroc_from_log` | **scraped from a training log** |
| `0.9985` | manifest → `oof_blend_auroc_from_log` | **scraped from a training log** |

The two computed figures agree to four decimal places across two independently
written files. The two scraped figures differ from them by roughly one
thousandth — small enough to look like rounding, large enough to change which
model appears best, and describing **a different quantity entirely**:
out-of-fold performance during training, not held-out performance after it.

`docs/METRICS.md` gets this right in prose. Its header reads
`| Run | Date | Test AUROC | OOF blend | …` and row 14 reads
`| 0.9975 | 0.9985 |` — two named columns for two quantities. **An earlier
draft of this census asserted that the roadmap table disagreed with the
artefact. It does not, and that claim is withdrawn.** The documentation has
been making this distinction in column headings for months; only the code
cannot represent it.

## 2. The artefact names its own provenance, in three places

```
oof_blend_auroc_from_log        0.9985
lr_stacker_auroc_from_log       0.9984
per_model_oof_auroc_from_log    {ten models, 0.9921 – 0.9984}
```

The `_from_log` suffix is **a naming convention doing a type's job**. Whoever
wrote this manifest understood that the origin of a number belongs with the
number, and had nowhere to put it except the key. A sealed record should give
that fact a field.

The remaining metric containers carry no such suffix and are computed:
`stacker_metrics_test`, `stacker_metrics_val`, `per_model_test_metrics`,
`per_model_val_metrics`.

## 3. What the manifest already supplies

Twenty-two top-level keys. Commit C does not need to invent requirements; it
needs to require what this artefact proves is obtainable.

| key | value |
|---|---|
| `git_head` | `80ac62ca7e83d35638274a01170d4c8f4f62c418` |
| `run_start_utc` / `run_end_utc` | `2026-05-26T10:38:56Z` / `13:53:31Z` |
| `python_version` | `Python 3.11.10` |
| `dataset` | `n_train` 1197216, `n_val` 154404, `n_test` 349067, `n_features` 78 |
| `artifact_sha256` | **two entries** |
| `vm` | instance, host, region, graphics processing unit, image, hourly rate |
| `observability`, `artifacts`, `anomalies_logged` | 5, 6 and 8 entries |

**`artifact_sha256` is the finding that most changes the design.** Run 14
already binds its metrics to artefact digests — the same mechanism PROD-1 built
independently for runtime attribution, arrived at twice from opposite ends.
Sealing does not need to introduce it; it needs to make it mandatory.

The two artefact manifests supply the rest: `ensemble.manifest.json` and
`scaler.manifest.json` each pin eight library versions, the interpreter, and
the platform.

## 4. Per-model metrics are stored as strings

`per_model_test_metrics` and `per_model_val_metrics` are each an eleven-element
array of objects with fields
`model, auroc, auprc, f1_macro, f1_weighted, mcc, brier` — and **every value is
a `str`**, not a number.

`EvaluationEvidence.__post_init__` already refuses this:

```python
if not isinstance(value, (int, float)) or isinstance(value, bool):
    raise RegistryInvariantError(
        f"metric {name!r} must be a real number, got {value!r}")
```

That refusal is correct and must stay. The consequence for Commit C is that a
sealing layer **must coerce explicitly and record that it coerced**, never
silently call `float()`. A string that looks like a number may be a rounded
rendering of one, and rounding is a transformation a sealed record should
declare.

## 5. One artefact carries a byte-order mark

Of the nine committed JSON artefacts under `outputs/`, **exactly one** begins
with `EF BB BF`:

```
BOM   outputs/run14/reproducibility_manifest.json
-     the other eight
```

It was written by PowerShell — the `"key":  value` double-space formatting is
`ConvertTo-Json`'s signature, and `Set-Content -Encoding UTF8` emits a mark.
Python's `json.loads` refuses it outright:

```
JSONDecodeError: Unexpected UTF-8 BOM (decode using utf-8-sig)
```

**So `encoding="utf-8-sig"` is not a workaround — it is the only encoding that
reads all nine**, since it handles marked and unmarked files alike. A loader
using plain `utf-8` reads eight artefacts and crashes on the one artefact this
project can actually seal.

This author's first two probes for this census used plain `utf-8` and died on
that line, after a full session of enforcing byte-order-mark-free output in
every installer written today. Recorded rather than quietly corrected.

## 6. A scientific claim lives inside the artefact

```
h1_diversity_finding: "Stacker test AUROC 0.9975 = catboost test AUROC 0.9975,
but stacker dominates on f1_macro (0.9775 vs 0.9632), MCC (0.9550 vs 0.9276),
Brier (0.0130 vs 0.0166). Ensemble lift is in calibration and
threshold-quality, not ranking power."
```

This is a **finding**, not a metric: the ensemble's advantage over its best
single member is not in ranking but in calibration and threshold behaviour. It
is also a direct argument against reading any single AUROC as the ensemble's
value.

A sealed record that keeps only numbers would discard it. Whether
`SealedEvaluation` should carry free-text findings is a design question, but
the census records that discarding this one would lose something the artefact
was deliberately written to preserve.

## 7. An incident recorded in `session_notes`, unresolved here

```
"Instance destroy command was inadvertently executed despite Block B gate
showing FAIL on ensemble.* files. Subsequent locator (Block I) confirmed all
73 files ... present locally. No data loss. Root cause: Block B gate used
fixed Test-Path; files lived one directory deeper than assumed. Anomaly A8
captured; Charter v1.2 patch (recursive locator + separate destroy script) in
Run 15 backlog."
```

A gate checked the wrong path, a teardown proceeded past its own FAIL, and
recovery was fortunate rather than designed. **This is the same defect class as
DOCKERCOPY-1** — a check pointed at an assumed location rather than the real
one.

Whether anomaly A8 and the Charter v1.2 patch were ever completed is **not
established by this census** and is not asserted. It belongs on the register as
a question, before Run 17 makes it a live one. Filed as **TEARDOWN-1**.

## 8. What this establishes for Commit C

A sealed evaluation must be able to say, for every number it carries:

1. **Where it came from** — computed from held-out predictions, or scraped from
   a training log. The artefact already encodes this in key names; the type
   should encode it in a field.
2. **Which artefact produced it** — `artifact_sha256` exists and should be
   mandatory rather than optional.
3. **Under what protocol** — `EvaluationProtocol` already requires a nonempty
   `protocol_id`, `split_kind`, `population_scope` and `label_policy`, and
   already carries `population_fingerprint`.
4. **Whether it was transformed on the way in** — string-to-float coercion
   declared, not silent.
5. **Whether the record is complete** — Run 10b's artefact declares
   `"status": "partial"` with three outputs `lost`, and a sealed record must be
   able to represent that without pretending otherwise.

`EvaluationProtocol`, `EvaluationEvidence` and `TrainingLineage` already exist
and already refuse the failures they were built for. **The indicated design is
a thin sealing layer over them, not a parallel hierarchy** — the same ruling
GATE-1 took when it extended the registry's promotion policy rather than
building a second one.

## 9. Register

**METRICORIGIN-1** — a metric's origin is part of the metric. `_from_log`
figures and computed figures share a flat mapping today; the spread between
them in Run 14 is 0.0010, and nothing in the type prevents them being read as
the same quantity. Closes when `SealedEvaluation` distinguishes them.

**TEARDOWN-1** — Run 14's session notes record a destroy command executing past
its own gate's FAIL, with the root cause a fixed path check. Whether the
recorded remedy was ever applied is unestablished.

52 carried in, two filed: **54 open.**

## 10. Method, and what was not done

Every statement above comes from reading committed artefacts on 2026-08-08 at
commit `764147d`: `reproducibility_manifest.json`, `metrics.json`, the two
artefact manifests, `METRICS.md`, and a byte-order-mark scan across all nine
committed JSON files under `outputs/`.

**No type was designed and no code was written.** This census exists so that
the design has a citable basis in the repository rather than in a conversation
— the same reason the BASELINE-1 census was committed on its own before
anything was built on it.
