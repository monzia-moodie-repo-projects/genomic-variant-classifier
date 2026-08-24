# SESSION 2026-08-24 -- a metric's origin becomes part of the metric

**Author: Monzia Moodie**
**Commits:** `c143788`, `8d029ee`
**Ratchet:** 5404 -> 5435
**Preceding head:** `99ab4ed`
**Ending head:** `8d029ee`

> Written at TWO unrecorded commits, as the previous record was. The three
> before it were written at three, four and six.

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `c143788` | session record, 2026-08-23 part 2 | NEUTRAL | 5404 | 5389p/15s, 899.16s |
| `8d029ee` | Commit C -- SealedEvaluation | ADDITION +31 | 5404 -> 5435 | 5420p/15s, 900.72s |

`METRICORIGIN-1` closes. This is the first SCIENTIFIC unit of the stretch;
everything before it was repository infrastructure.

---

## 1. Nothing was built until both censuses were read in full

The standing plan's first item is Commit C, and everything known about it came
from a single roadmap sentence. Two committed measurement documents are the
authority, and both were read end to end before a line was written.

`MEASUREMENT_2026-08-08_metricorigin-census.md` (10,027 bytes) establishes WHAT
a sealed record must carry. `MEASUREMENT_2026-08-08_baseline1-provenance-census.md`
(9,292 bytes) establishes WHICH runs may be sealed. The roadmap had been quoting
the first; I had been reading a summary of a document I had never opened.

### What the metric-origin census actually says

Run 14's manifest holds **four** figures a careless reader would call "Run 14's
area under the receiver operating characteristic curve":

```
0.9975  metrics.json -> auroc                     computed on the test split
0.9975  manifest -> stacker_metrics_test.auroc    the same computation
0.9984  manifest -> lr_stacker_auroc_from_log     scraped from a training log
0.9985  manifest -> oof_blend_auroc_from_log      scraped from a training log
```

The scraped pair describe *"a different quantity entirely: out-of-fold
performance during training, not held-out performance after it"*, and the census
names the consequence: **"That is precisely the mechanism by which 0.9847 came
to be published as a holdout metric."**

`docs/METRICS.md` has carried two named columns for months. The census WITHDRAWS
an earlier draft's claim that the roadmap disagreed with the artefact -- *"It
does not, and that claim is withdrawn"* -- and concludes: **only the code cannot
represent the distinction.** The `_from_log` suffix is *"a naming convention
doing a type's job"*.

### What the BASELINE-1 census actually says

**`0.9847` is UNATTRIBUTABLE.** Its earliest appearance is a commit SUBJECT
LINE; there is no Phase 2 artefact and nothing from Run 8 in the repository. It
was served by the application programming interface, baked into every container
image, cited as a benchmark baseline, and printed in the README as both
*"publication snapshot"* and *"Run 8 baseline"*.

The denominator convicts the claim: `n_val` in `outputs/run14/full/metrics.json`
is **exactly 154,404**, so the advertised cohort is Run 14's VALIDATION split,
whose measured figure is **0.9974** -- *"four lines away from the number that was
published against it."*

Three further findings that a summary would have lost:

- **The audit asked this on 2026-07-14 and it went unresolved for three and a
  half weeks** while the figure kept being served. *"BASELINE-1 is therefore not
  a discovery. It is a known unanswered question that was treated as settled
  everywhere except in the document that asked it."*
- **Run 15's `0.9847` is `F1_macro`** -- a four-digit coincidence settled by
  dates, and the census corrects its own author's PROD-1 commit message for
  having invited a lineage reading.
- **`TEMPORALCITE-1`**, filed there: *"a consistent difference between two
  quantities of unknown identity is not a measured comparison."* The census
  reverses an earlier ruling that a citation was untouchable because its
  arithmetic checked out -- *"correct arithmetic and insufficient scrutiny."*

Both censuses record their own probe defects rather than hiding them: a
suppressed `-ErrorAction SilentlyContinue` miss read as "no citations found",
and two probes that died on a byte-order mark *"after a full session of
enforcing byte-order-mark-free output in every installer written today.
Recorded rather than quietly corrected."*

---

## 2. Two live consumers were already waiting

`SealedEvaluation` was DEFINED NOWHERE and mentioned in exactly two files. My
first probe printed the FILENAMES and not the lines; for a name defined nowhere
and mentioned twice, the lines are the entire finding.

`api/attribution.py:381` -- the runtime attribution layer is already blocked:

> Commit C attaches a SealedEvaluation that names this digest and this roster
> fingerprint, and only then can this become APPLICABLE. ... Even a linked
> `sealed_evaluation_id` is NOT enough here.

There is a live enumeration member, `EvaluationApplicabilityStatus.
NO_SEALED_EVALUATION`, existing solely because the type did not. And the
enumeration states the acceptance criterion in its own docstring:
**"Resolving a digest authorises IDENTITY, not EVIDENCE."**

`monitoring/model_registry.py:375` -- `ModelRecord` already carries a
`sealed_evaluation_id` pointing at *"PROD-1's future `SealedEvaluation`, which
IS a different object ... Those may legitimately diverge, which is exactly why
they must not be the same record."*

---

## 3. Placement was measured, not assumed

The module imports three types from `monitoring/` and is placed in
`evaluation/`. Whether that direction is permitted was not something I knew, and
a layering test might exist.

MEASURED at `c143788`, by PARSING every import rather than grepping -- an import
inside a function body is still an import, and a name in a docstring is not:

```
evaluation/ -> monitoring/   1 import   alignment.py:57, module level
monitoring/ -> evaluation/   0 imports
```

Precedented, acyclic. The four enforced layering rules in the suite all govern
the VOCABULARY layer at the bottom -- `capabilities.py`, `thresholds.py` -- and
the models-to-data direction. None governs this.

`PROBE-LAYER-WORD-COLLISION-1`: my word match returned 112 test files because
**`agent_layer` contains the substring `layer`**.

---

## 4. The binding test refused, for the second time, and the second refusal is
the more valuable

```
snapshot: suite size    says   5404   live source says   5435
```

All thirty-one new cases PASSED -- `5419 + 15 skipped + 1 failed = 5435`. The
only failure was the roadmap's suite figure.

`RATCHET-MOVING-UNITS-RENDER-THREE-COUNTERS-1`. `render_roadmap_suite` was added
to the ROADMAP-BINDING installer ALONE, so the first ratchet-moving unit after
it installed a suite its own transaction invalidated. The first refusal exposed
a transcribed figure in one document; **this one exposes a standing
obligation.**

`install_plan.py:42` already states the principle and already owns
`render_ratchet` and `render_readme`. It should own this function too, so a
future installer cannot omit it by forgetting to copy it. That is a change to a
shipped module and belongs in its own unit -- recorded in the installer's
comment rather than smuggled into a feature commit.

---

## 5. What Commit C is

A thin sealing layer, as the census rules: *"EvaluationProtocol,
EvaluationEvidence and TrainingLineage already exist and already refuse the
failures they were built for. The indicated design is a thin sealing layer over
them, not a parallel hierarchy."*

The defect is one line -- `metrics: Mapping[str, float]` -- with the protocol
BESIDE the mapping rather than inside each entry. And `from_dict` at
`model_registry.py:260` performs a **silent `float()`**, which would swallow Run
14's string-valued per-model metrics without a trace.

Five requirements, each with a source: origin is a FIELD; `artifact_sha256`
becomes MANDATORY rather than being introduced; coercion is DECLARED;
partiality is representable; and a seal names a digest AND a roster fingerprint.

One requirement the censuses implied but did not state: a metric whose name
still ends `_from_log` may not declare a computed origin. The suffix is the
artefact's own statement of origin, and a seal may replace it but must not
contradict it.

Thirty-one cases, with fixtures that are Run 14's four real figures and Run
10b's three real lost outputs. One asserts the spread rounds to exactly 0.0010.
One asserts a twelve-model serving projection is NOT evidence for a
thirteen-model seal. One asserts the projection down to `EvaluationEvidence` is
LOSSY -- origin does not survive it, which is the defect made explicit.

**Eighteen guards sabotaged, eighteen detected**, each by its intended case,
against a zero-failure baseline and restoration.

---

## 6. Four defects in my own instruments

| identifier | what it was |
|---|---|
| `PROBE-ORIGIN-REGEX-OVERBROAD-1` | matching `computed` also matched **pre-computed**, filling a 194-line census with data-connector noise about externally supplied score files |
| `PROBE-LAYER-WORD-COLLISION-1` | `agent_layer` contains `layer`; 112 files matched, most irrelevant |
| harness registration order | `dataclasses._is_type` resolves `cls.__module__` through `sys.modules` WHILE the decorator runs, so executing a module before registering it raises `AttributeError` on `NoneType`. The module was never at fault. |
| a sabotage that did not parse | reported its guard as undetected. Redone with a mutation that parses, it made `"n/a"` silently become `0.0` and was caught by the intended test. **"My sabotage did nothing" is not "nothing checks this."** |

Also: a probe printed FILENAMES where the LINES were the finding, and an
extraction pattern assumed three blank lines after a function. Both located by
parsing instead.

---

## 7. Findings

### Closed
`METRICORIGIN-1`.

### Registered, open
`RATCHET-MOVING-UNITS-RENDER-THREE-COUNTERS-1`;
`ATTRIBUTION-STILL-REFUSES-1` -- the type exists but nothing composes a seal, so
`attribution.py` still returns `NO_SEALED_EVALUATION`; `TEMPORALCITE-1`;
`BASELINE-1`, whose answer is known and which *"closes when that fact is
recorded in the documents that still cite the number as established -- the
README and the roadmap"*.

### Unchanged
`ROOT-DIRECTORY-UNGOVERNED-1`, `EVIDENCE-DISPOSITION-INCONSISTENT-1`,
`ONTOLOGY-ZERO-LENGTH-REFUSAL-1`, `SUITE-TRANSITION-KIND-INCOMPLETE-1`,
`CERTIFICATION-SURFACE-UNIMPLEMENTED-1`,
`TRANSACTION-CANNOT-EXPRESS-DELETION-1`, `ARCHIVE-SEMANTIC-COLLISION-1`,
`ARCHIVE-PATCH-INFERRED-TEXT-1`, `AF-FIX-WORK-TRACKED-1`,
`POSTFLIGHT-FEATURE-COUNT-STALE-1`, `KAN-IMPORT-SIDE-EFFECT-1`,
`TRANSACTION-GIT-FAILURE-FAILS-OPEN-1`, `RESOURCE-HANDLE-LEAK-1`,
`MANIFEST-NONDETERMINISTIC-ACROSS-RUNS-1`, `DOWNLOADSHADOW-1`,
`FABRICATED-OBSERVATION-1`, `FABRICATED-DIGEST-2`, and the remaining
scientific register.

---

## 8. Ending state

```
HEAD                    8d029ee
ratchet                 5435
suite identity digest   a922ebef1c1d4875
gate                    5420 passed, 15 skipped, 0 failed
docs/ROADMAP.md         13,668 bytes, bound by 9 tests, three counters rendered
working tree            clean, including untracked
continuous integration  green through c143788; 8d029ee in flight
```

Suite identity chain:

```
5352 70a3b350199cf2ec -> 5385 1c8bc5a726662c69 -> 5395 f13709cd715c625c
-> 5404 66fddbc60fb28e9a -> 5435 a922ebef1c1d4875
```

## 9. Next intended action

The BASELINE-1 repair across the README and roadmap, which is what closes it.
`0.9847` still appears in the README as both "publication snapshot" and "Run 8
baseline", and the census establishes it cannot be attributed. Then DRIFT-1 with
README-1, OP-1 step 5 against STEP K, OP-2, and RETRAIN-GATE last.

## 10. Remaining uncertainty

Whether the continuous-integration alert workflow can ALERT. Measured this
session: it FIRES -- twelve `workflow_run` entries, 6 to 10 seconds each, after
every push run. Its failure branch remains unexecuted, because no run has
failed in the visible window.
