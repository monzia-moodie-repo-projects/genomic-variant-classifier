# SESSION 2026-08-24 part 2 -- writing the analysis is not applying it

**Author: Monzia Moodie**
**Commits:** `a65bb50`, `10e72a4`
**Ratchet:** 5435 (unchanged)
**Preceding head:** `8d029ee`
**Ending head:** `10e72a4`

> Written at TWO unrecorded commits, as the previous record was.

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `a65bb50` | session record, 2026-08-24 | NEUTRAL | 5435 | 5420p/15s, 907.68s |
| `10e72a4` | BASELINE-1 -- METHODS.md | NEUTRAL | 5435 | 5420p/15s, 912.43s |

`BASELINE-1` closes. With `METRICORIGIN-1` at `8d029ee`, the standing plan's
first two items are done.

---

## 1. The closure condition's premise was false

`MEASUREMENT_2026-08-08_baseline1-provenance-census.md` names what closes the
finding:

> It closes when that fact is recorded in the documents that still cite the
> number as established -- the README and the roadmap, which are not touched
> here.

**MEASURED 2026-08-24 at `a65bb50`: neither carries the figure.**

```
README.md        16,445 bytes  sha256 f0d477dac657e85a   no 0.9847, no 0.9863
docs/ROADMAP.md  13,668 bytes  sha256 97362acab8500336   no 0.9847, no 0.9863
```

The README was repaired at some point; the roadmap was succeeded by D2c and its
seventeen citations now live in the archive as frozen evidence. The census was
written sixteen days earlier at commit `0856fd7`, and **which citations survive
is a fact about HEAD, not about `0856fd7`.**

The surviving live claim was in a third document the census never named:

```
METHODS.md:6    **Holdout AUROC:** 0.9847 (gene-stratified, 154,404 variants)
METHODS.md:193  | AUROC (gene-stratified holdout) | 0.9847 |
```

`BASELINE1-SCOPE-INCOMPLETE-1`. That is the same claim, in the same form, as the
`HOLDOUT_AUROC` constant PROD-1 removed from `api/main.py` on 2026-08-07 -- the
unattributable figure fused with the denominator that convicts it, since
`n_val` is exactly `154,404` and its measured value is `0.9974`.

### Why it survived

`tests/unit/test_methods_feature_count.py` binds three things: the feature-count
sentence, the feature-group table sum, and HGMD's absence from the numbered
sources. **It binds no performance claim.** The enumeration never included one,
which is why this figure outlived the README's and the roadmap's. The scope gap
and the binding gap are the same fact seen twice.

### What the repair does not do

It substitutes **no corrected figure**, because the census establishes there is
none: *"The correct outcome is to record that, not to reconstruct a figure from
four disagreeing descriptions."*

It touches **no record**. Of 81 occurrences of `0.9847` in the tracked tree, 56
are in record documents -- the changelog, session records, both censuses, the
archived roadmap -- and editing them would falsify history. The four remaining
in `METHODS.md` are all inside the correction block, which QUOTES the removed
claim, and the installer asserts zero BARE occurrences outside a blockquote.

`METHODS.md` supplied its own repair form: lines 96-99 already carry a dated
correction for the identical class of defect, ending *"Restating the number by
hand would only reset the clock on the same defect."*

---

## 2. Writing the analysis is not applying it

The installer's docstring carries THREE PARAGRAPHS explaining that the
`authored()` predicate demands pure ASCII and *"would refuse the very file this
repair exists to fix"* -- `METHODS.md` has carried 62 non-ASCII bytes since long
before that convention: seven `>=`, six em dashes, three en dashes, a `<=`, a
subscript one and zero, an alpha, a set-membership sign.

**I then left the call in the code path.** The dry run refused with
`METHODS_postimage.md: non-ASCII`, exactly as predicted, by the check I had
predicted it with.

That is the session's clearest lesson and it belongs in the permanent record.
The correct predicate for a historical document -- no byte-order mark, no
carriage return, a terminal newline, and a non-ASCII count UNCHANGED -- is what
the unit now applies, and `authored()` is removed rather than left unused.

---

## 3. A NameError my own audit could not see

Removing `measure_package` as dead code was correct: it was proven uncalled.
But a CALL SITE referenced its RESULT --

```python
if before.count != EXPECTED_COLLECT or before.count != m["suite"]:
```

-- and `m` would have raised `NameError` at run time.

My undefined-name audit missed it because it collected every `Store` name in the
module into one flat set with **no scope awareness**, and an unrelated local `m`
in a nested helper three hundred lines away masked it.
`PROBE-SCOPE-BLIND-AUDIT-1`.

**It was found by READING the installer end to end** -- the standard applied to
every file received and, until this unit, not to files written. A scope-aware
checker then reported four more, all false positives: nested-function parameters
(`n(word)`, `document(post_head, status, error)`) and a vararg (`git(repo,
*args)`) that the first version of the checker did not collect.

---

## 4. PAYLOAD-DELIVERY-STALE-NAME-1, confirmed by three measurements

`Install_Baseline1_2026-08-24.py` named THREE different files in one session.
When the corrected version was published under the same name, the copy in
`Downloads` did not change -- twice, with the digest checked between attempts:

```
Install_Baseline1_2026-08-24.py      9AC7C38E...  exit 2
Install_Baseline1_2026-08-24.py      9AC7C38E...  exit 2   (unchanged after re-download)
Install_Baseline1_v2_2026-08-24.py   7E5574AF...  exit 0
```

The refusal looked IDENTICAL both times, so only the digest distinguished a
stale file from a real defect. Renaming resolved it on the first attempt.

Every re-cut installer now gets a version suffix. Three probes already followed
that rule today -- `Probe_PreflightD2C_v2`, `Probe_RoadmapStructure_v2`,
`Probe_RoadmapHeadlineFacts_v2` -- and the installers did not. This is what it
cost.

---

## 5. Findings

### Closed
`BASELINE-1`, in the form the census specified.

### Registered
`BASELINE1-SCOPE-INCOMPLETE-1`; `PROBE-SCOPE-BLIND-AUDIT-1`;
`PAYLOAD-DELIVERY-STALE-NAME-1`; `RUN-PLANNING-DOCS-UNMARKED-1` --
`docs/RUN9_SCIENTIFIC_DESIGN.md` and `scripts/run9_launch.md` state Run 8's
figure as `0.9863`, the other half of the audit's unresolvable question, but
both are dated planning artefacts for a run that has since happened and editing
them would falsify what Run 9 was planned against.

### Unchanged
`RATCHET-MOVING-UNITS-RENDER-THREE-COUNTERS-1`, `ATTRIBUTION-STILL-REFUSES-1`,
`TEMPORALCITE-1`, `ROOT-DIRECTORY-UNGOVERNED-1`,
`EVIDENCE-DISPOSITION-INCONSISTENT-1`, `ONTOLOGY-ZERO-LENGTH-REFUSAL-1`,
`SUITE-TRANSITION-KIND-INCOMPLETE-1`, `CERTIFICATION-SURFACE-UNIMPLEMENTED-1`,
`TRANSACTION-CANNOT-EXPRESS-DELETION-1`, `ARCHIVE-SEMANTIC-COLLISION-1`,
`ARCHIVE-PATCH-INFERRED-TEXT-1`, `AF-FIX-WORK-TRACKED-1`,
`POSTFLIGHT-FEATURE-COUNT-STALE-1`, `KAN-IMPORT-SIDE-EFFECT-1`,
`TRANSACTION-GIT-FAILURE-FAILS-OPEN-1`, `RESOURCE-HANDLE-LEAK-1`,
`MANIFEST-NONDETERMINISTIC-ACROSS-RUNS-1`, `DOWNLOADSHADOW-1`,
`FABRICATED-OBSERVATION-1`, `FABRICATED-DIGEST-2`, `DOCLOC-1`.

---

## 6. Ending state

```
HEAD                    10e72a4
ratchet                 5435
suite identity digest   a922ebef1c1d4875
gate                    5420 passed, 15 skipped, 0 failed
METHODS.md              14,683 bytes; states that 0.9847 cannot be attributed
working tree            clean, including untracked
continuous integration  green through a65bb50; 10e72a4 in flight
```

## 7. Next intended action

The standing plan, quoted in the roadmap: **DRIFT-1 with README-1**, then
**OP-1 step 5** against STEP K, **OP-2**, and **RETRAIN-GATE** last. Commit C
and the BASELINE-1 repair are done.

## 8. Remaining uncertainty

Whether the continuous-integration alert workflow can ALERT. It FIRES -- twelve
`workflow_run` entries in the visible window, 6 to 10 seconds each, after every
push run. Its failure branch remains unexecuted, because no run has failed.
