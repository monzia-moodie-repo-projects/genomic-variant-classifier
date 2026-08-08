# SESSION 2026-08-08 — GATE-1: four AUROC thresholds were never four gates

**Base: `3378659`. Result: `b702777`, pushed. Continuous Integration: success
on both workflows.**

**Ratchet 4462 → 4487 (+25). Armed full suite 4481 passed, 6 skipped, 0 failed
in 31m48s; 4487 collected. Skip surface unchanged at 6.**

Closes **GATE-1**, **PIPELINE-1**, **LSIF-1** and **ALIGNMENT-1**.
**ROSTER-1**, **EVALPROV-1** and **EWCSEL-1** stay open and now fail closed by
name.

---

## 1. The ontology, which was the whole defect

GATE-1's original census counted four AUROC thresholds and called them four
gates. They are **three classes of decision**, and one of them is not a policy
at all:

| value | question it answers | where it now lives |
|---|---|---|
| `0.90` | is the score↔label join credible? | `ScoreLabelAlignmentPolicy` |
| `0.97` | is production above an absolute floor? | `PromotionPolicy` |
| `0.002` | how far may a candidate regress? | `PromotionPolicy` |
| `0.9842` | — | **deleted** |

`0.9842` was `current production − 0.0005` over a figure whose provenance is
unestablished (**BASELINE-1**). A derived quantity of an unknown is not a
policy, so it is removed rather than typed.

## 2. The census correction

The original inventory **missed a fifth site**.
`scripts/forensics/verify_oof_alignment.py` declared `0.90` independently, with
the comment *"a correctly-joined base model should be well above chance"*.

It asks the same question as conformal calibration's floor, so it is **the same
policy, not a new gate** — a correction to the count, not growth in the
register. Both now consume one `ScoreLabelAlignmentPolicy` from
`evaluation/alignment.py`, and the forensic script does **not** import
`conformal` to obtain it: a general integrity check must not depend on a
specific statistical method for a number.

`ALIGNMENT-1` is filed and closed in the same entry so the correction has a
permanent identifier.

## 3. The numbers are typed, not justified

`0.97`, `0.002` and `0.90` all carry `PolicyProvenance` reporting
`LEGACY_PENDING_JUSTIFICATION`. `is_justified` is `False` and stays false until
somebody establishes them.

An architecture can be correct while its constants remain inherited, and
putting a threshold behind a good type does not validate it. The provenance is
a **normal field on a normal policy object** rather than a constant named
something alarming — a scary name becomes wallpaper the moment it is imported;
a field is read every time the policy is.

## 4. REGISTRY-1c — three incompatibilities, one visible

The drift workflow read a path `.gitignore` excludes wholesale, imported
`ModelRegistry` from the module that never had it, and called an attribute the
registry deliberately does not provide. The first branch exited before reaching
the third, so it reported a coherent-looking failure indefinitely.

It is now an **exit-code adapter** over `validate_current_production`. The
`3 = not checked, 1 = invalid, 0 = valid` semantics the file's own comments
record as hard-won are preserved exactly, but they live in Python where they
can be tested. **Continuous Integration knows exit codes, not arithmetic.**

`validate_current_production` collects **every** blocker rather than returning
on the first, so a failing run shows the whole picture instead of revealing the
next problem on the following run.

**The step is renamed** to *"Validate declared production registry state"*. A
committed file can establish that the **declared** state satisfies policy; it
cannot establish that a running process serves those bytes.

**Protocol equality now precedes every numeric comparison.** Previously the
absolute floor was checked before the protocol was known to match, so a number
could be judged before it was known to be interpretable. Judging first and
qualifying afterwards is how 0.9988 unseen-gene came to be compared with 0.9984
ordinary test.

## 5. LSIF-1 closes as a feature-space contract, not a rename

`current_pipe._prepare(pd.DataFrame(X_train_new))` supplied the **new** cohort
as its own reference, and the source admitted it: *"placeholder — ideally pass
X_train_old"*. The defect was known when it was written.

Feeding the real reference **through** the serving pipeline would have been
worse than leaving it: reference rows in the serving representation against new
rows from `DataPrepPipeline` estimates a ratio across two representation
functions — compatible widths, uninterpretable quantity.

So the call is **removed**. Both cohorts must arrive already engineered at the
same stage, and **column equality** is checked rather than column count: a
reordered column preserves the width while permuting every row, and width is
the only thing the previous code could have checked.

`DensityRatioStatus.SAME_POPULATION` yields weights of exactly one **by
declared policy** when both fingerprints are declared and agree, rather than
fitting an estimator on p/p and reporting its noise as adaptation. An absent
fingerprint is **not** a declaration of difference — it means the ratio must be
estimated, which is the safe default.

## 6. PIPELINE-1, and why EWCSEL-1 did not close with it

Three of four sites were mechanical: `InferencePipeline` exposes
`trained_models`, never `base_models`. One of those lines was written by this
author and shipped in `372cea1`.

**Correcting the attribute name alone would have made EWCSEL-1 worse.**
`getattr(m, "best_score_", 0.0)` over an all-equal keyspace returns dictionary
order, so the repair would have turned an unreachable defect into a silently
arbitrary one. It raises instead, and
`UNRESOLVED_ADAPTIVE_RETRAINING_BLOCKERS` names ROSTER-1, EVALPROV-1 and
EWCSEL-1 so that closing one blocker cannot open the path.

## 7. Measured, not assumed

Importing the `evaluation` package costs **749–926 ms** against **10–12 ms**
for `model_registry`. Placing the alignment policy under `evaluation/`
therefore costs the conformal path about 0.8 s of one-time import. Tolerable —
and now a number in the record rather than a preference.

## 8. Nine author defects, and what caught each

Every one was caught by an instrument executing, none by review.

1. A **text rule applied where a syntax-tree rule belonged**, forbidding the
   comments that record what was removed. YAML has no syntax tree and needs the
   text rule; Python does and does not.
2. A **digest pinned from a scratch directory** that never reached the tree.
3. A **forbidden string removed from one of the two paragraphs** carrying it.
4. A **hand-escaped nested string** producing an unterminated literal, rebuilt
   with `repr()` so quoting is generated rather than written.
5–7. **Three undefined names** — `Enum`, `current_record_id`,
   `new_population_fingerprint` — introduced by an edit set verified in a
   sandbox whose preamble **supplied every one of them**.
8. A **stale digest pin** in the ratchet installer, by exactly the fix applied
   between the two runs.
9. Prose claiming the ratchet installer "can now run" when its pin was already
   known stale — a summary running ahead of its measurement.

**Defects 5–7 are the day's lesson.** The piece-D block was parsed against its
author's scaffolding rather than against the file, so every name resolved by
construction. The import-resolution gate installed with REGISTRY-1 caught
`Enum` within minutes and named both call sites. The other two raise only when
the line executes, and `_retrain` is fail-closed — so a green suite would have
coexisted with two guaranteed runtime failures indefinitely. **Only scope
analysis finds those**, and that instrument had been written during DOCKERCOPY-1
and then not reused.

## 9. Acceptance

| | |
|---|---|
| base | `3378659` |
| result | `b702777` |
| diff | 11 files, 999 insertions, 89 deletions |
| targeted | 89 passed (18 import gate + 25 alignment + 46 registry) |
| unarmed full suite | 4481 passed, 6 skipped, 0 failed, 33 warnings, 19m21s |
| armed suite **before** the bump | **ERROR — the ratchet fired**, "Set the value to 4487" |
| coupled tests after the bump | 100 passed in 21.64s |
| armed full suite after | 4481 passed, 6 skipped, 0 failed, 33 warnings, 31m48s |
| collected | 4487, measured |
| Continuous Integration | **success, both workflows** |

Insertions reconcile: 134 + 173 + 100 + 1 = 408 accounted for, leaving 591
across the seven patched files; deletions 2, leaving 87; and 591 + 87 = 678 =
238 + 245 + 121 + 27 + 19 + 20 + 8.

## 10. Next

**Commit C — `SealedEvaluation`** — after a field-by-field source census.
BASELINE-1 is its first question: which experiment produced `0.9847` is
unestablished, and the repository has cited it under three incompatible
descriptions.

Then **DRIFT-1 with README-1**, **OP-1 step 5** against STEP K, **OP-2**, and
**RETRAIN-GATE** last — which needs ROSTER-1, EVALPROV-1 and EWCSEL-1 each to
have an answer.

Fifty-one follow-ups are open.
