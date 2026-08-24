# SESSION 2026-08-23 part 2 -- a plan says what it does not know

**Author: Monzia Moodie**
**Commits:** `78c433c`, `99ab4ed`
**Ratchet:** 5395 -> 5404 -> 5404
**Preceding head:** `b586778`
**Ending head:** `99ab4ed`

> Written at TWO unrecorded commits. The three preceding records were written at
> three, four and six, and the drift was named each time. Two is the correction.

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `78c433c` | the roadmap stops being able to rot | ADDITION +9 | 5395 -> 5404 | 5389p/15s, 900.94s |
| `99ab4ed` | the plan is re-derived | NEUTRAL | 5404 | 5389p/15s, 908.10s |

`78c433c` is also covered by the previous record's forward reference; it is
recorded here in full because its FAILURE is the substance.

---

## 1. The binding test caught a real defect on its first real run

`78c433c` installs nine cases binding `docs/ROADMAP.md` to its live sources.
Its FIRST apply attempt failed:

```
snapshot: suite size    says   5395   live source says   5404
```

The unit MOVES the ratchet 5395 -> 5404, and D2c had written the successor's
suite figure by TRANSCRIPTION. **The unit was self-invalidating: it installed a
check that its own transaction falsified.** The gate refused, the transaction
rolled back, nothing was committed.

`ROADMAP-SUITE-COUNTER-UNRENDERED-1`. `install_plan.py:42` had already stated
the principle -- *"Never independently write `expected_suite_size = N` and
`readme_badge = N`"* -- and the roadmap had quietly become a THIRD copy of that
number. `render_roadmap_suite` now sits beside `render_ratchet` and
`render_readme`: three counters, one measured count, and the render refuses
rather than guessing if its needle is not found exactly once.

**The check was correct; the unit was wrong.** That is the strongest evidence
the file could have produced about itself.

### What the binding enforces

Nine claim sites over seven quantities, with EQUALITY and no tolerance -- the
README binding's first version used `assert collected - n <= 50` and let a real
17-test drift pass while reporting green. A vanished claim site FAILS rather
than quietly ceasing to check. Blockquotes are exempt from the passing-count
ban, because the succession notice quotes the predecessor's own figures and a
document must be able to record its own history.

Two guards have no analogue in the README binding: the archive pointer must
RESOLVE, and the blob object identity is RECOMPUTED from the archived bytes. A
one-byte edit or a one-byte truncation of the 466,826-byte predecessor turns
three cases red.

Sixteen sabotage perturbations, sixteen detected, each by its intended case.
One initially reported NOTHING FAILED, and that was the HARNESS: the mutation
replaced a string not yet present in the document. **"Nothing failed" meant
"nothing changed", not "nothing checks this."**

---

## 2. The plan is re-derived, and the scale was not what I had been saying

`docs/ROADMAP.md` section 5 had stated since D2c that its plan had not been
re-derived since 2026-08-08. That was honest, and it was the last open item from
this stretch.

**MEASURED 2026-08-23 at `b586778`: EIGHTY-THREE commits since 2026-08-08.** I
had been saying fifteen. Fifteen was the count of commits in one working
session -- the boundary of one conversation's visibility substituted for the
boundary the roadmap declares. Sixty-three of the eighty-three predate that
session and are unread.

### So the section does not summarise them

Summarising unread work is `FABRICATED-OBSERVATION-1`, recorded one record
earlier, and **a plan is the worst place for it: a wrong entry directs future
effort rather than merely misinforming a reader.**

The new section quotes the archive's final `NEXT` verbatim as the standing plan
-- Commit C (SealedEvaluation), then the BASELINE-1 repair, DRIFT-1 with
README-1, OP-1 step 5 against STEP K, OP-2, RETRAIN-GATE last -- states that the
infrastructure stretch touched none of them, and points at `docs/CHANGELOG.md`
and `docs/sessions/` as the authoritative complete history.

The open register is named at fifty-four with its arithmetic quoted from source:
*"Fifty-two carried in ... Two filed. 52 + 2 = 54. The count RISES, and that is
correct: a census that finds two real things records both rather than tidying
the number."*

`METRICORIGIN-1` and `TEARDOWN-1` are quoted in full because a paraphrase would
lose what matters -- a 0.0010 spread between log-scraped and computed figures
sharing one flat mapping, and a destroy command that executed past its own
gate's FAIL with the Charter v1.2 patch's application NOT ESTABLISHED before
Run 17.

**No closure is asserted.** A search of the eighty-three commit messages found
115 distinct identifiers mentioned, and a mention is not a closure: a commit may
cite an item to say it is open, deferred, or blocking. That is
`PROBE-CLASSIFIER-COARSE-1` -- structure rendered as semantics -- and
reconciling the register is a separate unit whose judgements are scientific
rather than clerical.

### The guard that made the rewrite safe

Section 5 is the ONLY thing that may change, and the installer proves it:
everything BEFORE the heading byte-identical (7,874 bytes) and everything AFTER
the next top-level heading byte-identical (2,404 bytes). Exercised offline
against five deliberate violations before the payload was cut -- a word changed
before, a word changed after, the heading duplicated, the heading removed, a
trailing section deleted -- and each refused on its own clause.

---

## 3. A second fabricated digest, caught before shipping

`PRE_SHA` in the plan installer read
`10eb250bf84eb5e4a1e1c9ee9dc0d1e2b0e57e2b4a9e8e7c6d5b4a3928170615`: the correct
sixteen-character prefix and **forty-eight characters I invented**. Measured
shared prefix with the real value: exactly 16.

It was also **never compared**, which is how it survived the checks that caught
nothing.

That is the SECOND fabricated digest of the day. The first, in a preflight probe
this morning, shipped and did nothing because it too was uncompared. This one
was caught by asking a question I had not asked the first time: **is the pin
actually compared?**

Both defects were fixed together: the value is the measured digest of the
reconstruction the payload was authored against, and it IS compared. The dry run
then confirmed it against the file on disk -- `preimage digest matches the
authoring basis` -- which also proved `render_roadmap_suite` had done exactly
and only what it claimed at `78c433c`.

**A pin that is never compared is decoration.** That is now written into the
installer beside the constant.

---

## 4. Findings

### Closed
`ROADMAP-SUITE-COUNTER-UNRENDERED-1`; `ROADMAP-STALE-1` for the plan section.

### Registered
`FABRICATED-DIGEST-2` -- the second invented digest in one day, both uncompared.
The corrective is not "check digests" but "verify that every pin is READ by
something".

### Unchanged and open
`ROOT-DIRECTORY-UNGOVERNED-1` (89 tracked files at the repository root),
`EVIDENCE-DISPOSITION-INCONSISTENT-1` (five directories still holding committed
machine evidence), `ONTOLOGY-ZERO-LENGTH-REFUSAL-1` (awaiting
`IDENTITY_REPLACEMENT`), `SUITE-TRANSITION-KIND-INCOMPLETE-1`,
`CERTIFICATION-SURFACE-UNIMPLEMENTED-1`, `TRANSACTION-CANNOT-EXPRESS-DELETION-1`,
`ARCHIVE-SEMANTIC-COLLISION-1`, `ARCHIVE-PATCH-INFERRED-TEXT-1`,
`AF-FIX-WORK-TRACKED-1`, `POSTFLIGHT-FEATURE-COUNT-STALE-1`,
`KAN-IMPORT-SIDE-EFFECT-1`, `TRANSACTION-GIT-FAILURE-FAILS-OPEN-1`,
`RESOURCE-HANDLE-LEAK-1`, `MANIFEST-NONDETERMINISTIC-ACROSS-RUNS-1`,
`DOWNLOADSHADOW-1`, and the fifty-four-item scientific register.

---

## 5. Ending state

```
HEAD                    99ab4ed
ratchet                 5404
suite identity digest   66fddbc60fb28e9a
gate                    5389 passed, 15 skipped, 0 failed
docs/ROADMAP.md         13,668 bytes, bound by 9 tests, plan re-derived
archived predecessor    466,826 bytes, blob 990088a61365ef3de3a02fd34327c7c5f3134731
working tree            clean, including untracked
continuous integration  green through b586778; 99ab4ed in flight
```

## 6. Next intended action

The standing plan, unchanged and quoted in the roadmap: Commit C
(SealedEvaluation), then BASELINE-1, DRIFT-1 with README-1, OP-1 step 5 against
STEP K, OP-2, and RETRAIN-GATE last. Reconciling the fifty-four-item register
against eighty-three commits is a separate unit, and its judgements are
scientific rather than clerical.

## 7. Remaining uncertainty

Whether the continuous-integration alert workflow can alert. No run has failed
in the visible window, so the `workflow_run` failure branch remains unexecuted
against a real event payload.
