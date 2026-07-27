# SESSION 2026-07-26 -- P6 R2 provenance correction, phase 1: the golden capture

**Branch point:** `origin/main` at `23290dc`, ratchet 3121, badge 3121, tree clean.
Roadmap ordering item 4, "P6 R2 probe + superseding audit (bounded provenance repair)".

---

## 1. WHY A PHASE 1 EXISTS AT ALL

The roadmap's R2 plan is a five-step restructuring of
`scripts/probe_clean_cohort_p6_2026-07-25.py`. That probe generates the counters in
`docs/measurements/CLEAN_COHORT_P6_AUDIT_2026-07-25.txt`, which gate cohort-version-2
certification.

A restructuring of evidence-generating code may only be trusted if it reproduces the
prior answers EXACTLY. Synthetic fixtures prove logic; they cannot prove equivalence
on the real 4,420,180-row cohort. The correct order is therefore: freeze the current
answers from the real input FIRST, restructure SECOND, and require exact reproduction.

Phase 1 is that freeze. It changes no computation.

---

## 2. WHAT THE SOURCE READING FOUND, BEFORE ANY CODE WAS WRITTEN

### 2.1 The two disputed counts are computed over DIFFERENT universes

```python
# probe line 490-491  -> 63
p6_reprrow_label_changes = sum(1 for v in p6_reprrow_label
                               if base_vlabels.get(v) != p6_reprrow_label.get(v))
# probe line 493      -> 203
p6_group_label_changes  = sum(1 for v in p6_labels
                               if base_vlabels.get(v) != p6_labels.get(v))
```

`p6_reprrow_label` is keyed by variants that HAVE a P6 representative row.
`p6_labels` is keyed by EVERY variant. `run_p6` at lines 318-320 quarantines
`IRREDUCIBLE_CONFLICT` variants with `continue`, selecting no representative at all.

A quarantined variant therefore appears in the 203 universe and is absent from the
63 universe. **The roadmap's stated invariant `n01 + n11 == 203` assumed a shared
universe that the source does not have.** That assumption is falsified and replaced.

### 2.2 There are THREE estimands in the file, not two

| notion | value | universe | includes quarantined |
|---|---|---|---|
| representative ROW changed | 232 | row-index symmetric difference | yes |
| representative-row LABEL changed | 63 | variants with a P6 representative | no |
| group-adjudicated LABEL changed | 203 | all variants | yes |

Line 87 called the second "canonical"; line 65 used "canonical" for the basis of the
third. One word, two estimands, and a third notion nobody had named.

### 2.3 A POLICY INVARIANT, not an empirical accident

Every variant P6 newly quarantines necessarily has a group-label change. For P6 to
quarantine, the group must hold pathogenic and benign at the same unified best tier,
which forces `is_conflict = True` in `select_repr_row`; P0 then keeps a row only when
its legacy best-tier set is a single binary class. So the legacy label is necessarily
binary and the P6 label is necessarily None.

Consequently the "not applicable, group label unchanged" cell can only contain
variants quarantined by BOTH policies.

### 2.4 A SECOND overloaded label, previously unrecorded

The acceptance line reads `P6 explicit conflicts preserved (not discarded): 112`.
The quantity computed at line 494 is

```python
sum(1 for st in p6_states.values()
    if st in ("IRREDUCIBLE_CONFLICT", "AMBIGUOUS_AT_BEST_TIER"))
```

-- withheld-label STATES. `IRREDUCIBLE_CONFLICT` means an opposed binary at the best
tier, which need not contain any explicit "conflicting classifications" value at all.
The synthetic fixture demonstrates it: 3 published = 2 irreducible + 1 ambiguous, and
one of the irreducible groups contains no explicit conflict value. The real 112 is
decomposed by the golden capture in the same run.

---

## 3. WHAT WAS BUILT

### 3.1 An additive capture on the existing probe

`--emit-json PATH` writes every counter the run produced. It records values already
computed and already emitted; it changes no computation and no line of the artifact.

**Proven, not asserted.** A pristine baseline was captured from the `origin/main`
probe via `git stash`, then compared against the modified probe writing to the SAME
output path:

```
capture DISABLED -- stdout identical to baseline : True
capture DISABLED -- artifact sha256 574c8257f79cfbdb3918110a3a5fef12
capture ENABLED  -- artifact sha256 574c8257f79cfbdb3918110a3a5fef12
stdout gains exactly one line: WROTE <path>/capture.json
strict JSON (allow_nan=False) round-trips
```

Two earlier attempts failed on harness artifacts -- different output filenames, then
different temporary directories -- and the harness was corrected rather than the
assertion relaxed.

### 3.2 A permanent guard: tests/unit/test_p6_probe_contract.py

Eleven tests pinning the probe's answers on an eight-variant synthetic cohort, each
variant constructed against a specific code path rather than chosen arbitrarily. V8
is the decisive one, derived from the tier maps rather than guessed:

  * `criteria provided, conflicting classifications` is legacy tier 4, unified tier 3.
  * The legacy best-tier set is the pathogenic row alone, so P0 resolves and KEEPS it.
  * The unified best-tier set is BOTH rows, so P6 Rule 3 fires and QUARANTINES.
  * Legacy label 1, P6 label None: the group label changes and there is NO P6
    representative row to compare.

On that fixture the corrected reconciliation holds and the roadmap's original fails:

```
n01 + n11 + n_na1 = 1 + 0 + 1 = 2 = group-label changes     CORRECTED   holds
n10 + n11         = 1 + 0     = 1 = repr-label changes      CORRECTED   holds
n01 + n11         = 1        != 2                           ORIGINAL    FAILS
```

### 3.3 A near-miss worth recording

The guard's first version used `pytest.importorskip("pyarrow")` at module level.
Measured on the two interpreter legs:

```
3.11 (has pyarrow) : 11 passed
3.12 (no pyarrow)  :  1 skipped     <- eleven tests became one entry
```

`pyarrow==23.0.1` is pinned at requirements.txt:89 and is what the probe reads the
cohort with; it is not optional. `test_preflight_run17.py` and
`test_spliceai_parquet_default.py` already import it directly, and
`test_ci_failure_alert_workflow.py` records the reasoning for PyYAML verbatim.
Replaced with a direct import, so absence is a loud collection error.

**The suite-size ratchet cannot catch this.** Collection precedes skipping, so both
legs collect 11 either way. This is the mechanism by which the graph-neural-network
branch went untested for 508 Continuous Integration runs (roadmap 6.17).

---

## 4. THE SCHEMA CORRECTION THE EVIDENCE FORCES

The roadmap asked for five booleans. The live computation proves the second cannot
honestly be a total boolean. The typed contract is revised, not the data:

```python
representative_row_changed:        bool
representative_row_label_changed:  bool | None   # None when P6 selected no
                                                 # representative row
final_adjudicated_label_changed:   bool
trainability_changed:              bool
quarantine_changed:                bool
```

Four total boolean deltas and one nullable comparison whose applicability is
determined by representative-label availability. Encoding "not applicable" as False
would preserve the roadmap equation syntactically by changing the meaning of the
predicate, and would put two incompatible states into one cell.

---

## 5. OPEN, FOR PHASE 2

1. Run the capture against the real cohort; freeze the golden reference.
2. Confirm the regenerated artifact is byte-identical to the committed one -- which
   simultaneously proves additivity on real data AND that the cohort has not moved
   since 2026-07-25.
3. Restructure into ProbeConfig / load / compute / summarize / render, requiring
   exact reproduction of every golden counter.
4. Emit Table A, Table B, the transition matrices, and the corrected reconciliation.
5. Write CLEAN_COHORT_P6_AUDIT_2026-07-25_R2.txt as a superseding artifact; append
   only a supersession pointer to the original, preserving provenance.

---

---

## 6. PHASE 2 -- THE RECONCILIATION (2026-07-26)

### 6.1 The golden capture settled both open questions

```
states partition the universe   27 + 1,358,564 + 85 + 2,718,384 + 338,917 = 4,415,977  EXACT
trainable = the binary states   338,917 + 1,358,564 = 1,697,481 = P6 trn/pos/neg        EXACT
the "112" decomposes            IRREDUCIBLE_CONFLICT 85 + AMBIGUOUS_AT_BEST_TIER 27     EXACT
not-applicable = quarantined    4,415,977 - 4,415,892 = 85 = quarantined = irreducible  EXACT
```

**75.9 per cent of the published "explicit conflicts preserved: 112" is
IRREDUCIBLE_CONFLICT** -- opposed binaries that need not contain any explicit
"conflicting classifications" value. At most 27 of the 112 involve one. The
acceptance-readout label is materially wrong, not merely imprecise.

The not-applicable population is **85 variants**, exactly the quarantined set. The
nullable field now has a measured population behind it.

WHAT THE CAPTURE DOES NOT DETERMINE. `policy_table.P6.quar = 22` is a SYMMETRIC
DIFFERENCE while `p6_quarantined_variants = 85` is a cardinality. From
`85 + |base| - 2|intersection| = 22` there are 23 feasible solutions. Only if
`base_quar` is a subset of `p6_quar` does `|base_quar| = 63`. That is plausible and
unproven, so phase 2 MEASURES n_na1 rather than inferring it.

### 6.2 Architecture: a new module, not a restructured probe

The probe is now the golden reference. Restructuring it would put that reference at
risk for no gain. Instead `scripts/probe_p6_r2_reconciliation.py` IMPORTS its pure
adjudication functions, so the policy has exactly one source of truth, and layers
cleanly:

```
ProbeConfig (typed, frozen)  ->  load_inputs  ->  compute_policy_deltas
                                              ->  summarize  ->  render_report
```

`compute_policy_deltas` makes ONE pass and returns immutable `PolicyDelta` records.
Every table is derived from that single collection, so no two tables can silently
count different universes. `check_invariants` RETURNS its failures rather than
raising, because a single assert reports the first violation and hides the rest --
the wrong shape for evidence.

### 6.3 THREE DEFECTS IN MY OWN NEW CODE, ALL CAUGHT BY LOOKING AT OUTPUT

1. **`run_single_row_policy` returns two values, not three.** Caught at first run.
2. **The supersession block was appended SEVENTY-EIGHT times.** Writing
   `"\n" + "=" * 78 + "\n" f"..." "-" * 78 + ...` concatenates the ADJACENT
   LITERALS first, then applies `* 78` to the joined string. Measured: 166 lines
   where about 11 were expected, with the header repeated 78 times. Rebuilt as a
   list of lines so no literal ever neighbours a `*`.
3. **It claimed golden reproduction with no golden file loaded.** The golden checks
   were skipped, the failure list was therefore empty, and the report printed
   "reproduces the frozen reference EXACTLY". A claim that passed because it never
   ran -- the fourth instance of that defect class in one day. Now an absent golden
   prints NOT VERIFIED, exits 2, and REFUSES to supersede: an unverified run must
   never supersede verified evidence.

### 6.4 Verified

```
14 reconciliation tests + 11 probe-contract tests = 25 passed (Python 3.11)
matching golden   -> exit 0, supersession appended exactly ONCE, original preserved
re-run            -> still exactly one pointer (idempotent)
sabotaged golden  -> exit 1, failure named, original UNTOUCHED
absent golden     -> exit 2, NOT VERIFIED, original UNTOUCHED
3.10 floor: both files parse.  LF only, no byte-order mark, pure ASCII.
```

Ratchet 3132 -> 3146.

---

---

## 7. PHASE 2b -- THE GATE FIRED, AND WHAT IT FOUND (2026-07-26)

The first real run of the reconciliation FAILED, exactly as designed:

```
GOLDEN reproduction failed -- representative-row LABEL changed: recomputed 53,
                              frozen reference 63
exit 1; the original artifact was NOT superseded
```

That is the guard doing its job on a ten-variant disagreement in 4.4 million rows.

### 7.1 Four hypotheses, three falsified by reading the source

  * different `order` -- FALSE. Both use `list(range(n))`, and `select_repr_row`
    uses `sorted(idxs)`, ignoring the passed order for grouping.
  * different label-map construction -- FALSE. `variant_labels_from_kept` is
    `out[vid[i]] = labels[i]`, identical.
  * legacy quarantines where P6 does not -- FALSE STRUCTURALLY. The legacy-to-
    unified tier map is monotonic and MERGES tiers (L3->U3, L4->U3, L5->U4), so the
    unified best-tier set is always a SUPERSET of the legacy one. If legacy sees
    opposed classes at its best tier, unified does too. Hence base_quar is a subset
    of p6_quar.
  * a `(None, False, ...)` fall-through in `select_repr_row` -- FALSE. Every path
    for P0 returns either a row index or `(None, True, ...)`.

### 7.2 The fix does not depend on being right about the cause

The published figure is now REPLAYED rather than derived. Probe lines 464 and
514-517 build two maps and count, over the SECOND map's keys,
`base_vlabels.get(v) != p6_reprrow_label[v]`. The `.get()` is load-bearing: a
variant absent from the legacy map compares against None.

`replay_published_representative_label_changes` performs exactly that computation,
so the frozen counter is reproduced BY CONSTRUCTION. The stricter quantity -- both
sides present -- is still derived from the PolicyDelta collection, and a BRIDGE
classifies every variant where the two definitions disagree into one of four named
categories, with example variant identifiers printed. The difference is therefore
MEASURED on the next run rather than argued about.

### 7.3 A THIRD overloaded quantity

If the bridge reports `counted_but_legacy_had_NO_representative`, then the figure
published as 63 counts a P6 representative label differing from the ABSENCE of a
legacy representative -- a comparison against a missing row, not against another
label. That would be the third overloading in the same artifact, after "canonical"
and "explicit conflicts preserved". Both numbers are reported; neither is discarded.

Ratchet 3146 -> 3150.

---

---

## 8. PHASE 2c -- THE FIRST SUCCESSFUL RUN, AND WHAT IT FALSIFIED

`EXIT=0`, 3 min 22.9 s, golden reproduced exactly. Every internal check closes:

```
Table A total   4,415,870 = 4,415,660 + 157 + 24 + 29
Table B total         107 =        90 + 17
sum             4,415,977 = the variant universe
n10 + n11              53 = the strict representative-label changes
n01 + n11 + n_na1     203 = the group-adjudicated label changes
label transitions sum 4,415,977; those that CHANGED = 138+51+11+3 = 203
bridge  10 + 29 + 24 = 63 (published);  strict 29 + 24 = 53
```

**n11 = 29** and **n_na1 = 17** are measured for the first time. Both fall inside
the bounds the golden capture fixed (n_na1 <= 85; n01 + n11 in [118, 203]).

### 8.1 THE STRUCTURAL PROOF IN SECTION 7 IS FALSIFIED

Section 7 argued that `base_quar` is a SUBSET of `p6_quar`, so P6 could only ADD
quarantines. **The data says the reverse.**

```
Table B universe (EITHER side missing)                      107
  neither side has a representative (BOTH quarantined)       85
  legacy missing, P6 PRESENT       (legacy-only quarantine)   22
  P6 missing, legacy PRESENT       (NEWLY quarantined)         0
quarantine changes (symmetric difference)                    22
newly quarantined by P6 AND lost a binary label               0
```

So `p6_quar` is a STRICT SUBSET of `base_quar`: 107 against 85, difference 22.
**P6 never newly quarantines on this cohort; it UN-quarantines 22 variants**, and
the newly-quarantined cell measures exactly zero.

A SIXTH report defect, found on the run after the fix: Table B first emitted TWO
OVERLAPPING counts -- "no P6 representative 85" beside "no legacy representative
107" under a universe of 107, where 85 + 107 = 192. Two overlapping counts printed
as a breakdown read as a partition and are not one. Replaced by three DISJOINT
cells with an invariant that refuses any set of cells that fails to partition the
universe.

WHY THE PROOF FAILED. I considered only the "both classes at the best tier" case.
`select_repr_row` line 249 keeps a row only when `len(classes) == 1 AND classes <=
{0, 1}`. A legacy best tier holding only non-binary rows gives `classes == {None}`
-- one class, but NOT a subset of `{0, 1}` -- so the legacy policy falls through to
`conflict_irreducible` and QUARANTINES. The unified best tier, being a superset
because the map merges (legacy 4 -> unified 3), can include a binary row and give
P6 a label where legacy had none. That is the 22, and I missed it.

The claim written into the earlier draft -- "a variant P6 newly quarantines
necessarily loses a binary label" -- is structurally true and **vacuous**: the
population is empty. The report now states the measured direction instead.

### 8.2 FIVE DEFECTS IN MY OWN R2 REPORT, ALL OF THE DISEASE IT CORRECTS

  1. "of which both sides had a label : 53" -- FALSE. 29 of the 53 have a legacy
     representative whose own label is None. The strict count means both had a
     ROW. Overloaded wording, in the artifact written to remove overloaded wording.
  2. Table B labelled "variants with no P6 representative row (quarantined)" --
     it is variants where EITHER side lacks one, two different populations of 85
     and 22 reported as one.
  3. Table B's prose asserted the newly-quarantined mechanism as though it were
     operative, when the measurement is zero.
  4. The THIRD OVERLOADED QUANTITY note was printed BETWEEN the two reconciliation
     lines, orphaning the 203.
  5. `TableB` carried no decomposition at all, so the 85/22 split could not be
     reported even had the wording been right.

All five corrected; five tests added that fail if any returns. Ratchet 3150 -> 3154.

### 8.3 The cohort, as now measured

```
label transitions          None->None 2,718,482   0->0 1,358,513   1->1 338,779
                           None->1 138   None->0 51   0->None 11   1->None 3
```

P6 grants a binary label to 189 variants the legacy policy left unlabelled, and
withdraws one from 14. The net is +175 trainable variants, and the entire
adjudication debate concerns 203 variants -- 0.0046 per cent of the cohort.

---

---

## 9. PHASE 2e -- TABLE B BECOMES A JOINT TABLE

Phase 2d produced the right partition on real data -- 85 / 22 / 0 summing to 107 --
but stored the ROW and COLUMN marginals INDEPENDENTLY. That representation cannot
distinguish these two cohorts:

```
X: all 17 group-label changes fall among legacy-missing / P6-present variants
Y: all 17 fall among neither-side variants
```

Both serialise as `n_na0=90, n_na1=17, neither=85, legacy_only=22, p6_only=0`.
Correct margins, different science, and **no invariant over those marginals can
tell them apart** -- the same failure shape as every earlier defect in this
artifact, one level up.

### 9.1 Six stored cells, every margin derived

`TableB` now stores a 3x2 joint table and derives everything else as properties:
the three representative-availability row totals, the two label-change column
totals, the universe, both quarantine cardinalities, and the direction sentence.
There is one stored truth, so a count cannot be paired with the wrong population.

Emitted form:

```
                                            group-adjudicated label changed
                                                 no        yes      total
  neither side has a representative              85          0         85
  legacy missing, P6 present                      5         17         22
  P6 missing, legacy present                      0          0          0
  ---------------------------------------- ---------- ---------- ----------
  total                                          90         17        107
```

### 9.2 The direction sentence is DERIVED, and proven so

`quarantine_direction` compares the two derived cardinalities. On the real cohort
it reports **P6 UN-QUARANTINES**; on the synthetic fixture, where V8 is newly
quarantined, the same code reports **P6 NEWLY QUARANTINES**. A test asserts both.
The previous version hard-coded the real-cohort conclusion into a renderer that
also runs on synthetic data.

### 9.3 The test helper could not express the defect it tested

The old `delta()` set `p6_quarantined = quar or legacy_quar`, so a LEGACY-ONLY
quarantine -- legacy True, P6 False -- was inexpressible. That is exactly the
107-to-85 transition the real cohort exhibits. A helper that cannot construct the
case cannot test it. The two states are now independent inputs, `quarantine_changed`
is derived, and a parametrised test asserts all four combinations.

### 9.4 PolicyDelta now refuses inconsistent records

`__post_init__` rejects: an applicable comparison stored as None; a non-applicable
one stored as a boolean; `representative_row_changed` disagreeing with the row
identities; and `quarantine_changed` disagreeing with the quarantine states. Every
table is derived from this collection, so a malformed record propagates into every
published number; the constructor is the only place the check cannot be skipped.

Ratchet 3155 -> 3165. Forty-four tests across the two files.

---

---

## 10. PHASE 2f -- THE MACHINE-READABLE SIDECAR, AND CLOSURE

`CLEAN_COHORT_P6_AUDIT_2026-07-25_R2.json` is emitted beside the text report, from
the SAME `Reconciliation` instance:

```
Reconciliation
    +-- render_report()   for a human reader
    +-- serialize_json()  for exact regression checks and audit tooling
```

Values are never reconstructed independently. Doing so would reintroduce precisely
the failure mode this artifact exists to remove -- two descriptions of one thing
drifting apart.

It stores the six Table B cells and the four Table A cells, and DERIVES every
margin into a `derived` block, so a post-run gate can assert typed values instead
of parsing prose. `quarantine_direction` is a stable TOKEN
(`P6_UNQUARANTINES` / `P6_NEWLY_QUARANTINES` / `QUARANTINE_CARDINALITY_UNCHANGED`)
beside the human sentence: prose may be reworded, the token may not.

Strict serialization -- `sort_keys=True` for a stable diff, `allow_nan=False`
because a non-finite counter is a computation that failed silently. A run that
fails the golden check writes `golden_reproduced: false`, and a test asserts it.

### CLOSURE CRITERION, adopted

> Prose may describe only quantities derivable from the persisted joint structure.

Forty-eight tests across the two files. Ratchet 3165 -> 3169.

### The arc of this correction, for the record

The scientific numbers have been stable since the first successful run:
`n11 = 29`, `n_na1 = 17`, golden counters reproduced exactly. Every iteration since
has been about EVIDENCE REPRESENTATION, and each one found a real defect of the
same species the artifact was written to cure:

  * "canonical" naming two estimands            -- the original defect
  * "explicit conflicts preserved" counting states, not conflicts
  * the published 63 comparing against a MISSING ROW
  * "both sides had a label" when 29 of 53 had no legacy label
  * Table B labelled as one population when it held two
  * two OVERLAPPING counts printed as a partition (85 + 107 under 107)
  * independent MARGINALS concealing joint structure

The last was the deepest: correct row totals and correct column totals, with the
association between them unrecorded. The joint table ends that class, and the
sidecar ends the possibility of the prose and the numbers disagreeing.

---

*Written 2026-07-26.*
