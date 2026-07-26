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

*Written 2026-07-26.*
