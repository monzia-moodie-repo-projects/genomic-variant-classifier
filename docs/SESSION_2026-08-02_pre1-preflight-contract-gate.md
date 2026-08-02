# SESSION 2026-08-02 — PRE-1: a dead gate on the paid-launch path

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Commits: `f2cff8c` (PRE-1a), `db327c4` (PRE-1b). Both pushed.**
**Outcome: preflight section 13c runs for the first time, and passes for the first time. G1 PASS — 59 passed, 2 warned, 0 failed.**

Companion documents:
`SESSION_2026-08-01_op1-preflight-and-defect-register.md` (the OP-1 preflight),
`SESSION_2026-08-01_pop1a-label-eligible-population.md` (POP-1a).
**Section 6 of the second contains errors this document corrects — see section 5 below.**

---

## 1. How this was found

Not by looking for it. POP-1a's final preflight run on 2026-08-01 printed, in the
middle of an otherwise unremarkable summary:

```
=== 13c. RUN_17_PLAN.md feature contract matches the CODE ===
Method invocation failed because [System.Management.Automation.ErrorRecord]
does not contain a method named 'Trim'.
  Run_Preflight_Local.ps1:321
```

and then:

```
Passed: 53    Warned: 1    Failed: 1
```

**Section 13c contributed to none of the three.** It crashed, printed a
PowerShell error into the transcript, and the summary reported a clean-but-for-
one-known-issue preflight. The single failure was the expected uncommitted-tree
gate.

A gate that reports nothing when it breaks is worse than no gate, because it
looks like coverage.

---

## 2. PRE-1a — the crash

### 2.1 Measured cause

The import chain writes the imodelsx KAN repair banner to standard error. In
PowerShell, `2>&1` on a **native command** merges that stream as `ErrorRecord`
**objects**, not strings. So

```powershell
$codeCount = (& $venvPython -c "..." 2>&1).Trim()
```

produced a two-element array — one `ErrorRecord`, one `String` reading `95` —
and `.Trim()` member-enumerated across it and died on the element that has no
such method. **The interpreter itself succeeded: exit code 0, value 95.**

### 2.2 The guard existed, one line too late

Line 322 reads `if ($codeCount -notmatch '^\d+$') { Fail "13c: could not read…" }`.
It was written to catch a non-numeric result. The `.Trim()` on 321 threw before
it could run.

### 2.3 Confirmed pre-existing

The crash reproduces identically with `evaluator.py` stashed to its pre-POP-1a
state. Not caused by POP-1a.

### 2.4 The repair

Capture the merged stream, filter out `ErrorRecord` objects, take the last
remaining line. If nothing survives the filter, hand the guard the whole text so
its failure message carries the diagnosis rather than an empty string.

The `-c` argument was **extracted from the file rather than retyped**, so the
import path and printed expression are preserved byte-for-byte. This mattered:
the file could not be transferred to the working session — five consecutive
attempts arrived empty — and the installer was therefore built to **discover its
own anchor**, locating the single assignment combining `$codeCount`, the
interpreter invocation and `.Trim()`, printing it verbatim with twelve lines of
context, and refusing unless exactly one line matched and the numeric guard was
found within five lines below.

### 2.5 Proven before committing

Applied with the plan marker still at 97 and the code at 95, section 13c
reported:

```
FAIL  13c: RUN_17_PLAN.md asserts FEATURE_CONTRACT=97 but the code says 95
      (EXPECTED_TABULAR_FEATURE_COUNT). The plan misstates the contract under
      test. Fix the plan (or the code) BEFORE spending money.
```

Not a pass, not a crash — the exact failure the section was written to produce.
**`f2cff8c` was committed in that failing state, deliberately.** A repair that
only ever showed green would be a repair nobody had seen work. The same
discipline was applied on 2026-08-01 to POP-1a's scores regression test, which
was falsified against the pre-fix code before being trusted.

### 2.6 The defect class is closed, not just the instance

A search for the idiom `2>&1).` across `Run_Preflight_Local.ps1` returns **zero**
further hits. Of the six interpreter invocations — lines 123, 150, 256, 332, 375
and 404 — only 332 chained a string method onto the captured result. The rest
assign the raw value to a variable. PRE-1a fixed the only occurrence.

---

## 3. PRE-1b — what the crash was hiding

### 3.1 Four stale assertions, not one

`RUN_17_PLAN.md` was last committed **2026-07-13**. Four of its assertions went
stale afterwards, of which section 13c reads exactly one:

| line | assertion | truth |
|---|---|---|
| 11 | `<!-- FEATURE_CONTRACT: 97 -->` | 95 |
| 99 | `97-feature contract (88 + 3 + 6)` | 95, `86 + 3 + 6` |
| 118 | `B.D3` — 91 columns, `…COUNT=91` | 95 |
| 119 | `B.D4` — `KNOWN_ZERO_DEFAULT=25` | 24 |

Correcting only the marker would have restored the green light over three
remaining misstatements — which is precisely the condition the section's own
comment describes: *"G1 checked the plan for unfilled `<DECISION>` markers and
passed it, so the gate green-lit a paid run against a document that misstated
the very contract under test."*

### 3.2 The arithmetic, reconciled from commits

```
88   the Run-16 contract
 +3  finngen_r13_af_fin / af_nfsee / enrichment    -> 91   752335c  2026-06-27
 +6  KEGG (x2), COSMIC (x2), Nucleotide Transformer (x2)
                                                   -> 97   80eb9c8  2026-07-06
 -2  hgmd_is_disease_mutation, hgmd_n_reports      -> 95   4528414  2026-07-14
```

The HGMD pair sat **inside the 88** — `# HGMD (2)` appears between
`clingen_validity_score` and the end of that block in `TABULAR_FEATURES`, not
among the six new columns. So the decomposition becomes `86 + 3 + 6 = 95`, where
86 is the Run-16 contract less those two columns.

### 3.3 Why 97 → 95 is a leakage excision, not a count correction

Roadmap 6.21a. Two independent reasons, either sufficient.

**No licence.** HGMD Professional is a paid QIAGEN product this project does not
hold. Both columns were **constant zero for the life of the project**, occupying
two slots and *"making the roster overstate the science by two."*

**Label leakage, which survives the licence arriving.** HGMD "DM" means
disease-causing mutation; the training label here is ClinVar Pathogenic. Those
are the same quantity under two vendors' names. As a variant-level feature it is
an answer key, and **the gene-aware split cannot help, because the leak sits
inside every fold at the variant level.** A variant of uncertain significance —
precisely what this classifier exists to score — has no HGMD entry, so the flag
reads zero and the model leans benign: an excellent area under the receiver
operating characteristic curve on catalogued variants, and systematic
under-calling of the variants that matter.

If access is ever obtained, the note records the correct wiring: **gene-level and
leave-one-out** (`n_hgmd_dm_in_gene`, excluding the variant being scored),
mirroring `n_pathogenic_in_gene` — never as a variant-level flag.

The old tests asserted `hgmd_* == 0` when absent: **the defect written down as a
requirement.**

### 3.4 The digits rule, which governed the note

Lines 92-97 of the plan state that **any `<N>-feature` digit string in the file
is treated as a live assertion about the contract**, which is why the existing
correction block spells its history in words: *"ninety-one → ninety-seven."*

The note added by PRE-1b therefore reads *"ninety-seven → ninety-five."* Writing
it in digits would have planted a fresh stale assertion inside the very block
explaining why the last one happened, and 13c would then have failed on a note
about why it once failed.

The installer's post-check scans the whole file before and after and refuses
unless **exactly one** bare `<N>-feature` string remains, reading 95. Verified in
both directions: the clean run reported `BEFORE 1 (97) → AFTER 1 (95)`, and a
deliberately planted second digit string produced `AFTER 2` and exit 1.

### 3.5 Deliberately not touched

`C.2` (*"TABULAR_FEATURES 88→91 … closed in 752335c"*), `C.3`, the 2026-06-27 log
line, and **both `4/4` claims** — all accurate.

A residual, flagged and not edited: `C.3` records `KNOWN_ZERO_DEFAULT 27→25`,
while the log shows `5344ddb 27→29` and `e6447fb 29→25`, reading as
`27 → 29 → 25 → 24`. So `C.3` compresses two steps into one. Harmless, and
rewriting a historical record on a reading that could not be confirmed —
`git show 1bedf52` printed nothing for that constant — is the wrong instinct.

### 3.6 Proven

```
PASS  13c: plan FEATURE_CONTRACT (95) == code EXPECTED_TABULAR_FEATURE_COUNT (95)
```

**It has never passed before.** Built 2026-07-12 in three commits fourteen
minutes apart — `6f4904b` (*"plan said 91 features; the contract is 97. G1 now
DERIVES it"*), `01afe93` (*"machine-checkable, and the guard emphasis-proof"*),
`721a23e` (*"assert the feature contract, do not scrape it from prose"*) —
defeated by drift twenty-seven hours later, and unable to run at all until
2026-08-01.

Final state: **G1 PASS — 59 passed, 2 warned, 0 failed.** A full per-section
inventory accounts for every verdict; the count matches the summary exactly.

---

## 4. Two figures the author had wrong, corrected by measurement

Both were caught by running a command rather than by review, and both would have
made the repository worse.

**`4/4` is accurate.** On 2026-08-01 I reported *"7 passed"* for
`test_feature_count_contract.py`. It reports **4**; the 7 came from a different
file whose test names I had quoted. `B.D3`'s and `C.2`'s `4/4` are therefore
correct, and "correcting" them to 7 would have **made a true document false.**

**`KNOWN_ZERO_DEFAULT` is 24.** I was about to affirm 25 from a visual scan of an
unordered `frozenset` printed across several lines — exactly the pattern-match-
and-conclude shortcut this project's standing instructions forbid. `len()` says
24, and `tests/unit/test_harness_fixture_omim_molecular.py:65` asserts `== 24`
with a stated reason. So 24 is the **enforced** value, not silent drift — and
this constant is the one member of the cluster that already has the tripwire the
others lacked.

---

## 5. Errata for `SESSION_2026-08-01_pop1a-label-eligible-population.md`

That document was written before this work and section 6 carries readings since
superseded. It is **left in place**; this is an additive correction, on the same
reasoning the ratchet's placement erratum was handled additively on 2026-08-01 —
the error is itself evidence about how these records decay.

**E1 — POP-1a's commit hash.** Quoted as `4d20ade` in conversation across several
turns. The authoritative value, from `git log`, is **`1577f0b`**. The session
document itself does not name a hash, so no committed text is wrong; the
correction is recorded here so the conversational record is not trusted later.

**E2 — the staleness timeline, wrong three times.** Section 6.4 already records
two wrong readings. A third followed, and the plan's own text settled it. The
sequence:

- I first said the code moved on 20 and 21 July. **False** — those commits
  touched `variant_ensemble.py` without changing the constant.
- I then said the marker was *"stale on the day it was written, never correct."*
  **False** — it was correct when authored on 13 July.
- The plan's comment at lines 305-312 then showed the real story: on **2026-07-06**
  (`80eb9c8`) the contract went to 97, **the runbook was corrected (`61c2b04`),
  and the plan was not.** So the plan was *already known* to lag, was
  half-repaired, and the marker it then carried went stale again on 14 July.

Each correction came from anchored evidence, not from reasoning harder. Order
must be measured.

**E3 — how long 13c had been broken.** Section 6 does not state a duration, which
was correct: it depends on when the KAN repair began writing to standard error,
and that has **not** been established. No date is inferred here either.

---

## 6. Follow-ups — nine, none touched today beyond PRE-1a and PRE-1b

| id | item |
|---|---|
| ~~PRE-1a~~ | **CLOSED** — `f2cff8c` |
| ~~PRE-1b~~ | **CLOSED** — `db327c4` |
| **PRE-2** | *new.* Section 5's PASS line swallows the KAN banner, a `RemoteException` and a progress bar into its verdict text. Same merged-stream root cause; cosmetic, the check passes on `SMOKE_OK`. |
| **ZERO-1** | *new.* Stage 5 warns that **24 dead-connector defaults are still zero**, allowlisted and marked expected. The HGMD excision was, in part, exactly this: columns constant zero for the project's life while the roster counted them. Is each of the 24 still legitimately pending, or has the allowlist gone stale the way the plan did? A scientific question, not a preflight nicety. |
| ABS-1 | the ranking channel's refusal reported as `undefined_on_cohort`; `_absence_maps` takes `ranking_check` and never reads it |
| DEAD-1 | ~40 lines of dead absence computation in `evaluate` (1181-1220), discarded at 1240, already carrying a reason string that disagrees with the live one |
| LINT-1 | no lint gate; `ruff` reports 603 `I001`, 500 `UP045`, 409 `BLE001`, 267 `F401` |
| F821-1 | 18 undefined names; 7 are the deliberate `_ensure_sklearn` global injection, 9 need assessment, `metrics.py:1486` first |
| CMP-1 | `ModelComparison` carries a population fingerprint with no population scope beside it |
| INF-1 | an infinite reference label is pooled with `NaN` as *withheld*; it is corrupt, not missing |

**Also outstanding: `docs/ROADMAP.md`.** None of today's three commits touched it,
against the standing convention that it stays fresh after every session and
milestone. POP-1a, PRE-1a, PRE-1b and the nine follow-ups exist in session
documents and nowhere in the roadmap.

---

## 7. Final state

| item | value |
|---|---|
| `HEAD` = `origin/main` | `db327c4` |
| today's commits | `1577f0b` POP-1a, `f2cff8c` PRE-1a, `db327c4` PRE-1b |
| suite | 4134 passed, 6 skipped, 4140 collected |
| armed ratchet | `collected 4140 == EXPECTED_SUITE_SIZE 4140` |
| preflight | **G1 PASS — 59 passed, 2 warned, 0 failed** |
| warnings | 1000 Genomes absent (deferred B.D1); 24 dead-connector defaults (ZERO-1) |

Next: the roadmap; then POP-1b (the report surface — `n_source`,
`n_label_eligible`, `n_reference_label_withheld`, scope and parent fingerprint,
with a schema version bump); then REG-1 and OP-1. Then the drift monitor, whose
red is roadmap 6.20's fix working — *"THE SCHEDULED DRIFT MONITOR HAD NEVER
CHECKED ANYTHING"* — exactly as 13c's red was PRE-1a working.
