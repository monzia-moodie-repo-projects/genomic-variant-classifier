# SESSION 2026-08-22 -- the record acquires a contract

**Author: Monzia Moodie**
**Commits:** `31c279a`, `e1a5297`, `f62f40d`
**Ratchet:** 5213 -> 5222 -> 5222 -> 5237
**Preceding head:** `dffe31e`
**Ending head:** `f62f40d`

> **Record status:** pre-archive-migration. Future archive class: session
> notebook. Migration required.
>
> This is the SECOND record for 2026-08-22. The first,
> `SESSION_2026-08-21_to_08-22_authority-becomes-typed.md`, covers `084ece5`
> through `69ba5f6` and is a true record of what it covers. It is not amended:
> the archive is append-dominant, and a later record is the correct way to
> continue a chronology.

---

## 0. What this session did

It made the architecture-decision record a contract rather than a naming
convention, gave three completeness invariants owners that are not README prose,
and accepted the knowledge architecture that governs where every fact lives.

Three commits, three suite transitions, two of the three kinds ADR-0003 defines
now demonstrated in production.

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `31c279a` | invariant ownership | ADDITION +9 | 5213 -> 5222 | 5207 passed, 15 skipped, 1098.32s |
| `e1a5297` | ADR-0003 | NEUTRAL | 5222 | 5207 passed, 15 skipped, 946.84s |
| `f62f40d` | decision-record contract | ADDITION +15 | 5222 -> 5237 | 5222 passed, 15 skipped, 941.78s |

The third kind, `DELIBERATE_RETIREMENT`, remains unexercised. It is the path the
README decomposition will be the first to use.

---

## 1. INVARIANT-HANDOFF-1, applied before it was written

A census on 2026-08-22 over the **entire tracked corpus** -- 1,573 files, 1,565
textual, enumerated by `git ls-files` rather than by a hand-maintained directory
list -- asked which invariants `tests/unit/test_readme_claims.py` uniquely owns.

The first answer was wrong, and the instrument was mine. The probe reported
`OWNERS OUTSIDE test_readme_claims.py: 19` for the model roster. **That is a
count of files referencing a symbol, not a count of invariant owners.** Reading
what those nineteen files actually assert:

- eight construct `VariantEnsemble` and MUTATE `base_estimators` as a fixture;
- three enumerate the roster in order to iterate over it;
- `test_catboost.py` asserts one conditional member;
- `test_base_model_dropout_is_loud.py` asserts dropout is loud, which is the
  failure mode rather than the membership;
- `test_module_docstring_is_not_a_stale_roster.py` asserts the docstring does
  NOT enumerate the roster -- the opposite binding;
- `test_model_registry.py` and `test_runtime_attribution.py` pass a
  test-supplied roster.

**Nothing compared the runtime roster to an independently authored list.**

The corrected result, per invariant:

| invariant | apparent | real owner outside the README test |
|---|---:|---|
| `INV-SUITE-SIZE` | 9 | YES -- `tests/conftest.py` and `test_suite_size_ratchet.py` |
| `INV-FEATURE-CONTRACT-CARDINALITY` | 32 | YES -- the fail-loud guard at import in `variant_ensemble.py` |
| `INV-MODEL-ROSTER-COMPLETENESS` | 19 | **NO** |
| `INV-AGENT-REGISTRY-COMPLETENESS` | 14 | **NO** -- seven wiring tests each assert ONE agent; fifteen of twenty-two had no coverage |
| `INV-DRIFT-EXIT-CODE` | 1 | **NO** -- and the single reference is a COMMENT in the ratchet |

`31c279a` gave the three unowned invariants owners and **retired nothing**. The
overlap is the handoff proof.

Six of its nine tests are negative controls. They were executed before delivery
and then re-executed against a deliberately neutered checker: **six of six
failed**, so none is decorative.

---

## 2. ADR-METADATA-INCOMPLETE-1

ADR-0003 required accepted records to declare `Status`, `Date`, `Authority`,
`Domains` and `Measured at commit`. Measuring all three records side by side
before writing the checker:

```
field                ADR-0001   ADR-0002   ADR-0003
Status               PRESENT    PRESENT    PRESENT
Date                 PRESENT    PRESENT    PRESENT
Authority            PRESENT    PRESENT    PRESENT
Measured at commit   PRESENT    PRESENT    PRESENT
Domains              ABSENT     PRESENT    PRESENT
```

**ADR-0001 declared no domains -- and ADR-0001 is the record that introduces the
domain concept.** A checker requiring the field would have failed the record
that invented it. I authored all three and did not notice until they were placed
side by side, which is the argument for measuring rather than reading.

It is amended at `f62f40d` to `**Domains:** meta`: ADR-0001 governs no single
domain, it defines the lattice by which every domain is assigned. The amendment
declares itself with an `**Amended:**` field naming the finding, the reason, and
an explicit statement that no ruling, consequence or reasoning is altered. A
header field is an index entry, not a historical claim.

The checker and the amendment are ONE unit. Separating them would have left the
suite red in between.

The checker was written blind to its outcome. Run against the real headers it
passed twelve of thirteen and failed on exactly one thing: `ADR-0001 ... missing
['Domains']`. That is the strongest validation available -- an instrument that
detects the known defect and nothing else.

### The index is bound, because an index is a second copy

`docs/architecture/decisions/README.md` enumerates the accepted records. That
makes it a second copy of the record list -- the same shape that once let
`README.md` state a feature count in nine places with four different values. It
is therefore bound to the directory by a test, and breaking it was demonstrated
to fail in both directions, naming the unlisted record and the phantom entry,
before the installer was cut.

---

## 3. Two defects in my own instruments

### DOWNLOADS-SHADOWS-TOP-LEVEL-MODULES-1, confirmed

A pre-flight reported `AttributeError: module 'catalogue' has no attribute
'create'` while constructing `VariantEnsemble`. Settled by controlled
difference, four configurations, each in its own child interpreter so a poisoned
module table could not contaminate the next:

```
A  clean sys.path          OK    roster (13)
B  + <repo>/src            OK    roster (13)
C  + Downloads             FAIL  catalogue -> C:\Users\monzi\Downloads\catalogue.py
D  + Downloads + src       FAIL  same
```

The shadowing file is the project's own `evaluation/catalogue.py`, staged in
Downloads as a delivery payload from an earlier session. Python places a
script's directory at `sys.path[0]`, and Downloads holds **236 Python files**
including `catalogue.py`, `metrics.py`, `registry.py`, `install_plan.py`,
`repository_transaction.py` and `runtime_paths.py`. A bare `import catalogue`
from the spaCy stack resolves to the project module, which has no `create`.

Package-qualified imports are immune, which is why every installer so far
worked. Bare top-level imports are not. Every installer cut since removes its
own directory from `sys.path` before importing anything.

A stale `install_plan.py` or `repository_transaction.py` in Downloads would be a
far worse version of this hazard than `catalogue.py` was.

### PROBE-VERDICT-WITHOUT-EVIDENCE-1, mine, corrected

The pre-flight printed `AttributeError: ...` and discarded the traceback, so the
import chain that produced it was invisible. That is exactly the failure the
2026-08-21 correction note names -- *a lone verdict is unfalsifiable* --
reproduced inside an instrument built to prevent it. Every installer and probe
cut since prints a full traceback from its generic handler.

---

## 4. A measurement corrected

I attributed `31c279a`'s 1,098.32-second gate to its nine added tests, calling
it plausible rather than measured. The series refutes it:

```
b115bab   998.13s @ 5213      31c279a  1098.32s @ 5222
1c50680  1010.66s @ 5213      e1a5297   946.84s @ 5222
69ba5f6   954.89s @ 5213      f62f40d   941.78s @ 5237
dffe31e   962.20s @ 5213
```

`f62f40d` ran the largest suite of the session in the shortest time. Nine tests
cannot have cost a hundred seconds. The figure was run-to-run variance and the
attribution was wrong.

---

## 5. Findings

### Closed this session

| identifier | how |
|---|---|
| `ADR-METADATA-INCOMPLETE-1` | ADR-0001 amended and the contract enforced at `f62f40d` |

### Confirmed and open

| identifier | one line |
|---|---|
| `DOWNLOADS-SHADOWS-TOP-LEVEL-MODULES-1` | 236 modules ahead of site-packages whenever a script runs from Downloads; mitigated per-installer only |
| `PROBE-VERDICT-WITHOUT-EVIDENCE-1` | a verdict printed without its traceback; corrected in every instrument since |
| `TRANSACTION-GIT-FAILURE-FAILS-OPEN-1` | clean-tree and head-unmoved assertions return early when `_git` yields None |
| `RESOURCE-HANDLE-LEAK-1` | 869 occurrences across 51 sites; four in shipped code |
| `ATTESTATION-SCHEMA-DRIFT-1` | one declared schema version, two `acceptance` shapes |
| `ROADMAP-STALE-1` / `ROADMAP-ROLE-OVERLOAD-1` | symptom and root cause; ADR-0003 adopted, migration not begun |
| `METRICSTATUS-NAME-COLLISION-1` | two `MetricStatus` enumerations |
| `KAN-REPAIR-DUAL-AUTHORITY-1`, `KAN-IMPORT-SIDE-EFFECT-1` | two repair authorities; import-time global mutation |
| `PHASE-LIST-MEMBERSHIP-OVERLAP-1` | one feature in the guarded contract and in a phase list |
| `STATE-STORE-OWNERSHIP-1` | state identity by path convention rather than registry |
| `AGENT-LIVENESS-SEMANTICS-1`, `AGENT-RUNTIME-TIMESTAMP-SEMANTICS-1` | a gate that cannot fail by default; timestamps that are not per-agent |
| `GATE-WARNING-TAXONOMY-MISLEADING-1`, `GATE-ENVIRONMENT-SPLIT-1` | occurrences reported as sites; local and remote splits differ |
| `INVOCATION-DEPENDS-ON-SHELL-STATE-1` | a delivered command block depended on the working directory |

### Suspected

`CI-FAILURE-ALERT-UNVERIFIED-1` -- the `CI failure alert` workflow has fired
twelve times and succeeded twelve times, paired one-to-one with continuous
integration completions including a duplicated pair. It triggers on completion
rather than on failure, and presumably guards its alerting step with a
condition. **It has never been observed to fire.** A workflow showing a green
tick on every run is visually indistinguishable from one whose alerting step is
silently skipped by a condition bug. Same family as `AGENT-LIVENESS-SEMANTICS-1`
and the vacuous detritus iterator.

Also unexplained: `32503182315` and `32503183706`, two full continuous-
integration runs for one commit five seconds apart, roughly forty minutes of
duplicated compute.

### Unassigned, and blocking

`docs/measurements/` (353 references) and `docs/audits/` (23) have **no plane**
in ADR-0003. Neither can be migrated until one is assigned. This gates D2c.

---

## 6. Ending state

```
HEAD                    f62f40d
ratchet                 5237
tests/ collects         5237
gate                    5222 passed, 15 skipped, 0 failed
architecture records    4 files: ADR-0001, ADR-0002, ADR-0003, README index
working tree            clean, including untracked
detritus                none
transaction journals    0
continuous integration  32551443126 PASSED 19m26s at 31c279a
```

## 7. Next intended action

Assign a plane to `docs/measurements/` and `docs/audits/`, which requires
measuring their contents first. Then D2c: the legacy ROADMAP move by `git mv`,
proven by blob object identifier equality at source and destination rather than
by rename recognition. Then the current-state reconstruction, then the README
de-loading under `DELIBERATE_RETIREMENT`.

## 8. Remaining uncertainty

Whether the `CI failure alert` workflow can alert. It has no observed failure to
prove it against, and reading its condition is not the same as watching it fire.
