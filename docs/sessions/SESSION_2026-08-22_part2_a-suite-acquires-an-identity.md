# SESSION 2026-08-22 part 2 -- a suite acquires an identity

**Author: Monzia Moodie**
**Commits:** `a60f18f`, `88e844e`
**Ratchet:** 5237 -> 5262 -> 5303
**Preceding head:** `ba9060d`
**Ending head:** `88e844e`

> **Record status:** pre-archive-migration. Future archive class: session
> notebook. Migration required.
>
> Third record for 2026-08-22. The earlier two cover `084ece5`..`69ba5f6` and
> `31c279a`..`f62f40d` and are true records of what they cover. The archive is
> append-dominant; a later record continues a chronology.

---

## 0. What this session did

It found that half the installers had been verifying a suite change by counting
it, gave the concept one typed owner, and then gave the evidence format a
version that changes when its shape changes.

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `a60f18f` | SuiteTransition | ADDITION +25 | 5237 -> 5262 | 5247 passed, 15 skipped, 961.59s |
| `88e844e` | attestation version 2 | ADDITION +41 | 5262 -> 5303 | 5288 passed, 15 skipped, 949.86s |

---

## 1. SUITE-NEUTRAL-IDENTITY-1

ADR-0003 established that a count is not an identity. The ADDITION installers
honoured it. **The NEUTRAL installers did not.** They verified

```
collected == expected      and      ratchet == collected
```

and nothing more, so this passes as neutral:

```
before  {test_a, test_b, test_c}      after  {test_a, test_b, test_d}
```

Four installers had each carried a private notion of "neutral" and two were
wrong. That is semantic drift re-entering a system that had just removed it, and
the cure is the same one: **one semantic concept, one typed owner.**

Two units were published under the weaker check. They are proven genuinely
neutral -- see `CORRECTION_2026-08-22_a-neutral-transition-was-not-verified.md`,
placed beside those records rather than inside them. The finding stands
regardless: a correct answer from an invalid method is the defect, not an
exception to it.

### Three guards removed as dead code

A sabotage matrix disabled each guard in the new primitive and required the
suite to detect it. **Three were not detected**, and the cause was not missing
tests -- all three are provably unreachable:

- the NEUTRAL set-equality guard is reached only after both comparisons pass,
  and for NEUTRAL both expected sets are empty, so `after == before` follows;
- the count/identity cross-check asserts a set identity true for all finite
  sets, and its own comment said it could not fail;
- the ADDITION rising-count guard is entailed by the construction rules plus
  the comparisons above it.

They were removed. **Defence in depth that cannot fire is not defence** -- it is
the shape this project keeps finding, and it is worse than absence because it
reads as protection. Twelve guards remain; a second matrix over all twelve,
including the construction-time validations, detected every one.

---

## 2. ATTESTATION-SCHEMA-DRIFT-1

Nine attestations existed, all declaring `"schema_version": 1`, in **three**
shapes -- because every installer hand-built its own dictionary and each
recorded what the previous had learned to miss: `measured_by`, then the full
outcome taxonomy, then a top-level `suite_transition`, then `amendments`.

Version 2 refuses any undeclared field and enforces **cross-field consistency**,
so an attestation cannot contradict itself. The binding that matters most:

```
passed + skipped + xfailed  ==  counter.after
```

A gate summary and a collection count are two measurements of one suite, and
nothing until now required them to agree *inside the recorded evidence* -- the
same class of gap as an acceptance line that recorded zeroes because nothing
checked it against a gate.

**Version 1 documents are not migrated.** A schema written afterwards is not a
standard to hold earlier records to, and a record corrected in place is no
longer a record. `validate` refuses to judge them and says so.

The suite-transition unit produced identity digests that belonged in an
attestation and **deliberately did not add them**: widening a corrupt format to
carry better evidence corrupts the evidence. The schema was the prerequisite
that was owed first.

---

## 3. A suite now has an identity, and continuity is provable

```
5237  75fd25f457dfa55d      before a60f18f
5262  29978734b1cf6b8a      after a60f18f, and before 88e844e -- IDENTICAL
5303  972c352bd2b7ca08      after 88e844e
```

The middle line is the point. The digest `a60f18f` recorded as its `after` is
byte-for-byte the digest `88e844e` measured as its `before`. **Suite continuity
across a commit boundary is now proven by membership rather than by count** --
the first time this project has been able to say that.

---

## 4. Two defects in my own instruments

### PROBE-OVERREFUSAL-1

The probe examining whether the two neutral commits were genuinely neutral
reported `PREMISE DOES NOT HOLD` and `NOT PROVABLE`. The verdict is wrong. Its
filter matched *"this expression mentions the changelog"* rather than *"this
parametrization derives identities from content"*; the flagged parametrization
uses literal identifiers, so content cannot reach a node identity.

> A checker that fails closed is right to refuse and wrong to be believed
> without examination. A refusal is a claim, and a claim is checkable.

This is the mirror image of the vacuous checks this project keeps finding. Those
accept because they cannot reject; this one rejected because it could not
discriminate.

### A sabotage harness that crashed on its own mechanism

The first sabotage matrix failed with `AttributeError: 'NoneType' object has no
attribute '__dict__'`. `@dataclass` resolves `cls.__module__` through
`sys.modules`, so a module object must be **registered before its body
executes**. That was a defect in the harness, not in the primitive, and it was
fixed rather than worked around. Every installer since registers payload modules
in the same order.

---

## 5. Conventions measured rather than assumed

`a60f18f` was the first unit of the session to add a module to `src/`, so it was
the first to face fifteen tests that walk a source tree.

The stated convention that library modules carry a module-level
`logging.getLogger(__name__)` is **not enforced by any of them**. Confirmed two
independent ways: no walker asserts it, and `install_plan.py` -- the direct
sibling in the same package -- has no logger and is green. Adding an unused
logger to satisfy an unenforced convention would be decoration.

`transactions/__init__.py` is ninety-four lines of which ninety-one are
docstring: **no `__all__`, no re-exports**. The package-export discipline
enforced by `test_conformal_package_exports.py` is scoped to the conformal
package alone, so no `__init__` update was owed. Measured, not assumed.

---

## 6. Findings

### Closed

| identifier | how |
|---|---|
| `SUITE-NEUTRAL-IDENTITY-1` | one typed owner at `a60f18f`; twelve guards, all detectable |
| `ATTESTATION-SCHEMA-DRIFT-1` | version 2 with cross-field consistency at `88e844e` |
| `ADR-METADATA-INCOMPLETE-1` | closed earlier at `f62f40d` |

### Open

`TRANSACTION-GIT-FAILURE-FAILS-OPEN-1`, `RESOURCE-HANDLE-LEAK-1` (four shipped
sites), `ATTESTATION-NOT-PRESERVED-1` (ten attestations, all only in a downloads
directory, all cited by commit messages), `ROADMAP-STALE-1` and
`ROADMAP-ROLE-OVERLOAD-1`, `DOWNLOADS-SHADOWS-TOP-LEVEL-MODULES-1`,
`METRICSTATUS-NAME-COLLISION-1`, `KAN-REPAIR-DUAL-AUTHORITY-1`,
`KAN-IMPORT-SIDE-EFFECT-1`, `PHASE-LIST-MEMBERSHIP-OVERLAP-1`,
`STATE-STORE-OWNERSHIP-1`, `AGENT-LIVENESS-SEMANTICS-1`,
`AGENT-RUNTIME-TIMESTAMP-SEMANTICS-1`, `GATE-ENVIRONMENT-SPLIT-1`,
`PROBE-OVERREFUSAL-1`.

### Narrowed

`CI-FAILURE-ALERT-UNVERIFIED-1` moves from SUSPECTED to **REFUTED-BY-DESIGN with
one residual**. The workflow names the 2026-07-21 incident it exists for, where
main stayed red for two hours and nobody looked; it carries a `workflow_dispatch`
path that treats a missing payload as a failure precisely so the alert branch is
exercisable; and a test pins its trigger and permissions. What remains unproven
is that a real `workflow_run` failure event produces a notification -- no
continuous-integration run has failed in the visible window.

### Unassigned, and blocking

`docs/measurements/` (73 references from outside itself, four of them from
`src/`) and `docs/audits/` (23, all citations) have no plane in ADR-0003. The
first is load-bearing for source code and tests in a way the second is not.
Assigning them is a governance ruling.

---

## 7. Ending state

```
HEAD                    88e844e
ratchet                 5303
suite identity digest   972c352bd2b7ca08
gate                    5288 passed, 15 skipped, 0 failed
working tree            clean, including untracked
detritus                none
transaction journals    0
continuous integration  green through ba9060d; a60f18f and 88e844e in flight
```

## 8. Next intended action

`MIGRATION-SCOPE-ISOLATION-1` recorded, then D2c -- the legacy ROADMAP move by
`git mv`, proven by blob object identifier equality rather than by rename
recognition. D2c is **not** blocked by the unassigned planes: it moves
`docs/ROADMAP.md`, which is neither of them.

## 9. Remaining uncertainty

Whether the continuous-integration alert can alert. Reading its condition showed
the path is reachable; only a deliberate red run shows it works.
