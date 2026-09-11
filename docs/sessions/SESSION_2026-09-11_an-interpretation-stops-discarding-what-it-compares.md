# SESSION 2026-09-08 to 2026-09-11 -- an interpretation stops discarding what it compares

Predecessor at the start: `fdb5473aaf3f7a8d594da10b5f793d295dffe610`
Head at the end:          `50aea177e0da477b6d03edd4d6d31d1addc21624`

Four units installed. Five defects reproduced in an installed owner before any
repair was written. Three defects found in my own instruments, and one in the
delivery record itself. One finding refuted -- someone had already fixed it and
I had been reading a stale copy.

## 0. What this session did

    G  fdb5473 -> 48913c1   a transaction stops certifying what it could not observe
    H  48913c1 -> f808944   failed inspection stops meaning deletion eligibility
    O  f808944 -> 07bc7a5   a digest binds an encoding that preserves distinctions
    I  07bc7a5 -> 50aea177  the operations package lands, unblocked by O

Suite: 6255 -> 6263 -> 6299 -> 6318 -> 6470.
Every count MEASURED at the live repository inside a rolled-back transaction;
none computed from its predecessor plus a delta.

## 1. TRANSACTION-GIT-FAILURE-FAILS-OPEN-1 -- repaired by unit G

Recorded CONFIRMED and unaddressed in ADR-0003 and defined in
SESSION_2026-08-21_to_08-22 as "`_git` returns None on failure and both
clean-tree and head-unmoved assertions return early, silently".

### Falsified before repair

MEASURED 2026-09-08 against the payload unit T installed
(`fdc02af8c111573f48c838a113b432fb60e50396dcd37989783fb2ad70be7687`), in a
repository holding one unowned untracked file:

    git AVAILABLE    -> refused, TransactionError, working tree dirty
    git UNAVAILABLE  -> CONSTRUCTED, self._head is None

One git failure at construction disabled BOTH assertions for the transaction's
whole lifetime: `_assert_tree_clean` returned early because `_git` yielded
None, and `_assert_head_unmoved` returned early because `self._head` was None.

### The control is what makes it a measurement

An earlier attempt of this experiment failed with `TypeError` on BOTH arms -- a
wrong keyword -- and the conclusion was printed anyway. Without the
git-available arm refusing, a run in which construction failed for an unrelated
reason would look identical to the exposure.

### The stale-source error, first occurrence

That first experiment ran against a SANDBOX COPY of
`repository_transaction.py` (38,567 B) which is NOT the payload unit T
installed (43,802 B). The exposure was re-measured on the installed payload and
holds there. Digest verification against the installed artifact must PRECEDE a
measurement, not follow its failure.

## 2. The category-F fail-open -- repaired by unit H

MEASURED 2026-09-08, running the installed `scripts/forensics/cleanup_apply.py`
with a working directory outside any Git working tree:

    ### (F)  2 eligible (4.0B), 0 SKIPPED ###
      would-delete  2.0B  scripts/dump_thing.py
      would-delete  2.0B  scripts/patch_thing.py

`ok = not tracked(p)`, and `tracked()` returns False whenever `git ls-files`
exits non-zero -- which it always does outside a repository. FAILED INSPECTION
BECAME DELETION ELIGIBILITY. The docstring "Refuses to delete a tracked path
under any circumstance" is literally true and operationally empty.

Category E escaped only because it also required `ignored()`, which fails the
same way but in the safe direction.

Categories A-F are PRESERVED, read line by line from the installed file.
Production application is DELIBERATELY UNAVAILABLE: `--apply` exits 3 before
repository selection or discovery, and the script imports no executor.

### A destination error caught before it was pinned

The patch destination was first declared `scripts/cleanup_apply.py`, which does
not exist. The file is under `scripts/forensics/`. The error had propagated
into the payload: the script computed `parent.parent / "src"`, correct from
`scripts/` and wrong from `scripts/forensics/`. The bootstrap was REMOVED
rather than re-depth-counted -- `scripts/retire_backup_artifacts.py` imports the
package directly, which has no depth to get wrong.

## 3. Five defects in the suite-transition owner -- repaired by unit O

All reproduced against `94e58a79bca83a696ea07c72bf4cd5f0db05e0caeb970e1ace3a4a9a2403b424`
before a line of repair was written.

1. **A separator replacement passed as NEUTRAL.** `from_pytest_output` applied
   `.replace(chr(92), "/")` to the whole line, parameter text included. Two
   collections, each internally consistent, mapped to ONE identity; snapshots
   equal, digests equal, NEUTRAL ACCEPTED for a changed suite.

2. **A plain string was accepted as a kind.** `SuiteTransition(kind="addition")`
   constructed and verified; every branch was bypassed and `_checked` was set
   regardless.

3. **Duplicates collapsed on direct construction.** Three identities, two
   identical, yielded a count of 2. The reported-count witness existed only on
   the text route.

4. **The digest was ambiguous over accepted inputs.** A snapshot of ONE identity
   containing a newline and a snapshot of TWO without one shared a digest. NOT
   a SHA-256 collision -- an encoding that could not represent inputs the
   domain wrongly accepted.

5. **The public projection emitted contradictions.** NEUTRAL evidence with
   before_count 1, after_count 999 and digests "x" and "y" was projected into
   an attestation record.

### Why the existing cross-check could not catch the first

The listing/summary witness catches colliding identities appearing TOGETHER in
one collection. It cannot catch one REPLACING the other ACROSS collections,
because each collection is internally consistent and the difference vanishes
before `verify` is reached.

### The gate refused the first attempt, and was right to

    tests/unit/test_attestation_projection.py::
    test_hand_built_evidence_is_still_checked
        expected: "added identities are not this declaration"
        actual:   "the counts move by +1 while the difference sets move by +0"

The consistency check had been placed BEFORE `_assert_evidence_belongs_here`.
That test asserts the refusal REASON, not merely that a refusal happened.
Belonging is now settled first: whether evidence is THIS declaration's to judge
must be answered before its internal coherence.

## 4. The interpretation migration

MEASURED at f808944 over the `tests` scope: 6299 node identities, of which TEN
contain a backslash and therefore change VALUE under the repair. No test is
added or removed by that change.

    tests/unit/test_suite_size_ratchet.py               6
    tests/unit/test_adaptation_agent.py                 2
    tests/unit/test_calibration_binning_convention.py   2

Historical digests stay as recorded, bound to the implementation that produced
them. The boundary is `07bc7a5`.

### Unit I was unblocked without being edited

Unit I's first dry run refused at f808944:

    observed but not declared: ...[.../..//..//..//escape.json]
    declared but not observed: ...[.../..\..\..\escape.json]
    a count of +152 cannot distinguish these.

NOTHING IN UNIT I WAS EDITED. Unit O removed the rewrite and the same 152
identities matched exactly. A count of +152 was equal on both sides; only
identity comparison could tell them apart.

## 5. Defects in my own instruments

- **A checker built from remembered phrases.** A consistency checker searched a
  hardcoded list and reported zero contradictions while a finding record said a
  pass had not run beside another field reporting its result. Replaced with
  structure: `open_questions` and `resolved_questions` with disjoint key sets.

- **A hand-written dependency list.** Unit I was declared with ONE dependency.
  A clean re-collection failed at import: `archive_manifest` also imports
  `.classification`, `.identity` and `.roles`. The census now WALKS the import
  graph with `ast`. Re-collecting rather than carrying the count is what
  exposed it.

- **A regular expression over source text**, replaced by `ast`, after the first
  version reconstructed member names by slicing text around a match.

- **A text search reported as an enumeration of consumers, twice.**
  `git grep -F 'suite_transition'` matched two source modules; both reference
  it as a JSON key or dict field name and neither imports the owner. Final
  enumeration: ZERO source consumers, seven test consumers.

- **A corpus digest over paths**, cited as evidence that contents were
  unchanged. `membership_digest` is computed over `tuple(sorted(p for p, _o in
  rows))` -- the object identifiers are discarded.

- **A stale inventory in the delivery manifest.** `.files` was generated by
  walking the directory BEFORE the same script renamed the previous revision,
  recording a path that no longer existed and omitting the current one.

## 6. A finding refuted

The review-tier filter was reported as never running, silently admitting every
tier. The INSTALLED `real_data_prep.py` (73,528 B) RAISES when
`min_review_tier < 5` and no `ReviewStatus` column exists, and its message
names the silent-keep-everything failure as the reason. The report described a
20,926-byte project-knowledge copy that is not the installed source.

What survives is larger: THIS COHORT IS NOT BUILDABLE at default settings.

## 7. Findings

### Repaired

- `TRANSACTION-GIT-FAILURE-FAILS-OPEN-1` -- unit G, eight bound tests.
- The category-F fail-open in `scripts/forensics/cleanup_apply.py` -- unit H.
- Five owner defects -- unit O, nineteen bound tests.

### Open, and prepared for

- `SUITE-TRANSITION-KIND-INCOMPLETE-1`. Re-executed 2026-09-10 and still true:
  a pure rename is expressible only as `DELIBERATE_RETIREMENT`, recording a
  retirement where nothing was retired. Unit O installed the `else` branch that
  makes adding `IDENTITY_REPLACEMENT` safe; it did not add the member.
- `ONTOLOGY-ZERO-LENGTH-REFUSAL-1`, recorded as awaiting the same kind.

### Not a defect -- a property

`INSTALLER-BASELINE-COLLISION-1`. After 07bc7a5 the repository interprets with
the repaired parser, so a successor's baseline must be MEASURED, never carried.

## 8. Ending state

    HEAD           50aea177e0da477b6d03edd4d6d31d1addc21624
    suite          6470 collected; 6455 passed, 15 skipped, 0 failed
    working tree   clean, untracked included

## 9. Next intended action

Four decisions, none of them a measurement:

1. `Q-tier-remedy` -- rebuild the cohort via `scripts/augment_reviewstatus.py`,
   or set `min_review_tier=5` to disable tier filtering explicitly.
2. The 149 identifiers carrying contradictory labels across 785 rows.
3. `IDENTITY_REPLACEMENT` as its own unit.
4. Unit A, blocked on composed semantic verification and the maintenance
   channel, both NOT YET IMPLEMENTED.

## 10. Remaining uncertainty

- The two source modules that reference `suite_transition` were read and import
  nothing from the owner. Production paths the suite does not exercise remain
  unmeasured.
- The ten migrated identities are those CONTAINING A BACKSLASH. An identity
  whose meaning changes for another reason would not appear in that filter.
- Unit O's gate ran 1869.5 s against a prior observed maximum of 1355.5 s. No
  predictive model exists and no interval is claimed.
- Whether the cohort's 149 contradictory identifiers reflect genuine ClinVar
  disagreement between distinct variants or conflicting submissions on one is
  INDISTINGUISHABLE from the current table, because the identifier merges them.
