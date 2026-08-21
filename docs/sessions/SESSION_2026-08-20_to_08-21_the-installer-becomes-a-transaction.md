# SESSION 2026-08-20 to 2026-08-21 -- the installer becomes a transaction
**Author: Monzia Moodie**
**Commits:** `954343e` through `f125187` -- thirteen, across 2026-08-20 and 2026-08-21
**Ratchet:** 5131 -> 5213
**Preceding head:** `2755d73`

---

## What these two days were

One question, carried to its end: **can an installer be a single atomic
repository state transition?**

The answer that had been in force was no. An installer wrote files directly,
backed each one up to a `.bak_<timestamp>` sibling, and left that sibling
behind forever. Three sweeps across 2026-08-19 and 2026-08-20 retired 280
artefacts totalling roughly 22 megabytes, every one of them produced by a
mechanism nobody thought of as producing residue.

By the end of 2026-08-21 an installer had run that declared its complete write
set before touching the repository, proved the actual writes equalled the
declared ones, ran its acceptance gate while rollback was still available,
verified hygiene before commit and attested it after, and left **three changed
files and nothing else**.

The contrast is measured rather than argued. The conventional bootstrap that
landed the machinery SUCCEEDED and left six `.bak_` files. The transactional
installer FAILED its gate on an earlier attempt and left the repository
byte-identical.

---

## The commits

| commit | time | what |
|---|---|---|
| `954343e` | 08-20 00:43 | INSTALLER-TRANSACTION-1 step 5: one authority for repository hygiene |
| `719d14c` | 08-20 02:00 | SCRATCH-IGNORE-ENVIRONMENT-1: a test asserted a property of my working tree |
| `2ef629c` | 08-20 06:31 | docs: retirement manifest for the scratch-ignore repair |
| `15830cc` | 08-20 07:17 | RELOCATION-UNWIRED-1: a capability no consumer invoked |
| `b3c5e80` | 08-20 13:26 | DETRITUS-WALK-COST-1 and RELOCATION-FALSE-POSITIVE-1 |
| `559ca58` | 08-20 15:27 | docs: retirement manifest for the hygiene repairs |
| `775d16c` | 08-20 21:31 | INSTALLER-TRANSACTION-1 step 4: the conventional bootstrap |
| `be645d1` | 08-20 23:02 | GITIGNORE-UNANCHORED-INSTALL-1: a root rule that matched every depth |
| `5d3a955` | 08-20 23:10 | GITIGNORE-UNANCHORED-INSTALL-1: the anchor itself |
| `441d899` | 08-20 23:50 | docs: retirement manifest for the step-4 bootstrap |
| `4a3f13d` | 08-21 00:07 | RUNNER-COUNTER-SCOPE-1: the runner measured a scope the ratchet does not describe |
| `5864f73` | 08-21 00:43 | INVARIANT-SELF-REFERENCE-1: the runner names its own transaction |
| `f125187` | 08-21 01:01 | INSTALLER-TRANSACTION-1 step 4: the first transaction-installed payload |

Ratchet checkpoints, read from each commit's own blob:

```
2755d73  5131   (the preceding head)
954343e  5175   +44   the hygiene authority
719d14c  5176   +1    the scratch-ignore repair
15830cc  5180   +4    relocation wired
b3c5e80  5185   +5    the pruned walk and its repair
775d16c  5207   +22   the bootstrap machinery
f125187  5213   +6    the transactional payload
```

---

## One authority for repository hygiene (`954343e`)

`SECRET_PATTERNS` and `SECRET_CANARIES` were each defined TWICE -- in
`transactions/repository_transaction.py` and `scripts/retire_backup_artifacts.py`,
eleven and seven entries respectively. Measured by IMPORTING both modules rather
than comparing their source, they were identical: element for element, order
included.

One copy had been written by transcribing the other. Nothing enforced the
agreement, and a third consumer was about to arrive.

Agreement by discipline is exactly what makes duplication dangerous.

The installer's own census counted FIVE literal lists outside the new
authority, not the two predicted: `BACKUP_PATTERNS` had a single definition and
so was never a "pair". After the commit that figure is zero, and a test walks
`src/` and `scripts/` so a sixth cannot appear.

**Three questions that were being conflated** now have three names:

```
NOT_THIS_REPOSITORY   .venv312, .git -- not this project's artefacts
SCRATCH_ROOTS         .af_fix_work -- backups PERMITTED here
BACKUP_SHAPES         the shapes indicating rollback residue
```

`.gitignore` answers "should git normally show this path?", NOT "may this path
legitimately contain rollback detritus?". Scratch roots are DECLARED; a test
asserts the declaration and `.gitignore` CORRESPOND without either deriving
meaning from the other.

**A real defect surfaced while wiring it.** The retirement tool DELETED a backup
inside `.af_fix_work` whose original was resolvable. The twelve real scratch
files had survived every earlier sweep only because THEIR originals happened to
be untracked -- an accident of that data, not a policy. Measured at `2755d73`:
zero scratch backups have a resolvable original, so the fix is PREVENTIVE. No
artefact was rescued.

> An outcome one approves of, produced by a mechanism one has not checked, is
> not evidence the mechanism is right.

---

## A test that asserted a property of my working tree (`719d14c`)

Continuous integration FAILED at `954343e` on Python 3.11:

```
test_backup_artifacts.py:133
  test_the_declared_scratch_roots_are_also_git_ignored
AssertionError: .af_fix_work is declared scratch but git does not ignore it
```

`.gitignore:198` is `.af_fix_work/` -- a DIRECTORY rule. Given a bare name for
a path that does not exist, git cannot know it is a directory, so the rule does
not apply. Measured in a fresh clone:

```
query                     directory ABSENT   directory PRESENT
.af_fix_work              exit 1  NO MATCH   exit 0  matched
.af_fix_work/probe.bak    exit 0  matched    exit 0  matched
```

I had reasoned IN PROSE, one message before the failure, that `--no-index` made
the query about the RULE rather than the PATH. That reasoning was wrong, and it
was a claim about a second environment made from this one.

Sabotage shows why the second test is the load-bearing half:

```
revert the query, directory ABSENT   DETECTED
revert the query, directory PRESENT  MISSED
revert the second test's query       DETECTED
```

The missed row is the finding, not a gap.

**An operational note.** The first apply PASSED its gate at 4941, and the
printed `TO REVERSE:` block was then executed -- restoring three files from
their own backups before the commit ran. Diagnosed from the filesystem rather
than assumed: each target and its backup shared a modification time to the
millisecond, which only occurs when `Copy-Item` restores a backup over its
source, and the console history at lines 104611-104613 held the three reverse
commands. Fifteen minutes of suite time discarded. The reverse block exists for
a FAILED gate.

---

## A capability no consumer invoked (`15830cc`)

`repository_hygiene.resolve_relocation()` existed from the moment the authority
was written, FOR ONE NAMED CASE, and nothing called it. Measured by walking the
retirement tool's call sites:

```
hygiene functions the tool CALLS: ['in_scratch_root']
resolve_relocation is called    : False
```

`scripts/verify_written_cohorts.py.bak` sat UNCLASSIFIED through FOUR sweeps.
Called by hand the function answers immediately, naming
`scripts/forensics/verify_written_cohorts.py` and successor `0b93d302`.

> A capability that no consumer invokes is not a capability.

Reading `collect()` in full -- rather than the branch I needed -- also exposed a
SECOND duplication: five `CLASS_*` strings against `ArtifactClass` in the
authority. Two vocabularies for one concept, reintroduced immediately after the
pattern lists were consolidated. I would not have found it by wiring one
function.

A relocated preimage is RETAINED. Its bytes are not in git -- that is why the
derived path found nothing -- and whether it holds anything its successor lost
is a JUDGEMENT, not a computation. The manifest records the successor and the
evidence so that judgement can be made from the record.

---

## Four full walks, and a vacuous invariant (`b3c5e80`)

`iter_repository_detritus` called `rglob()` once PER PATTERN, and rglob cannot
prune. Measured on the live repository:

```
rglob("*.bak_*")   1.931s     rglob("*") whole tree  2.093s  135,832 entries
rglob("*.bak")     1.889s     os.walk with pruning   0.478s   43,070 files
rglob("*.orig")    1.900s
rglob("*.rej")     1.901s
end to end         7.617s for FIVE reported files
```

92,762 entries -- almost all of `.venv312` -- enumerated FOUR TIMES and
discarded. The rewrite prunes by mutating `dirs` in place. Measured afterwards
on a clean tree: **1.690s**, a 4.5-fold improvement for the FUNCTION. The
sixteenfold figure applies to the WALK alone; that correction is recorded in
`559ca58` beside the record rather than edited into it.

**The rewrite shipped a defect through a passing gate.** `resolve_relocation`
matched a tracked file by BASENAME without checking whether the original still
sat at its derived path:

```
README.md.bak_2026-08-20_065912
    derived original : README.md      EXISTS=True
    resolve_relocation -> README.md   <- claimed a relocation
```

The iterator asked unconditionally, excluded EIGHT ordinary artefacts, and
reported ZERO detritus. A VACUOUS invariant that would have passed forever.

**The suite did not catch it.** 4948 tests passed with the defect installed.
What caught it was the installer's structural gate PRINTING THE FILE LIST, and
reading `0 file(s)` as wrong rather than as success.

The repaired gate compares the iterator's output against the tool's own
classification as PATH LISTS and fails on any divergence. The previous gate
printed and required me to read; this one refuses.

---

## The bootstrap, and a rule for the root that matched every depth (`775d16c`, `be645d1`, `5d3a955`)

The transactional runner cannot install the code it depends on, so a
CONVENTIONAL installer landed the machinery -- and left `.bak_` artefacts,
honestly, as every installer before it has.

> Do not redesign the transaction system while simultaneously relying on that
> unfinished redesign to install itself.

`775d16c` was pushed and continuous integration FAILED. It carried
`tests/unit/test_install_plan.py` but NOT the module that test imports.
`.gitignore:250` read `install_*.py`, written for root-level seq-window
scaffolding, and a pattern without a leading slash matches at EVERY depth.

```
                              install_*.py    /install_*.py
install_root.py               ignored         ignored
scripts/install_runner.py     ignored         NOT ignored
src/pkg/install_plan.py       ignored         NOT ignored
```

**A correction to my own account, and the more important half.** I asserted
TWICE that `git add` fails silently on an ignored path. It does not:

```
git add pkg/ scripts/install_runner.py
    The following paths are ignored by one of your .gitignore files:
    hint: Use -f if you really want to add them.
    exit=1
    staged: pkg/normal.py
```

It warns, exits 1, and stages the rest. The bootstrap's `git add` DID print
that warning. I read the `git status` that followed, counted six entries, and
proceeded past a visible error. **A reading failure, not a tooling gap.**

Then `be645d1` committed the two rescued files but not the `.gitignore` change,
because the installer staged only its two targets and verified only those two.
The same shape as the omission it was repairing: **the presence of what was
expected was checked; the absence of what was needed was not.**

---

## Two refusals that were the design working (`4a3f13d`, `5864f73`)

**RUNNER-COUNTER-SCOPE-1.** The runner measured `tests/unit` and tried to render
a badge from it:

```
tests/       collects 5207     ratchet reads      5207
tests/unit   collects 4982     README badge reads tests-5207-
```

`render_readme` REFUSED rather than produce a plausible-looking counter, and
that refusal is the only reason the mismatch was found. The error rendered
CLEANLY -- nothing in the numbers themselves would have looked wrong.

The scope is not a choice: the ratchet's own header calls itself "the single
source of truth for how many tests exist", `tests/conftest.py` enforces it
against the COLLECTED total of `tests/`, and `test_readme_claims.py:222`
requires the badge to EQUAL it with no tolerance. `COUNTER_SCOPE` and
`ACCEPTANCE_SCOPE` now have two names and neither stands in for the other.

**INVARIANT-SELF-REFERENCE-1.** The first real transactional install gated at
4976 passed / 10 skipped / 1 failed:

```
FAILED test_no_detritus.py::test_no_incomplete_transaction_journals_remain
```

The gate runs INSIDE the apply transaction -- deliberately, so a failure still
has a rollback -- so that transaction's journal is legitimately non-terminal
while the test looks at it.

> "No incomplete journals" is a QUIESCENT-REPOSITORY property. Asserting it
> during an install asks whether a thing is finished while it is happening.

**And the failure proved the architecture.** Measured immediately afterwards:
detritus NONE, incomplete journals 0, journal root EMPTY, `git status` clean.
A transactional install that FAILED left the repository byte-identical.

The fix is narrow by construction: the runner exports `GVC_ACTIVE_TRANSACTION`
into the GATE'S CHILD ENVIRONMENT ONLY, and the test excludes EXACTLY ONE named
identifier. A second test asserts the exclusion stays narrow.

---

## The first transaction-installed payload (`f125187`)

```
the ratchet (5207) agrees with tests
measurement transaction 5207 -> 5213, then ROLLED BACK
plan digest 374d205830601b78, validated against the pristine tree
    create tests/unit/test_no_detritus.py
    patch  tests/EXPECTED_SUITE_SIZE
    patch  README.md
apply transaction 8170f9fe4332
write set  : proven equal to the declared set
gate: 4978 passed, 10 skipped, 0 failed
prospective hygiene: no detritus
committed; journal destroyed: True
attested hygiene: no detritus
no journals remain
```

Three changed files. Continuous integration green on both interpreters at
`5199 passed, 13 skipped, 1 xfailed` -- summing to the ratchet exactly.

The payload's own content is the invariant forbidding the residue every
previous installer left behind.

---

## Defects found across the two days

| identifier | what |
|---|---|
| SCRATCH-IGNORE-ENVIRONMENT-1 | a directory rule queried by bare name; passed locally, failed on the runner |
| RELOCATION-UNWIRED-1 | a function written for one case that no consumer ever called |
| the second vocabulary duplication | five `CLASS_*` strings against `ArtifactClass` |
| DETRITUS-WALK-COST-1 | four full tree walks where one pruned walk suffices |
| RELOCATION-FALSE-POSITIVE-1 | a relocation claimed for an original that never moved |
| RELOCATION-SELF-MATCH-1 | a committed backup reported as relocated to ITSELF |
| JOURNAL-CONSTRUCTION-RESIDUE-1 | a refused construction left an undiscoverable journal directory |
| the plan built after mutation | the payload's action read `patch` for a file that was CREATED |
| `validate_against` never called | written, tested, and unwired -- in code an hour old |
| GITIGNORE-UNANCHORED-INSTALL-1 | a root rule matching at every depth; two source files never committed |
| the unread `git add` warning | printed, exit 1, and read past -- twice asserted to be silent |
| RUNNER-COUNTER-SCOPE-1 | the runner measured a scope the ratchet does not describe |
| INVARIANT-SELF-REFERENCE-1 | an invariant asserting a property false during its own installation |

---

## What these days taught

**A fixture that contains only the thing being detected cannot show that
anything else is rejected.** Removing the shape filter from the detritus
iterator left all 100 tests passing, because every fixture held only
backup-shaped files. The same shape recurred three times.

**Reaching a guard can take more attempts than writing it.** The basename
self-match guard needed FOUR fixtures. Three could not reach the branch at all:
the backup was committed so the blob branch returned first; the stripped base
shared no tracked basename; the derived original existed so the unmoved guard
returned first. Tracing the function step by step found the reachable case;
reasoning about it did not.

**A defect can pass a 4,948-test suite and a structural gate.** What caught the
vacuous invariant was reading a printed file list. Gates that print require a
reader; gates that compare refuse on their own.

**The recurring failure is mine and has one shape.** I check for the presence of
what I intend and not the absence of what I need. It produced the missing
`.gitignore` in `be645d1`, the unread `git add` warning, the "no CHANGELOG.md"
conclusion drawn from searching only the repository root, and the claim of
twenty-one commits where `git rev-list --count` says thirteen.

**Corrections belong beside records, not inside them.** The 4.5-fold speedup
correction sits in `559ca58`, adjacent to the `b3c5e80` commit whose figure it
narrows.

---

## Where INSTALLER-TRANSACTION-1 stands

Steps 1, 2, 3, 3B, 4 and 5 of 8 are complete.

Remaining: convert the installer generator so new installers are transactional
by construction; make the hygiene invariant mandatory in continuous
integration; migrate the installers that remain supported capabilities, by
census rather than blindly; and remove the legacy backup-writing machinery with
its redundant ignore rules only after a caller census proves nothing depends on
them.

The `.gitignore` consolidation is still owed: the census found EIGHT backup
rules including THREE copies of `*.bak_*`, so removing one changes nothing.
