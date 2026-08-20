# SESSION 2026-08-19 -- rollback state leaves the repository

**Author: Monzia Moodie**

**Commits:** `9b072c2` through `41372ad` -- ten, all on 2026-08-19
**Ratchet:** 5023 -> 5120
**Preceding head:** `f89ce6b`

---

## What this day was

One question, pursued until it stopped generating defects: **where does an
installer's rollback state live?**

The answer that had been in force -- `<target>.bak_<timestamp>`, beside the
file, forever -- was asked to serve four incompatible purposes at once: crash
recovery, undo, historical archive and security evidence. It served none of
them reliably, and measuring that produced six further defects, every one of
them in an instrument rather than in the thing measured.

| commit | time | unit | files | lines |
|---|---|---|---|---|
| `9b072c2` | 11:21 | docs: session record for 08-17 to 08-19 | 2 | +382 |
| `a18ff26` | 12:38 | GITATTRIBUTES-UNGATED-1 | 3 | +282 -2 |
| `320e9cf` | 13:17 | docs: thirty-seven, not thirty-one | 2 | +71 |
| `5447362` | 15:17 | INSTALLER-TRANSACTION-1 step 1 | 2 | +2369 |
| `05f1a72` | 17:02 | INSTALLER-TRANSACTION-1 step 2 | 5 | +592 -5 |
| `be033e7` | 17:44 | INSTALLER-MANIFEST-OVERWRITE-1 | 1 | +32 |
| `ab36352` | 18:33 | docs: the paths package undercounted | 3 | +187 -2 |
| `9cba87f` | 19:06 | RETIREMENT-PATTERN-INCOMPLETE-1 | 2 | +1687 -7 |
| `06e75fe` | 20:12 | INSTALLER-TRANSACTION-1 step 3 | 5 | +1004 -2 |
| `41372ad` | 20:38 | docs: retirement manifest | 1 | +215 |

All figures quoted from `git show --stat`; all times from
`git log --date=format-local`.

Ratchet, quoted from `tests/EXPECTED_SUITE_SIZE`:

```
# 2026-08-19 -- 5023 -> 5027 (+4). OUTPUT-ROOT-CONFLATION-1.
# 2026-08-19 -- 5027 -> 5066 (+39). GITATTRIBUTES-UNGATED-1.
# 2026-08-19 -- 5066 -> 5082 (+16). INSTALLER-TRANSACTION-1 step 2: cache_root.
# 2026-08-19 -- 5082 -> 5120 (+38). INSTALLER-TRANSACTION-1 step 3.
```

The first of those four belongs to `f89ce6b`, which the previous document
covers.

---

## 1. `a18ff26` -- a rule file with no gate is a convention

**GITATTRIBUTES-UNGATED-1.** `.gitattributes` carried 37 rule lines and a
documented near-corruption of a test fixture, and NO test asserted any of them.
Delete `*.py text eol=lf` and nothing failed.

The binary rules matter most. The file records why, from 2026-07-12:

> tests/fixtures/alphafold/AF-E7ENB7-F1-model_v4.cif was committed as a
> 99,647-byte blob while the working copy was 101,171 bytes -- exactly 1,524
> carriage returns stripped, one per line.

Benign for an mmCIF parser. But the file states the consequence plainly: *had
this fixture been genuinely binary (a parquet, an .npy), normalization would
have SILENTLY CORRUPTED it rather than merely shortening it.* A `.npy` whose
bytes git has rewritten does not fail loudly. It loads, and the numbers are
wrong.

**Every assertion goes through `git check-attr`**, so git answers for itself
rather than the tests reimplementing its pattern semantics -- precedence, `**`
matching, later rules overriding earlier. check-attr also answers for paths
that do not exist, which is the point: MEASURED, zero tracked `.npy`, `.gz` or
`.sqlite` files, and the rules still resolve, so the guard covers the NEXT one
added.

The invariant is the INDEX, not the working tree. 124 of 981 tracked Python
files are CRLF here and that is correct under `core.autocrlf=true`. MEASURED: 0
with CRLF in the committed blob, and the guarded state is reachable -- with
`* -text`, a CRLF file commits as `i/crlf`.

### A weakness the pre-install probe found

Run from a temporary directory before any digest was pinned, an earlier draft
had 37 of 38 cases fail loudly on the wrong repository root -- correct -- while
**one passed**, because `git ls-files` had inherited the shell's working
directory. A test that passes wherever a clean repository happens to be current
is not testing THIS repository. `git` now runs with `-C <repo>`.

### An honest negative result

The `tests/fixtures/**` overrides are REDUNDANT while the general
`*.parquet binary` rules exist -- measured by building two repositories
differing only in those lines and comparing git's answers, which were
byte-identical. Sabotage confirmed it: deleting an override changes nothing.
They stay as defence-in-depth, and the test says so rather than pretending to
guard them.

---

## 2. `320e9cf` -- thirty-seven, not thirty-one

Two committed records stated `.gitattributes` carries **31 rules**. It carries
**37**. MEASURED three ways, all agreeing: 74 total lines, 37 non-blank
non-comment lines, 37 DISTINCT patterns -- no line shares a pattern, so no
counting method yields 31.

The figure came from reading a truncated terminal display rather than
enumerating, and I stated it twice before measuring.

**Superseded, not rewritten.** The changelog is newest-first, so the correction
sits above the claim; the session document gets an append-only addendum.
Everything else in those sentences stands.

And the installer **re-counted the rules itself** before writing, refusing
unless the answer was 37 rule lines and 37 distinct patterns. Publishing a
corrected figure on a measurement taken twenty minutes earlier is how the wrong
one got in.

---

## 3. `5447362` -- 148 rollback artefacts retired, one manifest kept

**INSTALLER-TRANSACTION-1 step 1.**

MEASURED: 148 `.bak_<timestamp>` files, 17,640,928 bytes, spanning 2026-08-10
to 2026-08-19. `README.md` alone had 31 and `tests/EXPECTED_SUITE_SIZE` 29 --
one per installer run. Invisible to `git status` because `.gitignore` carries
`*.bak_*`.

I found them by enumerating, having first said "six" from a listing of `docs/`.

### Three classifications, and why the distinction mattered

```
git_exact_preimage               139
superseded_uncommitted_preimage    8
secret_bearing                     1
unclassified                       0
```

The middle class is why *"the original is tracked"* was NOT sufficient grounds
for deletion. A tracked original says git has SOME version; it does not say git
has THESE bytes. `git hash-object` against every historical blob for that path
says the stronger thing.

### The secret-bearing artefact

`.env.bak_2026-08-15_205854`, 1,041 bytes, recorded by digest and structure
only: 22 total lines, 9 assignment lines, 2 bare non-comment lines, 8 distinct
names -- so `GITHUB_TOKEN` appeared TWICE. That is the corrupted state exactly,
and the mechanism of the credential incident is readable from structure alone.

Not recorded: the token, any identifying prefix, or any line content. Verified
before deletion.

> incident evidence != secret retention

### A sabotage result an absence-of-leak check could not find

Emptying `SECRET_PATTERNS` produced NO leak -- the shape reader never runs, so
nothing enters the manifest. But the artefact would be classified as ordinary
and deleted as routine detritus, with the secret-handling decision never
surfaced. A canary guard now refuses to run at all unless the classifier
recognises seven known shapes.

---

## 4. `05f1a72` -- a fifth path domain

**INSTALLER-TRANSACTION-1 step 2.** `state_root` defaults to
`<project>/.gvc-state` -- correct for agent state belonging to THIS checkout. A
transaction journal does not: it must survive an interrupted installer even if
the working tree is reset.

```
repository identity  -> project_root
artifact identity    -> artifact_root
checkout state       -> state_root
machine-scoped cache -> cache_root      <- new
```

Default: `LOCALAPPDATA` on Windows, `XDG_STATE_HOME` on POSIX, then
`home/.local/state` -- the fallback that ALWAYS resolves. MEASURED: with `HOME`
unset on Windows, `Path.home()` still returned `C:/Users/monzi` via
`USERPROFILE`.

### What cannot be tested from Windows

Passing a fake environment with `XDG_STATE_HOME="/home/runner/.local/state"`
selects the right BRANCH but produces `C:/home/runner/...`. **Path flavour is
baked into the platform, not the environment.** So the tests assert
RELATIONSHIPS -- outside the repository, beneath the chosen base, absolute,
project-named -- which hold on both platforms, and the literal POSIX form is
verified when the runner executes them.

Purely additive: ONE keyword-only construction site, `describe()` asserting
membership rather than equality, and NO test constructing `RuntimePaths`
directly. So no existing test was touched, and the gate confirmed it.

---

## 5. `be033e7` -- a manifest is evidence, not a scratch file

**INSTALLER-MANIFEST-OVERWRITE-1.** `--manifest` defaulted to a fixed path and
the script wrote it unconditionally.

A routine three-artefact cleanup overwrote the 148-artefact record:

```
docs/incidents/BACKUP_RETIREMENT_2026-08-19.json | 1976 +----------------
1 file changed, 20 insertions(+), 1956 deletions(-)
```

Recovered with `git checkout 5447362 -- <path>`. It survived only because it had
been committed minutes earlier.

**I predicted this one message too late** -- the overwrite risk was described in
the same message that issued the command causing it. Worse, the verification
line read *"no diff = the original record survives"* and printed unconditionally
beneath a diff of 1,976 changed lines. **Third unconditional label of the
session.**

The guard is refusal, not versioning: a target recording a DIFFERENT scan aborts
before deleting anything. Verified against the real loss, not only a fixture.

---

## 6. `ab36352` -- the paths package undercounted its own domains

The package docstring opened with *"one authority, three roots"* after
`cache_root` had landed at `05f1a72`. I added the field and did not re-derive
the enumeration two directories away.

**Corrected in place, not superseded.** `REQUIRED_PROVENANCE_CORRECTION` governs
RECORDS -- claims about what was believed at a past moment. A module docstring
is a LIVE DESCRIPTION of current structure, and one that describes the module
wrongly is simply wrong.

Verified by RUNTIME correspondence rather than by reading two files:
`dataclasses.fields(RuntimePaths)` returns exactly the four names the docstring
enumerates.

### An instrument defect, recorded because it did NOT bite

The targeted suite reported 40 where I expected 39. My AST case-counter treats a
`@parametrize` over a NAME as one case, and `test_runtime_paths.py` parametrizes
over `PROJECT_SENTINELS`. The file collects 24, not the 23 I had recorded.

No pinned figure was affected: every installer arithmetic came from pytest's
`tests/unit` collection, never from the per-file estimate. **The unreliable
instrument was never load-bearing, because the reliable one was always the
authority for anything pinned.**

---

## 7. `9cba87f` -- the sweep saw one shape of four

**RETIREMENT-PATTERN-INCOMPLETE-1.** The retirement tool scanned `*.bak_*`
ALONE. At `5447362` it retired 148 artefacts, reported *"remaining .bak_*
artefact(s): 0"*, and I wrote into that commit message that the repository held
zero backup artefacts.

MEASURED: **107 more** were sitting beside them, 2,828,345 bytes.

```
foo.py.bak_2026-08-19_164056      the PowerShell installers
foo.py.pre_cfgroot.bak            the Python appliers   <- NEVER SWEPT
foo.py.precosmic.bak              older appliers
foo.py.20260702_183508.bak        a dated convention
foo.py.bak                        bare
```

Every `scripts/apply_*.py` writes `.pre_<name>.bak`. A second accumulation ran
in parallel to the one I cleared, invisible to the tool built to clear it, for
the entire session.

> A filter that reports zero is not evidence of zero.

Extending the patterns alone would have classified everything as unresolvable,
because the original was derived by splitting on `.bak_`. `_derive_original` now
generates candidates and lets the FILESYSTEM decide; when none exists the backup
stays UNCLASSIFIED rather than guessed at.

And a classification-ordering defect found by testing: a `.env.pre_token.bak`
whose live `.env` had been removed fell through to `unclassified` -- safe, but
the manifest then held NO shape metadata for a credential-bearing artefact. The
secret check now runs FIRST.

### What the sweep found, and why 13 survive

```
git_exact_preimage               65
superseded_uncommitted_preimage  29
unclassified                     13
```

The 29 meet the standard the first sweep's 8 met: 13 distinct originals, EVERY
one tracked, ZERO without a committed successor.

The 13 retained are retained for two different reasons: twelve in
`.af_fix_work/`, ignored at `.gitignore:198` with zero tracked files -- a
deliberate scratch area; and one, `scripts/verify_written_cohorts.py.bak`, whose
original does not exist and has NO git history at all.

### A census defect of my own, corrected by the tool

An earlier probe printed `af_fix_work/`, `github/workflows/` and `gitignore.`
because it used `as_posix().lstrip("./")` -- and **`lstrip` takes a CHARACTER
SET, not a prefix**, so it stripped the leading dot from every hidden path.

---

## 8. `06e75fe` -- the transaction primitive

**INSTALLER-TRANSACTION-1 step 3.**

```
success       the repository keeps the changes and NOTHING else
failure       the repository is byte-identical to how it was found
interruption  a journal survives OUTSIDE the repository in a non-terminal
              state, discoverable by incomplete_transactions()
```

A journal inside the repository is REFUSED at construction. The defect that
produced 255 artefacts is structurally impossible rather than remembered.

Secret targets get NO on-disk preimage: bytes in process memory only, a manifest
carrying digest and structure but never content, scrubbed on commit.

Persistence reuses `JsonStateStore`, VERIFIED against the real module rather
than a stub -- including that `StateSchemaMismatch` descends from
`StateStoreError`, so an unenveloped journal is REPORTED rather than
propagated. My fixture stub declared the inheritance I assumed; only the real
module could settle it.

### The first sabotage run found three defects in my TESTS

**H9 passed for the wrong reason.** With containment removed, `../escape.py` did
not exist, so `patch()` raised *"does not exist"* -- same exception TYPE,
different cause.

**H8 hid a real behavioural difference.** Two digest checks exist; removing the
pre-write one let the post-write one raise, so the test passed WHILE the
corrupted bytes had already reached the target.

**H6 revealed the state table was never consulted.** Every state test was
satisfied by a method's own guard, so `_ALLOWED` had no test at all and
permitting every transition passed all 27 cases.

> A transition table with no test is a comment.

Two further mutations of my own then found two more gaps: the post-write digest,
and the canary guard's CALL SITE.

**Final: 38 tests, 12 of 12 mutations detected.**

---

## Corrections owed to earlier records

These are recorded here rather than edited into the commits they concern.

| record | what it says | what is true |
|---|---|---|
| `5447362` | the repository held zero backup artefacts | 107 remained, in a shape the filter never scanned |
| `be033e7` | the refusal was verified against the real manifest | asserted in the message; measured only afterwards |
| `9cba87f` | four sabotage mutations, two undetected | two were undetectable BY EXIT CODE, detectable by manifest inspection -- a sharper statement |

The third is worth keeping as a method note: **a sabotage matrix judging solely
on exit status will under-report.** Two of that unit's mutations changed only
what the manifest recorded.

---

## Register at close

| item | state |
|---|---|
| `GITATTRIBUTES-UNGATED-1` | CLOSED at `a18ff26` |
| `INSTALLER-MANIFEST-OVERWRITE-1` | CLOSED at `be033e7` |
| `RETIREMENT-PATTERN-INCOMPLETE-1` | CLOSED at `9cba87f` |
| `INSTALLER-TRANSACTION-1` | OPEN. Steps 1-3 of 8 complete. Remaining: one installer converted to the primitive; the crash matrix run against a CONVERTED installer; a repository-wide "no detritus" test; a migration sweep; and finally removing `*.bak_*` from `.gitignore` so recurrence becomes VISIBLE. |
| `PATHS-BY-INJECTION-1` | OPEN, Stage B of the path work |
| `CONFIG-DEAD-PATHS-1` | OPEN, a scope decision: 35 unreachable of 71; 7 stale, 28 roadmap |
| `ATOMIC-WRITE-DUPLICATION-1` | NEW. `representation_artifact.py` documents its copy of the idiom from `RunArtifactWriter._atomic_write`. Deliberate and documented, but two copies; the transaction primitive deliberately did not add a third. Consolidation is its own unit. |
| `WORKTREE-EOL-DRIFT-1` | OPEN, and NOT a defect |
| `ROOTFIX-VERIFY-TEXTUAL-1` | OPEN |
| `SHAREDSTATE-LOAD-WRITES-1` | OPEN |
| `PACKAGES-NO-INIT-1` | OPEN |
| `MIGRATION-RECORD-SEPARATOR-1` | OPEN |
| `CHANGELOG-DUP-2026-06-25` | OPEN |
| `LGBM-SKLEARN-FEATURE-NAME-WARNING-1` | OPEN, non-blocking, upstream |
| `PREFLIGHT-CREDENTIAL-USABILITY-1` | OPEN, a refinement |
| SESSION_2026-06-19 item 5 | OPEN |

---

## The method, and where it failed

**Six defects today, every one in an instrument rather than in the thing
measured:**

- a manifest addressed by a name the next event reused, which destroyed the
  record it existed to preserve;
- a filter covering one shape of four, which reported zero and meant nothing of
  the kind;
- a classification ordered so a credential file could fall through to
  `unclassified`;
- a census using `lstrip("./")` as though it were a prefix;
- three tests that passed for the wrong reason, or hid a real behavioural
  difference, or never exercised the table they were meant to cover;
- a case-counter blind to `@parametrize` over a name.

Each was found by RUNNING the tool against reality -- a real repository, the
real module rather than a stub, a deliberately broken variant. **None was found
by reading.**

The three that matter as standing rules:

> **A filter that reports zero is not evidence of zero.** It is evidence about
> the filter.

> **A test that passes for the wrong reason is worse than a missing test**,
> because it consumes the attention a missing test would attract.

> **A stub agrees with you.** Only the real module can contradict you.

And the label defect recurred a third time: a prose line stating a conclusion
regardless of what the command printed. It has now appeared beneath a sentinel
listing, a `docs/sessions` sweep, and a 1,976-line diff. A line that cannot fail
is not a check.
