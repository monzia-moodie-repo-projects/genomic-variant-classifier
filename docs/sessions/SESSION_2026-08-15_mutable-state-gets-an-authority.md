# SESSION 2026-08-15 -- mutable state gets an authority

**Author: Monzia Moodie**

**Commits:** `48907ec`, `a734ea1`, `c1fb110` (three)
**Ratchet:** 4910 -> 4964
**Preceding head:** `cc848b4`

---

## What this session was

Three units completing the runtime-path programme begun on 2026-08-14. That
day's `RUNTIME-PATHS-1` (`69a9597`) established WHERE things live; this session
establishes what a mutable store IS, makes the one live agent adopt it, and
reconciles the two divergent copies its absence had produced.

| commit | unit | files | lines |
|---|---|---|---|
| `48907ec` | STATE-STORE-1 | 5 | +621 -2 |
| `a734ea1` | LITERATURE-STATE-CWD-RELATIVE-1 | 7 | +593 -24 |
| `c1fb110` | STATE-FILE-DUPLICATES-1 | 5 | +587 -2 |

All figures quoted from `git show --stat`, not reconstructed.

Ratchet, quoted from `tests/EXPECTED_SUITE_SIZE`:

```
# 2026-08-15 -- 4910 -> 4932 (+22). STATE-STORE-1.
# 2026-08-15 -- 4932 -> 4949 (+17). LITERATURE-STATE-CWD-RELATIVE-1.
# 2026-08-15 -- 4949 -> 4964 (+15). STATE-FILE-DUPLICATES-1.
```

---

## 1. `48907ec` -- a store that is atomic, identified, and fails closed

**STATE-STORE-1.**

Two mutable JSON stores existed and behaved differently. `SharedState` writes
atomically and logs corruption while leaving the damaged file alone.
`version_monitor_agent.py:58-85` did neither:

```
_STATE_PATH = Path("data/agent_state.json")     # cwd-relative
except (json.JSONDecodeError, OSError):
    return {}                                    # corruption -> empty
_STATE_PATH.write_text(...)                      # non-atomic
```

**Those compound.** A crash mid-write truncates the file; truncation reads as
an empty store; the next `_set_many` persists that emptiness as the new truth.
This agent's entire purpose is detecting when upstream sources change, and its
ClinVar header hashes and AlphaMissense entity tags are exactly what that
sequence destroys.

### The mechanism was copied deliberately, not reinvented

`SharedState.save` was READ BEFORE THE STORE WAS DESIGNED and is correct: a
temporary file in the SAME directory so `os.replace` is an atomic rename,
cleanup on failure, and a re-raise so the caller sees it. Reproducing it
identically -- with `fsync` added -- avoids putting a second, subtly different
atomic write beside a working one. That is the parallel-vocabulary defect this
project keeps eliminating, and the intent is that `SharedState` later adopts
this module rather than the reverse.

### Three deliberate differences

**fsync before rename.** `SharedState` omits it, so `os.replace` is atomic
against other PROCESSES while the bytes may still sit in the operating system
cache at power loss.

**Corruption RAISES.** A store answering "empty" when it means "damaged" is the
same shape as a parser reporting a 310-kilobyte lock file as zero packages.

**Schema identity in the payload.** Two files named `agent_state.json` held
UNRELATED schemas -- a flat literature-scout key-value log and the
orchestrator's structured state. Reading the wrong one previously SUCCEEDED and
returned a dictionary that meant something else.

### Legacy payloads are readable, and said so

MEASURED before the store was written: `data/agent_state.json` is a BARE flat
dict of 25 keys with no envelope. A store that refused unenveloped payloads
could not read the data it exists to migrate. So `load()` accepts both and
reports which, and the caller decides.

### A sabotage mutation that stayed missed, and why that is correct

Replacing `values=dict(values)` with `values=values` went undetected. MEASURED:
`json.loads` builds a fresh object per call, so every load is already
independent and the copy defends an aliasing path that does not exist. The
mutation is INERT.

But my test was still wrong -- it mutated the returned mapping and re-read the
FILE, which passes either way. Replaced with two properties that are actually
true, one noted as holding by construction so a future caching layer breaks it
visibly.

**22 tests, 14 of 15 mutations detected, the fifteenth measured inert.**

---

## 2. `a734ea1` -- the agent adopts the store

**LITERATURE-STATE-CWD-RELATIVE-1.**

Six definitions at lines 58-85 become a store anchored to `RuntimePaths`, with
`_get` and `_set_many` kept as thin delegates so the three call sites at 156,
202 and 496 are untouched.

`_set` is DROPPED -- defined at 77-80, called NOWHERE, verified by an
abstract-syntax-tree call census across `src`, `scripts` and `tests`.

### The first attempt failed, and the gate caught it

A four-edit version was installed and the suite reported **4713 passed / 10
skipped / 1 FAILED**. The installer rolled back all five files, removed the two
created ones, cleaned three backups, and `git status` came back empty.

```
NameError: name '_STATE_PATH' is not defined
version_monitor_agent.py:532
    logger.info("LiteratureScoutAgent: state written to %s", _STATE_PATH)
```

`run()` LOGS the constant the block edit deletes. I had read that line and
quoted it in an earlier exchange -- and recorded it as "a log message" rather
than as a reference to a name I was about to remove.

> **Confirming the DEFINITIONS are gone says nothing about whether anything
> still LOADS them.**

The applier now carries a fifth edit for that line and a verifier rule
refusing any remaining `ast.Name` load of `_STATE_PATH`, `_load_state`,
`_save_state` or `_set`. Proven in both directions: the four-edit version is
REFUSED before writing and leaves the agent unchanged; the five-edit version
applies with no dangling references.

That is the seventh time in this three-day session I sized something from what
I had rather than what the code contains, and the first that only a runtime
failure could surface.

### The state boundary, in both directions

`/.gvc-state/` added to `.gitignore`, ANCHORED. MEASURED: `git check-ignore`
returned NOTHING for `.gvc-state/literature_scout/state.json`, so the first
real agent run would have left mutable state untracked in `git status`, where
someone eventually commits it. **That is REPORTS-DIR-IGNORED-1 inverted**, and
the leading slash keeps a NESTED `.gvc-state` under `src/` VISIBLE.

### Why new tests were required

Both pre-existing test files stub `_run_watch_targets` AND pass
`dry_run=True`. Line 495 reads `if not dry_run: _set_many(...)`, so NEITHER
ever reaches the store -- replacing it would have been invisible to the whole
suite. The new file drives it through an INJECTED store.

**16 tests, 9 of 9 mutations detected.**

---

## 3. `c1fb110` -- a migration record, not a file copy

**STATE-FILE-DUPLICATES-1.**

> A copy leaves no answer to "why does this store's history jump from
> 2026-06-13 to 2026-06-20?"

The record answers it, with the digest of every file involved, the key-set
comparison that justified the choice, and the reasoning stated in the document
rather than in a commit message someone must go looking for.

```
data/agent_state.json                       13,463 bytes  2026-06-13 15:14
    sha256 e28c673ba7a93ed7856755ef6bf9cd84b4ceaac95500fba7e350e9e8438479cb
src/.../agent_layer/data/agent_state.json   14,524 bytes  2026-06-19 22:30
    sha256 22fe38e94ce3bc8fd349e1fd4a6fbff51e4e6ed5c217503dfee095a8fe339e16
```

Same key set, no key unique to either, FIVE values differing, every one the
nested copy being a LATER observation: `last_run` advances,
`deps_outdated_count` rises 99 to 110, borb's tracked release moves 3.0.7 to
3.0.8.

### Re-measuring mattered

"The nested copy supersedes" is a claim about a MOMENT, and the adoption at
`a734ea1` changed where the agent writes. The canonical store was still absent,
so `version_monitor` had not run since -- verified, not assumed.

### Two clocks

The nested file's modification time is 2026-06-19 22:30 while the `last_run` it
holds is 2026-06-20T02:30:41 UTC -- four hours apart, Eastern Daylight Time's
offset. The same instant on two clocks. Both values are UTC-suffixed ISO 8601,
so comparing them as strings is sound.

### What the migration refuses

Not "the bigger file", not "the newer modification time". It requires IDENTICAL
KEY SETS -- a newer copy that has LOST a key is not a superset, and merging it
would discard a change-detection baseline -- and a strictly later ordering
value. It pins both source digests: if either changed since the comparison it
refuses, because a reconciliation justified by different bytes is not
justified.

### It deletes nothing

Both legacy files are RETAINED, named in the record, and verified unchanged
after the run. They are the only surviving evidence of what the cwd-relative
path produced, and this session proposed destroying supposedly-redundant files
three times and was wrong three times.

### Verified after the run

```
destination values == nested source : True
key count                          : 25 (nested 25)
schema / version / generation      : gvc.literature-scout-state / 1 / 1
both legacy digests                : unchanged
.gvc-state/... ignored             : .gitignore:103:/.gvc-state/
docs/migrations/... ignored        : NO -- evidence is visible
```

### Two defects in my own tests, both found by running them

I deleted the ordering key from one copy and asserted the ORDERING refusal --
but removing a key changes the KEY SETS, so the superset guard fires first. The
message told me.

And a sabotage giving `legacy_files_retained` a default of `()` went
UNDETECTED, because every test passes it explicitly. **A record whose fields can
be OMITTED can be built claiming no legacy files were retained when two were,
and the omission would look like a fact.** No field carries a default now.

**15 tests, 8 of 8 mutations detected.**

---

## Register at close

**Closed:** `STATE-STORE-1` established; `LITERATURE-STATE-CWD-RELATIVE-1` and
`STATE-FILE-DUPLICATES-1` closed.

| item | state |
|---|---|
| `MIGRATION-RECORD-SEPARATOR-1` | NEW, cosmetic. `destination_path` uses Windows separators while every other path uses forward slashes; `Path.relative_to()` returns platform-native. The record is immutable evidence and is NOT rewritten; the script should normalise for future runs. |
| `PROJECT-ROOT-HARDCODED-1` | `config.py:17` still holds the literal |
| `CONFIG-DEAD-PATHS-1` | two constants, four environment variables, zero readers |
| `OUTPUT-ROOT-CONFLATION-1` | addressable via `artifact_root` |
| `ROOTFIX-VERIFY-TEXTUAL-1` | correction now demonstrated in `apply_literature_state_adoption.py` |
| `SHAREDSTATE-LOAD-WRITES-1` | `_migrate` calls `save()` from inside `load()` |
| `PACKAGES-NO-INIT-1` | `monitoring/` and `training/` are namespace packages |
| `PREFLIGHT-TOKEN-SUBSTRING-1` | check 9 tests a substring; **currently FAILING** -- the placeholder was removed and no token replaced it, so any cloud run is gated |
| `CHANGELOG-DUP-2026-06-25` | a 26-line double-paste at lines 6546 and 6573 |
| SESSION_2026-06-19 item 5 | `run_agents.py` still has no chdir |

---

## The method, across three days

The recurring failure had one shape: **sizing a defect from the set already in
hand rather than the set it could inhabit.** One em-dash file when there were
two, one ignored directory when there were three, four agents when there were
five, four edits when there were five.

The paired failure was assertions that could not fail -- `if "root" not in
source` satisfied by `outputs_root`, a no-developer-path check blind to forward
slashes, a discovery test true whether it walked from `__file__` or `"."`, and
a defaulted field no test could notice. Sabotage found every one, which is why
each gate gets a mutation matrix.

And the gates worked. One unit failed on one test out of 4,724 and rolled back
five files cleanly. The cost was a twelve-minute cycle; the alternative was a
partially adopted module.


---

## Addendum -- later on 2026-08-15

This document was written after `c1fb110` and was accurate then. Two further
commits landed the same day, and one of them changed a state this record
describes as current. Recorded as an addendum rather than a rewrite: the body
above says what was true when it was written, and this says what changed.

| commit | unit | files | lines |
|---|---|---|---|
| `e08da7f` | docs: session record for 2026-08-15 | 2 | +306 |
| `a8cc484` | PREFLIGHT-TOKEN-SUBSTRING-1 | 5 | +590 -6 |

Ratchet 4964 -> 4997. Continuous integration on `a8cc484`: 4983 passed, 13
skipped, 1 xfailed on both Python 3.11 and 3.12, all 33 new cases passing on
Linux.

### PREFLIGHT-TOKEN-SUBSTRING-1 is CLOSED

The register above records it as **currently FAILING**, with cloud runs gated.
That was true when written. `a8cc484` closed it, and check 9 now answers:

```
check 9: True  (.env (length: 40))
```

A measured length, not the presence of a substring, and the value is never
emitted.

The defect was `preflight_check.py:259`, `if "GITHUB_TOKEN=" in content` -- a
substring search over the whole file that accepted a commented line, an empty
value, a placeholder, and a name merely ENDING with GITHUB_TOKEN. It reported a
usable credential twice on 2026-08-15 when `.env` held only placeholder text.

The floor is thirty, derived from two measured lengths: 22, the fragments
PowerShell `Add-Content` produced when it split a pasted token across two
lines; and 40, the real token. I first asserted a floor of 20, which does not
reject 22 -- an arithmetic error against a number measured minutes earlier.
Both wrong floors are sabotage cases.

### A PRECISION CORRECTION TO THAT UNIT'S CLAIMS

The unit's documentation states that every current GitHub credential format is
at least 40 characters and enumerates exact lengths for several. Those are
EXTERNALLY VERSIONED FACTS about another organisation's product, not properties
of this repository, and they should not be encoded as permanent truths.

What the check actually needs is narrower: reject obvious placeholders and
fragments, never expose the secret, and prefer usability over plausible syntax.

And a stronger distinction the unit does not yet draw:

    a well-formed candidate  IS NOT  a usable credential

A token can be forty characters and revoked, expired, or scope-deficient.
`_MIN_TOKEN_LENGTH = 30` is a syntactic sanity check and should be described as
one. The stronger contract is an authenticated probe -- a minimal GitHub
application programming interface request, with no token output -- reported
separately:

    credential configured   : yes
    credential authenticated: yes / no / not checked

That remains open as a refinement, not a defect.

### LGBM-SKLEARN-FEATURE-NAME-WARNING-1, and a renaming

An investigation into a `UserWarning` in the suite concluded on 2026-08-16.

    X does not have valid feature names, but LGBMClassifier was fitted with
    feature names

I first filed this as `ENSEMBLE-FEATURE-NAMES-1`, a name that assigns ownership
to the ensemble and implies it violated a schema contract. **It did not.**
Renamed to `LGBM-SKLEARN-FEATURE-NAME-WARNING-1`.

Established by a library-only reproduction with all project code removed:

```
lightgbm 4.6.0, scikit-learn 1.8.0
ARRAY only       feature-name warnings: 3   (three folds, three warnings)
DATAFRAME only   feature-name warnings: 0
```

LightGBM 4.6.0 synthesises feature names when fitted on unnamed array input;
scikit-learn 1.8.0 then observes estimator feature-name metadata against
unnamed prediction input and warns. `variant_ensemble.py` is internally
consistent -- line 2941's `.values` reaches both fit and predict, and no
`.fit(` appears anywhere between the start of `VariantEnsemble.fit` at 2782 and
the out-of-fold loop at 2927.

**Non-blocking, no correctness risk.** Column order is preserved; the same
array serves fit and predict. It contributes one of the 33 local warnings and
one of the 34 in continuous integration.

The three test failures seen during the investigation were an artefact of
running with `-W error`, which promotes the warning to an exception inside
`cross_val_predict`. The fail-loud guard at `variant_ensemble.py:2963` then
correctly refuses to let LightGBM vanish from the ensemble. **That guard is
vindicated by this investigation, not implicated by it.**

The eventual repair is NOT suppression. It is a model-specific dispatch giving
LightGBM DataFrames deliberately, mirrored in `_leakfree_oof` and
`predict_proba`, so the model carries real genomic feature identities rather
than synthetic `Column_0` names -- valuable for importance attribution,
checkpoint inspection and schema audits independent of the warning. The test
must assert the CONTRACT, not the symptom: that the estimator receives a
DataFrame whose columns equal the expected feature list, at fit and at predict.
Suppression makes a diagnostic disappear; DataFrame input makes its cause
disappear.

### THE METHODOLOGICAL FINDING, WHICH OUTLASTS THE WARNING

Locating that warning took eight probes. Seven were filtered searches, and each
filter encoded the hypothesis it was being used to test:

    predict_proba|.values|to_numpy     found `.values` at 2941, named it
    _RECALIBRATE|calibrat              found lightgbm in a set, named it
    a line range 2975-3050             found the final fit, cleared it
    ast.walk for FunctionDef "fit"     found CNN1DClassifier.fit, not the
                                       ensemble's, and I read 120 lines of the
                                       wrong function
    Select-String on a test run        matched nothing and I read the empty
                                       output as "the test passed" -- it had
                                       failed

The unfiltered traceback identified the execution path immediately. The
library-only reproduction established ownership conclusively. Both were
available from the first minute.

> **Discovery tools must not encode the hypothesis they are being used to
> test.**

Stated as a rule for this project:

> **OBSERVED FAILURE -> raw evidence first -> minimal reproduction second ->
> source narrowing third -> hypothesis testing last.**

For exceptions, the full traceback. For filesystem populations, the full
census. For state discrepancies, a complete key and value-shape comparison. For
dependency interactions, a project-free reproduction. For source structure, a
class-qualified enumeration rather than the first textual hit.

Each wrong location in this investigation cost a cycle. The unfiltered
traceback would have cost one.

### Register at the true close of 2026-08-15

| item | state |
|---|---|
| `PREFLIGHT-TOKEN-SUBSTRING-1` | **CLOSED** at `a8cc484`. Refinement open: distinguish a configured credential from an authenticated one. |
| `LGBM-SKLEARN-FEATURE-NAME-WARNING-1` | NEW, non-blocking, confirmed upstream interoperability behaviour. Renamed from `ENSEMBLE-FEATURE-NAMES-1`, which falsely assigned ownership. |
| `PROJECT-ROOT-HARDCODED-1` | OPEN, and the largest remaining item |
| `CONFIG-DEAD-PATHS-1` | OPEN |
| `OUTPUT-ROOT-CONFLATION-1` | OPEN |
| `ROOTFIX-VERIFY-TEXTUAL-1` | OPEN |
| `SHAREDSTATE-LOAD-WRITES-1` | OPEN |
| `PACKAGES-NO-INIT-1` | OPEN |
| `MIGRATION-RECORD-SEPARATOR-1` | OPEN |
| `CHANGELOG-DUP-2026-06-25` | OPEN |
| `WORKTREE-EOL-DRIFT-1` | OPEN |
| SESSION_2026-06-19 item 5 | OPEN |
