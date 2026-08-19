# SESSION 2026-08-17 to 2026-08-19 -- paths acquire an owner

**Author: Monzia Moodie**

**Commits:** `ed10e41`, `27267d0`, `ec8e51b`, `f89ce6b` (four, across three days)
**Ratchet:** 4997 -> 5027
**Preceding head:** `a8cc484`

---

## What this period was

One idea pursued to its conclusion: **a path derives from the authority that
owns what the path contains.** Repository identity, artifact identity and state
identity are distinct domains, and conflating them is how a developer's
absolute path became the value on every machine.

| commit | date | unit | files | lines |
|---|---|---|---|---|
| `ed10e41` | 2026-08-17 | docs: a correction beside the record | 2 | +191 |
| `27267d0` | 2026-08-18 | RUNTIME-SENTINEL-TEST-ARTEFACT-1 | 5 | +411 -3 |
| `ec8e51b` | 2026-08-18 | PROJECT-ROOT-HARDCODED-1 | 6 | +697 -12 |
| `f89ce6b` | 2026-08-19 | OUTPUT-ROOT-CONFLATION-1 | 5 | +557 -31 |

All figures quoted from `git show --stat`. Dates from
`git log --date=format-local`, because a continuous-integration log timestamp
is a different clock from a commit's and I have confused the two before.

Ratchet, quoted from `tests/EXPECTED_SUITE_SIZE`:

```
# 2026-08-17 -- 4997 -> 5006 (+10 new, -1 parametrized). RUNTIME-SENTINEL-TEST-ARTEFACT-1.
# 2026-08-18 -- 5006 -> 5023 (+17). PROJECT-ROOT-HARDCODED-1.
# 2026-08-19 -- 5023 -> 5027 (+4). OUTPUT-ROOT-CONFLATION-1.
```

---

## 1. `ed10e41` -- a correction beside the record, not inside it

The 2026-08-15 session document and the changelog both stated that
`PREFLIGHT-TOKEN-SUBSTRING-1` was CURRENTLY FAILING and that any cloud run was
gated. True when written, after `c1fb110`. Commit `a8cc484`, later the same
day, CLOSED it.

So the repository asserted an active blocker that was closed. Real evidence
drift, not an aesthetic one.

### Two principles pulled in opposite directions, and the conflict is stated

`REQUIRED_PROVENANCE_CORRECTION` holds that corrections belong BESIDE records,
never inside them. Against that, the stale line was not a record of what
happened -- it was a PRESENT-TENSE CLAIM about current state.

**Superseding satisfies both.** The changelog is newest-first, so a 2026-08-16
entry sits ABOVE the stale claim: a reader meets the correction before the
thing corrected. In-place editing cannot do that, and it would destroy the
record of what was believed when.

The structural gate asserts the original text SURVIVES in both files.

### A renaming, and why it mattered

I had filed a suite warning as `ENSEMBLE-FEATURE-NAMES-1` -- a name assigning
ownership to the ensemble and implying it violated a schema contract. It did
not. Renamed `LGBM-SKLEARN-FEATURE-NAME-WARNING-1`, established by a
library-only reproduction with ALL project code removed:

```
lightgbm 4.6.0 + scikit-learn 1.8.0
ARRAY only      feature-name warnings: 3   (three folds)
DATAFRAME only  feature-name warnings: 0
```

LightGBM synthesises feature names from unnamed array input; scikit-learn then
sees estimator metadata against unnamed prediction input and warns.
`variant_ensemble.py` is internally consistent.

---

## 2. `27267d0` -- a sentinel must exist where the root must be found

**RUNTIME-SENTINEL-TEST-ARTEFACT-1.** A defect in a module I had built three
days earlier.

```
PROJECT_SENTINELS = (
    "pyproject.toml",
    "src/genomic_variant_classifier",
    "tests/EXPECTED_SUITE_SIZE",      <-- a TEST-SUITE ARTEFACT
)
```

The third entry identifies a DEPLOYMENT root by a file every correct
deployment excludes. The conjunction could not hold in a container by
construction.

MEASURED against this repository's own build files:

```
Dockerfile:185    COPY . .          (trainer stage, WORKDIR /app)
.dockerignore     tests/            excluded

pyproject.toml                  in trainer image : True
src/genomic_variant_classifier  in trainer image : True
tests/EXPECTED_SUITE_SIZE       in trainer image : FALSE
```

So `resolve_project_root()` would have RAISED on import of any module reaching
`agent_layer.config` inside the trainer image -- where cloud training runs.
`config.py` is imported at MODULE SCOPE by thirteen modules, so the failure
would have been total.

The API image is unaffected for a different reason: Dockerfile lines 109-126
copy only `api/`, `models/`, `utils/`, `monitoring/model_registry.py` and the
package `__init__.py`, so `agent_layer/` is not present at all. Confirmed
independently with `python -X importtime`.

### Invisible by construction

Nothing imported `runtime_paths` until `PROJECT-ROOT-HARDCODED-1` needed it.
The existing test asserting "every sentinel is load-bearing" passed throughout,
because it only ever ran inside the repository -- where all three sentinels
exist.

**I did not find this by inspection.** `config.py` is imported at module scope
by thirteen modules, so I asked what happens if resolution FAILS, and that
question required measuring a SECOND ENVIRONMENT.

Two sentinels plus the declared project name lose NO discrimination: this
repository True; `C:/Users/monzi`, `C:/Projects`, `C:/Windows` and `C:/` all
False.

### The ratchet decreased by one, deliberately

`test_every_sentinel_is_load_bearing` is parametrized over
`PROJECT_SENTINELS`, so it dropped from three cases to two. Recorded as
`PHYLOPTEST-DUP-1` was. The test still asserts every REMAINING sentinel is
load-bearing; it has one fewer to check because one was wrong to require.

**Ten new tests, 5 of 5 mutations detected.** Every sentinel is now checked
against the REAL `.dockerignore`, and a second test rejects any sentinel under
an artefact root -- tests, docs, build, dist, notebooks, htmlcov, .github,
logs, outputs -- because that is a category error regardless of what any
particular `.dockerignore` says today.

---

## 3. `ec8e51b` -- the project root is resolved, never guessed

**PROJECT-ROOT-HARDCODED-1.**

```
PROJECT_ROOT = Path(os.getenv("GVC_PROJECT_ROOT",
                               r"C:\\Projects\\genomic-variant-classifier"))
```

`GVC_PROJECT_ROOT` is set NOWHERE -- not in continuous integration, not in the
Dockerfile, not in any script, not in the shell. So the fallback was the value
EVERY consumer received, and on the Linux runner it named a path that cannot
exist. Every prior run imported it and passed only because nothing
dereferenced it.

### The blast radius was smaller than I claimed

MEASURED by syntax tree: **4 definitions of `PROJECT_ROOT`, and THREE WERE
ALREADY CORRECT.** `c3_inventory.py:21`, `c3_sweep.py:29` and
`run11_preflight.py:27` all use `Path(__file__).resolve().parent.parent`.

I had framed 27 loads in `scripts/` as part of the problem. They are the part
of the codebase that already did this properly. Only `config.py:17` was ever
defective.

17 load-context references, 0 attribute accesses, 13 importers.

### The first attempt failed, and the gate caught it

A version patching `config.py` alone reported **4786 passed / 10 skipped / 1
FAILED**. The installer rolled back and `git status` came back empty.

```
FAILED test_agent_root_anchor.py
       ::test_the_default_TRACKS_the_environment_not_the_cwd
```

That test sets `GVC_PROJECT_ROOT` to a bare `/probe_anchor` -- a path that does
not exist -- and asserts the agent follows it. Under the old line `os.getenv`
accepted ANY string. **I wrote that test on 2026-08-14**, and its own docstring
states the assumption: *"config.py reads GVC_PROJECT_ROOT at import time"*. It
encoded the DEFECT as a contract.

Proven by a three-way matrix, not argued:

```
OLD test + OLD config    2 passed
OLD test + NEW config    1 FAILED    <- the gate's failure, reproduced
NEW test + NEW config    3 passed
```

### Two measurements that prevented wasted cycles

`test_agent_root_anchor.py` imports ONLY `subprocess`, `sys` and `pytest` -- no
`io`, no `Path`, no `os`. A first draft of the replacement used `io.open` and
would have raised `NameError`. The replacement has ZERO free names.

`Path.read_text(newline=...)` is Python 3.13+; this environment is 3.12, so the
call raised `TypeError` before any output -- and WITHOUT that argument
`read_text` translates CRLF to LF, making every `config.py` anchor match
nothing.

---

## 4. `f89ce6b` -- a path derives from the authority that owns it

**OUTPUT-ROOT-CONFLATION-1.**

```
SHAP_REPORT_DIR       = PROJECT_ROOT / "reports" / "shap"       (line 194)
LITERATURE_DIGEST_DIR = PROJECT_ROOT / "reports" / "literature"  (line 327)
```

Both are ARTIFACT DESTINATIONS computed from REPOSITORY identity. Where output
goes is a deployment decision, not a fact about where the source lives.

The line numbers had moved from 174 and 307: the `PROJECT-ROOT-HARDCODED-1`
comment block shifted both by twenty. Re-measuring caught that.

### One authority, not two

```python
_RUNTIME_PATHS = resolve_runtime_paths()
PROJECT_ROOT   = _RUNTIME_PATHS.project_root
SHAP_REPORT_DIR       = _RUNTIME_PATHS.reports_root / "shap"
LITERATURE_DIGEST_DIR = _RUNTIME_PATHS.reports_root / "literature"
```

NOT `resolve_project_root()` alongside a second `resolve_runtime_paths()` call.
That would be two authorities for one process -- the parallel-vocabulary defect
this project keeps removing -- and each call performs a full discovery walk.

The single call is a **CONFIGURATION SNAPSHOT**, not an optimisation. Runtime
path configuration is immutable for the lifetime of a process: fresh process,
fresh resolution; existing process, stable paths. A process whose artifact
identity silently moved mid-execution would be far harder to reason about than
one requiring a restart.

`_RUNTIME_PATHS` is PRIVATE deliberately. The authority belongs in
`paths.runtime_paths`; a public name would invite `from config import
RUNTIME_PATHS` imports replacing the old global constants with one global
service locator.

### Two tests from the previous commit were revised, and that is the point

```
test_project_root_is_assigned_from_the_resolver
test_the_resolver_is_imported
```

Both required a PARTICULAR implementation -- the exact call, the exact import
-- rather than the property the commit existed to guarantee. Appropriate as
migration guards; wrong as permanent architectural constraints. **This is the
milder form of the `/probe_anchor` mistake: encoding HOW as the contract for
WHAT** -- caught one commit later rather than four days later.

And a third asserted

```
SHAP_REPORT_DIR == PROJECT_ROOT / "reports" / "shap"
```

which LITERALLY SPECIFIED the defect being closed.

### The release-blocking test needs separated roots

MEASURED: on this workstation `artifact_root == project_root`, so the defect is
INVISIBLE under the default configuration. Testing only here can never validate
the boundary -- the same lesson the sentinel repair taught.

> **An artifact path contract must be tested in an environment where artifact
> identity DIFFERS from repository identity.**

`test_the_two_root_domains_can_DIVERGE` uses `GVC_ARTIFACT_ROOT`, the supported
injection mechanism, and asserts BOTH directions: reports follow artifact
identity, checkpoints do NOT. The repair is an OWNERSHIP correction, not a
blanket move of every path under `artifact_root` -- and two sabotage cases
confirm that half is load-bearing.

**6 of 6 mutations detected.**

---

## Register at close

| item | state |
|---|---|
| `RUNTIME-SENTINEL-TEST-ARTEFACT-1` | CLOSED at `27267d0` |
| `PROJECT-ROOT-HARDCODED-1` | CLOSED at `ec8e51b` |
| `OUTPUT-ROOT-CONFLATION-1` | CLOSED at `f89ce6b` |
| `PATHS-BY-INJECTION-1` | NEW, Stage B. Move `interpretability_agent` and `literature_scout_agent` from imported path constants to an injected `RuntimePaths`, so `config.py` retreats to configuration rather than acting as a filesystem locator. |
| `GITATTRIBUTES-UNGATED-1` | NEW. MEASURED 2026-08-19: `.gitattributes` carries 31 rules and a documented near-corruption of the AlphaFold fixture, and NO test asserts any of them. Delete `*.py text eol=lf` and nothing fails. A rule file with no gate is a convention, not a contract. |
| `CONFIG-DEAD-PATHS-1` | OPEN, a scope decision. Reachability over `config.py` finds 35 of 71 module-level assignments unreachable: 7 stale paths, 28 unwired roadmap constants (EWC, ResNet, replay, SHAP tuning, endpoints, LOG_LEVEL). The latter are planned interfaces whose consumers do not exist yet, not dead code. The applier ASSERTS the seven remain, so removal is deliberate. |
| `WORKTREE-EOL-DRIFT-1` | OPEN, and correctly characterised on 2026-08-11 as *"benign for commits; load-bearing for byte-exact tooling."* Count grown 102 -> 124 of 981 tracked Python files. I twice mischaracterised this item as newly urgent; the original wording was right both times. |
| `ROOTFIX-VERIFY-TEXTUAL-1` | OPEN |
| `SHAREDSTATE-LOAD-WRITES-1` | OPEN |
| `PACKAGES-NO-INIT-1` | OPEN |
| `MIGRATION-RECORD-SEPARATOR-1` | OPEN |
| `CHANGELOG-DUP-2026-06-25` | OPEN |
| `LGBM-SKLEARN-FEATURE-NAME-WARNING-1` | OPEN, non-blocking, confirmed upstream |
| `PREFLIGHT-CREDENTIAL-USABILITY-1` | OPEN, a refinement: distinguish a configured credential from an authenticated one |
| SESSION_2026-06-19 item 5 | OPEN |

---

## The method, and where it failed

**The rule this period produced, from the LightGBM investigation:**

> **Discovery tools must not encode the hypothesis they are being used to
> test.**
>
> Observed failure -> raw evidence first -> minimal reproduction second ->
> source narrowing third -> hypothesis testing last.

Locating that warning took eight probes. Seven were filtered searches, and each
filter decided in advance what could be seen -- including one `Select-String`
that matched nothing, which I read as "the test passed" when it had failed.

**The recurring failure of this period was narrower and sharper: a check whose
node handling is incomplete.** Four instances:

- an `ast.dump` search that matched the DOCSTRING explaining the defect it
  guarded against;
- a free-name detector reading `g.target.id`, blind to `for k, v in ...`
  because tuple unpacking is an `ast.Tuple`;
- a reachability census over `ast.Assign` only, missing eight `ast.AnnAssign`
  constants;
- a `FunctionDef` walk that returned `CNN1DClassifier.fit` when the ensemble's
  was wanted, after which I read 120 lines of the wrong function.

Each was a defect in a CHECK, not in what it checked. Every census here now
walks node types rather than assuming one.

**And twice I proposed repairing something already recorded correctly.**
`WORKTREE-EOL-DRIFT-1` said *"benign for commits; load-bearing for byte-exact
tooling"* on 2026-08-11 -- precisely the conclusion I reached on 2026-08-19
while believing it newly discovered. Reading the register before acting on it
is cheaper than re-deriving it.

**The gates worked.** One unit failed on one test out of 4,724 and rolled back
five files cleanly; another failed on one of 4,798 and rolled back two. Each
cost a cycle; the alternative was a partially adopted module.


---

## Addendum -- later on 2026-08-19: a count corrected

The register table above states that `.gitattributes` **carries 31 rules**. It
carries **37**.

MEASURED 2026-08-19, three ways, all agreeing:

```
total lines            : 74
non-blank, non-comment : 37
DISTINCT patterns      : 37
```

No line shares a pattern with another, so there is no counting method under
which 31 is defensible. The figure came from reading a truncated terminal
display rather than enumerating, and I stated it twice before measuring.

The table is NOT rewritten. Corrections belong beside records, and this is the
correction.

**Everything else in that entry stands.** The rule file was ungated, the
2026-07-12 AlphaFold near-corruption is real and is quoted accurately, and the
conclusion -- that a rule file with no gate is a convention rather than a
contract -- is unaffected by the count.

### Why a six-count error is worth an addendum

A count is a claim. This one was asserted twice, corrected only when someone
asked for the enumeration, and by then it was committed in two places. It is
the same failure this document already records four times in checks with
incomplete node handling -- a quantity produced by a method I did not name and
did not verify -- applied to prose instead of to code.

### The unit that produced it

`GITATTRIBUTES-UNGATED-1` closed at `a18ff26` (3 files, +282 -2). 39 cases, 8
of 8 sabotage mutations detected, continuous integration green at 5052 passed /
13 skipped / 1 xfailed against a ratchet of 5066.

Its own header records two further honest results: that the pre-install probe
from a temporary directory found one case passing on inherited working-directory
state, and that the `tests/fixtures/**` overrides are redundant while the
general rules exist.
