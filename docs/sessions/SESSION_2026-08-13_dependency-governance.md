# SESSION 2026-08-13 -- dependency governance: one vocabulary, six classifications

**Author: Monzia Moodie**

**Commits:** `569f4b1`, `f38c88b`, `bcefe49`, `5555c1a`, `7a27951`, `8cb0429` (six)
**Ratchet:** 4685 -> 4853
**Preceding head:** `cc350b9`

---

## What this session was

The PhyloP work closed, and then a much larger finding opened: the project had
**no single authority for what a dependency is**, and the instruments meant to
measure that were themselves defective.

| commit | unit | files | lines |
|---|---|---|---|
| `569f4b1` | PHYLOP-QUERY-INTEGRITY-1 | 5 | +965 -19 |
| `f38c88b` | BIGWIG-DEPENDENCY-CONTRACT-1 | 3 | +15 |
| `bcefe49` | DOC-EVIDENCE-STALE-1 | 1 | +65 -14 |
| `5555c1a` | DEPENDENCY-GOVERNANCE-1 | 9 | +1866 -2 |
| `7a27951` | DEPENDENCY-ONTOLOGY-1 | 4 | +492 -2 |
| `8cb0429` | REQFILES-NONASCII-1, PYTEST-ANYIO-REDIRECT-1 | 5 | +164 -6 |

---

## 1. `569f4b1` -- an uncovered position is absence, not a zero

**PHYLOP-QUERY-INTEGRITY-1**, carrying two defects in one method.

**PHYLOPBACKEND-1.** `fillna=0.0` was passed at the library boundary *before*
the `isnan` guard, so the guard could never fire on the pybigtools path. An
uncovered genomic position returned `0.0`, indistinguishable from a measured
conservation score of zero.

Measured against a real 659-byte bigWig file:

```
position      pybigtools(fillna=None)  pybigtools(fillna=0.0)  pyBigWig
1:900 (gap)        nan                      0.0  <- DEFECT      nan
1:500 (0.0)        0.0                      0.0                 0.0
```

**PHYLOPSWALLOW-1.** Every exception was absorbed into a sentinel and logged at
debug level, so `SOURCE_UNREADABLE` was unrepresentable.

### Three beliefs corrected by measurement

1. I claimed only the Run 17 preflight against the production asset could close
   the gap. Both libraries install from the Python Package Index, and pyBigWig
   can **write** bigWig files -- so a fixture could be built.
2. I classified an absent chromosome as a source fault. Both libraries **raise**
   for it, differently -- pybigtools a `KeyError`, pyBigWig a `RuntimeError` -- so
   it is an answer, not a fault.
3. A fake `_PyBigWig` returned `None` for an absent chromosome. No real version
   does.

Each belief was plausible and each was wrong, and the corrections came from
running the libraries rather than reasoning about them.

---

## 2. `f38c88b` -- nine tests that verify a repair were never run

**BIGWIG-DEPENDENCY-CONTRACT-1.** Fifteen lines, and the most consequential
small commit of the three days.

`pybigtools` was declared in `requirements.in:20` and `requirements.txt:235`.
**`pyBigWig` was declared nowhere.** So the nine real-asset parity tests written
to verify `569f4b1` had never executed in continuous integration -- they skipped
with "pybigtools and pyBigWig are both required", and pybigtools *was* present.

Verified against the Python Package Index: pyBigWig 0.3.25 ships five Linux
wheels and a source distribution, and **no Windows wheel**. Hence the platform
marker `pyBigWig>=0.3.25; sys_platform != "win32"`.

**The result is the causal signature:** delta passed +9, delta skipped -9, delta collected 0,
on both Python 3.11 and 3.12. A test that skips is not a test that passes, and
nothing distinguished the two until this was measured.

---

## 3. `bcefe49` -- a closed limitation left in the present tense

**DOC-EVIDENCE-STALE-1.** A docstring stated as current fact: *"The libraries
themselves are NOT installed... The Run 17 preflight against the real
9.19-gibibyte asset is what closes it."*

Every clause was false, and had been falsified **the same day it was written**.
Rewritten as a dated validation chronology, so a reader sees when each claim was
established rather than an assertion with no time index.

---

## 4. `5555c1a` -- three analyzers, one vocabulary

**DEPENDENCY-GOVERNANCE-1.** The largest unit of the three days: 1,866
insertions.

### The instrument failure that started it

A parser was pointed at `requirements-dev.lock` -- 310,494 bytes, 180 packages --
and reported **zero**. It is a `pip-compile --generate-hashes` artifact, so
every record spans continuation lines. The parser split before joining, handed
`aiobotocore==3.6.0 \` to `Requirement()`, caught the exception, and continued.
Every record failed identically and silently.

> A parser that silently drops every record looks exactly like a file that
> contains nothing.

That reading nearly supported deleting a hash-pinned supply-chain artifact.

### Why one model and not three

Two analyzers had each invented an identity rule -- `req.name.lower()` in one,
`p.lower().replace("-", "_")` in the other. Measured with packaging 26.0, a
naive `.lower()` disagrees with `canonicalize_name` on **six of ten** sampled
names. And measured against installed metadata, the hyphen-to-underscore
distribution-to-module guess disagrees with reality on **four of seven**:

```
pyBigWig        -> pyBigWig, pyBigWigTest    (guess: pybigwig)
beautifulsoup4  -> bs4                       (guess: beautifulsoup4)
pyyaml          -> yaml, _yaml               (guess: pyyaml)
python-dateutil -> dateutil                  (guess: python_dateutil)
```

`pyBigWig` is this project's own dependency and Python imports are
case-sensitive, so that guess finds nothing.

### The census reads syntax, not text

A text search for a package name matches comments, docstrings and string
literals. The census walks `ast.Import` and `ast.ImportFrom` nodes, and
classifies each site by **what the handler catches**:

```
except ImportError          optional
except ModuleNotFoundError  optional for a MISSING package; a plain
                            ImportError from a BROKEN one escapes
except Exception / bare     optional, indiscriminately
except ValueError           NOT optional -- ImportError escapes it
```

Three of eight handler shapes were classified wrongly before that repair.

### A performance error, measured rather than reasoned

`packages_distributions()` is **not cached** by importlib: 0.541 s first call,
0.536 s second. Calling it per package made six lookups cost 3.25 s against
0.001 s with a shared mapping, turning a 0.05 s test file into 17.75 s. Behind
an `lru_cache`: 0.68 s.

---

## 5. `7a27951` -- six measured classifications, each with its evidence

**DEPENDENCY-ONTOLOGY-1.** An import census over `src`, `scripts` and `tests` --
941 files walked, 941 parsed, zero failures -- measured what actually imports
what. **Not one of six packages was correctly scoped by the file it sits in.**

```
seaborn, jinja2   REPORTING. Unguarded imports at report_generator.py:45-46,
                  with sns.set_style() executing at MODULE SCOPE. Both declared
                  development-only, both absent from requirements-api.lock --
                  while matplotlib, two lines above them in the same import
                  block, is present at line 24.
pyfaidx           REFERENCE_SEQUENCE. 25 imports: 23 in scripts/, 2 in tests/,
                  18 of them hard.
pre-commit        DEV_TOOLING. Zero imports, and that is the CORRECT census
                  result for a console script.
httpx, anyio      UNRESOLVED, deliberately. Reached only through
                  fastapi.testclient.TestClient.
```

**Scope is two axes.** Neither "seaborn = development" nor "seaborn = runtime"
is true: it is a reporting dependency, required by the training and developer
profiles and absent from the API -- established by `python -X importtime -c
"import api.main"`, which shows no seaborn, jinja2, reports or report_generator
in the import graph.

`UNRESOLVED` is a first-class state. Recording `httpx` and `anyio` as
"transitive and misplaced" would assert more than the evidence carries.

---

## 6. `8cb0429` -- two files, four lines

**REQFILES-NONASCII-1** and **PYTEST-ANYIO-REDIRECT-1.**

Three em dashes across two tracked requirements files, replaced with the ` -- `
that `requirements.txt` already uses at lines 160, 166 and 177. Nothing was
broken -- these files pass through PowerShell `Set-Content`, `Copy-Item` and
installers that assert ASCII purity, so a mojibake byte surfaces during an
install rather than during review.

`pytest-anyio>=0.0.0` removed: a 3,559-byte package at version 0.0.0 whose own
summary says the plugin is built into anyio. **Proven, not quoted:** anyio
4.13.0 declares exactly one entry-point group -- `pytest11` pointing at
`anyio.pytest_plugin` -- and `pytest --trace-config` shows it registered.

### An error of mine, recorded

I framed the em dash as **one** file. The census across every `requirements*`
file found **two**. A repair aimed at the first would have fixed a symptom in
one location and declared the class closed.

---

## Errors in my own instruments, this session

Four assertions that could not fail, each in a property the code was written to
guarantee:

1. A reconciliation guard comparing `accounted == logical` -- true by
   construction when every record takes exactly one branch. Corrected three
   times before being replaced with a physical-line-count identity.
2. A parser test asserting on a `packaging` re-parse rather than the stored
   record, so a marker-discard mutation went undetected.
3. A census guard deleted rather than replaced, because no independent quantity
   existed to reconcile against.
4. A test file left stale while I read "21 passed" as success -- the tests were
   the old ones, passing against a module whose behaviour they no longer
   described.

The fix that worked in every case: **replace by parse-tree span, and read the
collected count from both the tree and pytest.**
