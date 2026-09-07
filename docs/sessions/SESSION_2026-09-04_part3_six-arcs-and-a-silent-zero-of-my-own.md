# SESSION 2026-09-04 part 3 -- six arcs, and a silent zero of my own

**Author: Monzia Moodie**
**Measured at commit:** 8678fc4
**Suite transition:** NEUTRAL. No production code changes, no test changes.

Every figure here was RE-MEASURED at `8678fc4` immediately before this record
was written. Nothing is carried from earlier in the session.

---

## 0. What this covers

Six arcs completed after `8678fc4` was pushed. None of them changed the
repository: five were filesystem or working-directory work, and one was an
enumeration. The repository is unchanged at `8678fc4` with a clean tree and
three agreeing counters -- `tests/EXPECTED_SUITE_SIZE`, the README badge
`tests-6237`, and `docs/ROADMAP.md` line 97 `6,237 collected`.

---

## 1. The home-directory repository: 849 objects classified, then removed

`HOME-DIRECTORY-REPOSITORY-HAS-NO-HISTORY-AND-ORPHANED-OBJECTS-1`, open since
2026-09-04, is CLOSED.

**Why the objects were unreachable, stated because the question was asked.**
Git stores blobs (file CONTENT, with no name), trees (the directory listing
that MAPS names to blobs), and commits (each pointing at one tree). Reachable
means walkable from a reference. `git fsck` reported:

```
Checking ref database: 100% (1/1)
notice: No default references
849 objects   848 dangling blobs + 1 dangling tree
```

No branch, no tag, no commit. The single tree was git's CANONICAL EMPTY TREE,
0 bytes. **A tree is the only thing that records a name, and the only tree
present recorded nothing.** So the path names were not lost through damage;
they never existed in that object store. The CONTENT was always readable --
`git cat-file -p` prints any blob -- and only the name-to-content mapping was
absent.

**All 848 blobs classified by reading their bytes**, not by sampling:

```
plain text or source            491   27,487,678 B
other binary                    345    2,101,433 B
bundled JavaScript                6   21,422,158 B
Windows executable or library     5  138,548,224 B
ICU Unicode data                  1   10,468,208 B
TOTAL                           848  200,027,701 B
```

The five `MZ`-headed files are Windows executables. The six JavaScript bundles
open `import { createRequire as __createRequire } from "node:module"` -- a
bundler's preamble. The Unicode file carries `CmnD` and a Unicode copyright
line: International Components for Unicode common data. **The 491 "plain text"
blobs are application logs**, every sampled one opening
`[2026-06-08T05:27:13.734Z] [INFO] Tracing collector status check: Not...`,
spanning 2025-12-14 to 2026-06-08.

A toolchain and six months of its own logs. Nothing of this project's.

**A METHOD CORRECTION.** An earlier statement in this session called these
"redistributable toolchain artifacts" on the evidence of the SIX LARGEST blobs.
Largest-first is a biased sample -- build outputs are large, logs and source are
small -- and the full census changed the description from "toolchain" to
"toolchain plus 491 log files". The conclusion held; the description did not.
Six of 848 is 0.7 percent, and deleting on it would have been extrapolation.

**Removal, with preconditions proven first.**

```
BEFORE  count 849   size 71,060 KiB   .git on disk 72,791,946 B
        rev-parse HEAD    exit 128 -- NO commit exists
        for-each-ref      no output -- NO branch, NO tag
AFTER   count 0     size 0            .git on disk     26,397 B
```

`git gc --prune=now --aggressive` reported `Nothing new to pack`, which is
correct: with zero reachable objects there is nothing to compress.
**72,765,549 bytes reclaimed by the prune.**

The empty repository was then removed, and the verification is a FAILURE:

```
Test-Path 'C:\Users\monzi\.git'                        False
git -C 'C:\Users\monzi' rev-parse --show-toplevel
    fatal: not a git repository (or any of the parent directories): .git
```

**The home directory no longer presents as a repository.** That is the hazard
the standing `git -C $Repo` rule existed to work around. 72,791,946 bytes total.

**A NUMBER THAT WAS ALWAYS TWO NUMBERS.** 200,027,701 bytes is the UNCOMPRESSED
content; the object store held it in 72,791,946 bytes. Both were stated before
the decision, because the larger figure had been in play all session and is not
the reclaim figure.

---

## 2. `pyproject.toml` in the home directory -- unrelated, and still present

`HOME-DIRECTORY-HAS-A-PYPROJECT-TOML-1` remains OPEN. Read in full, 419 bytes,
digest `3566710cc1b221886cd12a837da9ebd7d7472d9626275438f597e46d5fb14dc1`:

```
name = "pyquil"
description = "PyQuil is a Python library for quantum programming using Quil."
authors = [{name = "Rigetti Computing"}]
requires-python = ">=3.13"
[tool.uv.workspace]
members = ["weather"]
```

**It is not a copy of this project's manifest.** The repository's own is 6,437
bytes at `9220939fade23b35608fa1db0f0621cc22f805d1e31f3bc70625467219bdfcf0`.
This declares Rigetti Computing's quantum-programming library and a `uv`
workspace member named `weather` -- almost certainly a `uv init` or tutorial
left in the home directory, plausibly the same event that created the git
repository whose objects were a Node.js and Python toolchain.

Its `[tool.uv.workspace]` block is the live hazard: any `uv` command run from
beneath `C:\Users\monzi` resolves to this workspace. Removal is Monzia's
decision and is not taken here.

---

## 3. `variant_ensemble_cff925c.py` -- proven redundant, then removed

`SNAPSHOT-AT-A-COMMIT-KEPT-AS-A-WORKING-FILE-1`, and it is CLOSED.

**MEASURED:** 35,027 bytes at the repository ROOT, digest
`df13fe1f051fb06f2c99dc76bbba9ca5f92b75057d5d078065e71364e97bb4f5`. Not
tracked; `git status --untracked-files=all` showed NOTHING, because
`.gitignore:107` names the file EXPLICITLY -- not a pattern, an individual line,
sitting among `agent_layer.zip`, `lovd/`, `tumor gene data.txt` and `.claude/`.

**What distinguished it, from the parse tree of both files:**

```
                     snapshot                 live module
bytes                35,027                   157,288
neural framework     tensorflow.keras         torch, torch.nn, torch.utils.data
project imports      NONE                     six sibling modules
classes              4                        6  (+_IsotonicCalibrator, SequenceWindows)
top-level functions  2                        11
constants            3                        11  (incl. EXPECTED_TABULAR_FEATURE_COUNT)
pure ASCII           False                    True
```

A TensorFlow-era ancestor, predating the PyTorch migration, the model-roster
split into modules, the fail-loud feature-count guard, and the ASCII invariant.

**BYTE-IDENTICAL TO HISTORY.** `cff925c` is a real commit -- `cff925ca86072252e
555c3ed9dff582324f322ae`, 2026-03-26, *feat(phase4): algorithm benchmarking,
ESM-2, uncertainty decomposition, LOVD validation*. The module lived at
`src/models/variant_ensemble.py` then, BEFORE the `src/genomic_variant_
classifier/` layout, and:

```
git show cff925c:src/models/variant_ensemble.py
    35,027 B  DF13FE1F051FB06F2C99DC76BBBA9CA5F92B75057D5D078065E71364E97BB4F5
the working copy
    35,027 B  DF13FE1F051FB06F2C99DC76BBBA9CA5F92B75057D5D078065E71364E97BB4F5
```

Every byte recoverable by one command, at any time. Deleting it lost nothing.

**A REFUSAL THAT WAS READ CORRECTLY BECAUSE THE EXIT CODE WAS PRINTED.** The
first attempt used the CURRENT path and `git show` exited 128, producing 108
characters. `Measure-Object` reported `Lines: 1`, which reads exactly like a
one-line file. Only the exit code distinguished an error message from content.

**FOUR TRACKED RECORDS NAME IT**, and `git grep` exited 0 -- a real match, not
the silent zero a non-matching pathspec produces. `REMEDIATION_2026-07-11_
test-suite-red.md:629` describes it precisely: *"A third, older generation is
still on disk ... whose header reads '55 features -- must match
DataPrepPipeline._engineer_features'. Three generations of the same contract --
55, 65, 97 -- coexist in the tree."* It was evidence in a drift investigation
long since settled by `EXPECTED_TABULAR_FEATURE_COUNT`.

One correction owed to that record: it says UNTRACKED; the file was IGNORED.
`OPEN_ITEMS_2026-07-18.txt:380` recorded it correctly as `35,027  ignored`.

**FOUR SIBLINGS REMAIN**, named as a group in two records since 2026-05-08:
`catboost_wrapper.py` 22,180 B, `test_catboost.py` 17,718 B,
`NOTEBOOK_CELL_FIXES.py` 3,362 B, `patch_finngen_wiring.py` 6,490 B. Two others
named in those records are already gone. `ROOT-DIRECTORY-UNGOVERNED-1` covers
the class; none was measured for byte-identity and none is removed here.

---

## 4. The tee, and a silent zero I built

`REDIRECT-MANGLES-NON-ASCII-1` and `REDIRECT-2>&1-LOSES-OUTPUT-1` (open since
2026-08-26) are addressed for all four instruments.

**THE DESIGN CHOICE.** `Probe_AuthorityCatalog` uses an `emit()` that prints and
accumulates. Copying that to the other three meant converting 41, 63 and 42
print sites -- 146 single-occurrence edits, each a chance to miss one SILENTLY.
A tee wrapping `sys.stdout` is ONE insertion per probe and CANNOT drop a line,
because it never enumerates them. `atexit` covers normal return and `sys.exit`
alike, so no body was re-indented.

**AND IT FAILED, SILENTLY, ON ITS FIRST REAL RUN.**

```
UnicodeEncodeError: 'utf-8' codec can't encode character '\udcff'
gvc_register_teetest.txt       0 bytes
probe exit: 0
```

The probe's OWN encoding self-test emits `\udcff`, a surrogate from
`surrogateescape`. The probe reconfigures its stdout with
`errors="backslashreplace"`; my tee wrote with `write_text`, which encodes
STRICTLY. The exception was raised inside `atexit`, where Python prints a
traceback and **leaves the exit code at 0**. A zero-byte report, and a success
code.

That is the silent-zero class this session has been hunting, built into the
tool meant to stop reading output through a mangling pipe. It was caught by an
instrument written for `PROBE-CONSOLE-ENCODING-1` and `-2`, closed 2026-08-23.

**Three repairs, each for a distinct failure:** `errors="backslashreplace"`,
matching the probe's own stdout policy; an assertion that raises if the buffer
holds bytes and the file gets zero; and failure reported to the real stdout,
because `atexit` will not change the exit code.

**Driven on the exact failing case** -- `\udcff` plus an em dash, curly quotes,
an arrow and a traceback -- for all three probes, then run against the real
corpus:

```
Probe_SectionConvention   exit 0    84,184 B    64e68fa050ef4410b9d3f1ea59e85ae047b12b25b93a91484d2c632f704aeb29
Probe_StillOpenLedger     exit 0    16,459 B    a11e32b8f5d43518c81b271e7ff22e4b9db410ffd2748f07a730a90dafc30190
Probe_FindingRegister     exit 0   195,499 B    995a08c093791cccd4fbe3f619925fdd4eceb24e9f7e06e64b670536288e6460
```

Each grew by EXACTLY 2,220 bytes and each keeps its print-site count unchanged
at 41, 42 and 63.

**A CHECK THAT PASSED FOR THE WRONG REASON.** A verification searched the tee
output for `written to` and found it -- in the probe's OWN line about the
measurement report, not the tee's size notice. The correct behaviour is the
opposite of what it appeared to confirm: the notice is written AFTER
`sys.stdout` is restored, so it belongs on the console and NOT in the file.

---

## 5. Two environment facts, both measured the hard way

**PowerShell 7, not 5.1.** `Set-Content -Encoding Byte` failed with *'Byte' is
not a supported encoding name*; it was removed in PowerShell 6 in favour of
`-AsByteStream`. A 5.1 idiom used against a 7 host without checking -- the same
class as inventing `--out` on a probe that never had it. The size comparison
added beforehand caught it on all six files: `read back: 0 B  matches
cat-file size: False`.

**A THIRD here-string escape failure.** A backtick inside a `@"..."@`
here-string is PowerShell's line-continuation character; Python received an
unterminated string. Stated as a rule two messages earlier and then violated:
`HERE-STRING-ESCAPE-REPEATED-AFTER-BEING-NAMED-1`.

---

## 6. What is NOT claimed

That the four remaining root-level strays were measured. Only their names and
sizes were read; none was tested for byte-identity against history and none was
removed.

That the 132 identifiers with no dated mention, or the 707-token review bucket,
were examined. The register reports both; neither was read.

That `pyproject.toml` in the home directory should be removed. Its content and
its `uv` workspace hazard are recorded; the decision is not made here.

That the section-convention and still-open-ledger reports were READ in full.
They were produced and their sizes verified. Their content is a separate
reading.

That the tee is correct for every encoding. It was driven on one surrogate, one
em dash, curly quotes, one arrow and one traceback. Other inputs are untested.

---

## 7. Ending state, re-measured at writing time

```
HEAD == origin/main            8678fc45452f134de98e124798bf5c3aafaf7747
working tree                   clean, untracked included
docs/CHANGELOG.md              733,490 B  15db58c94f510631...
newest changelog heading       ## 2026-09-04 part 18
newest SESSION record          SESSION_2026-09-04_part2_the-chain-gains-one-link.md
tests/EXPECTED_SUITE_SIZE      the node-identifier list, first line a header comment
README badge                   tests-6237
docs/ROADMAP.md:97             6,237 collected

C:\Users\monzi\.git            ABSENT
C:\Users\monzi\pyproject.toml  PRESENT, 419 B, unrelated to this project
variant_ensemble_cff925c.py    ABSENT
data/external/grch38           ABSENT
data/external/reference/       Homo_sapiens.GRCh38.dna.primary_assembly.fa   3,151,425,851 B
                               Homo_sapiens.GRCh38.dna.primary_assembly.fa.fai       6,600 B
```

Twenty-one acceptance gates today, warnings at 33 in every one, durations 842.9
to 1355.5 seconds. NO VALIDATED PREDICTIVE MODEL EXISTS; the observation set is
stated and no interval is claimed.
