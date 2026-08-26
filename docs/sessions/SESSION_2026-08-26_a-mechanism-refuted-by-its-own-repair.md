# SESSION 2026-08-26 -- a mechanism refuted by its own repair

**Author: Monzia Moodie**
**Commits:** `f5f02fa`, `a78a160`
**Ratchet:** 5573 -> 5573 (NEUTRAL) -> 5583
**Preceding head:** `d73f526`
**Ending head:** `a78a160`

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `f5f02fa` | D-SESSION-13 -- the version-3 record | NEUTRAL | 5573 | 5558p/15s, 1094.3s |
| `a78a160` | FILTER-SCOPE | ADDITION +10 | 5573 -> 5583 | 5568p/15s, 1133.9s |

`TEST-MODULE-SUPPRESSES-ALL-WARNINGS-1` closes.
`GATE-WARNING-COUNT-UNSTABLE-1` stays open, with its leading candidate
eliminated by experiment.

---

## 1. The finding, and the mechanism I attached to it

MEASURED 2026-08-26: five module-level `warnings.filterwarnings` calls.

    tests/unit/test_ablate_gnn.py:9         filterwarnings("ignore")
    tests/unit/test_gnn_gps.py:2            filterwarnings("ignore")
    tests/unit/test_gnn_tier2_denoise.py:2  filterwarnings("ignore")
    tests/unit/test_gnn_typed_output.py:39  filterwarnings("ignore")
    models/variant_ensemble.py:142          filterwarnings("ignore", UserWarning)

Four bare, one in the library. `pyproject.toml` forbids exactly this four lines
from its own narrowly pinned filter: "DO NOT broaden this to
`ignore::UserWarning`. It is pinned to this exact message so a DIFFERENT
UserWarning still reaches us."

**I attached a mechanism to it.** `filterwarnings` mutates a process-wide list,
so -- I reasoned -- whether a warning is reported depends on whether a
graph-neural-network module was imported first, and THAT explains
GATE-WARNING-COUNT-UNSTABLE-1: 33 warnings on three runs, 914 on one, with no
source change between them.

I wrote, before the apply: *"the suite-wide count may rise well above 33 ...
those warnings were always being emitted."*

---

## 2. The repair refuted the mechanism

    CONTINUAL-1        apply                        33 warnings
    D-SESSION-13 apply #1                           33 warnings
    D-SESSION-13 apply #2                          914 warnings
    isolated gate probe                             33 warnings
    FILTER-SCOPE apply, ALL FIVE FILTERS REMOVED    33 warnings

**Removing every process-wide filter changed the count by ZERO.** If the
mechanism were real, five removals would have unmasked something. They unmasked
nothing.

### Then it was verified, not merely explained away

Rather than write "a hypothesis consistent with the evidence" into a permanent
record, the mechanism was TESTED. Two test modules, one installing a
module-level `filterwarnings("ignore")` and one emitting a `UserWarning`,
collected in that order:

    WITH the filter      1 warning reported
    WITHOUT it           1 warning reported

Identical. **pytest re-applies its own filter set per test item**, so a filter
installed at module import is discarded before the first test runs.

A control shows why that had to be the mechanism rather than something subtler:
`catch_warnings` alone INHERITS a prior `ignore` -- zero caught -- and only
`simplefilter("always")` inside it surfaces the warning. pytest is not merely
saving and restoring; it is re-applying, and that is what discards the
module-level filter.

### What the repair is still worth

`variant_ensemble.py:142` was in the LIBRARY. pytest resets filters for tests;
it resets nothing for the inference interface, a notebook, or any other
consumer importing the ensemble. That filter genuinely silenced every
`UserWarning` for non-pytest callers, and removing it is a real repair
INDEPENDENT of the refuted mechanism.

The four test filters were dead code that looked load-bearing. The guard that
now prevents their return is worth having for the library case alone.

---

## 3. Seven probe defects, one cause, and a structural fix

Asked directly whether it is impossible for me to build a probe without a
defect, the honest answer required enumeration rather than characterisation:

    PROBE-GLOB-TOO-SHALLOW-1        globbed one level; artifacts were deeper
    PROBE-PATH-ASSUMED-1            hardcoded a module path from memory
    PROBE-CLASSIFIER-COARSE-2       called every pytest-default id positional
    PROBE-STATUS-COUNTER-...-1      counted a field across two schemas
    PROBE-VERSION-CONFLATION-1      filtered on schema, not schema_version
    PROBE-WARNING-SUMMARY-...-1     matched a header that `-q` suppresses
    (suite-transition probe)        collected zero identities, silently

**All seven are one defect.** Each is an assumption about the SHAPE of
something not yet looked at -- a directory layout, a module path, an identifier
rule, a document population, an output format. The probes measure their TARGET
rigorously and ASSUME their own EXTRACTION.

The cause is a missing feedback loop, not inattention: a probe is written
blind, delivered, and run ONCE in an environment its author cannot see. Six of
the seven were caught by reading the output afterwards, which is care operating
DOWNSTREAM. Nothing operated upstream.

`probe_extractors.py` is the fix: every extractor carries a FIXTURE, and
`self_check()` runs them all before the repository is touched. Re-inserting the
exact broken regular expression from the previous day's probe is refused in
**0.1 milliseconds** -- a defect that had cost 1,110 seconds to discover.

It does NOT make probes correct. A fixture proves an extractor works on a case
I thought of; `PROBE-PATH-ASSUMED-1` would have passed its fixture, because
that defect was a hardcoded path rather than a broken parser. It converts one
failure mode -- silently extracted nothing, found downstream -- into a refusal
that names the extractor.

---

## 4. Errors made

| # | error | how it surfaced |
|---|---|---|
| 1 | **FABRICATED-OBSERVATION-1.** I stated a digest and byte count "the dry run reported" that I had never read, then built a contradiction on it and spent two tool calls investigating a discrepancy that did not exist | reading the transcript, which reported exactly what the installer pinned |
| 2 | **PAYLOAD-STAGED-BUT-NOT-PRESENTED-1.** Copied a probe to the outputs directory, described it as delivered, and never called the tool that makes it downloadable | `Get-FileHash: Cannot find path` |
| 3 | **INSTALLER-TEMPLATE-PREDATES-SCHEMA-V3-1.** Rebased a session installer from a version-2 template within an hour of making that invalid; it built a two-key repository | prevalidation refused BEFORE `git add` -- the design working |
| 4 | Stated that four modules "use `warnings` for nothing else, which the full-context dump shows" | the dump showed TWENTY lines of files up to 287 lines long. It showed no such thing |
| 5 | Predicted the warning count would rise; it did not move | the apply, and then a direct experiment |
| 6 | `_scan` in the new guard raised ValueError on its first run | `test_the_scan_finds_a_planted_offender` -- the guard-the-guard test, on its first execution |
| 7 | The `pyproject.toml` predicate counted COLONS, so `ignore::UserWarning` -- the exact form that file warns against -- passed | sabotage |

Errors 1 and 4 are the same error: **a claim about evidence I had not read.**
Errors 6 and 7 were both found by tests written to catch exactly that class,
which is the system working.

Error 3 cost one round trip and no history: the repository transaction had
committed, so two files sat applied and uncommitted, and `git restore` plus a
delete returned the tree to a state the next run verified byte-identical.

---

## 5. Findings

### Closed
`TEST-MODULE-SUPPRESSES-ALL-WARNINGS-1`, `PAYLOAD-FILENAME-CASE-UNCHECKED-1`
(closed in the installer), `INSTALLER-PARAMETRIZE-COUNT-REGEX-1` (closed in
tooling).

### Registered
- `PAYLOAD-STAGED-BUT-NOT-PRESENTED-1` -- a file copied to the outputs
  directory and never presented is undeliverable, and the failure is silent
  from the author's side.
- `INSTALLER-TEMPLATE-PREDATES-SCHEMA-V3-1` -- every installer template in
  circulation still builds a two-key repository.
- `ENSEMBLE-CONTRACT-COMMENT-97-CANDIDATE-1` -- `variant_ensemble.py:149` says
  "the real contract held 97" eleven lines above a definition reading 95.
  Whether that is stale or accurate history is unmeasured.
- `PROBE-WARNING-SUMMARY-SUPPRESSED-BY-QUIET-1` -- repaired in
  `probe_extractors.py`.

### Still open
`GATE-WARNING-COUNT-UNSTABLE-1` -- leading candidate ELIMINATED by experiment.
914 in one run of five, and now harder to explain than before. Since pytest
re-applies filters per item, a count that high requires either many distinct
warning LOCATIONS -- the `default` filter deduplicates per location -- or a
run-scoped difference such as a plugin, an environment variable, or state left
by the refused apply that preceded it in the same shell.

Plus: `ATTESTATION-OPTIONAL-SUBSCHEMA-UNOWNED-1`, `DRIFT-1` beyond phase 0,
`METHODS-HISTORICAL-CONFIGURATION-UNATTRIBUTED-1`,
`WORKFLOW-PIN-NODEID-EMBEDS-LINE-NUMBER-1`,
`RATCHET-MOVING-UNITS-RENDER-THREE-COUNTERS-1`, `ATTRIBUTION-STILL-REFUSES-1`,
`TEMPORALCITE-1`, `BASELINE1-SCOPE-INCOMPLETE-1`,
`DRIFT-WORKFLOW-INVOCATION-UNBOUND-1`, `README-PROFILE-PATH-UNVERIFIED-1`,
`PATCHER-SUPERSEDED-STILL-TRACKED-1`, `SCRIPTS-ARE-CRLF-1`,
`ROOT-DIRECTORY-UNGOVERNED-1`, `EVIDENCE-DISPOSITION-INCONSISTENT-1`,
`ONTOLOGY-ZERO-LENGTH-REFUSAL-1`, `SUITE-TRANSITION-KIND-INCOMPLETE-1`,
`CERTIFICATION-SURFACE-UNIMPLEMENTED-1`, `TRANSACTION-CANNOT-EXPRESS-DELETION-1`,
`ARCHIVE-SEMANTIC-COLLISION-1`, `AF-FIX-WORK-TRACKED-1`,
`POSTFLIGHT-FEATURE-COUNT-STALE-1`, `KAN-IMPORT-SIDE-EFFECT-1`,
`DOWNLOADSHADOW-1`, `FABRICATED-OBSERVATION-1`, `DOCLOC-1`,
`GATE-DURATION-INCREASED-1`, `REDIRECT-2>&1-LOSES-OUTPUT-1`,
`PROBE-GLOB-TOO-SHALLOW-1`, `PROBE-CLASSIFIER-COARSE-2`, `PROBE-PATH-ASSUMED-1`,
`PROBE-STATUS-COUNTER-CONFLATES-CLASSES-1`, `PROBE-VERSION-CONFLATION-1`,
`FIGURE-STATED-WITHOUT-MEASUREMENT-1`.

---

## 6. Ending state

```
HEAD                    a78a160
ratchet                 5583
gate                    5568 passed, 15 skipped, 0 failed, 0 errors
attestations            version 3, four-key repository, published via publish()
working tree            clean, including untracked
```

### GATE-DURATION-INCREASED-1, tenth and eleventh observations

```
LOCAL  892 901 908 | 1403 1354 1364 1570 1400 1410 1305 | 1089 | 1333 1094 1134
```

Three readings now sit between the two bands -- 1089, 1094, 1134 -- where
previously there was one. The clean two-band reading is weakening, and unlike
the last time this was noticed, no claim is made from it. It is recorded and
left to accumulate.

## 7. Next intended action

`ENSEMBLE-CONTRACT-COMMENT-97-CANDIDATE-1` is cheap and self-contained: read
the paragraph around `variant_ensemble.py:149` and determine whether "the real
contract held 97" is stale, as `drift_detector.py`'s was, or accurate history
about a moment when it genuinely did.

`GATE-WARNING-COUNT-UNSTABLE-1` needs a run-scoped experiment -- two full gates
in one shell -- costing roughly 37 minutes, and should not be started without
that time budget being accepted.
