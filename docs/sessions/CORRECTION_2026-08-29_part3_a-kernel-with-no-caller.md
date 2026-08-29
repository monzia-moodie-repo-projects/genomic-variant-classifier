# CORRECTION 2026-08-29 part 3 -- a kernel with no caller

**Author: Monzia Moodie**
**Applies to:** `b3619f2` and `81f6c4f` (their stated next actions), and
`docs/ROADMAP.md` section 6 (the DRIFT-1 row)
**Status:** the corrected claims are recorded here; neither commit is amended.

---

## 0. What went wrong

Two session records state a next action that measurement falsifies, and the
roadmap's authoritative status table has not moved since `abcb22e`.

`SESSION_2026-08-29_part2_an-invented-vocabulary-is-retired.md`, section 8:

> `SourceRegistry` exists and nothing in the drift package uses it yet.
> `SourceEvidenceManifest` accepts any syntactically valid source name, and the
> admission check that would refuse an undeclared one is not wired to anything.
> That wiring is the next unit.

**That is the wrong next unit**, and the reason is a measurement I had not
taken when I wrote it.

---

## 1. `DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1`

MEASURED 2026-08-29 at `b3619f2`, by parsing every tracked Python file and
counting construction sites -- calls to the type, and to `of` / `load` /
`from_records` on it -- separated into production and test:

| type | declared at | production | test |
|---|---|---|---|
| `SourceEvidenceManifest` | `source_release.py:311` | **0** | 30 |
| `SourceManifest` | `source_release.py:410` | **0** | 7 |
| `SourceArtifactKey` | `source_release.py:117` | **0** | 22 |
| `SourceArtifactIdentity` | `source_release.py:169` | **0** | 3 |
| `SourceDependency` | `source_release.py:254` | **0** | 4 |
| `SourceRegistry` | `data/source_registry.py:202` | **0** | 19 |

**Zero production construction sites across all six types.** The only non-test
modules importing them are `drift/__init__.py`, which re-exports, and
`source_delta.py`, which consumes types it never constructs.

`SourceRegistry` is imported by exactly one file: `tests/unit/test_source_registry.py`.

### Why that makes the admission check the wrong unit

This repository has ruled on guards with no reachable caller, twice.

`suite_transition.py` **DELETED three checks** because no reachable case
existed. `publish()`'s re-parse survived only once a reachable case was found.
And `preflight_data_guard.py` records of itself, after a repository-wide search
across every `.py`, `.sh`, `.ps1` and `.yaml` found zero invocations:

> A guard that is not invoked is not a guard; it is a comment that happens to
> be executable.

An admission check refusing an undeclared source, wired to a manifest that
nothing constructs, would be exactly that. It would also be untestable in the
only way that matters -- against a real acquisition -- because no real
acquisition reaches it.

### What is NOT claimed

That the kernel is wasted. It is a correct, tested, sabotage-verified identity
model, and Phase 1C requires one. What is measured is only that **nothing calls
it yet**, so the next unit must supply a caller rather than more guarantees.

That the tests are inadequate. 85 test sites across six types is thorough
coverage of behaviour; it is not evidence of use.

---

## 2. The roadmap row, stale since `abcb22e`

`docs/ROADMAP.md` section 6 carries the authoritative status table. Its DRIFT-1
row reads:

```
| DRIFT-1 | PHASE 0 CLOSED; the assessment itself remains open | abcb22e |
```

Since `abcb22e`, DRIFT-1 has moved through Phase 1A, 1B.1, 1B.2, 1B.3, 1B.4 and
1B.5 -- measured, by commit:

```
694da7f  a representation states its contract before it is compared
66e2737  a reserved layer gets its vocabulary
c77a1a9  four coordinate axes stop pretending to be one
cffc51f  the source kernel stops asserting what the data disproves
69e8524  the source declarations acquire a typed reader
ac14ab5  an invented vocabulary is retired for the registry that existed
```

The row is not wrong about Phase 0. It is silent about six commits since, which
in an authoritative status table directs future effort at work already done.

**The section's own design is sound and is preserved.** It says: *"stated by
identifier and commit rather than characterised, because a plan is the worst
place for a summary of work that has not been read."* The repair updates one
row's state and commit. It summarises nothing; `docs/CHANGELOG.md` and
`docs/sessions/` remain authoritative for the history.

### What was checked and is NOT stale

`Test suite | 5,705 collected` is current, and
`tests/unit/test_roadmap_counters_agree.py` binds it to the README badge and
`tests/EXPECTED_SUITE_SIZE`, so it cannot drift silently.

`ROADMAP-PROVENANCE-CLAIM-STALE-1` was repaired on 2026-08-26 and its repair
holds: the provenance paragraph no longer claims a measurement date that eleven
installers had moved past.

---

## 3. The corrected next action

**Not** the admission check.

The drift source kernel needs a CALLER before it needs further guarantees.
Phase 1C's reference profile is the unit that would construct a
`SourceEvidenceManifest` from real acquisition data, and it is blocked on a
decision recorded at `482c0c9`: `ARTIFACT-KEY-INSUFFICIENT-1` means persisting
a source manifest now would freeze a model measurement has already falsified.

So the sequence is: resolve the artifact-key question, then build the profile
that uses the kernel, and only then consider an admission check -- at which
point it will have a caller and can be tested against a real acquisition.

---

## 4. Why a correction rather than an amendment

`SESSION_2026-08-29_part2_an-invented-vocabulary-is-retired.md` is pinned by
digest `37fd69ec107350c2` in the attestation for `b3619f2`, and
`SESSION_2026-08-29_reading-replaces-building.md` by `b48485fcacc714ec` in
`81f6c4f`. Amending either would break those bindings.

`docs/ROADMAP.md` is different: it is a LIVING document that declares itself
"updated at the end of every session", and its counters are already patched by
every installer. Its stale row is repaired in place, in the same unit, because
that is what a living roadmap is for.
