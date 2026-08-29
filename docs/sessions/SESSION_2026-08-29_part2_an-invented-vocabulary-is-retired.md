# SESSION 2026-08-29 part 2 -- an invented vocabulary is retired

**Author: Monzia Moodie**
**Commits:** `69e8524`, `ac14ab5`
**Ratchet:** 5690 -> 5709 -> 5705
**Preceding head:** `81f6c4f`
**Ending head:** `ac14ab5`, pushed

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `69e8524` | SOURCE-REGISTRY | ADDITION +19 | 5690 -> 5709 | 5694p/15s, 1042.3s |
| `ac14ab5` | RETIRE-SOURCENAME | RETIREMENT 13/9 | 5709 -> 5705 | 5690p/15s, 1312.2s |

DRIFT-1 Phase 1B.5. The defect recorded at `02c13b4` and refined at `54989dc`
is repaired.

---

## 1. The defect being repaired

`SourceName` was installed at `cffc51f` on 2026-08-28, on the stated basis --
written into its own module docstring -- that "no source registry exists
anywhere in the repository".

`configs/data_manifest.yaml` calls itself the "Canonical registry of every data
source under data/" on its own THIRD LINE, declares 32 sources, and is read by
five scripts under `scripts/maintenance/`. The authority search that missed it
looked only at Python files: `AUTHORITY-SEARCH-SCOPED-TO-ONE-LANGUAGE-1`.

MEASURED 2026-08-29 against the manifest:

| quantity | manifest | `SourceName` |
|---|---|---|
| declared sources | 32 | 18 |
| sources it cannot name | -- | **16** |
| members declared nowhere | -- | 2 (`gencode`, `esm2`) |
| declared aliases | 8 | **0 accepted** |
| aliases invented | -- | 26 |

Four of the sixteen are `irreplaceable` and constrained: `tcga` and `topmed`
are `controlled`, `rnaseq` and `validation_cohort` are `review`. A vocabulary
that cannot name `tcga` cannot express a manifest containing it, so
`SourceEvidenceManifest` would have refused a governed source because of a
missing enum member rather than a scientific judgement.

---

## 2. `69e8524` -- the reader, before anything depends on it

`src/genomic_variant_classifier/data/source_registry.py`, 15,948 bytes.

`StoragePolicy` in `preflight_data_guard.py` was the template: a frozen
dataclass loaded from the same manifest, recording `source` -- the path it read
-- so a verdict can name its own provenance. `SourceRegistry.manifest_source`
does the same.

**What raw dictionary access permitted, and this refuses.** MEASURED: all five
maintenance scripts walk raw dictionaries.

```
meta.get("tier")            returns None for `teir`, and None != "controlled",
                            so the compliance gate ADMITS a controlled source
meta.get("location", ...)   one default spelled independently in three scripts
nothing validates           not `tier: contrlled`, not an alias equal to its
                            own canonical name, not one alias claimed twice,
                            not an alias that is also a canonical source
```

**Where it departs from `StoragePolicy`, deliberately.** `StoragePolicy.load`
warns and uses documented defaults when the manifest cannot be read, because
"refusing every run because a configuration file moved would be a worse failure
than the one being guarded against". That does not transfer: one cannot invent
32 declarations, and a fallback registry would silently answer questions about
evidence this project does not have. It RAISES.

**Two real-manifest tests.** `test_the_real_manifest_loads` and
`test_no_controlled_source_is_marked_for_sync_in_the_real_manifest` read
`configs/data_manifest.yaml` itself. `skipped` stayed at 15 across the gate, so
both RAN -- if the file had been absent they would have skipped silently and
the reader would have landed untested against reality.

Before this, the 32 declarations had **no test of any kind**. Only the
`storage:` block and one tier claim were bound.

---

## 3. `ac14ab5` -- the retirement

**The design.** `SourceArtifactKey.source` becomes a VALIDATED STRING, and
registry membership becomes an ADMISSION question answered by
`SourceRegistry.canonical_for`.

Threading a registry through every construction would make `SourceArtifactKey`
unconstructible without a readable file -- the collapse this package has twice
repaired: `RepresentationIdentity` carries no source state, and
`SourceArtifactIdentity` carries no retrieval time.

**`_SOURCE` is restored**, because dropping the enum removed the REGISTRY
question and not the SYNTAX one. It ALLOWS a leading digit, because `1kgp` is a
declared source -- and my first version of its test wrongly expected `9lives`
to be refused, which would have rejected a real source in the very unit
repairing that failure mode.

**`ArtifactKind` stays.** It is not in the manifest and nothing else declares
it. The manifest declares location, tier, class, aliases, version, acquire,
regenerate, sync and notes -- nothing about Variant Call Format versus parquet
versus FASTA. The census separated them; treating them together would have
been wrong.

### A consumer the symbol census could not see

`source_delta.py` imports none of the retired names. It used `t.source.value`,
reaching through `SourceTransition` to a field whose TYPE changed. Three tests
failed on that one line.

**A census of symbols cannot find a consumer that never names the symbol.**

### The transition, measured

Both trees collected and their identity sets differenced:

```
old 78 | new 74 | REMOVED 13 | ADDED 9 | UNCHANGED 65
test_representation_identity.py contributes NOTHING -- all 17 persist
```

The 13 removed are exactly the enum's own tests: five parametrized spellings
that resolved, five that were refused, the alias-diagnostics test, the raw-string
refusal, and the unregistered-authority refusal. Every one tested behaviour that
was WRONG.

---

## 4. Twelve boundaries sabotaged, twelve detected

Three were undetected at first, and only ONE was a real gap.

**The real gap:** a fixture that could not distinguish exact matching from
substring matching, because `"clinvarplus" in "clinvar"` is False and
`"clinvar" in "clinvar"` is True. The replacement uses names that genuinely
nest -- the manifest declares `dbsnp` with the alias `dbsnp156`, so a source
whose name is a PREFIX of another must not answer for it.

**The other two were mine:** sabotage cases that mutated
`SourceArtifactIdentity` while naming tests that bind `SourceArtifactKey`.
Re-run against the right anchors, both detect.

---

## 5. Errors made

| # | error | how it surfaced |
|---|---|---|
| 1 | The consumer census searched only `.py` -- the SAME scope error this unit repairs | an all-file-type search afterwards; no live consumer was missed, but it could have been |
| 2 | `_SOURCE` referenced without being defined | collection error; it was removed at `cffc51f` when the enum replaced it |
| 3 | Expected `9lives` to be refused | `1kgp` is a real declared source; the pattern must allow a leading digit |
| 4 | Three expectations carried the retired enum's display casing | `('clinvar',) != ('ClinVar',)` |
| 5 | `artifacts_of("ClinVar")` returned nothing | the key holds the canonical name; the standard says exactly one |
| 6 | The installer's `PINS` table named the PREVIOUS unit's payloads | twelve pinned digests reported as never read; all six payloads would have been delivered UNVERIFIED |
| 7 | Referenced `vocab_doc.txt` before writing it | `FileNotFoundError`, the third time today |
| 8 | Sliced the vocabulary module by the wrong boundary | `SourceName` and `_ALIASES` survived the cut |
| 9 | Called the staleness search a pre-apply guard | `HEAD: ac14ab5` in its own output; the apply had already run |

Error 6 is the most serious, and its repair generalises: the pin table is now
DERIVED from `PAYLOADS`, so a unit that changes its payload set cannot leave
the pins behind. That is the same repair as the derived `docs/sessions/` label
and the derived baseline commit -- **a value stated independently of the thing
it describes goes stale.**

---

## 6. Findings

### Closed
`AUTHORITY-SEARCH-SCOPED-TO-ONE-LANGUAGE-1` -- the consequence is repaired;
the habit is recorded here as error 1, made again in the same unit.

`ARTIFACT-STATES-A-CLAIM-ITS-CORRECTION-REFUTES-1` -- `source_release.py` and
`drift/__init__.py` no longer assert that no registry exists.

### Registered
`PIN-TABLE-NAMES-A-PREVIOUS-UNITS-PAYLOADS-1` -- repaired by derivation.

`SYMBOL-CENSUS-CANNOT-SEE-A-TYPE-CONSUMER-1` -- `source_delta.py` used a field
whose type changed while naming no retired symbol.

### Still open
`ARTIFACT-KEY-INSUFFICIENT-1`, `ARTIFACT-ORIGIN-UNMEASURABLE-FROM-CODE-1`,
`CACHE-KEY-DERIVED-FROM-PATHS-NOT-CONTENT-1`,
`CACHE-KEY-OPAQUE-AND-INCONSISTENT-1` (450,324,943 duplicated bytes),
`GTEX-BUILT-ARTIFACT-EXISTS-AT-TWO-PATHS-1`,
`PROBE-DIRECTORY-LARGELY-UNTESTED-1`.

---

## 7. Ending state

```
HEAD     ac14ab5 = origin/main, tree clean
ratchet  5705
gate     5690 passed, 15 skipped, 0 failed, 0 errors
suite    6937cf8536417101 -> 60c7535c9a4ffeea
```

The vocabulary module went from six declarations to one. No live code names the
retired symbols in any tracked file of any type; the six remaining mentions are
prose in records, a ratchet comment, and the registry test's docstring
explaining what it replaced.

## 8. Next intended action

`SourceRegistry` exists and nothing in the drift package uses it yet.
`SourceEvidenceManifest` accepts any syntactically valid source name, and the
admission check that would refuse an undeclared one is not wired to anything.

That wiring is the next unit, and it needs a measurement first: which
construction sites have a manifest available, and which are pure identity
operations that must stay file-independent.
