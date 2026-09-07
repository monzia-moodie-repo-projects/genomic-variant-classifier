# MEASUREMENT 2026-09-04 -- one phenomenon of three, and a finding cited for the wrong one

**Author: Monzia Moodie**
**Measured at commit:** b53ffba
**Applies to:** `ARTIFACT-KEY-INSUFFICIENT-1`, recorded at `482c0c9` and cited
in ten records.
**Status of the finding after this measurement:** OPEN, on two phenomena, not
the one the records describe.

---

## 0. Why this was measured

`INCIDENT_2026-08-28_artifact-identity-and-cache-keys.md` reads `Status: OPEN.
Measured, not repaired.` at `b67e30f`. `INCIDENT_2026-08-29_a-source-nobody-
declared.md` reads the same at `95f6c44`. Both govern Phase 1C, and the first
carries a ruling that blocks the production seam:

> **Phase 1C must not persist a source manifest yet.** Persisting
> `(source, artifact_kind)` would convert a known transient model error into a
> migration obligation.

Seventeen commits landed between `95f6c44` and `b53ffba`, several touching this
kernel: `SourceArtifactKey.of` was hardened on 2026-09-01, the v5 evidence
epoch was cut, and `product` exists as a field today. Carrying an eight-day-old
status forward is `STALE-BACKLOG-CARRIED-A-CLOSED-FINDING-1`, corrected twice
on 2026-09-04. So the blocker was re-measured before being allowed to govern.

---

## 1. The incident names THREE phenomena, and the records cite ONE

`INCIDENT_2026-08-28` section 1 is explicit that the fifteen measured
collisions are not one kind of thing:

```
A. several PUBLISHED PRODUCTS        GENCODE's three FASTAs
B. PARTITIONED MEMBERS of one product EVE/csv, 3,212 files, one per protein
C. PROJECT-DERIVED artifacts
   attributed to a PUBLISHER          ClinVar/primary_release, 18 files,
                                      NOT ONE a ClinVar publication
```

Ten records cite `ARTIFACT-KEY-INSUFFICIENT-1`. Every citation reachable from
them describes phenomenon A.

---

## 2. Phenomenon A: RESOLVED, by construction at b53ffba

The incident's own reproduction, run against the live kernel:

```
three GENCODE FASTAs, key (source, artifact_kind)
  canonical keys  [('gencode', 'sequence_fasta', '')]
  REFUSED: artifact key(s) ['gencode/sequence_fasta'] appear more than once

the SAME three, key (source, artifact_kind, product)
  canonical keys  [('gencode', 'sequence_fasta', 'lncRNA_transcripts'),
                   ('gencode', 'sequence_fasta', 'pc_transcripts'),
                   ('gencode', 'sequence_fasta', 'transcripts')]
  ACCEPTED: 3 dependencies, digest 73c9fac3a88fa883
```

The two-field form still refuses exactly as the incident recorded, so this is a
genuine before-and-after rather than a changed test. The state the incident
called *legitimate and unrepresentable* is representable.

`product` did not exist when the incident was written. It exists now and is
exercised 38 times inside `SourceArtifactKey`.

---

## 3. Phenomenon B: UNRESOLVED, and worse -- AMBIGUOUS

```
three EVE partitions as three PRODUCTS      ACCEPTED, 3 dependencies
the whole score set as ONE artifact          ACCEPTED, 1 dependency
```

BOTH are expressible and NOTHING IN THE TYPE SAYS WHICH IS CORRECT. The
incident's warning was that minting 3,212 product identifiers *would be as
wrong as forcing them into one artifact*; the key permits both wrongs and the
right answer indifferently.

A field that accepts every modelling of a distinction has not modelled it.
`PARTITION-AXIS-UNDECIDED-1`.

---

## 4. Phenomenon C: UNRESOLVED, and no field can express it

```
clean_DRIVE + grch38_pathfix as ClinVar products   ACCEPTED, 2 dependencies
```

Those are project-derived parquet files attributed to ClinVar because the PATH
contains the substring. They are accepted as ClinVar publications.

The fields, enumerated from the type rather than read from documentation:

```
SourceArtifactKey  ['source', 'artifact_kind', 'product']
ArtifactKind       primary_release, derived_index, vcf, variant_summary,
                   annotation_gtf, annotation_gff3, sequence_fasta,
                   constraint_table, score_track, network_edges
```

NO FIELD DISTINGUISHES A PUBLISHER'S BYTES FROM THIS PROJECT'S. `derived_index`
is a KIND, not a provenance marker: a publisher may publish an index.

The incident's sentence stands verbatim: *`primary_release` is not a KIND. It
partly encodes PROVENANCE.* `PUBLISHER-VERSUS-PROJECT-AXIS-ABSENT-1`.

---

## 5. What this decides

**The finding stays OPEN**, and the ruling it carries stays in force. Phase 1C
must still not persist a source manifest, because a manifest persisted today
could record `clinvar/primary_release/clean_DRIVE` as a ClinVar publication --
attributing this project's own derived file to a publisher, inside a scientific
reference record. That is exactly the migration obligation the incident refused
to create.

**`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` cannot close until this
does.** Its closure condition requires persistence, and persistence is blocked.
The `persistence -> reload` link built at `bc8b6ce` is a CAPABILITY; measured
at `85d0247`, zero production sites construct any kernel type, so nothing calls
`render()` and no manifest is persisted. The ruling has not been violated.

**The ten citing records describe a case that no longer reproduces.** They are
not wrong about the finding's status -- it is open -- but a reader following
them would investigate the GENCODE collision and find it fixed. That is a
documentary repair, owed to the records, not a design change.

---

## 6. What is NOT claimed

That `product` is the right model for phenomenon A. It REPRESENTS the three
GENCODE products; whether the schema should instead declare them as three
sources -- the form `omim_mim2gene` and `omim_genemap2` already take, cited by
`INCIDENT_2026-08-29` section 4 -- is a schema question this measurement does
not answer.

That the partition axis should be a field. B is ambiguous; whether the repair
is a field, a rule, or a declaration is a design decision. The incident's
closing warning applies: *plausible is what produced the defect being recorded
here.*

That phenomenon C requires a new field. It requires that the distinction be
EXPRESSIBLE somewhere; the key is one candidate location and not necessarily
the right one.

That the other findings in `INCIDENT_2026-08-28` were re-measured.
`ARTIFACT-ORIGIN-UNMEASURABLE-FROM-CODE-1` (3,263 of 3,273 artifacts with no
recoverable origin), `CACHE-KEY-DERIVED-FROM-PATHS-NOT-CONTENT-1` and
`CACHE-KEY-OPAQUE-AND-INCONSISTENT-1` (450,324,943 duplicated bytes) were NOT
re-run here. Their status is whatever the incident recorded on 2026-08-28, and
this measurement establishes nothing about them.

---

## 7. How this was found, which is the reusable part

I designed a P1 producer from the CODE alone -- `provenance/source.py`,
`hashing.py`, `data/source_registry.py`, `configs/data_manifest.yaml` -- and
concluded that the artifact-level declarations were missing and that siting
them was a new architecture decision.

They were not missing from the project's knowledge. They were missing from
mine. Both incidents predate this session, both are tracked, both say OPEN, and
both had already framed the question and ruled on the sequence.

MEASURED 2026-09-04: the records carry 246 DISTINCT FINDING IDENTIFIERS across
358 tracked markdown files. I worked the entire session from a list of
SEVENTEEN carried in session prose. `ARTIFACT-KEY-INSUFFICIENT-1` -- described
in its own incident as blocking Phase 1C -- was in none of them, and appears in
ten records.

This is the fifth time in one session that a question I was investigating had
already been answered in a record I had not opened, after `docs/CARRIED_ITEMS.md`,
the auditor closure at `fd6cd4e`, `configs/data_sources.json`, and
`INCIDENT_2026-08-29`. The register that enumerates all 246 is a probe I ran
myself this morning and read only in summary.

`RECORDS-SEARCHED-LESS-THAN-CODE-1`.
