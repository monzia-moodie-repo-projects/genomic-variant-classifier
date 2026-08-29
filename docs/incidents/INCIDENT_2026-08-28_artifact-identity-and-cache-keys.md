# INCIDENT 2026-08-28 -- artifact identity is insufficient, and the cache proves why

**Author: Monzia Moodie**
**Status:** OPEN. Measured, not repaired.
**Measured at:** `b67e30f`
**Severity:** HIGH for Phase 1C. The next persisted scientific reference format
would freeze a model already known to be incomplete.

---

## 0. What this records

Three independent measurements taken on 2026-08-28, after `cffc51f` landed the
Phase 1B.3 source kernel. Each falsifies part of the committed artifact
identity model. None is repaired here.

| finding | evidence |
|---|---|
| `(source, artifact_kind)` cannot represent the artifact estate | 15 colliding groups across 12 of 17 authorities |
| artifact origin is unknown for almost the whole estate | 4 of 3,273 classified from code |
| cache keys encode filesystem locations | 4 keys, one naming a vanished temporary directory |
| cache keys are opaque AND inconsistent | 450,324,943 bytes duplicated |

---

## 1. `ARTIFACT-KEY-INSUFFICIENT-1`

`SourceArtifactKey(source, artifact_kind)` was installed at `cffc51f` after
measurement proved `source` alone too coarse. It is too coarse for the SAME
REASON and against the SAME EVIDENCE.

GENCODE release 50 publishes three distinct transcript products:

```
gencode.v50.transcripts.fa.gz          all transcripts
gencode.v50.pc_transcripts.fa.gz       protein-coding transcripts
gencode.v50.lncRNA_transcripts.fa.gz   long non-coding RNA transcripts
```

All three become `SourceArtifactKey(GENCODE, SEQUENCE_FASTA)`, and
`SourceEvidenceManifest` refuses the duplicate:

```
artifact key(s) ['GENCODE/sequence_fasta'] appear more than once
```

**The state is legitimate and unrepresentable.** Reproduced against the
committed code before this was written.

### The census that found it also contained the counterexample earlier

The topology census of 2026-08-28 reported "GENCODE 3 kinds, 5 files". I read
that line, used it to justify `(source, artifact_kind)`, and did not ask why
five files occupy three kinds. The number that falsified the design was in the
measurement that motivated it.

### Three phenomena, not one

The product census found 15 collisions, and they are NOT the same kind of
thing:

**A. Several published products.** GENCODE's three FASTAs; ClinVar's `.vcf.gz`
and `_GRCh38.vcf.gz`. A product coordinate is genuinely missing.

**B. Partitioned members of one product.** `EVE/csv` holds 3,212 files --
`1433G_HUMAN.csv`, `BRCA1_HUMAN.csv`, one per protein. These are members of
ONE published score set. Minting 3,212 product identifiers would be as wrong
as forcing them into one artifact.

**C. Project-derived artifacts attributed to a publisher.**
`ClinVar/primary_release` holds 18 files and NOT ONE is a ClinVar publication:
`clean_DRIVE`, `clean_REGRESSED_17col_2026-07-08`, `grch38_pathfix`,
`grch38_alleleless_quarantine`, `smoke3000`. They were attributed to ClinVar
because the PATH contains the substring, and called `primary_release` because
the name ends in `.parquet`.

**`primary_release` is not a KIND. It partly encodes PROVENANCE.**

---

## 2. `ARTIFACT-ORIGIN-UNMEASURABLE-FROM-CODE-1`

A lineage census parsed all 1,037 tracked Python files, extracting path
literals and their enclosing call to distinguish creator sites from consumer
sites. Of 3,273 artifacts examined:

```
unknown          unknown      3263
test_fixture     candidate       4
cache_or_index   measured        2
project_derived  measured        2
quarantine       candidate       1
unknown          candidate       1

classified from CODE rather than naming    4 of 3273
PUBLISHER_BYTES asserted                   0
```

The invariant HELD: no filename promoted an artifact to publisher identity.

But **3,263 artifacts have no creator site, no consumer site, and no naming
evidence.** Their origin is not merely unrecorded; it is not recoverable from
the tracked code. They were produced by scripts that no longer exist, by code
constructing paths dynamically, or outside the repository.

This is a measurement, not a probe failure. It bounds what any provenance
model can claim about the existing estate.

---

## 3. `CACHE-KEY-DERIVED-FROM-PATHS-NOT-CONTENT-1`

Four of 41 cache keys embed a filesystem location:

```
omim_gene_table_genemap2=C_/Users/monzi/AppData/Local/Temp/tmpub9ux6hp/genemap2.txt.parquet
omim_gene_table_genemap2=data/external/omim/genemap2.txt.parquet
omim_gene_table_mim2gene=data/external/omim/mim2gene.txt_genemap2=None.parquet
omim_gene_table_mim2gene=G_/My Drive/genomic-variant-data/external/omim/mim2gene.txt
    _genemap2=G_/My Drive/genomic-variant-data/external/omim/genemap2.txt.parquet
```

One is keyed on a **temporary directory** that no longer exists and never will
again. One on a **Google Drive mount**, where `G_` is a sanitised drive letter.

Because a filename cannot contain a path separator, these are NESTED DIRECTORY
TREES: the cache created directories mirroring its inputs. Maximum depth 10.

Two of them hold **2,849 byte-identical bytes** -- `genemap2=None` and
`genemap2=.../genemap2.txt` produced the same result, so the second input did
not affect the output while it did affect the key.

### How this was found

The probe's own test could not WRITE its fixture: a filename cannot contain
`/`. That failure -- easy to dismiss as a harness quirk -- revealed that the
entries are directory trees, and that analysing the FILENAME would have
reported ZERO located keys while the defect was the directory structure.

---

## 4. `CACHE-KEY-OPAQUE-AND-INCONSISTENT-1`

37 of 41 keys carry an opaque hexadecimal suffix that cannot be inspected. The
`eve_eve_lookup` family fails in BOTH DIRECTIONS:

```
eve_eve_lookup_5c147a0fdaba4b1d   444,367,755 B   91fdcc46e6a50e99  | IDENTICAL
eve_eve_lookup_d24b674cd7ff366d   444,367,755 B   91fdcc46e6a50e99  | bytes

eve_eve_lookup_358ac572e67df34e   143,457 B       894932b8a590bac9  | DIFFERENT
eve_eve_lookup_f31f490df5a2d0f0   143,457 B       6f62f7e0e0a01aa2  | bytes
```

It **duplicates identical results and separates genuinely different ones**, in
one family. The key digests something the result partly depends on and partly
does not, and nothing in the name can say which.

### Duplicated bytes, accounted for exactly

```
        2,849 B  omim, two path-derived keys
        3,780 B  finngen_full_index vs finngen_r13_full_index
       20,130 B  dbsnp_af_lookup, two opaque keys
    5,930,429 B  1000genomes_kg_pop_af, two opaque keys
  444,367,755 B  eve_eve_lookup, two opaque keys
  ------------
  450,324,943 B  TOTAL
```

`finngen_full_index` and `finngen_r13_full_index` hold identical bytes under
names implying different releases. Whether one is stale is NOT established
here.

### Equal size is not equal content, demonstrated

Three equal-size groups have DIFFERENT digests, including
`TPIS_HUMAN.csv` and `TSHB_HUMAN.csv` at exactly 612,501 bytes each. Inferring
duplication from size would have been wrong in three of eleven cases.

---

## 5. What these findings decide

**`DerivedArtifactLineage` must NOT be modelled on the existing cache key.**
Neither form is fit: located keys are machine-specific and one references a
vanished temporary directory; opaque keys are unverifiable and demonstrably
inconsistent.

Derived identity must come from transformation identity plus domain-separated
parent identities plus logical output identity -- never from path names. This
incident is the evidence that the alternative has already failed in production,
in two different directions.

**Phase 1C must not persist a source manifest yet.** Persisting
`(source, artifact_kind)` would convert a known transient model error into a
migration obligation.

---

## 6. What is NOT claimed

- That the 3,263 unclassified artifacts are publisher bytes. Their origin is
  UNKNOWN, and UNKNOWN is the result.
- That any duplicate should be deleted. Byte equality is not redundant
  semantics; a deliberate copy and an accidental one look identical.
- That `finngen_full_index` is stale. Only that it is byte-identical to
  `finngen_r13_full_index`.
- That the product vocabulary should be `transcripts/all`,
  `transcripts/protein_coding`, `transcripts/lncrna`. Those are plausible, and
  plausible is what produced the defect being recorded here. Product identity
  must be derived from publisher semantics, not from varying filename
  substrings.

---

## 7. Status

OPEN. No file in the repository is changed by these measurements. Every figure
above was produced by a read-only probe whose extractors carry fixtures, and
each probe's self-check caught at least one defect in itself before it ran.
