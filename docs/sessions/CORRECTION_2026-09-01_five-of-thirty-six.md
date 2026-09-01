# CORRECTION 2026-09-01 -- five of thirty-six

**Author: Monzia Moodie**
**Applies to:** `ea84591`
(`MEASUREMENT_2026-09-01_where-phase-1c-would-attach.md`)
**Status:** the corrected claim is recorded here; `ea84591` is not amended.

---

## 0. What went wrong

`ea84591`, section 2, states:

> FOUR loaders already compute a source digest. [...] The expensive part --
> streaming a multi-gigabyte file to compute a digest -- is ALREADY DONE in
> these paths. A builder does not need a new seam there; it needs to receive
> what they already have.

The four are real. **They are also the exception, and I presented them as the
pattern.**

The same record, section 6, says the gap explicitly -- "whether the four
digest-computing loaders cover the sources a run actually consumes is
unmeasured" -- and I wrote a design implication on top of the unmeasured half
in the same document.

---

## 1. MEASURED 2026-09-01 at `ea84591`: five of thirty-six

Every tracked module under `src/genomic_variant_classifier/data/` was searched
for `sha256_file`, `source_sha256` or `compute_sha256`, and separately for a
reader call:

```
COMPUTE a source digest    5
    connector_gnomad_constraint.py
    constraint_canonicalize.py
    phylop.py
    phylop_cache.py
    phylop_ingest.py

READ but do NOT           31
    alphafold, alphamissense, clingen, connector_1kgp, connector_cosmic,
    database_connectors, dbnsfp, dbsnp, esm2, etl_polars, eve, finngen,
    genomic_lm, gtex, hgmd, kegg, kg_edges, lovd, omim, pipeline,
    primateai3d, protein_coords, reactome, real_data_prep, revel, rnaseq,
    seq_window_join, seq_window_manifest, sift_polyphen, spliceai,
    thousandgenomes
```

The five are TWO source families -- phyloP and gnomAD constraint -- not five
independent adopters of a convention.

**And every source the training frame actually joins is in the thirty-one.**
`real_data_prep.py` calls `_load_and_label(clinvar_path)`, `_join_spliceai`,
`_join_alphamissense`, `_join_uniprot`, and runs `DbNSFPConnector`,
`LOVDConnector`, `ESM2Connector` and thirteen more numbered steps. None of
those computes a source digest.

So "receive what they already have" describes **five of thirty-six modules and
none of the principal training inputs**.

---

## 2. What the same measurement found that is more useful

`AnnotationConfig` in `real_data_prep.py` declares **twenty-nine `*_path`
fields**, one per source:

```
dbnsfp_path, phylop_path, spliceai_path, alphamissense_path,
alphamissense_tsv_path, gtex_path, vep_path, omim_path,
omim_genemap2_path, clingen_path, dbsnp_path, eve_path,
eve_entry_map_path, hgmd_path, kg_path, finngen_path,
finngen_r13_path, lovd_path, esm2_cache_path,
esm2_uniprot_index_path, genomiclm_cache_path,
genomiclm_seq_windows_path, cosmic_path, kegg_path,
gnomad_constraint_path, reactome_path, alphafold_path,
alphafold_uniprot_index_path, rnaseq_path
```

Its own docstring: *"Paths and flags controlling which score connectors run
during DataPrepPipeline. All paths default to None -> connector runs in stub
mode."*

That is ONE DECLARATIVE OBJECT naming every source a run consumes, with
absence already meaning "this run does not use it". Ninety-three call sites is
too many to instrument; five cooperative loaders is too few to be
representative; twenty-nine declared fields in one config is neither.

**This is a CANDIDATE, not a decision.** `AnnotationConfig` and
`_annotate_scores` have not been read in full, and a config field is a
DECLARATION that a path was offered -- not evidence that a file was OPENED. The
rulings are explicit that the manifest must be execution-derived, and
"declared in a config" is closer to inventory than to provenance.

---

## 3. What stands unchanged in `ea84591`

Ninety-three read sites across fifty-three modules, judged by argument.

`phylop.py:536` stating the path-exclusion principle independently, and lines
531-534 recording what `CACHEIDENTITY-1` cost.

`FILE-DIGEST-HELPER-DEFINED-THREE-TIMES-1`, measured by execution: all three
produce the identical digest.

`ScienceClawLedger` as a third provenance system, read in full and set aside.

---

## 4. And one open question is now CLOSED

`ea84591` section 5 says "`domain_digest` in `_digest.py` solves the same
problem; WHETHER THEY AGREE IS UNMEASURED."

`_digest.py` read in full, 82 lines. Line 58:

```python
json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
```

The ledger's `_row_hash` uses `json.dumps(..., sort_keys=True,
separators=(",", ":"))`. **Identical canonical serialisation.**

They differ in what precedes it. `domain_digest` prepends
`genomic-variant-classifier:` plus a version-suffixed domain plus a NUL byte,
and REFUSES a domain carrying no version:

> A digest whose domain never changes cannot express a schema change, and
> records of two incompatible shapes would compare across it.

That refusal is what made `v3 -> v4` meaningful at `4bed1b8`. The ledger has no
domain at all, because it identifies rows within one log rather than kinds
across a system.

Compatible in encoding, deliberately incomparable in output. **Nothing to
reconcile**, and `ensure_ascii=True` is the one detail the ledger omits -- so a
non-ASCII field would serialise differently between them. Neither currently
carries one.

---

## 5. Why this is a correction and not an amendment

`MEASUREMENT_2026-09-01_where-phase-1c-would-attach.md` is pinned by digest
`f05353ef476f1930` in the attestation for `ea84591`. Amending it would break
that binding.

The measurement it records is correct. The DESIGN IMPLICATION drawn from it was
written before the coverage ratio was measured, in the same document that named
that ratio as unmeasured.

---

## 6. Status

`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` remains open. No file in the
repository is changed by this record.

The next measurement is `AnnotationConfig` and `_annotate_scores` in full: how
many of the twenty-nine declared paths are actually opened in a run, and
whether a declared path that is never read would enter a manifest built from
the config -- which would be inventory wearing provenance's clothes.
