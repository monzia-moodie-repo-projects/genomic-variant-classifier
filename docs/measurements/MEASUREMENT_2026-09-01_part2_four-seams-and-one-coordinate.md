# MEASUREMENT 2026-09-01 part 2 -- four seams measured, four rejected, one coordinate found

**Author: Monzia Moodie**
**Measured at:** `4eea19d`
**Status:** MEASUREMENT ONLY. Nothing is built.

---

## 0. Why this exists

`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` needs an attachment point
where a file is ACTUALLY OPENED. `ea84591` measured one candidate and
`4eea19d` corrected the implication drawn from it. This measures three more,
rejects all four on stated evidence, and records the one thing that survives.

---

## 1. Four candidates, four rejections

| candidate | measured | why it fails |
|---|---|---|
| every read site | 93 sites, 53 modules | 93 edits, 93 chances to miss one silently |
| digest-computing loaders | 5 of 36 | two source families; NO principal training input |
| `AnnotationConfig` | 29 `*_path` of 39 fields | `vep_path` is read NOWHERE |
| `BaseConnector` | 17 of 32 subclass it | `_load_cache` opens the CACHE, not the source |

### `vep_path` is the decisive one

MEASURED: of `AnnotationConfig`'s 39 annotated fields, 35 are read inside
`_annotate_scores` and 4 are not. Three of the four -- `kg_path`,
`finngen_path`, `finngen_r13_path` -- appear twice each elsewhere in the file,
so they are read at a different seam. **`vep_path` appears ZERO times
elsewhere.**

And `_annotate_scores` line 942 reads:

```python
vep = VEPConnector()
```

No path argument. The connector cannot accept one, so the field is vestigial.

A manifest built from the config would record VEP as a consumed source when
nothing opens it. The rulings state the failure exactly: "model did not use
GENCODE -> GENCODE does NOT enter the run evidence."
`CONFIG-DECLARES-A-PATH-NOTHING-READS-1`.

### And connector CONSTRUCTION is not evidence either

Most steps are unguarded:

```
 824  dbnsfp   = DbNSFPConnector(dbnsfp_file=ac.dbnsfp_path)
 834  phylop   = PhyloPConnector(phylop_file=ac.phylop_path)
 871  spliceai = SpliceAIConnector(vcf_path=ac.spliceai_path)
1150  lovd     = LOVDConnector(parquet_path=ac.lovd_path)
```

`None` passes straight through and the connector runs in stub mode -- the
file's own docstring says so. Only six of nineteen steps are guarded by a path
check. So the construction line executes whether or not a file is opened,
which is the same defect as instrumenting the config.

### `BaseConnector` is a fetch-and-cache base

`database_connectors.py:112`. Its methods:

```
_get(url, params)     HTTP with retry and backoff
_cache_path(key)      config.cache_dir / f"{source_name}_{key}.parquet"
_load_cache(key)      reads the PARQUET CACHE this project wrote
_save_cache(key, df)  writes it
fetch(**kwargs)       raise NotImplementedError
_to_canonical(df)     column conformance
```

`_load_cache` opens a cache, not a source. `fetch` is abstract, so every
subclass opens its own file its own way. And 15 of the 32 connector classes do
not inherit it at all.

---

## 2. What SURVIVES: `source_name`

Twenty-four connectors declare it, **including classes that do NOT inherit
`BaseConnector`**:

```
alphafold  alphamissense  cadd  clingen  1kgp  cosmic_cmc
gnomad_constraint  clinvar  gnomad  uniprot  dbnsfp  dbsnp
eve  gtex  hgmd  kegg  omim  phylop  reactome  revel
sift_polyphen  spliceai  1000genomes  vep
```

`BaseConnector` declares `source_name: str = "base"` and every subclass
overrides it. The convention is honoured more widely than the inheritance.

That is exactly `SourceArtifactKey.source`. The remaining coordinates --
artifact kind, release, coordinate context, digest -- are NOT on the connector,
and no shared method computes them, which is why only five of thirty-six
compute a digest: each did it locally when it needed one.

### Three names disagree with the manifest

```
connector          manifest
cosmic_cmc         cosmic
1000genomes        1kgp        (declared as an ALIAS of 1kgp)
sift_polyphen      declared nowhere
```

`connector_1kgp.py` uses `1kgp` and `thousandgenomes.py` uses `1000genomes` --
TWO connectors, TWO names, ONE source. `SourceRegistry.canonical_for` exists to
resolve exactly this, and the manifest already declares `1000genomes` as an
alias. That is the admission boundary the rulings place at the construction
point, with a live case waiting for it.

`CONNECTOR-SOURCE-NAMES-DISAGREE-WITH-THE-MANIFEST-1`.

---

## 3. `DATABASE-CONNECTORS-NOT-BYTE-EXACT-BY-TRANSCRIPT-1`

`src/genomic_variant_classifier/data/database_connectors.py`, 583 lines,
sha256 `1cff53a8fb9ed3f2`.

Every other file reconstructed from a console transcript this session was
BYTE-EXACT. This one is not: it contains 18 non-ASCII bytes that did not
survive the console encoding, so a transcript reconstruction differs from the
file.

**Any installer touching this file must NOT use a transcript reconstruction as
its preimage source.** It needs a base-64 capture, or a digest-only preimage
check with the payload authored another way.

Recorded now rather than discovered during an apply.

---

## 4. What this does NOT decide

Where the builder attaches. Four candidates are rejected and no fifth is
proposed here.

Whether `source_name` should be reconciled with the manifest. Three
disagreements are measured; whether `cosmic_cmc` should become `cosmic`, or the
manifest gain an alias, is a governance decision.

Whether `BaseConnector` should acquire an open-the-source method. That would be
a real seam, but it is a refactor of 32 classes and its own unit.

---

## 5. Status

`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` remains OPEN, and now
deliberately: four attachment points have been measured and rejected with
stated reasons rather than left unexamined. No file in the repository is
changed by this record.
