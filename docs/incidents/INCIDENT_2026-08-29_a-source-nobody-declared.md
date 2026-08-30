# INCIDENT 2026-08-29 -- a source nobody declared, and a validator that cannot find it

**Author: Monzia Moodie**
**Status:** OPEN. Measured, not repaired.
**Measured at:** `95f6c44`
**Severity:** MEDIUM. Nothing is corrupted; a validation gate reports failure on
intact data, and 636 MB of acquired evidence is invisible to every registry.

---

## 0. Why this was measured

`ARTIFACT-KEY-INSUFFICIENT-1`, recorded at `482c0c9`, states that
`SourceArtifactKey(source, artifact_kind)` cannot represent GENCODE release 50,
which publishes three transcript products that collapse to one key. It was
recorded as blocking Phase 1C.

Testing the manifest's own pattern against all four measured collision classes
left exactly ONE genuine product case:

| collision | files | already modelled by |
|---|---|---|
| `ClinVar/vcf` | 2 | `CoordinateContext` -- they differ by ASSEMBLY |
| `ClinVar/primary_release` | 18 | `acquire`/`regenerate` -- they are project-derived |
| `EVE/csv` | 3,212 | nothing; these are PARTITIONS, a different axis |
| `GENCODE/sequence_fasta` | 3 | **nothing. This is the genuine product case.** |

And GENCODE is one of two `SourceName` members declared nowhere in the
manifest's 32 sources. So the question became: is GENCODE a real dependency, or
a directory nothing reads?

---

## 1. `GENCODE-ACQUIRED-VALIDATED-UNDECLARED-1`

MEASURED at `95f6c44`. It is real, and substantial:

```
data/external/gencode/gencode.v50.annotation.gff3.gz        160,590,749 B
data/external/gencode/gencode.v50.annotation.gtf.gz         124,527,720 B
data/external/gencode/gencode.v50.transcripts.fa.gz         183,554,921 B
data/external/gencode/gencode.v50.pc_transcripts.fa.gz      129,953,566 B
data/external/gencode/gencode.v50.lncRNA_transcripts.fa.gz   37,879,655 B
data/external/gencode/GENCODE_v50_manifest.json                   1,869 B
data/external/gencode/GENCODE_v50_validation_report.json          1,393 B
data/external/gencode/release_50_ftp_listing.html                12,233 B
                                                     total  636,522,106 B
```

The publisher's own manifest and the File Transfer Protocol directory listing
were acquired alongside the data. That is an acquisition performed with care.

**Two tracked scripts name the files by path**, `scripts/audit_source_data_assets.py`
lines 21-25 and `scripts/validate_gencode_assets.py` lines 16-20. Both AUDIT or
VALIDATE. Searched across every tracked file of every type: **no module under
`src/` opens a GENCODE file.**

So the state is precise: **ACQUIRED, VALIDATED, NOT CONSUMED, AND DECLARED BY
NO REGISTRY.**

`configs/data_manifest.yaml` -- which calls itself the "Canonical registry of
every data source under data/" -- does not list it. 636 MB sits under `data/`
outside the registry that claims to cover everything under `data/`.

---

## 2. `CONFIG-DECLARES-A-SECOND-PATH-VOCABULARY-1`

`configs/data_sources.json`, 1,881 bytes, 25 lines, read in full. It is NOT a
second source registry: it declares no tier, no class, no aliases, no
provenance. It is a PATH MAP, and every value is an absolute machine path:

```json
"data_root": "G:\\My Drive\\genomic-variant-data",
"gencode":   "G:\\My Drive\\genomic-variant-data\\external\\gencode",
```

Fifteen external entries, all rooted at a Google Drive mount. Its `data_root`
is `genomic-variant-data` -- the OLD Drive root that
`genomic-variant-classifier/data/` supersedes.

**But it is a second NAMING authority, and its names disagree with the
manifest's:**

| `data_sources.json` | `data_manifest.yaml` |
|---|---|
| `gencode` | not declared |
| `omim_mim2gene` | `mim2gene` |
| `omim_genemap2` | `omim` |
| `clingen_gene_validity_csv` | `clingen` |
| `dbsnp_vcf_gz` | `dbsnp` |
| `phyloP100way_bw` | `phylop` |
| `eve_bulk_zip`, `eve_bulk_dir` | `eve` |

The standard, section 3, states: *"Aliases are forbidden: a source has exactly
one canonical name."* These are not declared aliases -- the manifest lists none
of them -- so they are a parallel vocabulary the auditor cannot see.

This carries the same defect as `CACHE-KEY-DERIVED-FROM-PATHS-NOT-CONTENT-1`
and `MANIFEST-REGENERATE-EMBEDS-A-MACHINE-PATH-1`: a machine-specific mount
hard-coded into configuration.

### How it was found, and what that says about the earlier search

My "32 declared sources" figure came from `configs/data_manifest.yaml` ALONE. I
found one registry and stopped looking.

That is the same shape as `AUTHORITY-SEARCH-SCOPED-TO-ONE-LANGUAGE-1`, except
the search SUCCEEDED and was therefore never widened. A successful search is
harder to distrust than a failed one.

---

## 3. `VALIDATOR-CHECKS-A-LOCATION-THE-DATA-LEFT-1`

`scripts/validate_gencode_assets.py`, 54 lines, read in full. It resolves:

```python
data_root = Path(os.environ.get("GENOMIC_DATA_ROOT",
                                r"G:\My Drive\genomic-variant-data"))
base = data_root / "external" / "gencode"
```

MEASURED at `95f6c44`:

```
GENOMIC_DATA_ROOT = 'G:\My Drive\genomic-variant-data'
G:\My Drive\genomic-variant-data\external\gencode          exists: False
C:\Projects\...\data\external\gencode                      exists: True, 636,522,106 B
```

**The environment variable is set, and points at a directory that does not
exist.** The variable and the hard-coded default are the same wrong location,
so the script resolves to the missing path either way.

Run today it emits five `MISSING` lines and exits 2, while all five files sit
intact in the repository data tree.

**The script is otherwise careful**: it checks existence, non-zero size, AND
reads the first megabyte through `gzip.open` -- which catches a truncated
download that a size check cannot. That is the same lesson
`ALIAS-MERGE-VERIFIES-BY-SIZE-NOT-DIGEST-1` repaired at `62d0a33`, already
applied here.

The defect is only the ROOT, and it is a one-line question: where does this
project keep its data now?

### No duplication

The Drive path does not exist, so the 636 MB exists ONCE. Stating this
explicitly because the alternative -- files in both places -- would have added
636 MB to the 450,324,943 duplicated bytes already measured in the cache, and
it is not the case.

---

## 4. What these decide

**The corrected sequence for `ARTIFACT-KEY-INSUFFICIENT-1` gains a step before
the artifact key.** GENCODE must be DECLARED before its product structure can
be modelled, because declaring it forces the product question into the
manifest's own schema -- where `mim2gene` versus `omim` already shows the
established form for one publisher's differently-governed products.

Whether three FASTA products become three source declarations or one
declaration with a product coordinate is a question the schema should answer,
not a type invented ahead of it.

---

## 5. What is NOT claimed

That `configs/data_sources.json` should be deleted. It may be read by something
this measurement did not look for; only its CONTENT has been read, not its
consumers.

That the validator should default to the repository tree. `GENOMIC_DATA_ROOT`
is set to a nonexistent path, and which of the two is wrong -- the variable or
the data's location -- is a decision, not a measurement.

That GENCODE should be consumed. It is acquired and unused; whether that is a
plan not yet executed or an acquisition that should be reclaimed is unknown.

That the 636 MB should be reclaimed. It is a public re-downloadable source, so
losing it costs a download -- but nothing has established that it is unwanted.

---

## 6. Status

OPEN. No file in the repository is changed by these measurements. Every figure
was produced by a read-only probe, and every file cited was read in full.
