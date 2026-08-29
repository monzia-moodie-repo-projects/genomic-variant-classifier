# CORRECTION 2026-08-29 -- a registry existed, and the search missed it

**Author: Monzia Moodie**
**Applies to:** `cffc51f` (the source kernel) and `b67e30f` (the session record)
**Status:** the corrected claim is recorded here; neither commit is amended.

---

## 1. The claim that was wrong

`SESSION_2026-08-28_claims-the-data-disproves.md`, section 3, records under
Phase 1A that source-release identity was the one fact of eight with **no
owner**, and the Phase 1B.2 authority search concluded:

> no source registry exists anywhere in the repository

That is false. `configs/data_manifest.yaml` describes itself on its own third
line:

```
# Canonical registry of every data source under data/. Single source of truth
# for the data-layout standard: the auditor, setup, and sync scripts all read it.
```

It is 407 lines, declares 32 sources, and is read by the auditor, the setup
scripts and the sync scripts. It is a live authority, not documentation.

### Why the search missed it

The Phase 1B.2 probe searched for PYTHON IDENTIFIERS -- `SOURCE_NAMES`,
`SourceName`, `KNOWN_SOURCES`, `SOURCE_REGISTRY`, `CANONICAL_SOURCES`,
`DATA_SOURCES`, `SourceRegistry` -- across tracked `.py` files. A registry
expressed as YAML matched none of those terms and lay outside the file set.

`AUTHORITY-SEARCH-SCOPED-TO-ONE-LANGUAGE-1`. The Layer B rule adopted on
2026-08-27 asks whether an owner exists; it does not say where to look, and I
looked only where an owner would be written in Python.

---

## 2. What the false negative produced

`SourceName` and `_ALIASES` in
`src/genomic_variant_classifier/monitoring/drift/source_vocabulary.py`,
installed at `cffc51f`, were written from a path census on the assumption that
nothing declared this. MEASURED 2026-08-29 against the manifest:

| quantity | manifest | `SourceName` |
|---|---|---|
| declared sources | 32 | 18 |
| sources I cannot name | -- | **16** |
| members not declared anywhere | -- | 2 (`gencode`, `esm2`) |
| declared aliases | 8 | **0 accepted** |
| aliases I invented | -- | 26 |

`resolve_source_name` REFUSES every alias this project actually uses:

```
clinvar      clinvar_fresh
spliceai     spliceai_scores
hgmd         hgmd_pro
dbsnp        dbsnp156
1kgp         1000g, onekg, 1000genomes
clingen      ClinGen-Gene-Disease-Summary
```

My twenty-six aliases -- `ncbi-clinvar`, `splice-ai`, `stringdb` and the rest --
are plausible spellings that appear nowhere in this repository.

### The omissions are not neutral

Four of the sixteen unnameable sources are `irreplaceable` and constrained:

```
tcga               controlled   dbGaP data use agreement; personal cloud likely BREACHES it
topmed             controlled   individual-level access is controlled
rnaseq             review       may be a disease cohort; tier unconfirmed
validation_cohort  review       independence from the label cohort unconfirmed
```

A vocabulary that cannot name `tcga` cannot express a manifest containing it.
`SourceEvidenceManifest` would refuse the record rather than admit a governed
source -- a refusal produced by a missing vocabulary entry, not by a scientific
judgement.

---

## 3. What the manifest already declares

Per source: `location`, `tier`, `class`, `aliases`, `version`, `acquire`,
`regenerate`, `sync`, `notes`.

**`class` is a durability axis** -- `irreplaceable`, `regenerable_expensive`,
`regenerable_cheap`, `public_redownloadable` -- that no type I proposed carries.

**`acquire` and `regenerate` already separate published from derived.** Measured:
29 sources have a non-empty `acquire`; 3 have an empty `acquire` and a non-empty
`regenerate`, and those three sit under a heading that names them:

```
# ---- BUILT ARTIFACTS (live at code-referenced paths; regenerable) ----
reactome_gene_pathways
gtex_gene_expression
rnaseq_gene_expression
```

That is the published-versus-derived union the rulings specify, already in use
as a data convention. `DerivedArtifactLineage` would be a fourth duplicate
authority this session.

**One authority's products are modelled as separate sources when their terms
differ**: `mim2gene` is the free OMIM gene map, declared distinct from the
licensed `omim`. So the GENCODE problem may not need an `ArtifactProductId`
coordinate at all -- this repository already promotes a product to a source
when licence or tier separates it.

---

## 4. Two defects the manifest itself carries

### `MANIFEST-LOCATION-CONTRADICTS-REGENERATE-OUTPUT-1`

`gtex_gene_expression` declares `location: external` while its regenerate
command writes `--out data/processed/gtex/gtex_v11_gene_expression.parquet`.

Both files exist and are BYTE-IDENTICAL -- 1,093,500 bytes, digest
`73985d43c41cc69f`, measured by the lineage census before the manifest was
read:

```
data/external/gtex_gene_expression.parquet
data/processed/gtex/gtex_v11_gene_expression.parquet
```

The duplicate the census could not explain is explained by the declaration that
contradicts itself.

### `MANIFEST-REGENERATE-EMBEDS-A-MACHINE-PATH-1`

```
regenerate: 'python scripts/build_gtex_parquet.py
             --gct "G:/My Drive/genomic-variant-data/external/gtex/
                    GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz"
             --out data/processed/gtex/gtex_v11_gene_expression.parquet'
```

A Google Drive mount hard-coded into a regeneration command. It cannot run on
any machine without `G:` mapped -- the same defect as
`CACHE-KEY-DERIVED-FROM-PATHS-NOT-CONTENT-1`, here in the authority that
declares what is regenerable.

### `MANIFEST-TIER-VOCABULARY-INCOMPLETE-1`

The header declares three tiers -- `public | academic | controlled`. MEASURED:
`review` appears three times, on `rnaseq`, `validation_cohort` and
`rnaseq_gene_expression`. Either the header is stale or the tier is undeclared.

---

## 5. What this corrects, and what it does not

CORRECTED: the claim that no source registry exists. It exists, it is
`configs/data_manifest.yaml`, and it predates the vocabulary that duplicates it
badly.

NOT CORRECTED, because it is not yet measured: whether `SourceName` should be
deleted outright or become a typed reader over the manifest. `SourceEvidenceManifest`
is not yet constructed from real data anywhere, so the defect is LATENT -- no
analysis has been refused because of it. That is the window in which to repair
it, and the repair is a separate unit.

NOT CLAIMED: that `gencode` and `esm2` should be added to the manifest. They are
absent from it and present on disk; whether that is a manifest gap or an estate
gap is unmeasured.

---

## 6. Why this is a correction rather than an amendment

`SESSION_2026-08-28_claims-the-data-disproves.md` is pinned by digest
`646644796711e95d` in the attestation for `b67e30f`. Amending it would break
that binding. The repository's convention -- established by
`CORRECTION_2026-08-21_a-comparison-needs-one-decoding.md` and
`CORRECTION_2026-08-22_a-neutral-transition-was-not-verified.md` -- is that
corrections sit beside records, never inside them.
