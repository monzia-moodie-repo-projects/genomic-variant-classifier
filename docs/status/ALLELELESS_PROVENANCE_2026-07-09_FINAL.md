# Allele-less Provenance and Cohort v3 Build — FINAL Status

**Date:** 2026-07-09
**Author of record:** GenAssoc engineering session (Monzia Moodie, sole developer)
**Supersedes:** the earlier `ALLELELESS_PROVENANCE_2026-07-09.md` written mid-arc, whose
recovery-oriented conclusion was overturned by the structural-variant finding documented in
§4.4 below. Retain the earlier file for lineage; this file is authoritative.
**Cohort touched:** none destructively. Canonical v2 remained intact throughout.
**Outcome:** Cohort v3 built and independently verified. The allele-less investigation is closed.

---

## 1. One-paragraph summary

The 19,988 allele-less (`na:na`) rows in the canonical cohort were investigated to determine
whether their missing alleles could be recovered from ClinVar. The investigation established,
with data at every step, that these rows are **correctly allele-less**: 19,976 are structural
variants (copy-number gains and losses, large deletions and duplications, microsatellite
repeat expansions, translocations, and inversions) that ClinVar represents in `variant_summary`
but deliberately omits from the allele-level Variant Call Format (VCF) file because they have
no simple reference/alternate allele, and 12 are single-nucleotide variants whose ClinVar
Variation Identifiers are absent from the VCF (withdrawn, re-versioned, or transcript-only
records). A prior recovery attempt that claimed to recover 544 alleles was found to be
spurious: it resolved each structural variant to a co-located single-nucleotide variant's
Variation Identifier by a positional lookup and borrowed that unrelated variant's point
allele. Cohort v3 was rebuilt by excluding all 19,988 allele-less rows on a fully
per-row-justified basis, yielding 4,400,192 rows, and was independently verified to be exactly
the canonical v2 with only those rows removed.

---

## 2. Definitions (first-use expansions)

- **ClinVar** — the National Center for Biotechnology Information's public archive of human
  genetic variants and their clinical significance.
- **VCF (Variant Call Format)** — a text file listing variants by chromosome, position,
  reference allele, and alternate allele. ClinVar's VCF contains variants that have a simple
  reference/alternate representation.
- **variant_summary** — a ClinVar tab-delimited table listing every ClinVar variant, including
  large structural variants that do not appear in the VCF.
- **VariationID** — ClinVar's stable per-variant identifier; carried in the cohort's
  `source_id` column.
- **CNV (copy number variant)** — a structural gain or loss of a genome segment, typically
  kilobases to megabases, with no single reference/alternate base pair.
- **SNV (single nucleotide variant)** — a single base-pair substitution, which does have a
  simple reference/alternate allele.
- **Allele-less / `na:na`** — a cohort row whose reference and alternate allele fields are both
  null (`na`).
- **GRCh38** — the reference human genome assembly the cohort is built on.

---

## 3. The canonical cohort (unchanged throughout)

- Path: `data/processed/clinvar_grch38_clean_v2_verified.parquet`
- MD5 checksum: `F3152F671E920A0A0C19A696563002E0`
- Rows: 4,420,180; columns: 16, including `variant_id`, `source_id`, `chrom`, `pos`, `ref`,
  `alt`, `pathogenicity`, and a `metadata` column holding Python dictionaries.
- Allele-less rows within it: 19,988.

Every diagnostic and build read the canonical cohort without modifying it. The MD5 was
re-checked at the start of the rebuild and matched, confirming no drift.

---

## 4. How the investigation unfolded (chronological, 2026-07-09)

### 4.1 The rebuild guard aborted, twice, for two different real reasons

The first attempt to build v3 by merging previously "recovered" alleles aborted at the
independent genome re-verification guard: 39 recoveries failed to match the reference genome at
their stated coordinate. This traced to a coordinate-basis conflict in the recovery table,
which stored the cohort's placeholder position alongside a reference/alternate pair captured at
the VCF's own (sometimes offset) position. A genuine coordinate-provenance defect, correctly
caught by the guard.

After that was addressed, the build aborted again at the set-reconciliation guard: kept plus
removed rows did not equal the allele-less total. The arithmetic was the clue — the number of
unique `variant_id` values among the allele-less rows (15,771) was smaller than the row count
(19,988).

### 4.2 The key-degeneracy discovery

The cohort's `variant_id` for allele-less rows is a fabricated placeholder of the form
`clinvar:CHROMOSOME:POSITION:None:None`. Because it encodes only locus, not variant identity,
**it collapses genuinely distinct co-located variants into one key**: 19,988 allele-less rows
share only 15,771 distinct `variant_id` values, leaving 4,217 duplicate-keyed rows in 1,915
collision groups. One group at `clinvar:22:18339130:None:None` held 91 distinct variants
(22q11/DiGeorge-region copy-number variants of differing extent); several groups mixed opposite
clinical categories (benign through pathogenic) under a single key. Merging or labeling by
`variant_id` would attach one variant's allele and label to distinct neighbors.

The same diagnostic identified the correct key. The `source_id` column is the ClinVar
VariationID (all sampled values matched a VariationID at the row's exact locus in
`variant_summary`), unique per row except for 14 identifiers mapping to two loci each
(pseudoautosomal-region genes on both X and Y, plus a few multi-mapping variants). The unique
per-row key is the triple `(source_id, chrom, pos)`, later confirmed on the real data to be
perfectly unique across all 19,988 rows.

### 4.3 The re-key, and the collapse audit

The recovery and rebuild tools were re-keyed onto the triple. Auditing the old recovery table
showed its 544 rows collapsed to only 389 distinct resolved identifiers, and a decisive probe
showed that for **all** 544 rows the resolved identifier differed from the row's own
`source_id`. The old recovery used a positional lookup that returned whichever VariationID
appeared first at the locus, then fetched and genome-verified that neighbor's allele. The
alleles verified against the genome — because they are real alleles at that position — but they
belonged to a different variant than the allele-less row.

### 4.4 The decisive finding: the rows are structural variants

Recovering strictly by each row's own `source_id` returned zero recoveries, because none of the
allele-less `source_id` values are present in the ClinVar VCF at all. A direct probe ruled out
a key-format bug (identifiers were clean, genuinely different numbers) and resolved the question
via the `variant_summary` Type field. Across all 19,988 allele-less rows, 19,976 are structural
types and only 12 are single-nucleotide variants:

| ClinVar Type | Count |
|---|---|
| copy number gain | 7,269 |
| copy number loss | 6,968 |
| Deletion | 3,788 |
| Duplication | 1,545 |
| Insertion | 144 |
| Indel | 140 |
| Microsatellite | 76 |
| Translocation | 22 |
| single nucleotide variant | 12 |
| Inversion | 10 |
| Complex | 10 |
| Variation | 4 |

The structural variants have no simple reference/alternate allele and are absent from the VCF
by ClinVar's design; they are genuinely allele-less. The 12 single-nucleotide variants were
examined individually and are all absent from both the raw and fresh VCF, with alleles present
only as coding HGVS in their ClinVar Name (strand-relative and deliberately not hand-parsed
into genomic alleles).

**Conclusion:** the entire recovery premise was mistaken. The allele-less rows are supposed to
be allele-less. The prior "544 recoveries" were spurious neighbor borrowing that, had it been
merged, would have injected wrong single-base alleles onto megabase structural variants — a
silent, systematic label corruption. Every guard abort across the arc was correct.

---

## 5. The final disposition and the v3 build

### 5.1 Type-aware classification

`classify_alleleless_by_type.py` assigns every allele-less row a per-row verdict, reason, and
ClinVar Type:

| Verdict | Count | Basis |
|---|---|---|
| `CONFIRMED_ALLELELESS_SV` | 19,976 | structural ClinVar Type with no simple allele |
| `CONFIRMED_ALLELELESS_SNV_NOT_IN_VCF` | 12 | single-nucleotide variant absent from both VCFs |
| `RECOVER_BY_SID_*` | 0 | none of the allele-less identifiers are in the VCF |

The tool retains a genuine recovery path (an SNV identifier present in the VCF with a
genome-verified allele would be recovered by its own `source_id`); it yields zero for the
current data but is correct and future-proof if ClinVar later adds these identifiers.

### 5.2 Rebuild

`rebuild_cohort_v3_by_sid.py` merges recoveries by the unique triple `(source_id, chrom, pos)`
and excludes every non-recovered allele-less row. With zero recoveries, it excluded all 19,988.
The full fail-loud guard chain ran (canonical MD5 check, refuse-overwrite, genome
re-verification, subset, per-row reconciliation, collision, duplicate-identifier,
zero-`na:na`-remaining).

- v2: 4,420,180 rows, MD5 `F3152F671E920A0A0C19A696563002E0` (unchanged)
- v3: `data/processed/clinvar_grch38_clean_v3_verified.parquet`, 4,400,192 rows,
  MD5 `5871AE9C0E18192FC49C2B7E97776114`
- Reconciliation: kept 0 + excluded 19,988 = 19,988; `na:na` remaining = 0;
  `reconciliation_ok = true`.
- Provenance artifacts: `data/processed/cohort_v3_reconciliation.json` and
  `data/processed/cohort_v3_excluded_alleleless.tsv` (per-row Type and reason for all 19,988).

### 5.3 Independent verification

`verify_v3_against_v2.py` read both parquet files and confirmed at the row level, not merely by
count: v3 has zero `na:na` rows; v3 row count equals v2 minus the allele-less rows (4,400,192);
all 4,400,192 non-allele-less v2 rows are present unchanged in v3 (keyed on source_id,
chromosome, position, reference, alternate), zero missing; zero unexpected extra rows; no
duplicate `variant_id`; schema identical to v2.

---

## 6. Standing lessons

1. **Identity by locus is not identity by variant.** Two distinct ClinVar variants routinely
   share a start position; a key built from position alone is degenerate. The correct per-row
   key is `(source_id, chrom, pos)`, where `source_id` is the true ClinVar VariationID already
   carried in the cohort.
2. **The guard-first discipline works.** Every abort surfaced a real defect (coordinate basis,
   key degeneracy, wrong-variant alleles). A build that "looks correct" is not correct until
   independent re-verification confirms it.
3. **A correction to a correction is often itself wrong on the first pass.** The way out is a
   data-derived control, never more reasoning. Two intermediate figures produced during this
   arc (an over-broad "1,816 recovered in collision groups" count and a misleading "544
   not-a-source_id" count) were themselves diagnostic artifacts, caught by re-scoping against
   the data rather than by argument.
4. **Do not fabricate data.** The 12 single-nucleotide variants' alleles exist only as coding
   HGVS; reconstructing genomic alleles from them is strand-dependent and error-prone, and was
   declined in favor of an authoritative-source-only policy.

---

## 7. Open item (next action)

The upstream cause remains to be fixed: `scripts/clean_cohort.py` routed these 19,988
structural variants into the main cohort as `na:na` rather than to the structural-variant
table. The fix should route allele-less rows by ClinVar Type (structural types to the
structural table) rather than leaving them allele-less in the main cohort or dropping them.
Building this fix correctly requires the actual source of `scripts/clean_cohort.py` and
`src/genomic_variant_classifier/data/allele_classify.py`, to be reviewed before any change,
followed by a regression test against the real files.

---

## 8. Artifacts produced this session

**PATHS UPDATED 2026-07-12** — these scripts were untracked for weeks and were dispositioned
on 2026-07-12 (they blocked the G1 pre-flight gate). They now live in **two** places, and the
split is not arbitrary: anything reachable from `tests/` or `src/` stayed in `scripts/`;
everything else was archived. See `scripts/forensics/README.md`.

Live tooling, still in **`scripts/`** (a test imports each of these):
`classify_alleleless_by_type.py`, `rebuild_cohort_v3_by_sid.py`, `recover_by_sourceid.py`.

Spent diagnostics, archived to **`scripts/forensics/`** (they produced the findings recorded
above; they are kept so this document remains reproducible, not because they are maintained):
`verify_v3_against_v2.py`, `diagnose_alleleless_keys_v3.py`, `diagnose_collision_groups_v2.py`,
`audit_recovery_collapse.py`, `probe_resolve_vs_sourceid.py`, `probe_sid_in_vcf.py`,
`probe_snv_alleleless.py`.

Tests (in `tests/`): `test_classify_alleleless_by_type.py`, `test_rebuild_cohort_v3_by_sid.py`,
`test_recover_by_sourceid.py`. Full suite green (45 tests) as of 2026-07-09.

Data (in `data/processed/`): `clinvar_grch38_clean_v3_verified.parquet`
(MD5 `5871AE9C0E18192FC49C2B7E97776114`), `cohort_v3_reconciliation.json`,
`cohort_v3_excluded_alleleless.tsv`.
