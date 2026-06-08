# Run-15 split feature-health audit (BEFORE baseline)

**Date:** 2026-06-08
**Tool:** `scripts/audit_split_feature_health.py` (read-only)
**Splits:** `outputs/run15_rerun_report/full/splits/` — 9 parquets
(`X_/meta_/y_` x train/val/test; train 1,038,974 / val 146,329 / test 304,711 rows;
X = 78 cols, meta = 58 cols, y = 1 col).
**Result:** 96 distinct columns; **54 healthy, 42 degenerate.**
**CSV:** `split_health_before.csv`.

## Key context: these splits are STALE
git-log on `src/genomic_variant_classifier/data` since the splits' mtime
(2026-06-06 04:45 UTC) shows both:
- `5a4d103` ESM-2 batched + device path
- `8696ede` protein-coords step + HGVSp parser

i.e. the splits were built **before** protein coords / HGVSp parsing / the ESM-2
device path existed. So a regen now is justified on staleness alone.

## The 42 degenerate columns, by ROOT CAUSE

### A. Fixed by this regen (1)
- `esm2_delta_norm` (ALL_ZERO) — silent-zero because ESM-2 had no protein coords.
  Device path + prefetched sequence cache (18,621 genes) + coords step remove the blocker.

### B. Null prerequisites the regen now populates (3)
- `protein_change`, `transcript_id`, `fasta_seq` (ALL_NULL) — populated by the
  coords/parser step that postdates these splits. VERIFY non-null after regen.

### C. Legitimately (near-)constant — NOT bugs, do not "fix" (2)
- `is_mitochondrial` NEAR_CONSTANT(0.9994) — true ~0.06% MT rate.
- `source_db` CONSTANT — single-source cohort.

### D. Data-source-blocked => Phase-2 database backlog (~36) — regen recomputes to ZERO again
- 1KGP: `af_1kg_afr/amr/eas/eur/sas`
- dbSNP: `dbsnp_af`
- GTEx: `gtex_is_eqtl`, `gtex_max_abs_effect`, `gtex_max_tpm`, `gtex_min_eqtl_pval`,
  `gtex_n_tissues_expressed`, `gtex_tissue_specificity`
- PhyloP: `phylop_score`
- FinnGen: `finngen_af_fin`, `finngen_af_nfsee`, `finngen_enrichment`
- OMIM: `omim_is_autosomal_dominant`, `omim_n_diseases`
- ClinGen: `clingen_validity_score`
- HGMD (procurement-blocked): `hgmd_is_disease_mutation`, `hgmd_n_reports`
- EVE (score files absent): `eve_score`
- MaxEntScan: `maxentscan_score`
- Protein structure: `alphafold_plddt`, `dist_to_active_site`,
  `solvent_accessibility`, `secondary_structure_context`, `has_uniprot_annotation`
- VEP codon/exon: `codon_position`, `exon_number`
- Splice distance: `dist_to_splice_site`, `is_canonical_splice`
- LOVD: `lovd_variant_class`
- Other: `n_known_pathogenic_protein_variants`

### ANOMALY to investigate before regen (not in any bucket above with confidence)
- `gene_constraint_oe`, `gene_is_constrained` (ALL_ZERO) while `loeuf` + `pli_score`
  (SAME gnomAD-constraint source) are HEALTHY. Suggests a field-mapping/merge gap
  in the constraint connector, NOT a missing source. Inspect the connector's
  output keys vs the engineered column names.

## Corrected prior belief
- `gnn_score` is **HEALTHY** here. The Run-14 all-zero merge-back issue is NOT
  present in these Run-15 splits.

## Healthy signal-carriers (54, partial)
cadd_phred, cadd_high, loeuf, pli_score, gerp_score, sift_score, sift_deleterious,
revel_score, revel_pathogenic, alphamissense_score, splice_ai_score, is_splice,
gnn_score, n_pathogenic_in_gene, consequence_severity, ...

## Decision implied
"Full re-annotation" now = **ESM-2 activation + protein coords + rebuilt splits**;
cost is ESM-2-forward-dominated (every other connector is a cheap lookup). The ~36
data-blocked columns are the Phase-2 database-wiring backlog and are expected to
remain zero until their sources are connected — they are NOT regressions.

## Tooling caveat (cosmetic)
The audit's "watched features present?" convenience scan uses substring matching,
so `consequence_severity` and `revel_*` spuriously match the `eve` tag. The
degeneracy detection itself is exact and unaffected.

## AFTER check
Re-run `scripts/audit_split_feature_health.py` on the regenerated splits:
`esm2_delta_norm` + the three ALL_NULL prerequisites must flip to healthy; the 54
healthy columns must stay healthy; the ~36 data-blocked stay zero (expected).
