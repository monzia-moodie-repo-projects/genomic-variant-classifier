---
document: RUN_11_FINDINGS_AND_CRITICAL_FIXES.md
date: 2026-05-24
based_on: data_quality_audit.csv (Run 10b splits, 1,700,687 variants × 78 features)
---

# Run 11 Findings: Data-Quality Audit + Code Review

## Finding 1: 38 of 78 features are 100% dead — but the count is misleading

The audit reports 38 DEAD features. However, the audit checks `(series == 0)`,
which is correct for post-scaled data. After `StandardScaler`, any constant-
value feature (whether constant-0.0 or constant-1.0 or constant-15.0) becomes
0.0 because `(x - mean) / std = 0 / 0 → 0`. So the 38 "DEAD" features are
features that are **constant across all 1.7M variants** — they contribute
literally zero information to any model.

### The 38 dead features by root cause:

**A. Connector never wired (6 features, 3 connectors — all fixable by passing CLI args):**
- `finngen_af_fin`, `finngen_af_nfsee`, `finngen_enrichment` → FinnGen (no `--finngen-path`)
- `pli_score`, `loeuf`, `syn_z`, `mis_z` → gnomAD constraint (no `--gnomad-constraint`)

Wait — that's 7, and gnomAD constraint was **top-5 in Runs 7-8** (loeuf #2,
syn_z #3, mis_z #4, pli_score #6). It is SILENT-ZERO in Run 10b because
`launch_run10b_skip_kan_v2.sh` never passes `--gnomad-constraint`.

**B. Connector wired but data path not on disk (varies):**
- `phylop_score` → PhyloP (no parquet on Vast.ai)
- `eve_score` → EVE (stub — needs HGVSp parser)
- `esm2_delta_norm` → ESM-2 (stub — needs HGVSp parser)
- `gene_constraint_oe`, `gene_is_constrained` → Old gnomAD constraint columns
  (these are from the Phase 1 `engineer_features()` path, separate from the
  Phase 3C `GnomADConstraintConnector`)

**C. Connector produces zeros because feature depends on upstream data not populated:**
- `codon_position` → VEP (needs VEP annotation run)
- `dbsnp_af` → dbSNP (no parquet)
- `omim_n_diseases`, `omim_is_autosomal_dominant` → OMIM (no path)
- `clingen_validity_score` → ClinGen (no path)
- `hgmd_is_disease_mutation`, `hgmd_n_reports` → HGMD (no Pro license)
- `has_uniprot_annotation`, `n_known_pathogenic_protein_variants` → UniProt
  (not populated by current pipeline — `_join_uniprot()` is never called
  because `uniprot_path` is not passed to `run()`)

**D. Pipeline stage never runs on Vast.ai:**
- `gnn_score` → GNN (STRING DB files present but GNN training failed)
- `maxentscan_score`, `dist_to_splice_site`, `exon_number`, `is_canonical_splice`
  → RNA splice pipeline (stub defaults)
- `alphafold_plddt`, `solvent_accessibility`, `secondary_structure_context`,
  `dist_to_active_site` → Protein structure pipeline (stub defaults)
- `af_1kg_afr/eur/eas/sas/amr` → 1KGP (no `--kg` path)
- `gtex_is_eqtl`, `gtex_min_eqtl_pval`, `gtex_max_abs_effect` → GTEx (no genes)

### What this means for Run 11:

The Run 10b TEST AUROC of 0.9970 was achieved with only **40 of 78 features
carrying any signal**. And of those 40 "healthy" features, several are just
at their default values (e.g., `cadd_phred` default 15.0 appears nonzero
after scaling because SOME variants have real CADD scores while others don't).

The real signal carriers are likely a small subset:
- `n_pathogenic_in_gene` (gene prevalence — the #1 feature, 3.3× next)
- `consequence_severity` and its binary derivatives
- `splice_ai_score` (148K nonzero in Run 9)
- `alphamissense_score` (206K nonzero in Run 9)
- Allele frequency features (`af_raw`, `af_log10`, bins)
- Variant type features (`is_snv`, `len_diff`, etc.)

**CRITICAL for Run 11:** Wiring `--gnomad-constraint` alone could recover
4 features that were top-5 in Runs 7-8. This is the single highest-value
fix after the existing LOVD/DbNSFP/FinnGen wiring.

---

## Finding 2: `_engineer_features()` has a redundant Phase 2 block at the bottom

In `real_data_prep.py`, lines after the `feats.fillna(0.0)` block:

```python
        # Phase 2 features — now active
        protein_change = df.get("protein_change", ...)
        feats["codon_position"]       = protein_change.apply(_parse_codon_position)
        feats["splice_ai_score"]      = df.get("splice_ai_score", ...)
        feats["alphamissense_score"]  = df.get("alphamissense_score", ...)
```

This block:
1. **OVERWRITES** `codon_position` that was already set 50 lines above
2. **REDUNDANTLY RE-SETS** `splice_ai_score` and `alphamissense_score`

The `codon_position` overwrite is problematic: it uses `_parse_codon_position`
on a `protein_change` column that is likely never populated (the column doesn't
come from any annotation step). This means `codon_position` is being set to 0
by the HGVSp parser (which returns 0 for empty input), overwriting whatever
value came from the VEP annotation step above.

**Fix:** Remove this redundant block entirely. The features are already
computed in the correct location above.

---

## Finding 3: Launch script missing `--gnomad-constraint`

`launch_run10b_skip_kan_v2.sh` passes:
- `--lovd-path` ✓
- `--dbnsfp-path` ✓
- `--gtex-genes` ✓
- `--spliceai` ✓
- `--alphamissense` ✓

But does NOT pass:
- `--gnomad-constraint` ✗ (4 features dead: loeuf, syn_z, mis_z, pli_score)
- `--finngen-path` ✗ (3 features dead)
- `--kg` ✗ (5 features dead: 1KGP population AFs)

**Fix for Run 11 launch script:** Add all three missing path arguments.

---

## Finding 4: "Healthy" features include default-value constants

The audit reports 40 "healthy" features, but some of these have nonzero
counts because their DEFAULT values are nonzero. After scaling, they have
variance only because some variants received real annotation values while
others kept the default.

Features where the "healthy" designation may be overstated:
- `cadd_phred` (default 15.0) — "healthy" but real CADD annotation was
  never wired (annotation_config.annotate_cadd=False by default)
- `sift_score` (default 0.5) — "healthy" only if DbNSFP provided real scores
- `polyphen2_score` (default 0.5) — same as sift
- `revel_score` (default 0.5) — same
- `gerp_score` (default 0.0) — same
- `alphamissense_score` (default 0.5) — "healthy" because 206K variants
  have real scores, so there IS real variance here

The audit script should be enhanced to also report the **default-value
fraction** — how many variants are at the feature's known default value.

---

## Finding 5: `variant_ensemble.py` TABULAR_FEATURES list vs actual features

The `TABULAR_FEATURES` list in variant_ensemble.py has 78 entries. The
`_engineer_features()` function in real_data_prep.py produces features that
should match. But there is no assertion in `_engineer_features()` that
`list(feats.columns) == TABULAR_FEATURES` — the assertion was removed during
the C5 namespace migration because the two lists diverged.

**Risk:** If the lists diverge (e.g., a feature is added to one but not the
other), models trained on one schema cannot be evaluated on the other. The
Run 10b splits have 78 columns, matching the TABULAR_FEATURES count, so
there's no current mismatch — but this is a fragile state.

**Recommendation:** Add the assertion back with a clear error message, or
at minimum add a post-regen check that verifies column alignment.

---

## Finding 6: `gnn_score` default discrepancy

In `variant_ensemble.py` `engineer_features()`:
```python
feats["gnn_score"] = df.get("gnn_score", ...).fillna(0.5)  # default 0.5
```

In `real_data_prep.py` `_engineer_features()`:
```python
feats["gnn_score"] = df.get("gnn_score", ...).fillna(0.5)  # default 0.5
```

But the audit shows `gnn_score` as 100% zero (DEAD). After scaling, constant
0.5 → 0.0. This is consistent: GNN never ran, so all values are 0.5, which
scales to 0.0.

However, when GNN DOES run and produces real scores (some 0.5, some not),
the default 0.5 will be indistinguishable from "GNN scored this gene at 0.5"
vs "GNN never scored this gene." This is a data-quality concern for when
GNN is activated.

**Recommendation:** Use `gnn_score = 0.0` as the default (absent = no
signal), not 0.5 (which means "average"). This is a minor change but
prevents future confusion.

---

## Summary: Run 11 Action Items from Findings

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| F1 | Add `--gnomad-constraint` to launch script | Recovers 4 top-5 features | 1 line |
| F2 | Add `--finngen-path` to launch script | Recovers 3 features | 1 line |
| F3 | Add `--kg` to launch script | Recovers 5 features | 1 line |
| F4 | Remove redundant Phase 2 block in _engineer_features() | Prevents codon_position overwrite | 3 lines removed |
| F5 | Enhance audit script with default-value detection | Better signal-vs-noise separation | 30 min |
| F6 | Add column alignment assertion | Prevents schema drift | 5 min |

With F1-F3, Run 11 would activate 12 additional features beyond the
Run 10b baseline, bringing the active feature count from 40 to 52+ (of 78).
