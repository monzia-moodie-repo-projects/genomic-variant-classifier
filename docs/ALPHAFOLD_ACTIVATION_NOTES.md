# AlphaFold Structural-Feature Activation (Phase D)

## Summary

The four AlphaFold structural features were **silent stub-constants** in the real
feature matrix prior to this work: `alphafold_plddt` (constant 50.0),
`solvent_accessibility` (0.5), `secondary_structure_context` (0),
`dist_to_active_site` (100.0). Only two CIFs were cached, there was no prebuilt
parquet, and no CLI wiring; the existing `ProteinStructurePipeline` fetched three live
REST APIs per gene at annotation time, which is unusable at cohort scale. The
`PHASE_2_FEATURES = []` comment falsely implied all Phase-2 features were active.

This activation makes all four features **real**, computed from AlphaFold mmCIF
structures, validated against the real `AF-E7ENB7` structure (a 98-residue BRCA1
fragment) before any code was shipped.

## Architecture (hybrid)

- **Cohort parquet** keyed on `(uniprot_accession, residue_pos)` -> `{plddt, rsa, ss,
  dist_active}`. Structural features are variant-independent, so a residue-level parquet
  is compact and reusable.
- **Connector** (`AlphaFoldConnector`) joins features at annotation time using the
  **canonical `protein_pos`** column that `real_data_prep` already computes at step
  10b/10c (AlphaMissense + HGVSp). This is the single source of truth for residue
  numbering -- no reimplementation, hence no drift.
- **Gene -> accession** via the local `uniprot_human_reviewed.parquet` `uniprot_id`
  column (verified 1:1: 20,190 genes = 20,190 distinct accessions, 0 duplicates).
- **Mandatory `wt_aa` cross-check** (fail-closed, mirrors the ESM-2 guard): before
  attaching a feature, the wild-type residue implied by the variant must match the
  residue at `protein_pos` in the UniProt sequence the structure is numbered against.
  Any mismatch (isoform disagreement) -> honest sentinel default, never a
  mismatched-isoform feature.

## Feature extraction (all validated against real AF-E7ENB7)

| Feature | Source | Validation |
|---|---|---|
| `alphafold_plddt` | mmCIF `B_iso_or_equiv` of the C-alpha atom | Exact; verified equal to `_ma_qa_metric_local` per residue |
| `secondary_structure_context` | `_struct_conf` DSSP records (parse-first); coordinate fallback when empty | Helix 3-21 / 51-71, strand at STRN residues, loop at TURN/BEND; distribution non-degenerate |
| `dist_to_active_site` | Real 3-D C-alpha Euclidean distance to nearest UniProt ACT_SITE/BINDING | Adjacent C-alpha spacing 3.83 A confirmed; active-site residue distance 0.0 |
| `solvent_accessibility` (RSA) | Shrake-Rupley SASA / Tien 2013 max-ASA, **clamped to [0,1]** | All residues in [0,1]; buried core (VAL11 = 0.48) < exposed terminus |

### Parse-first secondary structure

Two real CIFs disagreed: `AF-P38398` (full-length BRCA1) ships `_struct_conf` **empty**
(0 records), while `AF-E7ENB7` (98-residue fragment) ships **full DSSP** records
(`_struct_conf_type.criteria = DSSP`). AlphaFold CIFs therefore *sometimes* carry
DSSP secondary structure and *sometimes* do not, depending on model revision. The
builder is **parse-first**: it uses the file's DSSP `_struct_conf` when present (the
gold standard) and falls back to a coordinate-derived backbone-geometry assignment only
when the block is absent or empty. Residues not listed in `_struct_conf` (only
named-conformation residues appear) default to loop (0).

### RSA clamp to [0,1] -- reasoning (documented deliberately, NOT a silent fix)

Relative solvent accessibility is **defined** as a fraction in [0,1]: the ratio of a
residue's observed SASA to its maximum possible SASA. The Tien et al. (2013)
normalisation constants are derived from residues in extended Gly-X-Gly tripeptides.
Chain-terminal residues carry extra atoms (e.g. the C-terminal `OXT`) and unusually
high exposure, so their raw ratio can slightly exceed 1.0. On the validation structure,
exactly one residue -- the C-terminal ILE98 -- produced raw RSA 1.17; every other
residue (540 atoms' worth) was already in range, with the buried core low (VAL11 =
0.48) and the overall distribution sane (mean 0.42).

This is a **normalisation artefact, not a geometry error**: the underlying SASA is
correct; only the normalisation constant is slightly too small for a chain terminus. We
therefore **clamp RSA to [0,1]** (the definitional range). ILE98 -> 1.0, which is
correct (it is fully exposed). This preserves the real exposure signal rather than
discarding terminal residues.

Crucially, the clamp does **not** hide real failures. A genuine geometry failure
(corrupt coordinates, mislabelled atoms) produces RSA far outside this range. The
extractor enforces a **fail-loud guard** (`_RSA_FAIL_LOUD_MAX = 1.5`): any raw RSA above
1.5 or below 0 **raises `CIFParseError`** and the structure is skipped loudly, rather
than being silently clamped. The observed terminal artefact (1.17) is comfortably below
the guard; a corrupt multi-atom pileup (tested) reaches 3.48 and correctly raises.

## Feature count

`EXPECTED_TABULAR_FEATURE_COUNT` is unchanged by this activation. The four AF features are locked members
of `TABULAR_FEATURES`; this activation changes their *data source* (stub-constant ->
real), not the schema. No column is added or removed.

## Build (one-time, run by the developer)

```
python scripts/build_alphafold_parquet.py \
    --cohort data/processed/clinvar_grch38_clean.parquet \
    --uniprot-index data/external/uniprot/uniprot_human_reviewed.parquet \
    --out data/external/alphafold/alphafold_cohort.parquet
```

- ~18,302 of 19,383 missense genes resolve to an accession; 1,081 unresolvable ->
  honest defaults.
- Downloads each CIF once, **keeps raw** (Drive-backed), resumable.
- Mid-run disk guard aborts (exit 3) if free space < 5 GB; hard-errors (exit 4) if zero
  structures extracted.
- Bounded runs: `--max-genes N`; report-only: `--audit`.
- Final integrity asserts before writing: pLDDT in [0,100], RSA in [0,1], ss in {0,1,2}.

After building: `rclone copy` the parquet + raw CIFs to Drive; run the coverage-gate
audit and `pytest tests/unit/test_alphafold.py`.

## CLI / launcher wiring (applied by Install_alphafold_wiring)

1. `real_data_prep.py` `AnnotationConfig`: add `alphafold_path: Optional[Path] = None`
   and `uniprot_index_path: Optional[Path] = None` fields.
2. `real_data_prep.py` step 14: construct `AlphaFoldConnector(parquet_path=
   ac.alphafold_path, uniprot_index_path=ac.uniprot_index_path)` and call
   `annotate_dataframe` as the parquet-first path (defaults when path is None).
3. `run_phase2_eval.py`: add `--alphafold-path` and `--uniprot-index-path` argparse
   args; wire into the `AnnotationConfig(...)` construction.
4. `launch_run17_baseline.sh`: add `--alphafold-path` to the ARGS block with an
   abort-if-missing gate (mirrors the dbSNP block, exit 8) -- only when the developer
   opts to require it.

## Validation record

All extraction algorithms were validated against the real `AF-E7ENB7-F1-model_v4.cif`
before shipping. The production module `alphafold_features.py` passes 15/15 unit tests
including known-answer checks (pLDDT exact, DSSP SS, 3.83 A adjacent C-alpha, RSA
clamp + ordering), the fail-loud guard, the 1-based indexing tripwire, and the
connector's wt_aa fail-closed behaviour. The builder was validated end-to-end against
the real cached structure (98 residues, all features in range, real 3-D active-site
distances).
