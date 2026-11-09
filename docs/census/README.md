# Stage 1D-1a information-source census (set "ao")

Owner rulings 2026-11-09 (decision files 25691d66 and c34eb717). Nonbehavioural: no model, feature, scaler, graph or
inference code changes. Author: Monzia Moodie.

## Files

| File | What it is |
|---|---|
| `FEATURE_CENSUS_1D1a.json` | The ledger: 91 feature entries in contract order, 11 non-tabular information sources, 25 datasets, the measured surface and graph inventories, every classified consumer site, producer digests, and the registered defects. |
| `measure_1d1_correction.json` | Withdraws the 1D-1 audit's `default_indicator_train` and `eve_gene_coverage_train` (EVE-COVERAGE-IDENTITY-1: standardised matrices were compared with raw defaults) and records what replaces them: Run 15's EVE connector ran unconfigured (log line 83), and the 36 columns constant in every Run 15 split (scale-invariant). |
| `../../scripts/measure_1d1_correction.py` | Writes the correction from Run 15's files and the audit's file; `--check` re-derives it byte for byte. |
| `../../scripts/build_feature_census_1d1a.py` | The reviewed content, by hand; the measured content, computed. `--check` re-builds and byte-compares. |
| `../../src/genomic_variant_classifier/information_sources/` | `census.py` (the gate) and `consumers.py` (surface, site and graph measurement). |

Verify: `python -m genomic_variant_classifier.information_sources.census --repo .`

## What the gate verifies, and what it does not

It verifies: every declared feature exactly once, in order; every surface measured from the code (eleven: feature
engineering, tabular fit, scaler fit, STRING-GNN node attributes, hetero-GNN node attributes, split persistence,
inference, calibration, export, evaluation, continual training), each entry's consumers EQUAL to its measured surfaces,
every module that references a feature-contract or graph-construction symbol classified (aliases included); every
graph relation and edge attribute the code can build is a census entry; every cited producer's semantic-closure digest;
every code anchor present exactly once; admission monotone over dependencies; no unreviewed field.

It does not establish the scientific truth of the reviewer's assertions, and says so:
`Scientific truth of all source claims: NOT ESTABLISHED BY STRUCTURAL VALIDATION ALONE`. Literature references are
checked for form only.

## Refinements over the owner's integrated reference (gvc_1d1a_integrated_reference.py, sha256 4ca06c71)

- **R1 Measured consumption is mandatory.** The reference's `verify_census` defaulted `required_external_surfaces=()`
  and its command line passed none, so a census with no consumer inventory passed.
- **R2 Identity follows meaning.** The reference bound the census to the whole 3,825-line `variant_ensemble.py` (any
  model-code edit invalidated it) and did not bind the producers in other files at all. Here: the digest of the ordered
  contract, plus a comment-insensitive digest of each cited producer and every module-level definition it transitively
  references. The file digest is recorded, not gated.
- **R3 Evidence is checked:** `code:<path>#<text>` must occur exactly once.
- **R4 Monotone admission:** an admitted entry depends only on admitted entries; measured dependencies (the GNN scores'
  node attributes) must be declared.
- **R5 Graph information is census information:** five knowledge-graph relation sources, three STRING edge attributes,
  the STRING edge selection and the two GNN supervision labels, measured from `load_kg_edge_specs` and `StringDBGraph`.
- **R6 Label-criterion use:** whether curators assign the label from the same evidence (ACMG/AMP population data,
  computational evidence, variant type, gene-disease curation, clinical classification). It does not change a
  disposition; it tells stage 1D-2 which results need a criterion-independent evaluation.

Disposition records whether the INFORMATION is admissible. A defect of representation (a fill value that is also an
observation, a missing observation status) is registered, not used to exclude.

## Result (a measured output; no count was targeted)

Features: 55 admit_strict, 8 conditional_only, 12 exclude_strict, 4 retire_predictive, 12 withhold_pending_evidence.
Information sources: 4 conditional_only, 5 exclude_strict, 2 withhold_pending_evidence.

Beyond the 19 pinned rulings: PolyPhen-2 and REVEL (score and flag) and `n_tools_pathogenic` are conditional (clinically
supervised, overlap with the label set unknown); `codon_position` is excluded (two incompatible quantities in one
column); the five splice-context features and the three GTEx eQTL features are withheld (their production path writes
defaults only); the three RNA-seq summary features are withheld until their sample population is reviewed.

## Registered defects (in the ledger, with evidence)

EVE-COVERAGE-IDENTITY-1, MISSINGNESS-SEMANTICS-1 (the owner's), GNN-SCALE-SKEW-1, GNN-NODE-LABEL-COUNT-1,
KG-DISEASE-RELATION-1, STRING-CHANNEL-ORIGIN-1, CODON-POSITION-SEMANTICS-1, RNA-SPLICE-GATE-ORDER-1,
GTEX-EQTL-PRODUCTION-DEFAULT-1. GNN-FEATURE-ORDER-1 and PHYLOP-RECONCILE-1 remain as already recorded.
