# Stage 1D-1a information-source census (set "ao")

Owner rulings 2026-11-09 (decision files 25691d66 and c34eb717). Nonbehavioural: no model, feature, scaler, graph or
inference code changes. Author: Monzia Moodie.

## Files

| File | What it is |
|---|---|
| `FEATURE_CENSUS_1D1a.json` | The ledger: 91 feature entries in contract order, 11 non-tabular information sources, 27 datasets (25 at stage 1D-1a; `uniprot_index` added at stage 1D-1b.2, `hgnc` at stage 1D-1b.3), the measured surface and graph inventories, every classified consumer site, producer digests, and the registered defects (24 at stage 1D-1b.4). |
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
GTEX-EQTL-PRODUCTION-DEFAULT-1. GNN-FEATURE-ORDER-1 and PHYLOP-RECONCILE-1 remain as already recorded. Later stages register and repair defects in their own sections below; the ledger is the list (24 at stage 1D-1b.4).

## Stage 1D-1b.1 (set "ap"): typed observations and shadow admission

Owner ruling f2ec92cd (2026-11-09). Shadow mode: nothing is enforced and no production module imports this code.

| File | What it is |
|---|---|
| `ADMISSION_SHADOW_1D1b1.json` (replaced in 1D-1b.2 by `ADMISSION_SHADOW_1D1b2.json`; in git history) | For each of the 91 features and 11 information sources: its disposition; admitted or not under the `strict` and `conditional` regimes; its observation status (`not_recorded_by_producer` for every name -- no producer records one yet); the predictive surfaces that make it available to a fitted model; its registered defects. Then every name that reaches a predictive surface unadmitted, with the reason. Bound to the census ledger's SHA-256, contract digest and semantic digest. |
| `../../src/genomic_variant_classifier/information_sources/observation.py` | The observation contract for stage 1D-1b.2: six states, no sentinel on an absent state, no boolean as a number, the source named by SHA-256 when the state asserts something about it. |
| `../../src/genomic_variant_classifier/information_sources/admission.py` | Derives the report from the census after the census gate re-measures the repository. |

Verify (since 1D-1b.2): `python -m genomic_variant_classifier.information_sources.admission --repo . --check docs/census/ADMISSION_SHADOW_1D1b2.json`

Admission is transitive: a name is admitted under a regime only if its disposition is admitted there AND every measured
dependency is. Under the conditional regime `gnn_score` and `hetero_gnn_score` (both `conditional_only`) are therefore not
admitted: their node attributes include retired and excluded features, and their edges the excluded STRING selection.

Result (measured): strict -- 55 admitted, 47 names reach a predictive surface unadmitted; conditional -- 65 admitted, 37
(2 only through a dependency). Every one of the 91 features currently reaches a fitted model. Stage 1D-1c turns these into
refusals.

## Stage 1D-1b.2 (set "aq"): lineage integrity, source materialisation and the source-absence census

Owner ruling 7f4b69d0 (2026-11-09). Shadow mode: nothing is enforced and no production module imports this code.

| File | What it is |
|---|---|
| `ADMISSION_SHADOW_1D1b2.json` | The 1D-1b.1 report with lineage refused when cyclic or ungrounded, admission as the least fixed point, exposure graded (declared and statically reachable established; materialised, fitted input and inference input not established), and the statically reachable unadmitted names as `structural_exposure_risks`. |
| `SOURCE_ABSENCE_1D1b2.json` | What every feature becomes when its source is absent, measured by executing `engineer_features` (absent, NaN and varied frames) and, hermetically, the annotation chain; the raw inputs each feature reads; the lineage check of every derived feature. |
| `PILOTS_1D1b2.json` | The EVE and gnomAD allele-frequency pilots on the real source files and the Run 15 cohort: receipts, the legacy comparison, the extent of ABSENCE-AS-EVIDENCE-1 by label, and the digests of the code every step ran. Re-deriving it needs the source files (`scripts/measure_1d1b2_pilots.py`). |
| `../../src/genomic_variant_classifier/information_sources/lineage.py` | Cycle refusal (every cyclic component), dependency order, least fixed point, grounding. |
| `../../src/genomic_variant_classifier/information_sources/columns.py` | Ordered populations, columnar observations with reasons, derived-feature propagation rules. |
| `../../src/genomic_variant_classifier/information_sources/materialization.py` | The parser bound to the hashed bytes, measured receipts, the two pilot materialisers. |

Verify: `python -m genomic_variant_classifier.information_sources.absence_census --repo . --check docs/census/SOURCE_ABSENCE_1D1b2.json`

Regeneration order (each is bound to the one before): SOURCE_ABSENCE (its record must exist for the ledger) -> the ledger
(`scripts/build_feature_census_1d1a.py`) -> SOURCE_ABSENCE again (it records the ledger's SHA-256) -> ADMISSION_SHADOW.

## Stage 1D-1b.3 (set "ar"): source-bound graph identity and authorized labels

Owner rulings f2ec92cd (Unit 1D-1b.3) and 7f4b69d0 (sections 7-10); 7a2a6ade (section 13). Shadow mode: nothing is
enforced and no production module imports this code.

| File | What it is |
|---|---|
| `GRAPH_IDENTITY_1D1b3.json` | Measured on the owner's files (STRING v12 human links and info, HGNC complete set, ReactomePathways.gmt, the Run 15 cohort and splits): exact pair accounting, the quantization-model consistency, the four-way threshold cross-tabulation, the canonical HGNC mapping and its collisions, the G1, threshold-robust G1 and G2 gene graphs with their computed materialisations and census admission, the legacy StringDBGraph and its cache key, the cohort's canonical gene identity, and Reactome under the legacy and the global-size projections. Since stage 1D-1b.4 every graph is accepted by re-derivation from the pinned files, and the fit authorization is measured in `POPULATION_ISOLATION_1D1b4.json`. Binds the digests of the code that produced it (`scripts/measure_1d1b3_graph_identity.py`). |
| `STRING_READER_BENCHMARK_1D1b3.json` | The exact-pair reader's runtime, peak memory and scratch for five partition and chunk layouts, each in its own process; the pair table's content is identical in all (`scripts/benchmark_1d1b3_string_reader.py`). |
| `LABEL_SITES_1D1b3.json` | Every module that names the graph builders' label column or takes a `label_col` parameter, reviewed: role and reason. A test re-discovers the sites and refuses an unreviewed or stale one. |
| `../../src/genomic_variant_classifier/information_sources/graph_identity.py` | The exact-pair STRING reader; threshold classification (exact rational arithmetic at the boundary); HGNC canonical mapping; GraphPolicy, GraphMaterialization and accept_graph; the legacy cache key; per-source pathway relations with global set sizes and the induced-subgraph check. |
| `../../src/genomic_variant_classifier/information_sources/labels.py` | The label non-interference test and label-site discovery; since stage 1D-1b.4 `TrainingLabels` replaces the 1D-1b.3 label authority and store. |

Registered defects that were reproduced carry executable witnesses (`tests/unit/test_registered_defect_witnesses.py`): each
passes while its defect exists and fails once it is repaired, so the repair must remove the witness and the registration
together. Re-deriving the two records needs the source files, which a clone does not hold; `tests/unit/test_graph_identity_record.py`
refuses a record whose bound code has changed and checks that the registered defects state the measured numbers.

## Stage 1D-1b.4 (set "as"): population identity, fit authorization, claims, and the repaired label and graph contracts

Owner ruling 68ca322f and the owner's reference packages cee6713d, 7ba10b3b and 7b277d7c, audited and synthesized on
2026-11-13 (`AUDIT_1D1b4_REFERENCES_2026-11-13.md`). Shadow mode: nothing is enforced and no production module imports this code.

| File | What it is |
|---|---|
| `POPULATION_ISOLATION_1D1b4.json` | Measured on the owner's Run 15 split manifests (meta and y for training, validation and test) and the HGNC complete set: gene fields resolved to canonical gene SETS (and the 32 rows of ClinVar prose), variant identity, the three-way split audit, the STRICT and CONDITIONAL_COMPONENT fit authorities against the gene-direct rule and the 1D-1b.3 one-gene rule, the label mix, tempered component weights, the component-disjoint stratified checkpoint split against the legacy GNN early-stopping call, and one TrainingLabels receipt. Binds the digests of the code that produced it; `--check` re-measures it (`scripts/measure_1d1b4_population.py`). |
| `../../src/genomic_variant_classifier/information_sources/population.py` | PopulationRow (strict types; variant identity qualified by an enum), resolve_gene_field, split_components (variant and every listed HGNC locus), audit_split (the partitions a declared SplitProtocol requires), authorize_fit (STRICT; CONDITIONAL_COMPONENT, exploratory), tempered_component_weights, component_checkpoint_split. |
| `../../src/genomic_variant_classifier/information_sources/claims.py` | The claims authority: claim kinds and tiers, complete per-kind evidence scopes, trusted pins by content digest, rules that receive their scope, eligibility and scientific finding as separate axes; within-unit AUROC (area under the receiver operating characteristic curve) with two estimands and a paired unit bootstrap. |
| `../../src/genomic_variant_classifier/information_sources/labels.py` | `TrainingLabels`: built from a FitAuthorization, holds authorized native-integer labels only, records every read (authorized, denied, unknown) and every masked bulk read, cannot be copied. |
| `../../src/genomic_variant_classifier/information_sources/graph_identity.py` | `StringInfo` (content and digest as one object); `accept_graph` re-derives the graph from the pinned files; `verify_graph_consistency` is the self-check; bytes-backed read-only arrays and read-only mappings throughout. |

Repaired, with their witnesses and registrations removed: LABEL-TYPE-IDENTITY-1, LABEL-DENIED-READ-AUDIT-1,
GRAPH-MAPPING-RECEIPT-AUTHORITY-1, GRAPH-MUTABLE-MATERIALIZATION-1. Registered, with witnesses: GNN-EARLY-STOP-ROW-SPLIT-1 and
GENE-FIELD-PROSE-1. `tests/unit/test_population_record.py` refuses a record whose bound code has changed and checks that the
registered defects state the measured numbers.
