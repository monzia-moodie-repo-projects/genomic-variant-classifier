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

## Stage 1D-1c.1 A and B (set "at"): the columns.py repairs and the allele-frequency materialiser

Owner rulings 482cd291, 4bef3910 and b937681e (2026-11-14). Shadow mode: nothing is enforced and no production module imports this
code.

| File | What it is |
|---|---|
| `AF_REPAIR_1D1c1.json` | Measured on the pilot's population (DataPrepPipeline._load_and_label on the verified ClinVar cohort; identity version 1 equal to `PILOTS_1D1b2.json`'s) and the verified gnomAD file, with reference identity from the recorded GRCh38 window check: the three typed evidence columns and their receipt, the strict selection's census against the owner's pre-registered figures, the three sensitivity arms, a label-permutation check, every row classified against the current join, and the rare flags described beyond AUROC (counts, prevalence with exact intervals, likelihood ratios, mutual information, per split). Binds the digests of the code that produced it; `--check` re-measures it (`scripts/measure_af_repair_1d1c1.py`; about 80 s and 3.4 GB peak memory in the sandbox). |
| `../../src/genomic_variant_classifier/information_sources/allele_frequency.py` | The two logical sources of the gnomAD file under strict grammars, typed source states with details, `select_strict` (policy, arms, claim eligibility) and `audit_binary_flag`. |
| `../../src/genomic_variant_classifier/information_sources/columns.py` | Since stage 1D-1c.1: population identity version 2, read-only column arrays, derivation lineage, `derive(name, inputs)` from the registry. |

Registered, with executable witnesses: the nine gnomAD extraction and join defects listed in the ledger (GNOMAD-*-1 added
2026-11-15). `tests/unit/test_af_repair_record.py` refuses a record whose bound code has changed; the 1D-1b.2 pilot materialiser is
not modified, so `PILOTS_1D1b2.json` still binds.

## Stage 1D-1c.1C (set "au"): feature-semantics epochs and the member projection

Owner ruling aa999b96 (2026-11-16), audited in AUDIT_OF_RULING_1D1c1C_2026-11-16.md. No record changes its meaning: the census
semantic digest is unchanged (97f64bcb) and admission is unchanged (strict 55 / 47, conditional 65 / 37).

- The census identity of every epoch-2 feature is read through `models/feature_semantics.py` `census_identity`: epoch 2 calls the
  census entry `af_is_absent` by the name `af_is_observed_zero` (an explicit translation; the ledger is not regenerated).
- The ledger cites `models/feature_semantics.py` as a definition site, and registers LOEUF-SENTINEL-1 (the model feature `loeuf`
  is filled with 1.0, and Run 15's metadata holds 57,556 rows at exactly 1.0; it affects `loeuf` and `gene_is_constrained`),
  with an executable witness. 34 registered defects.
- `census.py`'s canonical bytes and symbol-closure digest moved, verbatim, to `src/genomic_variant_classifier/code_identity.py`
  (standard library only); `census.py` re-exports every name, so each caller and each sealed module's import line is unchanged.
  The models layer binds its fitted projections to code closures through it without importing the census.
- `FEATURE_CENSUS_1D1a.json`, `SOURCE_ABSENCE_1D1b2.json` and `ADMISSION_SHADOW_1D1b2.json` are re-derived: the recorded file
  digest, the surface anchors' lines, the builder's closure digest and the ledger digest each record cites.

## Stage 1D-1c.1C (set "au.1"): the corrective revision

Owner ruling ff77015c (2026-11-16), audited in AUDIT_OF_RULING_ff77015c_2026-11-16.md. No record changes its meaning: the
semantic digest is unchanged (97f64bcb) and admission is unchanged (strict 55 / 47, conditional 65 / 37).

- `FEATURE_CENSUS_1D1a.json` is re-derived for one field, the authority's capture digest of `models/variant_ensemble.py`
  (`VariantEnsemble.load` now asks the admission decision to admit the recorded fit protocol). `SOURCE_ABSENCE_1D1b2.json` and
  `ADMISSION_SHADOW_1D1b2.json` are re-derived for the ledger digest they cite, nothing else.
- `code_identity.closure_digest_of_text` computes a closure from source text already read; `symbol_closure_digest` reads the file
  and calls it, so every digest any record binds is unchanged (`tests/unit/test_corrections_1d1c1c_r1.py` compares them all).

## Stage 1D-1c.1C (set "av"): the qualified preparation and the reference-identity authority

Owner rulings aa999b96 and ff77015c. No record changes its meaning: the semantic digest is unchanged (97f64bcb) and admission
is unchanged (strict 55 / 47, conditional 65 / 37).

- `QUALIFIED_PREPARATION_1D1c1C.json` (new) records the qualified preparation on the real files: the three sources declared by
  `AF_REPAIR_1D1c1.json`'s inputs and verified by digest; the pilot population; the reference status (identity and verdicts:
  1,489,514 `True`, 500 `non_acgt_allele`); the canonical matrix's receipt and identity; the exact reconciliation
  (1,028,224 + 459,512 + 500 + 1,778 = 1,490,014; 1,487,736 sequence-resolved), equal to the AF ledger's census, selection
  identity and reference digest; the AF family missing on exactly the 461,790 unselected rows; and the chunked build compared
  with one builder call on the first 300,000 rows. No annotation source is read, so every non-AF feature is the builder's
  default: the record's subject is the AF stage. `scripts/measure_qualified_preparation_1d1c1c.py --check` re-measures it.
  It holds no machine-dependent bits (CANONICAL-IDENTITY-ENVIRONMENT-1): the per-column digests of the 88 exact columns
  reproduce the receipt's `exact_columns_sha256`, and `af_log10` (whose bits depend on numpy's processor dispatch and the
  platform's math library) is certified within 4 ULP of log10 computed to 113 bits -- the project's registered permissible
  error, not a numpy guarantee -- not sealed. The machine-dependent representation (every column's exact bits) is printed
  by the script and bound by a fit, never recorded here. Revision c (owner ruling 6fa240c0): the receipt (schema /3) states
  its qualification scope (allele-frequency family only, QUALIFIED-FRAME-PROVENANCE-1), the declared reference pin's digest
  and the bound's basis; the reference status's identity and result digest are unchanged.
- `FEATURE_CENSUS_1D1a.json` is re-derived: two consumer sites are classified (`information_sources/qualified_preparation.py`,
  `surface:feature_engineering`; `scripts/measure_qualified_preparation_1d1c1c.py`, `diagnostic`), and the authority's capture
  digest of `models/variant_ensemble.py` changes (`source_binding_` is saved and restored). `SOURCE_ABSENCE_1D1b2.json` and
  `ADMISSION_SHADOW_1D1b2.json` are re-derived for the ledger digest they cite, nothing else.
- `admission.compiled_roster` compiles a regime's tabular roster with withheld roots, through `effective_admission`; it equals
  this record's admitted features for both regimes (tested).

## Stage 1D-1c.2A (set "aw"): the computational-predictor source audit

Owner ruling e755519d (2026-11-17/18), audited in AUDIT_OF_RULING_e755519d_2026-11-18.md. Nothing in production changes and
nothing is fitted. The semantic digest changes (97f64bcb -> c19eb103) only through the reasons of four strict entries:
`sift_score`, `sift_deleterious`, `cadd_phred` and `cadd_high` are strict in principle but withheld until their source is
qualified. No disposition changes; admission is unchanged (strict 55 / 47, conditional 65 / 37).

| File | What it is |
|---|---|
| `PREDICTOR_SOURCES_1D1c2.json` (new) | Measured on the real files: the full dbNSFP index (85,299,339 rows, every predictor's default counted exactly against its neighbouring values), the pilot population's join through the current connector, GENCODE v50's MANE Select transcripts, the publisher's REVEL v1.3 archive (Zenodo 10.5281/zenodo.7072866, MD5 as published, SHA-256 measured; all 82,100,677 rows validated) and ClinVar's variant_summary. It corrects ABSENCE-AS-EVIDENCE-1 with three quantities derived from one partition (revision c, owner ruling ace92e09): 1,299,101 rows carry `revel_pathogenic = 1` from no index score or an unmatched 0.5; 1,242,065 of them have no publisher score in the GRCh38 view (by domain and with the GRCh37 rows the publisher left unresolved) and 57,036 have one. It compares the historical index with the publisher at every allele and measures two label-blind CANDIDATE selections (ClinVar's preferred name by allele and amino-acid pair: 191,427 missense rows; the MANE Select transcript of the cohort's gene_symbol: 160,973; the locus-resolved MANE coding sequences measure that gene's dependence on ClinVar). Labels are only counted after each selection. `scripts/measure_predictor_sources_1d1c2.py` writes five parts and `collect --check` reproduces the record byte for byte. |
| `../../scripts/measure_predictor_sources_1d1c2.py` | `index-defaults` (numpy and pyarrow only), `pilot`, `mane`, `revel` (a vectorised, refusing scan), `collect`. Parts hold no time, memory or environment value, so they are byte-reproducible; the console reports those. Each part is written as bytes of one canonical serialisation, so its bytes are the same on every platform, and a reader refuses any other bytes (revision b: text-mode writing gave CR LF line ends on Windows, which changed the part digests the record binds). |

- Three datasets are registered: `revel_v1_3` (the publisher's transcript-aware release; no producer consumes it yet), `gencode_v50`
  and `clinvar_variant_summary`. 32 datasets.
- Four defects are registered with executable witnesses: DBNSFP-INDEX-DEFAULT-BAKED-1, PREDICTOR-TRANSCRIPT-COLLAPSE-1,
  DBNSFP-PARSE-FAILURE-SILENT-DEFAULTS-1 and COHORT-TRANSCRIPT-UNDECLARED-1; ABSENCE-AS-EVIDENCE-1's statement is corrected.
  38 registered defects. `tests/unit/test_predictor_sources_record.py` refuses a record whose bound code changed and checks that
  the statements quote it.
- `observation.ObservationState` gains SOURCE_ROW_WITHOUT_VALUE, appended (code 7; codes 0-6 unchanged), with a required digest.
  Counts name an appended state only where it occurs (`columns.COUNTED_STATES`), so no record sealed before the append
  changes (the allele-frequency ledger and the qualified-preparation record re-measure byte for byte).
  REVEL v1.3 itself has no row without a score (measured); the state is for per-transcript sources such as dbNSFP v5.
- `FEATURE_CENSUS_1D1a.json`, `SOURCE_ABSENCE_1D1b2.json` and `ADMISSION_SHADOW_1D1b2.json` are re-derived in the documented order.

## Stage 1D-1c.2A: the dbNSFP containment and its read follow-up

The owner's containment ruling of 2026-11-21 (pull requests #82 and #83). No disposition, admission or summary count changes;
the semantic digest stays c19eb103.

- The containment retires DBNSFP-PARSE-FAILURE-SILENT-DEFAULTS-1: the legacy reader refuses a configured source it cannot
  read instead of returning defaults. 37 registered defects; the defect leaves the six features it was attached to.
  DBNSFP-INDEX-DEFAULT-BAKED-1 and PREDICTOR-TRANSCRIPT-COLLAPSE-1 are restated for historical indices.
- `FEATURE_CENSUS_1D1a.json`, `SOURCE_ABSENCE_1D1b2.json` and `ADMISSION_SHADOW_1D1b2.json` are re-derived for the
  connector's code digest: ledger e6d2e91f -> 9d5c8e66 (containment) -> 8ae60493 (read follow-up).
