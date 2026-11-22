# GenAssoc / genomic-variant-classifier - Living Roadmap

**Version: 2026-11-18 (v7: set "av" merged; predictor source audit set "aw")  |  Owner: Monzia Moodie**

Repo: github.com/monzia-moodie-repo-projects/genomic-variant-classifier

*Authoritative source of truth for PRESENT STATE and INTENDED FUTURE. History is not kept here: it is in the two
archives below, in `docs/CHANGELOG.md` (newest first, one entry per session) and in `docs/sessions/`.*

> **AUTHORITY SUCCESSION, 2026-11-16 (owner ruling aa999b96, section 20).** The predecessor of this document is
> preserved verbatim at
> [`docs/archive/legacy/ROADMAP_2026-08-23_to_2026-11-16.md`](docs/archive/legacy/ROADMAP_2026-08-23_to_2026-11-16.md) --
> 118,940 bytes, 1,192 lines, git blob object identifier `771a678b5512a8616df31079eb4fdf98847b218c`, byte-identical to
> what was live at `1fba4cb8`. It had reached 99.1% of the 120,000-byte bound that `tests/unit/test_roadmap_claims.py`
> enforces, almost all of it a dated changelog (2026-09-23 to 2026-11-15). The ruling: preserve the bytes, write a
> concise living roadmap, and do NOT raise the bound.
>
> **The first succession, 2026-08-23.** Its predecessor is preserved verbatim at
> [`docs/archive/legacy/ROADMAP_2026-03_to_2026-08-22.md`](docs/archive/legacy/ROADMAP_2026-03_to_2026-08-22.md) --
> 466,826 bytes, git blob object identifier `990088a61365ef3de3a02fd34327c7c5f3134731`, byte-identical to what was live
> at `f2b93ff`.
>
> **Nothing was deleted.** Git blobs are content-addressed: identical bytes yield the identical object identifier
> regardless of path, and the test re-derives both identities from the bytes on disk on every run.

# 1. Project identity & goals

Production-grade multi-modal genomic disease-association program. Core: an ACMG/AMP-style (American College of Medical
Genetics and Genomics / Association for Molecular Pathology) variant pathogenicity classifier over a ClinVar GRCh38 table
of 4,399,089 variants after cleaning, of which 1,490,324 are binary-labelled under the current review-status policy (both
measured 2026-07-24, docs/measurements/), 91 features, 13-model ensemble whose one authoritative score is the stored convex
blend (exact kernel; the logistic stacker is a research candidate) + STRING-DB GNN + KAN. The four protein-structure
features are quarantined (`quarantine_policy.py`).

**Dual goal (both first-class):** classify variants and infer disease categories and phenotypes; and empirically measure,
compare and validate machine-learning algorithms on large complex data, including KAN (Kolmogorov-Arnold Network), GNN
(graph neural network) and GraphGPS, even at small performance differences.

**Default stance: implement / keep / study. Never drop models or features on marginal-AUROC grounds.**

# 2. Phase model and maturity

| **Phase** | **Name** | **Scope** | **State** |
| --- | --- | --- | --- |
| F | Foundation | Bug-fixed core, ensemble + GNN + KAN, real ClinVar | DONE (engineering) |
| D | Data expansion and scientific repair | Package 1D: information admission, evidence semantics, isolation | IN PROGRESS |
| P | Performance/infra | Polars layer; Rust inference service | NOT STARTED |
| A | Advanced modeling | Julia training, VAE, Bayesian UQ, GraphGPS | NOT STARTED |
| X | Productionization | REST API, Docker, clinical evaluation and report | NOT STARTED |

"Foundation: DONE" is an ENGINEERING statement (owner ruling 7f4b69d0, section 15). Engineering maturity -- correct and
loud execution, reproducible locked environments, content-bound artefacts and exact rosters, guarded delivery -- is
established. Scientific maturity is reported separately:

| Question | Scientific maturity | State (2026-11-16) |
|---|---|---|
| Is the target scientifically defined? | no | binary ClinVar label; clinical, gene-disease and molecular-effect targets not yet separated |
| Is the information regime enforced? | partly | every ensemble fit now declares an explicit fit context (stage 1D-1c.1C); admission of the fitted roster is still shadow: 47 names statically reachable unadmitted under the strict regime (docs/census/ADMISSION_SHADOW_1D1b2.json); enforcement is 1D-1c.4 |
| Is absence kept apart from evidence? | partly | epoch 2 of the feature builder (`gvc.features/2`) keeps the allele-frequency family NaN where unobserved, and each fitted member projects it inside its own fit; under absence epoch 1 fabricates a value for 89 of 91 features (docs/census/SOURCE_ABSENCE_1D1b2.json); epoch 2 removes the six AF features from that count, so 83 still do; the AF feed into production is set "av" |
| Is the evaluation population valid? | partly | splits are gene-disjoint by SYMBOL only; 327 of Run 15's 16,964 split components span partitions (docs/census/POPULATION_ISOLATION_1D1b4.json); canonical isolation is 1D-1d, grouping (a): every independently resolved HGNC locus (owner ruling aa999b96) |
| Is graph information source-bound? | no | shadow only (1D-1b.3, 1D-1b.4); the GNNs still build their graphs with the legacy code |
| Does performance generalise to held-out genes under the admitted regime? | not measured | the leakage-free baseline ladder (1D-2) is the next scientific milestone |
| Are molecular or clinical claims independently supported? | no | none is made; the claims authority exists in shadow (1D-1b.4) |

A source has three separate statuses, never one: IMPLEMENTED (the connector exists), SCIENTIFICALLY ADMISSIBLE (its
census disposition under a regime) and OPERATIONALLY POPULATED (a materialisation shows its values reach the rows).

# 3. Current state snapshot

**Every number here names the artifact it is read FROM, and is maintained against it.** The table is re-derived, not
frozen: `tests/unit/test_roadmap_claims.py` and `tests/unit/test_roadmap_counters_agree.py` fail on any drift.

| Quantity | Value | Measured from |
|---|---|---|
| Tabular feature contract | **91** | `EXPECTED_TABULAR_FEATURE_COUNT`, and `len(TABULAR_FEATURES)` agrees |
| Phase-2 features (declared, not yet computed) | **0** | `PHASE_2_FEATURES` is empty |
| Phase-4 features | 4 | `PHASE_4_FEATURES` |
| Sequence features | 1 | `SEQUENCE_FEATURES` |
| Base-model roster | **13** | `len(VariantEnsemble().base_estimators)` on a live instance |
| Registered agents | **22** | `Orchestrator._register_agents()` -> `_agent_registry` |
| Test suite | **9,621 collected** | `tests/EXPECTED_SUITE_SIZE`, and the README badge agrees |
| Feature-semantics epochs | 2 | `models/feature_semantics.py`: `gvc.features/1` (historical) and `gvc.features/2` (three-valued AF) |
| Strict admission | 55 features admitted, 47 entries not | `docs/census/ADMISSION_SHADOW_1D1b2.json` (36 features + 11 non-tabular information sources: 6 graph relations, 3 edge attributes, 2 supervision labels) |

**The roster is BUILT, not declared.** There is no roster constant; `_build_estimators` produces `base_estimators`.
Each member's measured missing-value capability is `models/member.py` `ESTIMATOR_CAPABILITIES`, re-measured by the suite.

# 4. Data sources

Wired and populated in Run 15: ClinVar (labels and cohort), gnomAD v4 (allele frequency, LOEUF, pLI, mis_z, syn_z),
AlphaMissense, SpliceAI, dbNSFP (SIFT, CADD, REVEL, GERP, PolyPhen), STRING-DB v12 (GNN), Reactome. Implemented but not
populated, or partial: ESM-2, PhyloP, GTEx, 1000 Genomes, dbSNP, OMIM, ClinGen, FinnGen, MaxEntScan, VEP coding context,
EVE, LOVD (tiny coverage). Quarantined: AlphaFold structure. Blocked: HGMD (paid; label leakage). The per-source history and
the candidate verdicts are in the archives.

The computational predictors (stage 1D-1c.2A, docs/census/PREDICTOR_SOURCES_1D1c2.json) are read from a dbNSFP index that
filled every absent SIFT, PolyPhen-2, REVEL and CADD score with its default before caching and reduced each transcript list
to an extreme (DBNSFP-INDEX-DEFAULT-BAKED-1, PREDICTOR-TRANSCRIPT-COLLAPSE-1). The raw dbNSFP 5.4a academic release
(CC BY-NC-ND 4.0, under the owner's licence) was acquired on 2026-11-18 and is held outside the checkout: the GRCh38 table's
MD5 equals the publisher's (d2ad53e1; SHA-256 90d57358); it is not yet qualified or consumed. The publisher's transcript-aware REVEL v1.3 (Zenodo 10.5281/zenodo.7072866, verified) is acquired and registered
(`revel_v1_3`), not yet consumed. The cohort declares no transcript; ClinVar's own Name does (COHORT-TRANSCRIPT-UNDECLARED-1).
dbNSFP 5.4a's table (508 columns, read on the owner's machine) carries per transcript the protein position (`aapos`), the
VEP consequence (`HGVSp_VEP`, Ensembl 116 = GENCODE v50), a `MANE` flag and the predictor scores, as aligned lists; it also
carries ClinVar's own annotation (nine `clinvar_*` columns, `clinvar_clnsig` the classification among them), which no
feature may read.

The allele-frequency source is typed in `information_sources/allele_frequency.py` (two logical sources in one file; the
strict selection, claim eligibility and arms A/B/C separate; docs/census/AF_REPAIR_1D1c1.json). Its primary statistic is a
mixed, unfiltered, ClinVar-ascertained extract; strict admission does not certify FILTER or denominator quality.

# 5. Current execution and next milestones

**Package 1D, stage 1D-1c (the scientific repair; owner rulings b937681e, 482cd291, 4bef3910, aa999b96):**

| Unit | State |
|---|---|
| 1D-1c.1 A, B -- column identity, typed AF evidence | MERGED 2026-11-16 (pull request #77) |
| 1D-1c.1C set "au" -- model-facing core: semantic epochs, fitted member projections (P capability-aware, C common fold median, B research arm), the member adapter, the fit context, serving through member projections | MERGED 2026-11-16 (pull request #78) |
| 1D-1c.1C set "au.1" -- corrective revision (owner ruling ff77015c): content-keyed code identity, the B1 sample verified at every transform and its claim corrected, deployment admitted per recorded fit protocol | MERGED 2026-11-17 (pull request #79) |
| 1D-1c.1C set "av" -- data-facing: the explicit qualified preparation (verified, declared sources; the strict AF selection fed to the single builder; the canonical epoch-2 matrix, content-addressed), the GRCh38 reference-status authority (content-addressed reuse; novel variants validated against the pinned sequence), rosters compiled by the admission authority, the source binding of a fitted ensemble, the fresh-process end-to-end proof (ruling aa999b96 revised section 15), the real-data record `docs/census/QUALIFIED_PREPARATION_1D1c1C.json`; REVISION C (owner ruling 6fa240c0): trust boundaries made exact -- the matrix and the reference status valid by construction and immutable, three identities (logical, representation, payload) and a two-level store with exclusive publication, source bytes re-hashed at consumption, the reference pin declared, exact replay labelled | MERGED 2026-11-18 (pull request #80) |
| 1D-1c.2A set "aw" -- the computational-predictor source audit (owner ruling e755519d): the publisher's REVEL v1.3 acquired and verified (MD5 as published, SHA-256 measured), the full dbNSFP index and the pilot's join measured, ABSENCE-AS-EVIDENCE-1 corrected (revision c, owner ruling ace92e09: 1,299,101 positives from no index score or an unmatched 0.5, of which 1,242,065 have no publisher score in the GRCh38 view -- 1,238,371 of those outside REVEL's domain -- and 57,036 have one), four defects registered with witnesses, three datasets registered, two label-blind CANDIDATE selections measured (ClinVar's preferred name by allele and amino-acid pair, 191,427 missense rows; the MANE Select transcript of the cohort's gene, 160,973; the 163 differing rows are policy disagreements, all with another amino-acid change), the append-only state SOURCE_ROW_WITHOUT_VALUE (counted only where it occurs, so no sealed record changes); the record `docs/census/PREDICTOR_SOURCES_1D1c2.json` | THIS DELIVERY |
| 1D-1c.2B the allele -> transcript -> protein-consequence qualification and transcript-aware REVEL (owner ruling ace92e09: a clinical-context arm from ClinVar's preferred name and a prespecified arm from the GRCh38 locus, candidates kept apart from qualified contexts, agreement a sensitivity cohort); 2C SIFT, PolyPhen-2 and CADD source qualification (from dbNSFP 5.4a, acquired 2026-11-18); 2D derived-feature semantics (epoch `gvc.features/3`); 2E qualification. 1D-1c.3 EVE identity and the 310-row population reconciliation; 1D-1c.4 enforced admission; 1D-1c.5 train-serve certification | AFTER |
| 1D-1d canonical leakage-disjoint evaluation (components on the full identity graph before any eligibility projection) | AFTER 1D-1c |
| 1D-2 the corrected strict baseline and the controlled comparisons (P, C, B, AF-free; source-shift; rare benign) | AFTER 1D-1d |

**No scientific training is reachable, and no non-scientific fit is deployable.** A load through `model_admission` must
be admitted for the fit protocol the predictor records: the decision names the protocols it admits, a scientific fit
only by default (`require_admitted_fit`, set "au.1"). No scientific `FitContext` can be constructed (`models/fit_context.py`), and
`models/execution.py` `scientific_fit_context(intent)` refuses, naming the five readiness gates that have no certifying
authority yet (`READINESS_GATES`); the launchers require `--execution-protocol` and refuse a scientific run before
reading data. Historical replay of the epoch-1 pipeline is named as such and is never a
scientific artifact. The qualified preparation (`information_sources/qualified_preparation.py`, set "av") is explicit
(`PreparationMode.QUALIFIED`; nothing inferred), refuses an undeclared source, a non-missing incoming `allele_freq` and the
legacy join's 1000 Genomes and FinnGen columns, and records that training is refused. Its matrix is qualified for the
allele-frequency family ONLY (QUALIFIED-FRAME-PROVENANCE-1, stated in every receipt's `scope`): every other column is the
builder's value from the caller's frame, so it is not a source-qualified biological feature matrix, and its source
binding licenses exact replay only (`BINDING_SCOPE`; general inference on a new population is stage 1D-1c.5).

**Standing preconditions for any run** (unchanged): zero known defects before launch, an all-models smoke test with no skip
flags, a freshly generated preflight, and the wall-clock estimate stated and accepted before any command expected to exceed
fifteen minutes.

# 6. Open register

Present-state items, by identifier; each is recorded in full where it was filed.

- **Open owner decisions.** DECISION-POLICY-OVERLAP-1 (the threshold partition rule).
- **Decided 2026-11-18 (owner rulings e755519d, ace92e09).** SIFT, PolyPhen-2 and CADD come from dbNSFP under the owner's
  academic licence (5.4a, acquired); the project is strictly research and non-commercial. Stage 2B keeps two transcript
  policies with separate identities (clinical-context reconstruction from ClinVar's preferred name; a prespecified policy
  resolved from the GRCh38 locus as the primary prospective candidate) and their agreement as a sensitivity cohort; a
  different score under the two is a policy disagreement, not a source conflict. If LOEUF, SIFT and CADD are still
  unqualified at the first strict fit, `admission.compiled_roster` gives 49 features (measured: loeuf, gene_is_constrained,
  sift_score, sift_deleterious, cadd_phred and cadd_high withheld).
- **Registered at stage 1D-1c.2A (executable witnesses).** DBNSFP-INDEX-DEFAULT-BAKED-1, PREDICTOR-TRANSCRIPT-COLLAPSE-1,
  COHORT-TRANSCRIPT-UNDECLARED-1. The legacy reader now refuses parse failure and repeated raw allele keys. Historical
  parquet indices retain default fills and within-row extremes. Explicit raw sources take precedence over sibling indices.
  ABSENCE-AS-EVIDENCE-1 was corrected at 2A. SIFT and CADD stay
  strict in principle and are withheld from a scientific fit until qualified.
- **Ruled 2026-11-16 (ff77015c), LOEUF.** An unknown `gene_is_constrained` stays refused for a finite-requiring member
  until the constraint-source stage qualifies LOEUF (observed versus unobserved from upstream provenance, gene and
  transcript identity, release; label-blind). At the first strict baseline: admit a qualified LOEUF family, or compile a
  versioned roster withholding `loeuf` and its whole dependent closure from every estimator alike, through admission's
  least fixed point (`admission.compiled_roster`, set "av"). Measured: `loeuf` has three dependents (`gene_is_constrained`,
  `gnn_score`, `hetero_gnn_score`), but the GNN scores are already not admitted under either regime (they depend on the
  retired `n_pathogenic_in_gene`), so the closure that drops out today is `loeuf` and `gene_is_constrained` (strict roster
  55 -> 53 names). No structural zero with an
  indicator, no threshold of an imputed LOEUF. Continuous LOEUF is the preferred candidate; thresholds 0.35 (historical)
  and 0.45 (gnomAD v4.1.1) are separate identities.
- **Open, P0 before any deployment (executable witnesses, tests/unit/test_open_serving_defects_1d1c1c_r1.py).**
  RUNTIME-SERVING-IMAGE-1 (the API image is Python 3.11 against the 3.12 fitting locks, and installs neither torch nor
  imodelsx, so 4 of the 12 tabular members cannot be deserialized there); DOCKER-RUNTIME-BASE-UNPINNED-1 (the runtime
  stage's base image is a tag, not a digest); SERVING-LEGACY-ROSTER-1 (a projected pipeline declares the whole epoch
  contract, not its members' fitted roster: a smaller admitted roster is refused at serving).
- **Blocking a full mainline run.** GNN-FEATURE-ORDER-1 (members fitted before the GNN scores are injected);
  MAXTRAIN-SPLIT-REPERSIST-1; GENE-IDENTITY-SPLIT-1 (live splits group gene-symbol strings; 1D-1d).
- **Registered with executable witnesses or in the census ledger.** The gnomAD defects of 1D-1c.1, GNN-EARLY-STOP-ROW-SPLIT-1,
  GENE-FIELD-PROSE-1, KAN-NONFINITE-MISSING-INPUT-1 (measured; repaired by the member projection), LOEUF-SENTINEL-1 (the
  model feature `loeuf` is filled with 1.0 and Run 15's metadata carries 57,556 rows at exactly 1.0; constraint stage),
  CONSTRAINTPOLICY-1 (LOEUF threshold 0.35 kept for comparability; gnomAD v4.1.1 recommends 0.45). Repaired in set
  "au.1": CODE-IDENTITY-STALE-CACHE-1, MARGINAL-RESAMPLING-SAMPLE-UNBOUND-1, ENGINEERING-FIT-DEPLOYABLE-1;
  PROTOCOL-B-CONDITIONAL-BLINDING-CLAIM-1 (claim withdrawn: protocol B1 is a marginal-resampling diagnostic; missingness
  stays recoverable from the projected AF with the other features, witnessed).
- **Engineering, recorded.** DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1, SILENT-FALLBACKS-1, REPORT-PROVENANCE-ABSENT-1,
  WINDOWS-RSCRIPT-NOT-ON-PATH-1, SOURCE-SCANS-ENTER-IGNORED-DIRS-1, DATA-CENSUS-TOOL-1, INSTRUMENT-DIR-FD-AUDIT-EVENTS-1,
  ESM2-REAL-MODE-FIXTURE-1, ESM2-LIVE-UNIPROT-DEFAULT-1, FRESHNESS-DRY-RUN-1, AGENT-LOGGING-IN-LIBRARY-1,
  CI-INSTALLS-DEV-RANGES-1, METRICORIGIN-1, TEARDOWN-1, the S4Vectors 0.50.1 known limitation (requalification triggers in
  the predecessor archive).
- **Scheduled.** The legacy-roadmap carry-forward audit: 47 of the 55 item identifiers in the first archive were not
  mentioned in the roadmap when measured on 2026-09-28; absence is not resolution.

# 7. Modeling & infra roadmap

- Ensemble: RF, XGBoost, LightGBM, SVM (nystrom + bagged_rbf), LR, GBM, 1D-CNN, TabularNN; CatBoost; MC-Dropout; Deep
  Ensemble; KAN; combined by the stored convex blend. Per-model comparison every run, under a declared projection protocol.
- GNN (Phase D/A): bf16 AMP, PyG SparseTensor/CSR, GraphGPS, Laplacian PE/RWSE, 3-channel STRING weights. GPU-only.
- After the corrected baseline: AF and LOEUF representation and ablation experiments, native-versus-imputed missingness,
  the marginal-resampling sensitivity experiment B1 (conditional resampling B2 only once its assumptions are testable;
  differences between protocols are effects of declared interventions, not causal estimates of ascertainment), source
  and coverage shift, missingness by mechanism stratum, and independent rare-benign and molecular-effect benchmarks.
- Performance (P): Polars; Rust inference service. Advanced (A): Julia, VAE, Bayesian UQ. Productionization (X).

# 8. Standing disciplines

- Measure first (no estimate without a probe); ALL-MODELS smoke before training; pre-flight gate.
- Guarded, reversible installers; validation, branch publication, merge verification and branch deletion as separate owner steps.
- Incremental checkpointing; irreversible or cloud commands in separate blocks.
- Never drop models or features; scope ambiguity -> STOP and ask with options.
- Keep this roadmap current, present state only; discharge history to the archive by succession, never by raising the bound.

# 9. Blockers

- HGMD Professional - procurement.
- dbGaP / TOPMed / controlled TCGA / CPTAC-protected - need an institutional Signing Official.

# 10. Changelog since the succession

`docs/CHANGELOG.md` is the per-session record. Entries from 2026-09-23 to 2026-11-15 are in the predecessor archive.

- **2026-11-16 -- authority succession.** The predecessor preserved verbatim (address and blob identity above).
- **2026-11-16 -- PACKAGE 1D, STAGE 1D-1c.1C, SET "au": the fitted member owns its projection.** Owner ruling aa999b96,
  audited in AUDIT_OF_RULING_1D1c1C_2026-11-16.md. Two feature-semantics epochs (`gvc.features/1` historical
  and default; `gvc.features/2` with the allele-frequency family NaN wherever unobserved and `af_is_absent` renamed
  `af_is_observed_zero` through an explicit census translation). Every tabular member owns a fitted projection (schema 2)
  fitted on exactly its own fitting rows, imputing the family jointly into model-projected names, with no availability
  indicator; protocols P (capability-aware, the primary benchmark), C (common fold median, the control) and B
  (availability-blinded draw, research only and never served). `VariantEnsemble.fit` requires an explicit fit context;
  serving executes only the members' own projections and binds each projection identity. The inference service's import
  closure is measured at 25 modules, all shipped. 9,336 -> 9,436 collected (+104 / -4 by node identity).
- **2026-11-16 -- SET "au.1": the corrective revision of set "au" (owner ruling ff77015c, the owner's independent
  audit).** The projection's code identity is computed from the source bytes as they are at every transform (a cache
  keyed by content; it was keyed by symbol and returned a stale digest after an edit). Protocol B is renamed B1,
  marginal resampling, and its fitted sample is re-verified (digest, count, domain, order) at every transform; its claim
  is corrected. A load is admitted only for the fit protocols the decision names. Three serving defects registered.
  9,436 -> 9,464 collected (+33 / -5 by node identity; the 5 are renamed).
- **2026-11-17 -- SET "av": the data-facing stage 1D-1c.1C (owner rulings aa999b96 sections 13-15 and revised sections 10
  and 15, ff77015c sections 11-13).** The qualified preparation feeds the strict AF selection to the single builder and
  persists the label-blind canonical epoch-2 matrix (89 columns: the contract without the two label-derived columns),
  content-addressed and bound to its sources, policy, population and code; built in chunks of a row-local builder (an
  executable proof over the builder's source, and bit-identical to one call). The GRCh38 reference status is reused only
  by exact source, rule, reference and population; a novel variant is validated against the pinned sequence by the same
  rule. Rosters are compiled by the admission authority; a fitted ensemble is bound to its canonical matrix and refuses
  any other, also in a fresh process. On the real files: 1,028,224 + 459,512 + 500 + 1,778 = 1,490,014 and 1,487,736
  sequence-resolved, the selection and reference digests equal to the allele-frequency ledger's. The first owner run found
  CANONICAL-IDENTITY-ENVIRONMENT-1 (the record's digest differed on Windows): numpy's log10 bits depend on processor
  dispatch and the platform's math library (measured: 27,950 of 1,028,224 real rows, at most 2 ULP), so the identity seals
  every column but af_log10, which is certified within 4 ULP of a 113-bit reference at preparation and at every reuse.
  REVISION C (owner ruling 6fa240c0; audit AUDIT_OF_RULING_6fa240c0_2026-11-17.md), every defect reproduced against the
  candidate's code first: REFERENCE-VERDICT-ZIP-TRUNCATION-1 (a reference digest covered only the first rows; the
  preparation accepted 1 reason for 320 verdicts) -- the digest refuses unequal lengths and a `ReferenceStatus` is valid by
  construction; a reference status under another pin was accepted -- the pin is declared and compared; a `Parsed`
  wrapper's digest was trusted -- its bytes are hashed again where consumed (integrity.require_verified_bytes); an
  in-place edit of the live matrix changed predictions under an unchanged identity, and a matrix contradicting its receipt
  could be constructed -- `CanonicalMatrix` is valid by construction, bytes-backed and re-verified at bind, score and
  write; one logical identity could not hold a second certified `af_log10` representation, success was reported over a
  missing or corrupt payload, and a fixed temporary name let one writer publish another's bytes -- three identities
  (logical, representation, payload), `<logical>/<payload>.parquet` with its record, exclusive publication by a unique
  temporary file and a hard link, and a record without its intact payload refused. Measured on the real files: one
  identity (9619b172) holds the AVX-512 payload (0.575 ULP) and the baseline payload (1.552 ULP) side by side; the record
  reproduces byte for byte on both paths. The 4-ULP bound is the project's registered permissible error, not a numpy
  guarantee. 9,464 -> 9,555 collected (+91 / -0). Merged 2026-11-18 (pull request #80, main e15190f1).
- **2026-11-18 -- SET "aw": stage 1D-1c.2A, the computational-predictor source audit (owner ruling e755519d; audit
  AUDIT_OF_RULING_e755519d_2026-11-18.md).** The publisher's REVEL v1.3 copied from the project's Drive and verified (MD5
  equal to Zenodo's; SHA-256 815801d8); all 82,100,677 rows validated (no row without a score; 509,942 without a GRCh38
  position; every transcript identifier unversioned). The full dbNSFP index holds each predictor's default exactly on
  15.5 M (REVEL), 16.4 M (SIFT), 18.8 M (PolyPhen-2) and 44,611 (CADD) rows against neighbouring counts of about 44,000,
  17,400, 13,000 and 19,000; at the pilot's alleles the fabricated 0.5 encodes nonsense and canonical-splice variants
  (AUROC 0.738 for the label over all rows, but 0.489 within missense and 0.500 within nonsense) and the publisher has a 0.5
  at 66 of 129,779 (transcript unverified). Two label-blind candidate selections are measured; the record reproduces byte
  for byte. Nothing in production changes; no model is fitted. Revision b: the record's parts are written as canonical
  bytes on every platform (the owner's V1 differed only in their Windows line ends). Revision c (owner ruling ace92e09):
  the absence count and the selections carry exactly the meaning their computation supports, with the domain, GRCh37 and
  locus measurements behind them. 9,555 -> 9,593 collected (+38 / -0).
