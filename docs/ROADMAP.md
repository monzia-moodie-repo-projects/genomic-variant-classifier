# GenAssoc / genomic-variant-classifier - Living Roadmap

**Version: 2026-11-16 (v5: set "au" merged; corrective set "au.1")  |  Owner: Monzia Moodie**

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
| Test suite | **9,464 collected** | `tests/EXPECTED_SUITE_SIZE`, and the README badge agrees |
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

The allele-frequency source is typed in `information_sources/allele_frequency.py` (two logical sources in one file; the
strict selection, claim eligibility and arms A/B/C separate; docs/census/AF_REPAIR_1D1c1.json). Its primary statistic is a
mixed, unfiltered, ClinVar-ascertained extract; strict admission does not certify FILTER or denominator quality.

# 5. Current execution and next milestones

**Package 1D, stage 1D-1c (the scientific repair; owner rulings b937681e, 482cd291, 4bef3910, aa999b96):**

| Unit | State |
|---|---|
| 1D-1c.1 A, B -- column identity, typed AF evidence | MERGED 2026-11-16 (pull request #77) |
| 1D-1c.1C set "au" -- model-facing core: semantic epochs, fitted member projections (P capability-aware, C common fold median, B research arm), the member adapter, the fit context, serving through member projections | MERGED 2026-11-16 (pull request #78) |
| 1D-1c.1C set "au.1" -- corrective revision (owner ruling ff77015c): content-keyed code identity, the B1 sample verified at every transform and its claim corrected, deployment admitted per recorded fit protocol | THIS DELIVERY |
| 1D-1c.1C set "av" -- data-facing: the qualified AF preparation path, the content-addressed reference-status receipt, the fresh-process end-to-end proof (ruling aa999b96 revised section 15) | NEXT |
| 1D-1c.2 REVEL and predictor applicability; 1D-1c.3 EVE identity and the 310-row population reconciliation; 1D-1c.4 enforced admission; 1D-1c.5 train-serve certification | AFTER |
| 1D-1d canonical leakage-disjoint evaluation (components on the full identity graph before any eligibility projection) | AFTER 1D-1c |
| 1D-2 the corrected strict baseline and the controlled comparisons (P, C, B, AF-free; source-shift; rare benign) | AFTER 1D-1d |

**No scientific training is reachable, and no non-scientific fit is deployable.** A load through `model_admission` must
be admitted for the fit protocol the predictor records: the decision names the protocols it admits, a scientific fit
only by default (`require_admitted_fit`, set "au.1"). No scientific `FitContext` can be constructed (`models/fit_context.py`), and
`models/execution.py` `scientific_fit_context(intent)` refuses, naming the five readiness gates that have no certifying
authority yet (`READINESS_GATES`); the launchers require `--execution-protocol` and refuse a scientific run before
reading data. Historical replay of the epoch-1 pipeline is named as such and is never a
scientific artifact.

**Standing preconditions for any run** (unchanged): zero known defects before launch, an all-models smoke test with no skip
flags, a freshly generated preflight, and the wall-clock estimate stated and accepted before any command expected to exceed
fifteen minutes.

# 6. Open register

Present-state items, by identifier; each is recorded in full where it was filed.

- **Open owner decisions.** DECISION-POLICY-OVERLAP-1 (the threshold partition rule).
- **Ruled 2026-11-16 (ff77015c), LOEUF.** An unknown `gene_is_constrained` stays refused for a finite-requiring member
  until the constraint-source stage qualifies LOEUF (observed versus unobserved from upstream provenance, gene and
  transcript identity, release; label-blind). At the first strict baseline: admit a qualified LOEUF family, or compile a
  versioned roster withholding `loeuf` and its whole dependent closure (measured: `gene_is_constrained`, `gnn_score`,
  `hetero_gnn_score`) from every estimator alike, through admission's least fixed point. No structural zero with an
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
