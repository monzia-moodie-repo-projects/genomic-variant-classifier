# GenAssoc / genomic-variant-classifier - Living Roadmap

**Version: 2026-11-10 (v3; section 9 current to 2026-11-10, section 5 last re-derived 2026-09-04)  |  Owner: Monzia Moodie**

Repo: github.com/monzia-moodie-repo-projects/genomic-variant-classifier

*Authoritative source of truth for PRESENT STATE and INTENDED FUTURE. Updated at
the end of every session; Drive copy via rclone genvarcla:.*

> **AUTHORITY SUCCESSION, 2026-08-23.** The predecessor of this document is
> preserved verbatim at
> [`docs/archive/legacy/ROADMAP_2026-03_to_2026-08-22.md`](docs/archive/legacy/ROADMAP_2026-03_to_2026-08-22.md) --
> 466,826 bytes, 7,020 lines, git blob object identifier
> `990088a61365ef3de3a02fd34327c7c5f3134731`, byte-identical to what was live at
> `f2b93ff`.
>
> That document had become an append-only journal: 324 headings, roughly forty
> `ROADMAP delta` sections, and FOUR current-state snapshots superseding one
> another by name -- section 3 (2026-06-10), section 5 (2026-07-12), 6A (2026-07-15), 6C
> (2026-07-18) -- all still live in one file, with 38% of its bytes in two
> sections. A living roadmap that never discharges anything stops being
> readable, and an unreadable roadmap is not consulted.
>
> **Nothing was deleted.** Every delta, every superseded snapshot, every open
> register and the whole 2026-06-10 Appendix A remain in the archive, and the
> blob identity above is the proof: git blobs are content-addressed, so
> identical bytes yield the identical object identifier regardless of path.
>
> The same operation was performed once before, on 2026-06-10 -- recorded in section 9
> below as *"the pre-rebaseline repo-root ROADMAP.md archived verbatim into
> Appendix A"*. That succession archived INSIDE the same file, which is why the
> file then grew to 466 kilobytes. This one archives to a separate address.

*v3 note: this document carries PRESENT STATE. Discharged history lives in the
archive. Every headline number below is NAMED WITH ITS SOURCE so that it can be
re-derived rather than transcribed, and is maintained against that source
rather than frozen at the commit where the table was built.*

# 1. Project identity & goals

Production-grade multi-modal genomic disease-association program. Core: an ACMG/AMP-style (American College of Medical Genetics and Genomics / Association for Molecular Pathology) variant pathogenicity classifier over a ClinVar GRCh38 table of 4,399,089 variants after cleaning, of which 1,490,324 are binary-labelled under the current review-status policy (both measured 2026-07-24, docs/measurements/; corrected 2026-11-05 from "~1.49M cohort rows (from ~2.49M ClinVar missense)", whose second figure has no recorded source), 91 features, 13-model ensemble whose one authoritative score is the stored convex blend (exact kernel; the logistic stacker is a research candidate since 2026-11-07) + STRING-DB GNN + KAN. The four protein-structure features are quarantined (2026-09-22; `quarantine_policy.py`).

**Dual goal (both first-class):**

- Classify variants and infer disease categories/phenotypes.

- Empirically measure/compare/validate ML algorithms on large complex data, incl. KAN/GNN/GraphGPS, even at small performance differences.

**Default stance: implement / keep / study. Never drop models/features on marginal-AUROC grounds.**

# 2. Phase model

| **Phase** | **Name** | **Scope** | **Legacy label** | **State** |
| --- | --- | --- | --- | --- |
| F | Foundation | Bug-fixed core, ensemble+GNN+KAN, real ClinVar | Phase 1 / Phase 0 | DONE |
| D | Data expansion | Wire sources; activate dead columns; add new sources | Phase 2 (partly) | IN PROGRESS |
| P | Performance/infra | Polars layer; Rust inference service | roadmap 3/5 | NOT STARTED |
| A | Advanced modeling | Julia training, VAE, Bayesian UQ, GraphGPS | roadmap 4 | NOT STARTED |
| X | Productionization | REST API, Docker, clinical eval/report | Phase 2 (API/Docker) | NOT STARTED |

*We are in Phase D (data expansion). The engineering foundation is done; see the maturity states below before reading
"done" as anything more.*

## 2A. Two maturity states, reported separately (owner ruling 7f4b69d0, section 15)

"Foundation: DONE" is an ENGINEERING statement. The project's scientific maturity is a different question with a different
answer, and the classifier must not be described as scientifically validated because its engineering foundation is complete.

| Question | Engineering maturity | State (2026-11-10) |
|---|---|---|
| Does the code execute correctly; are failures loud? | yes | established: guarded suite, fail-loud contracts |
| Are dependencies reproducible? | yes | established: locked training and drift environments, verified on Windows and Linux |
| Are artefacts content-bound; are rosters and prediction identities exact? | yes | established (package 1) |
| Does delivery protect the working repository? | yes | established: guarded installers, receipts, post-merge checks |

| Question | Scientific maturity | State (2026-11-14) |
|---|---|---|
| Is the target scientifically defined? | no | binary ClinVar label; clinical, gene-disease and molecular-effect targets not yet separated |
| Is the information regime enforced? | no | shadow only: 47 names statically reachable unadmitted under the strict regime (docs/census/ADMISSION_SHADOW_1D1b2.json); enforcement is 1D-1c |
| Is absence kept apart from evidence? | no | 89 of 91 features fabricate a value when their source is absent (docs/census/SOURCE_ABSENCE_1D1b2.json); on the Run 15 cohort a missing gnomAD frequency alone separates the labels with AUROC 0.695 (docs/census/PILOTS_1D1b2.json) |
| Is the evaluation population valid? | partly | splits are gene-disjoint by SYMBOL only: read as gene SETS with variant identity, 327 of Run 15's 16,964 split components span partitions and validation and test share 54 genes (VARIANT-MULTIGENE-ISOLATION-1, docs/census/POPULATION_ISOLATION_1D1b4.json); 14 canonical genes straddle under two symbols (GENE-SYMBOL-SPLIT-IDENTITY-1); the shadow component authority would admit 910,273 of 1,038,974 training rows for exploratory fits (1D-1b.4); canonical gene and split identities are 1D-1d |
| Is graph information source-bound? | no | shadow only: the exact-pair STRING reader, computed graph identities and global-size pathway relations exist (1D-1b.3); graphs are accepted by re-derivation from the pinned files and materialisations are immutable (1D-1b.4); the GNNs still build their graphs with the legacy code |
| Are upstream supervision and label-criterion overlap characterised? | partly | recorded per feature in the census; criterion-independent evaluation is 1D-2 |
| Does performance generalise to held-out genes under the admitted regime? | not measured | the leakage-free baseline ladder B0-B4 (1D-2) is the next scientific milestone |
| Are molecular or clinical claims independently supported? | no | none is made; the claims authority that would decide one -- pinned, completely scoped evidence; eligibility and finding as separate axes -- exists in shadow (1D-1b.4) |

A source has three separate statuses, never one: an IMPLEMENTED connector (the code exists), a SCIENTIFICALLY ADMISSIBLE source
(its census disposition under a regime) and an OPERATIONALLY POPULATED source (a materialisation shows its values reach the rows).
Measured on the Run 15 cohort (1,490,014 rows): EVE is implemented, conditional_only, and was not populated in Run 15 (unconfigured;
with the release on disk it would observe 91,164 rows); gnomAD allele frequency is implemented, admit_strict, and populated for
992,044 rows -- none on chromosome X (GNOMAD-CHRX-LOCUS-PARSE-1).

# 3. Current state snapshot (2026-08-23)

**Every number here names the artifact it is read FROM, and is maintained
against it.** The table was constructed on 2026-08-23 at `f2b93ff`; the figures
in it are not frozen at that commit, and the right-hand column is what makes
each one re-derivable rather than transcribed.

> **ROADMAP-PROVENANCE-CLAIM-STALE-1, repaired 2026-08-26.** This paragraph
> previously read *"Every number here was MEASURED on 2026-08-23 at
> `f2b93ff`"*. That was true when written and false thereafter: eleven
> consecutive installers patched the collected count in the table below --
> 5,436 through 5,583 -- while leaving the sentence claiming where it had been
> measured. The figure was correct throughout; the PROVENANCE was not.
>
> It is the shape `TEMPORALCITE-1` records: a citation whose subject moved. A
> same-width substitution is invisible to a length check, and nothing read the
> prose around the number for eleven commits.
>
> `tests/unit/test_roadmap_counters_agree.py` now binds the three counters --
> this table, the README badge and `tests/EXPECTED_SUITE_SIZE` -- so a figure
> here that drifts from its source FAILS rather than merely misleading.

Earlier snapshots -- 2026-06-10, 2026-07-12, 2026-07-15 and 2026-07-18 -- are
in the archive, together with the supersession notices they wrote about each
other.

| Quantity | Value | Measured from |
|---|---|---|
| Tabular feature contract | **91** | `EXPECTED_TABULAR_FEATURE_COUNT`, and `len(TABULAR_FEATURES)` agrees |
| Phase-2 features (declared, not yet computed) | **0** | `PHASE_2_FEATURES` is empty |
| Phase-4 features | 4 | `PHASE_4_FEATURES` |
| Sequence features | 1 | `SEQUENCE_FEATURES` |
| Base-model roster | **13** | `len(VariantEnsemble().base_estimators)` on a live instance |
| Registered agents | **22** | `Orchestrator._register_agents()` -> `_agent_registry` |
| Test suite | **9,282 collected** | `tests/EXPECTED_SUITE_SIZE`, and the README badge agrees |

**Why the feature count reads 97 in the history.** HGMD was removed on
2026-07-13 -- `variant_ensemble.py:389` records *"Was 2 features; roster dropped
97 -> 95"*. Commit `80eb9c8` (2026-07-06) says *"->97 feat"* and was true when
written. Two measurements of one quantity, seven weeks apart.

**The roster is BUILT, not declared.** There is no roster constant;
`_build_estimators` produces `base_estimators`, which is what `fit()` writes
into `ensemble_completeness_["roster"]`. `tests/unit/test_readme_claims.py`
reads it from a live instance for exactly this reason, and records why a regular
expression over the source is not an acceptable substitute.

# 4. Data-source registry

## 4A. Wired & healthy

ClinVar (labels+cohort), gnomAD v4 (LOEUF, pLI, AF, **mis_z, syn_z, gene_constraint_oe**), AlphaMissense, SpliceAI, dbNSFP (SIFT/CADD/REVEL/GERP/PolyPhen), STRING-DB v12 (GNN -> gnn_score, confirmed real), n_pathogenic_in_gene, Reactome (reactome_pathway_count, wired 2026-06-08).

## 4B. Scaffolded but DEAD / partial (Phase D targets)

| **Source** | **Dead/partial column(s)** | **Access** | **Note** |
| --- | --- | --- | --- |
| ESM-2 | esm2_delta_norm (secondary), **esm2_llr** (primary, NEW) | local model+index | Phase 1 DONE 2026-06-10: esm2_llr LLR scorer (EsmForMaskedLM logits head; WT-marginal default, masked opt-in) + feature wired (79->80 lockstep; SIGNED, NOT clipped). CPU sign/index gate PASS; sign != class (continuous). Realizes after Run 16 coord-sync with esm2_model_name=esm2_t33_650M_UR50D. ESM C 600M = Phase 2 |
| PhyloP | phylop_score | free bigWig | conservation |
| GTEx | gtex_* (6) | free | eQTL/expression |
| 1000 Genomes | af_1kg_* (5) | free VCF | population AF -- ACTIVE 2026-06-15: kg_grch38_af.parquet built (chr1-22 + X, 437,668 variants = ~9.9% cohort; 5 super-pops non-zero); activate via --kg. chrY/MT structurally absent from the 1000G high-coverage panel (404-confirmed) -> 3,191 Y + 3,124 MT cohort variants get af_1kg=0; gnomAD Y/MT allele_freq RESOLVED 2026-06-16 (PAR X->Y fix): Y 1047/3155, MT 2731/3124 |
| dbSNP/RefSNP | dbsnp_af | free | DONE+VERIFIED 2026-06-26 (build_dbsnp_parquet.py; dbsnp157_cohort.parquet 3.75M rows, 46% AF>0). End-to-end audit 2026-07-01: 37.45% cohort coverage, dbsnp_af>0 confirmed through DbSNPConnector. Wired: --dbsnp-path -> AnnotationConfig -> real_data_prep step 10. |
| AlphaFold structure | alphafold_plddt, solvent_accessibility, secondary_structure_context, dist_to_active_site | free (AlphaFold DB) | **QUARANTINED 2026-09-22** -- out of the contract; both producers (`AlphaFoldConnector`, `ProteinStructurePipeline`) refuse, and a config requesting them is refused. Activation is NOT data + config: restoration is the Phase 1 repair (`quarantine_policy.py`; `docs/CONTAINMENT_2026-07-24.md` section 4). `has_uniprot_annotation` is a UniProt feature and remains in the contract |
| OMIM | omim_* (2) | free academic w/ reg. | disease/inheritance |
| ClinGen | clingen_validity_score | free API | **dtype drift: int vs float across prep/inference - fix before regen** |
| FinnGen | finngen_* (3) | free summary | population enrichment |
| MaxEntScan | maxentscan_score | free tool | splice strength |
| VEP | codon_position, exon_number | free tool | coding context |
| EVE | eve_score | free score files | needs score files + HGVSp coords |
| HGMD | hgmd_* (2) | PAID, blocked | label-leakage rules |
| LOVD | lovd_variant_class | free | tiny coverage |

*Note: gene_constraint_oe / gene_is_constrained moved OUT of 4B (constraint vestige resolved; oe now healthy and #2 feature).*

## 4C. New candidates - verdicts (unchanged from v1)

Strong fits: AlphaFold DB (DO), RefSNP/dbSNP (DO), COSMIC (DO, academic; feature NOT label), TCGA (OPTIONAL), Reactome (DONE), KEGG (OPTIONAL, overlaps Reactome). BioGRID overlaps STRING. dbGaP is an access prerequisite (blocked), not a connector. ProteomeXchange / SRA / ENA / DDBJ DRA / SILVA out of scope.

# 5. Immediate plan

> **NOT RE-DERIVED SINCE 2026-09-04 (recorded 2026-11-05).** The work of 2026-09-22 to 2026-11-05 -- the structural quarantine,
> containment, the gnomAD approval and source-monitor changes A, B, C1 and C2, the science stage (exact inference, the DANDELION
> ranking and its traces), environment qualification 1 to 9, the lockfile migration, the dependency profiles and the test-session
> guards -- is recorded in section 9 and in `docs/CHANGELOG.md`, and none of it is reflected in this section's plan. Which of the
> plan's items now come first is the owner's ruling, not a measurement, so this section is left as it was rather than rewritten.

**Last re-derived 2026-09-04 at `42780f4`.** The standing plan is quoted from
the archive verbatim below; what changed since 2026-08-08 is characterised, not
summarised, and the reason is given.

> **ROADMAP-PLAN-SECTION-STALE-SINCE-2026-08-29-1, repaired 2026-09-04.** This
> section stopped tracking on 2026-08-29. MEASURED at `42780f4`: of the
> twenty-three commits that landed from 2026-09-01 onward, **zero** were named
> anywhere in this document, the DRIFT-1 Phase 1 table ended at `ac14ab5`, and
> two passages had become false rather than merely incomplete -- the DRIFT-1
> state row, and the claim that Phase 1C is blocked on a finding closed at
> `4bed1b8` three days earlier.
>
> It is the shape `ROADMAP-PROVENANCE-CLAIM-STALE-1` records one section
> above, in a different register: there, prose outlived the number it
> described; here, a plan outlived the work it directed. A stale number
> misinforms a reader. A stale plan directs effort.
>
> **A search for this work under the wrong vocabulary found nothing and I
> nearly read that as absence.** This document uses `DRIFT-1 PHASE 1` and
> `Phase 1C`; the rulings use `3A` and `3B`. Searching here for `3B` returns
> zero, and zero was not evidence -- it was the wrong query.
> `KEYWORD-SEARCH-ASSUMED-A-SHARED-VOCABULARY-1`.

## The standing plan, quoted

From the archive's final `NEXT` section, dated 2026-08-08, reproduced exactly:

> **Commit C (SealedEvaluation)**, with both censuses now committed and citable.
> Then the BASELINE-1 repair across the README and roadmap, **DRIFT-1 with
> README-1**, **OP-1 step 5** against STEP K, **OP-2**, and **RETRAIN-GATE**
> last.

**Nothing in this section supersedes that.** The work committed between
2026-08-21 and 2026-08-23 was repository infrastructure: it did not touch Commit
C, BASELINE-1, DRIFT-1, OP-1, OP-2 or RETRAIN-GATE.

**Since 2026-08-23, three of those have moved.** Stated by identifier and
commit rather than characterised, because a plan is the worst place for a
summary of work that has not been read:

| Item | State | Commit |
|---|---|---|
| Commit C (SealedEvaluation) | CLOSED | recorded in `docs/sessions/` |
| BASELINE-1 | CLOSED | recorded in `docs/sessions/` |
| README-1 | CLOSED | `7cc213d` |
| DRIFT-1 | PHASE 0 CLOSED `abcb22e`; PHASE 1 CONTINUING; the kernel is still uncalled under `src/`, re-measured 2026-09-04 at `42780f4` | see below |
| OP-1 step 5, OP-2, RETRAIN-GATE | UNTOUCHED | -- |

DRIFT-1's phase 0 made the monthly job stop reporting a verdict it could not
reach; it did not make the assessment possible. `CONTINUAL-1` (`1ea45de`)
closed the sibling defect in the trainer, where a drift check that RAISED was
rendered as a finding of no drift.

**DRIFT-1 PHASE 1, 2026-08-27 to 2026-08-29.** Six commits, by identifier, in
the order they landed. Not summarised: `docs/CHANGELOG.md` and `docs/sessions/`
are authoritative, and a plan is the worst place for a summary of work that has
not been read.

| commit | unit |
|---|---|
| `694da7f` | a representation states its contract before it is compared |
| `66e2737` | a reserved layer gets its vocabulary and emits nothing yet |
| `c77a1a9` | four coordinate axes stop pretending to be one |
| `cffc51f` | the source kernel stops asserting what the data disproves |
| `69e8524` | the source declarations acquire a typed reader |
| `ac14ab5` | an invented vocabulary is retired for the registry that existed |

**DRIFT-1 PHASE 1 CONTINUED, 2026-09-01 to 2026-09-04.** Nine further units,
by identifier, in the order they landed. Not summarised, for the reason this
section already gives.

| commit | unit |
|---|---|
| `24bfb11` | 636 megabytes acquire a declaration |
| `4bed1b8` | three GENCODE products stop being one key |
| `accdf49` | the factory stops stringifying whatever it is handed |
| `4805033` | a digest that refuses when the file moves underneath it |
| `2d90c23` | freeze what an identity MEANS, before its owner moves |
| `1ef4ca5` | one class object, two import paths |
| `13daa3f` | a July retrieval record satisfied August evidence |
| `01cd4b4` | freeze what v4/schema3 emitted, before its retirement |
| `63c9f52` | one canonical identity epoch, one version authority |
| `b04e826` | retire the v4/schema3 evidence epoch for v5/schema4 |
| `086e2fa` | one grammar, one parser, and a canonical epoch spelling |
| `2f8dcb9` | three identity laws, two primitives, fourteen sabotages |
| `f9b4075` | the second equality site, and a whole-module static ban |

**BUILT AND NOT YET CALLED.** RE-MEASURED 2026-09-04 at `42780f4` by parsing
every tracked Python file -- 1,063 of them, zero parse failures -- and counting
construction sites for `SourceEvidenceManifest`, `SourceManifest`,
`SourceArtifactKey`, `SourceArtifactIdentity`, `SourceDependency` and
`SourceRegistry`:

| bucket | direct | factory |
|---|---|---|
| `src/` | **0** | **0** |
| `tests/` | 84 | 70 |
| other | 19 | 4 |

**Zero under `src/`, and the twenty-three elsewhere are all TOOLING**:
nineteen in `scripts/dev/build_provenance_migration_fixtures.py` and four in
`scripts/dev/freeze_source_evidence_epoch.py`, neither of which existed on
2026-08-29. A fixture generator and an epoch freezer are not the production
caller this finding waits for. `SourceRegistry` is still imported by exactly
one file -- `tests/unit/test_source_registry.py`, its own test.

Direct and factory constructions are counted SEPARATELY and never summed: a
factory call produces an instance without naming the constructor, so counting
one alone undercounts and merging them overstates.

**The 2026-08-29 figure of "85 test sites" is NOT COMPARABLE** and is not
restated as though it were. That record did not say whether 85 counted direct
constructions, factory calls, imports, or a sum; the two readings available
now are 84 and 154. A near match is not a match.

Recorded as `DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` in
`docs/sessions/CORRECTION_2026-08-29_part3_a-kernel-with-no-caller.md`, and
STILL OPEN. Its closure condition is sharper than when it was written --
`SESSION_2026-09-03_part2` states it: *closes only when a real computation
opens evidence, captures it, persists the manifest, reloads it, and downstream
logic consumes it.*

That is why the next DRIFT-1 unit supplies a CALLER rather than more
guarantees. An admission check wired to a manifest nothing constructs is what
`suite_transition.py` deleted three of, and what `preflight_data_guard.py`
records of itself: *a guard that is not invoked is not a guard; it is a comment
that happens to be executable.*

Phase 1C -- the reference profile that would construct a manifest from real
acquisition data -- was blocked on `ARTIFACT-KEY-INSUFFICIENT-1` (`482c0c9`).
**THAT BLOCKER IS CLOSED**, at `4bed1b8` on 2026-09-01, and four records agree:
`SESSION_2026-09-01_a-coordinate-the-evidence-required.md` lists it under
*Closed*, `docs/CHANGELOG.md` under *Fixed*, commit `11df0b5` states it, and
`tests/unit/test_source_release.py::test_three_GENCODE_products_are_three_keys`
binds it.

It closed **NARROWLY**. Of four measured collision classes, three dissolved
into axes that already existed -- assembly into `CoordinateContext`,
project-derived into `acquire`/`regenerate`, EVE's 3,212 files into a partition
axis -- and only GENCODE's three transcript products were a genuine missing
dimension. `SourceArtifactKey` gained an OPTIONAL product coordinate.

**The named blocker being closed is not the same as the phase being ready.**
This entry records the former and asserts nothing about the latter.

## What changed since, and why it is characterised rather than listed

**MEASURED 2026-08-23: eighty-three commits have landed since 2026-08-08.**
Twenty are the infrastructure stretch described in section 9 and in
`docs/sessions/`; the remaining sixty-three predate that stretch.

This section deliberately does **not** summarise eighty-three commits. A summary
of work that has not been read is exactly the failure this programme records as
`FABRICATED-OBSERVATION-1`, and a plan is the worst place for it: a wrong entry
here directs future effort rather than merely misinforming a reader.

Two sources are authoritative and complete for that history, and neither is
summarised here:

- `docs/CHANGELOG.md` -- newest-first, one entry per session, Attempted /
  Fixed / Failed / Learned.
- `docs/sessions/` -- one record per session, with measured figures, findings
  and their identifiers.

## The open register

**Fifty-four items**, enumerated by identifier in the archive's final delta,
with the arithmetic stated there: *"Fifty-two carried in ... Two filed.
52 + 2 = 54. The count RISES, and that is correct: a census that finds two real
things records both rather than tidying the number."*

Two carry detail that a paraphrase would lose, so they are quoted:

- **METRICORIGIN-1** -- a metric's origin is part of the metric. Log-scraped and
  computed figures share a flat mapping today; the spread between them in Run 14
  is 0.0010, and nothing in the type prevents them being read as one quantity.
  Closes when SealedEvaluation distinguishes them.
- **TEARDOWN-1** -- Run 14's session notes record a destroy command executing
  past its own gate's FAIL, root-caused to a fixed Test-Path check while the
  files lived one directory deeper. No data was lost, by fortune rather than
  design. Whether anomaly A8 and the recorded Charter v1.2 patch were ever
  applied is NOT ESTABLISHED, and must be before Run 17.

**No closure is asserted here.** A search of the eighty-three commit messages
found 115 distinct identifiers mentioned, and a mention is not a closure: a
commit may cite an item to say it is open, deferred, or blocking. Reconciling
the register against those commits is its own unit, and several entries are
judgements of scientific scope rather than facts about the repository.

## Standing preconditions for any run

Unchanged and not superseded: zero known defects before launch, an all-models
smoke test with no skip flags, a freshly generated preflight, and the wall-clock
estimate stated and accepted before any command expected to exceed fifteen
minutes.

# 6. Modeling & infra roadmap

- Ensemble: RF, XGBoost, LightGBM, SVM (nystrom + bagged_rbf), LR, GBM, 1D-CNN, TabularNN; CatBoost; MC-Dropout; Deep Ensemble; KAN; combined by the stored convex blend (the selected predictor; the logistic meta-learner is a research candidate, never served). Per-model comparison every run.

- GNN (Phase D/A): bf16 AMP, PyG SparseTensor/CSR, GraphGPS, Laplacian PE/RWSE, 3-channel STRING weights. GPU-only; 2-epoch probe first.

- CatBoost GPU-memory hardening (empty_cache between families / expandable_segments / order before torch models).

- Performance (P): Polars; Rust inference service. Advanced (A): Julia, VAE, Bayesian UQ. Productionization (X): REST API, Docker, clinical eval/report.

# 7. Standing disciplines

- Pre-flight gate; local mini-test before cloud; goal realignment each run.

- Measure-first (no estimates without a probe); ALL-MODELS smoke before training.

- Incremental checkpointing; irreversible/cloud cmds in separate re-paste blocks.

- Count-guarded, backup-first, idempotent, sandbox-validated patchers; byte-IO on Windows.

- **Background launch over SSH uses `< /dev/null`; read-only SSH checks use `-n -o ConnectTimeout=20 -o BatchMode=yes`; single-quoted SSH bodies; single-word grep patterns.**

- Document every run (algorithm comparison + metrics glossary); keep this roadmap current.

- Never drop models/features; scope ambiguity -> STOP + ask with options + pros/cons.

# 8. Blockers

- HGMD Professional - procurement; REVEL/VEST4/FATHMM/MutPred2 not labels if HGMD is a label source.

- dbGaP / TOPMed / controlled-TCGA / CPTAC-protected - need institutional Signing Official; blocked w/o R1 faculty sponsor.

- EVE - needs score files + HGVSp coords before eve_score is real.

# 9. Changelog

Every entry through 2026-08-08 is preserved verbatim in
[`docs/archive/legacy/ROADMAP_2026-03_to_2026-08-22.md`](docs/archive/legacy/ROADMAP_2026-03_to_2026-08-22.md),
together with roughly forty `ROADMAP delta` sections and the 2026-06-10
Appendix A. Entries are not restated here: a changelog that is copied forward
becomes a second copy of history, and a second copy is what this succession
exists to end.

- **2026-09-23 -- structural quarantine (ruled 2026-09-22).** The four protein-structure
  features left the active contract (95 -> 91). Producers, configuration, ensemble fitting,
  model loading, serving (which no longer zero-fills missing columns), cached splits and the
  scripts that fit on them all refuse them, per Monzia's September ruling (Option A) on
  `docs/CONTAINMENT_2026-07-24.md` section 4. Policy: `quarantine_policy.py`. Suite 6,758 ->
  6,871 collected (+114, 1 retired). Lands after the test-isolation prerequisite.
- **2026-09-23 -- containment completion (ruled 2026-09-23).** No unbound model load (one admission
  route: registry-measured binding plus a typed decision; gate C10 not implemented, so every real load
  refuses), all-or-nothing ensembles, historical execution denied, and CI repaired (the #882 failure,
  per-version reporting, image build on pull requests, tested-image release publishing). Suite 6,871 ->
  6,921 collected. Merged via #19 (CI green on the pull request and on `main`).
- **2026-09-24 -- gnomAD 4.1.1 approval recorded (issue #17).** Both monitor baselines 4.1.1;
  approval is NOT adoption -- every production gnomAD input is still v4.1, and migration is tracked
  separately. Producer claims must now EQUAL the independent witnesses as parsed identities (a stale,
  unsupported or substring-masked claim is refused). Merged via #20; verified on `main` by source-monitor
  run 36095779494 (three-layer check, 20 of 20). #17 closed 2026-09-25; successors #21 (4.1.2 constraint
  qualification) and #22 (constraint adoption for cohort v2), next review 2026-09-28.
  `docs/measurements/DECISION_2026-09-24_gnomad-4.1.1-approval.md`. Suite 6,921 -> 6,963.
- **2026-09-26 -- approval declaration installed (change A; owner rulings 2026-09-25/26).** The
  manifest (schema version 2) SELECTS a never-edited approval record by path and SHA-256; the
  record holds the 4.1.1 monitoring-baseline approval (scope `release_monitoring_baseline`,
  grant time unknown, evidence the decision pinned at 38987f54) and is read only after its bytes
  verify. The manifest reader is strict (no coercion, duplicate keys refused). CI admits approvals
  from Git objects before the suite (append-only history, evidence fetched and verified; Git 2.45+).
  The monitor does not read it yet (change B). Suite 6,963 -> 7,071. Ratified by the owner on #24
  at 2026-09-26T05:51:42Z, twelve minutes AFTER the merge (05:39:29Z); stated in the comment itself.
- **2026-09-26 -- CI runners pinned to ubuntu-24.04; release-grammar correction recorded.** All 14
  jobs across 7 workflows (13 were `ubuntu-latest`, which moves to Ubuntu 26 from 2026-10-19). The
  approval check's fetch list is de-duplicated. `DECISION_2026-09-26_release-grammar-four-part-correction.md`:
  four-component names stay review items (change B). Suite 7,071 -> 7,086.
- **2026-09-26 -- source-monitor release grammar and interpretation (change B; owner rulings 2026-09-25/26).**
  Every listed prefix is classified on the exact envelope `release/NAME/` by two independent parsers (adapter
  regex, verifier characters; 0 disagreements on 20,000 random prefixes and the owner reference). Two- and
  three-component names are ordered; four-component, prerelease and other names are unsupported REVIEW findings
  (exit 1) that block any absence claim -- never dropped (measured before: `5.0.0rc1` exited 0, `4.1.1.0` falsely
  alerted). Findings and witnesses name the RAW prefix. The monitor reads the approved release from the
  manifest-selected record; the verifier's pin must agree at runtime; each report carries a six-part
  interpretation fingerprint. PyYAML is installed hash-locked. The alert renders unsupported names; its
  2026-09-17 `[object Object]` xfail now passes and is locked in. Suite 7,086 -> 7,119. First real runs (#8, #9)
  verified: raw-prefix witness release/4.1.2/, fingerprint 4a39c162... reproduced from main's files.
- **2026-09-27 -- source-monitor run verifier, PREVIEW (change C, stage C1).** Trusted in-repository code
  (report_verifier.py + scripts/verify_monitor_run.py) verifies each run from GitHub's records and its report
  archive and reports the ruling's six results; a valid exit-1 run verifies with a review item. The new
  preview workflow (read-only; writes no issue) runs after every monitor run and on demand for a past run.
  The existing alert stays the only issue writer until C2. Reports now declare their run and attempt.
  Suite 7,119 -> 7,177.
- **2026-09-28 -- C1 accepted live; no test may publish to the workflow.** Preview run #1 verified run #8
  (all six results true, the 4.1.2 review item); preview run #2 refused a CI run (exit 2, all false). A defect
  followed: the C1 tests wrote FABRICATED verdicts onto CI run #896's job summaries through $GITHUB_STEP_SUMMARY.
  tests/conftest.py now removes the five workflow-command variables around every test; the verifier writes a
  summary only where told; an absent report is no longer called ambiguous. Suite 7,177 -> 7,181.
- **2026-09-29 -- interpretation contract version 2 (owner rulings 2026-09-28, review revision 3).** One
  dependency-light contract; the committed policy file is bound into the version-2 fingerprint (seventh part
  orchestrator_code); version 1 is admitted only for the three audited legacy commits; the run verifier
  reconstructs every part, replays under the run's own policy, treats validation findings as blocking and
  compares types strictly. Eight reproduced verifier defects fixed; no test can publish to the workflow
  before the interpreter starts; the policy file is byte-pinned in .gitattributes. Suite 7,181 -> 7,243.
- **2026-09-29 -- sixth publication channel closed.** The runner-variable census (CI run #899, names only)
  showed GITHUB_ARTIFACTS, documented by GitHub as the file that declares workflow artifacts for a step and added
  in actions/runner v2.336.0. It is now removed with the other five; tests/conftest.py holds the ONE list and the
  tests require ci.yml's removals to equal it. Suite 7,243 -> 7,244.
- **2026-09-30 -- C2: one issue writer (owner rulings 2026-09-29/30).** The verify job writes a typed, bound receipt; a
  separate publish job (the only one with issues: write) binds it to independently established facts, reconstructs prior
  dispatch from GitHub's execution history and posts AT MOST ONE comment to the pinned alert issue; uncertain outcomes are
  quarantined, never retried. The legacy alert workflow and its tests are removed, every protection carried. Receipt age
  policy: 900 s, 60 s clock allowance. Suite 7,244 -> 7,340. Activation needs the controlled cutover.
- **2026-10-01 -- C2 repairs and deployment configuration (owner rulings 2026-09-30 / 2026-10-01).** An
  unavailable checker receipt now reaches the writer (the evaluator identity no longer depends on evidence
  collection); receipt freshness is enforced immediately before the POST; deployment identities moved into
  a strictly validated configuration, with the source workflow and repository now authenticated by numeric
  id. Production cutover preceded isolated live qualification. This is a process deviation. Subsequent isolated
  qualification supplies compensating functional evidence. Suite 7,340 -> 7,371.
- **2026-10-01 -- C2 repairs 2, found by the isolated qualification.** The verifier's commit fetch is authenticated
  through that one process's environment (a private repository refused the anonymous fetch); a deployment refusal
  now records a no-POST outcome instead of leaving the attempt UNKNOWN. Suite 7,371 -> 7,373.
- **2026-10-01 -- C2 repairs 3: the attempt journal is the publish job's log.** Live qualification exercise 4 showed
  that after a re-run an earlier attempt's artifacts vanish from every GitHub listing, so prior dispatch was blind
  to earlier attempts; the publish job's log (retrievable per attempt by job id) now carries the journal, written at
  the gate before the POST. Suite 7,373 -> 7,383.
- **2026-10-02 -- C2 repairs 4: a conservative history reader.** Missing or ambiguous execution history (an empty
  job list, a missing or non-terminal step) and malformed journals (mistyped fields, more than one intent or outcome,
  wrong order, conflicting or invalid keys) are UNKNOWN, never no-dispatch; one action vocabulary is owned by the
  protocol. Suite 7,383 -> 7,400.
- **2026-10-02 -- C2 CLOSED: isolated live qualification preserved.** Two qualification rounds -- the second on the final
  runtime -- passed all five acceptance exercises and were independently reconciled; the exact evidence is preserved
  under records/verification/source-monitor-c2/ (typed manifest, verified offline by
  tests/unit/test_c2_qualification_record.py). Production cutover preceded isolated live qualification. This is a
  process deviation. Subsequent isolated qualification supplies compensating functional evidence. Narrative:
  docs/validated/C2_SOURCE_MONITOR_QUALIFICATION_2026-10-02.md. Suite 7,400 -> 7,419.
- **2026-10-04 -- Science stage, step 2: exact inference modules (owner rulings 2026-10-02 / 2026-10-03 / 2026-10-03b).**
  inference/exact_confirmation.py (the revised gene test P_g = max(b_g, min(1, M_g min_e t_eg)), explicit missingness,
  exact inclusive Holm, partial conjunction), inference/endpoints.py (known-positive recovery at 20 with the primary
  contrast DANDELION minus burden-only; assay-yield bounds with shared-gene cancellation), inference/analysis_contract.py
  (draft versus sealed; an independence claim seals only when documented disjoint). The C2 record tests now assert each
  scenario's contract-defined decision. Suite 7,419 -> 7,490.
- **2026-10-04 -- Science stage: the primary ranking method (owner ruling 2026-10-04).** inference/ranking.py: DANDELION
  with minimum-score gene aggregation over a frozen pair plan (incomplete execution refuses; significance flags and
  annotations cannot reorder), the top-k tie audit, fitting/evaluation universe identities and the three method
  identities. Measured: the published empirical-null calibration is defined but never called in any of DANDELION's
  73 commits, so the experiment reproduces the pinned implementation. Suite 7,490 -> 7,518.
- **2026-10-04 -- Science stage: the actual-call q-value backend trace (owner ruling 2026-10-03b).**
  scripts/dandelion/dandelion_backend_recorder.R observes, with trace() on the installed runtime, which branch each real
  DANDELION safe_qvalues call took; inference/backend_trace.py verifies the trace (exact-value digests, completeness,
  admissibility). Prerequisites recorded: DANDELION's R dependencies (igraph, qvalue) are not yet in renv.lock; the recorder
  is verified on R 4.3.3, the lockfile pins R 4.6.0. Suite 7,518 -> 7,535.
- **2026-10-05 -- Science stage: recorder version 2 and the numerical sensitivity audit (owner ruling 2026-10-04b).** The
  recorder refuses an unusable destination before any traced call and refuses a forked worker before writing; recorder-off
  and recorder-on runs in two fresh R processes produce byte-identical outputs. inference/ranking.py gains topk_audit
  (always-in / always-out / sensitive top-k membership over score intervals). Suite 7,535 -> 7,547.
- **2026-10-06 -- Environment qualification for R 4.6.1 (owner ruling 2026-10-05b, option B).** environment_qualification/:
  a clean probe of an explicit Rscript path; admission of the runtime-only lockfile change (/R/Version 4.6.0 -> 4.6.1, nothing
  else); a required-test outcome gate over a dedicated JUnit report (skips, absences, substitutions refused); and a receipt that
  binds the evidence to the exact candidate. Suite 7,547 -> 7,598.
- **2026-10-06 -- Environment qualification 2: no package substitution during installation (owner ruling 2026-10-06).**
  The R probe runs a FILE program and keeps complete evidence for every outcome; a source-only lockfile repair is admitted
  field by field; a sealed installation plan (one inspected artifact per package, bound to the lockfile, runtime and
  platform) is admitted before anything installs. Suite 7,598 -> 7,623.
- **2026-10-06 -- Environment qualification 3: the pre-install artifact inspector.** Every acquired package archive is
  judged by its INTERNAL DESCRIPTION, build metadata and digest -- never its filename -- and the installation plan now
  refuses a binary built for another R series (a CRAN binary built for R 4.7 would otherwise have been admitted for 4.6).
  Suite 7,623 -> 7,663.
- **2026-10-07 -- Environment qualification 4: R judges R's package metadata (owner ruling 2026-10-07).** The inspector's
  Python approximations of DESCRIPTION and version semantics were measurably wrong; Python now checks bytes and archive
  structure only, and the qualified R reads DESCRIPTION records and judges the dependency closure. The recovery contrast now
  reports the complete replacement sets. Suite 7,663 -> 7,679 (+43 / -27: one test file rewritten, no case dropped).
- **2026-10-10 -- Environment qualification 5: build planning (owner rulings 2026-10-08 to 2026-10-10).** Routes are selected
  from admitted artifacts (one per package; local builds = the reverse LinkingTo closure of packages without an admitted
  binary), installation is ordered, and build receipts and the installed dependencies a build used are admitted. Binary
  admission now counts native libraries in the archive instead of trusting NeedsCompilation. Suite 7,679 -> 7,706.
  CORRECTION (2026-10-09, measured on GitHub): this change was committed (6c3fdb6) and merged (a27ec6c, pull request #46) on
  2026-10-06; the date above and the branch name env-qual5-2026-10-10 are not its dates.
- **2026-10-07 -- Environment qualification 6: acceptance repairs (owner ruling 2026-10-07).** Ranking scores travel from R as
  lossless hexadecimal text; a replay may claim to be offline only under enforced, measured network isolation; and the R
  runtime is identified by its component files, not only its launcher. Suite 7,706 -> 7,735.
- **2026-10-07 -- Environment qualification 7: bounded admission repairs (owner ruling 2026-10-07b).** Scores cross from R only
  in the exact hexadecimal grammar; universe identities are unambiguous; replay rows are admitted by exact membership; a fixture
  run's loaded namespaces are checked against independent expectations; and an offline claim needs an enforcement mechanism's own
  evidence. Suite 7,735 -> 7,780.
- **2026-10-08 -- Environment qualification 8: one shared admission layer (owner rulings 2026-10-08, 2026-10-08b, 2026-10-08c).**
  Every script now uses one definition of artifact coverage, plan identity, library identity, lockfile transitions and the
  qualification decision; the older field-permission lockfile check is retired in its favour. Suite 7,780 -> 7,827.
- **2026-10-08 -- Environment qualification 9: the artifact-inventory verification record (owner rulings 2026-10-08c, 2026-10-08e,
  2026-10-08f).**
  Artifacts and runs stay outside the repository; the repository will hold portable, permanent records of what was observed about
  them -- requirements, content and locations kept distinct, each location observed once, every requirement's result derived, and
  incomplete searches never reported as absence -- with a replaceable index derived from them. Whether the environment is ready to
  replay is a separate decision taken against the installation plan and names every piece of code that determined it. The store
  root must be a plain canonical path (no link, junction or reparse point from the drive down), and a measurement is accepted only
  if the whole file census is unchanged at its end. The schema owner lands first; the first record follows from a measurement on
  the artifact store. Suite 7,827 -> 7,963.
- **2026-10-08 -- Science stage: predetermined method fixtures, the adjustment-to-endpoint trace and the exposure-failure policy
  (owner rulings 2026-10-08e, 2026-10-08f, 2026-10-08g).** Small fixtures with predictions frozen before any qualified run cover the
  native q-value route and every fallback; each fixture runs with the recorders off and on in fresh R processes, and a judge reports the
  input, adjustment, pair-decision, gene-aggregation and endpoint layers separately. Measured in the pinned code: an adjustment change
  alters DANDELION's own nominations but cannot move the minimum-score ranking, so it is never read as a change in Delta H(20). Every
  exposure now receives one status from the actual call, checked against what the inputs predict: a mixture-estimation failure
  withholds the primary ranking and Delta H(20) while the diagnostics finish (exposure completion, per-gene coverage, an exploratory
  partial ranking with "unscored" genes); the policy is a required element of the sealed analysis contract, and the feasibility stage
  withholds reference recovery and Delta H(20) altogether. The qualified run belongs to the isolated replay. Suite 7,963 -> 8,163.
- **2026-10-09 -- Science stage: the run intent and the reference-evaluator boundary (owner ruling 2026-10-09).** The analysis
  stage is now declared BEFORE execution in an immutable run intent that binds the sealed contract, the frozen pair plan, the input,
  environment and implementation identities and the release-policy identity; admission accepts only the bytes whose digest was
  recorded when the intent was sealed, so changing the stage alone breaks the binding, and an intent bound to another release policy
  is refused. Completeness is derived from admitted evidence -- no caller-supplied flag -- by the existing exposure validators plus the
  coverage rule (an empty eligible set, an unclassified or infrastructure outcome, or scores that disagree with the outcomes refuse;
  a mixture failure withholds). Reference evidence can be opened only inside the evaluator, only when the admitted intents and the
  re-derived assessment permit it; a feasibility run never opens it, and a confirmatory evaluation is a new identity that may reuse
  unchanged scores only with the feasibility history disclosed. The exposure recorder (version 2) also records each call's effective
  burden input, and a derived feasibility table shows which exposures share one burden-side estimation problem, with agreement across
  identical inputs. Measured while designing it: the repository had no operation-intent mechanism (a docstring claiming one is
  corrected). Next: environment admission and the isolated qualification, then the frozen feasibility run. Suite 8,163 -> 8,336.
- **2026-10-09 -- Environment qualification 9, completed: the first artifact-inventory record is committed.** The measurement of
  2026-10-09 13:40:36-13:42:13 UTC on the artifact store (collector cfc1cac8, implementation tree 6ef5da1f) found the content of all
  214 required content objects (313 requirements, every one "match"; 416 verified locations on one volume, 202 of them additional
  copies; no incomplete search; three R packages supplied by the runtime; one known gap: the evidence bundle of
  fixtures_20261007T022427Z was never retained, so it has no digest to verify). The readiness decision derived from it against the
  sealed replay plan v3 and candidate plan v2 found every selected installation input present (99 rows). That is HISTORICAL
  artifact-input availability, not runtime, dependency, behavioural or scientific validity, and authorizes nothing: installation
  admission must recheck the bytes it consumes. The record (REC-ece23653bf8e45dbad3da26303870dc6, sha256 bb716d48...2c22) entered
  the repository only after independent verification outside it -- strict round-trip through the typed owner, the collector's exact
  bytes, the summary and decision bindings, every loaded checkout module's digest, the run interval, and readiness RE-DERIVED from
  the record and the pinned plans -- and the installer binds it to the owner's evidence archive (0d14514d...8df6). Its bytes are
  pinned against line-ending conversion (ARTIFACT-INVENTORY-RECORDS-PINNED-1); the index is derived and replaceable. Suite 8,336 ->
  8,345.
- **2026-10-10 -- The lockfile migration is admitted; renv.lock is now R 4.6.1 with 102 packages (owner rulings 2026-10-08c/d/e/f).**
  The approved proposal (f21ac4bc: 90 provenance corrections in 13 packages, 15 additions including DANDELION and the locally built
  qvalue, R 4.6.0 -> 4.6.1, nothing else) was admitted against the actual baseline and candidate lockfiles and the two sealed
  installation plans: every input bound by digest (exact bytes and canonical text kept as separate domains), the regeneration
  differing only in its generator identity, the field differences equal to the approved transitions (each now required to state
  its effect on a later renv::restore), the artifact selection derived from the plans equal to the proposal's and to the candidate
  run's own report, each addition's lock label equal to its selected artifact's, and the candidate run's evidence archive the one the
  committed inventory found. The replaced lockfile, the candidate as renv wrote it, the proposal, its regeneration, the equivalence
  record, both plans and the run's difference report are preserved verbatim in a migration record
  (records/migrations/environment-qualification/lockfile/REC-722981980c854434ac9e3e7ec86956fe) whose admission is re-derived from
  those bytes by a test. This is a lockfile admission, not environment qualification. The first issue of this change failed to
  apply on Windows: one directory per preserved file made a 169-character path, 260 characters inside the installer's temporary
  clone, beyond the Windows limit. The record is now flat, and a repository PATH BUDGET (PATH-BUDGET-1: every tracked file path at
  most 150 characters and every directory at most 138, so any checkout or clone root up to 108 characters works without long-path
  settings) is enforced for every record and tested over every tracked path. Next: the isolated, network-disabled replay of the 102
  packages and the method tests on that library. Suite 8,345 -> 8,460.
- **2026-10-10 -- Repository history: the GitHub branch run9a-prep holds nothing main lacks, and may be retired.** GitHub reported
  "180 commits not in main" for run9a-prep (tip c1c01920, 2026-04-30). It is the April history that pull request #1 merged at
  2026-05-01 01:44 UTC (61c1b2b, the commit tagged run9a-baseline) and kept "briefly for reference"; the history rewrite of 2026-07-11
  (git filter-repo, which excised data/alphafold_cif_cache_2026-07-03.tar.gz and notebooks./gitkeep) renamed every commit on main and
  the identifiers quoted in messages, and the branch kept the old names. A count of commits says nothing about content, so a
  verification record now answers the content question commit by commit (repository_records/branch_equivalence.py;
  records/verification/repository-history/branches/REC-9ccf9fa0863b447292230fe1130850b0.json, sha256 9141d73e...19ff): of the 180
  branch-only commits, 169 have an identical counterpart on main (tree, author, committer, both dates, message, parents), 10 differ
  only by commit identifiers quoted in their messages, which map exactly to the counterparts' identifiers (13 pairs), and 1 (9bdd270)
  only added the excised notebooks./gitkeep; the tip's counterpart is the second parent of the pull request #1 merge. The branch
  reaches 180 commits and 2 trees that main does not, and NO blob: every file content it holds is on main. The record was derived
  byte-identically from the GitHub branch and from the owner's preserved bundle (history/run9a-prep_c1c01920aa0b_20261010T221514Z.bundle,
  sha256 1e83929f...e4f4, 3,796,302 bytes, kept in the artifact store); the installer re-derives it from that bundle and requires byte
  equality before committing -- measured on the owner's machine on 2026-10-11: byte-identical (9141d73e). Its bytes are pinned (BRANCH-EQUIVALENCE-RECORDS-PINNED-1). Deleting the GitHub branch is a separate,
  irreversible owner step taken only after this record is on main. Suite 8,460 -> 8,507.
- **2026-10-11 -- The Windows Sandbox isolation smoke qualification PASSED (revision c; the enforcement mechanism of the isolated
  replay, owner ruling 2026-10-08, Option A).** Measured on the owner's machine (build 29683, an ordinary window): all 26 checks passed
  (smoke_20261011T003302Z, evidence archive d142b790, terminal record 50d36b09) -- no network address, name resolution and three
  planned connections refused, the read-only mapping refused a write, the guest saw exactly the staged inputs, the host input and the
  configuration unchanged -- and the host itself closed the sandbox with `wsb stop` (exit 0; the first observation after the stop
  already showed the identifier unlisted and the virtual-machine worker exited). Why revision c: the first run (smoke_20261010T172959Z, archive 027b594c) passed
  all 20 isolation-evidence checks but its sandbox was still running almost five hours after the host asked for the window to be
  closed; the first launch registered the Windows Sandbox package (0.8.107.0) with the documented command-line interface `wsb`; and a
  non-elevated window reads empty command lines for processes an elevated window started (confirmed: the same reads succeed for
  processes the window started itself). Revision c identifies the sandbox by `wsb list` plus exactly one new worker and passes only
  if `wsb stop` closes it. The orphaned first sandbox was closed the same way first (exit 0, 0.7 s), from an ordinary window although
  an elevated one had started it. This qualifies the MECHANISM only: the replay itself (the 102 packages, preserved installers,
  inside the guest) remains to be built and run.
- **2026-10-11 -- Two Windows-only facts, measured by the second run of the run9a-prep installer (revision b, 01:53-02:02 UTC;
  nothing committed, the checkout verified unchanged).** (1) The installer's private temporary root works on the owner's machine (its
  probe stage passed), and the record re-derived byte-identically again. The user-wide TEMP\pytest-of-monzi was created 2026-10-09
  06:51:28 UTC, 0.8 s before the first of 973 tmp* folders that five test sessions left in TEMP (2026-10-09 06:51, 13:55, 17:39;
  2026-10-10 02:15, 10:28 UTC; about 20 minutes and 175 to 237 folders each); from an ordinary window it can be neither listed nor
  have its access list read, and renaming it there is refused (measured; my claim that an ordinary window could rename it was
  wrong). Its owner is read next from an administrator window, the one window that can move it. (2) Git for Windows refuses the path notebooks./gitkeep in an INDEX: 14 of the 40 new tests failed in their own
  history helper (Git 2.55.0.windows.5: read-cache.c verify_path calls compat/mingw.c is_valid_win32_path while core.protectNTFS is
  on, its default; no path component may end in a period or a space). The helper now writes trees with `git mktree`, which holds the
  excised path as a tree entry exactly as the real history does (+3 tests: the same trees an index writes for ordinary names, the
  excised entry, no index operation). A Linux build of that Git for Windows source with its Windows path check reproduced the 14
  failures exactly (14 failed, 356 passed, 2 skipped, the same node identities) and passes every test after the change; such a build
  now stands in for Git for Windows' path rules in this project's simulations, which a stock Linux git cannot.
- **2026-10-12 -- Test workspace hygiene: the suite stops leaving files in the checkout and in TEMP, and a guard makes it stay so.**
  Measured 2026-10-11: both validation clones that had run the full suite held models/best_gat.pt and compiled files, and the
  owner's TEMP held 973 tmp* folders from five test sessions. A full run at the same tree, with the owner's package versions and
  the installer's settings, left models/best_gat.pt (GNNTrainer created its checkpoint folder at construction and its default path
  is relative to the current directory; train_gnn_pipeline offered no way to choose it), 14 compiled files (two child interpreters
  in test_bootstrap_reconciliation.py started with a fresh environment) and, in its temporary root, 173 tmp* folders (72 mkdtemp
  calls in nine test files, plus library code). The 2026-09-23 entry below said the suite no longer wrote into the repository; its
  check ran without PyTorch (correction record beside it). Changed: the checkpoint folder is created where the file is written and
  the pipeline passes checkpoint_path through (default unchanged), every training test writes under tmp_path, and the two children
  run with -B. tests/conftest.py points tempfile and TMPDIR, TEMP and TMP at a folder inside pytest's base temporary directory from
  configuration on -- before any test module is imported (a first version as a session fixture still let PyTorch's cache folder
  escape during collection) -- so every temporary file, including child processes' and library code's, lands where pytest manages
  its lifetime. A working-tree guard (repository_hygiene/workspace_snapshot.py) snapshots the whole tree when a session starts and
  compares it when the test loop ends: anything added, removed or changed is named and fails the session even when every test
  passed; it never follows a link and does not enter the directories NOT_THIS_REPOSITORY declares. It compares at the end of the
  loop, not at session finish, because the first Continuous Integration run of this change (pull request #56, 2026-10-31) failed on
  both Python versions: pytest-cov writes .coverage and coverage.xml between the two (measured). The same run failed one guard test:
  it installs pytest 9.2.0 from requirements-dev.txt, which still calls plugins' terminal summaries under --no-summary; the test now
  holds on every pytest version. Open: TEST-R-PROJECT-PROFILE-1 (a test runs Rscript in the checkout at collection; the project
  profile activates renv and creates renv/library, which the guard does not enter by declaration); CI-INSTALLS-DEV-RANGES-1 (the
  gate installs requirements.txt and requirements-dev.txt as ranges, not the lock). Suite 8,507 -> 8,537.
- **2026-11-01 -- DEPENDENCY-PROFILES-1: the training-and-test profile is installed exactly from hashed, target-specific locks (owner
  rulings 2026-10-31).** Continuous Integration's pytest job installed requirements-api.lock, requirements.txt (57 exact pins, 7 ranges,
  1 unbounded) and requirements-dev.txt (7 ranges): nothing pinned the transitive dependencies, so on 2026-10-31 it tested pytest 9.2.0
  while the owner's validations ran 9.0.3; requirements-dev.lock (2026-05-07) was never installed and lacks 10 declared packages.
  Now: requirements-train-test-linux.lock (Python 3.11 and 3.12; two resolutions, byte-identical) and requirements-train-test-windows.lock
  (generated on the owner's machine), both resolved from requirements.txt + requirements-dev.txt under constraints-observed-venv312.txt
  (249 versions MEASURED in the owner's environment, which is preserved, not modified) by a pinned generator toolchain; every Windows pin
  equals the owner's measured version, and the Linux lock adds only Linux-specific packages. The job installs the lock with
  --require-hashes --no-deps, pins the build backend of three source-only packages (build-constraints-train-test.txt), and proves the
  environment IS the lock (deps/environment_observation.py, deps/environment_contract.py, scripts/check_dependency_profiles.py) before
  the suite runs. Every dependency file (16) and every install path (18) is declared in deps/governance.py; 16 install paths are
  not lock-qualified and say so (ci.yml among them: its drift and lockfile jobs are unchanged). pillow 12.1.1 -> 12.2.0 in
  requirements.txt (the owner's measured version). Suite 8,537 -> 8,583.
  CORRECTED: the entry above says requirements.txt is "ranges" -- see docs/sessions/CORRECTION_2026-11-01_requirements-txt-is-not-ranges.md.
- **2026-11-01 -- WARNINGS-AND-RESOURCES-1: an unclosed resource fails its test, and the package sets no warning filter.** A census of
  the merged tree with `-W always::ResourceWarning` found 850 unclosed-resource warnings at 42 places, all in this project's code
  (753 from one line of tests/unit/test_backup_artifacts.py; a directory iterator in scripts/preflight_check_data_paths.py; the child
  pipe of scripts/smoke_all_models.py), which the default filters never showed; all are closed, and pyproject.toml now makes
  ResourceWarning (and the unraisable-exception warning that carries it) an error. models/catboost_wrapper.py and models/kan.py
  wrapped their fits in a blanket `simplefilter("ignore")` that, measured, hid nothing of their own but on Python 3.12 governed every
  thread while active; both are removed and the package may hold no filter at any depth. The pykan fallback wrote ./model into the
  current directory (pykan's default auto_save) -- now off and tested; kan.py's dead efficient-kan path (an undefined name) and its
  false backend list are gone. pyproject.toml's claim that the scikit-learn delayed-warning filter is deterministic is corrected
  (SKLEARN-WARNING-FILTER-RACE-1: scikit-learn 1.8.0 resets the process-wide filter list in every worker thread). Open, recorded:
  SILENT-FALLBACKS-1 (catboost_wrapper's broad predict fallback and zero SHAP values on failure; KAN's silent backend substitution).
  Suite 8,583 -> 8,589. REVISION B (2026-11-02): pull request #59 failed both drift-monitor checks -- the configuration governs
  `pytest tests_drift/` too, and nannyml 0.13.1's usage telemetry (a Segment analytics event on every CBPE fit and estimate,
  identified by the machine's network hardware address) leaked /proc/self/cgroup. monitoring/performance_estimator.py switches
  that telemetry off (NML_DISABLE_USAGE_LOGGING) and imports nannyml without its implicit upward .env search
  (PYTHON_DOTENV_DISABLED, restored afterwards); seven drift tests with negative controls. Open, recorded:
  DRIFT-ENVIRONMENT-NOT-LOCKED (requirements-drift.txt pins seven packages; the rest resolves on the day).
- **2026-11-02 -- DRIFT-PROFILE-LOCK: the drift-monitoring profile is locked per target, and one installer serves every locked
  profile.** Closes DRIFT-ENVIRONMENT-NOT-LOCKED: the drift jobs of pull request #59 had resolved 137 distributions on the day
  (pyarrow 26.0.0 against training's 23.0.1 for the same parquet files; scipy differing between Python 3.11 and 3.12).
  requirements-drift-linux.lock and requirements-drift-windows.lock are resolved under constraints DERIVED from the training locks,
  so every shared distribution is at training's version except three departures each measured necessary (lightgbm, plotly,
  xgboost: nannyml's and FLAML's caps); wheels only. src/genomic_variant_classifier/deps/profiles.py declares both profiles once;
  scripts/install_locked_profile.py installs either exactly and proves the result IS the lock, for the pytest and drift jobs, the
  monthly drift monitor and a person's own machine. The monthly workflow's verify-model job could not import the project
  (ModuleNotFoundError by construction); it now installs nothing and computes its verdict. nannyml is imported in one place only.
  Install paths lock-qualified: 4 of 18. Suite 8,589 -> 8,607.
- **2026-11-02 -- CERTIFIED-AUPRC-DEGENERATE-REPLICATES-1 and RECALL-CURVE-FABRICATED-1: one bootstrap replicate engine, and curves
  only where they are defined.** The 33 warnings every full run had reported since 2026-08-26 (recorded as expected) were traced,
  call site by call site, to this project computing on single-class input. Two consequences were real. (1) The certified interval
  for the area under the precision-recall curve kept every replicate whose average precision was finite, while its accounting -- a
  separate replay -- counted single-class replicates as degenerate; scikit-learn 1.8.0's average precision answers 0.0 on an
  all-negative replicate and 1.0, silently, on an all-positive one. Measured on a 40-gene cohort with 2 positive genes: reported
  [0.0000, 0.8333], status OK, "863 valid, 137 degenerate", with all 1,000 values in the percentiles; over the two-class replicates,
  [0.4408, 0.8333]. Now ONE loop produces every replicate value and its count, and never calls the metric on fewer than two classes;
  intervals of metrics already undefined on one class are bit-identical (144 cases, four interval kinds). (2) On an all-negative
  cohort the report persisted a recall curve of 1.0 at every threshold, 0/0 that scikit-learn substituted; a curve is now computed
  only where its axes have members, and each absence names the class counts. Also: each curve's absence takes its cause from its
  own input channel (a refused `scores` array was recorded as undefined on the cohort); evaluate()'s unused duplicate absence maps
  are gone; metrics.py no longer imports nine scikit-learn names it never used, so the kernel imports without scikit-learn; a
  warning from scikit-learn's ranking module now fails its test. Zero warnings in the suite (Python 3.11 and 3.12; on Windows,
  2026-11-03: 8,545 passed, 83 skipped, 5 xfailed, 0 failed, no warnings). Merged 2026-11-03 as pull request #62 (2c2ba877).
  MEASURED 2026-11-03, the read-only census on the owner's machine: none of the 10 persisted reports is affected -- all are schema
  version 1 (2026-04-04 to 2026-06-13), written by ClinicalEvaluator._bootstrap_ci, which skipped single-class replicates from
  2026-03-12 (e1e9242) until it was replaced on 2026-07-26 (eca534e), after all ten. A report records neither the code nor the
  library versions that wrote it (REPORT-PROVENANCE-ABSENT-1, open), so one written by uncommitted code cannot be attributed.
  Suite 8,607 -> 8,633.
- **2026-11-03 -- SKLEARN-WARNING-FILTER-RACE-1: no thread-parallel scikit-learn in the suite, and a guard on the warning
  filters.** On Python 3.12 the warning filters are one process-wide list, and scikit-learn 1.8.0 resets and refills it inside
  every task of a thread-parallel call; concurrent tasks rebind it and can empty it. Measured inside pytest: 20 of 20 threaded
  forest fits left the process on a different list, one entry changed in meaning; a census of the full suite found 14 tests (all
  with threaded calls, none without) ending that way; on the owner's machine 448 copies of scikit-learn's pairing warning reached
  pytest through the ignore entry meant to hide them, and while a list is empty no `error::` entry applies. tests/conftest.py
  maps joblib's "threading" backend to its sequential one for the session (every thread request that names its backend,
  explicit backend="threading" included; process backends are untouched) and fails the teardown of any test whose warning state
  differs from the one bound before its setup -- and the session with it; entries libraries add while a test runs are listed in
  the guard's summary section. Both of scikit-learn's pairing warnings are now one error entry, by module. Child pytests that tests
  start run in one scrubbed environment (CI, width and colour settings changed what they printed). An independent review before
  delivery found five defects and several overstatements; each was measured and fixed. Production keeps its threads
  (scikit-learn #35089 is open upstream). Open, recorded: DRIFT-SUITE-UNGUARDED-1 (tests_drift/ has no conftest, so neither this
  guard nor the working-tree guard reaches it; CLOSED 2026-11-04, below); WINDOWS-RSCRIPT-NOT-ON-PATH-1 (in the owner's
  validations 28 tests skip because Rscript is not on PATH, and 10 more R tests skip for the pinned DANDELION package or unset
  fixture variables). Suite 8,633 -> 8,643. MERGED AND VERIFIED ON WINDOWS 2026-11-04: three -Validate runs, each 8,555 passed,
  83 skipped, 5 xfailed, 0 failed, no warnings, the guard 8,643 checked / 0 changed; pull request #63 (merge commit c98c5b92);
  the post-merge targeted run had no warnings (448 two post-merge blocks earlier). The 83 Windows skips: R 38, POSIX-only 15,
  symbolic links 13, pyBigWig and pybigtools 9, absent data or TODO 8.
- **2026-11-04 -- DRIFT-SUITE-UNGUARDED-1: the session guards in the repository root's conftest.py, for every suite.** The
  drift suite (tests_drift/, its own environment) never loaded tests/conftest.py, so it ran with no warning-filter guard, no
  working-tree guard, its temporary files in the system temporary folder, and only ci.yml's `env -u` before its tests and the
  workflow's publication files. Measured (pytest 9.0.3): a root conftest.py is loaded for `pytest tests/` and `pytest
  tests_drift/` alike, and not by a child pytest a test starts on files outside the repository. The guards that apply to every
  suite moved there by exact line ranges; tests/conftest.py keeps scripts/ on sys.path, the suite-size ratchet and the
  connector-cache isolation. A drift session now shows both guards' sections (21 checked, 0 changed; working tree unchanged);
  two drift tests prove the root conftest is loaded there and its preventions hold (both fail without it); at run time each
  guard hook has exactly one conftest implementation, the root's; a child session laid out like tests_drift/ fails on a test
  that rebinds the warning filters, writes into data/raw/cache, leaks a sys.path entry or writes into the checkout; and every
  child pytest of the suite is bounded to its own folder (--confcutdir). One behaviour changed, measured by the independent
  review: the guards are now set up before tests/conftest.py's connector-cache fixture. The repository root, newly on sys.path
  for the drift run, shadows nothing it imports. Drift suite 19 -> 21; suite 8,643 -> 8,647. MERGED AND VERIFIED ON WINDOWS
  2026-11-04: one -Validate run (121.3 minutes): 8,559 passed, 83 skipped, 5 xfailed, 0 failed, no warnings, coverage 83%, the
  guard 8,647 checked / 0 changed, the drift tests 21 passed (21 checked / 0 changed); pull request #64 (merge commit adb2e2fd,
  tree 679596de); the post-merge block: targeted 635 passed, 3 skipped, no warnings, drift 21 passed, your checkout unchanged.
- **2026-11-05 -- WARNING-FILTER-REORDER-AT-FIRST-IMPORT-1: no test's warning filters depend on which tests ran with it; every test
  file passes alone.** pytest collects inside warnings.catch_warnings() and discards what collection registered, so in a full run no
  test runs under the filters thirteen libraries register when first imported (requests, urllib3, numpy, pandas, scipy, torch, sympy,
  torch_geometric, accelerate, mlxtend -- a module-level simplefilter("always", DeprecationWarning) -- srsly, spaCy, pyspark); in a
  run of fewer files a library first imported INSIDE a test put its entries ahead of pyproject.toml's for the rest of that test.
  MEASURED on main (adb2e2fd) by running each of the 425 test files alone, one at a time, with Continuous Integration's options apart
  from coverage: 21 not clean -- 17 failed the warning-filter guard, 2 leaked tests/unit onto sys.path, 1
  (test_dbsnp_pop_preference.py) collected no test at all, 2 failed only because that run's Git was 2.43 -- and 378 passed with
  library entries "added during a test". With this change, each of the 426 files run alone: all clean (8,635 passed, 26 skipped, 6
  xfailed -- the whole suite). The root conftest.py now imports those libraries at session start, inside catch_warnings(record=True)
  (what they register discarded, what they warn handed to pytest's report, an absent one named, a failing one a usage error that
  carries its warnings and its writes); a check around collection stops the session if anything collected changes the warning state;
  and the per-test guard fails an added entry as it failed a moved one, saying where each stands. Tests import the tests/unit helpers
  as package modules instead of editing sys.path; test_dbsnp_pop_preference.py's checks became 8 tests. pyproject.toml pins
  torch_geometric 2.7.0's three import-time DeprecationWarnings about its own code (scoped by message and issuing module; a premise
  test re-measures them). README's repository structure and cohort sentence, and this document's header, section 1 and section 5's
  status, re-derived. Drift suite 21 -> 22; suite 8,647 -> 8,667 (+21 / -1). VALIDATED ON WINDOWS AND MERGED 2026-11-05: one
  -Validate run (04:20 to 06:21 UTC, 121.3 minutes): 8,579 passed, 83 skipped, 5 xfailed, 0 failed, coverage 83%, the warning-filter
  guard 8,667 checked / 0 changed (the eleven modules imported at session start in 28.78 s), targeted 756 passed, 3 skipped, drift 22
  passed, the checkout unchanged; candidate e28e0f13 (tree 6621b6ba, the sandbox's); pull request #65, merged with a merge commit
  (3c01564a, parents adb2e2fd and e28e0f13). The post-merge block passed on the owner's machine (07:49 to 08:05 UTC: the merge
  verified, the checkout fast-forwarded to 3c01564a, both candidate environments still exactly their Windows locks, targeted 756
  passed, 3 skipped, drift 22 passed, the checkout's 81 untracked files unchanged), and the branch was deleted (08:27 to 08:32 UTC).
- **2026-11-05 -- TESTS-REACH-THE-NETWORK-1 and GIT-VERSION-REQUIREMENT-UNSTATED-1: no test reaches the network; the Git requirement
  is stated by the reader itself.** MEASURED on the whole suite (8,667 tests, the gate's options, outcome unchanged by the
  instrument) with an audit hook in every Python process joined to the HTTPS proxy's refusal log: 15 tests reached the network -- 28
  proxied connections and 4 name lookups to rest.uniprot.org (ESM2Connector's default live lookup; 11 tests), ClinVar's FTP server,
  storage.googleapis.com, databases.lovd.nl, alphafold.ebi.ac.uk and api.github.com (the two data-freshness agents; 4 tests, one in
  a child interpreter) -- each refused in the sandbox and swallowed by the code; and collection started R five times in the
  checkout, where renv tried to bootstrap itself (7.72 s and 24 refused connections per start, measured alone). On machines with
  the network open those calls were made. The root conftest.py now sets the three Hugging Face offline variables for the session
  and refuses, inside the call, every non-loopback lookup, connection and datagram and every connection to a configured proxy; a
  test that tried fails at teardown even when the code swallowed the refusal, an attempt at session start or collection stops the
  session, and child processes and non-Python programs are stated as not covered. Each measured site was given a local source
  (AnnotationConfig.esm2_allow_network, the local UniProt index, an offline_freshness_sources fixture, an offline probe in the
  child) and the R module probes DANDELION once, lazily and outside the checkout (closing TEST-R-PROJECT-PROFILE-1 for that
  module). release_approval reads `git --version` once per reader and refuses below 2.45 with the requirement named. Open,
  recorded: ESM2-REAL-MODE-FIXTURE-1, PYCACHE-FROM-ISOLATED-CHILDREN-1, ESM2-LIVE-UNIPROT-DEFAULT-1, FRESHNESS-DRY-RUN-1,
  AGENT-LOGGING-IN-LIBRARY-1. REVISION B (same day): the owner's first Windows validation stopped at its targeted tests on ONE test
  of the change (1 failed, 883 passed, 14 skipped; the checkout verified unchanged) -- it read the program-start record in CPython's
  POSIX form, a list, where Windows reports one command line (PROGRAM-START-RECORD-PLATFORM-FORM-1; verified only on Linux). The
  recorder is now a fixture of tests/unit/conftest.py, installed once, keeping the platform's own form and calibrated by four tests
  against a real start on the machine that runs them; the guard now reads every proxy urllib.request.getproxies() can use, the
  Windows Internet settings included (GUARD-SYSTEM-PROXIES-1). Verified in the sandbox: 8,692 passed, 26 skipped, 6 xfailed; 0
  network attempts; the proxy recorded no refusal; drift 22 passed; 28 mutations caught across both deliveries. Suite 8,667 -> 8,724
  (+57 / -0). VALIDATED ON WINDOWS AND MERGED 2026-11-06: one -Validate run (2026-11-05 23:27 to 2026-11-06 01:22 UTC): 8,635 passed,
  84 skipped, 5 xfailed, coverage 83%, the network guard 8,724 checked / 0 tried, the warning-filter guard 8,724 / 0, drift 22
  passed, the clone unchanged; pull request #66, merged with a merge commit (0e5d73c4, parents 3c01564a and cf224271, tree
  2fd84792). The post-merge block's second run STOPPED: test_esm2_not_in_stub_mode skipped in the validation clone and passed in the
  owner's checkout, which holds the git-ignored UniProt index -- a count comparison between a clone and a checkout that holds local
  data. Its revision B compares each of the 910 targeted tests by node identity with the validation's full.xml and allows only
  that case, by name; it passed (04:26 to 04:39 UTC), and the branch was deleted. The finding started DATA-GATED-TESTS-1 (below).
- **2026-11-06 -- DATA-GATED-TESTS-1: no test's outcome depends on git-ignored local data, except three that check that data by
  design.** MEASURED on the owner's machine (the data census, 08:07 to 09:28 UTC): the whole suite in his checkout, on 0e5d73c4 and
  the validation's locked environment, under an instrument recording every path each test looked at (an audit hook and wrappers of
  the stat and os.path functions in every Python process, calibrated on that machine first), compared test by test with the
  validation's clean clone: 8,639 passed, 80 skipped, 5 xfailed against 8,635, 84, 5. Exactly 4 tests differ -- each skipped in the
  clone and passed in the checkout, each found by its paths: test_esm2_not_in_stub_mode (the UniProt index), the EVE (Evolutionary
  model of Variant Effect) module's two real-data checks, and the population stability index on the real Run-15 matrix. 46 tests read
  git-ignored data present there; of the 42 with the same outcome, 11 ran code that read the checkout's data without needing it (the
  Application Programming Interface startup's dbSNP index, preflight paths relative to the current folder, run_all's other gate over
  the real data tree, the freshness scan of data/), 24 source scans and 3 hygiene walks enter ignored folders, 1 audits the real tree
  by design, 2 only compare paths (the ClinVar ingest) and 1 reads the egg-info (EGG-INFO-SHADOW-1). The change: the ESM-2 test's
  three sequences committed as a UniProt FASTA fixture (pinned by accession, version and SHA-256; identical in the owner's index and
  UniProt's current entries), with a line-ending rule measured as necessary (FASTA-FIXTURE-EOL-1: without it a Windows clone wrote 49
  carriage returns into it); the three checks of local data declared with tests/local_data.py's `local_data` decorator (paths
  anchored at the repository root -- two were relative to the current folder -- pytest's own skip with one reason, `-m local_data`
  selects them), a contract that the set is exactly those three and every path git-ignored and untracked; the 11 given absent or
  temporary paths, each measured again under the instrument (api_without_local_data, with a contract that every test running the
  startup requests it); the isolated child of the dependency-profile test started with -B (PYCACHE-FROM-ISOLATED-CHILDREN-1). The
  drift suite, measured in the sandbox: no test reads a git-ignored path. Open, recorded: SOURCE-SCANS-ENTER-IGNORED-DIRS-1 (the 27
  scans and walks; the owner's ignored folders under src are the egg-info, agent_layer/data and agent_layer/logs), DATA-CENSUS-TOOL-1
  (the instrument is outside the repository; with it INSTRUMENT-DIR-FD-AUDIT-EVENTS-1, removal events carrying a directory descriptor
  read as current-folder paths). Suite 8,724 -> 8,753 (+29 / -0). REVISION B (2026-11-06): the owner's first V1 stopped at the
  targeted tests (6 failures, all carriage returns): the installer's `git apply` wrote the fixture under the old .gitattributes
  (FIXTURE-WRITTEN-BY-APPLY-1; the patch now goes into the index and git writes the files, every changed file's bytes checked), and a
  new test sent paths to git in text mode (STDIN-TEXT-MODE-1; now NUL-separated bytes, the repository's measured pattern of 2026-10-02).
  VALIDATED ON WINDOWS AND MERGED 2026-11-07: one -Validate run of revision B (01:52 to 03:55 UTC): 8,665 passed, 83 skipped, 5
  xfailed, coverage 83%, as forecast; test_esm2_not_in_stub_mode passed and the three local-data tests skipped with the one reason;
  drift 22 passed; pull request #67, merged with a merge commit (0e9db4e1, parents 0e5d73c4 and d1d30dc3, tree 2884c01f). The
  post-merge block compared all 810 targeted tests by node identity: 0 differences, the three local-data tests passing in the
  owner's checkout by name; both candidate environments still their Windows locks; the branch was deleted.
- **2026-11-07 -- PREDICTION-CONTRACT-1A: package 1 (the prediction contract), set 1a of three.** Owner rulings of
  2026-11-07 (decisions ccffb8cf, revision 5, and 38f15c7a): the stored convex blend of the fitted members' probabilities is the ONE authoritative
  score; the logistic stacker is a research candidate, never served and never substituted. Its kernel is
  `exact_binary64_products_sum_rne_v1` -- the exact real sum of the stored binary64 products, rounded once, nearest with ties to
  even, a result above one refused -- because the historical `P @ w` depends on the summation order and Dot2 is not exact (the
  owner's counterexamples reproduce: (1, 2^-52, 2^-106) with (1/2, 1/4, 1/4), and two 2^-1074 terms). MEASURED on Run 15's real
  883,127 x 13 out-of-fold matrix: the integrated kernel equals the owner's reference bit for bit on every row (and an independent
  exact method, fsum over error-free products, on every row); `P @ w` differs on 356,050 rows by at most 3 units in the last
  place (2.22e-16), with 0 call changes at 0.1/0.3/0.5/0.7/0.9 under both >= and >, and an identical total order (883,127
  distinct values), so every rank-based metric is unchanged. Set 1a: the observation and probability-block contract
  (prediction_contract.py), the exact kernel and the convex-blend specification (models/combiner.py), schema-2 uncertainty
  records whose identities close in their own units (entropy in nats, variance in probability squared; no stochastic pass is
  reported unavailable, never zero), `VariantEnsemble` scoring the declared blend from one member capture (the meta-learner
  fallback removed), serving with one scorer and one serializer (prediction schema 3.0.0: full-precision score, one tie rule;
  the five-tier labels, the unattributed counts and the unused fields removed), a readiness gate (`/predict` and `/batch` answer
  503 before any member call unless the registered production record, an EXACT roster and the same fitted predictor --
  `predictor_contract_sha256`, a new registry field -- all hold; a declared projection is readable and never ready), and
  transition refusals for the consumers that cannot yet consume the contract (calibrate_thresholds, run9_ablations, the
  score-reconstruction exporter, the SHAP writer; transition_policy.py, separate from the quarantine policy so its admission
  digest is unchanged). Gate C10 stays closed. Acceptance: docs/package1/ (the matrix, its revision 2 and the set-1a census
  ledger of all 282 score-surface candidates). Set 1b (out-of-fold identities, completion manifest, explanation and calibration
  writers) builds on merged 1a; set 1c (exporter metadata, thresholds, validators, forensic consumers) on merged 1b. Suite
  8,753 -> 8,875 (+151 / -29).
  VALIDATED ON WINDOWS AND MERGED 2026-11-07/08: one -Validate run (20:32 to 21:46 UTC; receipt 4d90c9fc): the whole suite 8,787
  passed, 83 skipped, 5 xfailed, exactly as forecast; targeted 1,210 passed, 5 skipped; drift 22 passed. Candidate 0daf3da8 (tree
  6bc717f5) published (23:51 UTC) and merged as pull request #68 with a merge commit: b051bb3d, parents 0e9db4e1 and 0daf3da8,
  tree 6bc717f5. The post-merge block (00:09 UTC) compared all 1,215 targeted tests by node identity against the validation: no
  difference; the branch was deleted (00:25 UTC).
- **2026-11-08 -- PREDICTION-CONTRACT-1B: package 1, set 1b of three.** Out-of-fold (OOF) rows now carry their ORIGINAL
  training-split row: `VariantEnsemble.fit` records each OOF row's fold (from the exact splits every member uses), its label and the
  fit-input size, and `evaluation/oof_identity.py` composes training-split row -> optional subsample -> `oof_fit_indices_` -> OOF
  row, verifying the composed labels against the fitted ones row by row; a failed binding blocks publication. A run is complete
  evidence only through `completion_manifest.json`, written last by `RunArtifactWriter.complete()` over every output's SHA-256
  (measured at write time and again at completion), the selected predictor and the prediction-block digests; a manifest left by an
  earlier run is retired (renamed, never onto an earlier retirement) before anything is rewritten, and every consumer
  re-measures it. Writers consume one bound capture per
  split (`VariantEnsemble.score_split`): predictions, calibration points with their real bin counts (separate from the
  per-consequence performance table), and the blend's exact member-score attributions phi = w (p - mu) against a declared
  background, with the closure residual recorded (a feature-level explanation is reported unavailable: the members' explanations
  share no definition, background or unit). The SHAP writer, which explained the logistic research candidate, is retired;
  run9_ablations is enabled (preflight before any fit, no broad handler around its deliverables); run_phase2_eval's `--max-train`
  path works (it raised AttributeError) and aligns the GNN, hetero-GNN and holdout gene groups with the subsample. The
  per-member `{name}_oof_indices.npy` sidecar is aligned with its OOF vector (until now the fold concatenation: measured on Run
  15, AUROC 0.499 when paired element-wise). Run 15's derived record (docs/package1/RUN15_OOF_IDENTITY_DERIVED_RECORD.json), from
  the artifact through a new inert pickle reader that runs no class's code: the stored `oof_fit_indices_` equal the reconstructed
  legacy carve exactly; AUROC through the mapping 0.85360 to 0.99906 per member, joined naively 0.4978 to 0.4999; the original
  file is not rewritten; its ENSEMBLE_STACKER rows are the stored blend under `P @ w` (the run's own code, read from the preserved
  pre-rewrite commit 4b9de1c3). FINDINGS, not fixed (owner decision): GNN-FEATURE-ORDER-1 -- run_phase2_eval fits the ensemble
  before the GNN blocks inject `gnn_score` and `hetero_gnn_score`, which are the constant 0.5 at fit time (measured), so the
  driver's census aborts and `fit` raises on a full cohort unless both dead-feature overrides are set, and with them the members
  would be fitted on a constant and evaluated on injected scores: this blocks a full mainline run (Run 17); MAXTRAIN-SPLIT-REPERSIST-1
  -- under `--max-train` the GNN block re-persists the subsampled `X_train` beside the full labels (the ablation preflight now
  refuses such a directory). Acceptance: docs/package1/PACKAGE1_SET1B_DELIVERY.md and the set-1b census ledger. Suite 8,875 ->
  8,922 (+58 / -11).
  VALIDATED ON WINDOWS AND MERGED 2026-11-08: one -Validate run (04:55 to 06:11 UTC; receipt dcee4aaa): the whole suite 8,834
  passed, 83 skipped, 5 xfailed, exactly as forecast; Run 15's derived record re-derived on Windows (digest b1330929, as on Linux).
  Candidate 7816bdc7 published (06:13 UTC) and merged as pull request #69 with a merge commit: a29af854, parents b051bb3d and
  7816bdc7, tree b0bb918b. The post-merge block (06:47 UTC) compared all 1,105 targeted tests by node identity: no difference; the
  branch was deleted (07:01 UTC).
- **2026-11-08 -- PREDICTION-CONTRACT-1C: package 1, set 1c of three; package 1 closes.** Every remaining consumer of the score
  reads BOUND evidence or refuses. A completed run's split predictions have one bound reader
  (`prediction_artifacts.load_completed_split_predictions`). The threshold workflow is migrated onto it and the repository's exact
  operating-point machinery (`evaluation/decision_policy.py`; MEASURED: the old search returned the top score for every target):
  thresholds on the served score stage, neutral band names, certification blockers, a record digest, refusals instead of
  fallbacks -- and, MEASURED, tail predictive values overlap on realistic runs, so the default floors refuse; the partition rule
  is an open owner decision (DECISION-POLICY-OVERLAP-1). Export metadata carries only evidence bound to the exported
  combiner (gate G7.2; `val_auroc` and `n_train` removed and refusing). The import-time threshold loader is gone; the four
  validators apply a policy only for their own predictor, measure training overlap instead of assuming it, stop fabricating a
  model input and rates, and map LOVD terms exactly (MEASURED: two benign terms were labelled pathogenic). Conformal calibration
  uses the selected score with a recorded substrate and the finite-sample order statistic. The reconstruction exporter is
  historical-execution denied; transition_policy.py is retired; the superseded patchers and the OOF alignment check refuse (gate
  G2.4); observability reads the selected blend's weights from a verified manifest. Acceptance:
  docs/package1/PACKAGE1_SET1C_DELIVERY.md and the set-1c census ledger. Suite 8,922 -> 8,959 (+53 / -16).
  VALIDATED ON WINDOWS AND MERGED 2026-11-08: one -Validate run (09:38 to 11:01 UTC; receipt 1e0fd3f8): the whole suite 8,871
  passed, 83 skipped, 5 xfailed, exactly as forecast; targeted 1,309 passed, 5 skipped; drift 22 passed. Candidate 7255c2db
  (tree db34ee7a) published and merged as pull request #70 with a merge commit: 4a7e6d96, parents a29af854 and 7255c2db. The
  post-merge block (11:03 to 11:34 UTC) compared all 1,314 targeted tests by node identity: no difference; the branch was deleted
  (11:34 to 11:36 UTC), after the post-merge record was written (owner ruling 2026-11-08, section 2.1).
- **2026-11-08 -- PACKAGE 1D, STAGE 1D-0: Run 15's evaluations are INVALIDATED_EVALUATION; a claim-disposition registry.**
  Owner ruling of 2026-11-08 (decision file 56ca0eff), extended by measurement. Run 15 ran at commit 4b9de1c3 (2026-06-05), before
  the train-only gene count and the gene-grouped stacking folds (both 2026-06-13, and both descend from it): MEASURED on the run's
  own files, every validation, test and unseen-gene-holdout row's `n_pathogenic_in_gene` equals the pathogenic labels of the
  evaluated rows of its gene, its own label included (0 mismatches over 146,329, 304,711 and 213,436 rows; the feature alone
  reaches AUROC 0.70 to 0.72); the splits grouped gene-symbol STRINGS, so 15.66%, 19.60% and 8.28% of those rows share a gene with
  training; the graph network that scored the holdout was trained on its labels; LOVD's own clinical class was an input. The C3
  claim, the test evaluation and the validation evaluation are INVALIDATED_EVALUATION -- not FAIL: no hypothesis is resolved and
  no inflation is estimated. docs/claims/ holds one record per claim, WRITTEN from the run's files by
  scripts/derive_run15_claim_dispositions.py (every input bound by SHA-256, the run's code facts read from the preserved
  pre-rewrite history; --check re-derives byte for byte), hash-chained and append-only; evaluation/claim_disposition.py reads it
  (require_admissible refuses an invalidated claim) and gates the current-claim documents: README, METRICS, this roadmap and the
  Run 17 scope may cite a Run 15 figure only beside its disposition and claim identifier. The ablation record and a probe's
  docstring, which stated the opposite of the measurement, carry dated corrections. The live driver's unseen-gene holdout -- the
  same design -- now records its C3 verdict as WITHHELD with the defects named. Open: GENE-IDENTITY-SPLIT-1 (the live splits still
  group gene-symbol strings: real_data_prep._gene_aware_split, split_protocol_v2._resolve_genes). Next: stage 1D-1. Suite 8,959 ->
  9,016 (+57 / -0).
  VALIDATED ON WINDOWS AND MERGED 2026-11-08: one -Validate run (receipt GVC_package_1d0_validation_20261108T201522Z, sha256
  01797bf8; candidate 3b9f112e, tree 7780f7a3; all 26 stages exit 0). The first post-merge block refused before merging, as
  designed; the second merged pull request #71 with a merge commit -- 7cb70ec2, parents 4a7e6d96 and 3b9f112e -- and compared the
  targeted tests by node identity (1,953 passed, 5 skipped, as in the validation; drift 22); the branch was then deleted.
- **2026-11-09 -- PACKAGE 1D, STAGE 1D-1a (set "ao"): the information-source census.** Owner rulings 25691d66 and c34eb717.
  Nonbehavioural. docs/census/FEATURE_CENSUS_1D1a.json reviews all 91 declared features and the 11 non-tabular information sources
  the graph code can build (five knowledge-graph relation sources, three STRING edge attributes, the STRING edge selection, two GNN
  supervision labels): meaning, producer, datasets, source state, supervision and threshold supervision, upstream overlap, timing,
  label-criterion use (ACMG/AMP evidence the curators assign the label from), disposition, reason and evidence. Consumption is
  MEASURED from the code (eleven surfaces; every consumer site classified, aliases included) and each entry's consumers must equal
  it; producers are bound by comment-insensitive semantic-closure digests; anchors must occur exactly once; admission is monotone
  over dependencies. information_sources/census.py verifies all of it and prints that scientific truth is NOT ESTABLISHED BY
  STRUCTURAL VALIDATION ALONE. Result, measured not targeted: 55 admit_strict, 8 conditional_only, 12 exclude_strict, 4
  retire_predictive, 12 withhold_pending_evidence; sources 4 / 5 excluded / 2 withheld. Registered: EVE-COVERAGE-IDENTITY-1 (the
  audit's EVE and default-indicator measurements are withdrawn: docs/census/measure_1d1_correction.json, re-derived from
  Run 15's files by scripts/measure_1d1_correction.py --check), MISSINGNESS-SEMANTICS-1,
  GNN-SCALE-SKEW-1, GNN-NODE-LABEL-COUNT-1, KG-DISEASE-RELATION-1 (ClinGen and OMIM shares_disease relations are accepted by the
  hetero-GNN), STRING-CHANNEL-ORIGIN-1 (edges selected on a score that includes curated pathways and text mining),
  CODON-POSITION-SEMANTICS-1, RNA-SPLICE-GATE-ORDER-1, GTEX-EQTL-PRODUCTION-DEFAULT-1. Next: stage 1D-1b, the admission authority
  and the observation contract. Suite 9,016 -> 9,056 (+40 / -0).
  VALIDATED ON WINDOWS AND MERGED 2026-11-09: one -Validate run (receipt GVC_feature_census_1d1a_validation_20261109T043508Z,
  sha256 f1b0f449; candidate e0fae59e, tree ff48e9a9; all 26 stages exit 0, the census stage included). Pull request #72 was merged
  with a merge commit -- f9736a8e, parents 7cb70ec2 and e0fae59e -- and the post-merge block compared the targeted tests by node
  identity (2,087 passed, 5 skipped, as in the validation; drift 22; both candidate environments still their Windows locks); the
  branch was then deleted.
- **2026-11-09 -- PACKAGE 1D, STAGE 1D-1b.1 (set "ap"): typed observations and SHADOW admission.** Owner ruling f2ec92cd.
  Nonbehavioural: no production module imports the new code (a test proves it). information_sources/observation.py is the
  observation contract stage 1D-1b.2 will make producers emit: six states; an observed value is a finite real number (booleans,
  NumPy's included, refused; NumPy numbers normalised to float); an absent state carries no sentinel; the state decides whether a
  source must be named by SHA-256 (observed, not in source, unsupported: required; source absent: forbidden). It imports only the
  standard library. information_sources/admission.py derives, from the census AFTER its gate re-measures the repository, a report
  that keeps three things apart -- admitted (per regime: strict, conditional), observed (not_recorded_by_producer for every name:
  no producer records it yet), available to the fitted model (the measured predictive surfaces) -- and records every name that
  reaches a predictive surface unadmitted, with its reason. Admission is TRANSITIVE (greatest fixed point over the measured
  dependencies), which the census enforced for admit_strict only: under the conditional regime gnn_score and hetero_gnn_score are
  conditional_only yet not admissible, because their node attributes include the retired and excluded features and their edges
  the excluded STRING selection. docs/census/ADMISSION_SHADOW_1D1b1.json is the report; --check re-derives it byte for byte.
  Result, measured: of 102 names, strict 55 admitted / 47 reach a model unadmitted; conditional 65 / 37 (2 only through a
  dependency). Registered for the stages that own them (ruling f2ec92cd; measured on the owner's STRING v12 human files, sha256
  28a2ea3b and 144de4b0): STRING-PAIR-ACCOUNTING-1 (an order-independent digest accepts a pair duplicated in both directions;
  exact pair accounting, 1D-1b.3; the owner's sharded audit accepts the real file: 178 s, 111 MiB memory, 854 MiB scratch, measured on a 2-core
  Linux container);
  STRING-THRESHOLD-RECONSTRUCTION-1 (rounded channels with a floored combined score fit 99.99514% of pairs, 333 miss by one
  thousandth; the measured-channel graph has 3 reconstruction artifacts outside the published one, so G1 = reconstruction
  INTERSECTED with the published graph, 84,987 edges; CORRECTED 2026-11-10 (owner ruling 7f4b69d0, section 7): "177
  rounding-sensitive" counted published-graph edges, of which 113 are G1 edges and 64 are excluded from G1 -- the threshold-robust
  G1 graph has 84,874 edges; 1D-1b.3); KG-COHORT-DEPENDENT-PROJECTION-1
  (kg_edges.co_membership_edges caps set size AFTER filtering to the cohort, so adding a gene can delete edges; 1D-1b.3);
  KG-RELATION-SOURCE-MERGE-1 (load_kg_edge_specs merges Reactome and KEGG into one shares_pathway relation, so a withheld and an
  excluded source cannot be told apart per edge; 1D-1b.3); STRING-GENE-MAPPING-1 (StringDBGraph.build maps by preferred_name,
  keeps an unmapped protein under its protein identifier, keeps the last row nx.Graph sees, and its name cache ignores
  force_refresh; measured on v12 human: 19,699 proteins, 19,699 distinct names, no collision, unmapped protein or self-loop --
  the realised risk is the namespace, preferred_name against HGNC identifiers; 1D-1b.3 and 1D-1d). Next: stage 1D-1b.2, source
  and observation materialisation. Suite 9,056 -> 9,077 (+21 / -0).
  VALIDATED ON WINDOWS AND MERGED 2026-11-10: one -Validate run (receipt GVC_admission_1d1b1_validation_20261109T233537Z, sha256
  ab93a82e; candidate c7c8422c, tree 04ba47bf; every stage exit 0, admission included). Pull request #73 was merged with a merge
  commit -- 18fa338e, parents f9736a8e and c7c8422c -- and the post-merge block compared the targeted tests by node identity
  (2,108 passed, 5 skipped, as in the validation; drift 22; both candidate environments still their Windows locks); the branch was
  then deleted.
- **2026-11-10 -- PACKAGE 1D, STAGE 1D-1b.2 (set "aq"): lineage integrity, source materialisation and the source-absence census.**
  Owner ruling 7f4b69d0. Shadow mode: no production module imports the new code (a test proves it); no model, feature, scaler,
  graph, split or inference behaviour changes. (1) ADMISSION-CYCLIC-LINEAGE-1 fixed by two independent mechanisms
  (information_sources/lineage.py): the census gate and the admission report refuse any cycle, naming every cyclic component
  (iterative Tarjan; self-loops included), and every leaf must name a registered dataset; admission is then the LEAST fixed point,
  evaluated bottom-up, so a cycle cannot admit itself even past the check -- equal to the 1D-1b.1 result on the acyclic census
  (tested). (2) Exposure is graded (DECLARED, STATICALLY_REACHABLE, MATERIALIZED, FITTED_INPUT, INFERENCE_INPUT): the report
  establishes the first two only, renames available_to_fitted_model to statically_reachable and violations to
  structural_exposure_risks (docs/census/ADMISSION_SHADOW_1D1b2.json, replacing ADMISSION_SHADOW_1D1b1.json). (3) The observation
  contract separates the LOGICAL source from the bytes read (observation.SourceIdentity; the owner's digest table) and adds
  INPUT_UNOBSERVED for derived values; columns.py holds a column per feature (float64 values, uint8 states, uint16 reasons)
  reconciled against an ordered population, and the propagation rule of every derived feature, reconciled with the census.
  (4) materialization.py binds the parser to the hashed bytes (one open, one pass; an early stop or unread remainder refuses),
  writes receipts from what it measured (parser identity = the materialiser's semantic-closure digest), and pilots EVE and gnomAD
  allele frequency, each reproducing its legacy connector's join exactly. (5) The source-absence census
  (absence_census.py, docs/census/SOURCE_ABSENCE_1D1b2.json) executes engineer_features on absent, NaN and varied frames and the
  annotation chain hermetically: 67 raw inputs (execution equals the source text); 89 of 91 features fabricate a value under
  absence, silently; "never joined" and "joined, not in it" are indistinguishable for all; absence alone sets af_is_absent,
  af_is_ultra_rare, is_snv and revel_pathogenic to 1; every derived feature reads only through its declared dependencies.
  (6) The pilots on the REAL source joins (docs/census/PILOTS_1D1b2.json; scripts/measure_1d1b2_pilots.py; EVE's 3,211 files,
  10.65 GB, parsed on the owner's machine, every file equal to the connector's own parse), on the Run 15 cohort (1,490,014 rows,
  Run 15's count): gnomAD frequency observed for 992,044 rows, the legacy join reproduced on every row; 497,970 rows (33.4%) have
  none and become "ultra-rare" -- 67.0% of pathogenic against 27.9% of benign rows (AUROC of the missingness alone 0.695); 40,419
  observed zeros are indistinguishable from them. EVE observes 91,164 rows (no observed score is exactly 0.5); 280 rows would be
  duplicated by the legacy join; the legacy join is reproduced on the other 1,489,734. REVEL is unscored for 1,169,388 rows, every
  one flagged pathogenic (90.1% of benign rows). Registered: ABSENCE-AS-EVIDENCE-1, GNOMAD-CHRX-LOCUS-PARSE-1 (every chromosome-X
  gnomAD record dropped by the locus parse: 63,929 cohort rows without a frequency; autosomes 990,350 observed, Run 15's logged
  count exactly), EVE-DUPLICATE-KEY-MULTIPLIES-ROWS-1. The pilot population is the Run 15 cohort built from
  clinvar_grch38_clean.parquet (1,490,014), not the canonical-review cohort of section 1 (1,490,324). Fixes: 1D-1c.
  Suite 9,077 -> 9,136 (+66 / -7 by node identity).
  VALIDATED ON WINDOWS AND MERGED 2026-11-10: one -Validate run (receipt GVC_admission_1d1b2_validation_20261110T055710Z, sha256
  18f5ba72; candidate 688f1caa, tree bc756c86; every stage exit 0, admission and absence included; 9,048 passed, 83 skipped, 5
  xfailed -- the forecast). Pull request #74 was merged with a merge commit -- 5b7335f6, parents 18fa338e and 688f1caa -- and the
  post-merge block compared the 2,379 targeted tests by node identity (2,374 passed and 5 skipped in the checkout; the
  validation's clone skipped 2 more, the local-data tests, which need data/external; no other difference; drift 22; both
  candidate environments still their Windows locks); the branch was then deleted.
- **2026-11-10 -- PACKAGE 1D, STAGE 1D-1b.3 (set "ar"): source-bound graph identity and authorized labels.** Owner rulings
  f2ec92cd (Unit 1D-1b.3), 7f4b69d0 (sections 7-10) and 7a2a6ade (section 13). Shadow mode: no production module imports the new
  code (a test proves it); no model, feature, scaler, graph, split or inference behaviour changes; no GNN is retrained.
  (1) information_sources/graph_identity.py reads STRING in ONE verified pass and matches the two directions of every pair
  EXACTLY in disk-backed partitions, refusing a missing reciprocal, a contradictory direction, a duplicate, a self-pair or a
  non-canonical score. On STRING v12 human (sha256 28a2ea3b): 13,715,404 records = 6,857,702 pairs x 2, 19,622 proteins, every
  pair symmetric; content identical for every layout; 54.3-58.4 s, 629-820 MiB peak, 342,885,100 bytes of scratch on a 2-core
  container (docs/census/STRING_READER_BENCHMARK_1D1b3.json). (2) The threshold is integer thousandths; G1 = restricted
  reconstruction (the five non-curated channels) >= 0.700 AND published >= 700; every comparison within floating-point error of
  the threshold is decided in exact rational arithmetic (52 pairs reconstruct to exactly 0.700); the four-way status with the
  published flag: selected robust 84,874, selected sensitive 113, excluded sensitive 64 + 9, excluded robust 151,879 +
  6,620,763 (the 1D-1b.1 figures, now from the exact reader); quantization model consistent for 6,857,369 pairs (333 miss by
  one thousandth). (3) Canonical identity: STRING preferred names resolved against HGNC (approved, else a unique previous
  symbol; aliases not used): 18,837 + 479, 1 ambiguous, 382 unresolved, and 4 HGNC identifiers claimed by two proteins after
  canonicalisation (approved-symbol precedence, reported). (4) GraphPolicy (regime, channels, threshold, quantization, edge
  reduction, canonicalisation, source-defined node universe, pinned sources) and GraphMaterialization (source, mapping, node
  universe, relation policy, edge content, schema) computed from the built graph; accept_graph recomputes every digest and
  refuses any mismatch. G1 83,613 gene edges, threshold-robust G1 83,501, G2 234,358, on 19,312 HGNC nodes; no STRING regime
  is strictly admissible (the census excludes the combined-score selection). The legacy StringDBGraph built 236,930 edges on
  16,201 preferred-name nodes and, given a different links file, returned the same cached graph: its cache key holds only the
  threshold (3 policies, 1 filename). (5) Pathway relations keyed by (source, relation) and (source, stable pathway id),
  weighted 1/(|S_s| - 1) by the GLOBAL canonical size: on ReactomePathways.gmt (sha256 cf17607f) and the Run 15 splits, the
  legacy projection loses 180,843 of its 732,550 train-train edges when the held-out genes join the universe, the global-size
  projection loses none (induced-subgraph consistency 0 / 0 / 0); the factorised hypergraph operator equals the projection
  (6e-14); the hypergraph model is stage 1D-3. (6) information_sources/labels.py: the label authority derives the fit's scope
  from canonical genes; the fold-scoped store records and refuses reads; the legacy hetero and homogeneous graph builders FAIL
  label non-interference when handed raw labels and pass on the authority's masked labels; every graph-label site is reviewed
  (docs/census/LABEL_SITES_1D1b3.json). On the Run 15 cohort the authority refuses the 206 training rows of the 14 genes whose
  two symbols straddle the split. Registered in the census ledger: STRING-PAIR-ACCOUNTING-1,
  STRING-THRESHOLD-RECONSTRUCTION-1, STRING-GENE-MAPPING-1, KG-COHORT-DEPENDENT-PROJECTION-1, KG-RELATION-SOURCE-MERGE-1,
  GRAPH-NODE-UNIVERSE-SPLIT-DEPENDENT-1 (the run driver's hetero node universe is the union of the split genes) and
  GENE-SYMBOL-SPLIT-IDENTITY-1 (owner: 1D-1d); dataset hgnc. Records: docs/census/GRAPH_IDENTITY_1D1b3.json. Admission
  unchanged: strict 55 / 47, conditional 65 / 37. Suite 9,136 -> 9,210 (+74 / -0, revision B included).
  REVISION B (2026-11-10). Windows V1 of the first candidate STOPPED (record GVC_admission_1d1b3_validate_run_20261110T112255Z,
  checkout verified unchanged): 6 tests in test_graph_identity.py failed because their fixtures, written with Path.write_text,
  had CRLF line endings on Windows (the Linux sandbox cannot see newline translation). The investigation found a latent SILENT
  defect behind it: a CRLF GMT made the last member of every gene set "unresolved", and a CRLF HGNC file was accepted whenever
  its required columns were not last. The measured sources hold 0 carriage returns, so no recorded number changes. Every reader
  now refuses a carriage return by name, in its header and in every row; fixtures are written as bytes; an AST guard keeps write_text without newline= out of
  the test modules that feed these readers; the sandbox now also runs the stage's tests under Windows newline emulation, which
  reproduces exactly the 6 Windows failures on the first candidate. Owner ruling 68ca322f registered: LABEL-TYPE-IDENTITY-1,
  LABEL-DENIED-READ-AUDIT-1, GRAPH-MAPPING-RECEIPT-AUTHORITY-1 (root cause: map_string_proteins takes the info digest as an
  argument, so a mapping of DIFFERENT identities claims the pinned source and is accepted), GRAPH-MUTABLE-MATERIALIZATION-1 --
  each with an executable witness (tests/unit/test_registered_defect_witnesses.py) that fails once it is repaired --
  VARIANT-MULTIGENE-ISOLATION-1 (measured: 729 multi-gene gene fields; as canonical gene sets, 419 genes on both sides of the
  Run 15 split and 309 linked gene groups straddling it, 219,128 rows; largest group 11 genes, so component-based splitting is
  feasible; owner 1D-1d), LABEL-ACCESS-DISCOVERY-SCOPE-1 and STRING-PROTEIN-GENE-REDUCTION-1. None is repaired here: the
  shadow scope is kept, as the ruling directs. Suite 9,197 -> 9,210 (+14 / -1 by node identity: one parametrized case renamed); against main +74 / -0.
  VALIDATED ON WINDOWS AND MERGED 2026-11-14: one -Validate run of revision B (2026-11-10 18:29 to 19:54 UTC; receipt
  GVC_admission_1d1b3b_validation_20261110T182927Z, sha256 98244de6; candidate 534bde78, tree 61de20ef; all 28 stages exit 0;
  9,122 passed, 83 skipped, 5 xfailed -- the forecast; test by test against set "aq"'s Windows run, the 9,136 common tests
  identical and the 74 new ones passed). The branch was published (record 7b936de2) and pull request #75 merged with a merge
  commit -- afbb8881, parents 5b7335f6 and 534bde78 (2026-11-14 06:28 UTC).
- **2026-11-14 -- PACKAGE 1D, STAGE 1D-1b.4 (set "as"): population identity, fit authorization, claims, and the repaired label
  and graph contracts.** Owner ruling 68ca322f and the owner's reference packages cee6713d ("TaskBound"), 7ba10b3b ("TaskScoped")
  and 7b277d7c ("EvidenceScope"), audited, probed and synthesized on 2026-11-13 (AUDIT_1D1b4_REFERENCES_2026-11-13.md). Shadow
  mode: no production module imports the new code (a test proves it); no model, feature, scaler, graph, split or inference
  behaviour changes. (1) information_sources/population.py: PopulationRow (strict str identities; variant identity qualified by an
  enum, VRS-verified only by the 1D-1d authority; a gene SET plus the tokens that did not resolve, one of which makes the set
  unknown); resolve_gene_field (prose flagged); split components over variant identity and every listed HGNC locus; the
  three-way audit of the partitions a declared protocol requires; two fit authorities: STRICT (refused unless the split is
  supported) and CONDITIONAL_COMPONENT (exploratory: a gene-complete training row whose component, built over the label-carrying
  training rows and every held-out row, holds no held-out row); tempered component weights with the effective sample size; the
  component-disjoint, class-stratified checkpoint split. (2) information_sources/claims.py: claim kinds and tiers, a complete
  scope per evidence kind (a split fact does not depend on the model; upstream independence and the statistical result depend on
  the evaluation population, so population A's evidence never certifies population B -- both references let it), undefined
  dimensions NOT_EVALUATED, pins by content digest and issuer, rules that receive their scope, eligibility and finding as
  separate axes; within-unit AUROC with two estimands and a paired unit bootstrap. (3) labels.TrainingLabels replaces the
  1D-1b.3 label authority and store: built from a FitAuthorization, it holds authorized native-integer labels only, records
  every read (authorized, denied, unknown) and every masked bulk read, and cannot be copied. (4) graph_identity: StringInfo
  (content and digest as one object); map_string_proteins takes no caller digest; a graph holds its mapping; accept_graph
  re-derives the graph from the pinned files (verify_graph_consistency is the self-check); bytes-backed read-only arrays and
  read-only mappings throughout. Measured on the Run 15 split manifests and the pinned HGNC file
  (docs/census/POPULATION_ISOLATION_1D1b4.json, 117 s, 1,969 MiB): 1,490,014 rows, every variant identifier and every (chrom,
  pos, ref, alt) distinct; 16,964 components (16,599 gene groups and 365 rows with no resolved gene), 327 spanning partitions;
  shared genes and components training-validation 145 / 115, training-test 291 / 229, validation-test 54 / 53; STRICT refused
  (violated); CONDITIONAL_COMPONENT authorizes 910,273 training rows (1,038,974 - 266 gene-incomplete - 128,435 in components
  holding a held-out row); a gene-direct rule would admit 14,967 more (26.73% pathogenic), each joined to a held-out row through
  another row; the 1D-1b.3 one-gene rule's 206 refused rows are all excluded; pathogenic 14.09% of training rows, 13.32% of the
  authorized, 19.58% of the excluded; effective sample size 81.5% / 36.6% / 4.76% of rows at alpha 0.25 / 0.5 / 1; the
  checkpoint split's inner validation is 13.319% positive against 13.320%. Re-measured on the same files with the repaired
  code: docs/census/GRAPH_IDENTITY_1D1b3.json (every graph accepted by re-derivation; every measured value identical; its label
  section moved to the population record) and docs/census/STRING_READER_BENCHMARK_1D1b3.json. Repaired, with their witnesses and
  registrations removed: LABEL-TYPE-IDENTITY-1, LABEL-DENIED-READ-AUDIT-1, GRAPH-MAPPING-RECEIPT-AUTHORITY-1,
  GRAPH-MUTABLE-MATERIALIZATION-1. Registered, with executable witnesses: GNN-EARLY-STOP-ROW-SPLIT-1 (the GNN early-stops on a
  row-level split: 155,488 of its 155,847 early-stopping rows share a gene symbol with its training rows; owner 1D-3) and
  GENE-FIELD-PROSE-1 (32 rows hold ClinVar prose in gene_symbol; owner 1D-1d). Admission unchanged: strict 55 / 47,
  conditional 65 / 37; semantic digest unchanged (97f64bcb). Still the owner's for 1D-1d: which loci bind a split group
  ((a) every listed HGNC locus, recommended and implemented here, or (b) protein-coding loci only). Suite 9,210 -> 9,282 (+86 /
  -14 by node identity).
- **KNOWN LIMITATION (2026-10-08) -- the frozen environment pins S4Vectors 0.50.1.** Upstream 0.50.3 (commit 841b1d2, 2026-09-16)
  fixes a comparator that ignored the fourth key: in 0.50.1, matchIntegerQuads and selfmatchIntegerQuads with method "quick" return no
  match for elements sharing the first three keys with an earlier element. No affected call was identified within the examined
  implementation (explicit method "quick" callers searched in the R code of ten Bioconductor dependents and of this repository) and
  the intended workload (method "auto" selects "quick" only above 2^29 elements). That is a scoped finding, not proof of
  unreachability. REQUALIFICATION TRIGGERS: new code calling these functions, or duplicated / match / unique on genomic ranges, with
  method "quick"; or inputs approaching 2^29 elements. Separately, 22 of the 102 pinned packages were behind the live indexes on
  2026-10-08 -- a currency observation, not a defect; package updates are a separate reviewed change.
- **SCHEDULED -- legacy-roadmap carry-forward audit.** 47 of the 55 item identifiers in
  docs/archive/legacy/ROADMAP_2026-03_to_2026-08-22.md are not mentioned in this roadmap (measured 2026-09-28).
  Absence is not resolution: SMOKE-1 (tests/smoke_test_imports.py is never collected and no workflow runs it) is
  verified still open. Each identifier needs its last recorded status and a check against current code.

- **2026-09-23 -- test-isolation prerequisite.** The suite no longer writes into
  the repository. Measured before: a full run left 38 files (CatBoost training
  files, 32 checkpoint files, an outputs/ mismatch file, a freshness report),
  truncated logs/train.log at collection, and rewrote the orchestrator's
  agent_state.json. After: a content-digest inventory of the whole tree is
  unchanged. Suite 6,745 -> 6,758 collected (+13, 0 removed). Details in
  `docs/CHANGELOG.md`. CORRECTED 2026-10-12: not true for the graph-neural-network
  tests, which that check could not run (PyTorch absent); see
  `docs/sessions/CORRECTION_2026-10-12_the-suite-still-wrote-into-the-repository.md`.

- **2026-08-23 -- authority succession.** The predecessor was preserved verbatim
  at the archive address above and this document took the live path, in one
  atomic transaction. Blob object identity proves the archived bytes are the
  bytes that were live. `docs/CHANGELOG.md` remains the per-session record and
  is unaffected.
