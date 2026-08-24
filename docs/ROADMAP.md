# GenAssoc / genomic-variant-classifier - Living Roadmap

**Version: 2026-08-23 (v3)  |  Owner: Monzia Moodie**

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
archive. Every headline number below was measured from the package on
2026-08-23 and is named with its source, so that it can be re-derived rather
than transcribed.*

# 1. Project identity & goals

Production-grade multi-modal genomic disease-association program. Core: an ACMG/AMP-style variant pathogenicity classifier over ~1.49M cohort rows (from ~2.49M ClinVar missense), 95 features, 13-model ensemble + stacking meta-learner + STRING-DB GNN + KAN.

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

*We are in Phase D (data expansion). Foundation is done.*

# 3. Current state snapshot (2026-08-23)

**Every number here was MEASURED on 2026-08-23 at `f2b93ff`, and the source is
named.** Earlier snapshots -- 2026-06-10, 2026-07-12, 2026-07-15 and 2026-07-18
-- are in the archive, together with the supersession notices they wrote about
each other.

| Quantity | Value | Measured from |
|---|---|---|
| Tabular feature contract | **95** | `EXPECTED_TABULAR_FEATURE_COUNT`, and `len(TABULAR_FEATURES)` agrees |
| Phase-2 features (declared, not yet computed) | **0** | `PHASE_2_FEATURES` is empty |
| Phase-4 features | 4 | `PHASE_4_FEATURES` |
| Sequence features | 1 | `SEQUENCE_FEATURES` |
| Base-model roster | **13** | `len(VariantEnsemble().base_estimators)` on a live instance |
| Registered agents | **22** | `Orchestrator._register_agents()` -> `_agent_registry` |
| Test suite | **5,443 collected** | `tests/EXPECTED_SUITE_SIZE`, and the README badge agrees |

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
| AlphaFold structure | alphafold_plddt, solvent_accessibility, secondary_structure_context, dist_to_active_site, has_uniprot_annotation | free (AlphaFold DB) | stub-mode step; activation = data + config |
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

**Last re-derived 2026-08-23 at `b586778`.** The standing plan is quoted from
the archive verbatim below; what changed since 2026-08-08 is characterised, not
summarised, and the reason is given.

## The standing plan, quoted

From the archive's final `NEXT` section, dated 2026-08-08, reproduced exactly:

> **Commit C (SealedEvaluation)**, with both censuses now committed and citable.
> Then the BASELINE-1 repair across the README and roadmap, **DRIFT-1 with
> README-1**, **OP-1 step 5** against STEP K, **OP-2**, and **RETRAIN-GATE**
> last.

**Nothing in this section supersedes that.** The work committed between
2026-08-21 and 2026-08-23 was repository infrastructure: it did not touch Commit
C, BASELINE-1, DRIFT-1, OP-1, OP-2 or RETRAIN-GATE, and none of those is closed.

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

- Ensemble: RF, XGBoost, LightGBM, SVM (nystrom + bagged_rbf), LR, GBM, 1D-CNN, TabularNN + meta-learner; CatBoost; MC-Dropout; Deep Ensemble; KAN. Per-model comparison every run.

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

- **2026-08-23 -- authority succession.** The predecessor was preserved verbatim
  at the archive address above and this document took the live path, in one
  atomic transaction. Blob object identity proves the archived bytes are the
  bytes that were live. `docs/CHANGELOG.md` remains the per-session record and
  is unaffected.

