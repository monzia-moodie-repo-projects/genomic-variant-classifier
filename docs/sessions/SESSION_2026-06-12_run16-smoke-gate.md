# SESSION 2026-06-12 -- Run-16 all-models smoke gate cleared

## Session Overview

This session drove the Run-16 `--fast` all-models smoke from a chain of silent-deadzone
and environment blockers to a clean, complete end-to-end run. CI was confirmed green first
(the historical red runs #316/#317 predated the ESM-2 Hub-flake skip-guard fee2e63). Each
blocker the smoke surfaced was a real Run-16 risk that the input preflight alone did not
catch -- which is exactly why the all-models smoke is the authoritative gate.

HEAD advanced 369b87e -> a7fe43e -> 5f068dc -> 9c037f1.

**Net result:** a complete `--fast` smoke over a real ref/alt cohort (724.0s): all 13 base
models trained, meta-learner blended, 81-feature matrix, both classes, every former deadzone
populated, evaluator report printed, all artifacts saved. ENSEMBLE_STACKER test AUROC 0.9934
[95% CI 0.9866-0.9984], AUPRC 0.9543, MCC 0.8572, Brier 0.0267. CI green. The gate is cleared;
remaining work is staging, not debugging.

---

## Arc 1 -- gnomAD-constraint wiring + preflight fifth check

### Root cause
- train.py built AnnotationConfig WITHOUT gnomad_constraint_path -> GnomADConstraintConnector
  ran in stub mode -> loeuf constant -> gene_constraint_oe (Run-15 #2 feature) would deadzone
  in a train.py-driven Run 16. Same wiring-gap class as ecd0474 (ESM-2). The engineer_features
  loeuf fallback was already correct; only the data-feeding flag was missing.

### Fix
- Added `--gnomad-constraint`, threaded gnomad_constraint_path into AnnotationConfig (0af34f3).
- Added preflight check #5 (gnomAD-constraint TSV present, >=1 MB) -- 76519f6.
- Smoke confirms: gnomAD constraint matched 1588/1681 variants, 1588 genes pLI>0.

---

## Arc 2 -- ReviewStatus cohort augment + preflight sixth check

### Root cause
- train.py hardcodes min_review_tier=3 (no CLI override). `_load_and_label` raises when the
  cohort lacks ReviewStatus and tier<5 (a deliberate guard against silently keeping all review
  levels). The seq-cohort rebuild for the CNN work had dropped ReviewStatus -> regen would abort.

### Fix
- Re-augmented clinvar_grch38_clean_seq.parquet via scripts/augment_reviewstatus.py from the
  ClinVar VCF CLNREVSTAT: matched 3,974,573; tier<=3 keeps 1,490,014/1,686,333 labeled (~88%).
  ReviewStatus is used to derive review_tier then dropped -> 81-feature count + schema unaffected.
- Added preflight check #6 (cohort ReviewStatus present) -- a7fe43e. Preflight now 6 checks,
  13 unit tests, exit 0 over the augmented cohort.
- The preflight passing exit 0 over a cohort the regen aborts on (ReviewStatus, and earlier the
  protein-coord coverage corruption) is the recurring lesson: the preflight is necessary but not
  sufficient; the all-models smoke is the authoritative gate.

---

## Arc 3 -- AlphaMissense cache OOM (bug + regen strategy)

### Root cause
- `_get_lookup` checks the cache first via `_load_cache("scores_hg38")` = `pd.read_parquet` on
  the entire ~71M-row alphamissense_scores_hg38.parquet -> a 16 GiB allocation -> OOM, regardless
  of cohort size. The cohort-filtering in `_parse_tsv`/`_parse_parquet` (whose comments cite a
  prior Run-15 OOM) is never reached when the cache exists. Cache schema confirmed
  [lookup_key:string, alphamissense_score:float] -- the 71M lookup_key strings are what balloon.

### Fix / strategy
- Workaround: move the cache aside (.OOMbak) -> connector falls through to the memory-bounded
  chunked `_parse_tsv` (500k-row chunks, cohort-filtered). Smoke then annotated 216/1681 with
  score!=0.5, no OOM.
- Regen strategy: ship ONLY the TSV (not the 740 MB cache) to Vast.ai so the regen takes the
  memory-safe chunked path regardless of box RAM. This also de-risks the regen, since the 16 GiB
  load is cohort-independent and would OOM a <~20 GB box.
- Optional later: cohort-filtered cache read (pyarrow iter_batches + per-batch isin on lookup_key)
  to keep cache speed without the 16 GiB load -- schema now known; not required.

---

## Arc 4 -- UTF-8 stdio + ASCII-clean (the cp1252 encoding crash)

### Root cause
- The smoke trained all 13 models and computed every metric, then crashed at PHASE 4 in
  evaluator.print_report on `print(sep)` where sep = U+2500 box-drawing -- a Windows cp1252
  console cannot encode it. Same family: the non-fatal `\u0394` (Greek delta) logging errors in
  variant_ensemble blend-AUROC and the `->` em-dash mojibake in connector warnings. Not a
  data/model bug; would not fire on Linux/Vast.ai (UTF-8).

### Fix
- `_force_utf8_stdio()` before logging.basicConfig (5f068dc) reconfigures stdout/stderr to UTF-8;
  the evaluator report (print->stdout) now completes. 4 unit tests.
- Under PowerShell `2>&1` the merged stderr handler stayed cp1252, so the Greek-delta `logger.info`
  still raised (cp1252 can encode em-dashes, not Greek). ASCII-cleaned variant_ensemble.py
  (U+0394->'delta', U+2014->'-'; 6 chars, 6 lines) -- 9c037f1 -- with a regression test asserting
  the module stays ASCII. evaluator.py keeps its report Unicode (stdout path works).

---

## Arc 5 -- The green smoke + watch-items

### Result (smoke, esm2_t6_8M for speed; 1681 variants, 1176 train / 330 test)
- All deadzones live: AlphaMissense 216 (score!=0.5); protein-coord coverage gate PASS 0.9600;
  ESM-2 delta 212/216, LLR 205/216 (11 wt_aa-vs-sequence mismatches ~5%); gnomAD constraint
  1588 pLI>0; 81 features; both classes.
- 13 base AUROCs; ENSEMBLE_STACKER test 0.9934. Best base random_forest 0.9942.

### Watch-items (carry to the full 1.49M regen -- NOT gate blockers; valid non-degenerate OOF)
- cnn_1d OOF 0.5132 / test 0.4782 (near/below random) -- reproducible across two runs; consistent
  with data-starvation at 1176 samples. If still ~0.5 at full scale -> real architecture/scaling
  bug to chase.
- kan OOF 0.8848 / test 0.7438 -- below the ~0.99 tree models; likely smoke-size.
- Feature importance top-3 are gene-level: gene_has_known_disease (14.0), consequence_severity
  (11.5), n_pathogenic_in_gene (9.8). On the full regen, re-confirm gene-disjoint splits and that
  gene-level counts are not computed across the test fold (the C3-vetted prevalence signal).
- protein structure mean pLDDT=50.0 = AlphaFold-structure stub default (not activated; known).
  Minor 225-vs-216 missense count difference between the structure step and ESM-2/coords steps.

---

## Next steps (dependency order)
1. Production-flag decision: confirm whether Run 16 also passes `--gnomad` (allele-freq),
   `--uniprot`, `--dbnsfp-path`, `--lovd-path`, `--finngen-path`, and their exact paths -- the
   smoke deliberately stubbed DbNSFP/OMIM/ClinGen/dbSNP/EVE/LOVD/Reactome/FinnGen.
2. Launch-contract doc (write against the confirmed flag set).
3. Vast.ai staging: ship the TSV (not the cache) + the 18.64 MB protein-coord index co-located
   under data/external/alphamissense/; verify on box read-only (~0.97, ~18 MB).
4. Split-regen with all flags -> schema baseline refresh 78->81 (build_schema_baseline.py ->
   run_schema_drift_check.py green) -> launch preflight (SSH host/port, instance, key) -> Run 16.

## Commits this session
- 0af34f3 fix(train): wire --gnomad-constraint
- 76519f6 feat(preflight): fifth check -- gnomAD-constraint TSV
- 369b87e docs: protein-coord index corruption + repair
- a7fe43e feat(preflight): sixth check -- cohort ReviewStatus
- 5f068dc fix(train): force UTF-8 stdout/stderr
- 9c037f1 fix(logging): ASCII-clean variant_ensemble.py
