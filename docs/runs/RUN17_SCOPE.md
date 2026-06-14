# RUN 17 — Scope & Pre-Flight Gate

**Status:** NOT LAUNCHED. This document is the pre-flight gate. Every box below must be
checked (PASS) before any billable GPU time is spent. Author: Monzia Moodie.

> Sourced from: standing run-gates, `docs/ROADMAP.md` (§3 snapshot, §7 standing disciplines,
> §9 changelog), and a direct read of `scripts/run_phase2_eval.py` on 2026-06-13. All flag
> names and paths below were verified against the script, not memory.

---

## 1. Objective

Run 17 = the next full-cohort training run after Run 15 (sealed, commit `032a2ab`,
Test AUROC 0.9984). Its **single new purpose** is to **activate two feature groups that are
present-but-constant** in the current schema:

- **`gnn_score`** — via `--string-db auto` (STRING-DB v12, GNN over the PPI graph).
- **`af_1kg_afr/eur/eas/sas/amr`** (5 cols) — via `--kg <1000G_phase3_AF.parquet>` (the parquet MUST carry per-superpopulation AF columns; fill_population_af reads them -- connector landed 2026-06-13, a0ce407).

**This is a VALUE activation, not a schema change.** Both groups are already among the
81 columns in `data/reference/schema/schema_baseline.json`; Run 17 turns their values from
constant to real. The feature **count stays 81** and the **schema hash is unchanged**, so no
baseline rebuild is required (confirmed: numeric dtypes are family-identity under the new
dtype-family gate; commit `37d60be`).

**Out of scope for Run 17** (separate activations, do NOT bundle): `esm2_llr` realization
(needs the coord-index sync + `--esm2-model esm2_t33_650M_UR50D`), `maxentscan_delta`,
`reactome_pathway_count` value population, dbSNP/AlphaFold stub activation.

---

## 2. VERIFIED command surface (`scripts/run_phase2_eval.py`)

| Knob | Real flag | Notes (verified in script) |
|---|---|---|
| ClinVar cohort | `--clinvar` (REQUIRED) | only required arg |
| gnomAD | `--gnomad` | help example: `data/processed/gnomad_v4_exomes.parquet` |
| SpliceAI | `--spliceai` | |
| AlphaMissense | `--alphamissense` | |
| gnomAD constraint | `--gnomad-constraint` | default `data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv` |
| dbNSFP | `--dbnsfp-path` | default `data/external/dbnsfp/dbnsfp_clinvar_index.parquet` |
| **GNN / `gnn_score`** | **`--string-db auto`** | `'auto'` → config-default STRING file; trains GNN, overwrites `gnn_score` (L343-476) |
| **1000G / `af_1kg_*`** | **`--kg <parquet>`** | **a Phase-3 AF *parquet*, not a VCF; flag is `--kg`, not `--kg-path`** (L66-68, L227) |
| smoke subsample | `--max-train N` | subsamples train to N (L286-295) |
| model skips | `--skip-svm/-nn/-cnn/-kan` | **NONE of these for the all-models smoke** |
| folds | `--n-folds` (default 5) | |
| holdout | `--unseen-gene-holdout` | gene-disjoint eval (used in Run 15 UGH) |
| output | `--output` | run output dir |

> ⚠ `scripts/train.py` is a DIFFERENT entrypoint and has **no** `--string-db`, `--kg`, or
> `--max-train`. The Run 17 activation and the all-models smoke run through
> **`run_phase2_eval.py`** only.

---

## 3. Pre-flight gates (all must be PASS)

### Gate A — Code & tests green
- [ ] **Full suite green.** `python -m pytest -q` → **956 passed / 6 skipped**, zero new warnings.
      *(Last known: 956/6 at commit `37d60be`.)*
- [ ] **GNN test path runnable locally** (was a carried blocker). `python -m pytest tests/unit/test_ablate_gnn.py -v` → **passes, does NOT skip**.
      *RESOLVED this session:* `torch_scatter`/`torch_sparse` uninstalled → PyG native scatter; VersionMonitorAgent `pyg_abi_alert == ""`, companions `absent`.
- [ ] **`.fillna` downcast FutureWarning closed** (was a carried blocker). RESOLVED this session via `@_suppress_fillna_downcast` (commit `4d56423`); suite warnings 220 → 41.
- [ ] **Zero open BUG incidents**; all prior anomalies closed.
- [ ] **All `<DECISION>` resolved** — see §5; the `n_pathogenic_in_gene` computation-scope audit is OPEN and gates this run.

### Gate B — Schema gate green (against the CORRECT matrix)
- [ ] Run the schema gate on a matrix that **matches the 81-col baseline**:
      ```powershell
      python scripts/run_schema_drift_check.py --matrix models\smoke_run16b\splits\X_train.parquet
      ```
      **PASS = `RESULT: schema matches baseline (green)`.**
- [ ] After the Run 17 regen, re-run the gate on the **fresh** `X_train.parquet` → green (81 cols).
- [ ] ⚠ **DO NOT** validate against `outputs/run15_rerun_report/full/splits/X_train.parquet` —
      that matrix is **stale (78 cols)**, predating `esm2_llr` / `maxentscan_delta` /
      `reactome_pathway_count`; it returns RED by design.
- [ ] ⚠ **FOOTGUN — fix or avoid:** `build_schema_baseline.py` `DEFAULT_MATRIX` still points at the
      stale run15 path. Re-running it **without** `--matrix` would silently regress the baseline 81→78.
      Either pass `--matrix` explicitly every time, or repoint `DEFAULT_MATRIX` to the run16b-smoke
      (or Run 17) matrix before any rebuild.

### Gate C — Data availability (the activation prerequisites)
- [ ] **STRING-DB present** for `--string-db auto`: confirm the config-default STRING v12 links
      file exists on the target box. (Roadmap §4A: wired & healthy.)
- [ ] **1000G Phase-3 AF parquet present** for `--kg`: confirm the file exists and is the
      **parquet** the connector expects (`kg_path=Path(args.kg)`, L227).
      ⚠ Roadmap §4B lists 1000 Genomes as *scaffolded but DEAD/partial* — **this file may not exist
      yet.** If it does not: **DECISION required** — acquire/convert the Phase-3 AF parquet, or
      defer `af_1kg_*` to Run 18 and run Run 17 with `gnn_score` only (omit `--kg`).
- [ ] Core inputs present: `--clinvar`, `--gnomad`, `--spliceai`, `--alphamissense`,
      `--gnomad-constraint`, `--dbnsfp-path` (all at their known paths).

### Gate D — All-models smoke (NO billable training until this is PASS)
Run the **tiny, local** all-models smoke with both activations on, **no skips**:
```powershell
python scripts/run_phase2_eval.py `
  --clinvar data\processed\clinvar_grch38.parquet `
  --gnomad data\processed\gnomad_v4_exomes.parquet `
  --spliceai data\external\spliceai\spliceai_index.parquet `
  --alphamissense <alphamissense_index> `
  --gnomad-constraint data\external\gnomad\gnomad.v4.1.constraint_metrics.tsv `
  --dbnsfp-path data\external\dbnsfp\dbnsfp_clinvar_index.parquet `
  --string-db auto `
  --kg <1000G_phase3_AF.parquet> `
  --max-train 3000 `
  --n-folds 5 `
  --output outputs\run17_smoke 2>&1 | Tee-Object outputs\run17_smoke.log
```
- [ ] **No `Traceback`, no model errors, no unexpected skips** in the log.
- [ ] **Every base model produced a non-degenerate OOF** (RF, XGB, LightGBM, SVM, LR, GBM,
      1D-CNN, TabularNN, CatBoost, MC-Dropout, Deep Ensemble, KAN, GNN meta).
- [ ] **`gnn_score` non-degenerate (HARD GATE, `INCIDENT_2026-06-04`):** the in-run gate fires if
      `gnn_score` is constant after `--string-db`. Confirm a non-zero std in the log
      (`gnn_score mean=… std=…`), and as the post-hoc mirror run
      `python scripts/verify_gnn_score.py` → PASS. A constant `gnn_score` = silent injection
      failure = **BLOCK**.
- [ ] **`af_1kg_*` non-degenerate** (if `--kg` supplied): confirm the 5 columns have real
      variance, not the constant default, in the smoke matrix.
- [ ] ⚠ **GNN runtime caveat (measure-first):** `--string-db auto` trains the GNN for 100 epochs
      on the STRING graph regardless of `--max-train`; on the CPU laptop this may be slow. PROBE
      the smoke wall-clock first (roadmap §7 "no estimates without a probe"). If infeasible locally,
      run this exact smoke on the Vast.ai GPU box as a pre-billing dry run.

### Gate E — Environment / GPU / checkpoint / budget
- [ ] Target GPU selected at run time via `vastai search offers` (lowest $/hr suitable 4090,
      `dlperf>=80 pcie_bw>=12`); do NOT overpay.
- [ ] Per-estimator checkpoint + OOF saved right after each AUROC log; abort if any single
      checkpoint > 30 min.
- [ ] Symlinks `/workspace/{data,outputs}` → repo (`rm -rf` before `ln -s`).
- [ ] Budget stated and accepted before launch (Run 15 ref: ~11.5 h, ~$6 on a 4090).

### Gate F — Single preflight script
- [ ] **ONE** preflight script fills ALL run variables (instance id, SSH host/port, key
      `C:\Users\monzi\.ssh\id_lambda_run8`, repo paths, data paths) with validation, so no var is
      hand-typed at launch. ⚠ No such script was found in `scripts/` at audit time — **build it as
      part of Run 17 prep** if it does not exist.

### Gate G — Irreversible commands isolated
- [ ] `vastai destroy`, `rm -rf`, force-push live in a **separate paste block**, run only after
      manual verification (roadmap §7).

---

## 4. Launch command (full run — only after Gates A–G are all PASS)
Same as the Gate-D smoke **minus `--max-train`**, plus the gene-disjoint holdout, writing to the
real run dir:
```powershell
python scripts/run_phase2_eval.py `
  --clinvar data\processed\clinvar_grch38.parquet `
  --gnomad data\processed\gnomad_v4_exomes.parquet `
  --spliceai data\external\spliceai\spliceai_index.parquet `
  --alphamissense <alphamissense_index> `
  --gnomad-constraint data\external\gnomad\gnomad.v4.1.constraint_metrics.tsv `
  --dbnsfp-path data\external\dbnsfp\dbnsfp_clinvar_index.parquet `
  --string-db auto `
  --kg <1000G_phase3_AF.parquet> `
  --unseen-gene-holdout `
  --n-folds 5 `
  --output outputs\run17
```
(On the GPU box, paths are the symlinked `/workspace/...` equivalents.)

---

## 5. Open decisions & risks (resolve before launch)
- **`<DECISION>` — `n_pathogenic_in_gene` computation scope** (roadmap §5): confirm
  train-only-per-fold vs corpus-wide; if corpus-wide, recompute train-only to close the
  leakage question the UGH 0.9988 result left open. **Gate A blocks on this.**
- **1000G parquet existence** (Gate C): if absent → Run 17 = `gnn_score` only; `af_1kg_*`
  slips to Run 18.
- **GNN local smoke feasibility** (Gate D): probe first; may force the smoke onto GPU.
- **`DEFAULT_MATRIX` footgun** (Gate B): repoint or always pass `--matrix`.

---

## 6. Post-run documentation (non-negotiable, roadmap §7)
- [ ] `docs/CHANGELOG.md` append (Attempted/Failed/Fixed/Learned).
- [ ] `docs/sessions/SESSION_2026-…_run17.md`.
- [ ] Per-model algorithm comparison + metrics glossary update (`docs/METRICS.md`).
- [ ] `docs/ROADMAP.md` §3 snapshot + §9 changelog updated; Run 17 sealed-commit recorded.
- [ ] If `gnn_score`/`af_1kg_*` now real: move them out of the "deferred/constant" census line.
