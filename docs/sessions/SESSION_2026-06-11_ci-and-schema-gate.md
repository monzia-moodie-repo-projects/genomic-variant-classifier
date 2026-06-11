# SESSION 2026-06-11 -- CI optional-deps fix + schema-drift activation and preflight gate

## Session Overview

This session closed the prior agent-layer repair docs, fixed a CI regression caused by
undeclared optional dependencies, and then activated the SchemaDriftMonitorAgent end to end:
built a versioned Run-15 schema baseline, added a `from_baseline` loader, and shipped a
standalone preflight schema gate usable before any regen or training run.

HEAD evolved `4843ff2`-predecessor chain through `21d94c4` across 6 commits.

**Net result:** CI green again (verified at #304); schema agent moved from a capability-only
state to a real, exit-code-returning preflight gate with a committed 78-column contract;
full unit suite grew 873 -> 876 -> 880 passed (6 skipped throughout).

---

## Arc 1 -- Close of the 2026-06-10 agent-layer docs

- Closed the prior session's docs into the `docs/` canonicals (routing fix; root strays removed).
- Commit `0bbeb6d` (docs/session 2026-06-10 agent-layer).

---

## Arc 2 -- CI regression: agent-layer optional deps (pandera, river)

### Root cause
- CI was red at #302 (`21e835d`) and #303 (`0bbeb6d`); green through #301.
- `schema_drift_agent.py` imported `pandera.pandas` and `annotation_policy_agent.py` imported
  `from river import drift` at module top level. The Orchestrator imports every drift wrapper,
  so the entire agent layer became un-importable in CI, which installs only the declared
  requirements (pandera and river are declared nowhere). `pytest -x` masked the river failure
  behind the pandera collection error. scipy IS declared, so label_shift was unaffected.

### Fix (no requirements changed -- matches the repo's importorskip convention)
- pandera made lazy (imported inside `detect()`); river guarded by try/except (`river_drift=None`).
- `test_schema_drift_monitor_agent.py` switched to `pytest.importorskip`.
- Added `scripts/simulate_ci_no_optional_deps.py` -- reproduces the lib-absent CI env in-process
  to validate import-safety before pushing.

### Verification
- Local full suite unchanged (873/6); simulate gate exit 0; CI #304 (`92ff4a2`) green on 3.11 + 3.12.
- Incident filed: `docs/incidents/INCIDENT_2026-06-11_ci-optional-deps.md` (`66fe67f`).
- Placement script tracked (`4843ff2`).
- Lesson recorded in the incident: a run gate's "full suite green" must mean CI green (different
  env), not just local-venv green; the simulate script is the prevention tool.

---

## Arc 3 -- Schema-drift activation (first delivery against the 2026-06-10 backlog item)

### Schema baseline
- `scripts/build_schema_baseline.py` reads the sealed Run-15 `X_train.parquet` and writes
  `data/reference/schema/schema_baseline.json` (ordered `expected_dtypes` + sha256 hash + provenance).
- Captured contract: **78 columns, all float64**, hash
  `db43fd918bdfa4d0b096ba7df1c9c045bc5563e072803586b538200639df65bc`,
  source `outputs/run15_rerun_report/full/splits/X_train.parquet`.

### from_baseline loader
- Added `SchemaDriftAgent.from_baseline(baseline_path, output_dir)` (classmethod) that rebuilds
  the pandera schema from `expected_dtypes` with **`nullable=True`**. This is the critical detail:
  Run-15 has many degenerate (all-NaN / all-zero) columns; without nullable columns, validating
  an unchanged matrix against its own baseline would raise false nullability violations and report
  red. Validated with real pandera: unchanged-incl-NaN -> green/0 violations; mutated -> red.
- pandera imported lazily inside the method (keeps the layer CI-importable).
- `tests/unit/test_schema_drift_activation.py`: ok/green on unchanged (incl. NaN col), ok/red on
  drift, and default-construction-still-awaiting_baseline (preserves the existing contract).
- Commit `e0a76a1`; suite 873 -> 876.

### Preflight schema gate
- `scripts/run_schema_drift_check.py`: load baseline -> head-read a feature matrix -> detect ->
  print column/dtype diff -> exit code. Exit codes match `run_drift_monitor.py`: 0 green,
  2 drift, 3 usage/env error.
- Efficiency: reads only the first parquet batch (default 4096 rows). Parquet stores dtypes in
  its schema, so a head-read is dtype-exact while keeping memory bounded on full-cohort matrices.
- `tests/unit/test_run_schema_drift_check.py`: exit-code contract (0 / 2 / 3).
- Decision: committed `data/reference/schema/schema_baseline.json` as a **versioned contract**
  (`git check-ignore` returned nothing -> not ignored). A future regen that changes the schema now
  shows up as a reviewable diff in one file.
- Commit `21d94c4`; suite 876 -> 880.

### Real-data validation (the proof, not just unit tests)
- `run_schema_drift_check.py --matrix outputs/run15_rerun_report/full/splits/X_train.parquet`
  -> severity green, byte-identical hash (`db43fd91...` both sides), exit 0.
- `--matrix .../meta_train.parquet` -> severity red, exit 2: 18 added, 38 removed,
  15 dtype changes (float64 -> int64/float32), 53 pandera violations. Proves the gate fires on
  real data, not only on synthetic mutations.

---

## Findings logged this session

### Drift-wiring (three findings; remediation captured as ROADMAP action items)
1. The eight agent-layer drift MonitorAgents (incl. Schema) are registered in
   `Orchestrator._register_agents` but invoked by nothing: absent from `PIPELINE_DEFINITIONS`,
   and `run_agents.py --pipeline full` runs only the four framework agents. "Operational/registered"
   is not "scheduled/invoked".
2. `.github/workflows/drift_monitor.yml` is effectively inert: its GDrive download step is a stub
   that creates an empty dir, so the very next step hits "No reference splits available -- skipping"
   and exits 0; it also references the **stale** `outputs/phase2_with_gnomad/splits/` path
   (pre-Run-15).
3. `scripts/run_drift_monitor.py` covers distributional (PSI/KS/MMD) + label drift via
   `monitoring/drift_detector.DriftDetector`, but **not** schema/column/dtype drift -- the gate
   built this session fills that gap and is additive, not duplicative.

### Feature-count spread (TO VERIFY -- not asserted)
- Three figures coexist: notes say **79**, on-disk `X_train` is **78** (verified green by the gate),
  and `docs/ROADMAP.md` still says **"Live (64 features)"**. The `meta_train` diff shows the
  identifier/label/target columns (`label`, `variant_id`, `gene_symbol`, `consequence`,
  `pathogenicity`, `clinical_sig`, `fasta_seq`, ...) live in `meta_*`, separate from the 78 `X_*`
  features -- so "79" likely counted a meta/label column or `esm2_llr` prospectively. The
  split-contract count is 78. Reconcile the 64/78/79 spread before this hardens further.
- The `meta_train` "removed" list shows `af_1kg_afr/amr/eas/eur/sas` ARE in the 78-column
  `X_train` baseline, yet `ROADMAP.md` lists `population_1kg_af` under PHASE_4_FEATURES (pending).
  Likely present-but-degenerate placeholders; **verify whether populated or zero** and reconcile
  the "pending" label. Recorded as to-verify, not concluded.

### Observed, non-blocking (recorded so they are not re-investigated cold)
- The `0xc0000139` torch_scatter/torch_sparse fatal-exception tracebacks during collection are the
  KNOWN benign `importorskip` skip-path of `test_ablate_gnn.py` (GNN ablation). The suite still
  reports 880 passed / 6 skipped; this is not a failure.
- 220 warnings in the full run, dominated by a pandas Downcasting FutureWarning in
  `models/variant_ensemble.py` (`.fillna` calls, lines ~378-477), plus sklearn LR ConvergenceWarning
  (max_iter) and a LightGBM feature-names UserWarning. Pre-existing tech debt; the pandas
  Downcasting behavior changes in pandas 3.0 and should be addressed before that upgrade.

---

## Git chain (this session)

```
21d94c4 feat(schema-drift): preflight schema gate (run_schema_drift_check) + versioned Run-15 baseline contract
e0a76a1 feat(schema-drift): SchemaDriftAgent.from_baseline + activation tests; baseline builder
4843ff2 chore: track CI-fix docs placement script
66fe67f docs(incident): 2026-06-11 CI optional-deps (pandera/river) -- resolved, verified green at #304
92ff4a2 fix(ci): lazy-import pandera (schema) + river (annotation) so agent layer imports without optional deps; importorskip schema test
0bbeb6d docs(session): 2026-06-10 agent-layer repair -- 13 operational; routes to docs/ canonicals
```

## Verification summary

- Suite progression: 873 -> 876 (`e0a76a1`) -> 880 (`21d94c4`) passed, 6 skipped throughout.
- simulate_ci_no_optional_deps gate: exit 0 (import-safety preserved by lazy pandera in from_baseline).
- CI #304 (`92ff4a2`) verified green. CI for `e0a76a1` and `21d94c4`: confirm on Actions
  (expected green; new tests importorskip pandera/pyarrow and skip in CI).
- Real-data gate smoke: X_train green/0; meta_train red/2.

## Next steps

1. Confirm CI green for `e0a76a1` and `21d94c4` on Actions.
2. Reconcile the 64/78/79 feature-count spread; verify `af_1kg_*` populated-vs-placeholder.
3. Drift-wiring remediation (see ROADMAP action items 2026-06-11): pipeline-wire the drift agents;
   fix the stale path + GDrive stub in `drift_monitor.yml`; add the schema gate as a
   `drift_monitor.yml` step; reconcile the two drift systems into one documented entrypoint.
4. Continue populating reference baselines for the remaining seven drift agents
   (schema is now the worked example).
