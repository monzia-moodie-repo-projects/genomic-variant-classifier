# Session 2026-06-14 -- hetero_gnn_score live eval-overwrite + SchemaDriftMonitorAgent activation

Two feature commits on top of 23f0034 (2026-06-13 v2), then this docs touch.

## 1. hetero_gnn_score live eval-overwrite -- a54ef38

run_phase2_eval gains opt-in `--hetero-gnn` + `--kg-edges source:path`. A block placed after
the gnn non-degeneracy gate and before the ensemble eval, PARALLEL to and SEPARATE from the
gnn_score block (preserving the homogeneous-vs-heterogeneous comparison):

- builds the training df (X_train + gene_symbol from meta_train.parquet + acmg_label);
- excludes BOTH gnn_score and hetero_gnn_score from node features (no self-feeding);
- STRING interacts_with edges from --string-db (cohort-restricted) + KG relations from
  --kg-edges (reactome/kegg/go/clingen/omim);
- trains a HeteroGNNTrainer, scores every gene, overwrites hetero_gnn_score per split
  (val/test gene_symbol from meta_*.parquet), re-persists the split parquets;
- WARNS (not exit) on a degenerate result -- hetero_gnn_score is a comparison feature;
- wrapped in try/except: a missing dep or edge source degrades to a logged warning.

Two testable helpers in hetero_gnn_scorer.py: load_kg_edge_specs ('source:path' -> {relation:
edges}, cohort-restricted, multi-source-per-relation merged + deduped) and string_graph_to_edges
(nx.Graph -> cohort-restricted edge list). Until run with --hetero-gnn, hetero_gnn_score stays the
0.5 default, exactly mirroring gnn_score's default-until-activated behavior. +4 tests
(test_hetero_kg_wiring.py). Run-17 activation:
`--string-db auto --hetero-gnn --kg-edges reactome:data/external/reactome/ReactomePathways.gmt`.
ONLY REMAINING hetero item: schema_baseline regen 81->82 from the real matrix.

## 2. SchemaDriftMonitorAgent activation -- 6a05481

First delivery against "populate the 8 drift agents' reference baselines".

Root cause that all 8 drift agents report awaiting_baseline: Orchestrator constructs every agent
as `agent_cls(self._state)` (no detector/reference), so each hits `_detect -> None ->
awaiting_baseline` (DriftMonitorBase). The baseline lives in the DETECTOR, not the agent.

- `SchemaDriftMonitorAgent.from_default_baseline(state, *, matrix_path, baseline_path, output_dir)`
  loads the SchemaDriftAgent detector from the canonical baseline
  (data/reference/schema/schema_baseline.json -- the same path run_schema_drift_check.py and
  build_schema_baseline.py use). Matrix resolves arg -> GVC_SCHEMA_CURRENT_MATRIX env -> None.
  Missing baseline -> inactive agent (graceful), never raises.
- Orchestrator now prefers `from_default_baseline(state)` when defined, else `cls(state)` -- the
  single generic enabler the other seven reuse.

The schema agent runs active detection once a matrix is supplied; it is no longer awaiting its
*baseline*, only its run-time *matrix*. Verified end-to-end (test_schema_drift_baseline.py, 5 tests):
matching matrix -> ok/green; added column -> ok/red with columns_added; absent baseline ->
awaiting_baseline; GVC_SCHEMA_CURRENT_MATRIX resolution; bare ctor -> awaiting_baseline (unchanged).

### Buildability split for the seven remaining
- BUILDABLE NOW (no trained model): LabelShift (reference label distribution from cohort labels),
  Infrastructure (pinned packages + DAG hash), likely AnnotationPolicy + AdversarialSubmission
  (config/heuristic references).
- RUN-17-DEPENDENT (need predictions): Concept (NannyML CBPE AUROC + BBSE), Calibration (per-class
  posteriors + ECE), FairnessSubgroup (per-subgroup predictions).

## Suite
1000 -> 1004 (hetero wiring) -> 1009 (schema activation), 6 skipped, 41 warnings (all pre-existing:
LGBM feature-names, n_components>n_samples, lbfgs ConvergenceWarning; zero new). HEAD 6a05481 on
origin/main.

## Next
- Next buildable-now drift baseline: LabelShift or Infrastructure (each defines from_default_baseline;
  the orchestrator hook already routes it).
- Run-17 prep: schema_baseline regen 81->82 from the real matrix; then the Run-17-dependent drift
  baselines (Concept / Calibration / FairnessSubgroup).
