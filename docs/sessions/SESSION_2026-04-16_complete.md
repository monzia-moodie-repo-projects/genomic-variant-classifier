# Session 2026-04-16 (complete) — Run 8 + GNN/AM fixes

## Run 8 Results
- Holdout AUROC: 0.9863  AUPRC: 0.9461  MCC: 0.8482  F1: 0.9226  Brier: 0.0358
- OOF blend: 0.9938 (RF 0.9921, XGB 0.9932, LGB 0.9930, GBM 0.9891, CatBoost 0.9930)
- AlphaMissense: 206,131 variants annotated, ranked 7th of 78 features
- Infrastructure: Vast.ai RTX 4090, $0.388/hr, 4270s, 1.8GB artifacts in GCS

## Bugs fixed this session
1. variant_ensemble.py SyntaxError: Phase 2 code inside unclosed assert ( — fixed
2. TABULAR_FEATURES frozen at 21 vs 78 produced by engineer_features() — fixed
3. AlphaMissense _parse_parquet: wrong 5-col schema returned instead of lookup_key/score — fixed
4. AlphaMissense stale cache with wrong schema — deleted on instance
5. GNN: int(string_db_path) ValueError — fixed; string_db_path wired to _local_links

## Bugs open for Run 9
- TF models (tabular_nn, cnn_1d, mc_dropout, deep_ensemble): no tensorflow on PyTorch image
  Fix: pip install tensorflow on instance, or migrate to PyTorch equivalents
- ESM-2: stub mode (no transformers installed) — pip install transformers on instance
- SpliceAI: 0 variants annotated — connector reads parquet as TSV; needs parquet branch like AM

## Infrastructure learnings
- Vast.ai auto-starts tmux — no manual tmux new-session needed
- Vast containers lack systemd — sudo shutdown fails, container exits naturally
- SA key parallel composite upload GET check fails — non-blocking warning only
- All training data now in GCS — future relaunches pull in ~3 min, no scp needed

## Commits this session
- d94b000 feat: sync TABULAR_FEATURES 78; Lambda startup; Phase 2 join methods
- de19e95 docs: SESSION_2026-04-16 + CHANGELOG
- 3326636 fix(alphamissense): parquet index support; _parse_parquet branch
- 5297711 fix(alphamissense): _parse_parquet return schema
- 8291bff docs: CHANGELOG AM fix + Run 8 launched
- f7c89d4 docs: Run 8 complete
- (pending) fix(gnn): string_db path vs int threshold
