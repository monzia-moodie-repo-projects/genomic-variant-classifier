# INCIDENT 2026-05-31 -- GNN score 100% zero in Run 14 (and prior)

**Status:** ROOT CAUSE IDENTIFIED -- operational gate-skip; no GNN code change required
**Severity:** MEDIUM -- the GNN modality was silently dead in every run that did not pass --string-db
**HEAD:** 18bbba1

## Summary
gnn_score is mean 0 / std 0 / min 0 / max 0 across all Run-14 splits (train 1,197,216;
val 154,404; test 349,067). The GNN contributed nothing to the ~0.9974 test AUROC.

## Root cause (from run14_master.log -- not inferred)
The GNN block is gated behind `if args.string_db:` (run_phase2_eval.py:272). The 05-26 run
passed string_db=None and the launch reported "STRING DB not found (GNN will skip)" (log lines
31/53/466/467). gnn_score kept its engineer_features default (0.0).

## Code assessment -- no GNN change needed
gene_symbol sourced from meta_train.parquet (Patch 6b, run_phase2_eval.py:300-325); injection
loop intact (389-412); Patch-6a split re-persist intact (433-438); GNNScorer is gene-level
(gnn.py:585, score(gene_symbol)->float). The code is correct; it never ran.

## Fix
1. Stage STRING DB on the VM (links + info under data/external/string/).
2. Pass --string-db auto (CWD = repo root; info path is hardcoded relative).
3. Hard pre-flight gate scripts/preflight_gnn.py: abort if GNN expected but STRING DB absent.
4. Post-condition: [GNN-TRACE] has_gene_symbol=True and gnn_score std > 0.

## Note for the leakage audit
gnn_score is gene-level; under gene-disjoint eval it is uninformative for unseen genes. Include
it in the ablation matrix. Verified-good in baseline pre-flight: STRING DB present (139.6 MB / 1.97 MB).
