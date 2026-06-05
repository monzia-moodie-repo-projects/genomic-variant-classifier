# INCIDENT 2026-06-04 — GNN `gnn_score` degenerate (constant 0.5) in Run 15

## Status
DIAGNOSED. Root cause confirmed against live source
(`scripts/run_phase2_eval.py`, `src/genomic_variant_classifier/models/gnn.py`).
Fix designed, NOT yet implemented. Run 15 completed `TRAIN_OK | GNN_FAIL`; the
GNN contributes nothing to Run 15 (gnn_score importance = 0.0).

## Symptoms
Post-run verifier (`scripts/verify_gnn_score.py`) FAILED on all three splits and
wrote `GNN_VERIFY_FAILED`:
```
[FAIL] X_train.parquet: rows=1,038,974 nunique=1 std=0.000000 nonzero_frac=1.0000
[FAIL] X_val.parquet:   rows=146,329   nunique=1 std=0.000000 nonzero_frac=1.0000
[FAIL] X_test.parquet:  rows=304,711   nunique=1 std=0.000000 nonzero_frac=1.0000
VERDICT: DEGENERATE
```
In-run trace: `GNN scores injected into test split (mean=0.500)` /
`[GNN-TRACE] post-injection split=test ... min=0.5000 max=0.5000 std=0.0000`.
`feature_importance.csv`: `gnn_score,0.0`. The GNN itself trained fine —
`GNN training complete. Best val AUC: 0.6509` — so this is **not** a training
failure.

## Root cause (exact)
The score map handed to the injector is **empty**, so every lookup falls to the
default.

1. `run_phase2_eval.py` builds the GNN training frame as:
   ```python
   gnn_df = X_train.copy().reset_index(drop=True)
   gnn_df["gene_symbol"] = _meta_train["gene_symbol"].fillna("")...
   gnn_df["acmg_label"] = y_train.values
   ```
   `X_train` is the 78-column numeric feature matrix. `gnn_df` therefore has
   **no `variant_id` column** (only `gene_symbol` and `acmg_label` were added).

2. `GNNScorer.from_trainer` (gnn.py) builds its gene map only via `variant_id`:
   ```python
   vid_to_gene = {}
   if "variant_id" in variant_df.columns and "gene_symbol" in variant_df.columns:
       vid_to_gene = dict(zip(variant_df["variant_id"]..., variant_df["gene_symbol"]...))
   gene_accumulator = {}
   for vid, score in zip(dataset.variant_ids, proba):
       gene = vid_to_gene.get(str(vid), "")
       if gene:
           gene_accumulator.setdefault(gene, []).append(float(score))
   gene_scores = {g: float(np.mean(s)) for g, s in gene_accumulator.items()}
   ```
   With no `variant_id` column the `if` is False → `vid_to_gene == {}` →
   `gene` is always `""` → `if gene:` always False → `gene_accumulator == {}` →
   **`gene_scores == {}`**.

3. `build_pyg_dataset` compounds it: `vids.append(str(row.get("variant_id", "")))`
   yields all `""` (same missing column), so even a variant_id-keyed
   accumulation could not work.

4. `GNNScorer.score()` then returns `DEFAULT_SCORE = 0.5` for every gene
   (`self.gene_scores.get(gene, 0.5)` on an empty dict), and the injector
   `X_split["gnn_score"] = split_df["gene_symbol"].map(gnn_scorer.score)` writes
   0.5 to every row of every split. Hence `std=0` on train, val, AND test.

### Compounding structural issue
Even after populating `gene_scores`, the map would only contain **training**
genes. The Run 15 main split is gene-disjoint (0 shared genes), so val/test
genes would still miss and fall to 0.5. A correct fix must therefore score
**every node in the STRING graph**, not just training focal genes.

## Why it was not fatal (and surfaced only post-hoc)
The GNN block is wrapped in `except Exception` (soft-warning, "continuing
without GNN") and the launcher's `verify_gnn_score.py` gate is **informational**
— it correctly detected the degeneracy (`nunique`/`std` based — the verifier
is sound) and wrote `GNN_VERIFY_FAILED`, but it does not abort the run, so the
pipeline finished `TRAIN_OK`. No code path raised on a zero-variance feature.

## Fix (to implement before the re-run; behind the smoke gate)
1. **Inductive all-node scorer (gnn.py).** Add `node_genes: list[str]` to
   `SharedFocalGraph` (populate with `all_genes` in `build_pyg_dataset`,
   propagate in `subset()`). Add `GNNScorer.from_full_graph(trainer, dataset)`
   that runs **one forward over all nodes** (`focal_idx = arange(n_nodes)`),
   softmaxes to per-node P(pathogenic), and builds `gene_scores` keyed by
   `node_genes`. This yields scores for all ~16,201 STRING genes, so train and
   the gene-disjoint val/test all get real, varying scores. Switch
   `run_phase2_eval.py` to use it.
2. **Hard-abort on degeneracy (run_phase2_eval.py).** After injection, assert
   `gene_scores` is non-empty AND each split's `gnn_score.std() > 0` (e.g.
   `> 1e-6`); raise instead of soft-continue so a degenerate GNN fails the run
   immediately rather than completing `TRAIN_OK | GNN_FAIL`.
3. **Traceability (minor).** Carry `variant_id` into `gnn_df` from `meta_train`
   for per-variant provenance, even though the inductive scorer removes the
   dependence on it.
4. Keep `verify_gnn_score.py` unchanged — it did its job.
5. Verify via the ALL-MODELS smoke gate: a tiny no-skip run must show
   `gnn_score` with `std > 0` on all splits and a non-zero feature importance.

## Source needed to finalize
`scripts/verify_gnn_score.py` (confirm exact pass/fail thresholds) and
`src/genomic_variant_classifier/data/splits.py` (`unseen_gene_holdout_split`,
to confirm the gene-disjoint property of the main split).

## Relationship to Patch 6b (INCIDENT_2026-04-30)
Patch 6b fixed GNN **training** input (`gene_symbol` sourced from
`meta_train.parquet`), which is why the GNN now trains (val AUC 0.6509). This
incident is the **next** failure downstream: the trained scores never reach the
feature matrix because the scorer's gene map is empty. Patch 6b was necessary
but not sufficient.

## References
- `scripts/run_phase2_eval.py` (GNN block; `gnn_df` construction; injection loop)
- `src/genomic_variant_classifier/models/gnn.py`
  (`GNNScorer.from_trainer`, `build_pyg_dataset`, `SharedFocalGraph`)
- `docs/incidents/INCIDENT_2026-04-30_gnn-gene-symbol-keyerror.md` (Patch 6b)
