# INCIDENT 2026-06-16 — hetero `hetero_gnn_score` inert (0.5 default) in val/test under gene-disjoint splits

## Status
DIAGNOSED — root cause confirmed against live source (`scripts/run_phase2_eval.py`
hetero block; `src/genomic_variant_classifier/models/hetero_gnn_scorer.py`;
`models/hetero_gnn.py`). FIX IMPLEMENTED and validated torch-free; in-env smoke
confirmation pending (no torch in the diagnosis sandbox). The **per-split audit
detection has LANDED**; the scorer/eval fix ships as `apply_hetero_inductive_fix.py`
(pending the in-env smoke + push). The full-flag Run-17 smoke printed
`RESULT: GREEN -- safe to launch` despite the defect.

## Symptoms
The `--hetero-gnn` full-flag smoke injected real scores into train but the **0.5
default into val and test**:
```
[HETERO-GNN] train injected mean=0.296 std=0.0688 nunique=1411
[HETERO-GNN] val   injected mean=0.500 std=0.0000 nunique=1
[HETERO-GNN] test  injected mean=0.500 std=0.0000 nunique=1
```
For contrast, the homogeneous `gnn_score` (already on the `from_full_graph` path)
was healthy on every split in the same run:
```
GNN scores injected into train/val/test split ... nonzero_frac=1.0000  (std ~0.10 each)
```
So `hetero_gnn_score` carries signal in **training only** and is a constant 0.5 at
eval/inference — the heterogeneous modality is inert exactly where it is measured.

## Root cause (exact)
The hetero scorer's gene map contains **only training-focal genes**, and the splits
are gene-disjoint, so every val/test gene misses and falls to the default.

1. `run_phase2_eval.py` builds the hetero training frame from `X_train` only:
   ```python
   hetero_df = X_train.copy().reset_index(drop=True)
   hetero_df["gene_symbol"] = _ht_mt["gene_symbol"].fillna("")...
   cohort_genes = sorted({g for g in hetero_df["gene_symbol"] if g})   # TRAIN genes
   fg = build_hetero_focal_graph(hetero_df, string_edges, kg_by_rel, ...)
   ```
   `build_hetero_focal_graph` derives its node set from the frame it is handed
   (`genes = sorted({str(g) for g in variant_df["gene_symbol"].dropna()})`), so the
   graph has **only the ~1,413 train genes** as nodes.

2. `HeteroGNNScorer.from_trained(_htr, fg)` scores exactly those nodes:
   `gene_scores = {gene: score for gene in fg.node_genes}` → ~1,413 entries.

3. The injector maps `gene_symbol -> hetero_scorer.score(gene)`, and
   `score()` returns `DEFAULT_SCORE = 0.5` for any gene absent from the map.
   The production splits are **gene-disjoint** (0 shared genes), so **no** val/test
   gene is in the 1,413-entry map → every val/test row gets 0.5 → `std=0`,
   `nunique=1`.

This is the same focal-only flaw that `INCIDENT_2026-06-04` fixed for the
homogeneous `gnn_score`. That incident's "Compounding structural issue" section
stated the rule explicitly: *"the map would only contain training genes ... a
correct fix must score every node in the STRING graph, not just training focal
genes."* The hetero scorer (added 2026-06-13/14, after that fix) reused the
focal-only `from_trained` path and reintroduced the flaw for the new modality.

## Why it was not fatal — and why it surfaced only post-hoc (the blind spots)
Three independent guards each failed to catch it:

1. **Train-only degeneracy guard.** The eval-side non-degeneracy check inspected
   **only `X_train`**:
   ```python
   _hchk = X_train["hetero_gnn_score"]
   if _hchk.nunique() <= 1 or float(_hchk.std()) == 0.0: logger.warning(...)
   ```
   `X_train` was non-degenerate (`nunique=1411`), so no warning fired. The val/test
   constancy was structurally invisible to a train-only guard.

2. **Soft-continue + "comparison feature" framing.** The hetero block is wrapped in
   `except Exception` (warn-and-continue), and `hetero_gnn_score` is treated as a
   comparison feature, so even an outright degenerate injection only warns; it never
   aborts the run.

3. **Audit concatenated the splits.** `audit_smoke_feature_population.py` pooled
   `X_train`+`X_val`+`X_test` into one frame before checking `nunique`. Train's
   variance (`nunique=1411`) masked val/test's constancy in the pool, so the one
   tool meant to catch dead features reported `hetero_gnn_score` as POPULATED.

Net effect: the defect is **not metric-fatal** — a feature that is constant on the
eval splits cannot inflate the held-out AUROC, so the reported numbers are honest —
but the heterogeneous-vs-homogeneous comparison (a first-class project goal) was
**not actually being evaluated**, the model wastes a feature slot, and it risks
overfitting to a train-only gene-identity shortcut.

## Fix (implemented — `apply_hetero_inductive_fix.py`)
Port the `from_full_graph` pattern to the heterogeneous scorer; mirror the regular
GNN exactly so the two modalities stay methodologically comparable.

1. **`hetero_gnn_scorer.py`** — `build_hetero_focal_graph(..., genes=None)` gains a
   `genes` override. When supplied, the **node set spans the train+val+test union**,
   while node features and focal supervision are still computed from `variant_df`
   (TRAIN). Val/test genes become real nodes with **zero features**, scored by graph
   structure. `node_genes = list(genes)` in `build_hetero_gene_graph` keeps every
   passed gene as a node (it sanitizes only edges), so the union genes are all present.
2. **`hetero_gnn_scorer.py`** — added `HeteroGNNScorer.from_full_graph(trainer, fg)`,
   a faithful sibling of `GNNScorer.from_full_graph`, scoring every node of `fg`.
3. **`run_phase2_eval.py`** — `cohort_genes` is now the train+val+test gene union
   (loaded from `meta_val`/`meta_test`); STRING `interacts_with` and KG
   `shares_pathway` edges span that union; the graph is built with `genes=cohort_genes`;
   the scorer is built via `from_full_graph`; and the non-degeneracy guard is widened
   from `X_train` to **all three splits** so a future regression is loud, not silent.
4. **Tests** — `tests/unit/test_hetero_inductive_fix.py` (+5, torch-free): the union
   node set makes val/test nodes with zero features and train-only focal; a scorer
   over the union resolves gene-disjoint val/test (not 0.5) while genuine unknowns
   still default; `from_full_graph` is a classmethod; `genes=` is exposed and
   backward-compatible (`genes=None` preserves the legacy node set).

### Leakage analysis
No label or feature leak. Val/test genes contribute **only graph structure** — STRING
and Reactome edges, which are external prior knowledge, not derived from labels. Their
node features are zero and no loss is computed on them; message passing during training
gives them structural connectivity only. This is the identical posture to the regular
GNN's full-graph scorer.

## Audit change (the detection gap — LANDED)
`scripts/audit_smoke_feature_population.py` was rewritten to evaluate **each split
separately** (`_load_splits` instead of the concatenating `_load_matrix`): a
FAIL-severity feature dead/absent in **any** split now FAILs the audit, with a
per-split `train=ok val=DEAD test=DEAD` breakdown and a diagnostic note. This
directly catches the focal-only failure mode going forward. `reactome_pathway_count`
was downgraded to WARN in the same change (it is dead-by-missing-data — it needs its
own parquet from `scripts/build_reactome_parquet.py`; `--kg-edges reactome:...gmt`
only feeds the hetero graph, not the feature). Regression: `tests/unit/test_run17_audit_persplit.py`
(+5), including the exact `hetero alive in train / dead in val+test -> FAIL` case.

## Validation
- **Torch-free (diagnosis sandbox):** both edited files `py_compile`; the union
  node-set assembly (`assemble_node_features_and_focal`) yields val/test nodes with
  zero features + train-only focal; `HeteroGNNScorer` over a union map resolves
  val/test; existing hetero suite (`test_hetero_gnn*`, `test_preflight_hetero_gate`)
  unchanged; patcher is EOL-agnostic and idempotent on LF and CRLF trees.
- **In-env (pending, the real gate):** run the full-flag smoke with `--hetero-gnn`;
  the log must show `[HETERO-GNN] node set spans all splits: union=...` and
  `val`/`test injected ... nunique > 1` (not `nunique=1, std=0.0000`); then
  `python scripts\audit_smoke_feature_population.py <outdir>\splits --run17` must read
  `hetero_gnn_score ... train=ok val=ok test=ok`; full `pytest -q` green.

## Relationship to INCIDENT_2026-06-04 (gnn-score-injection-degenerate)
Same class of bug, parallel modality. The 2026-06-04 fix added
`GNNScorer.from_full_graph` (all ~16,201 STRING nodes) for the homogeneous score and
documented that any gene scorer must cover every node. The heterogeneous scorer,
introduced afterward, reused the focal-only `from_trained` path and reintroduced the
flaw. **Lesson: when a structural fix lands on one of two parallel modalities, port
it to BOTH, and make the verifier/audit check the property on every split, not just
train.**

## References
- `scripts/run_phase2_eval.py` — hetero block: cohort-gene union, `genes=` graph
  build, `HeteroGNNScorer.from_full_graph`, per-split non-degeneracy guard.
- `src/genomic_variant_classifier/models/hetero_gnn_scorer.py` —
  `build_hetero_focal_graph(genes=)`, `HeteroGNNScorer.from_full_graph`.
- `src/genomic_variant_classifier/models/hetero_gnn.py` — `build_hetero_gene_graph`
  (`node_genes = list(genes)`; edge-only sanitization).
- `scripts/audit_smoke_feature_population.py` — per-split detection (`_load_splits`).
- `tests/unit/test_hetero_inductive_fix.py`, `tests/unit/test_run17_audit_persplit.py`.
- `docs/incidents/INCIDENT_2026-06-04_gnn-score-injection-degenerate.md` — homogeneous sibling.
- `docs/incidents/INCIDENT_2026-04-30_gnn-gene-symbol-keyerror.md` — Patch 6b (GNN training input).
