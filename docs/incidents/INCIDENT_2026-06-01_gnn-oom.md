# INCIDENT 2026-06-01 — GNN 64 GB OOM

## Summary

The GNN component attempted a ~64 GB allocation and OOM'd when training on the real STRING graph.

## Root Cause

`build_pyg_dataset` replicated the entire STRING graph (edge_index + 3-channel edge_attr) into every one of the ~4,873 per-variant `Data` objects. `DataLoader(batch_size=32)` then concatenated 32 full graphs, and `GATConv(edge_dim=3).lin_edge` materialized a tensor on the order of (32 × ~474K edges, heads × hidden) ≈ 64 GB. Node features were already gene-level means with the focal indicator on the per-gene node, so the model was effectively gene-level already.

## Fix (Option B)

Rewrote `gnn.py` to a single shared graph with batched focal readout, trained full-batch transductively (one forward over the shared graph per epoch). New `SharedFocalGraph` dataclass holds one graph plus per-sample focal index / label / variant id; `in_channels = n_feats` (the redundant focal indicator dropped); `GNNScorer.from_trainer` uses parallel arrays. A glue patch passes `string_kwargs` so the in-pipeline graph build uses local STRING files, not the network.

## Validation

Validated against real torch_geometric on a synthetic homophilous graph (val AUC 0.95) and on the real graph (16,201 nodes / 236,930 edges): no OOM, finite losses, `gnn_score` std 0.0302, range [0.150, 0.551], 2,299 / 2,446 genes scored. 5 GNN tests green; import lists all 7 public symbols. Full suite 651 passed.

## Follow-on Finding (not a defect)

On CPU the full-graph forward is ~2,293.8s/epoch (~38 min); this is inherent CPU slowness for edge-attention scatter/gather, not a bug. The GNN is a GPU operation; GPU epoch time must be measured before committing to 100 epochs. A pure-efficiency follow-up (move graph tensors to device once per fit/predict rather than per epoch) is deferred to the next `gnn.py` touch.

## Lessons

- Never replicate a shared static graph per sample; use one shared graph + focal readout.
- A bare `except Exception` previously masked GNN crashes as soft warnings — avoid in training pipelines.

## Status

FIXED. Applied to working tree; commit pending.
