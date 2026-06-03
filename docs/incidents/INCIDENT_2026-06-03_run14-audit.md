# INCIDENT 2026-06-03 — Run 14 audit: GNN skipped, no gene-disjoint holdout, leakage-inflated AUROC

## Summary

Run 14's headline test AUROC 0.9975 / val 0.9974 is not a trustworthy generalization estimate. Three independent reasons, all evidenced in `outputs/run14/run14_master.log`.

## Findings

1. **GNN entirely skipped.** Lines 466–467: `args.string_db=None` → `[GNN-TRACE] gate-skipped ... ENTIRE GNN BLOCK skipped`. Run 14 is an ensemble-without-graph result. The run did not error; it logged a WARNING and proceeded — a silent degradation.
2. **No gene-disjoint / unseen-gene holdout.** A grep for gene-disjoint / unseen / holdout / leak returned only `Dev (test) Holdout (val)` and the target-met line. The "Holdout (val)" is an ordinary random split, not gene-disjoint.
3. **Label-derived proxy dominant.** `n_pathogenic_in_gene` importance 464.35, nonzero on all 1,700,687 rows (`source=ClinVar(derived)`). With train/test sharing genes, the model can score by gene identity rather than variant biology — gene-level leakage. Every base model clustered at 0.9955–0.9984 is the signature.

The log's line 538 ("Next: REST API + Docker deployment") is therefore premature: it acts on a number that does not yet mean what it appears to.

## Corroboration

The same session's rich-source verification showed that wiring real sources cut `n_pathogenic_in_gene` importance from ~1213.5 to ~272.3 and promoted CADD/LOEUF/GERP, confirming the proxy is doing the work in leakage-prone configurations.

## Resolution / Required Action

The honest number requires the rich run with `--string-db auto` (GNN on), all source paths, and `--unseen-gene-holdout`. This is now enforced by `scripts/preflight_gate.py`, which hard-fails a launch missing any of those — converting the Run 14 silent degradation into a pre-launch abort.

## Status

DOCUMENTED. Run 14 results retained as a leakage-baseline reference only; not for citation as a generalization metric.
