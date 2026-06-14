# Session 2026-06-13 (v2) -- hetero_gnn_score schema + scorer; LiteratureScout broadening

Continuation of 2026-06-13 (after the leakage/agents/KG session, 32bb9ef). Three clean
commits on 32bb9ef; suite 992 -> 1000 passed / 6 skipped / 41 warnings (all pre-existing).

  547e2dc  feat(models): hetero_gnn_score 82nd feature + hetero-GNN trainer/scorer
  a42e723  feat(agents): LiteratureScout provenance -- journal, authors, publication_date
  a9c0326  feat(agents): LiteratureScout Zenodo source + scope + journal allow-list boost

## 1. Heterogeneous-KG GNN: scorer + 82nd feature (547e2dc)

`models/hetero_gnn_scorer.py` completes the hetero track (engine 54158f7 + connectors
8c19f9b landed last session). It mirrors `gnn.py`'s GNNTrainer/GNNScorer exactly: build
one shared multi-relation gene graph (STRING `interacts_with` + KG relations from
`kg_edges`), train `HeteroVariantGNN` with a focal-node loss, score every gene node, and
return a `gene_symbol -> score` map with the same 0.5-default contract as `GNNScorer`. The
bug-prone data assembly (gene-mean node features + focal/label alignment) is split into a
torch-free helper unit-tested without PyG; the train/score path is PyG-gated.

**Schema decision -- Option A (separate 82nd feature, NOT a graph replacement).** Settled by
the project's own dual goal: `gnn_score` is the homogeneous STRING-GNN signal; replacing its
graph with the multi-relation KG would destroy the homogeneous-vs-heterogeneous comparison,
a first-class deliverable. So `hetero_gnn_score` is a parallel feature filled the same way
`gnn_score` is (default 0.5 until activated at eval time).

The schema bump was the delicate part (a prior 81->51 trim broke 40 contract tests). Mapped
the full contract surface first: `EXPECTED_TABULAR_FEATURE_COUNT` (two `len==EXPECTED`
guards), the three `list(feats.columns)==TABULAR_FEATURES` order guards, and the
reactome-last guard. `hetero_gnn_score` inserted immediately after `gnn_score` in
TABULAR_FEATURES AND in both builders, same position -> order-match holds, reactome stays
last. EXPECTED 81 -> 82. Full suite 992 (now validating 82 cols); focused contract re-check
107 passed.

## 2. LiteratureScout broadening (a42e723 + a9c0326)

Two commits. **Provenance (a42e723):** `authors` / `publication_date` / `journal` captured
from all sources and carried into the SharedState candidate record + the emitted
FEATURE_CANDIDATE_ADDED event. New testable PubMed efetch helpers: `_parse_pubmed_article`
(journal Title -> ISOAbbreviation fallback; authors incl. CollectiveName; multi-AbstractText
join) and `_parse_pubmed_pub_date` (ArticleDate -> PubDate -> MedlineDate). **Source + scope
(a9c0326):** new `_fetch_zenodo` (Zenodo /api/records; try/except -> logged warning, never a
crash) + `_parse_zenodo_hit`; PubMed queries 11 -> 19 and keywords 32 -> 46 into the
architecture/methodology gaps (GNN, knowledge graph, self-supervised, contrastive, foundation
model, calibration/uncertainty, AlphaFold-structure, splicing); a 20-venue journal allow-list
+ 0.15 relevance boost. `_strip_html` strips tags BEFORE decoding entities so entity-encoded
`&lt;`/`&gt;` survive (a real ordering bug caught on a fixture).

## 3. Deferred -- Run-17 prep (tracked, both need the real 82-col matrix)

- **schema_baseline.json regen 81 -> 82** from the real matrix
  (`build_schema_baseline.py --allow-schema-change`). NOT edited in place -- that would attach
  82 columns to an 81-column `captured_from` source (provenance lie). No unit test depends on it.
- **run_phase2_eval live overwrite** -- a HeteroGNNScorer built from STRING + KG files fills
  `hetero_gnn_score` with real values (parallel to the `gnn_score` overwrite, opt-in flag).
  Until then `hetero_gnn_score` is a 0.5 constant, exactly mirroring `gnn_score`'s
  default-until-activated behaviour (so shipping the schema bump first is harmless).

## 4. Carried (pre-existing, unchanged)

- `test_ablate_gnn` skips locally on the torch_scatter/torch_sparse 0xc0000139 DLL load
  failure (GNN coverage absent on the Windows box) -- confirm runnable before Run 17 activates
  graph features.
- pandas `.fillna` downcasting FutureWarning in `variant_ensemble.py` wants an explicit cast.

## Key decisions / learnings

- Adding a feature is far safer than removing one: the connector-flow guards check NAMED
  features, so a pure addition leaves them green; only the count + order + reactome-last guards
  have teeth, all satisfied by a same-position insert.
- The schema baseline is a SNAPSHOT of a real matrix, not a mirror of TABULAR_FEATURES; it
  correctly lags a schema change until the next capture. Editing it in place to chase the code
  would corrupt `captured_from`.
- Defensive external fetchers: every new source parse uses `.get()` + try/except so an API
  shape or network surprise degrades to a logged warning, never a crash or a silent bad row.
