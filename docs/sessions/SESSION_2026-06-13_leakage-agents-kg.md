# Session 2026-06-13 -- Leakage L1/L2, AdaptationAgent, af_1kg, hetero-GNN engine, KG connectors

Owner: Monzia Moodie. Base: 5d69182. Head after session: 8c19f9b (origin/main).
Suite: 989 passed / 6 skipped / 41 warnings (all pre-existing).

## Goal
Close the n_pathogenic_in_gene leakage the Run-15 UGH 0.9988 result left open, advance the
agent layer and the heterogeneous-KG modeling track, and resurrect the dead af_1kg_* columns --
all under audit-first discipline with clean, separable commits.

## What shipped (6 commits)

1. **fix(leakage): train-only n_pathogenic_in_gene post-split (689787f).**
   enrich_gene_counts computes the count corpus-wide before the gene-disjoint GroupShuffleSplit,
   so a held-out gene's count derived from its own held-out labels -- a corpus-scope leak. Recompute
   train-only in _gene_aware_split, remap to every split (unseen genes -> 0), recompute
   gene_has_known_disease in lockstep. Standing probe scripts/audit_npathogenic_leakage.py:
   lone-feature test AUROC 0.7181 (corpus) -> ~0.50 (train-only). +4 tests.

2. **fix(leakage): Level-2 leak-free stacking OOF (6b38985).**
   The meta-learner's inner OOF used StratifiedKFold over a full-train count -> fold-held-out rows
   saw fold-train peers' counts. Inner CV switched to gene-disjoint GroupKFold with per-fold
   train-only recompute of n_pathogenic_in_gene + gene_has_known_disease. Standalone measurement:
   leaky OOF 0.7755 vs leak-free 0.6633 (+0.1122 inflation removed). gene_symbol threaded from
   meta_train.parquet (row-aligned with X_train); gene_symbol=None = legacy path byte-for-byte; the
   full-model fit (used for inference) is unchanged. +4 tests.

3. **feat(agents): AdaptationAgent (636c6df).**
   Consumes the version_monitor SharedState section (deps_major_bumps, python_alert, pyg_abi_alert);
   in evaluate mode builds a throwaway isolated venv, installs the candidate, runs the suite, parses
   the pytest summary, records a verdict in an append-only JSONL ledger. Plan-only by default; never
   mutates the live env. Wired version_monitor + adaptation pipelines into the orchestrator --
   VersionMonitorAgent was registered but in NO pipeline, so it had never executed. +10 tests.

4. **feat(data): resurrect af_1kg_* (a0ce407).**
   The five super-population AF columns were silently all-zero -- no connector wrote them
   (fill_missing_af fills only the global allele_freq). fill_population_af maps per-superpopulation
   AF columns from a 1000G parquet (key chrom:pos:ref:alt, path+mtime cache, [0,1] clip, dup de-dup,
   partial-population warnings); wired into _join_gnomad with an all-zero guard. build_1kg_parquet.py
   emits the per-population parquet from a 1000G VCF. +5 messy-data tests. NOTE: corrects the prior
   claim that "--kg activates af_1kg_*" -- --kg previously only filled allele_freq; the connector is
   what activates the per-population columns.

5. **feat(models): heterogeneous KG GNN engine (54158f7).**
   models/hetero_gnn.py: a torch-free builder (gene->index, per-relation edge sanitisation: drop
   unknown genes, dedup, undirected, self-loop policy, feature-shape guard) + a torch_geometric
   HeteroConv({(gene, rel, gene): SAGEConv}) model over one node type and many relations. Forward
   hardened with an explicit per-layer root transform + empty-relation filtering, so isolated nodes,
   empty relations, and edgeless graphs all stay finite. Additive -- does not touch the homogeneous
   STRING GNN (gnn.py) or the feature schema. +5 tests (3 builder run anywhere; 2 real HeteroConv
   forwards run under .venv312 where torch_geometric is present).

6. **feat(data): KG gene-gene edge connectors (8c19f9b).**
   data/kg_edges.py: every KG we integrate has the same shape (genes belong to sets; two genes share
   an edge iff they co-occur in a set), so one co_membership_edges primitive + thin adapters
   (reactome_edges, kegg_edges, go_edges, clingen_edges, omim_edges) feed the hetero-GNN builder.
   The primitive guards set-size explosion (skip > max_set_size=200), restricts to cohort genes
   before pairing, de-dups to canonical a<b order, drops self-pairs. GMT parsing survives
   malformed/duplicate/CRLF lines; CSV parsing auto-locates the header through a preamble (ClinGen)
   and errors clearly on a missing column. KG_SOURCES provenance registry. +5 tests (incl. handoff).

## Metrics observed this session
- L1 leakage probe (lone-feature, gene-disjoint test): corpus-wide 0.7181 -> train-only ~0.50.
- L2 inner-OOF standalone: leaky 0.7755 vs leak-free 0.6633 (delta 0.1122).
- co_membership explosion guard: a 50-gene set at max_set_size=10 suppressed ~1225 edges (logged).

## Decisions resolved
- **n_pathogenic_in_gene scope (OPEN -> RESOLVED):** corpus-wide was leaking; now train-only at both
  Level 1 (data prep) and Level 2 (stacking OOF). The Run-17 Gate-A leakage <DECISION> is closed.
- **af_1kg_* (dormant / "wire or retire" -> WIRED):** fill_population_af + build_1kg_parquet.py;
  activation at Run 17 = --kg <1000G per-superpopulation AF parquet>.

## Known follow-ups (queued, not blockers)
- Live hetero_gnn_score wiring: a HeteroGNNScorer mirroring GNNScorer (train HeteroVariantGNN on
  STRING + KG relations over cohort genes; read out per-gene score; map by gene_symbol), plus the
  schema decision -- hetero_gnn_score as a guarded 82nd feature (both builders in lockstep,
  EXPECTED_TABULAR_FEATURE_COUNT 81->82) vs enriching the existing gnn_score graph in place.
- LiteratureScout broadening (journal allow-list incl. Zenodo own-fetcher; methodology/architecture
  scope; author + publication-date capture).
- Run 17 launch: Gate A unblocked by L1+L2; confirm the 1000G per-superpopulation parquet for af_1kg.
- KG source acquisition: Reactome GMT (CC0) + ClinGen CSV (public) directly; GO/KEGG via MSigDB
  (registration); OMIM license-restricted.

## Process note
The commit regrouping's git add -p split was answered y/y instead of n/y, landing the af_1kg
_join_gnomad wiring in the L1 commit. A first corrective rebase silently no-op'd and was
force-pushed content-identical; a second, hardened rebase relocated the block correctly (per-commit
counts 0 then 1; final tree byte-identical). Full write-up: docs/incidents/INCIDENT_2026-06-13_rebase-noop.md.
