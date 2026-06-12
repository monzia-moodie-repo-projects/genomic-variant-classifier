# Run 17 Scope -- 1000 Genomes + STRING-DB GNN (COMMITTED, tracked)

These are commitments with hard acceptance criteria, not deferrals. Both must be fully
validated, integrated, and cleared before Run 17 trains. "Done right" = thorough test +
check + validate + verify; thoroughness is not rushing. No feature ships above the
integrated whole. Author: Monzia Moodie.

---

## Track A -- 1000 Genomes AF (`--kg-path`)

Goal: recover null allele_freq via the 1000G global-AF fallback, and resolve the five
af_1kg_* per-population stubs (wire a real source OR formally retire them).

- A1  Acquire 1000G Phase 3 VCFs (ftp.1000genomes.ebi.ac.uk .../release/20130502/).
- A2  Write scripts/build_1kg_parquet.py -> parquet {variant_id "chrom:pos:ref:alt"
      (no chr prefix), allele_freq float}. Unit test on a small VCF slice.
- A3  Build the parquet; `python scripts/locate_1kg.py` must report USABLE
      (schema PRESENT + KEY-FORMAT MATCH YES).
- A4  Add `--kg-path` to scripts/train.py, threaded into AnnotationConfig.kg_path
      (mirror the --lovd-path wiring exactly). Add a wiring unit test.
- A5  Re-smoke with --kg-path. GATE: log shows "1000G fallback: filled N / <nulls>"
      with N > 0 AND the post-smoke null-AF count drops vs Run-16b.
- A6  DECISION on af_1kg_*: either wire a per-population 1000G source (build extends A2)
      OR formally retire the 5 stub features from the schema (and document). No silent
      permanent stubs.

Acceptance (A): locate_1kg USABLE; --kg-path wiring test green; re-smoke shows AF-fill
> 0 and null-AF drop; af_1kg_* either POPULATED or formally retired; schema drift green.

---

## Track B -- STRING-DB GNN (`gnn_score` live, LEAKAGE-FREE)

Goal: make gnn_score a real ensemble feature, produced without leaking labels.

- B1  Audit STRING v12 data: are 9606.protein.links.detailed.v12.0 +
      9606.protein.info.v12.0 staged locally, or must the box pull stringdb-downloads.org?
      Confirm PyG/torch_geometric on the box.
- B2  Read train_gnn_pipeline + GNNTrainer + GNNScorer + gnn_optim in full; map the
      build-graph -> train-GAT -> score interface and its inputs/outputs.
- B3  DESIGN the leakage gate (the crux): cross-fit to the gene-disjoint splits -- train
      the GNN only on train-fold genes; score test-fold genes by graph propagation, never
      on a GNN that saw their labels. This mirrors the OOF discipline the tabular stack
      already uses.
- B4  Orchestration script: build STRING graph -> (per fold) train GAT -> GNNScorer.
      score_dataframe -> write a leakage-free gnn_score column back into the cohort.
- B5  GNN VALIDATION SMOKE: GAT trains to a sane AUROC; gnn_score non-degenerate with
      real gene coverage; a held-out-gene check confirms NO leakage (test-gene scores do
      not encode test labels).
- B6  Integrate: cohort gains live gnn_score; tabular re-smoke confirms gnn_score
      POPULATED. Compare ensemble metrics WITH vs WITHOUT gnn_score under identical
      gene-disjoint splits -- the gain must be real, not leakage.

Acceptance (B): GNN smoke green (sane AUROC, non-degenerate gnn_score, gene coverage,
held-out-gene no-leak check PASS); tabular re-smoke shows gnn_score POPULATED; WITH-vs-
WITHOUT ablation documents the true contribution; schema drift green; metrics-glossary
entry for gnn_score (formula/range/why/leakage-control).

---

## Cross-cutting (both tracks, before Run 17 trains)

- Every new/activated feature passes the feature-population audit at full scale.
- Schema baseline re-sealed if the column set/dtypes change; drift-check green.
- Ablation harness quantifies each new source's marginal contribution (no_kg / no_gnn
  masks) -> metrics glossary.
- Gene-disjoint integrity verified end-to-end (no cross-fold or cross-stage leakage).
- All work documented per standing discipline (CHANGELOG / ROADMAP / SESSION / incidents).
