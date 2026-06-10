#!/usr/bin/env python3
"""patch_readme_phase1.py -- refresh README.md from Run-8-era to Run 15 + Phase 1.

Updates ONLY verifiable current facts (Run 15 metrics, 80-feature contract, ESM-2
LLR). Leaves historical run records and unverifiable claims (agent count, DB count,
histopathology branch) untouched -- those are flagged for a separate pass.

Per-edit resilient: each edit is independent; anchors that do not match exactly
once are reported as MISS (loud, non-zero exit), never silently skipped. Line-ending
aware (preserves the file's existing LF/CRLF). Backup-first, idempotent.
Author: Monzia Moodie.
"""
from __future__ import annotations

import datetime as _dt
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RM = REPO / "README.md"

# (old, new, marker, label) -- authored with \n; normalized to the file's EOL at runtime
EDITS = [
    ("[![Holdout AUROC](https://img.shields.io/badge/Holdout%20AUROC-0.9847-brightgreen.svg)]()",
     "[![Holdout AUROC](https://img.shields.io/badge/Holdout%20AUROC-0.9984-brightgreen.svg)]()",
     "AUROC-0.9984", "badge: AUROC 0.9847 -> 0.9984"),
    ("[![Variants](https://img.shields.io/badge/Training%20variants-1.70M-blue.svg)]()",
     "[![Variants](https://img.shields.io/badge/Training%20variants-1.49M-blue.svg)]()",
     "variants-1.49M", "badge: variants 1.70M -> 1.49M"),
    ("[![Features](https://img.shields.io/badge/Tabular%20features-78-blue.svg)]()",
     "[![Features](https://img.shields.io/badge/Tabular%20features-80-blue.svg)]()",
     "Tabular%20features-80", "badge: features 78 -> 80"),
    ("[![Tests](https://img.shields.io/badge/Tests-501%20passing-success.svg)]()",
     "[![Tests](https://img.shields.io/badge/Tests-862%20passing-success.svg)]()",
     "Tests-862", "badge: tests 501 -> 862"),
    ("**Holdout AUROC: 0.9847** on 154,404 gene-stratified expert-reviewed ClinVar variants.",
     "**Run 15 (sealed 2026-06-09, commit 032a2ab): Test AUROC 0.9984 / Val 0.9983 / "
     "unseen-gene-holdout 0.9988** on gene-stratified expert-reviewed ClinVar variants (Test n=304,711).",
     "Test AUROC 0.9984 / Val 0.9983", "headline L23: Run-8 -> Run 15"),
    ("Run-8 holdout AUROC **0.9863** / test AUROC **0.9833** on the full 1.70 M-variant",
     "The model trains on a ~1.49 M-variant cohort drawn from ~2.49 M ClinVar missense variants, now a",
     "drawn from ~2.49 M ClinVar missense variants, now a", "headline L24"),
    ("78-feature matrix (Vast.ai RTX 4090, 4,270 s wall-clock, 1.8 GB artifacts).",
     "80-feature matrix (Run 15: Vast.ai RTX 4090, ~11.5 h, ~$6). Earlier Run-8: holdout 0.9863 / "
     "test 0.9833 on 78 features.",
     "80-feature matrix (Run 15:", "headline L25"),
    ("graph. Input features span **78 dimensions** drawn from eighteen biological databases.",
     "graph. Input features span **80 dimensions** drawn from eighteen biological databases.",
     "**80 dimensions**", "architecture: 78 -> 80 dimensions"),
    ("(one-hot encoded) combined with ESM-2 protein language model embeddings (HuggingFace\n"
     "`transformers` backend, SQLite cache, scalar L2-delta embedding) capturing evolutionary\n"
     "and structural variant context. ESM-2 silent-zero failure modes are explicitly\n"
     "detected by `tests/unit/test_esm2_activation.py` per `INCIDENT_2026-04-17`.",
     "(one-hot encoded) combined with ESM-2 protein-language-model features (HuggingFace\n"
     "`transformers` backend). Two signals are derived: the scalar L2 embedding-delta\n"
     "(`esm2_delta_norm`, secondary) and -- as of Phase 1 -- the primary log-likelihood-ratio\n"
     "`esm2_llr` (`logit[mut] - logit[wt]` from the ESM-2 650M masked-LM head; WT-marginal by\n"
     "default, masked-marginal opt-in). `esm2_llr` is SIGNED (negative = more damaging) and\n"
     "enters the ensemble as a CONTINUOUS feature -- its sign is not a class label (even benign\n"
     "variants score negative), so the model learns the threshold. ESM-2 silent-zero failure\n"
     "modes are detected by `tests/unit/test_esm2_activation.py` per `INCIDENT_2026-04-17`.",
     "the primary log-likelihood-ratio", "sequence branch: add esm2_llr LLR"),
    ("| Foundation model | ESM-2 scalar L2 delta (HF transformers, SQLite cache) | Active when HGVSp populated |",
     "| Foundation model | ESM-2 650M: `esm2_llr` LLR (primary) + scalar L2 delta (secondary), HF transformers | "
     "Phase 1 done; full-cohort scoring after Run-16 coord-sync |",
     "ESM-2 650M: `esm2_llr` LLR (primary)", "model registry: ESM-2 row"),
    ("## Feature set (78 features)",
     "## Feature set (80 features)",
     "## Feature set (80 features)", "feature-set header: 78 -> 80"),
    ("| ESM-2 (pending HGVSp parser, Run 10) | 1 | esm2_delta_norm |",
     "| ESM-2 (650M) | 2 | esm2_delta_norm (secondary), esm2_llr (primary, signed LLR) |",
     "| ESM-2 (650M) | 2 |", "feature table: ESM-2 row 1 -> 2"),
    ("| FinnGen R12 AF | 3 | finngen_af_fin, finngen_af_nfsee, finngen_enrichment |",
     "| FinnGen R12 AF | 3 | finngen_af_fin, finngen_af_nfsee, finngen_enrichment |\n"
     "| Reactome | 1 | reactome_pathway_count |",
     "| Reactome | 1 | reactome_pathway_count |", "feature table: add Reactome row"),
    ("GET  /info            Model metadata, 78 features, drift status",
     "GET  /info            Model metadata, 80 features, drift status",
     "Model metadata, 80 features", "API /info: 78 -> 80"),
    ("  features/      - engineer_features (78-column pipeline, runtime sync assertion)",
     "  features/      - engineer_features (80-column pipeline, runtime sync assertion)",
     "engineer_features (80-column", "layout: 78 -> 80 column"),
    ("# Train (full ensemble, 78 features)",
     "# Train (full ensemble, 80 features)",
     "full ensemble, 80 features", "train cmd: 78 -> 80"),
    ("| Run 10 | scheduled | Vast.ai RTX 4090 | -- | Phase-1.7 launch script + dual-layer preflight; targets locked test recovery |",
     "| Run 10 | scheduled | Vast.ai RTX 4090 | -- | Phase-1.7 launch script + dual-layer preflight; targets locked test recovery |\n"
     "| Run 14 | 2026-06-03 | Vast.ai RTX 4090 | 0.9975 | commit eb11029; SNV/indel leakage traced entirely to null ref/alt records (no real-allele leakage) |\n"
     "| **Run 15** | **2026-06-09** | **Vast.ai RTX 4090** | **0.9984** (test) | **commit 032a2ab; Val 0.9983 / unseen-gene-holdout 0.9988; 79 features. ESM-2 650M LLR + 80-feature contract added 2026-06-10 (Phase 1), realized at next regen** |",
     "| **Run 15** | **2026-06-09**", "run table: add Run 14 + Run 15"),
    ("- **Phase 4 -- Algorithm expansion and benchmarking.** Wire ESM-2 fully via\n"
     "  the in-flight HGVSp parser (Run 10), run KAN through the benchmark harness\n"
     "  against MLP, integrate Deep Ensemble uncertainty into VUS flagging, and\n"
     "  fuse GNN gene embeddings with `TABULAR_FEATURES` before stacking. Tracked\n"
     "  in `docs/ROADMAP.md`.",
     "- **Phase 4 -- Algorithm expansion and benchmarking.** ESM-2 upgraded to the 650M\n"
     "  masked-LM with a log-likelihood-ratio feature (`esm2_llr`, Phase 1 -- done); next are\n"
     "  ESM C 600M and a full-cohort regen after the Run-16 coordinate-index sync. Run KAN\n"
     "  through the benchmark harness against MLP, integrate Deep Ensemble uncertainty into\n"
     "  VUS flagging, and fuse GNN gene embeddings with `TABULAR_FEATURES` before stacking.\n"
     "  Tracked in `docs/ROADMAP.md`.",
     "ESM-2 upgraded to the 650M", "Phase 4 roadmap: ESM-2 LLR status"),
]


def main() -> int:
    if not RM.exists():
        print(f"ABORT: missing {RM}")
        return 2
    raw = RM.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    norm = (lambda s: s.replace("\n", nl)) if nl != "\n" else (lambda s: s)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(RM, f"{RM}.bak_{ts}")

    applied = skipped = missed = 0
    misses = []
    for old, new, marker, label in EDITS:
        if norm(marker) in raw:
            print(f"  skip (already applied): {label}")
            skipped += 1
            continue
        o = norm(old)
        n = raw.count(o)
        if n != 1:
            print(f"  MISS ({n}x, expected 1): {label}")
            misses.append(label)
            missed += 1
            continue
        raw = raw.replace(o, norm(new), 1)
        print(f"  ok: {label}")
        applied += 1

    RM.write_bytes(raw.encode("utf-8"))
    print(f"\napplied={applied} skipped={skipped} missed={missed} "
          f"(backup -> README.md.bak_{ts})")
    if misses:
        print("MISSED anchors (left unchanged -- verify text and re-run):")
        for m in misses:
            print(f"  - {m}")
        print("Regenerate nothing; README.docx is not maintained. Re-run after fixing anchors.")
        return 1
    print("DONE. All edits applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
