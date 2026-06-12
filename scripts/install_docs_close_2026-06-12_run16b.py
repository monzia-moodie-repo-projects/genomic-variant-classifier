#!/usr/bin/env python3
"""install_docs_close_2026-06-12_run16b.py -- close the 2026-06-12 Run-16b session.

Creates docs/sessions/SESSION_2026-06-12_run16b.md and APPENDS dated deltas to
docs/CHANGELOG.md and docs/ROADMAP.md. Idempotent (marker-guarded appends; session
file skipped if present), newline-preserving (CRLF repo), ASCII-only, backup-free
(append-only). Run from the repo root. Author: Monzia Moodie.

NEXT after running: python scripts/make_roadmap_docx.py ; review git diff ; commit + push.
"""
from __future__ import annotations

import sys
from pathlib import Path

SESSION_PATH = Path("docs/sessions/SESSION_2026-06-12_run16b.md")
CHANGELOG = Path("docs/CHANGELOG.md")
ROADMAP = Path("docs/ROADMAP.md")

CHANGELOG_MARKER = "2026-06-12 -- Run-16b smoke gate"
ROADMAP_MARKER = "2026-06-12 -- Run 16 launch-ready"

SESSION_BODY = """# SESSION 2026-06-12 -- Run-16b smoke gate, schema re-seal, source finalization

HEAD entering: 4da4219 (schema baseline re-seal 78->81).

## Arc 1 -- Run-16b all-models smoke CLEARED
Added gnomAD-AF, dbNSFP, LOVD to the proven flag set; full --fast smoke
(models/smoke_run16b, 962s, no OOM, no crash). 13 base models, 81 features,
ENSEMBLE_STACKER test AUROC 0.9994 (up from 0.9934 without the new sources). Feature
matrix population verified in splits/X_*.parquet: af_log10, cadd_phred, sift_score,
revel_score, n_tools_pathogenic POPULATED; LOVD all-default (smoke-size; verify >0 at
full scale).

## Arc 2 -- Connector audits corrected three wrong source picks
- --lovd-path -> data/external/lovd/lovd_all_variants.parquet (train.py-canonical), NOT
  the greedy-glob am_lovd_genes.parquet (an AlphaMissense artifact).
- --uniprot OMITTED: _join_uniprot reads source_id + pathogenicity; the only on-disk
  uniprot parquet has gene_symbol/uniprot_id/sequence -> KeyError / silent-dead.
- dbNSFP: connector hard-codes its cache name to dbnsfp_clinvar_index.parquet (2.69M);
  the 895 MB dbnsfp_full_index.parquet is never read. Docstring drift (said full_index)
  fixed by patch_dbnsfp_docstring.py. OOM avoided by using the ClinVar index directly.

## Arc 3 -- Schema baseline re-sealed 78 -> 81 (run16b-smoke)
Pre-seal probe confirmed all 3 splits share an identical 81-col float64 schema; new cols
esm2_llr (live), maxentscan_delta + reactome_pathway_count (sealed dormant). Sealed from
the smoke X_train; green vs all 3 splits. Authoritative gate remains the full-regen
schema drift-check on Vast.ai.

## Arc 4 -- Feature-population gate (built + hardened)
audit_smoke_feature_population.py: v1 mis-targeted the pre-scoring checkpoint
(clinvar_enriched.parquet, 1931 rows) and checked raw connector names -> false FAIL.
Corrected to read splits/X_*.parquet + engineered names. Noted: the per-source
default-check is unsound on standard-scaled splits; the all-constant scan is the reliable
detector (36/81 columns constant: known stubs + gnn_score + af_1kg_* + uniprot + lovd).

## Arc 5 -- 1KGP + GNN investigated -> committed Run-17 scope (NOT deferred)
- No --kg-path flag in train.py; ThousandGenomesConnector fills only combined allele_freq
  (af_1kg_* never activate via it); no kg parquet staged; build_1kg_parquet.py absent.
- No --string-db flag; GNN (gnn.py: StringDBGraph / VariantGAT / GNNTrainer / GNNScorer)
  is complete but unwired; gnn_score is a df.get placeholder. Live gnn_score requires
  gene-disjoint cross-fitting to avoid label leakage.
- Both committed to docs/roadmap/RUN17_SCOPE.md with hard acceptance criteria.

## Arc 6 -- Launch contract v2 + tree hygiene
docs/launch/LAUNCH_CONTRACT_run16.md v2 (validated flag set, ship/do-not-ship manifest,
on-box blocking gates, dormant-by-design watch-items). Quarantined stale
clinvar_grch38_clean_seq (1).parquet (18-col, no ReviewStatus). dbNSFP docstring fixed;
redundant promoted dbnsfp_full_index.parquet removed.

## Tools delivered this session
audit_run16_data_sources.py, prep_dbnsfp_cache.py, audit_smoke_feature_population.py,
verify_schema_seal_inputs.py, locate_1kg.py, patch_dbnsfp_docstring.py.

## Watch-items carried to Run 16 (full scale)
- cnn_1d test 0.4782 -- at/below random across 2 smokes; full scale discriminates
  architecture-defect vs data-starvation.
- kan Brier 0.2223 (poor calibration; ranks fine for the stacker).
- CIRCULARITY: cadd_phred is #1 importance; CADD/REVEL/SIFT/AlphaMissense are
  ClinVar-trained. Run a no_meta_predictors ablation; document in the metrics glossary.
- LOVD: expect lovd_variant_class > 0 at full scale (else a join-key bug, not coverage).
- real_data_prep.py:501 FutureWarning (gnomAD fillna downcast) -- tech debt.
- Dormant-by-design (NOT bugs): gnn_score, af_1kg_*, uniprot features. Activate in Run 17.
"""

CHANGELOG_DELTA = """## 2026-06-12 -- Run-16b smoke gate + schema re-seal + source finalization

Fixed:
- dbNSFP cache-name docstring drift (dbnsfp.py): said dbnsfp_full_index.parquet; code
  hard-codes dbnsfp_clinvar_index.parquet. Corrected (patch_dbnsfp_docstring.py).
- Quarantined stale clinvar_grch38_clean_seq (1).parquet (18-col, no ReviewStatus).

Added (validated via models/smoke_run16b: 962s, ENSEMBLE_STACKER test AUROC 0.9994):
- Run-16 production flag set: --gnomad, --dbnsfp-path (ClinVar index; OOM-safe),
  --lovd-path (lovd_all_variants.parquet). --uniprot omitted (wrong on-disk schema).
- Schema baseline re-sealed 78 -> 81 (run16b-smoke): +esm2_llr, +maxentscan_delta,
  +reactome_pathway_count (latter two dormant). Green vs all 3 smoke splits.

Learned:
- DbNSFPConnector._cache_path hard-codes dbnsfp_clinvar_index.parquet; the 895 MB full
  index is never read (no OOM risk from the connector).
- ThousandGenomesConnector fills only combined allele_freq; af_1kg_* have no source wired.
- GNN (gnn.py) is complete but unwired; gnn_score is a placeholder; live integration
  needs gene-disjoint cross-fitting to avoid leakage.
- Feature-population audit must target splits/X_*.parquet (not the pre-scoring checkpoint)
  and use varies-checks on standard-scaled data.
"""

ROADMAP_DELTA = """## 2026-06-12 -- Run 16 launch-ready; Run 17 scope committed

Run 16 (tabular): VALIDATED + launch-ready. Flag set frozen
(docs/launch/LAUNCH_CONTRACT_run16.md). Schema sealed at 81 (run16b-smoke). gnn_score,
af_1kg_*, and uniprot features dormant-by-design (sealed, will activate later).

Run 17 (COMMITTED, not deferred -- docs/roadmap/RUN17_SCOPE.md):
- Track A: 1000 Genomes AF -- build_1kg_parquet.py + --kg-path + validate AF-fill;
  resolve af_1kg_* per-population stubs (wire or formally retire).
- Track B: STRING-DB GNN -- gnn_score live + LEAKAGE-FREE via gene-disjoint cross-fitting,
  held-out-gene no-leak check, WITH/WITHOUT ablation.
Both gated by full-scale feature-population audit + schema drift-check + gene-disjoint
integrity verification before Run 17 trains.
"""


def append_if_absent(path: Path, delta: str, marker: str) -> str:
    raw = path.open("r", encoding="utf-8", newline="").read() if path.exists() else ""
    if marker in raw:
        return "already present (no-op)"
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if text and not text.endswith("\n"):
        text += "\n"
    text += "\n" + delta.strip("\n") + "\n"
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.open("w", encoding="utf-8", newline="").write(text.replace("\n", nl))
    return "appended"


def main() -> int:
    # session doc (create-once)
    if SESSION_PATH.exists():
        print(f"session: exists, skip ({SESSION_PATH})")
    else:
        SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        SESSION_PATH.open("w", encoding="utf-8", newline="\n").write(SESSION_BODY)
        print(f"session: created {SESSION_PATH}")

    print(f"CHANGELOG: {append_if_absent(CHANGELOG, CHANGELOG_DELTA, CHANGELOG_MARKER)}")
    print(f"ROADMAP:   {append_if_absent(ROADMAP, ROADMAP_DELTA, ROADMAP_MARKER)}")
    print("NEXT: python scripts/make_roadmap_docx.py ; review git diff ; commit + push.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
