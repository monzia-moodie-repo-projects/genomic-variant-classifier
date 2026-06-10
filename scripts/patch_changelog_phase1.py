#!/usr/bin/env python3
"""patch_changelog_phase1.py -- append Phase 0 + Phase 1 + hygiene to docs/CHANGELOG.md.

docs/CHANGELOG.md is chronological (newest at the BOTTOM), so this appends a new
dated section at the end (distinct from the roadmap S9 changelog, which is
newest-first). Append-only, idempotent via marker. Author: Monzia Moodie."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CL = REPO / "docs/CHANGELOG.md"
MARKER = "## 2026-06-10 (cont.): Phase 0 gene-resolution + Phase 1 ESM-2 LLR feature"

SECTION = "\n" + MARKER + "\n\n" + (
    "- Phase 0 (commit fd5e293): new data/gene_symbols.py (normalize_gene_symbol,\n"
    "  gene_symbol_candidates; full symbol then ;-split components; never splits '-',\n"
    "  protecting HLA-A / NKX2-1 / readthrough fusions). Wired into esm2 (_get_sequence\n"
    "  candidate loop + _missing_genes aggregate log), eve (fixed a real case-drift bug:\n"
    "  variant _gene_symbol .fillna(\"\") un-upper-cased vs an upper-cased lookup; now\n"
    "  normalizes both keys + drops empty-gene rows), protein_pipeline (get_accession\n"
    "  candidate loop). Suite 849 passed / 1 skipped.\n"
    "- Phase 1 (commit fd612e9): ESM-2 650M LLR scorer + esm2_llr feature.\n"
    "  - Scorer (data/esm2.py): _load_transformers_mlm (EsmForMaskedLM logits head,\n"
    "    distinct from the EsmModel embedding loader); _llr_from_logit_row\n"
    "    (logit[mut]-logit[wt]; partition function cancels -> normalization-domain-\n"
    "    invariant); _score_llr (WT-marginal = 1 pass/protein default; masked-marginal\n"
    "    opt-in; skips wt_aa-vs-sequence mismatches, counted); annotate_llr.\n"
    "  - CPU correctness gate (scripts/probe_esm2_llr.py) PASS: TP53 R175H/R248Q/R273H\n"
    "    WT-marginal -9.13 / -11.04 / -9.61 (pathogenic, negative); benign P72R -6.09;\n"
    "    every wt_aa matched the residue at its token index; WT- and masked-marginal\n"
    "    agree in sign.\n"
    "  - CALIBRATION: LLR sign is NOT a class label -- benign P72R also scores negative,\n"
    "    just less so. esm2_llr is a CONTINUOUS feature; the ensemble learns the\n"
    "    threshold (never a hard LLR<0 => pathogenic cutoff).\n"
    "  - Feature wired 79 -> 80: TABULAR_FEATURES += esm2_llr (after esm2_delta_norm);\n"
    "    EXPECTED_TABULAR_FEATURE_COUNT 79->80; INFERENCE_FEATURE_COLUMNS auto-derived\n"
    "    (list(TABULAR_FEATURES)). Assembled at BOTH sites (real_data_prep +\n"
    "    variant_ensemble) SIGNED with NO clip -- clipping would have silently zeroed the\n"
    "    pathogenic signal; a regression test fails loudly if a clip is reintroduced.\n"
    "  - Harness reference slice (correctness_harness.build_reference_slice) now\n"
    "    populates esm2_llr with a signed range -- a live feature, NOT added to\n"
    "    KNOWN_ZERO_DEFAULT (that set is dead-connectors only).\n"
    "  - Model default stays esm2_t6_8M_UR50D (CI fast, no 2.5GB download); regen MUST\n"
    "    set esm2_model_name=esm2_t33_650M_UR50D (printed in the step-16b log).\n"
    "  - Full suite 862 passed / 1 skipped.\n"
    "- Repo hygiene (commit a59d728): tracked prior-session diagnostics\n"
    "  (probe_uniprot_index, diagnose_esm2_coverage, clinvar_name_probe) + the step-10b\n"
    "  coverage-gate patcher (patch_add_protein_coord_coverage_gate, for committed\n"
    "  34e125a); .gitignore += *_bak_* (consolidation backups used _bak_, escaping the\n"
    "  existing *.bak_*).\n"
    "- Carried: Phase 2 = ESM C 600M (Cambrian, \"Built with ESM\"); Phase 3 = GPU regen +\n"
    "  LLR recalibration (signed-feature scaling); stale step-count log denominators\n"
    "  (/16, /17 vs 18 steps) cleanup; clingen int/float dtype drift before regen.\n"
)


def main() -> int:
    if not CL.exists():
        print(f"ABORT: missing {CL}")
        return 2
    text = CL.read_text(encoding="utf-8")
    if MARKER in text:
        print("  skip (already applied): Phase 0/1 changelog section present")
        return 0
    if not text.endswith("\n"):
        text += "\n"
    CL.write_text(text + SECTION, encoding="utf-8")
    print("  ok: appended Phase 0 + Phase 1 + hygiene section to docs/CHANGELOG.md")
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
