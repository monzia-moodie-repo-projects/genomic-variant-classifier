#!/usr/bin/env python3
"""patch_roadmap_phase1_llr.py -- in-place ROADMAP.md updates for Phase 1 (LLR).

Live-status edits (middle-path: live fields in place, changelog append-only):
  1. S1 architecture feature count 79 -> 80 (current code contract)
  2. S3 snapshot date -> 2026-06-10
  3. S3 suite count 804 -> 862 (stale live field; was the Run-15-seal count)
  4. S4B ESM-2 row -> Phase 1 DONE (esm2_llr primary; secondary delta; signed/no-clip)
  5. feature glossary: add esm2_llr line after esm2_delta_norm
  6. changelog: append Phase 0 + Phase 1 sub-bullets under the 2026-06-10 entry

S3 line 37 ("79 features", Run 15) and line 41 (delta ~99.7% zero in Run 15) are
LEFT as historical record. Regenerate ROADMAP.docx after. Author: Monzia Moodie.
"""
from __future__ import annotations

import datetime as _dt
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RM = REPO / "docs/ROADMAP.md"

EDITS = [
    (  # 1
        "ClinVar missense), 79 features, 13-model ensemble + stacking meta-learner + STRING-DB GNN + KAN.",
        "ClinVar missense), 80 features, 13-model ensemble + stacking meta-learner + STRING-DB GNN + KAN.",
        "ClinVar missense), 80 features, 13-model",
        "S1 feature count 79 -> 80",
    ),
    (  # 2
        "# 3. Current state snapshot (2026-06-09)",
        "# 3. Current state snapshot (2026-06-10)",
        "snapshot (2026-06-10)",
        "S3 snapshot date -> 2026-06-10",
    ),
    (  # 3
        "- Suite: 804 passed / 1 skipped.",
        "- Suite: 862 passed / 1 skipped (2026-06-10, post Phase-0/1 gene-resolution + ESM-2 LLR wiring; features 79 -> 80).",
        "862 passed / 1 skipped",
        "S3 suite count 804 -> 862",
    ),
    (  # 4
        "| ESM-2 | esm2_delta_norm | local model+index | code-FIXED 2026-06-10: the ~3,451 cap was a STALE protein-coord index (gate 34e125a; local ceiling 96.6%); gene-resolution hardened (Phase 0); realizes ~2.4M scores after the Run 16 coord-sync; LLR + 650M/ESM C migration in progress |",
        "| ESM-2 | esm2_delta_norm (secondary), **esm2_llr** (primary, NEW) | local model+index | Phase 1 DONE 2026-06-10: esm2_llr LLR scorer (EsmForMaskedLM logits head; WT-marginal default, masked opt-in) + feature wired (79->80 lockstep; SIGNED, NOT clipped). CPU sign/index gate PASS; sign != class (continuous). Realizes after Run 16 coord-sync with esm2_model_name=esm2_t33_650M_UR50D. ESM C 600M = Phase 2 |",
        "Phase 1 DONE 2026-06-10: esm2_llr LLR scorer",
        "S4B ESM-2 row -> Phase 1 DONE",
    ),
    (  # 5
        "esm2_delta_norm       -- ESM-2 embedding L2 distance (wt vs. mut); ~+0.03-0.06 AUROC",
        "esm2_delta_norm       -- ESM-2 embedding L2 distance (wt vs. mut); ~+0.03-0.06 AUROC (SECONDARY)\n"
        "esm2_llr              -- ESM-2 650M log-likelihood-ratio (logit[mut]-logit[wt]); SIGNED, negative=damaging; CONTINUOUS (sign != class; benign TP53 P72R also negative ~-6.09); WT-marginal default / masked opt-in (PRIMARY)",
        "esm2_llr              -- ESM-2 650M",
        "feature glossary += esm2_llr",
    ),
    (  # 6
        "  * Roadmap consolidation: the pre-rebaseline repo-root ROADMAP.md archived verbatim into Appendix A and removed from repo root; *.bak_* gitignored; README live-link disambiguated. Single ground-truth living roadmap.",
        "  * Roadmap consolidation: the pre-rebaseline repo-root ROADMAP.md archived verbatim into Appendix A and removed from repo root; *.bak_* gitignored; README live-link disambiguated. Single ground-truth living roadmap.\n"
        "  * Phase 0 (commit fd5e293): shared gene_symbols.py resolution helper wired into esm2/eve/protein_pipeline; aggregate missing-gene logging; fixed a real eve case-drift bug; safe ;-join recovery. Suite 849 passed.\n"
        "  * Phase 1: ESM-2 650M LLR scorer (annotate_llr; EsmForMaskedLM logits head; WT-marginal default, masked opt-in) + esm2_llr feature wired (TABULAR_FEATURES 79->80, both assembly sites, SIGNED/NOT clipped; INFERENCE_FEATURE_COLUMNS auto-derived). CPU sign/index gate PASS (TP53 hotspots negative; benign P72R less negative). CALIBRATION: LLR sign != class -> continuous feature (no hard cutoff). Harness reference slice populates esm2_llr (live, NOT allowlisted). Suite 862 passed / 1 skipped. Model default stays 8M; regen sets esm2_model_name=esm2_t33_650M_UR50D (visible in step-16b log).",
        "Phase 1: ESM-2 650M LLR scorer (annotate_llr",
        "changelog += Phase 0/1 sub-bullets",
    ),
]


def main() -> int:
    if not RM.exists():
        print(f"ABORT: missing {RM}")
        return 2
    text = RM.read_text(encoding="utf-8")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(RM, f"{RM}.bak_{ts}")
    for old, new, marker, label in EDITS:
        if marker in text:
            print(f"  skip (already applied): {label}")
            continue
        n = text.count(old)
        if n != 1:
            print(f"ABORT: anchor '{label}' found {n}x (expected 1); nothing written")
            return 3
        text = text.replace(old, new, 1)
        print(f"  ok: {label}")
    RM.write_text(text, encoding="utf-8")
    print(f"DONE. (backup -> ROADMAP.md.bak_{ts})  Regenerate .docx after.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
