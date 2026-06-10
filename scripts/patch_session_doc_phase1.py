#!/usr/bin/env python3
"""patch_session_doc_phase1.py -- append the Phase 0 + Phase 1 record to today's
session doc (append-only; never overwrites). Idempotent via section marker.
Author: Monzia Moodie."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs/sessions/SESSION_2026-06-10.md"
MARKER = "## Phase 1 -- ESM-2 650M LLR scorer + feature"

SECTION = '''

## Phase 0 + Phase 1 -- shared gene-symbol resolution + ESM-2 650M LLR (2026-06-10)

### Phase 0 -- shared gene-symbol resolution (commit fd5e293)
- New `data/gene_symbols.py`: `normalize_gene_symbol`, `gene_symbol_candidates`
  (full symbol then ;-split components; NEVER splits `-`, protecting HLA-A /
  NKX2-1 / readthrough fusions). 25-case unit test.
- Wired into esm2.py (`_get_sequence` candidate loop + `_missing_genes`
  accumulation + aggregate missing-gene log), eve.py (fixed a real case-drift
  bug: variant `_gene_symbol` was `.fillna("")` un-upper-cased vs an
  upper-cased lookup; now normalizes both keys + drops empty-gene rows),
  protein_pipeline.py (`get_accession` candidate loop).
- database_connectors.UniProtConnector assessed + excluded (panel ingestion,
  case-insensitive server-side query -> helper would be cosmetic).

## Phase 1 -- ESM-2 650M LLR scorer + feature

**Decision (protein-LM research report):** add `esm2_llr` (log-likelihood-ratio)
as the PRIMARY protein-LM feature; keep `esm2_delta_norm` (embedding delta) as
SECONDARY. Method = WT-marginal (one forward pass per protein) by default;
masked-marginal opt-in (`method="masked"`). Model = ESM-2 650M
(`esm2_t33_650M_UR50D`).

**Model-default decision:** the `AnnotationConfig` default stays
`esm2_t6_8M_UR50D` (keeps CI fast; no 2.5 GB download). The regen MUST set
`esm2_model_name=esm2_t33_650M_UR50D` + `esm2_uniprot_index_path`. The new
step-16b log prints the model name, so an accidental 8M run is VISIBLE, not
silent. (Note: config.yaml does not exist at the repo root; the effective
config is the AnnotationConfig dataclass, set at regen invocation.)

**Scorer (data/esm2.py):** `_load_transformers_mlm` (EsmForMaskedLM logits head,
distinct from the EsmModel embedding loader); `_llr_from_logit_row`
(= logit[mut] - logit[wt]; the partition function cancels in the difference, so
the value is invariant to normalization domain -- full vocab vs 20 AAs);
`_mlm_logit_matrix`; `ESM2Connector._score_llr` (WT-marginal = 1 pass/protein;
masked = 1 pass/unique position; skips wt_aa-vs-sequence mismatches, counted in
`_llr_n_mismatch`); `annotate_llr` (adds `esm2_llr`; 0.0 = neutral/unscored;
requires the transformers backend).

**CPU correctness gate (scripts/probe_esm2_llr.py) -- GATE PASS:**

| variant     | label      | WT-marginal | masked | index/wt |
|-------------|------------|-------------|--------|----------|
| TP53 R175H  | pathogenic | -9.13       | -5.97  | OK       |
| TP53 R248Q  | pathogenic | -11.04      | -9.55  | OK       |
| TP53 R273H  | pathogenic | -9.61       | -6.29  | OK       |
| TP53 P72R   | benign     | -6.09       | -2.73  | OK       |

Every wt_aa matched the sequence residue at its token index; the three
DNA-binding hotspots are clearly negative; WT- and masked-marginal agree in sign.

**CALIBRATION FINDING (critical for recalibration): LLR sign is NOT a class
label.** The common benign polymorphism P72R also scores negative (-6.09) --
just less so than the hotspots. `esm2_llr` therefore enters the ensemble as a
CONTINUOUS feature whose threshold the model learns; a hard `LLR < 0 =>
pathogenic` cutoff would misclassify benign variants. Recorded in the
`annotate_llr` docstring and the metrics glossary.

**Feature wiring (79 -> 80):**
- `TABULAR_FEATURES += esm2_llr` (immediately after `esm2_delta_norm`);
  `EXPECTED_TABULAR_FEATURE_COUNT` 79 -> 80. `INFERENCE_FEATURE_COLUMNS` is
  `list(TABULAR_FEATURES)` (derived) -> auto-propagated; contract tripwire green.
- **Clip trap caught + avoided:** both assembly sites build `esm2_delta_norm`
  with `.clip(lower=0.0)` (correct for a norm). `esm2_llr` is SIGNED and is
  assembled with NO clip; clipping would have silently zeroed the entire
  pathogenic signal. A dedicated regression test fails loudly if a clip is ever
  reintroduced (verified by sabotage in the sandbox).
- Both parallel assembly sites (data/real_data_prep.py + models/variant_ensemble.py)
  updated in lockstep.
- Step 16 calls `esm2.annotate_llr(df)` on the same connector; the step-16b log
  prints the model name + scored count.

**Harness zero-audit catch (the audit working as intended):** stage-5 flagged
`esm2_llr` as a new all-zero column in `build_reference_slice` (no 650M model
runs in a unit-test slice). Per the harness convention (live features are
populated; `KNOWN_ZERO_DEFAULT` is dead-connectors only -- cf. the
clingen_validity_score note), the fix POPULATES `esm2_llr` with a signed range
(-12..4) in `build_reference_slice`, NOT an allowlist entry. Allowlisting would
have falsely branded it a dead connector and masked future regressions.

**Memory note:** with model=650M the delta path loads EsmModel and the LLR path
loads EsmForMaskedLM (different cache keys) -> ~2x base weights (~5 GB). Fine on
the RTX 4090. Single-pass optimization (one EsmForMaskedLM yielding logits +
hidden_states) deferred.

**Validation ladder:** pure LLR math (sandbox) -> real-model sign/index
(GATE PASS) -> scorer control-flow (9 torch-free unit tests) -> feature wiring +
no-clip (4 tests incl. the clip-trap guard) -> harness slice (6 tests). Full
suite: **862 passed, 1 skipped** (the lone skip = `test_ablate_gnn`
torch_geometric ABI noise, pre-existing/benign; 220 pre-existing pandas/sklearn
warnings, none new).

**Files:** scripts/{probe_esm2_llr, patch_esm2_llr_scorer,
patch_esm2_llr_feature_wiring, patch_harness_reference_slice_esm2_llr}.py;
tests/unit/{test_esm2_llr, test_esm2_llr_feature_wiring}.py; src edits to
data/esm2.py, models/variant_ensemble.py, data/real_data_prep.py,
agent_layer/harness/correctness_harness.py.

**Carried / next:** Phase 2 = swap ESM C 600M (Cambrian SDK, "Built with ESM"
attribution). Phase 3 = regen with 650M + recalibration (LLR is signed --
recalibrate feature scaling; expected to beat the 8M embedding-delta baseline).
Stale step-count log denominators (`/16`, `/17` against 18 actual steps) need a
dedicated cleanup pass. Regen must set `esm2_model_name=esm2_t33_650M_UR50D`
(now visible in the step-16b log).
'''


def main() -> int:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    existing = DOC.read_text(encoding="utf-8") if DOC.exists() else "# Session 2026-06-10\n"
    if MARKER in existing:
        print("  skip (already applied): Phase 1 session section present")
        return 0
    DOC.write_text(existing + SECTION, encoding="utf-8")
    print(f"  ok: appended Phase 0 + Phase 1 section to {DOC.name}")
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
