#!/usr/bin/env python3
"""install_docs_close_cnn_rna.py -- close the 2026-06-11 (late PM) CNN + RNA-delta
session. Appends to docs/CHANGELOG.md and docs/ROADMAP.md and creates the session
log. Idempotent (marker-guarded), backup-first, no-BOM, newline-preserving, ASCII.
After running, regenerate the docx:  python scripts\\make_roadmap_docx.py
Author: Monzia Moodie."""
from __future__ import annotations
import shutil, sys
from pathlib import Path

CHANGELOG = Path("docs/CHANGELOG.md")
ROADMAP   = Path("docs/ROADMAP.md")
SESSION   = Path("docs/sessions/SESSION_2026-06-11_cnn-rna-activation.md")

CHANGELOG_MARKER = "<!-- docs-close: e3bcd79 cnn-rna-activation -->"
ROADMAP_MARKER   = "## ROADMAP delta -- 2026-06-11 (late PM, CNN + RNA activation)"

CHANGELOG_ENTRY = """<!-- docs-close: e3bcd79 cnn-rna-activation -->
## 2026-06-11 (late PM) -- CNN real-sequence + RNA MaxEntScan-delta activation

### Fixed
- 1D-CNN trained on poly-A placeholders: train.py gated the CNN on the deprecated,
  empty single `fasta_seq` column (notna=0 cohort-wide) and raised NotImplementedError
  on the real-sequence path. Repointed the gate and X_seq plumbing to the live
  [fasta_seq_ref, fasta_seq_alt] delta windows -- test-side from meta_test, train-side
  from the already-persisted meta_train.parquet (gene-split-aligned to X_train via the
  shared train_idx). NotImplementedError removed; NO DataPrepPipeline.run() signature
  change (fb12c0f).
- RNA-splice maxentscan_score dead (default 0.0 for every variant): rna_pipeline read the
  same empty single `fasta_seq`. Repointed to score the ref/alt windows and emit a NEW
  variant-specific feature maxentscan_delta = score(alt) - score(ref) (e3bcd79).

### Added
- maxentscan_delta registered in TABULAR_FEATURES, BOTH _engineer_features blocks
  (variant_ensemble.py and real_data_prep.py), and the RNA-off default-fill tuple;
  EXPECTED_TABULAR_FEATURE_COUNT 80 -> 81. INFERENCE_FEATURE_COLUMNS auto-derives
  (list(TABULAR_FEATURES)); the feature-count contract is green at 81/81.
- Correctness-harness reference slice now populates maxentscan_delta (non-zero synthetic)
  so stage-5 silent-zero detection stays honest -- deliberately NOT allowlisted in
  KNOWN_ZERO_DEFAULT (it is a live feature that should carry signal).
- tests/unit/test_train_cnn_activation.py; tests/unit/test_rna_maxentscan_delta.py.
- Idempotent patchers: patch_train_cnn_activation.py and patch_rna/ve/rdp/
  correctness_harness_maxentscan_delta.py.

### Verification
- Full suite 893 passed / 6 skipped (e3bcd79). The torch-gated CNN test trains end-to-end
  on a 2-column ref/alt delta frame and returns finite probabilities. maxentscan_delta is
  nonzero for a real ref!=alt splice variant and 0 for ref==alt / non-splice / legacy
  single-fasta_seq fallback.
- The correctness harness caught the one defect this session: maxentscan_delta added without
  its reference-slice entry tripped stage-5 (all-zero outside the dead-connector allowlist);
  py_compile, the feature-count contract, and the targeted tests all passed it through.

### Learned
- Activation precondition (load-bearing): real_data_prep NEVER adds fasta_seq* columns; they
  ride ONLY from the input parquet (_load_and_label preserves all input columns). Both the CNN
  and maxentscan_delta activate ONLY when Run 16 uses
  --clinvar data\\processed\\clinvar_grch38_clean_seq.parquet (the ref/alt cohort). With the
  default clinvar_grch38.parquet they degrade SILENTLY to inert (CNN dropped to placeholders,
  maxentscan_delta all-zero) -- no crash, no signal.
- New standing run-gate: every new tabular feature must appear, POPULATED, in the correctness-
  harness reference slice; it was the only gate that caught this session's feature/slice drift.
- The always-donor MaxEntScan selection bug does NOT collapse the delta (the variant base at
  window center lies inside the donor 9-mer); biology-correct donor/acceptor selection is a
  separate tracked fix.

### Commits
- fb12c0f (CNN real-sequence activation), e3bcd79 (maxentscan_delta + harness slice + contract
  bump). Both on origin/main.
"""

ROADMAP_DELTA = """## ROADMAP delta -- 2026-06-11 (late PM, CNN + RNA activation)

### Done
- [x] 1D-CNN activated on real [fasta_seq_ref, fasta_seq_alt] delta windows; train-side via the
  persisted meta_train.parquet (gene-split-aligned to X_train); NotImplementedError removed
  (fb12c0f). SUPERSEDES the "Open -- blocking Run 16" item "CNN train-sequence
  NotImplementedError ... (INCIDENT_2026-05-30)".
- [x] RNA MaxEntScan activated: maxentscan_delta = score(alt) - score(ref), a NEW
  variant-specific splice-disruption feature. The MaxEntScan source moves from Section 4B
  (Scaffolded but DEAD/partial) to LIVE. maxentscan_score keeps its meaning (ref-window score).
  EXPECTED_TABULAR_FEATURE_COUNT 80 -> 81 (e3bcd79).

### Run 16 launch contract -- ADDITION (preflight MUST Test-Path)
- --clinvar data\\processed\\clinvar_grch38_clean_seq.parquet  (the ref/alt cohort). Without it
  BOTH the CNN and maxentscan_delta degrade to inert: _load_and_label preserves input columns,
  but real_data_prep never adds fasta_seq* itself, so the ref/alt windows exist on the frame
  ONLY if the input cohort carries them. Joins the existing --esm2-model esm2_t33_650M_UR50D,
  --esm2-uniprot-index, and --alphamissense requirements.

### Open -- post-regen / parallel (updated)
- [ ] Schema baseline refresh: regenerate data/reference/schema/schema_baseline.json from the
  post-Run-16 X_train. Target = EXPECTED_TABULAR_FEATURE_COUNT (now 81: +esm2_llr +maxentscan_delta
  vs the sealed-78 baseline). SUPERSEDES the earlier "Schema baseline refresh 78 -> 79" line. The
  pre-existing 78/79/80 spread (Feature-count reconciliation, TO VERIFY) reconciles AT this regen
  by diffing actual X_train columns against TABULAR_FEATURES -- not asserted here.
- [ ] Always-donor MaxEntScan selection bug: donor/acceptor choice is bounds-based (always donor
  for a 101bp window), so maxentscan_delta measures a donor perturbation even for acceptor-region
  variants. Biology-correct selection (drive from dist_to_donor/dist_to_acceptor) is the next RNA
  item. Does NOT block Run 16.

### Standing discipline -- ADDITION
- Every new tabular feature must appear, POPULATED (non-zero / non-degenerate), in the
  correctness-harness reference slice (build_reference_slice). This session the harness stage-5
  silent-zero tripwire was the ONLY gate that caught a feature added without its slice entry.
"""

SESSION_DOC = """# Session 2026-06-11 (late PM) -- CNN real-sequence + RNA MaxEntScan-delta activation

## Summary
Two pre-Run-16 feature activations landed on origin/main, full suite green throughout
(893 passed / 6 skipped):
- fb12c0f -- 1D-CNN activated on real [fasta_seq_ref, fasta_seq_alt] delta windows.
- e3bcd79 -- RNA MaxEntScan delta (maxentscan_delta); EXPECTED_TABULAR_FEATURE_COUNT 80 -> 81.

Both were view-first / probe-before-change: every consumer and call site was read and the core
logic validated in a sandbox before any file was written, in line with the project's
"wired != populated != non-zero" failure mode.

## CNN real-sequence activation (fb12c0f)
The cohort's single `fasta_seq` column is deprecated and empty (notna=0 across 4.4M rows), while
[fasta_seq_ref, fasta_seq_alt] are ~100% populated. train.py gated the CNN on the empty column,
so the CNN always ran on "A"*101 placeholders and the real-sequence branch raised
NotImplementedError (claiming meta_train was unavailable). In fact DataPrepPipeline._save_splits
already persists meta_train.parquet (gene-split-aligned to X_train: both are
df.iloc[train_idx].reset_index, scaling preserves order, parquet preserves order). The fix:
gate on the ref/alt columns; build test-side windows from meta_test and train-side from
meta_train.parquet via seq_window_join.attach_delta_windows; remove the raise. No run() signature
change. The ensemble routes X_seq to the CNN via .iloc[idx].reset_index, which is DataFrame-safe,
and CNN1DClassifier._pair_arrays hits its real delta path on a 2-column frame. The CNN is a base
estimator (its OOF feeds the stacker), so NO feature-matrix schema change.

## RNA MaxEntScan delta (e3bcd79)
rna_pipeline.annotate_dataframe (wired at _annotate_scores step 13, rna_pipeline default True) read
the same empty single `fasta_seq`, so maxentscan_score defaulted to 0.0 everywhere. Repointed to a
_window_score helper applied to BOTH ref and alt windows: maxentscan_score stays the ref-window
score; maxentscan_delta = score(alt) - score(ref) is the new variant-specific signal. Graceful
fallback to legacy single fasta_seq (ref==alt -> delta 0) then defaults.

Registration touched the feature-count contract: maxentscan_delta added to TABULAR_FEATURES, both
_engineer_features blocks (variant_ensemble.py and real_data_prep.py), and the RNA-off default
tuple; EXPECTED_TABULAR_FEATURE_COUNT 80 -> 81. INFERENCE_FEATURE_COLUMNS is list(TABULAR_FEATURES),
so it auto-tracks; the contract test stays green at 81/81 with no edit to it or api/pipeline.py.

## Activation precondition (load-bearing)
real_data_prep does NOT add fasta_seq* columns anywhere; _load_and_label is pd.read_parquet with no
column subsetting, so ref/alt ride through ONLY if the input cohort carries them. Run 16 MUST pass
--clinvar data\\processed\\clinvar_grch38_clean_seq.parquet or BOTH the CNN and maxentscan_delta
degrade silently to inert. Added to the Run-16 launch contract.

## Verification
- Full suite 893 passed / 6 skipped (was 892 + 1 failed before the harness slice fix).
- CNN: torch-gated end-to-end test trains on a 2-column ref/alt delta frame, returns finite proba.
- RNA: maxentscan_delta nonzero for a real ref!=alt splice variant; 0 for ref==alt / non-splice /
  legacy single fasta_seq; graceful (no crash) when neither ref/alt nor fasta_seq is present.
- The correctness harness stage-5 tripwire caught maxentscan_delta added without its reference-slice
  entry -- the only gate that did. Fixed by populating the slice (NOT allowlisting it).

## Carry-forward
- Schema baseline refresh: regenerate from post-Run-16 X_train; target EXPECTED_TABULAR_FEATURE_COUNT
  (now 81). Reconciles the pre-existing 78/79/80 spread at the regen.
- Run-16 launch contract: --clinvar ref/alt cohort + --esm2-model 650M + --esm2-uniprot-index +
  --alphamissense, all Test-Path-gated in the ONE preflight script.
- Always-donor MaxEntScan selection bug: biology-correct donor/acceptor selection from
  dist_to_donor/dist_to_acceptor is the next RNA item (does not block Run 16).
- New standing run-gate: every new feature must appear populated in the correctness-harness slice.

## Key learnings
- meta_train.parquet was already persisted; the "needs run() plumbing" raise was unnecessary.
  Reading the actual _save_splits body (not memory) found the signature-free path.
- A feature can be wired (in the contract, in engineer_features) and still be silently zero if its
  upstream column is absent from the input cohort -- the launch-contract --clinvar requirement is
  the real activation switch.
- The correctness harness earns its keep: it caught drift that py_compile, the feature-count
  contract, and the targeted unit tests all passed.
"""

def _read_preserve(path: Path) -> str:
    # newline="" disables universal-newline translation so CRLF is detectable
    # (Path.read_text lacks a newline param before Python 3.13).
    with path.open("r", encoding="utf-8", newline="") as f:
        return f.read()

def _write_preserve(path: Path, text: str) -> None:
    # newline="" writes the string verbatim (no platform translation); utf-8 (no BOM).
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text)

def _nl(raw: str) -> str:
    return "\r\n" if "\r\n" in raw else "\n"

def _append(path: Path, marker: str, body: str) -> str:
    if not path.exists():
        return f"MISSING: {path} (not appended)"
    raw = _read_preserve(path)
    if marker in raw:
        return f"already present, skipped: {path.name}"
    nl = _nl(raw)
    text = raw.replace("\r\n", "\n")
    if not text.endswith("\n"):
        text += "\n"
    text = text + "\n" + body.rstrip("\n") + "\n"
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    _write_preserve(path, text.replace("\n", nl))
    return f"appended: {path.name}"

def _create(path: Path, body: str) -> str:
    if path.exists():
        return f"already present, skipped: {path.name}"
    nl = "\r\n"
    if ROADMAP.exists():
        nl = _nl(_read_preserve(ROADMAP))
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_preserve(path, body.replace("\n", nl))
    return f"created: {path.name}"

def main() -> int:
    print(_append(CHANGELOG, CHANGELOG_MARKER, CHANGELOG_ENTRY))
    print(_create(SESSION, SESSION_DOC))
    print(_append(ROADMAP, ROADMAP_MARKER, ROADMAP_DELTA))
    print("NEXT: regenerate the docx (run make_roadmap_docx.py in scripts/), then review git diff.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
