#!/usr/bin/env python3
"""
append_changelog_2026-06-19.py  --  Monzia Moodie

Idempotent, LF-preserving append of the 2026-06-19 Run-17 launch-kit entry to docs/CHANGELOG.md.
Skips if a '## 2026-06-19 ' header already exists. Run from repo root.
"""
from __future__ import annotations
import sys
from pathlib import Path

TARGET = Path("docs/CHANGELOG.md")
HEADER = "## 2026-06-19 -- "
ENTRY = """\
## 2026-06-19 -- Run-17 launch kit complete; resume-load bug fixed; RNA-seq gene-prior ablation (commit 988439c)

### Added
- `scripts/launch_run17_baseline.sh` -- Run-17 launcher (forked from launch_run15): wires
  `--kg` / `--rnaseq-path` / `--hetero-gnn` / `--kg-edges reactome:<gmt>`; elevates gtex/reactome/kg/rnaseq
  to hard-fail-if-absent (LOVD if-present); `--skip-svm` only; read-only kg+rnaseq column probe;
  `--esm2-uniprot-index` intentionally ABSENT (ESM-2/EVE remain stubbed pending the HGVSp parser).
  OUTDIR=outputs/run17_baseline/full.
- `scripts/merge_1kg_parquets.py` (atomic concat + dup-drop + super-pop non-zero validation) and
  `scripts/probe_1kg_superpop_info.py` (streamed `AF_<POP>` header probe).
- `tests/unit/test_launch_run17.py` (14 activation/required-flag/abort assertions + bash syntax check) and
  `tests/unit/test_run_phase2_resume_load.py` (static guard: resume must route through `VariantEnsemble.load`).
- `docs/RNASEQ_ABLATION_FINDINGS_2026-06-19.md` -- full 5-config ablation write-up + inference contract.

### Fixed
- **Resume crash** (`scripts/run_phase2_eval.py`): resuming after data-prep called raw `joblib.load` on the
  format_version=2 orchestrator DICT, then `ensemble.evaluate()` -> `'dict' object has no attribute
  'evaluate'`. Now routes through `VariantEnsemble.load()`. (patch_run_phase2_resume_load.py)
- **preflight --emit-kg omitted --rnaseq-path**: added `--rnaseq-path` to `preflight_gate.REQUIRED_PATHS`
  (single source of truth) and `_build_mirror_parser`. (patch_preflight_rnaseq_required.py)
- **test_launch_run17 bash check**: passing a Windows path to Git-Bash failed as backslash (escape mangling
  -> `C:Projects...`) AND as forward-slash (`C:/...` is not a `/c/` MSYS mount). Now syntax-checks the
  launcher TEXT via `bash -n -c <content>` -- no path translation, robust on Windows + Linux.
  (patch_test_launch_run17_bashcheck.py)
- **preflight_run17.py module docstring** refreshed (comment-only; code already `EXPECTED_SCHEMA_COLS=87`):
  81-column -> 87-column, `n_columns must be 81` -> 87, `1000G Phase-3` -> `1000G 30x high-coverage GRCh38`.
  (patch_preflight_run17_docstring.py)

### Verified
- Run-17 preflight: **GO -- 0 fail, 0 warn, 23 ok** (87-col schema baseline hash efca0d85a28d; kg carries all
  5 super-pop AF cols; hetero-GNN + reactome kg-edges; STRING cached graph present -> no download).
- `test_launch_run17.py` + `test_run_phase2_resume_load.py` + `test_preflight_run17.py`: **52 passed**.
- kg parquet re-derived: chr1-22 (426,358) + chrX `.v2` (11,310) -> 437,668 unique; super-pop non-zero
  AFR 291432 / EUR 205292 / EAS 154084 / SAS 188461 / AMR 251739.

### Findings -- RNA-seq ablation (reduced-context: spliceai-cache + pLDDT=50 stub + rnaseq + clinvar-derived only; max-train 5000, gene-disjoint, 10 base models)
- Held-out test/val AUROC: full 0.9360/0.9461; drop_de 0.9346/0.9461; gene_shuffle 0.9354/0.9383;
  drop_all 0.9304/0.9370; no_rnaseq 0.9304/0.9370 (== drop_all -> wiring sane).
- Total rnaseq marginal value +0.0056 test / +0.0091 val (~0.6-0.9 pt). DE-block +0.0014 test / 0.0000 val.
- Gene-shuffle retention DISAGREES across splits (test ~89% retained -> non-gene-specific; val ~14% ->
  gene-specific) at <=0.009 magnitude -> **INCONCLUSIVE at this scale**.
- Within-gene AUROC (genes >=2 of each class): test 0.9512 wtd / 0.9261 unwtd (780 genes); val 0.9479 /
  0.9240 (344). Discrimination is variant-level even where ALL gene-level features (incl
  `n_pathogenic_in_gene`) are constant. CONCLUSION: rnaseq importance is a tree split-bias toward
  high-cardinality continuous features (redundant gene-prior), not gene-identity/tissue-contrast reliance.
- INFERENCE CONTRACT: saved base models consume RAW (unscaled) X; applying `scaler.joblib` before
  `predict_proba` double-scales -> trees collapse ~0.45-0.50, blend 0.6083. Standalone inference MUST feed raw X.

### Open
- Gene-shuffle ablation unsettled at this scale -- re-run at Run-17 scale (full feature set, larger
  `--max-train`, >=3 seeds) to settle non-gene-specific vs gene-specific.
- **RECONCILE**: the 2026-06-15 CHANGELOG entry already documents this exact 437,668 kg build (26342e9). This
  session's parquet is content-equivalent (identical variant count + super-pop counts; only parquet-container
  bytes differ), so 988439c added a duplicate ~6 MB blob. Confirm whether 06-18/19 was a planned
  reproducibility re-derivation (so the data build is not double-counted -- the 06-19 deliverable is the
  launch-kit/integration). Consider a content-hash guard before re-committing the parquet, or Git LFS.
- GPU provisioning (Run 17) pending: `Run_Preflight_VM.sh` exit 0 -> all-models smoke (`--max-train ~3000`,
  no `--skip` beyond `--skip-svm`, `--string-db auto`) before any spend.
"""


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found (run from repo root)", file=sys.stderr); return 2
    raw = TARGET.read_bytes()
    text = raw.decode("utf-8-sig")
    if HEADER in text:
        print("[skip] CHANGELOG already has a 2026-06-19 entry"); return 0
    # normalize working text to LF (file is LF), ensure exactly one blank line before the new entry
    body = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    new = body + "\n\n" + ENTRY.rstrip("\n") + "\n"
    TARGET.write_bytes(new.encode("utf-8"))  # LF, no BOM (matches file)
    print(f"[appended] docs/CHANGELOG.md += 2026-06-19 entry ({len(ENTRY)} chars, LF)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
