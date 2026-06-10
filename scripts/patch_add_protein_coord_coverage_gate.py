#!/usr/bin/env python3
"""patch_add_protein_coord_coverage_gate.py

Author: Monzia Moodie

Adds a FAIL-LOUD coverage gate to the step-10b protein-coordinate annotation in
src/genomic_variant_classifier/data/real_data_prep.py.

WHY
---
Run 15 ran fresh data-prep, loaded the AlphaMissense protein-coord index, and step
10b populated only 3,461 protein_pos (the Vast box held a stale ~3,461-row index;
the local 2.41M index covers 96.7% of missense). The pipeline logged the count at
INFO and trained for 11.5 hours with a dead esm2_delta_norm. This gate makes that
condition abort BEFORE any model trains.

WHAT IT DOES (3 surgical, count-guarded edits)
  1. inserts a pure, testable helper `_assert_protein_coord_coverage(df, min_cov)`
     immediately after `_parse_codon_position`
  2. adds `min_protein_coord_coverage: float = 0.50` to AnnotationConfig
  3. inserts a gate call right after the step-10b "variants with protein_pos" log

SAFETY: backup-first (.bak_<ts>), each anchor must match exactly once (abort
otherwise), idempotent (re-running is a no-op), py_compile-gated (reverts on
failure). No behavioural change except the new abort path.
"""
from __future__ import annotations

import datetime as _dt
import py_compile
import shutil
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/real_data_prep.py")

HELPER_ANCHOR = (
    "def _parse_codon_position(hgvsp: object) -> int:\n"
    "    if not hgvsp:\n"
    "        return 0\n"
    "    m = _HGVSP_CODON_RE.search(str(hgvsp))\n"
    "    return int(m.group(1)) if m else 0\n"
)

HELPER_BLOCK = '''

def _assert_protein_coord_coverage(df: pd.DataFrame, min_cov: float) -> float:
    """Fail loud if the AlphaMissense protein-coordinate merge covered too few
    missense variants.

    AlphaMissense supplies (protein_pos, wt_aa, mut_aa) for ~97% of canonical
    missense SNVs, so a near-zero coverage means the coord index parquet is stale
    or missing on this box -- the silent ESM-2 zero that capped Run 15 at 3,451 of
    ~2.49M missense. Aborts BEFORE any model trains rather than baking a dead
    esm2_delta_norm into a multi-hour run. Returns the coverage fraction.
    """
    is_mm = (
        df.get("is_missense", pd.Series([0] * len(df), index=df.index))
        .fillna(0)
        .astype(int)
    )
    n_mm = int(is_mm.sum())
    if n_mm == 0:
        return 1.0
    pp = df.get("protein_pos", pd.Series([pd.NA] * len(df), index=df.index))
    n_pp_mm = int((is_mm.astype(bool) & pp.notna()).sum())
    cov = n_pp_mm / n_mm
    if cov < min_cov:
        raise ValueError(
            f"Protein-coord coverage {cov:.4f} ({n_pp_mm}/{n_mm} missense) < "
            f"min_protein_coord_coverage={min_cov}. The AlphaMissense protein-coord "
            f"index (data/external/alphamissense/alphamissense_protein_index.parquet) "
            f"is stale or missing for THIS cohort/box (expected ~0.97). Rebuild and "
            f"ship it to the training box before training -- see ESM-2 coverage incident."
        )
    return cov
'''

CONFIG_ANCHOR = (
    "    reactome_path: Optional[Path] = None  # Phase D: Reactome gene pathway-count parquet\n"
)
CONFIG_ADD = (
    "    reactome_path: Optional[Path] = None  # Phase D: Reactome gene pathway-count parquet\n"
    "    min_protein_coord_coverage: float = 0.50  # Phase D: fail-loud gate on step-10b coord coverage (observed ~0.97; <0.50 => stale/missing index)\n"
)

GATE_ANCHOR = (
    '        logger.info(\n'
    '            "Score annotation 10b (protein coords): %d variants with protein_pos.",\n'
    '            int(df.get("protein_pos", pd.Series([pd.NA] * len(df), index=df.index)).notna().sum()),\n'
    '        )\n'
)
GATE_ADD = GATE_ANCHOR + (
    '        _coord_cov = _assert_protein_coord_coverage(df, ac.min_protein_coord_coverage)\n'
    '        logger.info("Protein-coord coverage gate PASS: %.4f of missense have coords.", _coord_cov)\n'
)

MARKER = "_assert_protein_coord_coverage"


def _apply(text: str, anchor: str, replacement: str, label: str) -> str:
    n = text.count(anchor)
    if n != 1:
        raise SystemExit(f"ABORT [{label}]: anchor matched {n} times (need exactly 1). No change written.")
    return text.replace(anchor, replacement, 1)


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"ABORT: {TARGET} not found. Run from the repo root.")
    src = TARGET.read_text(encoding="utf-8")

    if MARKER in src:
        print("Already patched (marker present). No-op.")
        return 0

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_suffix(TARGET.suffix + f".bak_{ts}")
    shutil.copy2(TARGET, backup)
    print(f"backup -> {backup}")

    out = src
    out = _apply(out, HELPER_ANCHOR, HELPER_ANCHOR + HELPER_BLOCK, "helper")
    out = _apply(out, CONFIG_ANCHOR, CONFIG_ADD, "config")
    out = _apply(out, GATE_ANCHOR, GATE_ADD, "gate")

    TARGET.write_text(out, encoding="utf-8")
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(backup, TARGET)
        raise SystemExit(f"ABORT: py_compile failed, reverted from backup.\n{exc}")

    print("OK: 3 edits applied; py_compile clean.")
    print("  + _assert_protein_coord_coverage helper")
    print("  + AnnotationConfig.min_protein_coord_coverage = 0.50")
    print("  + step-10b coverage gate call")
    return 0


if __name__ == "__main__":
    sys.exit(main())
