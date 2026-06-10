#!/usr/bin/env python3
"""patch_add_protein_coord_coverage_gate_v2.py

Author: Monzia Moodie

Corrected step-10b protein-coord coverage gate. v1 raised UNCONDITIONALLY, which
broke 12 stub-mode tests: when no AlphaMissense source is present, the connector
intentionally leaves protein_pos unset and ESM-2 stubs to 0.0 (valid path for unit
tests and boxes without the 613 MB TSV). v2 enforces the gate ONLY when a coord
source (built cache OR AlphaMissense TSV) is actually present -- which is exactly
the Run 15 condition (a stale cache WAS loaded at 0.2% coverage) and not the stub
case.

Apply to the CLEAN (pre-v1) file. If v1 was applied, REVERT first:
    Copy-Item "src\\genomic_variant_classifier\\data\\real_data_prep.py.bak_<ts>" \
              "src\\genomic_variant_classifier\\data\\real_data_prep.py" -Force

Three count-guarded edits: two pure helpers after _parse_codon_position, the config
field, and a CONDITIONAL gate call after the step-10b log. Backup-first, idempotent,
py_compile-gated.
"""
from __future__ import annotations

import datetime as _dt
import py_compile
import shutil
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/real_data_prep.py")
MARKER = "_assert_protein_coord_coverage"

HELPER_ANCHOR = (
    "def _parse_codon_position(hgvsp: object) -> int:\n"
    "    if not hgvsp:\n"
    "        return 0\n"
    "    m = _HGVSP_CODON_RE.search(str(hgvsp))\n"
    "    return int(m.group(1)) if m else 0\n"
)

HELPER_BLOCK = '''

def _protein_coord_source_present(cache_path: Path, am_path: object) -> bool:
    """True iff a protein-coord SOURCE is available (a built cache file, or the
    AlphaMissense TSV) -- i.e. the connector is NOT in stub mode. The coverage gate
    is enforced ONLY when this is True. Stub mode (no source) is a valid path --
    unit tests and boxes without the 613 MB TSV -- and must never raise; the
    connector already warns there.
    """
    if am_path is not None and Path(str(am_path)).exists():
        return True
    return Path(str(cache_path)).exists()


def _assert_protein_coord_coverage(df: pd.DataFrame, min_cov: float) -> float:
    """Fail loud if the AlphaMissense protein-coordinate merge covered too few
    missense variants.

    AlphaMissense supplies (protein_pos, wt_aa, mut_aa) for ~97% of canonical
    missense SNVs, so a near-zero coverage WHEN A SOURCE IS PRESENT means the coord
    index is stale or mismatched on this box -- the silent ESM-2 zero that capped
    Run 15 at 3,451 of ~2.49M missense. Aborts BEFORE any model trains. Returns the
    coverage fraction. (Only called when _protein_coord_source_present is True.)
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
            f"min_protein_coord_coverage={min_cov}. A protein-coord source IS present "
            f"but covers almost no missense -- the AlphaMissense index "
            f"(data/external/alphamissense/alphamissense_protein_index.parquet) is stale "
            f"or mismatched for THIS cohort/box (expected ~0.97). Rebuild and ship it to "
            f"the training box before training -- see ESM-2 coverage incident."
        )
    return cov
'''

CONFIG_ANCHOR = (
    "    reactome_path: Optional[Path] = None  # Phase D: Reactome gene pathway-count parquet\n"
)
CONFIG_ADD = CONFIG_ANCHOR + (
    "    min_protein_coord_coverage: float = 0.50  # Phase D: fail-loud gate on step-10b coord coverage WHEN a source is present (observed ~0.97; <0.50 => stale/mismatched index)\n"
)

GATE_ANCHOR = (
    '        logger.info(\n'
    '            "Score annotation 10b (protein coords): %d variants with protein_pos.",\n'
    '            int(df.get("protein_pos", pd.Series([pd.NA] * len(df), index=df.index)).notna().sum()),\n'
    '        )\n'
)
GATE_ADD = GATE_ANCHOR + (
    '        # Coverage gate -- enforce ONLY when a coord source is present (NOT in stub\n'
    '        # mode). A source present + near-zero coverage is the Run 15 silent-zero.\n'
    '        if _protein_coord_source_present(pc.cache_path, ac.alphamissense_path):\n'
    '            _coord_cov = _assert_protein_coord_coverage(df, ac.min_protein_coord_coverage)\n'
    '            logger.info("Protein-coord coverage gate PASS: %.4f of missense have coords.", _coord_cov)\n'
    '        else:\n'
    '            logger.info("Protein-coord coverage gate SKIPPED (stub mode: no AlphaMissense source present).")\n'
)


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
        raise SystemExit(
            "ABORT: marker already present. If this is the BROKEN v1 gate, revert from "
            "the .bak_<ts> backup first, then re-run this v2 patcher."
        )

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_suffix(TARGET.suffix + f".bak_{ts}")
    shutil.copy2(TARGET, backup)
    print(f"backup -> {backup}")

    out = src
    out = _apply(out, HELPER_ANCHOR, HELPER_ANCHOR + HELPER_BLOCK, "helpers")
    out = _apply(out, CONFIG_ANCHOR, CONFIG_ADD, "config")
    out = _apply(out, GATE_ANCHOR, GATE_ADD, "gate")

    TARGET.write_text(out, encoding="utf-8")
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(backup, TARGET)
        raise SystemExit(f"ABORT: py_compile failed, reverted from backup.\n{exc}")

    print("OK: 3 edits applied; py_compile clean.")
    print("  + _protein_coord_source_present helper (stub-mode guard)")
    print("  + _assert_protein_coord_coverage helper")
    print("  + AnnotationConfig.min_protein_coord_coverage = 0.50")
    print("  + CONDITIONAL step-10b coverage gate (skips in stub mode)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
