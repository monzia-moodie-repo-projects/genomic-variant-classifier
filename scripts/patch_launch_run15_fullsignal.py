#!/usr/bin/env python3
"""patch_launch_run15_fullsignal.py

Convert scripts/launch_run15_baseline.sh from the minimal honest-baseline input
set to the FULL-SIGNAL Run 15 config decided in RUN_15_PLAN.md v2 (B7/B8/B9/B5):

  + preflight HARD-GATE: gnomAD-constraint + dbNSFP ClinVar-index (fail loud, not
    silent-default)
  + ARGS: --gnomad-constraint, --dbnsfp-path, --lovd-path (ON-if-present guard),
    --unseen-gene-holdout
  + header L14-16 rewritten so the doc matches the new intent

Properties: idempotent (no-op if already patched), count-guarded (asserts each
anchor occurs exactly once), backup-first (.bak), line-ending-preserving.
Run from the repo root.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET = Path("scripts/launch_run15_baseline.sh")

# dbNSFP lives at data/external/dbnsfp/ (the --help path), NOT RUN_15_PLAN's
# data/raw/cache/... which was wrong. gnomAD-constraint + LOVD paths verified on disk.
CONSTRAINT = "$DATA/external/gnomad/gnomad.v4.1.constraint_metrics.tsv"
DBNSFP = "$DATA/external/dbnsfp/dbnsfp_clinvar_index.parquet"


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    if not TARGET.exists():
        return fail(f"{TARGET} not found (run from repo root)")

    raw = TARGET.read_bytes()
    nl = b"\r\n" if b"\r\n" in raw else b"\n"
    text = raw.replace(b"\r\n", b"\n").decode("utf-8")

    # ---- idempotency guard --------------------------------------------------
    if "--gnomad-constraint" in text and "--unseen-gene-holdout" in text:
        print("ALREADY PATCHED (found --gnomad-constraint and --unseen-gene-holdout); no-op.")
        return 0

    # ---- edit 1: preflight required-list (fail-loud on missing) -------------
    a1 = '    "$DATA/external/alphamissense/AlphaMissense_hg38.tsv.gz" \\\n'
    if text.count(a1) != 1:
        return fail(f"preflight anchor count={text.count(a1)} (expected 1)")
    add1 = a1 + f'    "{CONSTRAINT}" \\\n    "{DBNSFP}" \\\n'
    text = text.replace(a1, add1, 1)

    # ---- edit 2: ARGS block -------------------------------------------------
    a2 = 'ARGS="$ARGS --output $OUTDIR"\n'
    if text.count(a2) != 1:
        return fail(f"ARGS anchor count={text.count(a2)} (expected 1)")
    insert2 = (
        f'ARGS="$ARGS --gnomad-constraint {CONSTRAINT}"\n'
        f'ARGS="$ARGS --dbnsfp-path {DBNSFP}"\n'
        '# LOVD is ON-if-present (RUN_15_PLAN B9): guard so absence never silently zeroes it.\n'
        'LOVD_PARQUET="$DATA/external/lovd/lovd_all_variants.parquet"\n'
        'if [ -f "$LOVD_PARQUET" ]; then\n'
        '    ARGS="$ARGS --lovd-path $LOVD_PARQUET"\n'
        '    echo "==> LOVD wired: $LOVD_PARQUET" | tee -a "$LOG"\n'
        'else\n'
        '    echo "==> LOVD absent ($LOVD_PARQUET); proceeding without it (B9 if-present)" | tee -a "$LOG"\n'
        'fi\n'
        'ARGS="$ARGS --unseen-gene-holdout"\n'
    )
    text = text.replace(a2, insert2 + a2, 1)

    # ---- edit 3: header L14-16 ---------------------------------------------
    a3 = (
        "#   - no LOVD / dbNSFP required: the honest-baseline input set only.\n"
        "#   - no --unseen-gene-holdout: that is the C3 ablation (a second full retrain),\n"
        "#     not the baseline.\n"
    )
    if text.count(a3) != 1:
        return fail(f"header anchor count={text.count(a3)} (expected 1)")
    r3 = (
        "#   - FULL-SIGNAL (RUN_15_PLAN v2 B7/B8/B9): gnomAD-constraint + dbNSFP +\n"
        "#     LOVD(if present) wired so the honest baseline holds Run-14's feature\n"
        "#     set CONSTANT and isolates the de-leaking effect, not a feature change.\n"
        "#   - --unseen-gene-holdout ON (B5, C3 gate >= 0.95): adds ~5h (second retrain).\n"
    )
    text = text.replace(a3, r3, 1)

    # ---- backup + write (preserve original line ending) ---------------------
    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copyfile(TARGET, bak)
    TARGET.write_bytes(text.encode("utf-8").replace(b"\n", nl))

    print(f"OK: patched {TARGET}  (backup: {bak})")
    print("  [1/3] preflight hard-gate += gnomAD-constraint, dbNSFP")
    print("  [2/3] ARGS += --gnomad-constraint, --dbnsfp-path, --lovd-path(guarded), --unseen-gene-holdout")
    print("  [3/3] header L14-16 rewritten to full-signal intent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
