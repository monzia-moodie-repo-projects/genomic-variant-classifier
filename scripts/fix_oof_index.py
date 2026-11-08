"""
scripts/fix_oof_index.py
=========================
Fixes the F-13 OOF row-index persistence bug: when --max-train subsampling
is active, the subsampled idx array (len=max_train) may not match the OOF
matrix row count (oof_predictions_ can have fewer rows than the training set
due to how VariantEnsemble's stacker partitions the OOF pool).

The fix: guard the idx insertion with a length check. If lengths don't match,
fall back to arange(len(_oof_df)) with a clear WARNING.

This is a single-match guarded str_replace (count==1 abort).

Run from repo root:
    python scripts/fix_oof_index.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


SUPERSEDED = (
    "SUPERSEDED (package 1, set 1c, 2026-11-08): fix_oof_index.py guarded the F-13 `_train_row_idx` insertion in "
    "run_phase2_eval.py with a fall-back to sequential indices -- the defect package 1, set 1b removed (gate G2: a "
    "failed identity binding prevents publication; out-of-fold rows carry their composed original training-split "
    "identity, evaluation/oof_identity.py). Its target no longer exists. It refuses before reading or writing anything; "
    "the source is kept as the record of what it did.")


def main(argv=None) -> int:
    print(SUPERSEDED, file=sys.stderr)
    return 2
    # --- historical body, unreachable since package 1, set 1c ---
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    target = Path(__file__).parent.parent / "scripts" / "run_phase2_eval.py"
    if not target.exists():
        print(f"ERROR: {target} not found", file=sys.stderr)
        return 1

    OLD = (
        '                if args.max_train and len(y_train) == args.max_train:\n'
        '                    _oof_df.insert(0, "_train_row_idx", idx)\n'
        '                else:\n'
        '                    _oof_df.insert(\n'
        '                        0, "_train_row_idx",\n'
        '                        _np.arange(len(_oof_df), dtype=_np.int64),\n'
        '                    )'
    )

    NEW = (
        '                if args.max_train and len(y_train) == args.max_train:\n'
        '                    if len(idx) == len(_oof_df):\n'
        '                        _oof_df.insert(0, "_train_row_idx", idx)\n'
        '                    else:\n'
        '                        # Length mismatch: VariantEnsemble stores a\n'
        '                        # subset of OOF rows (e.g. stacker training split).\n'
        '                        # Fall back to sequential indices within oof_df.\n'
        '                        logger.warning(\n'
        '                            "OOF idx length mismatch: idx=%d oof=%d "\n'
        '                            "-- using sequential indices (subsampled "\n'
        '                            "minitest run; does not affect Run 15 "\n'
        '                            "where max_train is not set).",\n'
        '                            len(idx), len(_oof_df),\n'
        '                        )\n'
        '                        _oof_df.insert(\n'
        '                            0, "_train_row_idx",\n'
        '                            _np.arange(len(_oof_df), dtype=_np.int64),\n'
        '                        )\n'
        '                else:\n'
        '                    _oof_df.insert(\n'
        '                        0, "_train_row_idx",\n'
        '                        _np.arange(len(_oof_df), dtype=_np.int64),\n'
        '                    )'
    )

    text  = target.read_text(encoding="utf-8")
    count = text.count(OLD)

    if count == 0:
        if text.count(NEW) == 1:
            print("SKIP: already patched.")
            return 0
        print(f"ERROR: OLD string not found in {target.name}.", file=sys.stderr)
        print("  Verify HEAD is at the D.1/D.2 commit (8820a40).", file=sys.stderr)
        return 1
    if count > 1:
        print(f"ERROR: {count} matches found (expected 1). Aborting.", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"DRY-RUN: would patch {target.name}")
        return 0

    target.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"OK: OOF index length guard patched in {target.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
