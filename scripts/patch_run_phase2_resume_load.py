#!/usr/bin/env python3
"""
patch_run_phase2_resume_load.py  --  Monzia Moodie

Fix the resume-path crash in scripts/run_phase2_eval.py:

    Resuming: loading existing ensemble from .../ensemble.joblib
    ERROR Pipeline failed: 'dict' object has no attribute 'evaluate'

VariantEnsemble.save() persists a format_version=2 ORCHESTRATOR DICT (config, meta_learner,
blend_weights_, per-model paths). VariantEnsemble.load() (classmethod) reconstructs the object
from that dict. The resume branch wrongly used a raw `joblib.load()`, which returns the dict, so
the subsequent `ensemble.evaluate(...)` died -- AFTER the expensive data-prep. On a paid GPU box a
crash-and-resume (e.g. during the --unseen-gene-holdout second retrain) would therefore lose the
whole run at the evaluate step.

This patch swaps the raw load for the proper classmethod (one line) and drops the now-unused
`import joblib as _jl`. EOL/BOM-safe, idempotent, count-guarded, single-file.
"""
from __future__ import annotations
import sys
from pathlib import Path

TARGET = Path("scripts/run_phase2_eval.py")

OLD = (
    '        if _ensemble_path.exists():\n'
    '            import joblib as _jl\n'
    '\n'
    '            logger.info("Resuming: loading existing ensemble from %s", _ensemble_path)\n'
    '            ensemble = _jl.load(_ensemble_path)\n'
)
NEW = (
    '        if _ensemble_path.exists():\n'
    '            logger.info("Resuming: loading existing ensemble from %s", _ensemble_path)\n'
    '            # Reconstruct the VariantEnsemble from the format_version=2 orchestrator dict.\n'
    '            # A raw joblib.load() returns the dict (no .evaluate()) and crashed every resume\n'
    "            # after data-prep (AttributeError: 'dict' object has no attribute 'evaluate').\n"
    '            ensemble = VariantEnsemble.load(_ensemble_path)\n'
)
ALREADY = 'ensemble = VariantEnsemble.load(_ensemble_path)'


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found (run from repo root)", file=sys.stderr); return 2
    raw = TARGET.read_bytes()
    crlf = raw.count(b"\r\n"); lf = raw.count(b"\n") - crlf
    eol = "\r\n" if crlf >= lf else "\n"
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")

    if ALREADY in text:
        print("[skip] resume path already uses VariantEnsemble.load")
        return 0
    if text.count(OLD) != 1:
        print(f"ERROR: resume anchor found {text.count(OLD)}x (expected 1); not patching", file=sys.stderr)
        return 3
    text = text.replace(OLD, NEW, 1)
    TARGET.write_bytes(text.replace("\n", eol).encode("utf-8"))
    print(f"[patched] resume path -> VariantEnsemble.load  (eol={'CRLF' if eol!=chr(10) else 'LF'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
