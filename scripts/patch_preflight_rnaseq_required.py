#!/usr/bin/env python3
"""
patch_preflight_rnaseq_required.py  --  Monzia Moodie

Run 17's RNA-seq features were just made real, but scripts/preflight_gate.py (the single source
of truth that BOTH validates a launch command AND drives preflight_run17 --emit-kg) does not list
--rnaseq-path. Consequence: --emit-kg omits it, and the gate never existence-checks it -> the
rnaseq_* features silently fire ZERO in a real Run-17 launch (the exact failure we forbid).

This patch adds --rnaseq-path to preflight_gate.py in TWO places (the only two needed because the
command is derived from REQUIRED_PATHS):
  1. REQUIRED_PATHS  -> emit includes it + validate() existence-checks it (FAIL if missing).
  2. _build_mirror_parser flag tuple -> the command parser recognises --rnaseq-path when supplied
     (else flagval() returns None and validate() would FAIL even on a correct command).

EOL-agnostic (detects dominant newline, edits in LF space, re-emits original EOL). Idempotent
(no-op if already patched). Count-guarded (each edit must apply exactly once). Read-modify-write
on bytes (BOM/encoding-robust). Does NOT touch any other file.
"""
from __future__ import annotations
import sys
from pathlib import Path

TARGET = Path("scripts/preflight_gate.py")

REQ_ANCHOR = '    "--reactome-path":     "external/reactome_gene_pathways.parquet",'
REQ_INSERT = '    "--rnaseq-path":       "external/rnaseq_gene_expression.parquet",'

PARSER_ANCHOR = '              "--auroc-target", "--output", "--gtex-path", "--reactome-path"):'
PARSER_REPLACE = ('              "--auroc-target", "--output", "--gtex-path", "--reactome-path",\n'
                  '              "--rnaseq-path"):')


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found (run from repo root)", file=sys.stderr); return 2
    raw = TARGET.read_bytes()
    crlf = raw.count(b"\r\n")
    lf_only = raw.count(b"\n") - crlf
    eol = "\r\n" if crlf >= lf_only else "\n"
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")

    already_req = REQ_INSERT in text
    already_parser = '"--rnaseq-path"' in text and PARSER_ANCHOR not in text

    # 1. REQUIRED_PATHS
    if not already_req:
        if text.count(REQ_ANCHOR) != 1:
            print(f"ERROR: REQUIRED_PATHS anchor found {text.count(REQ_ANCHOR)}x (expected 1); aborting",
                  file=sys.stderr); return 3
        text = text.replace(REQ_ANCHOR, REQ_ANCHOR + "\n" + REQ_INSERT, 1)
        print("[patched] REQUIRED_PATHS += --rnaseq-path")
    else:
        print("[skip] REQUIRED_PATHS already has --rnaseq-path")

    # 2. mirror parser tuple
    if PARSER_ANCHOR in text:
        if text.count(PARSER_ANCHOR) != 1:
            print(f"ERROR: parser anchor found {text.count(PARSER_ANCHOR)}x (expected 1); aborting",
                  file=sys.stderr); return 4
        text = text.replace(PARSER_ANCHOR, PARSER_REPLACE, 1)
        print("[patched] _build_mirror_parser += --rnaseq-path")
    elif '"--rnaseq-path"' in text:
        print("[skip] mirror parser already has --rnaseq-path")
    else:
        print("ERROR: parser anchor not found and --rnaseq-path absent; aborting", file=sys.stderr); return 5

    out = text.replace("\n", eol).encode("utf-8")
    TARGET.write_bytes(out)
    print(f"[ok] wrote {TARGET} (eol={'CRLF' if eol == chr(13)+chr(10) else 'LF'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
