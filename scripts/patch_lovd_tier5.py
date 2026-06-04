#!/usr/bin/env python3
"""Set min_review_tier=0 -> 5 in the two LOVD pipeline.run() tests so the new
fail-loud guard does not raise on their no-ReviewStatus fixtures (5 = tier filter
disabled; same all-rows-kept behavior they already relied on). Count-guarded,
idempotent, ast-validated, line-ending agnostic."""
from __future__ import annotations
import ast, shutil, sys
from pathlib import Path

EDITS = [
    ("        min_review_tier=0,  # accept the fixture's review_status\n",
     "        min_review_tier=5,  # 5 = tier filter disabled; fixture's lowercase 'review_status' never matched 'ReviewStatus', and the fail-loud guard now rejects tier<5 without it\n",
     1),
    ('        min_review_tier=0,\n        output_dir=tmp_path / "splits_no_lovd",\n',
     '        min_review_tier=5,  # tier filter disabled (no ReviewStatus col); avoids fail-loud guard\n        output_dir=tmp_path / "splits_no_lovd",\n',
     1),
]
MARKER = "tier filter disabled"

def main(path_str: str) -> int:
    path = Path(path_str)
    raw = path.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    data = raw.replace("\r\n", "\n")
    if MARKER in data:
        print(f"SKIP: {path} already updated (idempotent)"); return 0
    for old, _new, n in EDITS:
        if data.count(old) != n:
            print(f"ABORT: anchor count={data.count(old)} (want {n}); no change. Head: {old.splitlines()[0]!r}"); return 2
    out = data
    for old, new, _n in EDITS:
        out = out.replace(old, new, 1)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: invalid: {e}"); return 3
    final = out.replace("\n", nl) if nl == "\r\n" else out
    shutil.copy2(path, path.with_suffix(path.suffix + ".tier5.bak"))
    path.open("w", encoding="utf-8", newline="").write(final)
    print(f"patched {path}; {len(EDITS)} edits; endings={'CRLF' if nl == chr(13)+chr(10) else 'LF'}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "tests/unit/test_lovd_annotation_reaches_training_matrix.py"))
