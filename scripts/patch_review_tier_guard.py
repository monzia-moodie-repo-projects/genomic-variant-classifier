#!/usr/bin/env python3
"""Part 2: fail-loud guard for the review-tier filter in real_data_prep.py.
Inserts, immediately before the `if self.config.exclude_conflicting:` line
(unambiguous ASCII anchor; avoids the Unicode/ASCII log-string ambiguity):
  - `df = df.drop(columns=["review_tier"])` as the last stmt of the ReviewStatus
    branch, so review_tier can never leak into the feature matrix; and
  - an `elif self.config.min_review_tier < 5: raise ValueError(...)` so a missing
    ReviewStatus column with a real tier request fails loud instead of silently
    keeping all review levels.
Count-guarded, idempotent, ast-validated, line-ending agnostic."""
from __future__ import annotations
import ast, shutil, sys
from pathlib import Path

MARKER = "self.config.min_review_tier < 5"
ANCHOR = "        if self.config.exclude_conflicting:\n"
NEW = (
    '            df = df.drop(columns=["review_tier"])\n'
    "        elif self.config.min_review_tier < 5:\n"
    "            raise ValueError(\n"
    '                f"min_review_tier={self.config.min_review_tier} requested but "\n'
    "                \"the cohort has no 'ReviewStatus' column, so the review-tier \"\n"
    '                "filter cannot be applied (it would silently keep all review "\n'
    '                "levels). Re-build the cohort with ReviewStatus "\n'
    '                "(scripts/augment_reviewstatus.py) or set min_review_tier=5 "\n'
    '                "to disable tier filtering explicitly."\n'
    "            )\n"
    "\n"
    "        if self.config.exclude_conflicting:\n"
)

def main(path_str: str) -> int:
    path = Path(path_str)
    raw = path.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    data = raw.replace("\r\n", "\n")
    if MARKER in data:
        print(f"SKIP: {path} already has the fail-loud guard (idempotent)"); return 0
    if data.count(ANCHOR) != 1:
        print(f"ABORT: anchor count={data.count(ANCHOR)} (want 1); no change"); return 2
    out = data.replace(ANCHOR, NEW, 1)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched source invalid: {e}"); return 3
    final = out.replace("\n", nl) if nl == "\r\n" else out
    shutil.copy2(path, path.with_suffix(path.suffix + ".tierguard.bak"))
    path.open("w", encoding="utf-8", newline="").write(final)
    print(f"patched {path}; endings={'CRLF' if nl == chr(13)+chr(10) else 'LF'}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "src/genomic_variant_classifier/data/real_data_prep.py"))
