#!/usr/bin/env python3
"""Part 4: add a ReviewStatus-present-and-populated NO-GO gate to
scripts/preflight_run15_baseline.py, so a cohort that would make
--min-review-tier RAISE at run start is caught locally/free, before any VM
spend. Three edits: docstring item 5, the check function, the checks-list
registration. Count-guarded, idempotent, ast-validated, line-ending agnostic."""
from __future__ import annotations
import ast, shutil, sys
from pathlib import Path

MARKER = "def check_review_status"

DOC_OLD = "  4. cohort guard unit-test file present\n"
DOC_NEW = (
    "  4. cohort guard unit-test file present\n"
    "  5. ReviewStatus column present + populated (so --min-review-tier actually\n"
    "     filters and does not RAISE at run start)\n"
)

FUNC_ANCHOR = "def check_guard_once() -> tuple[bool, str]:\n"
FUNC_NEW = (
    "def check_review_status() -> tuple[bool, str]:\n"
    "    p = Path(CLEAN)\n"
    "    if not p.exists():\n"
    "        return False, f\"missing {CLEAN}\"\n"
    "    try:\n"
    "        import pyarrow.parquet as pq\n"
    "\n"
    "        names = pq.read_schema(p).names\n"
    "    except Exception as e:  # noqa: BLE001\n"
    "        return False, f\"could not read parquet schema ({e})\"\n"
    "    if \"ReviewStatus\" not in names:\n"
    "        return False, (\n"
    "            \"ReviewStatus ABSENT -> --min-review-tier would RAISE at run start \"\n"
    "            \"(run scripts/augment_reviewstatus.py to attach it)\"\n"
    "        )\n"
    "    d = pd.read_parquet(p, columns=[\"ReviewStatus\"])\n"
    "    n_nonempty = int((d[\"ReviewStatus\"].fillna(\"\").astype(str).str.len() > 0).sum())\n"
    "    return n_nonempty > 0, f\"present; non-empty={n_nonempty:,}/{len(d):,}\"\n"
    "\n"
    "\n"
    "def check_guard_once() -> tuple[bool, str]:\n"
)

LIST_ANCHOR = '        ("cohort guard present exactly once", *check_guard_once()),\n'
LIST_NEW = (
    '        ("ReviewStatus present + populated (tier filter)", *check_review_status()),\n'
    '        ("cohort guard present exactly once", *check_guard_once()),\n'
)

def main(path_str: str) -> int:
    path = Path(path_str)
    raw = path.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    data = raw.replace("\r\n", "\n")
    if MARKER in data:
        print(f"SKIP: {path} already has check_review_status (idempotent)"); return 0
    for label, old in [("docstring", DOC_OLD), ("func anchor", FUNC_ANCHOR), ("list anchor", LIST_ANCHOR)]:
        if data.count(old) != 1:
            print(f"ABORT: {label} count={data.count(old)} (want 1); no change"); return 2
    out = data.replace(DOC_OLD, DOC_NEW, 1).replace(FUNC_ANCHOR, FUNC_NEW, 1).replace(LIST_ANCHOR, LIST_NEW, 1)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched source invalid: {e}"); return 3
    final = out.replace("\n", nl) if nl == "\r\n" else out
    shutil.copy2(path, path.with_suffix(path.suffix + ".rsgate.bak"))
    path.open("w", encoding="utf-8", newline="").write(final)
    print(f"patched {path}; 3 edits; endings={'CRLF' if nl == chr(13)+chr(10) else 'LF'}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "scripts/preflight_run15_baseline.py"))
