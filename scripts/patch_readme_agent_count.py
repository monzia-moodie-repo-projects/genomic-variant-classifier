#!/usr/bin/env python3
"""patch_readme_agent_count.py -- reconcile README agent count to the measured 13.

Fixes the badge/intro/ASCII/bullet "seven" vs the table/tree "13" inconsistency and
the stale Python version. Single-line, EOL-agnostic, count-guarded (each edit must
match exactly once or the script aborts without writing). Author: Monzia Moodie.
"""
from __future__ import annotations
import sys
from pathlib import Path

README = Path("README.md")
EDITS = [
    ("Core%20agents-7-blueviolet", "Core%20agents-13-blueviolet"),
    ("agent layer of seven core monitoring agents", "agent layer of thirteen specialised agents"),
    ("7 core agents + drift suite", "13 specialised agents"),
    ("A monitoring layer of seven core agents plus a committed drift-detection suite",
     "A monitoring layer of thirteen specialised agents"),
    ("34/34 passing on Py 3.14.3", "34/34 passing on Python 3.12.10"),
]

def main() -> int:
    if not README.exists():
        print(f"ABORT: not found: {README.resolve()}"); return 1
    txt = README.read_text(encoding="utf-8")
    # pre-flight: every old string present exactly once (idempotent: skip ones already applied)
    plan = []
    for old, new in EDITS:
        c_old, c_new = txt.count(old), txt.count(new)
        if c_old == 1:
            plan.append((old, new))
        elif c_old == 0 and c_new >= 1:
            print(f"  skip (already applied): {old!r}")
        else:
            print(f"ABORT: anchor count != 1 for {old!r} (old={c_old}, new={c_new})"); return 1
    if not plan:
        print("no-op: all edits already applied"); return 0
    for old, new in plan:
        txt = txt.replace(old, new, 1)
    README.write_text(txt, encoding="utf-8")
    print(f"applied {len(plan)} edit(s): " + "; ".join(o.split()[0] + '...' for o, _ in plan))
    return 0

if __name__ == "__main__":
    sys.exit(main())
