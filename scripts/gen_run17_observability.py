#!/usr/bin/env python3
"""Generate run17_observability.py from run15_observability.py.

The observability script is config-agnostic (--outputs-dir/--log/--report-dir are all
required CLI args; nothing about a run is hardcoded except run-id strings in comments,
the report title, and the two output filenames). So adaptation = pure run-id repoint,
zero logic change. Each substitution is anchored + counted; the result must differ from
the source ONLY in the run15->run17 token sites (verified by the caller's diff).

Output filenames become run17_observability.{json,md} -- the Run17 postflight gates on
exactly these names (must match).

Usage: python gen_run17_observability.py <run15_observability.py> <out_path>
"""
from __future__ import annotations
import sys
from pathlib import Path

MARKER = "run17_observability"

# (anchor, replacement, expected_count). Each must occur exactly its count in the source.
SUBS = [
    ("# run15_observability.py", "# run17_observability.py", 1),
    ("Maximum-information extractor for Run 15.", "Maximum-information extractor for Run 17.", 1),
    ("# Target:   genomic-variant-classifier, Run 15 (commit set at launch)",
     "# Target:   genomic-variant-classifier, Run 17 (commit set at launch)", 1),
    ("scripts/run15_observability.py", "scripts/run17_observability.py", 1),
    ("--report-dir /workspace/run15_report", "--report-dir /workspace/run17_report", 1),
    ('f"# Run 15 Observability Report"', 'f"# Run 17 Observability Report"', 1),
    ('report_dir / "run15_observability.json"', 'report_dir / "run17_observability.json"', 1),
    ('report_dir / "run15_observability.md"', 'report_dir / "run17_observability.md"', 1),
]

# These run15 mentions are in the run14-history usage example (--log /workspace/run11_master.log,
# outputs/run9_fresh) -- those are illustrative examples, NOT run-id-coupled; we leave them as-is
# EXCEPT the report-dir which IS run-coupled (handled above). The Created date is left as the run15
# authored date intentionally? No -- update it too for accuracy.
DATE_SUB = ("# Created:  2026-05-27", "# Created:  2026-06-28", 1)  # run17 authored date


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python gen_run17_observability.py <run15_observability.py> <out_path>")
        return 2
    src_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    if not src_path.exists():
        print(f"ERROR: {src_path} not found")
        return 2

    raw = src_path.read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        print("ERROR: source has a UTF-8 BOM; expected BOM-free.")
        return 2
    text = raw.decode("utf-8")

    # Validate every anchor occurs exactly its count BEFORE applying
    problems = []
    for anchor, _r, cnt in SUBS:
        n = text.count(anchor)
        if n != cnt:
            problems.append(f"  anchor x{n} (expected {cnt}): {anchor[:60]}")
    # date sub is optional-but-expected-once; warn not fail if absent
    date_n = text.count(DATE_SUB[0])
    if problems:
        print("ANCHOR VALIDATION FAILED -- nothing written:")
        print("\n".join(problems))
        return 1

    for anchor, repl, _cnt in SUBS:
        text = text.replace(anchor, repl, 1)
    if date_n == 1:
        text = text.replace(DATE_SUB[0], DATE_SUB[1], 1)
    else:
        print(f"  NOTE: Created-date anchor occurs {date_n}x; left unchanged (non-fatal).")

    out_path.write_text(text, encoding="utf-8", newline="")
    print(f"GENERATED: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
