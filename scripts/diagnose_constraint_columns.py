"""
scripts/diagnose_constraint_columns.py
======================================
Locate WHERE feature columns are ASSIGNED (given a real value) versus merely
DECLARED (listed in a feature list / zero-filled), to explain silent-zero /
all-null columns. Built for the gnomAD-constraint anomaly (gene_constraint_oe
+ gene_is_constrained ALL_ZERO while loeuf + pli_score are healthy), but works
for any column set via --columns.

Read-only: scans source + config text, assigns no blame it can't show you the
line for. A column that has DECLARED/FILLNA hits but ZERO assignment hits is a
zero-filled vestige -- the most common silent-zero cause.

USAGE
-----
  python scripts/diagnose_constraint_columns.py
  python scripts/diagnose_constraint_columns.py --columns esm2_delta_norm gnn_score

EXIT: 0 always (diagnostic), 1 only if src dir missing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_DEFAULT_COLS = ["gene_constraint_oe", "gene_is_constrained", "loeuf", "pli_score"]
# gnomAD-constraint raw source keys worth surfacing alongside the columns
_SOURCE_HINTS = ["oe_lof", "oe_mis", "oe_lof_upper", "lof_oe", "lof.oe",
                 "pLI", "pli", "constraint", "constrained", "syn_z", "mis_z"]


def _classify(line: str, col: str) -> str:
    """ASSIGN | DECLARE | REF for an occurrence of `col` in `line`."""
    s = line.strip()
    cq = re.escape(col)
    # assignment forms: col = ... | df["col"] = ... | out['col'] = ... | "col": ...
    if re.search(rf'(^|[^\w])({cq})\s*=(?!=)', s):
        return "ASSIGN"
    if re.search(rf'\[\s*[\'"]{cq}[\'"]\s*\]\s*=(?!=)', s):
        return "ASSIGN"
    if re.search(rf'[\'"]{cq}[\'"]\s*:', s):  # dict / yaml mapping target
        return "ASSIGN"
    # declaration / fill forms: inside a quoted list, or a fillna/default line
    if re.search(rf'[\'"]{cq}[\'"]', s) and ("fillna" in s or "default" in s
                                              or "FEATURES" in s or "[" in s):
        return "DECLARE"
    return "REF"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="trace column assignment vs declaration")
    ap.add_argument("--src", default="src/genomic_variant_classifier")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--columns", nargs="+", default=_DEFAULT_COLS)
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        print(f"src dir not found: {src.resolve()} -- STOP.")
        return 1

    files = sorted(src.rglob("*.py"))
    cfg = Path(args.config)
    if cfg.exists():
        files.append(cfg)

    cols = list(args.columns)
    hits: dict[str, list[tuple[str, str, int, str]]] = {c: [] for c in cols}
    src_hint_hits: list[tuple[str, int, str]] = []

    for f in files:
        try:
            text = f.read_text(encoding="utf-8-sig", errors="replace")
        except Exception as e:
            print(f"  !! cannot read {f}: {e}")
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for c in cols:
                if re.search(rf'(^|[^\w]){re.escape(c)}([^\w]|$)', line):
                    hits[c].append((_classify(line, c), str(f), i, line.strip()[:140]))
            for h in _SOURCE_HINTS:
                if re.search(rf'(^|[^\w]){re.escape(h)}([^\w]|$)', line):
                    src_hint_hits.append((str(f), i, line.strip()[:140]))

    print(f"scanned {len(files)} files under {src.resolve()}"
          + (f" (+{cfg.name})" if cfg.exists() else "") + "\n")

    for c in cols:
        rows = hits[c]
        kinds = {k: sum(1 for r in rows if r[0] == k) for k in ("ASSIGN", "DECLARE", "REF")}
        verdict = ("NO ASSIGNMENT -> zero/null-filled vestige (silent-zero cause)"
                   if kinds["ASSIGN"] == 0 and rows else
                   "assigned" if kinds["ASSIGN"] > 0 else
                   "ABSENT from source (not referenced anywhere)")
        print(f"=== {c}  [ASSIGN={kinds['ASSIGN']} DECLARE={kinds['DECLARE']} "
              f"REF={kinds['REF']}]  -> {verdict}")
        for kind, fp, ln, txt in rows:
            rel = Path(fp).name
            print(f"    {kind:7s} {rel}:{ln}: {txt}")
        if not rows:
            print("    (no occurrences)")
        print()

    if src_hint_hits:
        print(f"=== gnomAD-constraint source-key references "
              f"({len(src_hint_hits)}) -- where raw constraint fields are read ===")
        seen = set()
        for fp, ln, txt in src_hint_hits:
            key = (Path(fp).name, ln)
            if key in seen:
                continue
            seen.add(key)
            print(f"    {Path(fp).name}:{ln}: {txt}")

    print("\nReading: a healthy column (loeuf/pli_score) should show ASSIGN hits in a "
          "connector. A dead column with only DECLARE hits and ASSIGN=0 is populated "
          "by nothing -> the engineer_features fillna(0)/default makes it constant-zero. "
          "Cross-check the source-key block: if the raw o/e field IS read but lands in "
          "loeuf instead of gene_constraint_oe, that's a rename the schema didn't follow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
