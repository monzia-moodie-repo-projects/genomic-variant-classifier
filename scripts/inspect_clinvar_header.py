r"""
Print ClinVar VCF header (##) lines relevant to release date / review status.
Stops at the first data line, so it reads only the header (fast on a multi-GB gz).

Usage (from repo root):
    .venv312\Scripts\python inspect_clinvar_header.py
    .venv312\Scripts\python inspect_clinvar_header.py data\raw\clinvar\clinvar_GRCh38.vcf.gz
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

DEFAULT = Path(r"data\raw\clinvar\clinvar_GRCh38.vcf.gz")
KEYS = ("date", "eval", "review", "clnrev", "source", "version", "reference")


def main() -> int:
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not p.exists():
        print(f"NOT FOUND: {p.resolve()}")
        return 1
    shown = 0
    with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("#"):
                break  # header ends at the first data row
            s = line.rstrip("\n")
            if s.startswith("##") and any(k in s.lower() for k in KEYS):
                print(s)
                shown += 1
    print(f"\n[{shown} matching header lines | file: {p.resolve()}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
