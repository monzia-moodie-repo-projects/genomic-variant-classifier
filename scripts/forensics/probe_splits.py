#!/usr/bin/env python
"""probe_splits.py (2026-07-10)

Read-only inspector for the existing gene-disjoint split machinery, so split_protocol_v2 can be
built as an EXTENSION of the proven code (real_data_prep._gene_aware_split, splits.py) rather than
a fork. Prints every split-related function signature, the gene-disjoint / hash-holdout logic, the
leakage-safe n_pathogenic_in_gene remap, and the current fraction fields.

ASCII-safe: every printed line that echoes file content is sanitized (scanned repo files contain
non-ASCII characters such as the subset symbol, which crash a Windows cp1252 console). Also forces
stdout to replace un-encodable bytes as a second guard.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

# Belt-and-suspenders: never let a stray non-ASCII byte crash the console.
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(".")
TARGETS = [
    r"src/genomic_variant_classifier/data/splits.py",
    r"src/genomic_variant_classifier/data/real_data_prep.py",
    r"src/genomic_variant_classifier/data/pipeline.py",
]
PATTERNS = re.compile(
    r"def .*split|def .*holdout|def _gene_aware|gene_disjoint|unseen_gene|gene_stratified|"
    r"train_test_split|StratifiedKFold|GroupKFold|val_fraction|test_fraction|conformal_fraction|"
    r"holdout_frac|n_pathogenic_in_gene|isdisjoint|assert .*gene|DataPrepConfig",
    re.IGNORECASE)


def _ascii_safe(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def line(c="-", n=78):
    print(c * n)


def main() -> int:
    print("=" * 78)
    print("SPLIT MACHINERY PROBE (inform split_protocol_v2 -- extend, do not fork)")
    print("=" * 78)
    any_found = False
    for rel in TARGETS:
        p = ROOT / rel
        if not p.exists():
            print(_ascii_safe(f"=== ABSENT: {rel} ==="))
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(_ascii_safe(f"=== READ ERROR {rel}: {type(e).__name__}: {e} ==="))
            continue
        lines = txt.splitlines()
        print(_ascii_safe(f"=== {rel} : {len(lines)} lines ==="))
        hits = [(i + 1, ln) for i, ln in enumerate(lines) if PATTERNS.search(ln)]
        if not hits:
            print("  (no split-related lines matched)")
        for i, ln in hits:
            any_found = True
            print(_ascii_safe(f"  L{i}: {ln.strip()[:150]}"))
        line()
    print("PROBE COMPLETE." if any_found else "PROBE COMPLETE (no matches).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
