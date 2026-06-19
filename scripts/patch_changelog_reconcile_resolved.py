#!/usr/bin/env python3
"""
patch_changelog_reconcile_resolved.py  --  Monzia Moodie

Resolve the open RECONCILE bullet in the 2026-06-19 CHANGELOG entry: the 1KGP parquet re-derivation
was a reproducibility rebuild that became an unintended content-equivalent re-commit (988439c), and the
content-hash guard is now shipped. Exact-string, idempotent, LF-preserving. Run from repo root.
"""
from __future__ import annotations
import sys
from pathlib import Path

TARGET = Path("docs/CHANGELOG.md")
MARKER = "RESOLVED -- reproducibility rebuild"

OLD = (
    "- **RECONCILE**: the 2026-06-15 CHANGELOG entry already documents this exact 437,668 kg build (26342e9). This\n"
    "  session's parquet is content-equivalent (identical variant count + super-pop counts; only parquet-container\n"
    "  bytes differ), so 988439c added a duplicate ~6 MB blob. Confirm whether 06-18/19 was a planned\n"
    "  reproducibility re-derivation (so the data build is not double-counted -- the 06-19 deliverable is the\n"
    "  launch-kit/integration). Consider a content-hash guard before re-committing the parquet, or Git LFS."
)
NEW = (
    "- **RESOLVED -- reproducibility rebuild (not a new data version):** 2026-06-18/19 re-derived the 1KGP\n"
    "  GRCh38 AF parquet during reconciliation/preflight work; output matched the prior 2026-06-15 build\n"
    "  (26342e9) -- 437,668 variants, identical super-population counts. Commit 988439c is therefore\n"
    "  content-equivalent and operationally redundant (6,672,110 -> 6,677,510 bytes, 0 logical change), not a\n"
    "  new dataset. FIX SHIPPED: `scripts/kg_semantic_hash.py` (semantic hash over sorted key + AF columns,\n"
    "  parquet container bytes ignored) + `write_parquet_if_changed` wired into `merge_1kg_parquets.py`\n"
    "  (build logs the hash); the merge step now skips the rewrite when the semantic hash is unchanged,\n"
    "  preventing future equivalent re-commits. Regression: `tests/unit/test_kg_semantic_hash.py` (8 passed)."
)


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found (run from repo root)", file=sys.stderr); return 2
    raw = TARGET.read_bytes()
    text = raw.decode("utf-8-sig")
    if MARKER in text:
        print("[skip] CHANGELOG RECONCILE item already marked RESOLVED"); return 0
    norm = text.replace("\r\n", "\n").replace("\r", "\n")
    c = norm.count(OLD)
    if c != 1:
        print(f"ERROR: RECONCILE bullet found {c}x (expected 1); not patching", file=sys.stderr); return 3
    norm = norm.replace(OLD, NEW)
    TARGET.write_bytes(norm.encode("utf-8"))  # LF, no BOM (matches file)
    print("[patched] CHANGELOG 2026-06-19 RECONCILE -> RESOLVED (+ fix-shipped note)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
