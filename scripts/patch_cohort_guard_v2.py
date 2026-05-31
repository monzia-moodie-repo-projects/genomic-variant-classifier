#!/usr/bin/env python3
"""Make _assert_clean_cohort resilient to inputs that lack a `variant_id` column.

Fixes the LOVD-test regression introduced by commit 1720c0a: the guard ran
`df["variant_id"].duplicated()` inside `_load_and_label`, but raw ClinVar / tiny fixtures only
carry chrom/pos/ref/alt at that stage (variant_id is built later), so it raised KeyError. The
duplicate check now prefers `variant_id` when present and otherwise derives the identity from the
chrom:pos:ref:alt locus, skipping only when no identity columns exist. Fail-loud behaviour on a
dirty production cohort is preserved.

Surgical (replaces ONLY the duplicate-identity block), idempotent, fails safe if the anchor is not
uniquely found, and preserves the file's newline style with no BOM (Windows-safe).
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/real_data_prep.py")

OLD = (
    '        if bool(df["variant_id"].duplicated().any()):\n'
    '            raise ValueError(\n'
    '                f"duplicate variant_id in {source}; run scripts/clean_cohort.py --apply."\n'
    '            )\n'
)
NEW = (
    '        if "variant_id" in df.columns:\n'
    '            _key = df["variant_id"]\n'
    '        elif all(c in df.columns for c in ("chrom", "pos", "ref", "alt")):\n'
    '            _key = (\n'
    '                df["chrom"].astype(str) + ":" + df["pos"].astype(str)\n'
    '                + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str)\n'
    '            )\n'
    '        else:\n'
    '            _key = None\n'
    '        if _key is not None and bool(_key.duplicated().any()):\n'
    '            raise ValueError(\n'
    '                f"duplicate variant identity in {source}; run scripts/clean_cohort.py --apply."\n'
    '            )\n'
)


def main() -> int:
    if not TARGET.exists():
        print(f"NOT FOUND: {TARGET}")
        return 2
    raw = TARGET.read_bytes()
    text = raw.decode("utf-8")
    nl = "\r\n" if b"\r\n" in raw else "\n"
    work = text.replace("\r\n", "\n")
    if "duplicate variant identity" in work:
        print("already patched (idempotent no-op)")
        return 0
    n = work.count(OLD)
    if n != 1:
        print(f"ANCHOR not uniquely found (count={n}); NO changes made.")
        return 3
    backup = TARGET.with_name(TARGET.name + ".bak_2026-05-31b")
    backup.write_bytes(raw)
    work = work.replace(OLD, NEW, 1)
    TARGET.write_bytes(work.replace("\n", nl).encode("utf-8"))
    print(f"patched {TARGET}  (backup: {backup.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
