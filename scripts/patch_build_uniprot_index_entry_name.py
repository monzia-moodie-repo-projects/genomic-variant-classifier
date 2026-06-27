#!/usr/bin/env python3
"""patch_build_uniprot_index_entry_name.py

Add UniProt's entry name (the 'id' field, e.g. '1433G_HUMAN') to the local
UniProt index so EVE can resolve its per-protein filenames (entry names) to HGNC
gene symbols. Verified live: UniProt fields=id,accession,gene_primary,sequence ->
headers 'Entry Name','Entry','Gene Names (primary)','Sequence'; 1433G_HUMAN -> YWHAG.

Change set (all anchor-based + idempotent, LF-safe):
  1. request URL:  &fields=accession,gene_primary,sequence
                -> &fields=id,accession,gene_primary,sequence
  2. parser: detect the 'Entry Name' column and emit an entry_name column.
     index columns become: gene_symbol, uniprot_id, entry_name, sequence
  3. docstrings updated to reflect the new column.

Backward-compatible: ESM-2 reads gene_symbol/sequence and ignores the added
column; the new column is purely additive.

  python scripts/patch_build_uniprot_index_entry_name.py            # apply
  python scripts/patch_build_uniprot_index_entry_name.py --check    # report only
"""
from __future__ import annotations

import argparse
from pathlib import Path

TARGET = Path("scripts/build_uniprot_index.py")
MARKER = "entry_name"  # presence of the output column name signals already-patched

REPLACEMENTS = [
    # 1. URL: add the 'id' field (entry name) as the first field.
    (
        '    "&fields=accession,gene_primary,sequence"\n',
        '    "&fields=id,accession,gene_primary,sequence"\n',
    ),
    # 2a. docstring column note (module).
    (
        "    columns: gene_symbol, uniprot_id, sequence   (one canonical row per gene)\n",
        "    columns: gene_symbol, uniprot_id, entry_name, sequence   (one canonical row per gene)\n",
    ),
    # 2b. parser docstring.
    (
        '    """Parse a UniProt TSV (accession, gene_primary, sequence) into a deduped\n'
        '    [gene_symbol, uniprot_id, sequence] frame. First row per gene wins."""\n',
        '    """Parse a UniProt TSV (id, accession, gene_primary, sequence) into a deduped\n'
        '    [gene_symbol, uniprot_id, entry_name, sequence] frame. First row per gene wins.\n'
        '    entry_name is UniProt\'s \'id\' (e.g. 1433G_HUMAN); EVE keys per-protein files on it."""\n',
    ),
    # 2c. detect the entry-name column right after the accession column.
    (
        '    acc = cols.get("entry", list(df.columns)[0])\n'
        '    gene_col = next((cols[c] for c in cols if "gene" in c), None)\n',
        '    acc = cols.get("entry", list(df.columns)[0])\n'
        '    entry_col = cols.get("entry name")  # UniProt \'id\' field -> \'Entry Name\' (e.g. 1433G_HUMAN)\n'
        '    gene_col = next((cols[c] for c in cols if "gene" in c), None)\n',
    ),
    # 2d. emit entry_name in each row + output columns.
    (
        "        seen.add(g)\n"
        "        rows.append((g, str(r[acc]).strip(), s))\n"
        '    return pd.DataFrame(rows, columns=["gene_symbol", "uniprot_id", "sequence"])\n',
        "        seen.add(g)\n"
        '        en = str(r[entry_col]).strip().upper() if entry_col else ""\n'
        "        rows.append((g, str(r[acc]).strip(), en, s))\n"
        '    return pd.DataFrame(rows, columns=["gene_symbol", "uniprot_id", "entry_name", "sequence"])\n',
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()

    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found (run from repo root).")
        return 2

    src = TARGET.read_text(encoding="utf-8")

    if 'columns=["gene_symbol", "uniprot_id", "entry_name", "sequence"]' in src:
        print("OK (idempotent): entry_name already present in build_uniprot_index.py.")
        return 0

    problems = []
    for old, _new in REPLACEMENTS:
        n = src.count(old)
        if n != 1:
            problems.append(f"anchor occurs {n}x (need 1): {old.splitlines()[0][:60]!r}")
    if problems:
        print("FAIL: cannot safely anchor:")
        for p in problems:
            print(f"  - {p}")
        return 3

    patched = src
    for old, new in REPLACEMENTS:
        patched = patched.replace(old, new, 1)

    if ns.check:
        print("CHECK: all 5 anchors found; would add the 'id' field + entry_name column.")
        return 0

    backup = TARGET.with_suffix(TARGET.suffix + ".pre_entry_name.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline="\n")
        print(f"OK: backup -> {backup}")
    TARGET.write_text(patched, encoding="utf-8", newline="\n")
    if b"\r\n" in TARGET.read_bytes():
        print("FAIL: CRLF detected in written file.")
        return 5
    print(f"OK: patched {TARGET}")

    ok = True
    for needle in ['"&fields=id,accession,gene_primary,sequence"',
                   'entry_col = cols.get("entry name")',
                   'columns=["gene_symbol", "uniprot_id", "entry_name", "sequence"]']:
        present = needle in patched
        print(f"  {'OK' if present else 'MISSING'}  {needle[:55]}")
        ok &= present
    try:
        compile(patched, str(TARGET), "exec")
        print("  py-compile OK")
    except SyntaxError as e:
        print(f"  py-compile FAIL: {e}")
        ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
