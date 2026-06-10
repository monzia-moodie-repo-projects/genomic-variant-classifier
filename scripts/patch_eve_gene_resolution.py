#!/usr/bin/env python3
"""patch_eve_gene_resolution.py -- Phase 0 (part 2 of 2): eve.py.

Wires the shared gene-symbol helper into EVEConnector._annotate so the
gene_symbol merge key is (a) case-normalized on BOTH sides (fixes the latent
case-drift bug), (b) resolves semicolon-joined multi-gene symbols to the first
component present in the EVE lookup, and (c) drops unusable empty-gene lookup
rows so they cannot spuriously match an empty variant gene_symbol. Never splits
on '-'. No row explosion (merge stays 1:1).

Count-guarded, backup-first, idempotent, py_compile-gated. Author: Monzia Moodie.
Requires src/genomic_variant_classifier/data/gene_symbols.py in place.
"""
from __future__ import annotations

import datetime as _dt
import py_compile
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVE = REPO / "src/genomic_variant_classifier/data/eve.py"

_IMPORT_OLD = (
    'from genomic_variant_classifier.data.database_connectors import BaseConnector, FetchConfig\n'
    '\n'
    'logger = logging.getLogger(__name__)'
)
_IMPORT_NEW = (
    'from genomic_variant_classifier.data.database_connectors import BaseConnector, FetchConfig\n'
    'from genomic_variant_classifier.data.gene_symbols import (\n'
    '    gene_symbol_candidates,\n'
    '    normalize_gene_symbol,\n'
    ')\n'
    '\n'
    'logger = logging.getLogger(__name__)'
)

_ANNOTATE_OLD = (
    '        gene_symbol = result.get(\n'
    '            "gene_symbol",\n'
    '            pd.Series([""] * len(result), index=result.index),\n'
    '        ).fillna("")\n'
    '        result["_gene_symbol"] = gene_symbol\n'
    '\n'
    '        # Only attempt to join for rows with a valid aa_change\n'
    '        has_key = result["_aa_change"].notna()\n'
    '\n'
    '        score_table = lookup.rename(\n'
    '            columns={"gene_symbol": "_gene_symbol", "aa_change": "_aa_change"}\n'
    '        )\n'
    '\n'
    '        result = result.merge(\n'
    '            score_table,\n'
    '            on=["_gene_symbol", "_aa_change"],\n'
    '            how="left",\n'
    '        )'
)
_ANNOTATE_NEW = (
    '        gene_symbol = result.get(\n'
    '            "gene_symbol",\n'
    '            pd.Series([""] * len(result), index=result.index),\n'
    '        ).fillna("")\n'
    '\n'
    '        # Only attempt to join for rows with a valid aa_change\n'
    '        has_key = result["_aa_change"].notna()\n'
    '\n'
    '        score_table = lookup.rename(\n'
    '            columns={"gene_symbol": "_gene_symbol", "aa_change": "_aa_change"}\n'
    '        )\n'
    '        # Normalize the lookup gene key so case/whitespace never blocks a\n'
    '        # match, and drop unusable empty-gene rows so they cannot spuriously\n'
    '        # match an empty variant gene_symbol.\n'
    '        score_table["_gene_symbol"] = score_table["_gene_symbol"].map(\n'
    '            normalize_gene_symbol\n'
    '        )\n'
    '        score_table = score_table[score_table["_gene_symbol"] != ""]\n'
    '        _lookup_genes = set(score_table["_gene_symbol"])\n'
    '\n'
    '        def _resolve_gene(_raw: object) -> str:\n'
    '            # First candidate present in the EVE lookup wins: recovers\n'
    '            # semicolon-joined multi-gene symbols and fixes case drift.\n'
    '            # Never splits on "-".\n'
    '            for _cand in gene_symbol_candidates(_raw):\n'
    '                if _cand in _lookup_genes:\n'
    '                    return _cand\n'
    '            return normalize_gene_symbol(_raw)\n'
    '\n'
    '        result["_gene_symbol"] = gene_symbol.map(_resolve_gene)\n'
    '\n'
    '        result = result.merge(\n'
    '            score_table,\n'
    '            on=["_gene_symbol", "_aa_change"],\n'
    '            how="left",\n'
    '        )'
)

EDITS = [
    (_IMPORT_OLD, _IMPORT_NEW,
     'from genomic_variant_classifier.data.gene_symbols import (',
     'eve: import shared gene-symbol helper'),
    (_ANNOTATE_OLD, _ANNOTATE_NEW,
     'score_table = score_table[score_table["_gene_symbol"] != ""]',
     'eve: _annotate gene-key normalization + candidate resolution'),
]


def main() -> int:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not EVE.exists():
        print(f"ABORT: missing {EVE}")
        return 2
    text = EVE.read_text(encoding="utf-8")
    shutil.copy2(EVE, f"{EVE}.bak_{ts}")
    for old, new, marker, label in EDITS:
        if marker in text:
            print(f"  skip (already applied): {label}")
            continue
        n = text.count(old)
        if n != 1:
            print(f"ABORT: anchor for '{label}' found {n}x (expected 1); no changes written")
            return 3
        text = text.replace(old, new, 1)
        print(f"  ok: {label}")
    EVE.write_text(text, encoding="utf-8")
    try:
        py_compile.compile(str(EVE), doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"ABORT: py_compile failed: {exc}")
        return 4
    print(f"py_compile clean: eve.py  (backup -> eve.py.bak_{ts})")
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
