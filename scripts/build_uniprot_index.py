#!/usr/bin/env python
"""
scripts/build_uniprot_index.py
Author: Monzia Moodie

Builds a local UniProt sequence index so the ESM-2 connector never makes a
live REST call at run time (the cause of the Run-15 smoke stall, instance
40187155). ONE bulk streamed download of the reviewed human proteome (~20k
entries, ~30 MB), parsed to:

    data/external/uniprot/uniprot_human_reviewed.parquet
    columns: gene_symbol, uniprot_id, entry_name, sequence   (one canonical row per gene)

Run once, locally, then SCP the parquet up with the other Run-15 inputs:
    python scripts/build_uniprot_index.py
"""
from __future__ import annotations

import gzip
import io
import sys
from pathlib import Path

import pandas as pd
import requests

# One bulk request (NOT per-gene). Reviewed (Swiss-Prot), human, 3 fields.
_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=organism_id:9606+AND+reviewed:true"
    "&fields=id,accession,gene_primary,sequence"
    "&format=tsv&compressed=true"
)
_OUT = Path("data/external/uniprot/uniprot_human_reviewed.parquet")
_TIMEOUT = 600  # the stream can take a minute or two


def parse_uniprot_tsv(text: str) -> pd.DataFrame:
    """Parse a UniProt TSV (id, accession, gene_primary, sequence) into a deduped
    [gene_symbol, uniprot_id, entry_name, sequence] frame. First row per gene wins.
    entry_name is UniProt's 'id' (e.g. 1433G_HUMAN); EVE keys per-protein files on it."""
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str).fillna("")
    cols = {c.lower(): c for c in df.columns}
    acc = cols.get("entry", list(df.columns)[0])
    entry_col = cols.get("entry name")  # UniProt 'id' field -> 'Entry Name' (e.g. 1433G_HUMAN)
    gene_col = next((cols[c] for c in cols if "gene" in c), None)
    seq_col = next((cols[c] for c in cols if "sequence" in c), None)
    if gene_col is None or seq_col is None:
        raise ValueError(f"unexpected UniProt columns: {list(df.columns)}")

    rows, seen = [], set()
    for _, r in df.iterrows():
        raw_gene = str(r[gene_col]).strip()
        g = raw_gene.split()[0].upper() if raw_gene else ""
        s = str(r[seq_col]).strip()
        if not g or not s or g in seen:
            continue
        seen.add(g)
        en = str(r[entry_col]).strip().upper() if entry_col else ""
        rows.append((g, str(r[acc]).strip(), en, s))
    return pd.DataFrame(rows, columns=["gene_symbol", "uniprot_id", "entry_name", "sequence"])


def main() -> int:
    print(f"Downloading reviewed human proteome from UniProt ...\n  {_URL}")
    resp = requests.get(_URL, timeout=_TIMEOUT, stream=True)
    resp.raise_for_status()
    raw = resp.content
    # compressed=true -> gzip; decompress (fall back to plain if already text)
    try:
        text = gzip.decompress(raw).decode("utf-8")
    except (OSError, gzip.BadGzipFile):
        text = raw.decode("utf-8")

    df = parse_uniprot_tsv(text)
    if df.empty:
        print("ERROR: parsed 0 sequences -- check the UniProt response/columns.", file=sys.stderr)
        return 1

    # sanity: a few well-known disease genes should be present
    for g in ("BRCA1", "TP53", "MLH1"):
        if g not in set(df["gene_symbol"]):
            print(f"WARNING: expected gene {g} not found in index.", file=sys.stderr)

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUT, index=False)
    print(
        f"Wrote {len(df):,} genes -> {_OUT}  "
        f"(median seq len {int(df['sequence'].str.len().median())} aa)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
