from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd


PARQUET_PATH = Path("data/processed/gtex/gtex_v11_gene_expression.parquet")
EXPECTED_COLUMNS = [
    "gene_symbol",
    "gtex_max_tpm",
    "gtex_n_tissues_expressed",
    "gtex_tissue_specificity",
]


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> int:
    if not PARQUET_PATH.exists():
        fail(f"missing parquet artifact: {PARQUET_PATH}")
    if PARQUET_PATH.stat().st_size <= 0:
        fail(f"zero-byte parquet artifact: {PARQUET_PATH}")

    df = pd.read_parquet(PARQUET_PATH)

    if list(df.columns) != EXPECTED_COLUMNS:
        fail(f"unexpected columns: {list(df.columns)!r}")

    if len(df) != 73321:
        fail(f"unexpected row count: {len(df)}")

    if not df["gene_symbol"].is_unique:
        fail("gene_symbol is not unique")

    if df["gene_symbol"].isna().any():
        fail("gene_symbol contains nulls")

    if (df["gtex_max_tpm"] < 0).any():
        fail("negative gtex_max_tpm detected")

    if not df["gtex_n_tissues_expressed"].between(0, 68).all():
        fail("gtex_n_tissues_expressed outside [0, 68]")

    if not df["gtex_tissue_specificity"].between(0, 1).all():
        fail("gtex_tissue_specificity outside [0, 1]")

    print(f"OK parquet: {PARQUET_PATH}")
    print(f"rows={len(df)}")
    print(f"bytes={PARQUET_PATH.stat().st_size}")
    print(f"sha256={sha256(PARQUET_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
