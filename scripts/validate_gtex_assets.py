from __future__ import annotations

import gzip
import sys
from pathlib import Path

GTEX_DIR = Path(r"G:\My Drive\genomic-variant-data\external\gtex")

FILES = [
    "GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz",
    "GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_tpm.gct.gz",
]

def fail(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(2)

def validate_gct(path: Path) -> None:
    if not path.exists():
        fail(f"missing: {path}")
    if path.stat().st_size <= 0:
        fail(f"zero-byte: {path}")

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        first = handle.readline().rstrip("\n")
        second = handle.readline().rstrip("\n")
        third = handle.readline().rstrip("\n")
        fourth = handle.readline().rstrip("\n")

    if not first.startswith("#1."):
        fail(f"{path.name}: invalid GCT marker: {first!r}")

    dims = second.split("\t")
    if len(dims) < 2 or not dims[0].isdigit() or not dims[1].isdigit():
        fail(f"{path.name}: invalid dimensions line: {second!r}")

    header = third.split("\t")
    if len(header) < 3:
        fail(f"{path.name}: header too short")

    if header[0].lower() not in {"name", "id"}:
        fail(f"{path.name}: unexpected first header column: {header[0]!r}")

    if not fourth:
        fail(f"{path.name}: no first data row")

    print(f"OK: {path.name}; rows={dims[0]}; cols={dims[1]}; header_cols={len(header)}")

def main() -> int:
    for name in FILES:
        validate_gct(GTEX_DIR / name)
    print("\nGTEx GCT validation passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
