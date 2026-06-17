from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

DATA_ROOT = Path(r"G:\My Drive\genomic-variant-data")
MANIFESTS = [
    DATA_ROOT / "manifests" / "source_assets_manifest.json",
    DATA_ROOT / "manifests" / "gtex_v11_manifest.json",
]

GTEX_FILES = [
    DATA_ROOT / "external/gtex/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz",
    DATA_ROOT / "external/gtex/GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_tpm.gct.gz",
]

GENCODE_FILES = [
    DATA_ROOT / "external/gencode/gencode.v50.annotation.gtf.gz",
    DATA_ROOT / "external/gencode/gencode.v50.annotation.gff3.gz",
    DATA_ROOT / "external/gencode/gencode.v50.transcripts.fa.gz",
    DATA_ROOT / "external/gencode/gencode.v50.pc_transcripts.fa.gz",
    DATA_ROOT / "external/gencode/gencode.v50.lncRNA_transcripts.fa.gz",
]

HGNC_FILE = DATA_ROOT / "external/hgnc/hgnc_complete_set.txt"


def fail(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def require_file(path: Path) -> None:
    if not path.exists():
        fail(f"missing file: {path}")
    if not path.is_file():
        fail(f"not a file: {path}")
    if path.stat().st_size <= 0:
        fail(f"zero-byte file: {path}")


def validate_gzip(path: Path) -> None:
    require_file(path)
    total = 0
    with gzip.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            total += len(chunk)
    if total <= 0:
        fail(f"gzip readable but empty: {path}")
    print(f"OK gzip: {path.name}; uncompressed_bytes={total}")


def validate_gct(path: Path) -> None:
    require_file(path)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        marker = f.readline().rstrip("\n")
        dims = f.readline().rstrip("\n")
        header = f.readline().rstrip("\n")
        first_row = f.readline().rstrip("\n")

    if not marker.startswith("#1."):
        fail(f"invalid GCT marker in {path.name}: {marker!r}")

    parts = dims.split("\t")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        fail(f"invalid GCT dimensions in {path.name}: {dims!r}")

    if not first_row:
        fail(f"missing first data row in {path.name}")

    print(f"OK GCT: {path.name}; rows={parts[0]}; cols={parts[1]}; header_cols={len(header.split(chr(9)))}")


def validate_manifest(path: Path) -> None:
    require_file(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        fail(f"manifest is not a list: {path}")
    if not data:
        fail(f"manifest is empty: {path}")

    for record in data:
        p = Path(record["path"]) if "path" in record else DATA_ROOT / "external/gtex" / record["name"]
        require_file(p)

        expected_bytes = int(record["bytes"])
        actual_bytes = p.stat().st_size
        if actual_bytes != expected_bytes:
            fail(f"byte mismatch for {p}: manifest={expected_bytes}, actual={actual_bytes}")

        expected_hash = record["sha256"].upper()
        actual_hash = sha256(p)
        if actual_hash != expected_hash:
            fail(f"sha256 mismatch for {p}")

    print(f"OK manifest: {path.name}; records={len(data)}")


def validate_hgnc(path: Path) -> None:
    require_file(path)
    with path.open("rt", encoding="utf-8", errors="replace") as f:
        header = f.readline()
    if "hgnc_id" not in header.lower() or "symbol" not in header.lower():
        fail(f"HGNC header missing expected fields: {header[:300]!r}")
    print(f"OK HGNC: {path.name}")


def main() -> int:
    for m in MANIFESTS:
        validate_manifest(m)

    for f in GENCODE_FILES:
        validate_gzip(f)

    for f in GTEX_FILES:
        validate_gct(f)

    validate_hgnc(HGNC_FILE)

    print("\nAll source data asset checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
