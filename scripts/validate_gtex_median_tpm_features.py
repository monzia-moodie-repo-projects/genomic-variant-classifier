from __future__ import annotations

import csv
import hashlib
import math
import subprocess
import sys
import tempfile
from pathlib import Path


FEATURE_PATH = Path("data/processed/gtex/gtex_v11_gene_median_tpm_features.tsv")
BUILD_SCRIPT = Path("scripts/build_gtex_median_tpm_features.py")

EXPECTED_ROWS = 74628
EXPECTED_TISSUES = 68
EXPECTED_HEADER = [
    "gene_id",
    "gene_symbol",
    "gtex_tissue_count",
    "gtex_max_median_tpm",
    "gtex_mean_median_tpm",
    "gtex_nonzero_tissue_count",
    "gtex_nonzero_tissue_fraction",
    "gtex_top_tissue",
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


def require_file(path: Path) -> None:
    if not path.exists():
        fail(f"missing file: {path}")
    if not path.is_file():
        fail(f"not a file: {path}")
    if path.stat().st_size <= 0:
        fail(f"zero-byte file: {path}")


def parse_float(value: str, field: str, row_idx: int) -> float:
    try:
        x = float(value)
    except ValueError:
        fail(f"non-numeric {field} at row {row_idx}: {value!r}")
    if math.isnan(x) or math.isinf(x):
        fail(f"invalid {field} at row {row_idx}: {value!r}")
    return x


def parse_int(value: str, field: str, row_idx: int) -> int:
    try:
        return int(value)
    except ValueError:
        fail(f"non-integer {field} at row {row_idx}: {value!r}")


def validate_table(path: Path) -> None:
    require_file(path)

    seen_gene_ids: set[str] = set()
    row_count = 0
    top_tissues: set[str] = set()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")

        if reader.fieldnames != EXPECTED_HEADER:
            fail(f"unexpected header: {reader.fieldnames!r}")

        for row_count, row in enumerate(reader, start=1):
            gene_id = row["gene_id"]
            gene_symbol = row["gene_symbol"]

            if not gene_id:
                fail(f"empty gene_id at row {row_count}")
            if gene_id in seen_gene_ids:
                fail(f"duplicate gene_id at row {row_count}: {gene_id}")
            seen_gene_ids.add(gene_id)

            if not gene_symbol:
                fail(f"empty gene_symbol at row {row_count}: {gene_id}")

            tissue_count = parse_int(row["gtex_tissue_count"], "gtex_tissue_count", row_count)
            max_tpm = parse_float(row["gtex_max_median_tpm"], "gtex_max_median_tpm", row_count)
            mean_tpm = parse_float(row["gtex_mean_median_tpm"], "gtex_mean_median_tpm", row_count)
            nonzero_count = parse_int(
                row["gtex_nonzero_tissue_count"],
                "gtex_nonzero_tissue_count",
                row_count,
            )
            nonzero_fraction = parse_float(
                row["gtex_nonzero_tissue_fraction"],
                "gtex_nonzero_tissue_fraction",
                row_count,
            )
            top_tissue = row["gtex_top_tissue"]

            if tissue_count != EXPECTED_TISSUES:
                fail(f"unexpected tissue count at row {row_count}: {tissue_count}")
            if max_tpm < 0:
                fail(f"negative max TPM at row {row_count}: {max_tpm}")
            if mean_tpm < 0:
                fail(f"negative mean TPM at row {row_count}: {mean_tpm}")
            if mean_tpm > max_tpm and max_tpm > 0:
                fail(f"mean TPM exceeds max TPM at row {row_count}: mean={mean_tpm}, max={max_tpm}")
            if not (0 <= nonzero_count <= EXPECTED_TISSUES):
                fail(f"bad nonzero tissue count at row {row_count}: {nonzero_count}")
            if not (0.0 <= nonzero_fraction <= 1.0):
                fail(f"bad nonzero tissue fraction at row {row_count}: {nonzero_fraction}")
            if abs(nonzero_fraction - (nonzero_count / EXPECTED_TISSUES)) > 1e-6:
                fail(
                    f"nonzero fraction mismatch at row {row_count}: "
                    f"{nonzero_fraction} vs {nonzero_count / EXPECTED_TISSUES}"
                )
            if not top_tissue:
                fail(f"empty top tissue at row {row_count}")

            top_tissues.add(top_tissue)

    if row_count != EXPECTED_ROWS:
        fail(f"unexpected row count: {row_count}, expected {EXPECTED_ROWS}")

    if len(top_tissues) < 10:
        fail(f"suspiciously low top-tissue diversity: {len(top_tissues)}")

    print(f"OK table: {path}")
    print(f"rows={row_count}")
    print(f"unique_gene_ids={len(seen_gene_ids)}")
    print(f"unique_top_tissues={len(top_tissues)}")


def validate_reproducible() -> None:
    require_file(BUILD_SCRIPT)
    require_file(FEATURE_PATH)

    with tempfile.TemporaryDirectory() as tmp:
        rebuilt = Path(tmp) / "rebuilt_gtex_features.tsv"

        cmd = [
            sys.executable,
            str(BUILD_SCRIPT),
            "--out",
            str(rebuilt),
        ]

        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
        )

        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            fail(f"rebuild failed with exit code {result.returncode}")

        require_file(rebuilt)

        original_hash = sha256(FEATURE_PATH)
        rebuilt_hash = sha256(rebuilt)

        if original_hash != rebuilt_hash:
            fail(f"rebuild hash mismatch: original={original_hash}, rebuilt={rebuilt_hash}")

        print(f"OK reproducible rebuild: sha256={original_hash}")


def main() -> int:
    validate_table(FEATURE_PATH)
    validate_reproducible()
    print("\nGTEx derived feature validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
