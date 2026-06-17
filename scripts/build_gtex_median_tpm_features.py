from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_config(path: Path) -> dict:
    if not path.exists():
        fail(f"missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_float(value: str) -> float:
    try:
        x = float(value)
    except ValueError:
        return math.nan
    return x


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/data_sources.json",
        help="Path to data source config JSON.",
    )
    parser.add_argument(
        "--out",
        default="data/processed/gtex/gtex_v11_gene_median_tpm_features.tsv",
        help="Output TSV path.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    gtex_dir = Path(cfg["external"]["gtex"])

    src = gtex_dir / "GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz"
    if not src.exists():
        fail(f"missing GTEx median TPM file: {src}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    tissue_count = None

    with gzip.open(src, "rt", encoding="utf-8", errors="replace") as handle, out.open(
        "w", encoding="utf-8", newline="\n"
    ) as sink:
        marker = handle.readline().rstrip("\n")
        dims = handle.readline().rstrip("\n")
        header = handle.readline().rstrip("\n").split("\t")

        if not marker.startswith("#1."):
            fail(f"invalid GCT marker: {marker!r}")

        if len(header) < 4:
            fail(f"unexpected GTEx header with {len(header)} columns")

        id_col = header[0]
        desc_col = header[1]
        tissues = header[2:]
        tissue_count = len(tissues)

        if id_col.lower() not in {"name", "id"}:
            fail(f"unexpected first header column: {id_col!r}")
        if desc_col.lower() not in {"description", "desc"}:
            fail(f"unexpected second header column: {desc_col!r}")

        sink.write(
            "gene_id\tgene_symbol\tgtex_tissue_count\tgtex_max_median_tpm\t"
            "gtex_mean_median_tpm\tgtex_nonzero_tissue_count\t"
            "gtex_nonzero_tissue_fraction\tgtex_top_tissue\n"
        )

        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                fail(
                    f"row width mismatch at data row {rows + 1}: "
                    f"expected {len(header)}, got {len(parts)}"
                )

            gene_id = parts[0]
            gene_symbol = parts[1]
            values = [parse_float(x) for x in parts[2:]]

            if any(math.isnan(x) for x in values):
                fail(f"NaN/non-numeric TPM value at gene {gene_id}")

            max_tpm = max(values)
            mean_tpm = sum(values) / len(values)
            nonzero = sum(1 for x in values if x > 0)
            top_idx = values.index(max_tpm)
            top_tissue = tissues[top_idx]

            sink.write(
                f"{gene_id}\t{gene_symbol}\t{len(values)}\t"
                f"{max_tpm:.8g}\t{mean_tpm:.8g}\t{nonzero}\t"
                f"{nonzero / len(values):.8g}\t{top_tissue}\n"
            )

            rows += 1

    if rows <= 0:
        fail("no GTEx rows written")
    if tissue_count is None or tissue_count <= 0:
        fail("no GTEx tissues detected")

    print(f"OK wrote: {out}")
    print(f"rows={rows}")
    print(f"tissues={tissue_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
