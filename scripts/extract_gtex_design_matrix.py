from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pandas as pd


GTEX_GCT = Path(r"G:\My Drive\genomic-variant-data\external\gtex\GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_tpm.gct.gz")
DESIGN = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\designs\gtex_v11_Brain_Cortex_vs_Whole_Blood.tsv")
OUT_MATRIX = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\gtex_v11_Brain_Cortex_vs_Whole_Blood.gene_tpm.tsv")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    if not GTEX_GCT.exists():
        fail(f"missing GTEx GCT: {GTEX_GCT}")
    if not DESIGN.exists():
        fail(f"missing design file: {DESIGN}")

    design = pd.read_csv(DESIGN, sep="\t", dtype=str)
    samples = design["sample_id"].tolist()

    if len(samples) != len(set(samples)):
        fail("duplicate sample_id values in design")

    with gzip.open(GTEX_GCT, "rt", encoding="utf-8") as handle:
        marker = handle.readline().rstrip("\n")
        dims = handle.readline().rstrip("\n")
        header = handle.readline().rstrip("\n").split("\t")

    if not marker.startswith("#1."):
        fail(f"invalid GCT marker: {marker!r}")

    if header[:2] != ["Name", "Description"]:
        fail(f"unexpected GCT header start: {header[:2]}")

    available = set(header[2:])
    missing = [s for s in samples if s not in available]
    if missing:
        fail(f"{len(missing)} design samples missing from GCT; first={missing[:5]}")

    usecols = ["Name", "Description"] + samples

    print("Reading selected GTEx columns...")
    df = pd.read_csv(
        GTEX_GCT,
        sep="\t",
        compression="gzip",
        skiprows=2,
        usecols=usecols,
        dtype={"Name": str, "Description": str},
    )

    if df.empty:
        fail("extracted matrix is empty")

    if list(df.columns[:2]) != ["Name", "Description"]:
        fail(f"unexpected extracted columns: {df.columns[:2].tolist()}")

    numeric = df[samples].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        fail("non-numeric or NaN TPM values detected")

    if (numeric < 0).any().any():
        fail("negative TPM values detected")

    OUT_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_MATRIX, sep="\t", index=False)

    print(f"OK wrote: {OUT_MATRIX}")
    print(f"shape={df.shape[0]} genes x {len(samples)} samples plus Name/Description")
    print(f"bytes={OUT_MATRIX.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
