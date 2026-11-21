"""Inspect the explicitly named historical ClinVar index; coverage is evaluated at replay."""
from pathlib import Path
import pyarrow.parquet as pq

CLINVAR_IDX = Path("data/external/dbnsfp/dbnsfp_clinvar_index.parquet")
REQUIRED = {"chrom", "pos", "ref", "alt", "sift_score", "polyphen2_score", "revel_score",
            "cadd_phred", "phylop_score", "gerp_score"}


def main() -> int:
    if not CLINVAR_IDX.is_file():
        print(f"Historical index missing: {CLINVAR_IDX}")
        return 2
    metadata = pq.read_metadata(CLINVAR_IDX)
    missing = REQUIRED - set(metadata.schema.to_arrow_schema().names)
    if missing or metadata.num_rows == 0:
        print(f"Invalid historical index footer: {metadata.num_rows} rows, missing {sorted(missing)}")
        return 1
    print(f"Historical index footer: {metadata.num_rows} rows, {metadata.num_row_groups} row groups.")
    print(f"Explicit replay path: {CLINVAR_IDX}. Verify its digest against the intended run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
