from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {
    "sample_id",
    "condition",
    "batch",
    "study_id",
    "tissue",
    "assay_platform",
    "design_family",
    "contrast_id",
    "source_matrix",
    "leakage_guard",
}

FORBIDDEN_COLUMNS = {
    "pathogenicity",
    "clinical_significance",
    "clinsig",
    "clnsig",
    "label",
    "y",
    "target",
    "variant_id",
    "clinvar_id",
    "hgmd_id",
}


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    if len(sys.argv) != 2:
        fail("Usage: python validate_rnaseq_de_design.py <design.tsv>")

    path = Path(sys.argv[1])
    if not path.exists():
        fail(f"missing design file: {path}")
    if path.stat().st_size <= 0:
        fail(f"zero-byte design file: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        fail(f"missing required columns: {sorted(missing)}")

    forbidden = FORBIDDEN_COLUMNS & {c.lower() for c in df.columns}
    if forbidden:
        fail(f"leakage-risk columns present: {sorted(forbidden)}")

    placeholder_cols = []
    for col in ["sample_id", "condition", "study_id", "contrast_id", "source_matrix"]:
        if df[col].str.contains("REPLACE|TEMPLATE", case=False, regex=True).any():
            placeholder_cols.append(col)

    is_template = bool(placeholder_cols)

    if not is_template:
        n_conditions = df["condition"].nunique(dropna=True)
        if n_conditions < 2:
            fail("design must contain at least two conditions")

        counts = df["condition"].value_counts()
        if (counts < 3).any():
            fail(f"each condition should have at least 3 samples; got {counts.to_dict()}")

        if df["sample_id"].duplicated().any():
            dupes = df.loc[df["sample_id"].duplicated(), "sample_id"].head(10).tolist()
            fail(f"duplicate sample IDs: {dupes}")

    guards = set(df["leakage_guard"].str.lower())
    if not any(
        "clinvar" in g or "variant_label" in g or "not_variant_label" in g
        for g in guards
    ):
        fail("leakage_guard must explicitly state independence from variant/ClinVar labels")

    print("OK RNA-seq DE design")
    print(f"path={path}")
    print(f"rows={len(df)}")
    print(f"template={is_template}")
    print(f"conditions={df['condition'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
