from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


SRC = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\gtex_v11_Brain_Cortex_vs_Whole_Blood.rnaseq_gene_expression.parquet")
OUTDIR = Path(r"G:\My Drive\genomic-variant-data\external\rnaseq\ablations")


def write(df: pd.DataFrame, name: str) -> None:
    out = OUTDIR / name
    df.to_parquet(out, index=False)
    print("wrote", out, df.shape)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SRC)

    required = {
        "gene_symbol",
        "rnaseq_mean_log_tpm",
        "rnaseq_detection_rate",
        "rnaseq_log2_cv",
        "rnaseq_log2fc",
        "rnaseq_de_neglog10p",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")

    if not df["gene_symbol"].is_unique:
        raise SystemExit("gene_symbol is not unique")

    drop_de = df.copy()
    drop_de["rnaseq_log2fc"] = 0.0
    drop_de["rnaseq_de_neglog10p"] = 0.0
    write(drop_de, "rnaseq_drop_de.parquet")

    drop_all = df.copy()
    for col in required - {"gene_symbol"}:
        drop_all[col] = 0.0
    write(drop_all, "rnaseq_drop_all.parquet")

    rng = np.random.default_rng(20260618)
    shuffle = df.copy()
    cols = [
        "rnaseq_mean_log_tpm",
        "rnaseq_detection_rate",
        "rnaseq_log2_cv",
        "rnaseq_log2fc",
        "rnaseq_de_neglog10p",
    ]
    perm = rng.permutation(len(shuffle))
    shuffle[cols] = shuffle.loc[perm, cols].to_numpy()
    write(shuffle, "rnaseq_gene_shuffle.parquet")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
