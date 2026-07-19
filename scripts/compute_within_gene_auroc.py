from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


RUN_DIR = Path("outputs/rnaseq_pred_write_smoke")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def within_gene(path: Path, min_per_class: int = 2) -> pd.DataFrame:
    df = pd.read_parquet(path)

    required = {"gene_symbol", "label", "y_score", "split"}
    missing = required - set(df.columns)
    if missing:
        fail(f"{path}: missing columns {sorted(missing)}")

    rows = []
    for gene, g in df.groupby("gene_symbol", dropna=False):
        labels = g["label"].astype(int)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())

        if n_pos < min_per_class or n_neg < min_per_class:
            continue

        try:
            auc = roc_auc_score(labels, g["y_score"])
        except ValueError:
            continue

        rows.append(
            {
                "gene_symbol": gene,
                "n": len(g),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "within_gene_auroc": float(auc),
            }
        )

    out = pd.DataFrame(rows).sort_values(["n", "gene_symbol"], ascending=[False, True])
    return out


def main() -> int:
    all_rows = []
    for split in ["test", "val"]:
        p = RUN_DIR / f"predictions_{split}.parquet"
        if not p.exists():
            fail(f"missing predictions: {p}")
        w = within_gene(p)
        w.insert(0, "split", split)
        all_rows.append(w)
        print(f"{split}: genes with both classes={len(w)}")
        if len(w):
            weighted = np.average(w["within_gene_auroc"], weights=w["n"])
            unweighted = w["within_gene_auroc"].mean()
            print(f"{split}: weighted within-gene AUROC={weighted:.4f}")
            print(f"{split}: unweighted within-gene AUROC={unweighted:.4f}")

    out = pd.concat(all_rows, ignore_index=True)
    path = RUN_DIR / "within_gene_auroc.parquet"
    out.to_parquet(path, index=False)
    print(f"wrote {path} {out.shape}")
    print(out.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
