"""audit_npathogenic_leakage.py -- quantify the n_pathogenic_in_gene leakage.

enrich_gene_counts() computes n_pathogenic_in_gene CORPUS-WIDE (full labeled df,
pre-split). Under a gene-disjoint split (groups=gene_symbol) a test gene has NO train
rows, so its count comes entirely from test labels -> direct label leakage. This probe
measures, on the actual stored splits:
  1. gene-disjointness (shared genes between train and test; should be ~0),
  2. how many TEST rows are "leaked" (corpus count > 0 but train-only count == 0),
  3. the leakage magnitude: test-set AUROC of the corpus-wide count used AS A SCORE
     vs the train-only count used as a score (a large gap = leakage), and
  4. a sanity check that the stored n_pathogenic_in_gene equals the recomputed
     corpus-wide count (confirms the stored feature is the leaky one).

Reads meta_{train,test}[,val].parquet (gene_symbol + label). No training, no writes.
Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


def _load(meta_dir: Path, name: str, gene_col: str, label_col: str):
    p = meta_dir / f"meta_{name}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    missing = [c for c in (gene_col, label_col) if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{p} missing {missing}; present columns: {list(df.columns)}. "
            f"Re-run with --gene-col/--label-col pointing at the right names."
        )
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Quantify n_pathogenic_in_gene corpus-wide leakage.")
    ap.add_argument("--splits-dir", type=Path, required=True,
                    help="dir with meta_train.parquet / meta_test.parquet [/ meta_val.parquet]")
    ap.add_argument("--gene-col", default="gene_symbol")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--feature-col", default="n_pathogenic_in_gene",
                    help="stored feature column to sanity-check against the recomputed corpus count")
    args = ap.parse_args()

    tr = _load(args.splits_dir, "train", args.gene_col, args.label_col)
    te = _load(args.splits_dir, "test", args.gene_col, args.label_col)
    va = _load(args.splits_dir, "val", args.gene_col, args.label_col)
    if tr is None or te is None:
        raise SystemExit(f"need at least meta_train + meta_test in {args.splits_dir}")
    parts = [d for d in (tr, va, te) if d is not None]
    full = pd.concat(parts, ignore_index=True)

    g, lab = args.gene_col, args.label_col
    # (1) gene-disjointness
    shared = set(tr[g]) & set(te[g])
    print(f"[1] gene-disjointness: train genes={tr[g].nunique()} test genes={te[g].nunique()} "
          f"shared={len(shared)}  ({'GENE-DISJOINT' if not shared else 'OVERLAPPING'})")

    # corpus-wide vs train-only pathogenic counts
    corpus = full[full[lab] == 1].groupby(g).size()
    train_only = tr[tr[lab] == 1].groupby(g).size()
    te = te.copy()
    te["_corpus"] = te[g].map(corpus).fillna(0).astype(int)
    te["_train"] = te[g].map(train_only).fillna(0).astype(int)

    # (2) leaked test rows: corpus count > 0 while train-only count == 0
    leaked = ((te["_train"] == 0) & (te["_corpus"] > 0)).sum()
    print(f"[2] leaked TEST rows (corpus>0 & train-only==0): {leaked}/{len(te)} "
          f"({100*leaked/max(len(te),1):.1f}%)  <- feature derived purely from non-train labels")

    # (3) leakage magnitude: test AUROC of corpus vs train-only count AS A SCORE
    y = te[lab].values
    try:
        auc_corpus = roc_auc_score(y, te["_corpus"].values)
        auc_train = roc_auc_score(y, te["_train"].values)
        print(f"[3] test AUROC of the COUNT used as a lone score: "
              f"corpus-wide={auc_corpus:.4f}  train-only={auc_train:.4f}  "
              f"gap={auc_corpus-auc_train:+.4f}  <- gap is the leakage")
    except ValueError as e:
        print(f"[3] AUROC skipped: {e}")

    # (4) sanity: stored feature == recomputed corpus count?
    if args.feature_col in te.columns:
        match = (te[args.feature_col].fillna(0).astype(int) == te["_corpus"]).mean()
        print(f"[4] stored '{args.feature_col}' == recomputed corpus count on test rows: "
              f"{100*match:.1f}%  <- ~100% confirms the stored feature is corpus-wide (leaky)")
    else:
        print(f"[4] '{args.feature_col}' not in meta_test; skip stored-vs-recomputed check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
