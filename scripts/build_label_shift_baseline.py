#!/usr/bin/env python3
"""build_label_shift_baseline.py  --  Monzia Moodie

Capture the label-shift reference baseline (classes + p_train + reference_confusion) for
LabelShiftMonitorAgent. p_train is the training-set class distribution; reference_confusion
is the model's column-stochastic C[pred, true] = P(pred|true) on a labeled validation set,
so it REQUIRES a trained model's validation predictions (a Run-17 artifact). Mirrors
build_schema_baseline.py. Output: data/reference/label_shift/label_shift_baseline.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT = Path("data/reference/label_shift/label_shift_baseline.json")
DEFAULT_CLASSES = ("B", "LB", "VUS", "LP", "P")


def _load_labels(path: Path, column=None) -> list:
    p = Path(path)
    if p.suffix == ".npy":
        return [str(x) for x in np.load(p, allow_pickle=True)]
    df = pd.read_parquet(p)
    col = column or df.columns[0]
    return [str(x) for x in df[col]]


def build_baseline(y_train, y_val_true, y_val_pred, classes) -> dict:
    """p_train from y_train; reference_confusion = C[pred, true] = P(pred|true) from the
    validation (true, pred) pairs (column-stochastic, so C @ p_train is the expected
    predicted marginal -- matching LabelShiftAgent.detect)."""
    classes = tuple(str(c) for c in classes)
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    ptr = np.bincount([idx[str(c)] for c in y_train], minlength=k).astype(float)
    ptr = ptr / max(ptr.sum(), 1.0)
    M = np.zeros((k, k))  # M[true, pred]
    for t, p in zip(y_val_true, y_val_pred):
        M[idx[str(t)], idx[str(p)]] += 1.0
    rowsum = M.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    C = (M / rowsum).T  # column-stochastic C[pred, true] = P(pred|true)
    return {
        "classes": list(classes),
        "p_train": [float(x) for x in ptr],
        "reference_confusion": [[float(x) for x in row] for row in C],
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val_true)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the label-shift reference baseline (Run-17 artifact).")
    ap.add_argument("--y-train", type=Path, required=True, help="train labels (parquet/npy)")
    ap.add_argument("--y-val-true", type=Path, required=True, help="validation true labels (parquet/npy)")
    ap.add_argument("--y-val-pred", type=Path, required=True, help="validation predicted labels from the model (parquet/npy)")
    ap.add_argument("--label-column", default=None, help="column name if parquet (default: first column)")
    ap.add_argument("--classes", nargs="*", default=list(DEFAULT_CLASSES))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    ytr = _load_labels(args.y_train, args.label_column)
    yvt = _load_labels(args.y_val_true, args.label_column)
    yvp = _load_labels(args.y_val_pred, args.label_column)
    baseline = build_baseline(ytr, yvt, yvp, args.classes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  classes = {baseline['classes']}")
    print(f"  p_train = {[round(x, 4) for x in baseline['p_train']]}  (n_train={baseline['n_train']}, n_val={baseline['n_val']})")


if __name__ == "__main__":
    main()
