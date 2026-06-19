#!/usr/bin/env python3
"""
aggregate_rnaseq_ablation.py  --  Monzia Moodie

Aggregate the Run-17-scale RNA-seq ablation: read metrics.json from each
<runs-root>/<config>_seed<N>/ dir, report per-config mean+/-std test/val AUROC across seeds, and compute
the gene-shuffle RETENTION on each split to settle whether rnaseq carries gene-specific signal:

    retention = (gene_shuffle - drop_all) / (full - drop_all)

  ~1.0 => shuffling genes barely hurts => rnaseq value is NOT gene-specific (a redundant high-cardinality
          prior, as the small-scale run suggested).
  ~0.0 => shuffling destroys the value => rnaseq IS gene-specific (tissue/expression contrast matters).
A test-vs-val disagreement (as seen at small scale) => still inconclusive; reported explicitly.

metrics.json schema (run_phase2_eval.py): top-level "auroc" (ENSEMBLE_STACKER test) and "val_auroc" (val).
Fails loud, listing the JSON keys, if those are absent.

  python scripts/aggregate_rnaseq_ablation.py --runs-root outputs/run17_ablation [--out summary.csv]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _read_metrics(run_dir: Path) -> tuple[float, float]:
    mp = run_dir / "metrics.json"
    if not mp.exists():
        raise FileNotFoundError(f"missing {mp}")
    m = json.loads(mp.read_text())
    if "auroc" not in m or "val_auroc" not in m:
        raise KeyError(f"{mp}: need 'auroc' + 'val_auroc'; got keys {sorted(m)}")
    return float(m["auroc"]), float(m["val_auroc"])


def _parse_dir(name: str) -> tuple[str, int | None]:
    # "<config>_seed<N>" -> (config, N); "<config>" -> (config, None)
    if "_seed" in name:
        cfg, _, s = name.rpartition("_seed")
        try:
            return cfg, int(s)
        except ValueError:
            return name, None
    return name, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    root = Path(args.runs_root)
    if not root.is_dir():
        print(f"ERROR: --runs-root not a directory: {root}", file=sys.stderr); return 2

    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            t, v = _read_metrics(d)
        except FileNotFoundError:
            print(f"[warn] {d.name}: no metrics.json (run incomplete?) -- skipped", file=sys.stderr); continue
        cfg, seed = _parse_dir(d.name)
        rows.append({"config": cfg, "seed": seed, "test_auroc": t, "val_auroc": v})

    if not rows:
        print("ERROR: no metrics.json found under runs-root", file=sys.stderr); return 3
    df = pd.DataFrame(rows).sort_values(["config", "seed"]).reset_index(drop=True)

    print("=== per-run AUROC ===")
    print(df.to_string(index=False))

    agg = (df.groupby("config")
             .agg(n=("test_auroc", "size"),
                  test_mean=("test_auroc", "mean"), test_std=("test_auroc", "std"),
                  val_mean=("val_auroc", "mean"), val_std=("val_auroc", "std"))
             .reset_index())
    print("\n=== per-config mean +/- std ===")
    print(agg.to_string(index=False))

    def _single(cfg, col):
        s = df.loc[df["config"] == cfg, col]
        return float(s.mean()) if len(s) else None

    full_t, full_v = _single("full", "test_auroc"), _single("full", "val_auroc")
    floor_cfg = "drop_all" if (df["config"] == "drop_all").any() else ("no_rnaseq" if (df["config"] == "no_rnaseq").any() else None)
    floor_t = _single(floor_cfg, "test_auroc") if floor_cfg else None
    floor_v = _single(floor_cfg, "val_auroc") if floor_cfg else None

    verdict = None
    if None in (full_t, full_v, floor_t, floor_v):
        print("\n[verdict] need both 'full' and a floor (drop_all/no_rnaseq) to compute retention -- "
              f"have full={full_t is not None}, floor({floor_cfg})={floor_t is not None}", file=sys.stderr)
    else:
        sh = df[df["config"] == "gene_shuffle"]
        if sh.empty:
            print("\n[verdict] no gene_shuffle runs -- cannot assess gene-specificity", file=sys.stderr)
        else:
            def _ret(shuf, full, floor):
                denom = full - floor
                return float("nan") if abs(denom) < 1e-9 else (shuf - floor) / denom
            ret_t = [_ret(x, full_t, floor_t) for x in sh["test_auroc"]]
            ret_v = [_ret(x, full_v, floor_v) for x in sh["val_auroc"]]
            mt, mv = np.nanmean(ret_t), np.nanmean(ret_v)
            print("\n=== gene-shuffle RETENTION  (1=non-gene-specific, 0=gene-specific) ===")
            print(f"  full:  test={full_t:.4f} val={full_v:.4f}   floor({floor_cfg}): test={floor_t:.4f} val={floor_v:.4f}")
            print(f"  test retention per seed: {[round(x,3) for x in ret_t]}  mean={mt:.3f}")
            print(f"  val  retention per seed: {[round(x,3) for x in ret_v]}  mean={mv:.3f}")
            agree = (mt > 0.66) == (mv > 0.66)
            if not agree:
                verdict = (f"INCONCLUSIVE: test ({mt:.2f}) and val ({mv:.2f}) retention disagree across the "
                           "0.66 line -- gene-specificity not settled even at Run-17 scale.")
            elif mt > 0.66 and mv > 0.66:
                verdict = (f"NON-GENE-SPECIFIC: retention high on both splits (test {mt:.2f}, val {mv:.2f}) -- "
                           "rnaseq acts as a redundant high-cardinality prior, not tissue/expression contrast.")
            else:
                verdict = (f"GENE-SPECIFIC: retention low on both splits (test {mt:.2f}, val {mv:.2f}) -- "
                           "shuffling gene<->expression destroys the value; rnaseq encodes gene-level signal.")
            print(f"\n[verdict] {verdict}")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\n[ok] wrote per-run table -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
