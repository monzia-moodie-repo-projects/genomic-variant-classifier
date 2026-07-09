#!/usr/bin/env python
"""
evaluate_predictions.py  (2026-07-08)  -- READ-ONLY. Writes nothing.
==========================================================================
APPLIES THE METRIC STACK TO THE REAL ENSEMBLE PREDICTIONS.

`RunArtifactWriter.save_test_predictions` writes, for every run it touches:

    label, ensemble_prob, <model>_prob ..., variant_id, gene_symbol,
    consequence, chrom, pos, ref, alt

Those files exist for `run10/full` and for all three `ablation_run15` arms. They are the
only artifacts that let us evaluate the ENSEMBLE rather than a single feature, and they
carry `ref`/`alt`, so every metric can be stratified by variant representation.

WHAT THIS SETTLES

    1. Is AUROC 0.998 a model result or a cohort artifact?
       `probe_run14_univariate_leakage.py` showed `cadd_phred` ALONE reaches AUROC 0.9761
       on SNVs and exactly 0.5000 on every indel class (constant there -- the coordinate
       bug). The overall 0.7613 is Simpson's paradox. The ensemble's headline may decompose
       the same way: strong on SNVs, and merely reading `is_indel` on the 14% of rows that
       are indels and 62-78% pathogenic. Stratify and see.

    2. Does removing `n_pathogenic_in_gene` matter?
       `run9_ablations.py:165` calls it "directly circular with the label ... top feature
       every run; importance 391 in Run 15", yet `ablation_results.parquet` reports
       delta-AUROC = 0.0002 for `no_gene_prevalence`. Compare the arms PER STRATUM: a leak that
       lives in a subpopulation is invisible in a pooled number.

    3. What is the TRUE calibration?
       `evaluator.py::_calibration_error` bins on `[0.9, 1.0)`, so every `p == 1.0` -- a
       pure tree leaf -- falls in no bin and is silently dropped. The stack's ECE closes the
       top bin. On a 20%-pure-leaf split the reported ECE was 86.5% too small.
       (docs/audits/EVALUATION_STACK_AUDIT_2026-07-08.md sec 3.1)

READ THE OUTPUT WITH THESE CAVEATS

    * `outputs/run14/full/splits/` and the ablation splits carry `gnn_score` as a DEFAULT
      CONSTANT -- DataPrepPipeline writes splits BEFORE run_phase2_eval.py trains the GNN,
      and the in-memory overwrite never reaches disk (run9_ablations.py docstring). The
      persisted matrix is therefore not the matrix the mainline ensemble trained on, and
      `no_gnn` is a no-op.
    * `fit_seconds` for the run15 arms is 16,233 / 2,293 / 1,552 at identical `n_folds=3`.
      If `--max-train` differed between arms they are not comparable. `max_train` is NOT in
      the manifest config, but `split_hashes` IS. Check them before trusting any delta.
    * Train and test share the deletion coordinate corruption identically, so no metric
      computed here can detect it.

USAGE (from project root, .venv312 active)
    python scripts/evaluate_predictions.py --pred outputs/run10/full/test_predictions.parquet
    python scripts/evaluate_predictions.py \\
        --pred outputs/ablation_run15/full/test_predictions.parquet \\
        --pred outputs/ablation_run15/no_gene_prevalence/test_predictions.parquet \\
        --pred outputs/ablation_run15/no_gene_level/test_predictions.parquet \\
        --stratify representation --base-models --n-boot 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from genomic_variant_classifier.evaluation.metrics import (
    auroc, auprc, evaluate, no_skill_auprc, stratified_evaluate,
)

LABEL_CANDIDATES = ("label", "y", "y_true", "target", "is_pathogenic")
SCORE_CANDIDATES = ("ensemble_prob", "ensemble_proba", "proba", "prediction", "y_proba")

PANEL = ["n", "n_pos", "pos_rate", "auroc", "auprc", "auprc_no_skill", "auprc_lift",
         "brier", "ece", "cal_slope", "cal_intercept", "calibration_valid"]


def representation(ref: str, alt: str) -> str:
    if not ref or not alt:
        return "empty"
    if len(ref) == 1 and len(alt) == 1:
        return "SNV"
    if len(alt) < len(ref) and ref.startswith(alt):
        return "padded_deletion"
    if len(ref) < len(alt) and alt.startswith(ref):
        return "padded_insertion"
    if ref[0] == alt[0]:
        return "padded_other"
    return "delins"


def _resolve(df: pd.DataFrame, candidates: tuple[str, ...], what: str) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise KeyError(f"cannot resolve the {what} column from {candidates}.\n"
                   f"  columns present: {list(df.columns)}")


def _strata(df: pd.DataFrame, how: str) -> pd.Series:
    if how == "representation":
        if {"ref", "alt"} <= set(df.columns):
            return pd.Series([representation(str(r), str(a)) for r, a in zip(df["ref"], df["alt"])])
        parts = df["variant_id"].astype(str).str.split(":", expand=True)
        return pd.Series([representation(r, a) for r, a in zip(parts[3], parts[4])])
    if how not in df.columns:
        raise KeyError(f"--stratify {how!r} is not a column. Present: {list(df.columns)}")
    return df[how].astype("string").fillna("(missing)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Apply the metric stack to test_predictions.parquet")
    ap.add_argument("--pred", action="append", required=True,
                    help="path to a test_predictions.parquet; repeat to compare arms")
    ap.add_argument("--stratify", default="representation",
                    help="'representation' (from ref/alt) or any column name, e.g. 'consequence'")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--score-col", default=None)
    ap.add_argument("--base-models", action="store_true", help="also score every *_prob column")
    ap.add_argument("--n-boot", type=int, default=0, help="bootstrap reps for AUROC/AUPRC CIs")
    ap.add_argument("--min-n", type=int, default=30)
    a = ap.parse_args(argv)

    paths = [Path(p) for p in a.pred]
    for p in paths:
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 2

    print("=" * 104)
    print("METRIC STACK APPLIED TO ENSEMBLE PREDICTIONS")
    print(f"  stratify by: {a.stratify}   bootstrap reps: {a.n_boot or 'off'}")
    print("=" * 104)

    headline: dict[str, dict] = {}
    strat_tables: dict[str, pd.DataFrame] = {}
    base_tables: dict[str, pd.DataFrame] = {}

    for p in paths:
        arm = f"{p.parent.name}"
        df = pd.read_parquet(p)
        try:
            lc = a.label_col or _resolve(df, LABEL_CANDIDATES, "label")
            sc = a.score_col or _resolve(df, SCORE_CANDIDATES, "score")
        except KeyError as exc:
            print(f"ABORT for {p}: {exc}", file=sys.stderr)
            return 3
        y = df[lc].to_numpy().astype(int)
        s = df[sc].to_numpy(dtype=float)

        print(f"\n{'-'*104}\n{arm}   ({p})")
        print(f"  rows {len(df):,}   label='{lc}'   score='{sc}'   "
              f"pos_rate {100*no_skill_auprc(y):.4f}%")
        bad = (~np.isfinite(s)).sum()
        if bad:
            print(f"  WARNING: {bad:,} non-finite scores -- excluded from every metric")

        headline[arm] = evaluate(y, s, n_boot=a.n_boot)
        strata = _strata(df, a.stratify)
        strat_tables[arm] = stratified_evaluate(y, s, strata, n_boot=a.n_boot, min_n=a.min_n)

        if a.base_models:
            probs = [c for c in df.columns if c.endswith("_prob") and c != sc]
            rows = []
            for c in probs:
                v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                if not np.isfinite(v).any():
                    rows.append({"model": c, "auroc": np.nan, "auprc": np.nan, "note": "all non-finite"})
                    continue
                nu = int(pd.Series(v).nunique(dropna=True))
                rows.append({"model": c, "auroc": auroc(y, v), "auprc": auprc(y, v),
                             "nunique": nu, "note": "CONSTANT" if nu <= 1 else ""})
            if rows:
                base_tables[arm] = pd.DataFrame(rows).set_index("model").sort_values(
                    "auroc", ascending=False)

    # ---- headline ----------------------------------------------------------
    print("\n" + "=" * 104)
    print("HEADLINE  (auprc_lift = auprc / pos_rate -- a lift near 1.0 is a coin flip)")
    print("=" * 104)
    hd = pd.DataFrame(headline).T
    cols = [c for c in PANEL if c in hd.columns]
    print(hd[cols].to_string())
    if not bool(hd["calibration_valid"].all()):
        print("\n  calibration_valid=False => the score is not in [0,1]; brier/ece/cal_* are NaN.")
    print("\n  NOTE: this ECE closes the top bin. `evaluator.py::_calibration_error` bins")
    print("  [0.9, 1.0), dropping every p == 1.0 -- so eval_report.json's ECE is SMALLER")
    print("  than the truth, by 86.5% on a 20%-pure-leaf split. Compare the two.")

    # ---- stratified --------------------------------------------------------
    print("\n" + "=" * 104)
    print(f"STRATIFIED BY {a.stratify.upper()}")
    print("=" * 104)
    for arm, t in strat_tables.items():
        print(f"\n  {arm}")
        show = [c for c in ("n", "pos_rate", "auroc", "auprc", "auprc_no_skill",
                            "auprc_lift", "ece", "cal_slope") if c in t.columns]
        print(t[show].round(4).to_string())
        print("\n  A stratum whose auprc_lift ~= 1.0 is being predicted no better than its own")
        print("  base rate, however high its raw AUPRC. A stratum with auroc == 0.5000 exactly")
        print("  has a CONSTANT score there -- the model is not discriminating at all.")

    # ---- deltas ------------------------------------------------------------
    if len(strat_tables) > 1:
        arms = list(strat_tables)
        ref = arms[0]
        print("\n" + "=" * 104)
        print(f"DELTA vs '{ref}'   (negative => the ablation HURT; the feature was doing work)")
        print("=" * 104)
        for other in arms[1:]:
            common = strat_tables[ref].index.intersection(strat_tables[other].index)
            d = pd.DataFrame({
                "n": strat_tables[ref].loc[common, "n"],
                "d_auroc": strat_tables[other].loc[common, "auroc"] - strat_tables[ref].loc[common, "auroc"],
                "d_auprc": strat_tables[other].loc[common, "auprc"] - strat_tables[ref].loc[common, "auprc"],
                "d_auprc_lift": strat_tables[other].loc[common, "auprc_lift"] - strat_tables[ref].loc[common, "auprc_lift"],
            })
            print(f"\n  {other}  -  {ref}")
            print(d.round(5).to_string())
        print("\n  A pooled delta of ~0 can hide a large per-stratum delta of opposite signs.")
        print("  Read the strata, not the ALL row.")

    # ---- base models -------------------------------------------------------
    if base_tables:
        print("\n" + "=" * 104)
        print("PER-BASE-MODEL DISCRIMINATION")
        print("=" * 104)
        for arm, t in base_tables.items():
            print(f"\n  {arm}")
            print(t.round(5).to_string())
            const = t.index[t.get("note", pd.Series(dtype=str)) == "CONSTANT"].tolist()
            if const:
                print(f"    >>> CONSTANT predictions from: {const} -- these models contributed nothing.")

    print("\n" + "=" * 104)
    print("BEFORE CITING ANY DELTA ABOVE, CONFIRM THE ARMS SHARE A TRAINING SET:")
    print("  Get-Content outputs\\ablation_run15\\full\\manifest.json               # split_hashes")
    print("  Get-Content outputs\\ablation_run15\\no_gene_prevalence\\manifest.json")
    print("  fit_seconds are 16,233 / 2,293 / 1,552 at identical n_folds=3. `--max-train`")
    print("  subsamples TRAIN and is not recorded in the config. If X_train_full hashes")
    print("  differ across arms, the ablation table is void.")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
