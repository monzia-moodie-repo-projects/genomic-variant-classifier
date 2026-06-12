#!/usr/bin/env python3
"""audit_smoke_feature_population.py -- after the Run-16b re-smoke, prove the three
newly-activated sources reached the FEATURE MATRIX the models trained on (the splits),
instead of silently filling defaults.

IMPORTANT: it reads the SPLITS (X_train/X_val/X_test.parquet), NOT clinvar_enriched.parquet.
clinvar_enriched.parquet is the pre-scoring base-cohort checkpoint (1931 rows, pre tier
filter, raw placeholder columns) -- auditing it gives a false FAIL. The scored, engineered
matrix the ensemble actually consumes is _save_splits' X_*.parquet (1681 rows, 81 features).

Checks the ENGINEERED columns (not raw connector outputs):
  gnomAD  -> af_log10            EXPECT it varies (gnomAD populated some AF)
  dbNSFP  -> cadd_phred (15.0), sift_score (0.5), revel_score (0.5),
             n_tools_pathogenic (0)   EXPECT populated (ClinVar index covers ClinVar)
  LOVD    -> lovd_variant_class (0)   WARN-only if all 0 (tiny-smoke overlap is plausible)

HARD-FAIL (exit 1): a FAIL-severity feature is absent, all-null, all-default, or constant.
WARN (exit 0):      LOVD all-default at smoke scale.

STRICTLY READ-ONLY.

Usage:  python scripts/audit_smoke_feature_population.py [splits_dir_or_parquet]
        default: models/smoke_run16b/splits
Author: Monzia Moodie."""
from __future__ import annotations

import sys
from pathlib import Path

# column -> (default | None, severity, source).  default=None => "varies" check (nunique>1).
EXPECT = {
    "af_log10":           (None, "fail", "gnomAD"),
    "cadd_phred":         (15.0, "fail", "dbNSFP"),
    "sift_score":         (0.5,  "fail", "dbNSFP"),
    "revel_score":        (0.5,  "fail", "dbNSFP"),
    "n_tools_pathogenic": (0,    "fail", "dbNSFP"),
    "lovd_variant_class": (0,    "warn", "LOVD"),
}
SPLIT_FILES = ["X_train.parquet", "X_val.parquet", "X_test.parquet"]


def _load_matrix(path: Path):
    import pandas as pd
    if path.is_dir():
        parts = []
        for fn in SPLIT_FILES:
            fp = path / fn
            if fp.exists():
                parts.append(pd.read_parquet(fp))
        if not parts:
            return None, f"no X_*.parquet found in {path}"
        return pd.concat(parts, ignore_index=True), f"{len(parts)} split file(s) under {path}"
    if path.is_file():
        return __import__("pandas").read_parquet(path), str(path)
    return None, f"path not found: {path}"


def main() -> int:
    import pandas as pd

    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/smoke_run16b/splits")
    print("=" * 78)
    print(f" Run-16b feature-matrix population audit: {arg}")
    print("=" * 78)
    df, src = _load_matrix(arg)
    if df is None:
        print(f" FAIL: {src}")
        print("       (point me at the splits dir, e.g. models\\smoke_run16b\\splits)")
        return 2
    print(f" matrix: {len(df)} rows x {df.shape[1]} cols  ({src})")

    present = set(df.columns)
    hard_fail = False
    print("\n[newly-activated source features]")
    for col, (default, severity, source) in EXPECT.items():
        if col not in present:
            print(f"  {col:<20} ({source:<6}) ABSENT  -> not in feature matrix  [{severity.upper()}]")
            if severity == "fail":
                hard_fail = True
            continue
        s = df[col]
        n = len(s)
        n_null = int(s.isna().sum())
        nun = s.dropna()
        n_distinct = int(nun.nunique())
        if default is None:
            populated = n_distinct > 1
            metric = f"distinct={n_distinct}"
        else:
            n_nondefault = int((nun != default).sum())
            populated = n_nondefault > 0
            metric = f"nondefault={n_nondefault} (default={default})"
        rng = f"[{nun.min():.4g}, {nun.max():.4g}]" if len(nun) else "[n/a]"
        verdict = "POPULATED" if populated else ("ALL-NULL" if n_null == n else "ALL-DEFAULT")
        flag = ""
        if not populated:
            flag = f"  <-- {severity.upper()}"
            if severity == "fail":
                hard_fail = True
        print(f"  {col:<20} ({source:<6}) {verdict:<11} {metric:<28} null={n_null:<5} range={rng}{flag}")

    print("\n[all-constant numeric scan (supplementary)]")
    num = df.select_dtypes("number")
    consts = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
    if consts:
        for c in consts:
            val = num[c].dropna().iloc[0] if num[c].notna().any() else "all-null"
            print(f"  CONSTANT  {c} = {val}")
    else:
        print("  (none -- every numeric column varies)")

    print("\n" + "=" * 78)
    if hard_fail:
        print(" VERDICT: FAIL -- a FAIL-severity source feature is dead in the matrix.")
        print("          Do NOT promote to the full regen; check that source's log hit-count.")
        return 1
    print(" VERDICT: PASS -- all FAIL-severity new sources reached the matrix. A LOVD WARN")
    print("          is expected at smoke scale; re-confirm lovd_variant_class>0 at full scale.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
