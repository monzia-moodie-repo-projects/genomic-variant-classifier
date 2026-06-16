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

Usage:  python scripts/audit_smoke_feature_population.py [splits_dir_or_parquet] [--run17]
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
# Run-17 no-defer feature set. Score/AF features use the "varies" (nunique>1) check (default=None) so
# the audit does not depend on exact default constants (verified defaults: gnn/hetero 0.5, reactome 0,
# af_1kg 0.0). dbNSFP/LOVD keep explicit-default checks. Valid against a FULL-FLAG smoke
# (--kg --hetero-gnn --kg-edges --string-db auto). is_mitochondrial / chrY-MT gnomAD af are NOT asserted
# here -- their density is a full-Run-17-scale property (MT ~0.07% of cohort => ~2 rows at smoke_n=3000).
EXPECT_RUN17 = {
    "af_log10":               (None, "fail", "gnomAD"),
    "gnn_score":              (None, "fail", "STRGNN"),
    "hetero_gnn_score":       (None, "fail", "hGNN"),
    "reactome_pathway_count": (None, "warn", "Reactm"),
    "af_1kg_afr":             (None, "fail", "1000G"),
    "af_1kg_eur":             (None, "fail", "1000G"),
    "af_1kg_eas":             (None, "fail", "1000G"),
    "af_1kg_sas":             (None, "fail", "1000G"),
    "af_1kg_amr":             (None, "fail", "1000G"),
    "cadd_phred":             (15.0, "fail", "dbNSFP"),
    "sift_score":             (0.5,  "fail", "dbNSFP"),
    "revel_score":            (0.5,  "fail", "dbNSFP"),
    "n_tools_pathogenic":     (0,    "fail", "dbNSFP"),
    "lovd_variant_class":     (0,    "warn", "LOVD"),
}

SPLIT_FILES = ["X_train.parquet", "X_val.parquet", "X_test.parquet"]


def _load_splits(path: Path):
    """Return [(split_name, df), ...] -- PER SPLIT, so a feature alive in train but dead in val/test
    (e.g. a focal-only gene score under gene-disjoint splits) is caught instead of being masked by
    concatenation. A single parquet file is returned as one ("matrix") split."""
    import pandas as pd
    if path.is_dir():
        out = []
        for fn in SPLIT_FILES:
            fp = path / fn
            if fp.exists():
                out.append((fn.replace("X_", "").replace(".parquet", ""), pd.read_parquet(fp)))
        return (out, f"{len(out)} split file(s) under {path}") if out else (None, f"no X_*.parquet in {path}")
    if path.is_file():
        return [("matrix", __import__("pandas").read_parquet(path))], str(path)
    return None, f"path not found: {path}"


# Activation hints for features that are dead-by-data (not a wiring bug) or have a known failure mode.
NOTES = {
    "reactome_pathway_count": "needs the Reactome parquet (scripts/build_reactome_parquet.py); "
                              "--kg-edges reactome:...gmt only feeds the hetero-GNN graph, NOT this feature",
    "lovd_variant_class": "very sparse (~369 cohort variants) -- near-zero at smoke scale",
    "hetero_gnn_score": "must be scored inductively across ALL splits; focal/train-only scoring leaves "
                        "val/test at the 0.5 default under gene-disjoint splits",
}


def _is_populated(series, default) -> bool:
    s = series.dropna()
    if len(s) == 0:
        return False
    if default is None:
        return s.nunique() > 1
    return int((s != default).sum()) > 0


def main() -> int:
    import pandas as pd

    run17 = "--run17" in sys.argv[1:]
    argv = [a for a in sys.argv[1:] if a != "--run17"]
    expect = EXPECT_RUN17 if run17 else EXPECT
    label = "Run-17 full-flag" if run17 else "Run-16b"
    arg = Path(argv[0]) if argv else Path("models/smoke_run16b/splits")
    print("=" * 78)
    print(f" {label} feature-matrix population audit (PER-SPLIT): {arg}")
    print("=" * 78)
    splits, src = _load_splits(arg)
    if splits is None:
        print(f" FAIL: {src}")
        return 2
    names = [n for n, _ in splits]
    total = sum(len(d) for _, d in splits)
    print(f" matrix: {total} rows across splits {names}  ({src})")

    hard_fail = False
    print("\n[newly-activated source features -- checked in EACH split]")
    for col, (default, severity, source) in expect.items():
        statuses = []
        dead_splits = []
        absent_splits = []
        for sname, sdf in splits:
            if col not in sdf.columns:
                absent_splits.append(sname)
                statuses.append(f"{sname}=ABSENT")
                continue
            ok = _is_populated(sdf[col], default)
            statuses.append(f"{sname}={'ok' if ok else 'DEAD'}")
            if not ok:
                dead_splits.append(sname)
        bad = bool(dead_splits or absent_splits)
        flag = ""
        if bad:
            if severity == "fail":
                hard_fail = True
                flag = "  <-- FAIL"
            else:
                flag = "  <-- WARN"
        print(f"  {col:<22} ({source:<6}) {' '.join(statuses):<34}{flag}")
        if bad and col in NOTES:
            print(f"      note: {NOTES[col]}")

    # supplementary: per-split constant scan on the concatenated frame
    cat = pd.concat([d for _, d in splits], ignore_index=True)
    print("\n[concatenated all-constant numeric scan (supplementary)]")
    num = cat.select_dtypes("number")
    consts = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
    print(("  " + ", ".join(consts)) if consts else "  (none -- every numeric column varies across the pool)")

    print("\n" + "=" * 78)
    if hard_fail:
        print(" VERDICT: FAIL -- a FAIL-severity feature is dead/absent in at least one split.")
        print("          A feature alive only in train (e.g. focal-only gene scores) is NOT acceptable:")
        print("          it is inert at val/test/inference. Fix before promoting to the full regen.")
        return 1
    print(" VERDICT: PASS -- every FAIL-severity feature is populated in ALL splits.")
    print("          WARN features (LOVD sparsity, Reactome-needs-parquet) are expected; see notes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
