#!/usr/bin/env python3
"""
scripts/pandas3_equivalence_harness.py -- pandas 2.x -> 3.x upgrade equivalence harness.

PURPOSE
-------
Reverses the 2026-04-29 pandas==2.3.3 pin (which guarded against the pandas 3.0
string-dtype default change). Before/after this upgrade we must PROVE the feature
matrix and -- critically -- every merge's join-match count are byte-for-byte
identical, because a string-dtype mismatch at a merge key drops join rows SILENTLY
(no error). This harness produces that proof.

It runs ONLY data-prep + feature engineering (NO models, NO GNN) on a FIXED, seeded
cohort, then serializes a reference bundle:
  - features.parquet         : the concatenated X_train/X_val/X_test feature matrix
  - dtypes.json              : per-column dtype (str) of the feature matrix
  - shape_labels.json        : row/col counts + label balance per split
  - merge_counts.json        : per-merge join-match counts (the string-dtype proof)
  - feature_hash.txt         : sha256 over a canonical (sorted, rounded) dump
  - pandas_version.txt       : the pandas version that produced this bundle
  - warnings.json            : every (category, file, line, message) captured, with
                               counts -- defeats the once-per-site dedup so the
                               .fillna downcast offender list is DEFINITIVE.

USAGE
-----
  # Phase 1 (on pandas 2.3.3):  python scripts/pandas3_equivalence_harness.py \
  #     --cohort <fixed_cohort.parquet> --gnomad data/processed/gnomad_v4_exomes.parquet \
  #     --out baseline_pandas233
  # Phase 6 (on pandas 3.0.4):  ... --out result_pandas304
  # Then:                        python scripts/pandas3_equivalence_harness.py \
  #     --compare baseline_pandas233 result_pandas304

The cohort is passed PRE-BUILT (deterministic) so both runs see the IDENTICAL rows.
Build it once with --build-cohort (seeded), which writes the cohort + its sha256.

NOTE on instrumentation: we wrap pandas.DataFrame.merge to record, per call site,
the left row count, right row count, and result row count. This is read-only telemetry;
it does not alter merge behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
import warnings
from pathlib import Path


# --------------------------------------------------------------------------- #
# Merge instrumentation: wrap DataFrame.merge to count join matches per call site.
# --------------------------------------------------------------------------- #
_MERGE_LOG = []


def _install_merge_counter():
    import pandas as pd

    _orig_merge = pd.DataFrame.merge

    def _counting_merge(self, right, *args, **kwargs):
        result = _orig_merge(self, right, *args, **kwargs)
        try:
            # caller site = the first frame above this wrapper not in pandas internals
            stack = traceback.extract_stack()
            site = "unknown"
            for fr in reversed(stack[:-1]):
                if "pandas" not in fr.filename.replace("\\", "/").split("/site-packages/")[-1][:6]:
                    site = Path(fr.filename).name  # line-INSENSITIVE (file drifts as code is edited)
                    break
            on = kwargs.get("on")
            how = kwargs.get("how", "inner")
            try:
                left_n = len(self)
                right_n = len(right)
                out_n = len(result)
            except Exception:
                left_n = right_n = out_n = -1
            _MERGE_LOG.append({
                "site": site, "on": str(on), "how": str(how),
                "left_rows": left_n, "right_rows": right_n, "result_rows": out_n,
            })
        except Exception:
            # telemetry must never break the merge
            pass
        return result

    pd.DataFrame.merge = _counting_merge


# --------------------------------------------------------------------------- #
# Warning capture: record EVERY warning (always), with counts per site.
# --------------------------------------------------------------------------- #
def _canonical_feature_hash(df) -> str:
    import numpy as np
    import pandas as pd

    cols = sorted(df.columns.astype(str).tolist())
    buf = hashlib.sha256()
    buf.update(("|".join(cols)).encode("utf-8"))
    for c in cols:
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            vals = np.round(s.fillna(0.0).to_numpy(dtype="float64"), 8)
            buf.update(vals.tobytes())
        elif pd.api.types.is_integer_dtype(s):
            buf.update(s.fillna(0).to_numpy(dtype="int64").tobytes())
        else:
            buf.update(("|".join(map(str, s.fillna("").tolist()))).encode("utf-8"))
    return buf.hexdigest()


def _build_cohort(clinvar_path: str, n: int, seed: int, out_path: str):
    import pandas as pd

    src = pd.read_parquet(clinvar_path)
    if len(src) > n:
        src = src.sample(n=n, random_state=seed).reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    src.to_parquet(out_path, index=False)
    h = hashlib.sha256(Path(out_path).read_bytes()).hexdigest()
    Path(out_path + ".sha256").write_text(f"{h}  {Path(out_path).name}\n", encoding="utf-8")
    print(f"[cohort] wrote {len(src)} rows -> {out_path}")
    print(f"[cohort] sha256 {h}")
    print(f"[cohort] (commit this file + its .sha256 so both runs use IDENTICAL rows)")
    return 0


def _run_prep(cohort_path: str, gnomad_path: str, out_dir: str,
              simulate_pandas3: bool) -> int:
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if simulate_pandas3:
        try:
            pd.set_option("future.no_silent_downcasting", True)
            print("[sim] future.no_silent_downcasting = True (pandas-3 behavior on 2.x)")
        except Exception as e:
            print(f"[sim] could not set option (already pandas 3.x?): {e}")

    _install_merge_counter()

    from genomic_variant_classifier.data.real_data_prep import (
        AnnotationConfig, DataPrepConfig, DataPrepPipeline,
    )

    repo = Path(".")
    ext = repo / "data" / "external"
    proc = repo / "data" / "processed"

    def _p(rel):
        pp = repo / rel
        return pp if pp.exists() else None

    ann = AnnotationConfig(
        spliceai_path=_p("data/external/spliceai/spliceai_index.parquet"),
        esm2_uniprot_index_path=None,  # ESM-2 runs (real model) regardless; tiny cohort = fast
        alphamissense_path=_p("data/external/alphamissense/AlphaMissense_hg38.tsv.gz"),
        gtex_genes=[],
        gtex_path=None,
        kg_path=None,
        gnomad_constraint_path=_p("data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv"),
        lovd_path=_p("data/external/lovd/lovd_all_variants.parquet"),
        dbnsfp_path=_p("data/external/dbnsfp/dbnsfp_clinvar_index.parquet"),
        reactome_path=None,
        clingen_path=None,
        rnaseq_path=None,
        finngen_path=None,
        finngen_r13_path=None,
        omim_path=None,
        omim_genemap2_path=None,
        phylop_path=None,
        dbsnp_path=None,
        eve_path=None,
        eve_entry_map_path=None,
    )
    prep = DataPrepPipeline(
        config=DataPrepConfig(min_review_tier=3, output_dir=out / "splits"),
        annotation_config=ann,
    )

    captured = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        X_train, X_val, X_test, y_train, y_val, y_test, meta_val, meta_test = prep.run(
            clinvar_path=str(cohort_path),
            gnomad_path=gnomad_path,
        )
        for w in wlist:
            captured.append({
                "category": w.category.__name__,
                "file": f"{Path(w.filename).name}:{w.lineno}",
                "message": str(w.message)[:300],
            })

    # Concatenate splits into one feature matrix (sorted columns for determinism)
    X_all = pd.concat([X_train, X_val, X_test], axis=0, ignore_index=True)
    X_all = X_all.reindex(sorted(X_all.columns.astype(str)), axis=1)

    X_all.to_parquet(out / "features.parquet", index=False)
    (out / "dtypes.json").write_text(
        json.dumps({c: str(X_all[c].dtype) for c in X_all.columns}, indent=2, sort_keys=True),
        encoding="utf-8")
    (out / "shape_labels.json").write_text(json.dumps({
        "train_rows": int(len(X_train)), "val_rows": int(len(X_val)), "test_rows": int(len(X_test)),
        "n_features": int(X_all.shape[1]),
        "train_pos": int(pd.Series(y_train).sum()),
        "val_pos": int(pd.Series(y_val).sum()),
        "test_pos": int(pd.Series(y_test).sum()),
    }, indent=2, sort_keys=True), encoding="utf-8")

    # Aggregate merge counts deterministically (sort by site+on+how)
    agg = {}
    for m in _MERGE_LOG:
        k = f"{m['site']}|on={m['on']}|how={m['how']}"
        if k not in agg:
            agg[k] = {"left_rows": m["left_rows"], "right_rows": m["right_rows"],
                      "result_rows": m["result_rows"], "calls": 1}
        else:
            agg[k]["result_rows"] += m["result_rows"]
            agg[k]["calls"] += 1
    (out / "merge_counts.json").write_text(
        json.dumps(dict(sorted(agg.items())), indent=2, sort_keys=True), encoding="utf-8")

    # Warning counts (defeats once-per-site dedup): count by (category,file,message-prefix)
    wagg = {}
    for w in captured:
        k = f"{w['category']}|{w['file']}|{w['message'][:80]}"
        wagg[k] = wagg.get(k, 0) + 1
    (out / "warnings.json").write_text(
        json.dumps(dict(sorted(wagg.items())), indent=2, sort_keys=True), encoding="utf-8")

    fhash = _canonical_feature_hash(X_all)
    (out / "feature_hash.txt").write_text(fhash + "\n", encoding="utf-8")
    (out / "pandas_version.txt").write_text(pd.__version__ + "\n", encoding="utf-8")

    print(f"[done] pandas {pd.__version__}")
    print(f"[done] feature matrix: {X_all.shape[0]} rows x {X_all.shape[1]} cols")
    print(f"[done] feature_hash: {fhash}")
    print(f"[done] merges recorded: {len(_MERGE_LOG)} ({len(agg)} distinct sites)")
    print(f"[done] downcast warnings: " + str(sum(
        v for k, v in wagg.items() if 'Downcasting' in k or 'downcast' in k)))
    print(f"[done] bundle -> {out}")
    return 0


def _compare(a_dir: str, b_dir: str) -> int:
    import pandas as pd

    a, b = Path(a_dir), Path(b_dir)
    problems = []

    va = (a / "pandas_version.txt").read_text().strip()
    vb = (b / "pandas_version.txt").read_text().strip()
    print(f"[compare] {a.name} (pandas {va})  vs  {b.name} (pandas {vb})")

    ha = (a / "feature_hash.txt").read_text().strip()
    hb = (b / "feature_hash.txt").read_text().strip()
    if ha == hb:
        print(f"[ok] feature_hash IDENTICAL ({ha[:16]}...)")
    else:
        problems.append(f"feature_hash DIFFERS: {ha[:16]} vs {hb[:16]}")

    da = json.loads((a / "dtypes.json").read_text())
    db = json.loads((b / "dtypes.json").read_text())
    if da == db:
        print(f"[ok] dtypes IDENTICAL ({len(da)} columns)")
    else:
        keys = set(da) | set(db)
        diffs = [f"{k}: {da.get(k,'<absent>')} -> {db.get(k,'<absent>')}"
                 for k in sorted(keys) if da.get(k) != db.get(k)]
        problems.append("dtypes DIFFER:\n    " + "\n    ".join(diffs))

    ma = json.loads((a / "merge_counts.json").read_text())
    mb = json.loads((b / "merge_counts.json").read_text())
    merge_diffs = []
    for k in sorted(set(ma) | set(mb)):
        ra = ma.get(k, {}).get("result_rows")
        rb = mb.get(k, {}).get("result_rows")
        if ra != rb:
            merge_diffs.append(f"{k}: result_rows {ra} -> {rb}")
    if not merge_diffs:
        print(f"[ok] merge join-match counts IDENTICAL ({len(ma)} merge sites)")
    else:
        problems.append("MERGE JOIN-MATCH COUNTS DIFFER (string-dtype break!):\n    "
                        + "\n    ".join(merge_diffs))

    sa = json.loads((a / "shape_labels.json").read_text())
    sb = json.loads((b / "shape_labels.json").read_text())
    if sa == sb:
        print(f"[ok] shapes + label balance IDENTICAL")
    else:
        problems.append(f"shapes/labels DIFFER: {sa} vs {sb}")

    # Cell-level diff if hashes mismatch (locate the offending columns)
    if ha != hb:
        try:
            fa = pd.read_parquet(a / "features.parquet")
            fb = pd.read_parquet(b / "features.parquet")
            if fa.shape == fb.shape:
                fa = fa.reindex(sorted(fa.columns.astype(str)), axis=1)
                fb = fb.reindex(sorted(fb.columns.astype(str)), axis=1)
                col_diffs = []
                for c in fa.columns:
                    try:
                        if not fa[c].equals(fb[c]):
                            ne = (fa[c].fillna(0).to_numpy() != fb[c].fillna(0).to_numpy()).sum() \
                                if pd.api.types.is_numeric_dtype(fa[c]) else \
                                (fa[c].fillna("").astype(str).to_numpy() != fb[c].fillna("").astype(str).to_numpy()).sum()
                            col_diffs.append(f"{c}: {ne} cells differ")
                    except Exception as e:
                        col_diffs.append(f"{c}: compare error {e}")
                if col_diffs:
                    problems.append("COLUMN-LEVEL diffs:\n    " + "\n    ".join(col_diffs))
        except Exception as e:
            problems.append(f"cell-level diff failed: {e}")

    print()
    if problems:
        print("RESULT: NOT EQUIVALENT -- DO NOT TRUST THE UPGRADE")
        for p in problems:
            print("  [FAIL] " + p)
        return 1
    print("RESULT: EQUIVALENT -- feature matrix, dtypes, merge counts, shapes all identical")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="pandas 2.x->3.x equivalence harness")
    ap.add_argument("--build-cohort", action="store_true",
                    help="Build a fixed seeded cohort parquet from --clinvar.")
    ap.add_argument("--clinvar", default="data/processed/clinvar_grch38_clean.parquet")
    ap.add_argument("--cohort", default="data/_pandas3/cohort_2k_seed42.parquet")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gnomad", default="data/processed/gnomad_v4_exomes.parquet")
    ap.add_argument("--out", default=None)
    ap.add_argument("--simulate-pandas3", action="store_true",
                    help="Set future.no_silent_downcasting=True (pandas-3 behavior on 2.x).")
    ap.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"), default=None)
    args = ap.parse_args(argv)

    if args.compare:
        return _compare(args.compare[0], args.compare[1])
    if args.build_cohort:
        return _build_cohort(args.clinvar, args.n, args.seed, args.cohort)
    if args.out:
        return _run_prep(args.cohort, args.gnomad, args.out, args.simulate_pandas3)
    ap.error("specify one of: --build-cohort | --out DIR | --compare A B")


if __name__ == "__main__":
    sys.exit(main())
