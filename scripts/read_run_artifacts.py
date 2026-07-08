#!/usr/bin/env python
"""
read_run_artifacts.py  (2026-07-08)  -- READ-ONLY. Writes nothing.
==========================================================================
WHY THIS EXISTS

    `docs/incidents/INCIDENT_2026-07-08_deletion-reviewstatus-loss.md` sec 6.2 asserts
    that "no run artifact records which cohort regime produced it". That assertion was
    made before anyone looked. The artifacts DO exist, for some runs:

        outputs/run10/full/{eval_report.json, manifest.json, test_predictions.parquet}
        outputs/run16/eval_report.json
        outputs/ablation_run15/{full,no_gene_level,no_gene_prevalence}/...
        outputs/ablation_probe/timing/...
        outputs/ablation_*/ablation_results.parquet

    `EvaluationReport.prevalence` is the AUPRC no-skill floor. `manifest.json` carries
    `git_sha` and the full `config` -- including, if it was passed, `min_review_tier`.
    Together they pin each run to a cohort regime from ground truth rather than from
    log-grepping.

    Absent for run14 and run17: `run_phase2_eval.py` never calls `RunArtifactWriter`
    (its `_write_model_manifest` is a different, joblib-sidecar function). Only
    `run9_ablations.py` does. See docs/audits/EVALUATION_STACK_AUDIT_2026-07-08.md sec 2.

WHAT IT PRINTS

    1. One row per `eval_report.json`: n, prevalence, AUROC (+CI), AUPRC (+CI),
       **auprc_lift = auprc / prevalence** (the number nobody has ever quoted),
       MCC, F1, Brier, ECE, MCE, and whether the consequence/gene breakdowns are empty
       (the "silent-empty trap" locked by test_evaluator_meta.py).
    2. One row per `manifest.json`: run_id, ablation, git_sha, and every config key,
       with `min_review_tier` called out.
    3. Every `ablation_results.parquet`, in full. **`ablation_run15` contains
       `no_gene_prevalence` and `no_gene_level`** -- ablations that already measure what
       happens when the gene-prevalence features (`n_pathogenic_in_gene`,
       `gene_has_known_disease`) are removed. Read this BEFORE asserting anything about
       the INCIDENT_2026-06-13 leak.

CAVEAT ON MEMORY

    Each eval_report.json is 4-15 MB because it embeds the full ROC / PR / calibration
    curves. They are loaded one at a time and the curves discarded immediately.

USAGE (from project root)
    python scripts/read_run_artifacts.py
    python scripts/read_run_artifacts.py --root outputs --show-config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCALARS = [
    "model_name", "n_samples", "n_pathogenic", "n_benign", "prevalence",
    "auroc", "auroc_ci_lo", "auroc_ci_hi",
    "auprc", "auprc_ci_lo", "auprc_ci_hi",
    "mcc", "f1", "brier_score", "calibration_ece", "calibration_mce",
]

CONFIG_KEYS_OF_INTEREST = [
    "min_review_tier", "n_folds", "max_train", "exclude_conflicting",
    "test_size", "val_size", "unseen_gene_holdout", "seed", "random_state",
    "skip_svm", "skip_kan", "skip_cnn", "hetero_gnn", "gnn_epochs",
]


def _read_eval_report(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as fh:
        d = json.load(fh)
    row = {k: d.get(k) for k in SCALARS}
    prev = d.get("prevalence")
    ap = d.get("auprc")
    row["auprc_no_skill"] = prev
    row["auprc_lift"] = (ap / prev) if (prev and ap is not None) else None
    row["n_consequence_rows"] = len(d.get("consequence_breakdown") or [])
    row["n_gene_error_rows"] = len(d.get("gene_errors") or [])
    for op in ("at_sensitivity_90", "at_sensitivity_95", "at_high_ppv"):
        o = d.get(op)
        row[f"{op}_present"] = o is not None
        if o:
            row[f"{op}_thr"] = o.get("threshold")
            row[f"{op}_ppv"] = o.get("ppv")
            row[f"{op}_sens"] = o.get("sensitivity")
    del d  # free the curves immediately
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read the run artifacts that exist.")
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--show-config", action="store_true", help="dump every manifest config key")
    a = ap.parse_args(argv)

    root = Path(a.root)
    if not root.exists():
        print(f"ERROR: {root} not found", file=sys.stderr)
        return 2

    # ---------------- 1. eval reports ------------------------------------
    reports = sorted(root.rglob("eval_report.json"))
    print("=" * 100)
    print(f"EVAL REPORTS  ({len(reports)} found under {root})")
    print("=" * 100)
    if not reports:
        print("  none. run_phase2_eval.py does not call RunArtifactWriter.save_eval_report().")
    rows = []
    for p in reports:
        try:
            r = _read_eval_report(p)
        except Exception as exc:  # noqa: BLE001 -- surface, never swallow
            print(f"  FAILED to read {p}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        r["path"] = str(p.relative_to(root))
        r["mb"] = round(p.stat().st_size / 1_048_576, 1)
        rows.append(r)

    if rows:
        df = pd.DataFrame(rows).set_index("path")
        core = ["mb", "model_name", "n_samples", "n_pathogenic", "prevalence",
                "auroc", "auroc_ci_lo", "auroc_ci_hi",
                "auprc", "auprc_no_skill", "auprc_lift"]
        print("\n--- discrimination, against the AUPRC no-skill floor ---")
        print(df[core].to_string())
        print("\n  auprc_lift = auprc / prevalence. A lift near 1.0 is a coin flip on that")
        print("  cohort, however high the raw AUPRC looks.")

        cal = ["mcc", "f1", "brier_score", "calibration_ece", "calibration_mce",
               "n_consequence_rows", "n_gene_error_rows"]
        print("\n--- calibration and breakdowns ---")
        print(df[[c for c in cal if c in df.columns]].to_string())
        print("\n  n_consequence_rows == 0 means the breakdown was SILENTLY EMPTY:")
        print("  meta lacked a 'consequence' column (test_evaluator_meta.py locks this trap).")
        print("  calibration_ece is UNDER-REPORTED: _calibration_error's last bin is")
        print("  [0.9, 1.0), so every p == 1.0 falls in no bin at all.")
        print("  (docs/audits/EVALUATION_STACK_AUDIT_2026-07-08.md sec 3.1)")

        ops = [c for c in df.columns if c.startswith("at_")]
        if ops:
            print("\n--- clinical operating points ---")
            print(df[ops].to_string())

    # ---------------- 2. manifests ----------------------------------------
    manifests = sorted(root.rglob("manifest.json"))
    print("\n" + "=" * 100)
    print(f"MANIFESTS  ({len(manifests)} found)")
    print("=" * 100)
    for p in manifests:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED to read {p}: {exc}", file=sys.stderr)
            continue
        cfg = d.get("config", {}) or {}
        print(f"\n  {p.relative_to(root)}")
        print(f"    run_id={d.get('run_id')!r}  ablation={d.get('ablation')!r}  "
              f"git_sha={d.get('git_sha')!r}")
        hits = {k: cfg[k] for k in CONFIG_KEYS_OF_INTEREST if k in cfg}
        if hits:
            print(f"    config (keys of interest): {hits}")
        else:
            print(f"    config keys present: {sorted(cfg)[:12]}{' ...' if len(cfg) > 12 else ''}")
        if "min_review_tier" in cfg:
            print(f"    >>> min_review_tier = {cfg['min_review_tier']}  "
                  f"(3 => regime v1 deletion-censored; absent/5 => regime v0)")
        else:
            print("    >>> min_review_tier ABSENT from config -- cannot pin the regime here")
        if a.show_config:
            print("    full config:")
            for k in sorted(cfg):
                print(f"      {k} = {cfg[k]!r}")
        ver = d.get("versions", {}) or {}
        if ver:
            print(f"    versions: {ver}")

    # ---------------- 3. ablation results ---------------------------------
    ablations = sorted(root.rglob("ablation_results.parquet"))
    print("\n" + "=" * 100)
    print(f"ABLATION RESULTS  ({len(ablations)} found)")
    print("=" * 100)
    for p in ablations:
        print(f"\n  {p.relative_to(root)}")
        try:
            adf = pd.read_parquet(p)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {exc}", file=sys.stderr)
            continue
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(adf.to_string(index=False))
        if "ablation" in adf.columns:
            names = set(adf["ablation"].astype(str))
            leak_ablations = {"no_gene_prevalence", "no_gene_level"} & names
            if leak_ablations:
                print(f"\n    >>> LEAKAGE ABLATIONS PRESENT: {sorted(leak_ablations)}")
                print("    These already measure what happens when the gene-prevalence features")
                print("    (n_pathogenic_in_gene, gene_has_known_disease) are removed. Compare")
                print("    their AUROC/AUPRC to 'full'. A large drop implicates the")
                print("    INCIDENT_2026-06-13 pre-split leak; a small drop exonerates it.")

    print("\n" + "=" * 100)
    print("NOTE ON n_pathogenic_in_gene AFTER THE 2026-06-13 FIX")
    print("=" * 100)
    print("  The fix computes the count from TRAIN labels only. The main split is")
    print("  gene-disjoint (GroupShuffleSplit), so a TEST gene has no train variants and")
    print("  its count is IDENTICALLY ZERO. tests/unit/test_d1_d2.py::TestRecomputeNpig")
    print("  ::test_test_npig_always_zero asserts exactly that.")
    print()
    print("  So the fix did not remove the feature -- it replaced a label leak with a")
    print("  TRAIN/TEST DISTRIBUTION SHIFT: the model learns on n_pathogenic_in_gene in")
    print("  [0, N] and predicts with it pinned at 0. If the model learned")
    print("  'high count => pathogenic', every unseen gene is pushed toward benign.")
    print("  Verify directly:  X_test['n_pathogenic_in_gene'].nunique()")
    print("    nunique > 1 on a PRE-fix run (e.g. run14)  -> the leak was live")
    print("    nunique == 1 on a POST-fix run             -> dead feature + shift")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
