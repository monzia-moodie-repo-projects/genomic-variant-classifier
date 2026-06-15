#!/usr/bin/env python3
"""
scripts/smoke_all_models.py -- ALL-MODELS pre-launch smoke gate.

Standing law (2026-06-04): before every full/cloud run, fit EVERY ensemble model
at tiny scale with NO --skip flags and FAIL the launch if any model is missing,
errors, is skipped, or returns a degenerate OOF. A run is not valid if any model
in a model-comparison project silently drops out.

What it does
------------
1. Runs scripts/run_phase2_eval.py with --max-train SMOKE_N (default 3000),
   --string-db auto, the real seq-windows, and NO --skip-* flags, into a temp
   output dir. At ~3000 training rows SVM (O(n^2)) and KAN (100k self-cap) both
   run, so every model's fit path is exercised cheaply.
2. Asserts the required model backends are importable (catboost, kan, mc_dropout).
3. Asserts run_phase2_eval exited 0 (the GNN std>0 hard gate is inside it).
4. Asserts per_model_metrics.csv contains the FULL expected roster with finite
   AUROC (a model that errored is dropped from the ensemble -> absent here -> FAIL).
5. Asserts the log has no 'OOF failed' / 'Traceback' / 'skipping' lines.
6. Asserts gnn_score is non-degenerate via scripts/verify_gnn_score.py.

Exit 0 = green (safe to launch the full run); non-zero = blocked.

Usage (on the VM, from repo root, before launching the full run):
  python scripts/smoke_all_models.py \
      --clinvar data/processed/clinvar_grch38_clean.parquet \
      --gnomad data/processed/gnomad_v4_exomes.parquet \
      --spliceai data/external/spliceai/spliceai_index.parquet \
      --alphamissense data/external/alphamissense/AlphaMissense_hg38.tsv.gz \
      --seq-windows data/processed/clinvar_grch38_clean_seq.parquet \
      --gnomad-constraint data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv \
      --dbnsfp-path data/external/dbnsfp/dbnsfp_clinvar_index.parquet \
      --lovd-path data/external/lovd/lovd_all_variants.parquet
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _stream_child(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, str]:
    """Run cmd, streaming its merged stdout/stderr LIVE to this console AND to log_path, while also
    accumulating the full text for the post-run assertions. Replaces capture_output=True, which buffered
    everything until the child exited (a multi-hour smoke then looked hung -- no output, hidden temp log).
    PYTHONUNBUFFERED=1 is forced so the child flushes promptly.
    """
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    lines: list[str] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
            lines.append(line)
        proc.wait()
    return proc.returncode, "".join(lines)

# Models that must always be in the roster at smoke scale (no skips, n<100k).
_ALWAYS = {
    "random_forest", "xgboost", "lightgbm", "logistic_regression",
    "gradient_boosting", "tabular_nn", "cnn_1d", "svm", "svm_bagged_rbf",
}
_LOG_RED_FLAGS = ("OOF failed", "Traceback", "skipping", "DEGENERATE")


def expected_roster() -> tuple[set[str], list[str]]:
    """Full expected model set + any missing-backend errors (which are fatal)."""
    errors: list[str] = []
    roster = set(_ALWAYS)
    try:
        from genomic_variant_classifier.models.variant_ensemble import (
            _CATBOOST_AVAILABLE, _KAN_AVAILABLE, _MC_DROPOUT_AVAILABLE,
        )
    except Exception as exc:  # noqa: BLE001
        return roster, [f"cannot import availability flags: {exc}"]
    if not _KAN_AVAILABLE:
        errors.append("KAN backend NOT available (pip install imodelsx) - KAN is a required comparison model")
    else:
        roster.add("kan")
    if not _CATBOOST_AVAILABLE:
        errors.append("CatBoost backend NOT available - required comparison model")
    else:
        roster.add("catboost")
    if not _MC_DROPOUT_AVAILABLE:
        errors.append("mc_dropout/deep_ensemble backend NOT available - required comparison models")
    else:
        roster.add("mc_dropout")
        roster.add("deep_ensemble")
    return roster, errors


def check_outputs(outdir: Path, log_text: str, expected: set[str]) -> tuple[bool, list[str]]:
    """Pure check logic (unit-testable without torch/data)."""
    msgs: list[str] = []
    ok = True

    # (a) per-model metrics present + full roster + finite AUROC
    pmm = outdir / "per_model_metrics.csv"
    if not pmm.exists():
        return False, [f"[FAIL] missing {pmm}"]
    df = pd.read_csv(pmm, index_col=0)
    present = set(df.index)
    missing = expected - present
    if missing:
        ok = False
        msgs.append(f"[FAIL] models missing from roster (errored/skipped): {sorted(missing)}")
    else:
        msgs.append(f"[ok] all {len(expected)} expected models present: {sorted(expected)}")
    for m in sorted(expected & present):
        auroc = df.loc[m, "auroc"] if "auroc" in df.columns else float("nan")
        if pd.isna(auroc):
            ok = False
            msgs.append(f"[FAIL] {m}: AUROC is NaN (degenerate fit)")
    if "ENSEMBLE_STACKER" not in present:
        ok = False
        msgs.append("[FAIL] ENSEMBLE_STACKER missing (blend did not form)")

    # (b) log red flags
    for flag in _LOG_RED_FLAGS:
        n = log_text.count(flag)
        if n:
            ok = False
            msgs.append(f"[FAIL] log contains {n}x '{flag}'")
    if ok:
        msgs.append("[ok] log clean (no OOF failed / Traceback / skipping / DEGENERATE)")
    return ok, msgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="All-models pre-launch smoke gate")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--clinvar", required=True)
    ap.add_argument("--gnomad")
    ap.add_argument("--spliceai")
    ap.add_argument("--alphamissense")
    ap.add_argument("--seq-windows", dest="seq_windows")
    ap.add_argument("--gnomad-constraint", dest="gnomad_constraint")
    ap.add_argument("--dbnsfp-path", dest="dbnsfp_path")
    ap.add_argument("--lovd-path", dest="lovd_path")
    ap.add_argument("--string-db", dest="string_db", default="auto")
    ap.add_argument("--smoke-n", type=int, default=3000)
    ap.add_argument("--n-folds", dest="n_folds", type=int, default=3)
    ap.add_argument("--min-review-tier", dest="min_review_tier", type=int, default=3)
    ap.add_argument("--keep-output", action="store_true")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    eval_py = repo / "scripts" / "run_phase2_eval.py"
    verify_py = repo / "scripts" / "verify_gnn_score.py"
    if not eval_py.exists():
        print(f"[FAIL] {eval_py} not found")
        return 2

    print("== ALL-MODELS SMOKE GATE ==")
    # KAN_FAIL guard: patch the imodelsx bare-name bug in a separate process
    # BEFORE the eval subprocess below, so that subprocess imports the
    # corrected source (a same-process patch would not take effect).
    _patch = repo / "scripts" / "patch_imodelsx_kan.py"
    if _patch.exists():
        _pr = subprocess.run([sys.executable, str(_patch)], cwd=str(repo),
                             capture_output=True, text=True)
        print((_pr.stdout or _pr.stderr).strip())
        if _pr.returncode != 0:
            print("RESULT: BLOCKED (imodelsx KAN patch failed)")
            return 1
    expected, backend_errors = expected_roster()
    if backend_errors:
        for e in backend_errors:
            print(f"[FAIL] {e}")
        print("\nRESULT: BLOCKED (required model backend missing)")
        return 1
    print(f"[ok] backends available; expecting roster: {sorted(expected)}")

    outdir = Path(tempfile.mkdtemp(prefix="smoke_all_models_"))
    cmd = [
        sys.executable, str(eval_py),
        "--clinvar", args.clinvar,
        "--string-db", args.string_db,
        "--max-train", str(args.smoke_n),
        "--n-folds", str(args.n_folds),
        "--min-review-tier", str(args.min_review_tier),
        "--output", str(outdir),
    ]
    for flag, val in [
        ("--gnomad", args.gnomad), ("--spliceai", args.spliceai),
        ("--alphamissense", args.alphamissense), ("--seq-windows", args.seq_windows),
        ("--gnomad-constraint", args.gnomad_constraint),
        ("--dbnsfp-path", args.dbnsfp_path), ("--lovd-path", args.lovd_path),
    ]:
        if val:
            cmd += [flag, val]
    # NOTE: deliberately NO --skip-* flags. That is the whole point.

    print(f"[run] {' '.join(cmd)}")
    print(f"[outdir] {outdir}")
    print(f"[log] {outdir / 'smoke.log'}  (tailing this also works while the smoke runs)")
    print("[note] full DataPrepPipeline runs on the FULL cohort before --max-train applies; on a CPU-only\n"
          "       box this can take hours. Output streams live below; on a GPU box it is fast.", flush=True)
    rc, log_text = _stream_child(cmd, repo, outdir / "smoke.log")
    proc = SimpleNamespace(returncode=rc)

    ok = True
    if proc.returncode != 0:
        ok = False
        print(f"[FAIL] run_phase2_eval exited {proc.returncode} (GNN hard-gate or pipeline error)")
    else:
        print("[ok] run_phase2_eval exited 0")

    out_ok, msgs = check_outputs(outdir, log_text, expected)
    for m in msgs:
        print("  " + m)
    ok = ok and out_ok

    # GNN non-degeneracy (mirror of the in-run hard gate)
    if verify_py.exists():
        vp = subprocess.run(
            [sys.executable, str(verify_py), str(outdir / "splits")],
            cwd=str(repo), capture_output=True, text=True,
        )
        if vp.returncode == 0:
            print("[ok] gnn_score non-degenerate (verify_gnn_score)")
        else:
            ok = False
            print("[FAIL] gnn_score degenerate (verify_gnn_score):")
            print("    " + vp.stdout.strip().replace("\n", "\n    "))

    if not args.keep_output:
        shutil.rmtree(outdir, ignore_errors=True)
    else:
        print(f"[info] smoke outputs kept at {outdir}")

    print("\nRESULT:", "GREEN -- safe to launch" if ok else "BLOCKED -- do NOT launch")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
