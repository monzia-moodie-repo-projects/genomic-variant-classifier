#!/usr/bin/env python3
# =============================================================================
# run14_observability.py
# =============================================================================
# Purpose:  Maximum-information extractor for Run 14. Runs on the Vast.ai VM
#           AFTER training completes and BEFORE the instance is destroyed.
#           Emits a single structured JSON + Markdown report capturing
#           every observable signal we can pull from the run artifacts.
#
# Author:   Monzia Moodie
# Created:  2026-05-26
# Target:   genomic-variant-classifier @ commit bf2f665, Run 14
#
# What it captures (per the "maximize info per run" directive):
#   1. Per-model train + OOF wall-clock time
#   2. Per-model OOF metrics (AUROC, AUPRC, F1, MCC, Brier)
#   3. Per-model test-set metrics (when locked test eval ran)
#   4. KAN backend confirmation (imodelsx vs pykan vs MLP fallback)
#   5. LightGBM device used (cpu vs gpu vs cuda)
#   6. CNN_1D skip confirmation
#   7. Feature non-zero rate (which features actually carried signal)
#   8. Blend weights (meta-learner contribution per base model)
#   9. Peak GPU memory observed
#  10. Master-log error and warning summary
#  11. Per-model artifact size on disk
#  12. Total wall-clock + cost estimate
#  13. Git commit, instance ID, host info for reproducibility
#
# Usage on VM:
#   python3 /workspace/genomic-variant-classifier/scripts/run14_observability.py \
#       --outputs-dir /workspace/genomic-variant-classifier/outputs/run9_fresh \
#       --log /workspace/run11_master.log \
#       --report-dir /workspace/run14_report \
#       --instance-id 37999999 \
#       --hourly-rate 0.74
#
# Idempotent: safe to re-run; overwrites the report directory.
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def sh(cmd: str, timeout: int = 30) -> str:
    """Run a shell command, return stdout (empty string on failure)."""
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=timeout)
        return out.decode("utf-8", errors="replace").strip()
    except Exception as e:
        return f"<error: {type(e).__name__}: {e}>"


def safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"<error: {e}>"


# -----------------------------------------------------------------------------
# Log scrapers
# -----------------------------------------------------------------------------

MODEL_NAMES = [
    "random_forest", "xgboost", "lightgbm", "gradient_boosting",
    "logistic_regression", "catboost", "tabular_nn",
    "mc_dropout", "deep_ensemble", "kan", "cnn_1d",
]


def parse_log_for_per_model_metrics(log_text: str) -> dict[str, dict[str, Any]]:
    """
    Scan the master log for per-model OOF/test metrics. The training script
    emits lines like:
        ==> random_forest OOF AUROC: 0.9964
        ==> xgboost test_AUROC=0.9974 AUPRC=0.9913 F1=0.9768 MCC=0.9536 Brier=0.0124
    This parser is tolerant of either format.
    """
    out: dict[str, dict[str, Any]] = {m: {} for m in MODEL_NAMES}

    # Pattern A: "<model> OOF AUROC: 0.NNNN" (with or without "==>" prefix).
    # A7 fix 2026-05-27: the Python logger emits without "==>"; original regex
    # required it and matched nothing on real training logs.
    pat_a = re.compile(r"\b(\w+)\s+OOF\s+AUROC[:=]\s*([0-9.]+)", re.IGNORECASE)
    for m in pat_a.finditer(log_text):
        name, val = m.group(1).lower(), float(m.group(2))
        if name in out:
            out[name]["oof_auroc"] = val

    # Pattern B: "==> <model> test_AUROC=0.NNNN AUPRC=0.NNNN F1=... MCC=... Brier=..."
    pat_b = re.compile(
        r"==>\s+(\w+)\s+test_AUROC=([0-9.]+)(?:\s+AUPRC=([0-9.]+))?(?:\s+F1=([0-9.]+))?"
        r"(?:\s+MCC=([0-9.]+))?(?:\s+Brier=([0-9.]+))?",
        re.IGNORECASE,
    )
    for m in pat_b.finditer(log_text):
        name = m.group(1).lower()
        if name not in out:
            continue
        for i, key in enumerate(("test_auroc", "test_auprc", "test_f1", "test_mcc", "test_brier"), start=2):
            v = m.group(i)
            if v is not None:
                out[name][key] = float(v)

    # Pattern C: "<model> trained in NNNs" / "trained in HH:MM:SS"
    pat_c = re.compile(r"(\w+)\s+trained\s+in\s+([0-9hms:.\s]+)", re.IGNORECASE)
    for m in pat_c.finditer(log_text):
        name = m.group(1).lower()
        if name in out:
            out[name]["train_time_raw"] = m.group(2).strip()

    return out


def read_per_model_metrics_files(outputs_dir: Path) -> dict[str, dict[str, Any]]:
    """
    A7 fix 2026-05-27: prefer structured outputs over log-grep.
    Reads:
      - {outputs_dir}/per_model_metrics.csv      -> test metrics (auroc, auprc, f1_macro, f1_weighted, mcc, brier)
      - {outputs_dir}/per_model_metrics_val.csv  -> val metrics (same schema)
      - {outputs_dir}/models/*_meta.json         -> OOF AUROC + saved_at_utc + n_samples

    Returns dict keyed by model name with merged metrics. Returns empty dict
    if no source files are present (caller should fall back to log-grep).
    """
    import csv as _csv  # local import to avoid changing top-level imports
    out: dict[str, dict[str, Any]] = {}

    if not outputs_dir.exists():
        return out

    # OOF AUROC from per-model meta JSONs
    models_dir = outputs_dir / "models"
    if models_dir.exists():
        for meta_path in sorted(models_dir.glob("*_meta.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                name = meta.get("name") or meta_path.stem.replace("_meta", "")
                d = out.setdefault(name, {})
                if "oof_auroc" in meta:
                    d["oof_auroc"] = float(meta["oof_auroc"])
                if "saved_at_utc" in meta:
                    d["saved_at_utc"] = meta["saved_at_utc"]
                if "n_samples" in meta:
                    d["n_samples"] = int(meta["n_samples"])
            except Exception as e:
                # Don't let one bad meta-json break the whole read
                out.setdefault(f"_error_{meta_path.stem}", {})["error"] = f"{type(e).__name__}: {e}"

    # Test metrics from per_model_metrics.csv
    test_csv = outputs_dir / "per_model_metrics.csv"
    if test_csv.exists():
        try:
            with test_csv.open("r", encoding="utf-8", newline="") as fp:
                reader = _csv.reader(fp)
                header = next(reader, None)
                if header:
                    metric_cols = [h.strip() for h in header[1:]]
                    for row in reader:
                        if not row or not row[0].strip():
                            continue
                        name = row[0].strip()
                        d = out.setdefault(name, {})
                        for i, metric in enumerate(metric_cols, start=1):
                            if i < len(row):
                                try:
                                    d[f"test_{metric}"] = float(row[i])
                                except ValueError:
                                    pass
        except Exception as e:
            out.setdefault("_error_test_csv", {})["error"] = f"{type(e).__name__}: {e}"

    # Val metrics from per_model_metrics_val.csv
    val_csv = outputs_dir / "per_model_metrics_val.csv"
    if val_csv.exists():
        try:
            with val_csv.open("r", encoding="utf-8", newline="") as fp:
                reader = _csv.reader(fp)
                header = next(reader, None)
                if header:
                    metric_cols = [h.strip() for h in header[1:]]
                    for row in reader:
                        if not row or not row[0].strip():
                            continue
                        name = row[0].strip()
                        d = out.setdefault(name, {})
                        for i, metric in enumerate(metric_cols, start=1):
                            if i < len(row):
                                try:
                                    d[f"val_{metric}"] = float(row[i])
                                except ValueError:
                                    pass
        except Exception as e:
            out.setdefault("_error_val_csv", {})["error"] = f"{type(e).__name__}: {e}"

    return out



def parse_log_for_kan_backend(log_text: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "patch_applied": False,
        "patch_message": None,
        "backend_used": None,
        "fit_succeeded": None,
        "error": None,
    }
    if "imodelsx_patch: fixed 3 bare-name refs" in log_text:
        info["patch_applied"] = True
        info["patch_message"] = "fixed 3 bare-name refs"
    elif "imodelsx_patch: already patched" in log_text:
        info["patch_applied"] = True
        info["patch_message"] = "already patched"

    if re.search(r"kan.*backend.*imodelsx", log_text, re.IGNORECASE):
        info["backend_used"] = "imodelsx"
    elif re.search(r"kan.*backend.*pykan", log_text, re.IGNORECASE):
        info["backend_used"] = "pykan"
    elif re.search(r"kan.*backend.*mlp|kan.*mlp\s+fallback", log_text, re.IGNORECASE):
        info["backend_used"] = "mlp_fallback"

    if re.search(r"kan.*(traceback|nameerror|attributeerror|runtimeerror)", log_text, re.IGNORECASE):
        m = re.search(r"(NameError|AttributeError|RuntimeError|ValueError)[^\n]+", log_text)
        info["error"] = m.group(0) if m else "unknown"
        info["fit_succeeded"] = False
    elif re.search(r"==>\s+kan\s+OOF\s+AUROC", log_text, re.IGNORECASE):
        info["fit_succeeded"] = True

    return info


def parse_log_for_lightgbm_device(log_text: str) -> dict[str, Any]:
    info: dict[str, Any] = {"device": None, "cuda_attempted": False, "fit_succeeded": None}
    if "CUDA Tree Learner was not enabled" in log_text:
        info["cuda_attempted"] = True
        info["device"] = "cpu_fallback"
    elif re.search(r"lightgbm.*device_type[:=]\s*cpu", log_text, re.IGNORECASE):
        info["device"] = "cpu"
    elif re.search(r"lightgbm.*device_type[:=]\s*(cuda|gpu)", log_text, re.IGNORECASE):
        info["device"] = "gpu_or_cuda"

    if re.search(r"==>\s+lightgbm\s+OOF\s+AUROC", log_text, re.IGNORECASE):
        info["fit_succeeded"] = True
    elif re.search(r"lightgbm.*(error|failed|skipped)", log_text, re.IGNORECASE):
        info["fit_succeeded"] = False
    return info


def parse_log_for_cnn_skip(log_text: str) -> dict[str, Any]:
    info = {"skipped": False, "reason": None, "auroc": None}
    if "CNN_1D skipped" in log_text:
        info["skipped"] = True
        info["reason"] = "no fasta_seq data"
    m = re.search(r"==>\s+cnn_1d\s+OOF\s+AUROC[:=]\s*([0-9.]+)", log_text, re.IGNORECASE)
    if m:
        info["auroc"] = float(m.group(1))
    return info


def parse_log_for_errors(log_text: str) -> list[str]:
    errors = []
    bad = re.compile(
        r"(?im)^.*(error|exception|traceback|nameerror|attributeerror|importerror|valueerror|runtimeerror|failed).*$"
    )
    benign = re.compile(r"(?i)(UserWarning|FutureWarning|DeprecationWarning|imodelsx_patch)")
    for line in bad.findall(log_text):
        if not benign.search(line) and line.strip():
            errors.append(line.strip()[:300])
    return errors[-40:]  # last 40


def parse_log_for_timestamps(log_text: str) -> dict[str, Any]:
    """Find the first and last timestamp in the log to estimate wall-clock."""
    pat = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")
    matches = pat.findall(log_text)
    if not matches:
        return {"first": None, "last": None, "elapsed_seconds": None}
    first, last = matches[0], matches[-1]
    try:
        t0 = datetime.fromisoformat(first.replace(" ", "T"))
        t1 = datetime.fromisoformat(last.replace(" ", "T"))
        return {"first": first, "last": last, "elapsed_seconds": (t1 - t0).total_seconds()}
    except Exception:
        return {"first": first, "last": last, "elapsed_seconds": None}


# -----------------------------------------------------------------------------
# Outputs-directory scrapers
# -----------------------------------------------------------------------------

def scan_artifacts(outputs_dir: Path) -> dict[str, Any]:
    """Walk the outputs dir, collect file sizes + key artifact info."""
    info: dict[str, Any] = {"exists": outputs_dir.exists(), "files": [], "total_size_mb": 0.0}
    if not outputs_dir.exists():
        return info
    total = 0
    for p in outputs_dir.rglob("*"):
        if p.is_file():
            sz = p.stat().st_size
            total += sz
            rel = p.relative_to(outputs_dir).as_posix()
            info["files"].append({"path": rel, "size_mb": round(sz / 1024 / 1024, 3)})
    info["total_size_mb"] = round(total / 1024 / 1024, 1)
    info["file_count"] = len(info["files"])
    return info


def feature_nonzero_rate(outputs_dir: Path) -> dict[str, Any]:
    """If the training split parquets exist, compute non-zero rate per feature."""
    info: dict[str, Any] = {"available": False, "features": {}}
    try:
        import pandas as pd
    except ImportError:
        info["error"] = "pandas not importable"
        return info

    candidates = [
        outputs_dir / "splits" / "X_train.parquet",
        outputs_dir / "X_train.parquet",
    ]
    x_path = next((p for p in candidates if p.exists()), None)
    if x_path is None:
        info["error"] = "X_train.parquet not found"
        return info

    try:
        df = pd.read_parquet(x_path)
        info["available"] = True
        info["n_rows"] = int(len(df))
        info["n_cols"] = int(len(df.columns))
        info["x_train_path"] = x_path.as_posix()
        for col in df.columns:
            try:
                nz = float((df[col] != 0).sum()) / max(len(df), 1)
                info["features"][col] = round(nz, 4)
            except Exception:
                info["features"][col] = None
        # Sort + count dead features
        dead = [c for c, v in info["features"].items() if v is not None and v < 0.001]
        info["dead_features"] = dead
        info["dead_feature_count"] = len(dead)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def blend_weights(outputs_dir: Path) -> dict[str, Any]:
    """If a meta-learner / blend weights file exists, read it."""
    info: dict[str, Any] = {"available": False}
    candidates = [
        outputs_dir / "blend_weights.json",
        outputs_dir / "meta_learner_coefs.json",
        outputs_dir / "ensemble_blend.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                info["available"] = True
                info["path"] = c.as_posix()
                info["weights"] = json.loads(c.read_text())
                return info
            except Exception as e:
                info["error"] = f"{type(e).__name__}: {e}"
    return info


# -----------------------------------------------------------------------------
# Environment / host info
# -----------------------------------------------------------------------------

def host_info() -> dict[str, Any]:
    return {
        "hostname": sh("hostname"),
        "uname": sh("uname -a"),
        "python_version": sys.version.split()[0],
        "cuda_devices": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null"),
        "disk_workspace": sh("df -h /workspace 2>/dev/null | tail -1"),
        "pip_versions": sh("pip list 2>/dev/null | grep -iE '^(lightgbm|xgboost|catboost|imodelsx|pykan|torch|numpy|pandas|scikit-learn|networkx)\\s'"),
    }


def git_info(repo_dir: Path) -> dict[str, Any]:
    cmd = f"cd {repo_dir} && git rev-parse HEAD && git rev-parse --abbrev-ref HEAD && git log -1 --format='%h %s'"
    return {"raw": sh(cmd)}


# -----------------------------------------------------------------------------
# Report writers
# -----------------------------------------------------------------------------

def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Run 14 Observability Report")
    lines.append(f"_Generated {report['generated_at']} on {report['host']['hostname']}_")
    lines.append("")
    lines.append(f"**Instance:** {report['instance_id']}  ")
    lines.append(f"**Commit:** `{report.get('git', {}).get('raw', 'unknown')}`  ")
    lines.append(f"**Master log:** `{report['log_path']}`  ")
    lines.append(f"**Outputs dir:** `{report['outputs_dir']}`")
    lines.append("")
    lines.append("---")

    # Wall-clock + cost
    elapsed = report["timing"].get("elapsed_seconds")
    if elapsed:
        h = elapsed / 3600
        cost = h * report.get("hourly_rate", 0.0)
        lines.append("## Wall-clock + cost")
        lines.append(f"- First timestamp: `{report['timing']['first']}`")
        lines.append(f"- Last timestamp:  `{report['timing']['last']}`")
        lines.append(f"- Elapsed: **{h:.2f} h** ({int(elapsed)} s)")
        lines.append(f"- Estimated cost: **${cost:.2f}** at ${report.get('hourly_rate', 0.0):.3f}/hr")
        lines.append("")

    # Per-model metrics table
    lines.append("## Per-model metrics")
    lines.append("")
    lines.append("| Model | OOF AUROC | Test AUROC | AUPRC | F1 | MCC | Brier | Train time |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m, d in report["per_model"].items():
        if m.startswith("_"):
            continue  # skip error buckets
        # A7 fix: prefer f1_macro (CSV schema) then fall back to f1 (log-scrape schema)
        f1_val = d.get("test_f1_macro", d.get("test_f1", "—"))
        lines.append(
            f"| {m} | {d.get('oof_auroc', '—')} | {d.get('test_auroc', '—')} | "
            f"{d.get('test_auprc', '—')} | {f1_val} | "
            f"{d.get('test_mcc', '—')} | {d.get('test_brier', '—')} | "
            f"{d.get('train_time_raw', '—')} |"
        )
    lines.append("")

    # KAN status
    lines.append("## KAN status")
    k = report["kan"]
    lines.append(f"- Package patch applied: **{k['patch_applied']}** ({k['patch_message']})")
    lines.append(f"- Backend used: **{k['backend_used']}**")
    lines.append(f"- Fit succeeded: **{k['fit_succeeded']}**")
    if k["error"]:
        lines.append(f"- Error: `{k['error']}`")
    lines.append("")

    # LightGBM status
    lines.append("## LightGBM status")
    lg = report["lightgbm"]
    lines.append(f"- Device: **{lg['device']}**")
    lines.append(f"- CUDA attempted: {lg['cuda_attempted']}")
    lines.append(f"- Fit succeeded: **{lg['fit_succeeded']}**")
    lines.append("")

    # CNN_1D status
    lines.append("## CNN_1D status")
    c = report["cnn_1d"]
    lines.append(f"- Skipped: **{c['skipped']}** ({c['reason']})")
    if c["auroc"] is not None:
        lines.append(f"- AUROC reported anyway: {c['auroc']} (if 0.5, sanity check failed)")
    lines.append("")

    # Feature non-zero
    lines.append("## Feature non-zero rate (signal coverage)")
    fnz = report["feature_nonzero"]
    if fnz.get("available"):
        lines.append(f"- X_train shape: {fnz['n_rows']:,} x {fnz['n_cols']}")
        lines.append(f"- Dead features (non-zero rate < 0.001): **{fnz['dead_feature_count']}**")
        if fnz["dead_features"]:
            lines.append("  - " + ", ".join(f"`{d}`" for d in fnz["dead_features"]))
        # Top 10 most populated features
        sorted_feats = sorted(
            ((k, v) for k, v in fnz["features"].items() if v is not None),
            key=lambda x: x[1], reverse=True,
        )
        lines.append("- Top 10 populated features:")
        for k, v in sorted_feats[:10]:
            lines.append(f"  - `{k}`: {v:.3f}")
    else:
        lines.append(f"- Not available: {fnz.get('error', 'unknown')}")
    lines.append("")

    # Blend weights
    lines.append("## Blend weights")
    bw = report["blend_weights"]
    if bw.get("available"):
        lines.append(f"- Source: `{bw['path']}`")
        for model, w in bw["weights"].items():
            lines.append(f"  - {model}: {w}")
    else:
        lines.append("- Not available")
    lines.append("")

    # Errors
    lines.append("## Recent errors / warnings")
    errs = report["errors"]
    if errs:
        for e in errs:
            lines.append(f"- `{e}`")
    else:
        lines.append("- No errors detected.")
    lines.append("")

    # Artifacts
    lines.append("## Artifact inventory")
    a = report["artifacts"]
    lines.append(f"- Total size: **{a.get('total_size_mb', 0)} MB** across {a.get('file_count', 0)} files")
    lines.append("- Largest 10:")
    largest = sorted(a.get("files", []), key=lambda f: f["size_mb"], reverse=True)[:10]
    for f in largest:
        lines.append(f"  - `{f['path']}` ({f['size_mb']} MB)")
    lines.append("")

    # Host info
    lines.append("## Host + environment")
    h = report["host"]
    for k, v in h.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--repo-dir", default="/workspace/genomic-variant-classifier")
    ap.add_argument("--instance-id", default="unknown")
    ap.add_argument("--hourly-rate", type=float, default=0.0)
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    log_path = Path(args.log)
    report_dir = Path(args.report_dir)
    repo_dir = Path(args.repo_dir)

    report_dir.mkdir(parents=True, exist_ok=True)

    log_text = safe_read(log_path) if log_path.exists() else ""

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instance_id": args.instance_id,
        "hourly_rate": args.hourly_rate,
        "log_path": log_path.as_posix(),
        "outputs_dir": outputs_dir.as_posix(),
        "log_size_bytes": log_path.stat().st_size if log_path.exists() else 0,
        "git": git_info(repo_dir),
        "host": host_info(),
        "timing": parse_log_for_timestamps(log_text),
        "per_model": None,  # filled below; structured-files preferred (A7 fix 2026-05-27)
        "kan": parse_log_for_kan_backend(log_text),
        "lightgbm": parse_log_for_lightgbm_device(log_text),
        "cnn_1d": parse_log_for_cnn_skip(log_text),
        "errors": parse_log_for_errors(log_text),
        "artifacts": scan_artifacts(outputs_dir),
        "feature_nonzero": feature_nonzero_rate(outputs_dir),
        "blend_weights": blend_weights(outputs_dir),
    }

    # A7 fix 2026-05-27: prefer structured outputs over log-grep
    per_model_structured = read_per_model_metrics_files(outputs_dir)
    if per_model_structured:
        report["per_model"] = per_model_structured
        report["per_model_source"] = "structured"
    else:
        report["per_model"] = parse_log_for_per_model_metrics(log_text)
        report["per_model_source"] = "log_scrape"

    json_path = report_dir / "run14_observability.json"
    md_path = report_dir / "run14_observability.md"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown_report(report, md_path)

    print(f"OBS_REPORT_JSON: {json_path}")
    print(f"OBS_REPORT_MD:   {md_path}")
    print(f"OBS_LOG_SIZE:    {report['log_size_bytes']:,} bytes")
    print(f"OBS_ARTIFACTS:   {report['artifacts'].get('total_size_mb', 0)} MB")
    print(f"OBS_DEAD_FEATS:  {report['feature_nonzero'].get('dead_feature_count', '—')}")
    print(f"OBS_KAN_BACKEND: {report['kan']['backend_used']}")
    print(f"OBS_LGB_DEVICE:  {report['lightgbm']['device']}")
    print(f"OBS_ERRORS:      {len(report['errors'])}")


if __name__ == "__main__":
    main()
