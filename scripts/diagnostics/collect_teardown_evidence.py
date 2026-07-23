#!/usr/bin/env python3
"""collect_teardown_evidence.py -- fold the teardown-probe artifacts into one report.

WHY
===
The Teardown abort diagnostic publishes one artifact per arm (nine arms), and each
artifact holds a JSON result, a per-arm log, the resolved environment, and any GNU
Debugger backtrace. That is thirty-odd files. Reading them one at a time invites
exactly the error this investigation exists to avoid: noticing one arm and missing
the arm that actually matters.

This reads every artifact -- extracting the downloaded .zip files first if they have
not been unpacked -- and writes ONE plain-text report: a per-arm table, the failure
signatures, the captured backtraces, and an explicit verdict section that states
which construction arm is the smallest to abort and whether any mitigation arm
suppressed it.

It reads only. It never modifies the repository and never writes outside --out.

    python collect_teardown_evidence.py ^
        --input "C:\\Users\\monzi\\Downloads\\teardown_artifacts" ^
        --out   "C:\\Users\\monzi\\Downloads\\teardown_evidence.txt"

Author: written for Monzia Moodie, 2026-07-23.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Arms in bisect order: a construction arm that aborts names the culprit by the
# smallest subset of the stack it needs, so the order is meaningful, not cosmetic.
CONSTRUCTION_ORDER = [
    "numpy_only", "pandas_no_parquet", "pyarrow_read", "pyarrow_write",
    "pyarrow_direct", "baseline_real_script",
]
MITIGATION_ARMS = ["arrow_io_threads_1", "omp_num_threads_1", "all_threads_1"]


def _extract_any_zips(root: Path) -> list[Path]:
    extracted = []
    for archive in sorted(root.glob("*.zip")):
        target = root / archive.stem
        target.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
            extracted.append(target)
        except zipfile.BadZipFile:
            print(f"  WARNING: {archive.name} is not a readable zip; skipped",
                  file=sys.stderr)
    return extracted


def _load_results(root: Path) -> list[dict]:
    results = []
    for path in sorted(root.rglob("result_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  WARNING: could not parse {path}: {exc}", file=sys.stderr)
            continue
        data["_source"] = str(path)
        results.append(data)
    return results


def _collect_backtraces(root: Path) -> list[tuple[str, str]]:
    out = []
    for path in sorted(root.rglob("backtrace.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text or text == "no core dumps produced by this arm":
            continue
        out.append((str(path), text))
    return out


def _verdict(results: list[dict]) -> list[str]:
    by_arm = {r.get("arm"): r for r in results}
    lines: list[str] = []

    aborting = [a for a in CONSTRUCTION_ORDER
                if by_arm.get(a, {}).get("n_failures", 0) > 0]
    if aborting:
        smallest = aborting[0]
        lines.append(f"SMALLEST CONSTRUCTION ARM THAT ABORTED: {smallest}")
        lines.append("  This names the culprit by construction: the abort requires "
                     "no more of the")
        lines.append("  stack than this arm loads.")
    else:
        lines.append("NO CONSTRUCTION ARM ABORTED.")
        bounds = [by_arm[a]["upper_bound_95_one_sided"]
                  for a in CONSTRUCTION_ORDER if a in by_arm]
        if bounds:
            lines.append(f"  Tightest one-sided 95% upper bound on the rate: "
                         f"{min(bounds):.6f}")
        lines.append("  This bounds the rate; it does NOT prove the arms are safe. "
                     "If the")
        lines.append("  baseline also did not abort, the trigger likely needs the "
                     "full pytest")
        lines.append("  process context rather than a bare subprocess loop.")

    lines.append("")
    baseline = by_arm.get("baseline_real_script", {})
    baseline_failed = baseline.get("n_failures", 0) > 0
    if baseline_failed:
        lines.append("MITIGATION ARMS (baseline DID abort, so these are informative):")
        for arm in MITIGATION_ARMS:
            record = by_arm.get(arm)
            if not record:
                lines.append(f"  {arm:22s} MISSING")
                continue
            n = record["n_failures"]
            verdict = "suppressed the abort" if n == 0 else "did NOT suppress"
            lines.append(f"  {arm:22s} failures={n:<5d} -> {verdict}")
        lines.append("  An arm that suppresses while the baseline aborts identifies "
                     "the responsible")
        lines.append("  thread pool causally.")
    else:
        lines.append("MITIGATION ARMS: not informative this run, because the "
                     "baseline did not abort.")
        lines.append("  A mitigation cannot be shown to suppress something that "
                     "did not occur.")
    return lines


def build_report(root: Path) -> str:
    _extract_any_zips(root)
    results = _load_results(root)
    backtraces = _collect_backtraces(root)

    out: list[str] = []
    out.append("TEARDOWN-ABORT DIAGNOSTIC -- CONSOLIDATED EVIDENCE")
    out.append("=" * 70)
    out.append(f"generated : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    out.append(f"source    : {root}")
    out.append(f"arms found: {len(results)}  (expected 9)")
    out.append("")

    if not results:
        out.append("NO result_*.json FOUND. Check that the artifacts were downloaded")
        out.append("and that --input points at the folder containing them.")
        return "\n".join(out) + "\n"

    versions = {json.dumps(r.get("library_versions", {}), sort_keys=True)
                for r in results}
    out.append("RESOLVED ENVIRONMENT")
    out.append("-" * 70)
    for blob in sorted(versions):
        for key, value in sorted(json.loads(blob).items()):
            out.append(f"  {key:10s} {value}")
        out.append("")
    if len(versions) > 1:
        out.append("  NOTE: arms did not all resolve the same versions. That is "
                   "itself a finding.")
        out.append("")

    out.append("PER-ARM RESULTS")
    out.append("-" * 70)
    header = f"  {'arm':24s} {'iters':>7s} {'fails':>6s} {'rate':>10s} {'95% bound':>11s} {'secs':>7s}"
    out.append(header)
    out.append("  " + "-" * (len(header) - 2))
    order = CONSTRUCTION_ORDER + MITIGATION_ARMS
    for arm in order + [r["arm"] for r in results if r.get("arm") not in order]:
        record = next((r for r in results if r.get("arm") == arm), None)
        if record is None:
            out.append(f"  {arm:24s} {'MISSING':>7s}")
            continue
        out.append(f"  {arm:24s} {record['iterations']:>7d} "
                   f"{record['n_failures']:>6d} {record['observed_rate']:>10.6f} "
                   f"{record['upper_bound_95_one_sided']:>11.6f} "
                   f"{record['elapsed_seconds']:>7.1f}")
    out.append("")

    failing = [r for r in results if r.get("n_failures", 0) > 0]
    out.append("FAILURE DETAIL")
    out.append("-" * 70)
    if not failing:
        out.append("  no arm produced a non-zero exit")
    for record in failing:
        out.append(f"  arm {record['arm']}")
        out.append(f"    returncode counts: {record['returncode_counts']}")
        for signature, count in record.get("stderr_signatures", {}).items():
            out.append(f"    x{count}: {signature}")
        first = record.get("first_failure")
        if first:
            out.append(f"    first failure at iteration {first['iteration']}, "
                       f"returncode {first['returncode']}")
            out.append(f"    stdout before the abort: {first['stdout_tail']!r}")
            out.append("    stderr:")
            for line in first["stderr_full"].splitlines():
                out.append(f"      {line}")
        out.append("")

    out.append("BACKTRACES")
    out.append("-" * 70)
    if not backtraces:
        out.append("  no core dumps were produced by any arm")
    for path, text in backtraces:
        out.append(f"  from {path}")
        for line in text.splitlines():
            out.append(f"    {line}")
        out.append("")

    out.append("VERDICT")
    out.append("-" * 70)
    out.extend("  " + line for line in _verdict(results))
    out.append("")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="folder holding the downloaded artifact zips or folders")
    parser.add_argument("--out", required=True, help="path for the consolidated report")
    args = parser.parse_args(argv)

    root = Path(args.input).resolve()
    if not root.is_dir():
        print(f"ERROR: --input is not a directory: {root}", file=sys.stderr)
        return 2

    report = build_report(root)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8", newline="\n")

    print(report)
    print(f"[written] {out_path}  ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
