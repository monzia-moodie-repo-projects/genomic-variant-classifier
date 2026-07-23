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
import math
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
# Round-two arms, referenced against pyarrow_read rather than the real script.
READ_REFERENCE = "pyarrow_read"
READ_MITIGATION_ARMS = ["read_cpu_count_1", "read_arrow_io_threads_1",
                        "read_all_threads_1"]
CONVERSION_PAIR = ("to_pandas_explicit", "to_pandas_no_threads")


def _extract_any_zips(root: Path) -> tuple[list[str], list[str]]:
    """Unpack downloaded archives. Returns (extracted, failed) by name so the
    report can state what happened rather than leaving it to stderr, where it is
    easy to miss."""
    extracted: list[str] = []
    failed: list[str] = []
    for archive in sorted(root.rglob("*.zip")):
        target = archive.parent / archive.stem
        target.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
            extracted.append(archive.name)
        except (zipfile.BadZipFile, OSError) as exc:
            failed.append(f"{archive.name}: {type(exc).__name__}: {exc}")
    return extracted, failed


def _inventory(root: Path, limit: int = 40) -> list[str]:
    """Describe what is actually present. A tool that reports only what it failed
    to find forces the reader to guess between an empty folder, an unreadable
    archive and a wrong path -- three situations with three different fixes."""
    lines: list[str] = []
    if not root.exists():
        return [f"  the path does not exist: {root}"]
    entries = sorted(root.rglob("*"))
    files = [e for e in entries if e.is_file()]
    dirs = [e for e in entries if e.is_dir()]
    lines.append(f"  directories: {len(dirs)}    files: {len(files)}")
    if not entries:
        lines.append("  the folder is COMPLETELY EMPTY")
        return lines
    by_suffix: dict[str, int] = {}
    for f in files:
        by_suffix[f.suffix.lower() or "<no suffix>"] = by_suffix.get(
            f.suffix.lower() or "<no suffix>", 0) + 1
    lines.append(f"  file types: {dict(sorted(by_suffix.items()))}")
    lines.append("  entries:")
    for entry in entries[:limit]:
        try:
            size = entry.stat().st_size if entry.is_file() else 0
        except OSError:
            size = -1
        kind = "dir " if entry.is_dir() else "file"
        lines.append(f"    {kind} {size:>12} {entry.relative_to(root)}")
    if len(entries) > limit:
        lines.append(f"    ... and {len(entries) - limit} more")
    return lines


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
    base_f = baseline.get("n_failures", 0)
    base_n = baseline.get("iterations", 0) or 0
    if base_f <= 0 or base_n <= 0:
        lines.append("MITIGATION ARMS: not informative, because the reference arm "
                     "did not abort.")
        lines.append("  A mitigation cannot be shown to suppress something that "
                     "did not occur.")
        return lines

    base_rate = base_f / base_n
    lines.append(f"MITIGATION ARMS (reference baseline_real_script: {base_f}/{base_n} "
                 f"= {base_rate:.6f})")
    for arm in MITIGATION_ARMS:
        record = by_arm.get(arm)
        if not record:
            lines.append(f"  {arm:22s} MISSING")
            continue
        n_fail = record["n_failures"]
        n_iter = record.get("iterations", 0) or 0
        # POWER GATE. Claiming "this mitigation suppressed the abort" from zero
        # events is only defensible if zero events would have been UNLIKELY had the
        # mitigation done nothing. Expected events under the baseline rate is the
        # honest yardstick: below about three, P(zero anyway) exceeds five percent
        # and the arm simply cannot distinguish suppression from chance.
        expected = base_rate * n_iter
        p_zero_if_useless = math.exp(-expected) if expected > 0 else 1.0
        if n_fail > 0:
            lines.append(f"  {arm:22s} failures={n_fail:<5d} -> DID NOT suppress")
        elif expected < 3.0:
            lines.append(f"  {arm:22s} failures=0     -> UNDERPOWERED, proves nothing")
            lines.append(f"  {'':22s}   expected {expected:.2f} events if the "
                         f"mitigation did nothing;")
            lines.append(f"  {'':22s}   P(zero anyway) = {p_zero_if_useless:.3f}. "
                         f"Zero here is not evidence.")
        else:
            lines.append(f"  {arm:22s} failures=0     -> suppressed "
                         f"(expected {expected:.1f}, P(zero if useless) = "
                         f"{p_zero_if_useless:.4f})")
    # Round two: mitigations referenced against the high-rate reproducer, plus the
    # direct test of the conversion hypothesis.
    ref = by_arm.get(READ_REFERENCE)
    if ref and ref.get("n_failures", 0) > 0 and ref.get("iterations", 0):
        ref_rate = ref["n_failures"] / ref["iterations"]
        lines.append("")
        lines.append(f"READ-REFERENCED MITIGATIONS (reference {READ_REFERENCE}: "
                     f"{ref['n_failures']}/{ref['iterations']} = {ref_rate:.6f})")
        for arm in READ_MITIGATION_ARMS:
            rec = by_arm.get(arm)
            if not rec:
                lines.append(f"  {arm:24s} MISSING")
                continue
            expected = ref_rate * (rec.get("iterations", 0) or 0)
            p0 = math.exp(-expected) if expected > 0 else 1.0
            if rec["n_failures"] > 0:
                lines.append(f"  {arm:24s} failures={rec['n_failures']:<5d} -> DID NOT suppress")
            elif expected < 3.0:
                lines.append(f"  {arm:24s} failures=0     -> UNDERPOWERED "
                             f"(expected {expected:.2f})")
            else:
                lines.append(f"  {arm:24s} failures=0     -> SUPPRESSED "
                             f"(expected {expected:.1f}, P(zero if useless) = {p0:.2e})")

        a, b = CONVERSION_PAIR
        ra, rb = by_arm.get(a), by_arm.get(b)
        if ra and rb:
            lines.append("")
            lines.append("CONVERSION HYPOTHESIS (is Arrow-to-pandas the trigger?)")
            lines.append(f"  {a:24s} failures={ra['n_failures']}/{ra['iterations']}")
            lines.append(f"  {b:24s} failures={rb['n_failures']}/{rb['iterations']}")
            if ra["n_failures"] > 0 and rb["n_failures"] == 0:
                exp_b = (ra["n_failures"]/ra["iterations"]) * rb["iterations"]
                lines.append(f"  -> CONFIRMED: the conversion aborts, and disabling its")
                lines.append(f"     thread use removes the abort (expected {exp_b:.1f} "
                             f"had it done nothing).")
            elif ra["n_failures"] == 0:
                lines.append("  -> NOT confirmed: the explicit conversion did not abort,")
                lines.append("     so the trigger lies elsewhere in pandas.read_parquet.")
            else:
                lines.append("  -> NOT confirmed: both arms abort, so thread use is not")
                lines.append("     the discriminating factor.")

    have_read_reference = bool(
        ref and ref.get("n_failures", 0) > 0
        and any(a in by_arm for a in READ_MITIGATION_ARMS))
    if base_rate * base_n < 3.0 and not have_read_reference:
        lines.append("")
        lines.append("  WARNING: the reference arm is too rare to power ANY of these")
        lines.append("  comparisons. Re-run the mitigations against the highest-rate")
        lines.append("  reproducing arm instead, where suppression would be decisive.")
    return lines


def build_report(root: Path) -> str:
    extracted, failed_zips = _extract_any_zips(root)
    results = _load_results(root)
    backtraces = _collect_backtraces(root)

    out: list[str] = []
    out.append("TEARDOWN-ABORT DIAGNOSTIC -- CONSOLIDATED EVIDENCE")
    out.append("=" * 70)
    out.append(f"generated : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    out.append(f"source    : {root}")
    out.append(f"arms found: {len(results)}  (expected 9)")
    out.append("")

    if extracted or failed_zips:
        out.append("ARCHIVES")
        out.append("-" * 70)
        for name in extracted:
            out.append(f"  extracted: {name}")
        for problem in failed_zips:
            out.append(f"  FAILED   : {problem}")
        out.append("")

    if not results:
        out.append("NO ARM RESULTS FOUND")
        out.append("-" * 70)
        out.append("No result_*.json exists anywhere beneath the input path. What is")
        out.append("actually there:")
        out.extend(_inventory(root))
        out.append("")
        out.append("WHICH SITUATION THIS IS, AND WHAT TO DO")
        out.append("-" * 70)
        if not root.exists():
            out.append("  The path does not exist. Re-check the --input value.")
        elif not any(root.rglob("*")):
            out.append("  The folder is EMPTY. Nothing was ever downloaded into it.")
            out.append("  Either the diagnostic workflow has not been dispatched yet,")
            out.append("  or it ran but its artifacts were not downloaded. Confirm the")
            out.append("  run finished on the Actions page and that its Summary lists")
            out.append("  nine teardown-probe artifacts, then download them here.")
        elif failed_zips:
            out.append("  Archives were present but could not be opened (listed above).")
            out.append("  Re-download them; a truncated transfer is the usual cause.")
        else:
            out.append("  Files are present but none is a result_*.json. Either the")
            out.append("  input path points at the wrong folder, or the workflow run")
            out.append("  failed before the probe produced any result. Open the run and")
            out.append("  check whether the 'Run the probe arm' step succeeded.")
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
    # No evidence is not success. Exit non-zero so a wrapper, or a reader
    # skimming the tail, cannot mistake an empty run for a completed one.
    if "NO ARM RESULTS FOUND" in report:
        print("[no evidence] no arm results were found; exiting 1", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
