r"""
preflight_gate.py - refuse to launch run_phase2_eval.py unless the rich-run
manifest is fully satisfied. Closes the silent-degradation class proven by
Run 14 (falsy --string-db -> GNN skipped) and the Run 15 mis-scope.

Grounded in `run_phase2_eval.py --help` (verified 2026-06-03).

Modes:
  --check "<full command>"   validate an externally-built launch command
  --emit                     print the canonical validated command (paths must exist)
  --data-root PATH           data root for path checks/emit (default: data)
  --n-train N                training-set size, to apply the >100k SVM rule (default: 1200000)
  --ack-omit a,b             acknowledge intentional omission of optional sources (e.g. finngen,kg)

Exit code 0 only if every check passes (no FAIL, no unacknowledged ACK-NEEDED).
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

# --- manifest (relative to --data-root) ---
REQUIRED_PATHS = {
    "--clinvar":           "processed/clinvar_grch38_clean.parquet",
    "--seq-windows":       "processed/clinvar_grch38_clean_seq.parquet",
    "--gnomad":            "processed/gnomad_v4_exomes.parquet",
    "--spliceai":          "external/spliceai/spliceai_index.parquet",
    "--alphamissense":     "external/alphamissense/AlphaMissense_hg38.tsv.gz",
    "--gnomad-constraint": "external/gnomad/gnomad.v4.1.constraint_metrics.tsv",
    "--dbnsfp-path":       "external/dbnsfp/dbnsfp_clinvar_index.parquet",
    "--lovd-path":         "external/lovd/lovd_all_variants.parquet",
    "--gtex-path":         "external/gtex_gene_expression.parquet",
    "--reactome-path":     "external/reactome_gene_pathways.parquet",
}
REQUIRED_VALUES = {"--string-db": "auto", "--min-review-tier": "3", "--n-folds": "5"}
REQUIRED_PRESENT_VALUE = ["--output"]            # must be present, value free
REQUIRED_FLAGS = ["--unseen-gene-holdout"]       # C3 falsifier
FORBIDDEN_FLAGS = ["--skip-nn", "--skip-cnn", "--skip-kan"]  # would diminish the battery
SCALE_SKIP = "--skip-svm"                        # help: "required at >100k samples"
ACK_OPTIONAL = {"--finngen-path": "finngen", "--kg": "kg"}   # documented Run-9-bug defaults if omitted

STORE_TRUE = {"--skip-nn", "--skip-svm", "--skip-kan", "--skip-cnn", "--unseen-gene-holdout"}


def _build_mirror_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    for f in ("--clinvar", "--seq-windows", "--gnomad", "--spliceai", "--alphamissense",
              "--kg", "--gnomad-constraint", "--lovd-path", "--dbnsfp-path", "--finngen-path",
              "--string-db", "--max-train", "--n-folds", "--min-review-tier",
              "--auroc-target", "--output", "--gtex-path", "--reactome-path"):
        p.add_argument(f)
    p.add_argument("--gtex-genes", nargs="*", default=[])
    p.add_argument("--kg-edges", nargs="*", default=[])
    p.add_argument("--hetero-gnn", action="store_true")
    for f in STORE_TRUE:
        p.add_argument(f, action="store_true")
    return p


def _parse_candidate(command: str) -> argparse.Namespace:
    # posix=False so Windows backslash paths (data\\processed\\x.parquet, C:\\...) survive -- POSIX shlex
    # treats '\\' as an escape and silently eats it, mangling the path so it "does not exist". Then strip any
    # surrounding quotes posix=False leaves on quoted tokens. Forward-slash paths are unaffected (cross-platform).
    toks = shlex.split(command, posix=False)
    toks = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in toks]
    # drop everything up to and including the script name, if present
    for i, t in enumerate(toks):
        if t.endswith("run_phase2_eval.py"):
            toks = toks[i + 1:]
            break
    ns, _unknown = _build_mirror_parser().parse_known_args(toks)
    return ns


def validate(ns: argparse.Namespace, data_root: str, n_train: int, ack: set[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    g = vars(ns)

    def flagval(flag):  # argparse dest
        return g.get(flag.lstrip("-").replace("-", "_"))

    for flag, rel in REQUIRED_PATHS.items():
        v = flagval(flag)
        if not v:
            rows.append(("FAIL", f"{flag} missing/empty (silent-zero trap per --help)"))
            continue
        if flag == "--seq-windows" and str(v).strip() == "":
            rows.append(("FAIL", f"{flag} empty -> CNN forced to poly-A (silent CNN degradation)"))
            continue
        p = Path(v)
        rows.append(("OK", f"{flag} = {v}") if p.exists()
                     else ("FAIL", f"{flag} path does not exist: {v}"))

    for flag, want in REQUIRED_VALUES.items():
        v = flagval(flag)
        if not v:
            extra = " (falsy --string-db skipped the ENTIRE GNN in Run 14)" if flag == "--string-db" else ""
            rows.append(("FAIL", f"{flag} missing/falsy{extra}"))
        elif want is not None and str(v) != want:
            rows.append(("WARN", f"{flag} = {v} (manifest expects {want})"))
        else:
            rows.append(("OK", f"{flag} = {v}"))

    for flag in REQUIRED_PRESENT_VALUE:
        rows.append(("OK", f"{flag} = {flagval(flag)}") if flagval(flag)
                     else ("FAIL", f"{flag} missing"))

    for flag in REQUIRED_FLAGS:
        rows.append(("OK", f"{flag} present") if flagval(flag)
                    else ("FAIL", f"{flag} absent (C3 falsifier required for an honest holdout)"))

    for flag in FORBIDDEN_FLAGS:
        if flagval(flag):
            rows.append(("FAIL", f"{flag} present -> diminishes the full model battery"))

    if n_train > 100_000 and not flagval(SCALE_SKIP):
        rows.append(("WARN", f"{SCALE_SKIP} absent at n_train={n_train:,}; --help says required >100k (RBF O(n^2)). "
                             f"Confirm the harness auto-skips, else this stalls/crashes."))

    for flag, key in ACK_OPTIONAL.items():
        if not flagval(flag) and key not in ack:
            rows.append(("ACK-NEEDED", f"{flag} omitted -> documented Run-9-bug default. "
                                       f"Re-run with --ack-omit {key} to acknowledge, or supply the path."))
    return rows


def emit(data_root: str) -> str:
    missing = [(f, Path(data_root) / rel) for f, rel in REQUIRED_PATHS.items()
               if not (Path(data_root) / rel).exists()]
    if missing:
        for f, p in missing:
            print(f"FAIL  {f}: not found at {p}", file=sys.stderr)
        sys.exit(2)
    parts = []
    for f, rel in REQUIRED_PATHS.items():
        parts.append(f'{f} {Path(data_root).joinpath(rel).as_posix()}')
    for f, v in REQUIRED_VALUES.items():
        parts.append(f"{f} {v}")
    parts.append("--skip-svm")
    parts.append("--unseen-gene-holdout")
    parts.append("--output outputs/run15_rich/full")
    return "python scripts/run_phase2_eval.py " + " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-flight gate for run_phase2_eval.py")
    ap.add_argument("--check")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--n-train", type=int, default=1_200_000)
    ap.add_argument("--ack-omit", default="")
    a = ap.parse_args()

    if a.emit:
        print(emit(a.data_root))
        return 0
    if not a.check:
        print("nothing to do: pass --check \"<command>\" or --emit", file=sys.stderr)
        return 2

    ack = {x.strip() for x in a.ack_omit.split(",") if x.strip()}
    ns = _parse_candidate(a.check)
    rows = validate(ns, a.data_root, a.n_train, ack)
    order = {"FAIL": 0, "ACK-NEEDED": 1, "WARN": 2, "OK": 3}
    for level, msg in sorted(rows, key=lambda r: order[r[0]]):
        print(f"  {level:<10} {msg}")
    fails = [r for r in rows if r[0] in ("FAIL", "ACK-NEEDED")]
    print(f"\n{'GATE BLOCKED' if fails else 'GATE PASSED'}: "
          f"{sum(1 for r in rows if r[0]=='FAIL')} fail, "
          f"{sum(1 for r in rows if r[0]=='ACK-NEEDED')} ack-needed, "
          f"{sum(1 for r in rows if r[0]=='WARN')} warn.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
