#!/usr/bin/env python3
"""GO/NO-GO pre-flight for the clean-baseline run (existing code, clean cohort).

Verifies every hard gate before any VM spend:
  1. clean cohort present, 0 null alleles, 0 duplicate variant_id
  2. cohort guard (_assert_clean_cohort) present EXACTLY once in real_data_prep.py
     (catches a non-applied guard AND an accidental double-apply)
  3. STRING DB links + info present and non-empty (GNN can run)
  4. cohort guard unit-test file present
Plus an informational git-status line. Exit 0 = GO, 1 = NO-GO.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import pandas as pd

CLEAN = "data/processed/clinvar_grch38_clean.parquet"
RDP = "src/genomic_variant_classifier/data/real_data_prep.py"
STRING_DIR = "data/external/string"
LINKS_GLOB = "*protein.links.detailed*.txt.gz"
INFO_GLOB = "*protein.info*.txt.gz"
MIN_LINKS = 10_000_000
MIN_INFO = 100_000
GUARD_TEST = "tests/unit/test_cohort_guard.py"


def check_clean_cohort() -> tuple[bool, str]:
    p = Path(CLEAN)
    if not p.exists():
        return False, f"missing {CLEAN}"
    d = pd.read_parquet(p, columns=["ref", "alt", "variant_id"])
    n_null = int(d["ref"].isna().sum() + d["alt"].isna().sum())
    n_dup = int(d["variant_id"].duplicated().sum())
    return (n_null == 0 and n_dup == 0), f"rows={len(d):,} null={n_null} dup={n_dup}"


def check_guard_once() -> tuple[bool, str]:
    p = Path(RDP)
    if not p.exists():
        return False, f"missing {RDP}"
    n = p.read_text(encoding="utf-8").count("def _assert_clean_cohort")
    return n == 1, f"_assert_clean_cohort definitions={n} (want exactly 1)"


def check_string() -> tuple[bool, str]:
    d = Path(STRING_DIR)
    if not d.is_dir():
        return False, f"missing dir {STRING_DIR}"
    links = sorted(d.glob(LINKS_GLOB))
    info = sorted(d.glob(INFO_GLOB))
    if not links:
        return False, f"no links file ({LINKS_GLOB})"
    if not info:
        return False, f"no info file ({INFO_GLOB})"
    ls, isz = links[0].stat().st_size, info[0].stat().st_size
    return (ls >= MIN_LINKS and isz >= MIN_INFO), f"links={ls:,}B info={isz:,}B"


def check_guard_test() -> tuple[bool, str]:
    return Path(GUARD_TEST).exists(), GUARD_TEST


def git_status() -> str:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=20)
        n = len([ln for ln in out.stdout.splitlines() if ln.strip()])
        return f"{n} uncommitted path(s)" if n else "clean"
    except Exception as e:  # noqa: BLE001
        return f"git unavailable ({e})"


def main() -> int:
    checks = [
        ("clean cohort (0 null / 0 dup)", *check_clean_cohort()),
        ("cohort guard present exactly once", *check_guard_once()),
        ("STRING DB present (GNN gate)", *check_string()),
        ("cohort guard test file present", *check_guard_test()),
    ]
    print("== RUN-15 BASELINE PRE-FLIGHT ==")
    hard_ok = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        hard_ok = hard_ok and ok
    print(f"  [INFO] git working tree: {git_status()}")
    print(f"  [INFO] launch must use: --clinvar {CLEAN} --string-db auto  (and NOT --skip-cnn)")
    print("\nVERDICT:", "GO" if hard_ok else "NO-GO (resolve FAILs above before any VM spend)")
    return 0 if hard_ok else 1


if __name__ == "__main__":
    sys.exit(main())
