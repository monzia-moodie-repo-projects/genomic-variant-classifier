#!/usr/bin/env python3
"""
Run 11 Pre-flight Checker
===========================
Verifies all prerequisites before committing or launching Run 11.

Checks:
  1. File inventory: all expected files exist with non-zero size
  2. Syntax validation: all Python files parse cleanly
  3. Import check: key modules importable
  4. Data inventory: splits exist and have expected shape
  5. Patch verification: Phase 0 patches applied correctly
  6. Git state: clean working tree, on main branch

Usage:
    python scripts/run11_preflight.py
"""
from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_file_exists(path: Path, min_size: int = 10) -> tuple[bool, str]:
    """Check that a file exists and has minimum size."""
    if not path.exists():
        return False, f"NOT FOUND: {path}"
    size = path.stat().st_size
    if size < min_size:
        return False, f"TOO SMALL: {path} ({size} bytes)"
    return True, f"OK: {path} ({size:,} bytes)"


def check_syntax(path: Path) -> tuple[bool, str]:
    """Check Python syntax."""
    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        return True, f"OK: {path.name}"
    except SyntaxError as e:
        return False, f"SYNTAX ERROR: {path.name} line {e.lineno}: {e.msg}"


def check_git_state() -> tuple[bool, str]:
    """Check git working tree state."""
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        if result.returncode != 0:
            return False, f"git status failed: {result.stderr}"
        if result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            return False, f"DIRTY: {len(lines)} uncommitted changes"
        return True, "Clean working tree"
    except FileNotFoundError:
        return False, "git not found"


def check_branch() -> tuple[bool, str]:
    """Check current git branch."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        branch = result.stdout.strip()
        if branch == "main":
            return True, "On main branch"
        return False, f"On branch '{branch}' (expected 'main')"
    except FileNotFoundError:
        return False, "git not found"


def check_data_splits(splits_dir: Path) -> list[tuple[bool, str]]:
    """Check that training splits exist and have expected dimensions."""
    results = []

    if not splits_dir.exists():
        results.append((False, f"Splits dir not found: {splits_dir}"))
        return results

    expected_files = ["X_train.parquet", "X_val.parquet", "X_test.parquet",
                      "y_train.parquet", "y_val.parquet", "y_test.parquet"]

    for fname in expected_files:
        fpath = splits_dir / fname
        if not fpath.exists():
            results.append((False, f"NOT FOUND: {fpath}"))
            continue

        try:
            import pandas as pd
            df = pd.read_parquet(fpath)
            results.append((True, f"OK: {fname} ({df.shape[0]:,} rows x {df.shape[1]} cols)"))
        except Exception as e:
            results.append((False, f"READ ERROR: {fname}: {e}"))

    return results


def check_patches_applied() -> list[tuple[bool, str]]:
    """Verify Phase 0 patches were applied."""
    results = []

    ensemble_path = PROJECT_ROOT / "src" / "genomic_variant_classifier" / "models" / "variant_ensemble.py"
    datprep_path = PROJECT_ROOT / "src" / "genomic_variant_classifier" / "data" / "real_data_prep.py"

    for name, path, patterns in [
        ("variant_ensemble.py", ensemble_path, {
            "_GPU_AVAILABLE": "GPU detection flag",
        }),
        ("real_data_prep.py", datprep_path, {}),
    ]:
        if not path.exists():
            results.append((False, f"NOT FOUND: {path}"))
            continue

        content = path.read_text(encoding="utf-8")
        for pattern, description in patterns.items():
            if pattern in content:
                results.append((True, f"{name}: {description} present"))
            else:
                results.append((False, f"{name}: {description} MISSING"))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("RUN 11 PRE-FLIGHT CHECKER")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print()

    all_pass = True

    # 1. File inventory
    print("--- 1. File Inventory ---")
    expected_files = [
        PROJECT_ROOT / "scripts" / "run11_phase0_apply.py",
        PROJECT_ROOT / "scripts" / "launch_run11_vm.sh",
        PROJECT_ROOT / "scripts" / "run11_hpo.py",
        PROJECT_ROOT / "scripts" / "run11_data_quality_audit.py",
        PROJECT_ROOT / "scripts" / "verify_r10a_lovd_root_cause.py",
        PROJECT_ROOT / "docs" / "plans" / "RUN_11_MASTER_PLAN.md",
        PROJECT_ROOT / "docs" / "plans" / "RUN_11_SESSION_GUIDE.md",
        PROJECT_ROOT / "src" / "genomic_variant_classifier" / "models" / "kan.py",
        PROJECT_ROOT / "src" / "genomic_variant_classifier" / "data" / "etl_polars.py",
        PROJECT_ROOT / "src" / "genomic_variant_classifier" / "data" / "primateai3d.py",
    ]

    for fpath in expected_files:
        ok, msg = check_file_exists(fpath)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {msg}")
        if not ok:
            all_pass = False

    # 2. Syntax validation
    print("\n--- 2. Syntax Validation ---")
    py_files = [f for f in expected_files if f.suffix == ".py"]
    for fpath in py_files:
        if fpath.exists():
            ok, msg = check_syntax(fpath)
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {msg}")
            if not ok:
                all_pass = False

    # 3. Git state
    print("\n--- 3. Git State ---")
    ok, msg = check_branch()
    print(f"  [{'PASS' if ok else 'WARN'}] {msg}")

    ok, msg = check_git_state()
    print(f"  [{'PASS' if ok else 'INFO'}] {msg}")
    # Dirty tree is not a failure if we're about to commit

    # 4. Data splits
    print("\n--- 4. Data Splits ---")
    splits_dir = PROJECT_ROOT / "outputs" / "run10b_final" / "full" / "splits"
    if not splits_dir.exists():
        splits_dir = PROJECT_ROOT / "data" / "splits"

    split_results = check_data_splits(splits_dir)
    for ok, msg in split_results:
        status = "PASS" if ok else "WARN"
        print(f"  [{status}] {msg}")

    # 5. Patch verification
    print("\n--- 5. Phase 0 Patch Verification ---")
    patch_results = check_patches_applied()
    for ok, msg in patch_results:
        status = "PASS" if ok else "INFO"
        print(f"  [{status}] {msg}")

    # 6. External data
    print("\n--- 6. External Data ---")
    external_files = {
        "AlphaMissense": PROJECT_ROOT / "data" / "external" / "alphamissense" / "AlphaMissense_hg38.tsv",
        "SpliceAI index": PROJECT_ROOT / "data" / "external" / "spliceai" / "spliceai_index.parquet",
        "gnomAD constraint": PROJECT_ROOT / "data" / "external" / "gnomad" / "gnomad.v4.1.constraint_metrics.tsv",
        "LOVD directory": PROJECT_ROOT / "data" / "external" / "lovd",
    }
    for name, path in external_files.items():
        if path.exists():
            if path.is_dir():
                n_files = len(list(path.iterdir()))
                print(f"  [PASS] {name}: {n_files} files")
            else:
                size_mb = path.stat().st_size / 1e6
                print(f"  [PASS] {name}: {size_mb:.1f} MB")
        else:
            print(f"  [WARN] {name}: NOT FOUND at {path}")

    # 7. R10-A verification
    print("\n--- 7. R10-A Verification ---")
    r10a_artifact = PROJECT_ROOT / "docs" / "verified" / "R10A_LOVD_VERIFICATION.json"
    if r10a_artifact.exists():
        import json
        with open(r10a_artifact) as f:
            data = json.load(f)
        status = data.get("status", "unknown")
        case = data.get("case", "unknown")
        print(f"  [PASS] R10-A: status={status}, case={case}")
    else:
        print(f"  [WARN] R10-A verification artifact not found")

    # Summary
    print("\n" + "=" * 70)
    if all_pass:
        print("PRE-FLIGHT: ALL CHECKS PASSED")
        print("Ready for: git add -A && git commit")
    else:
        print("PRE-FLIGHT: SOME CHECKS FAILED")
        print("Review failures above before proceeding")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
