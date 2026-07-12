"""collapse_engineer_features.py -- single source of truth for the feature matrix.
2026-07-11. Guarded, reversible, idempotent installer.

WHAT IT DOES
------------
Replaces the ~435-line body of `DataPrepPipeline._engineer_features`
(src/genomic_variant_classifier/data/real_data_prep.py) with a delegation to
`variant_ensemble.engineer_features`, leaving exactly ONE implementation of the
feature matrix in the codebase.

WHY
---
The project carried TWO independent, hand-synced implementations:

    variant_ensemble.engineer_features    <- what the 5-stage CORRECTNESS HARNESS
                                             validates (correctness_harness.py:59)
    DataPrepPipeline._engineer_features   <- what the TRAINING PIPELINE actually runs

The harness therefore audited a code path the pipeline never executed. A silent zero,
a wrong clip, or a truncating cast introduced in the pipeline's own block was
STRUCTURALLY INVISIBLE to the gate built to catch exactly that. The gate could read
green while the training matrix was wrong.

The codebase said so itself, and asked to be kept in sync by hand:
    variant_ensemble.py:121  "(65 features -- must match DataPrepPipeline._engineer_features)"
    variant_ensemble.py:340  "Mirrors DataPrepPipeline._engineer_features() ..."
That comment says 65. The contract actually holds 97. The hand-sync had already lapsed.

EVIDENCE THIS IS SAFE
---------------------
scripts/prove_engineer_features_equivalence.py, 2026-07-11:
  117 comparisons, ZERO divergences -- exact on column set, column ORDER, dtype and
  values (NaN positions included, no float coercion). Including:
    C2  a minimal frame with EVERY connector column absent -> forces every df.get default
    C4  16 integral columns made fractional -- among them clingen_validity_score, the
        exact column of INCIDENT_2026-05-30_clingen-int-truncation
    C5  NaN injection, C6 extremes, C7 empty frame
  Log: outputs/engineer_equiv_2026-07-11b.log

WHAT THE DELEGATION PRESERVES (deliberately)
--------------------------------------------
  * The LOCAL import of variant_ensemble. real_data_prep must NOT import the heavy ML
    stack at module import time -- tests assert the package imports without sklearn/torch
    (test_orchestrator_lazy_registry, test_evaluation_metrics). Keep it inside the method.
  * `.reset_index(drop=True)` on the returned frame. The old method ended with it; the
    equivalence proof compared values via .to_numpy() and therefore never checked the
    INDEX. Dropping it could silently misalign downstream joins/splits.

WHAT IT UPGRADES
----------------
The old fail-loud guard compared only the feature COUNT against
EXPECTED_TABULAR_FEATURE_COUNT. A count is a weak proxy: it cannot catch different names
at the same count, a different column ORDER (an estimator fed a bare ndarray trusts
position implicitly), or different values. Its own comment concedes the 88-vs-91 R13
drift "went unnoticed for a full 13-hour run". The delegation guards NAME AND ORDER
against TABULAR_FEATURES.

SAFETY
------
  * refuses if the method cannot be located by AST
  * refuses (cleanly, exit 0) if already collapsed -- idempotent
  * backs the file up before writing
  * ast.parse()es the result; on any syntax error it RESTORES the backup and aborts
  * prints the exact revert command

    python scripts\\collapse_engineer_features.py            # dry run, shows the plan
    python scripts\\collapse_engineer_features.py --apply    # writes
"""
from __future__ import annotations

import argparse
import ast
import shutil
import sys
from datetime import date
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/real_data_prep.py")
MARKER = "SINGLE SOURCE OF TRUTH (collapsed 2026-07-11)"

REPLACEMENT = '''    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the tabular feature matrix for training.

        SINGLE SOURCE OF TRUTH (collapsed 2026-07-11).

        This method used to be a ~435-line SECOND implementation of feature
        engineering, hand-kept in sync with variant_ensemble.engineer_features via
        comments that said "must match" and "Mirrors". That sync had already lapsed:
        the header in variant_ensemble.py claimed 65 features while the contract held
        97, and -- worse -- the five-stage correctness harness imports
        engineer_features from variant_ensemble, so the gate validated a code path the
        training pipeline never ran. A silent zero or a truncating cast introduced
        here was structurally invisible to the gate designed to catch it.

        The two were proved equivalent before this collapse:
        scripts/prove_engineer_features_equivalence.py -- 117 comparisons, zero
        divergences, exact on column set, column ORDER, dtype and values (NaN positions
        included). It included a minimal frame that forces every df.get default, and 16
        integral columns made fractional -- among them clingen_validity_score, the exact
        column of INCIDENT_2026-05-30_clingen-int-truncation.
        Log: outputs/engineer_equiv_2026-07-11b.log

        The pipeline now trains on EXACTLY the matrix the harness validates.
        """
        # LOCAL import, deliberately. real_data_prep must not pull the heavy ML stack in
        # at module-import time -- the suite asserts the package imports with sklearn and
        # torch blocked (test_orchestrator_lazy_registry, test_evaluation_metrics).
        from genomic_variant_classifier.models.variant_ensemble import (
            TABULAR_FEATURES as _CONTRACT,
            engineer_features as _engineer,
        )

        feats = _engineer(df)

        # Fail-loud CONTRACT guard. The guard this replaces compared only the feature
        # COUNT, and its own comment conceded that is how the 88-vs-91 R13 drift "went
        # unnoticed for a full 13-hour run". A count cannot catch different names at the
        # same count, nor a different column ORDER -- and order matters: an estimator
        # fitted on a named DataFrame but fed a bare ndarray trusts position implicitly
        # (cf. the standing "X does not have valid feature names" LightGBM warning).
        _expected = list(_CONTRACT)
        _actual = list(feats.columns)
        if _actual != _expected:
            _missing = [c for c in _expected if c not in _actual]
            _extra = [c for c in _actual if c not in _expected]
            _misordered = (not _missing and not _extra)
            raise ValueError(
                "engineer_features output violates the TABULAR_FEATURES contract -- "
                "refusing to train on a wrong matrix.\\n"
                f"  expected {len(_expected)} columns, got {len(_actual)}\\n"
                f"  MISSING from output : {_missing}\\n"
                f"  UNEXPECTED in output: {_extra}\\n"
                f"  ORDER differs only  : {_misordered}"
            )

        # reset_index preserved from the original method. The equivalence proof compared
        # values via .to_numpy() and therefore never checked the INDEX; dropping this
        # could silently misalign downstream joins and splits.
        return feats.reset_index(drop=True)
'''


def _find_method(src: str) -> tuple[int, int]:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DataPrepPipeline":
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name == "_engineer_features":
                    return m.lineno, m.end_lineno  # 1-based, inclusive
    raise SystemExit("ABORT: DataPrepPipeline._engineer_features not found by AST.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the change (default: dry run)")
    args = ap.parse_args()

    if not TARGET.is_file():
        raise SystemExit(f"ABORT: {TARGET} not found. Run from the repository root.")

    src = TARGET.read_text(encoding="utf-8")

    if MARKER in src:
        print("OK (idempotent): already collapsed -- no changes needed.")
        return 0

    start, end = _find_method(src)
    lines = src.splitlines(keepends=True)
    n_removed = end - start + 1

    print(f"  target        : {TARGET}")
    print(f"  method span   : lines {start}..{end}  ({n_removed} lines)")
    print(f"  replacing with: a delegation to variant_ensemble.engineer_features "
          f"({len(REPLACEMENT.splitlines())} lines)")
    print(f"  net           : -{n_removed - len(REPLACEMENT.splitlines())} lines")

    if not args.apply:
        print("\nDRY RUN. Re-run with --apply to write.")
        return 0

    backup = TARGET.with_suffix(f".py.bak_{date.today().isoformat()}")
    shutil.copy2(TARGET, backup)
    print(f"\n  backup written: {backup}")

    new_src = "".join(lines[: start - 1]) + REPLACEMENT + "".join(lines[end:])

    try:
        ast.parse(new_src)
    except SyntaxError as exc:
        shutil.copy2(backup, TARGET)
        raise SystemExit(
            f"ABORT: the rewritten file does not parse ({exc}). "
            f"Original RESTORED from {backup}. No changes made."
        )

    TARGET.write_text(new_src, encoding="utf-8", newline="")
    print(f"  written       : {TARGET}")
    print()
    print("  NEXT (both are gates, run them):")
    print("    python scripts\\prove_engineer_features_equivalence.py   # must still pass (now trivially)")
    print("    python -m pytest -q")
    print()
    print(f"  REVERT: copy /Y {backup} {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
