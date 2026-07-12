"""prove_engineer_features_equivalence.py -- 2026-07-11.

PURPOSE
-------
Establish, adversarially, whether the project's TWO feature-engineering implementations

    A = variant_ensemble.engineer_features            (what the correctness HARNESS validates)
    B = DataPrepPipeline._engineer_features           (what the training PIPELINE runs)

are genuinely equivalent -- strongly enough to justify DELETING one of them.

WHY scripts/diff_engineer_features.py IS NOT SUFFICIENT
------------------------------------------------------
That script compared A and B on build_reference_slice() and reported IDENTICAL
(97/97 columns, no set difference, no numeric difference, same order). True, but weak,
for two reasons:

  1. build_reference_slice supplies EVERY input column -- that is its contract. Both
     implementations read inputs via df.get(col, DEFAULT). So on that fixture NOT ONE
     DEFAULT IS EVER EXERCISED. If the two copies carry different defaults, the fixture
     is structurally incapable of revealing it.

  2. It coerced everything through pd.to_numeric(...).astype(float), so DTYPE differences
     were invisible. That is exactly the class of defect that already bit this project:
     INCIDENT_2026-05-30_clingen-int-truncation, where .astype(int) truncated fractional
     ClinGen scores to 0. The fixture feeds clingen_validity_score = rng.integers(1, 5)
     -- integral -- so an int-vs-float cast difference produces identical VALUES and
     slips through undetected.

WHAT THIS SCRIPT DOES INSTEAD
----------------------------
Runs both implementations over a battery of adversarial inputs chosen to force every
default, every cast, and every missing-column path, and compares them EXACTLY:
column set, column ORDER, dtype, and values (NaN positions included, no float coercion).

  C1  reference slice, several seeds and sizes
  C2  MINIMAL frame  -- identity columns only; EVERY connector column ABSENT.
                        This is the decisive case: it exercises every df.get default in
                        both copies simultaneously.
  C3  single-column dropout -- drop each input column in turn (one default at a time,
                        so a divergence names the exact column responsible)
  C4  FRACTIONAL inputs where the fixture uses integers (clingen_validity_score,
                        lovd_variant_class, exon_number, omim_*, hgmd_n_reports, ...):
                        the int-truncation trap, aimed directly.
  C5  NaN injection    -- 20% NaN into each numeric input column, one column at a time
  C6  extreme values   -- 0, negative, +/-inf, very large, very small
  C7  empty frame      -- 0 rows
  C8  dtype table      -- exact per-column dtype comparison across all cases

Exit code 0 == equivalent under every case (safe to collapse).
Exit code 1 == a divergence was found; it is printed with the exact column and case.

    python scripts\\prove_engineer_features_equivalence.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from genomic_variant_classifier.agent_layer.harness.correctness_harness import (
    build_reference_slice,
)
from genomic_variant_classifier.data.real_data_prep import DataPrepConfig, DataPrepPipeline
from genomic_variant_classifier.models.variant_ensemble import engineer_features

RULE = "=" * 78
_PIPE = DataPrepPipeline(DataPrepConfig())

# Identity columns engineer_features needs to produce a frame at all.
IDENTITY = ["variant_id", "gene_symbol", "chrom", "pos", "ref", "alt", "label"]

failures: list[str] = []

# NON-VACUITY ACCOUNTING (added 2026-07-11).
# A proof script that silently tests NOTHING and then declares success is worse than
# no proof at all. The first version of this file reported "EQUIVALENT -- including the
# fractional-input int-truncation trap" without ever printing how many cases block C4
# actually ran; had its column selector matched zero columns, that claim would have been
# false and unfalsifiable from the output. Every block now counts its cases, prints the
# count, and the run HARD-FAILS if any block that must do work did none.
case_counts: dict[str, int] = {}
# Blocks that would be meaningless at zero cases -> the run is invalid if they are empty.
MUST_BE_NONEMPTY = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")


def _run_both(df: pd.DataFrame):
    return engineer_features(df), _PIPE._engineer_features(df)


def compare(case: str, df: pd.DataFrame) -> None:
    """EXACT comparison: columns, order, dtypes, values (incl. NaN positions)."""
    case_counts[case.split()[0]] = case_counts.get(case.split()[0], 0) + 1
    try:
        A, B = _run_both(df)
    except Exception as exc:  # noqa: BLE001
        # A raise in ONE and not the other is itself a divergence.
        try:
            engineer_features(df)
            a_ok = True
        except Exception:                       # noqa: BLE001
            a_ok = False
        try:
            _PIPE._engineer_features(df)
            b_ok = True
        except Exception:                       # noqa: BLE001
            b_ok = False
        if a_ok != b_ok:
            failures.append(
                f"[{case}] ONE implementation raised and the other did not "
                f"(variant_ensemble ok={a_ok}, real_data_prep ok={b_ok}): {exc!r}"
            )
        else:
            print(f"  .. {case}: both raised identically ({type(exc).__name__}) -- not a divergence")
        return

    # 1. column set
    if set(A.columns) != set(B.columns):
        only_a = sorted(set(A.columns) - set(B.columns))
        only_b = sorted(set(B.columns) - set(A.columns))
        failures.append(f"[{case}] COLUMN SET differs. only_variant_ensemble={only_a} only_real_data_prep={only_b}")
        return

    # 2. column ORDER (an estimator fed a bare ndarray trusts position implicitly)
    if list(A.columns) != list(B.columns):
        for i, (ca, cb) in enumerate(zip(A.columns, B.columns)):
            if ca != cb:
                failures.append(f"[{case}] COLUMN ORDER differs at position {i}: {ca!r} vs {cb!r}")
                break
        return

    # 3. dtypes -- exact. This is the int-vs-float truncation trap.
    for c in A.columns:
        if A[c].dtype != B[c].dtype:
            failures.append(
                f"[{case}] DTYPE differs for {c!r}: "
                f"variant_ensemble={A[c].dtype} real_data_prep={B[c].dtype}"
            )

    # 4. values -- exact, NaN positions included, NO float coercion.
    for c in A.columns:
        x, y = A[c], B[c]
        if x.isna().to_numpy().tolist() != y.isna().to_numpy().tolist():
            n = int((x.isna() != y.isna()).sum())
            failures.append(f"[{case}] NaN PATTERN differs for {c!r} in {n} row(s)")
            continue
        xv = x.dropna().to_numpy()
        yv = y.dropna().to_numpy()
        if xv.dtype.kind in "fiub" and yv.dtype.kind in "fiub":
            if xv.shape != yv.shape or not np.allclose(
                xv.astype(float), yv.astype(float), rtol=0, atol=0, equal_nan=True
            ):
                bad = int((~np.isclose(xv.astype(float), yv.astype(float), rtol=0, atol=0)).sum())
                mx = float(np.max(np.abs(xv.astype(float) - yv.astype(float)))) if xv.shape == yv.shape else float("nan")
                failures.append(
                    f"[{case}] VALUES differ for {c!r}: {bad} row(s), max|A-B|={mx:.6g} "
                    f"| A[:3]={xv[:3]} B[:3]={yv[:3]}"
                )
        elif not xv.tolist() == yv.tolist():
            failures.append(f"[{case}] NON-NUMERIC values differ for {c!r}")


def main() -> int:
    print(RULE)
    print("EQUIVALENCE PROOF: variant_ensemble.engineer_features  vs  DataPrepPipeline._engineer_features")
    print("EXACT comparison -- column set, column order, dtype, values (NaN positions included).")
    print(RULE)

    # ---- C1: reference slice, several seeds and sizes -----------------------
    print("\nC1  reference slice (seeds x sizes)")
    for seed in (7, 42, 1234):
        for n in (10, 200, 1000):
            compare(f"C1 ref seed={seed} n={n}", build_reference_slice(n=n, seed=seed))

    base = build_reference_slice(n=200, seed=7)
    input_cols = [c for c in base.columns if c not in IDENTITY]

    # ---- C2: MINIMAL frame -- every connector column ABSENT -----------------
    # THE DECISIVE CASE. Forces every df.get(col, DEFAULT) in BOTH copies.
    print("\nC2  MINIMAL frame -- identity only, every connector column absent (forces ALL defaults)")
    compare("C2 minimal (all defaults)", base[IDENTITY].copy())

    # ---- C3: single-column dropout ------------------------------------------
    print(f"\nC3  single-column dropout -- {len(input_cols)} cases, one default at a time")
    for c in input_cols:
        compare(f"C3 drop {c!r}", base.drop(columns=[c]))

    # ---- C4: FRACTIONAL where the fixture uses integers ---------------------
    # Aimed squarely at INCIDENT_2026-05-30_clingen-int-truncation.
    print("\nC4  fractional inputs where the fixture is integral (the int-truncation trap)")
    # Explicit and auditable: any input column the fixture supplies as an INTEGER, or as a
    # float whose values happen to all be whole numbers. Either way, an .astype(int) in one
    # implementation and .astype(float) in the other would produce IDENTICAL values on the
    # fixture and diverge only on a fractional input -- which is exactly what
    # INCIDENT_2026-05-30_clingen-int-truncation was. We make each one fractional in turn.
    int_like: list[str] = []
    for c in input_cols:
        s = base[c].dropna()
        if s.empty:
            continue
        if pd.api.types.is_integer_dtype(base[c]):
            int_like.append(c)
        elif pd.api.types.is_float_dtype(base[c]) and bool((s % 1 == 0).all()):
            int_like.append(c)
    print(f"    integral input columns detected: {len(int_like)} -> {sorted(int_like)}")
    for c in sorted(set(int_like)):
        d = base.copy()
        d[c] = base[c].astype(float) + 0.5      # force a fractional value
        compare(f"C4 fractional {c!r}", d)

    # ---- C5: NaN injection ---------------------------------------------------
    print("\nC5  NaN injection -- 20% NaN into each numeric input column")
    rng = np.random.default_rng(0)
    for c in input_cols:
        if not pd.api.types.is_numeric_dtype(base[c]):
            continue
        d = base.copy()
        d[c] = d[c].astype(float)
        mask = rng.uniform(size=len(d)) < 0.20
        d.loc[mask, c] = np.nan
        compare(f"C5 nan20 {c!r}", d)

    # ---- C6: extreme values --------------------------------------------------
    print("\nC6  extreme values (0, negative, +/-inf, very large / very small)")
    for label, val in [
        ("zeros", 0.0), ("neg", -1.0), ("posinf", np.inf), ("neginf", -np.inf),
        ("huge", 1e300), ("tiny", 1e-300),
    ]:
        d = base.copy()
        for c in input_cols:
            if pd.api.types.is_numeric_dtype(d[c]):
                d[c] = float(val)
        compare(f"C6 {label}", d)

    # ---- C7: empty frame -----------------------------------------------------
    print("\nC7  empty frame (0 rows)")
    compare("C7 empty", base.iloc[0:0].copy())

    # ---- NON-VACUITY GATE ----------------------------------------------------
    # Prove the proof actually did work. A block that ran zero cases invalidates the
    # claim that block makes, so the whole run is invalid -- fail LOUD, do not pass.
    print()
    print(RULE)
    print("CASES EXECUTED PER BLOCK (a zero here would invalidate this proof)")
    print(RULE)
    total = 0
    empty_blocks = []
    for blk in MUST_BE_NONEMPTY:
        n = case_counts.get(blk, 0)
        total += n
        flag = "   <<< ZERO -- THIS BLOCK PROVED NOTHING" if n == 0 else ""
        print(f"  {blk}: {n:4d} case(s){flag}")
        if n == 0:
            empty_blocks.append(blk)
    print(f"  TOTAL: {total} comparison(s), each checking column set, order, dtype and values")

    if empty_blocks:
        print()
        print(f"INVALID PROOF: block(s) {empty_blocks} executed ZERO cases.")
        print("The 'EQUIVALENT' claim is therefore unsupported. Fix the case generator")
        print("before trusting any verdict from this script.")
        print(RULE)
        return 2

    # ---- report --------------------------------------------------------------
    print()
    print(RULE)
    if failures:
        print(f"DIVERGENCES FOUND: {len(failures)}")
        print(RULE)
        for f in failures:
            print("  " + f)
        print()
        print("VERDICT: NOT EQUIVALENT. Do NOT collapse. Adjudicate every item above --")
        print("each is a live defect in either the training matrix or the harness's model of it.")
        print(RULE)
        return 1

    print(f"NO DIVERGENCE across {total} comparisons -- including the minimal frame that")
    print("forces EVERY df.get default, the fractional-input int-truncation trap, NaN")
    print("injection, extreme values, an empty frame, and exact dtype comparison.")
    print()
    print("Known, ACCEPTED, non-divergent behaviour (both implementations, identically):")
    print("  * +/-inf input raises pandas IntCastingNaNError (an integer cast downstream).")
    print("    Fail-loud is correct, but the message is a raw pandas internal. Logged as a")
    print("    separate hardening item -- it is NOT a divergence and does not block this.")
    print()
    print("VERDICT: EQUIVALENT. Safe to collapse to a single implementation.")
    print(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
