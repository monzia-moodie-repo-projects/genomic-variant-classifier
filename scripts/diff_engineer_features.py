"""diff_engineer_features.py -- the single-source-of-truth audit (2026-07-11).

THE PROBLEM
-----------
The feature matrix has TWO independent, hand-maintained implementations:

    variant_ensemble.engineer_features          <- what the 5-stage CORRECTNESS HARNESS
                                                   validates (correctness_harness.py:59),
                                                   what api/pipeline.py serves, and what
                                                   the unit tests exercise.

    DataPrepPipeline._engineer_features         <- what the DATA-PREP / TRAINING PIPELINE
    (real_data_prep.py:1260, ~440 lines)           actually runs to build the matrix the
                                                   models are trained on.

The second does not delegate to the first. It rebuilds `feats` from scratch, column by
column. The codebase says so in its own words:

    variant_ensemble.py:121  "Feature definitions (65 features -- must match
                              DataPrepPipeline._engineer_features)"
    variant_ensemble.py:340  "Mirrors DataPrepPipeline._engineer_features() ..."
    scripts/install_docs_close_cnn_rna.py:34
                             "... registered in TABULAR_FEATURES, BOTH
                              _engineer_features blocks"

Two copies, kept in sync BY HAND. That is a permanent drift generator.

WHY IT MATTERS MORE THAN STYLE
------------------------------
Stage 5's zero-audit -- and the whole G1 pre-flight gate that rests on it -- imports
engineer_features from variant_ensemble. So the gate validates ONLY that copy. The code
that actually builds the training matrix is NEVER exercised by the harness. A silent
zero, a wrong clip, or a truncating cast introduced in the pipeline's own block is
STRUCTURALLY INVISIBLE to the gate built to catch exactly that class of defect.
The gate can be green while the training matrix is wrong. That is the Run-15 silent
zero, one level up.

Drift is already provable in the documentation: the header says 65 features;
TABULAR_FEATURES actually holds 97 (verified 2026-07-11). The "must match" contract has
demonstrably not been maintained. Whether the MATRICES differ is what this script
settles -- and until it runs, divergence must not be asserted in either direction.

    python scripts\\diff_engineer_features.py

READ THE VERDICT, THEN ADJUDICATE
---------------------------------
If DIVERGED, every diverging column is a live defect in EITHER the training matrix OR
the harness's model of it. Each must be adjudicated on its merits BEFORE the two are
collapsed into one implementation -- otherwise the merge silently changes the training
matrix underneath already-trained models. Do not merge blind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from genomic_variant_classifier.agent_layer.harness.correctness_harness import (
    build_reference_slice,
)
from genomic_variant_classifier.data.real_data_prep import DataPrepConfig, DataPrepPipeline
from genomic_variant_classifier.models.variant_ensemble import (
    TABULAR_FEATURES,
    engineer_features,
)

RULE = "=" * 78


def main() -> int:
    df = build_reference_slice()

    A = engineer_features(df)                                      # HARNESS-validated
    B = DataPrepPipeline(DataPrepConfig())._engineer_features(df)  # PIPELINE-trained

    a, b = list(A.columns), list(B.columns)
    sa, sb = set(a), set(b)
    contract = list(TABULAR_FEATURES)

    print(RULE)
    print("COLUMN COUNTS")
    print(RULE)
    print(f"  TABULAR_FEATURES (the declared contract) : {len(contract)}")
    print(f"  variant_ensemble.engineer_features       : {len(a)}   [harness validates this]")
    print(f"  DataPrepPipeline._engineer_features      : {len(b)}   [pipeline TRAINS on this]")

    print()
    print(RULE)
    print("SET DIFFERENCES")
    print(RULE)
    only_a = sorted(sa - sb)
    only_b = sorted(sb - sa)
    print(f"  ONLY in variant_ensemble  ({len(only_a)})  -- harness audits it; pipeline never builds it:")
    for c in only_a:
        print(f"      - {c}")
    if not only_a:
        print("      (none)")
    print(f"  ONLY in real_data_prep    ({len(only_b)})  -- pipeline builds it; harness NEVER audits it:")
    for c in only_b:
        print(f"      + {c}")
    if not only_b:
        print("      (none)")

    print()
    print(RULE)
    print("CONTRACT COMPLIANCE (vs TABULAR_FEATURES)")
    print(RULE)
    miss_a = [c for c in contract if c not in sa]
    miss_b = [c for c in contract if c not in sb]
    print(f"  declared but MISSING from variant_ensemble ({len(miss_a)}): {miss_a or '(none)'}")
    print(f"  declared but MISSING from real_data_prep   ({len(miss_b)}): {miss_b or '(none)'}")

    print()
    print(RULE)
    print("SHARED COLUMNS -- NUMERICALLY DIFFERENT (same name, different values)")
    print(RULE)
    shared = [c for c in a if c in sb]          # keep variant_ensemble's order
    diffs = []
    for c in shared:
        x = pd.to_numeric(A[c], errors="coerce").fillna(0).to_numpy(dtype=float)
        y = pd.to_numeric(B[c], errors="coerce").fillna(0).to_numpy(dtype=float)
        if x.shape != y.shape or not np.allclose(x, y, equal_nan=True):
            n_bad = int((~np.isclose(x, y, equal_nan=True)).sum()) if x.shape == y.shape else -1
            max_abs = float(np.nanmax(np.abs(x - y))) if x.shape == y.shape else float("nan")
            diffs.append((c, n_bad, max_abs, x[:3], y[:3]))

    if not diffs:
        print("  (none -- every shared column is numerically identical)")
    for c, n_bad, max_abs, xs, ys in diffs:
        print(f"  ! {c}")
        print(f"        rows differing : {n_bad} / {len(A)}")
        print(f"        max |A - B|    : {max_abs:.6g}")
        print(f"        variant_ensemble[:3] : {np.round(xs, 6)}")
        print(f"        real_data_prep[:3]   : {np.round(ys, 6)}")

    print()
    print(RULE)
    print("COLUMN ORDER")
    print(RULE)
    # Order matters: models fitted on a named DataFrame but predicted on a bare ndarray
    # trust positional order implicitly (cf. the LGBMClassifier feature-name warning).
    common_order_a = [c for c in a if c in sb]
    common_order_b = [c for c in b if c in sa]
    if common_order_a == common_order_b:
        print("  shared columns appear in the SAME order in both. OK.")
    else:
        print("  *** SHARED COLUMNS ARE IN A DIFFERENT ORDER. ***")
        print("  This is silently dangerous: an estimator fitted on one ordering and fed the")
        print("  other as a bare ndarray produces wrong predictions with NO error.")
        for i, (ca, cb) in enumerate(zip(common_order_a, common_order_b)):
            if ca != cb:
                print(f"      first divergence at position {i}: "
                      f"variant_ensemble={ca!r}  real_data_prep={cb!r}")
                break

    identical = not (sa ^ sb) and not diffs and common_order_a == common_order_b
    print()
    print(RULE)
    print("VERDICT:", "IDENTICAL -- safe to collapse to one implementation."
          if identical else
          "DIVERGED -- adjudicate every item above BEFORE collapsing. Do not merge blind.")
    print(RULE)
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
