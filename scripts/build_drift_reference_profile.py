#!/usr/bin/env python3
"""Build the AGGREGATE-ONLY drift reference profile from the raw reference matrix.

Created 2026-07-13 (roadmap 6.20).

RUN THIS WHERE THE RAW COHORT LIVES. The output is committable; the input is not.

    python scripts/build_drift_reference_profile.py \
        --reference outputs/run15_rerun_report/full/splits/X_train.parquet \
        --out       data/reference/drift/run15_reference_profile.json

WHY
---
The scheduled drift monitor needs a reference distribution. It used to try to download the
23.8 MB `X_train.parquet` from Google Drive -- a download that was never implemented, so the
workflow reported "no drift" every month having never looked at anything (roadmap 6.20).

The raw matrix stays here. What ships is a HISTOGRAM.

NOTE -- a correction, recorded because it was nearly baked into the repository:
An earlier draft of this script claimed X_train could not be published because dbNSFP is
`tier: controlled` / "LICENSED (paid)". That is FALSE. The "LICENSED (paid)" note at
data_manifest.yaml:286 belongs to **hgmd**. dbNSFP is `tier: academic`,
`class: public_redownloadable` (line 86). The controlled tier is omim, hgmd, cosmic, tcga,
topmed -- and all four controlled-tier columns in X_train (omim_n_diseases,
omim_is_autosomal_dominant, hgmd_is_disease_mutation, hgmd_n_reports) are CONSTANT ZERO,
never populated, carrying no licensed information at all.

The profile is still the right artifact -- 1.4 MB of histograms, committable, no credentials,
no cloud, and it redistributes no per-variant annotation from any source. It makes the
licensing question moot instead of answering it under pressure. But the reason has to be the
true one.

WHAT THE PROFILE CONTAINS
-------------------------
Per feature: the 1st/99th percentile, ten bin COUNTS, a quantile grid, mean, standard
deviation, and the finite-row count. That is everything `DriftDetector._psi` takes from the
reference -- so the Population Stability Index is reproduced EXACTLY, bit-for-bit, and the
per-feature drift action (which depends on PSI alone) is unchanged.

It contains NO variant. NO identifier. NO per-variant annotation value. It is a few hundred
kilobytes, it commits beside the code, it needs no credentials and no cloud, and it lets the
drift monitor actually run -- monthly, on a hosted runner, for the first time.

WHAT IT CANNOT DO
-----------------
The Maximum Mean Discrepancy and Szekely-Rizzo energy tests are MULTIVARIATE permutation tests
and need real reference samples. From a profile they are NOT RUN -- and are reported as not
run, never as passing. `DriftDetector.check` escalates on `mmd_pvalue < 0.001`; silently
substituting a passing p-value there would permanently disarm that escalation while appearing
to work, which is precisely the class of defect this work exists to end.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("build_drift_reference_profile")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", type=Path, required=True,
                   help="Raw reference feature matrix (parquet). NOT committed; stays local.")
    p.add_argument("--out", type=Path,
                   default=Path("data/reference/drift/run15_reference_profile.json"),
                   help="Where to write the aggregate profile. THIS is committed.")
    p.add_argument("--n-bins", type=int, default=10,
                   help="PSI bins. MUST match DriftDetector's n_bins or every PSI changes.")
    # Verification is ON and cannot be silently skipped. A profile that has not been proven
    # equivalent to the raw matrix is a profile that may be quietly wrong in every PSI it ever
    # produces -- and it would look completely healthy. The escape hatch is explicit, named
    # for what it costs you, and it makes the script exit non-zero so it can never be mistaken
    # for a certified build.
    p.add_argument("--skip-verification-I-ACCEPT-AN-UNPROVEN-PROFILE",
                   dest="skip_verify", action="store_true",
                   help="Skip the PSI equivalence proof. The profile is then UNCERTIFIED and "
                        "the script exits 3. Do not commit the result.")
    args = p.parse_args()

    from genomic_variant_classifier.monitoring.drift_reference_profile import (
        DriftReferenceProfile,
    )

    if not args.reference.is_file():
        logger.error("Reference matrix not found: %s", args.reference)
        return 2

    logger.info("Reading %s ...", args.reference)
    X_ref = pd.read_parquet(args.reference)
    logger.info("Reference: %d rows x %d features", *X_ref.shape)

    numeric = X_ref.select_dtypes(include=[np.number])
    dropped = [c for c in X_ref.columns if c not in numeric.columns]
    if dropped:
        logger.warning(
            "Dropping %d non-numeric column(s) -- the drift detector only handles numeric "
            "features: %s", len(dropped), dropped,
        )
    X_ref = numeric

    profile = DriftReferenceProfile.from_reference(
        X_ref, source=str(args.reference), n_bins=args.n_bins,
    )
    out = profile.save(args.out)

    # ------------------------------------------------------------------ verify --
    if args.skip_verify:
        logger.error(
            "VERIFICATION SKIPPED. This profile is UNCERTIFIED: nothing has checked that the "
            "Population Stability Index it produces matches the raw matrix. Exiting 3. "
            "DO NOT COMMIT THIS FILE."
        )
        return 3
    else:
        logger.info("VERIFYING: PSI from the profile must EXACTLY equal PSI from the raw matrix.")
        from genomic_variant_classifier.monitoring.drift_detector import DriftDetector

        detector = DriftDetector.from_reference(X_ref=X_ref)
        reloaded = DriftReferenceProfile.load(out)

        # Compare against a PERTURBED copy, so the PSI values are non-zero and the comparison
        # is meaningful. Comparing the reference against itself gives PSI ~ 0 for every
        # feature and would prove nothing.
        rng = np.random.default_rng(0)
        X_new = X_ref.iloc[rng.choice(len(X_ref), min(50_000, len(X_ref)), replace=False)].copy()
        for col in X_new.columns[: min(10, X_new.shape[1])]:
            X_new[col] = X_new[col] * 1.15 + 0.05    # a deliberate shift

        worst = 0.0
        worst_feat = ""
        mismatches = 0
        for i, feat in enumerate(detector.feature_names):
            ref_col = detector.ref_data[:, i]
            ref_col = ref_col[np.isfinite(ref_col)]
            new_col = X_new[feat].to_numpy(dtype=np.float64)
            new_col = new_col[np.isfinite(new_col)]

            psi_raw = detector._psi(ref_col, new_col)
            psi_prof = reloaded.psi(feat, new_col)

            delta = abs(psi_raw - psi_prof)
            if delta > worst:
                worst, worst_feat = delta, feat
            if delta > 1e-9:
                mismatches += 1
                logger.error(
                    "PSI MISMATCH on %r: raw=%.12f profile=%.12f delta=%.3e",
                    feat, psi_raw, psi_prof, delta,
                )

        if mismatches:
            logger.error(
                "%d feature(s) disagree. The profile is NOT equivalent to the raw matrix and "
                "MUST NOT be used -- every PSI in every drift report would be wrong. "
                "Refusing to certify.", mismatches,
            )
            return 1

        # `worst_feat` stays "" when EVERY delta is exactly 0.0 -- which is the outcome we
        # want, and which the first real run produced (78/78 features, worst delta 0.000e+00).
        # Printing "on ''" made a perfect result look like a bug. Say what actually happened.
        if worst == 0.0:
            logger.info(
                "VERIFIED: PSI is BIT-IDENTICAL on all %d features. Every delta was exactly "
                "0.0 -- not 'within tolerance', not 'close'. The profile is a faithful "
                "stand-in for the reference matrix.",
                len(detector.feature_names),
            )
        else:
            logger.info(
                "VERIFIED: PSI identical on all %d features to within %.3e (worst: %r).",
                len(detector.feature_names), worst, worst_feat,
            )

    size_kb = out.stat().st_size / 1024
    logger.info("")
    logger.info("Profile written: %s (%.0f KB)", out, size_kb)
    logger.info("  features:       %d", len(profile.feature_names))
    logger.info("  reference rows: %d", profile.n_ref_samples)
    logger.info("")
    logger.info("This file contains HISTOGRAM COUNTS and QUANTILE GRIDS only -- no variant,")
    logger.info("no identifier, no per-variant annotation value. It is safe to commit, and")
    logger.info("it is what lets the drift monitor run without the 23.8 MB cohort matrix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
