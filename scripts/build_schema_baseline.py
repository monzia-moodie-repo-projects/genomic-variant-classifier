#!/usr/bin/env python3
"""build_schema_baseline.py -- capture the expected feature-matrix schema (ordered
column names + dtypes + canonical hash) from a reference split, writing the artifact
that SchemaDriftMonitorAgent compares incoming matrices against.

Uses SchemaDriftAgent.hash_schema so the stored hash matches the live detector exactly.
Reads the full matrix via pandas (a few seconds for the ~25 MB X_train) so dtypes match
what detect() observes. Author: Monzia Moodie.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from genomic_variant_classifier.agent_layer.agents.schema_drift_agent import SchemaDriftAgent

DEFAULT_OUT = Path("data/reference/schema/schema_baseline.json")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build schema-baseline JSON from a reference feature matrix.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--matrix", type=Path, help="reference feature-matrix parquet (a stale default risked regressing the baseline, so there is none)")
    src.add_argument(
        "--from-contract", action="store_true",
        help="Derive the baseline from THE CODE: run the real feature builder "
             "(engineer_features) over the correctness harness's fully-populated fixture, and "
             "capture the resulting columns and dtypes. Then assert the column set equals "
             "TABULAR_FEATURES. Use this when no current matrix exists (e.g. before Run 17).",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output baseline JSON")
    ap.add_argument("--run-label", default="run15", help="provenance label for the source run")
    ap.add_argument("--allow-schema-change", action="store_true", help="permit a column-set change vs the existing baseline (otherwise a mismatch ABORTS to prevent silent regression)")
    ap.add_argument(
        "--verify-against", type=Path, default=None,
        help="A REAL persisted feature matrix (parquet). After building, assert that the "
             "baseline's dtype for every column this matrix also has MATCHES the matrix. "
             "Use it with --from-contract: the column set then comes from the code (so it "
             "cannot go stale) and the dtypes are PROVEN against real data (so they cannot "
             "be guessed wrong). Exits 1 on any disagreement.",
    )
    args = ap.parse_args()

    # =======================================================================================
    # --from-contract : DERIVE THE SCHEMA FROM THE CODE, NOT FROM A TRANSCRIPTION
    # Added 2026-07-13 (roadmap 6.22).
    #
    # The committed baseline had drifted TEN COLUMNS behind TABULAR_FEATURES. Its own
    # `captured_from` field admitted why:
    #
    #     "derived: run16b-smoke baseline + hetero_gnn_score(float64) for Run 17
    #      + 5 rnaseq_* (float64) surgically added for Run-17 RNA-seq branch"
    #
    # It was not captured. It was HAND-MAINTAINED -- edited by whoever last remembered to edit
    # it. cosmic_recurrence, cosmic_sig_tier, finngen_r13_af_fin, finngen_r13_af_nfsee,
    # finngen_r13_enrichment, genomiclm_delta_norm, genomiclm_llr, kegg_disease_pathway_flag,
    # kegg_pathway_count and omim_n_diseases_molecular were all added to the feature contract
    # and never added here. The schema gate -- whose entire job is to catch a column set that
    # has silently changed -- was itself silently ten columns out of date, and would have
    # fired on Run 17 for a "drift" that was really its own staleness.
    #
    # That is root pattern (a): a thing written down once, by hand, becomes a lie on a
    # schedule. Patching in the missing ten would merely restart the clock.
    #
    # So: derive the column set and the dtypes by RUNNING THE REAL FEATURE BUILDER. The
    # baseline can then never disagree with the code that produces the matrices it guards,
    # and tests/unit/test_schema_baseline_matches_contract.py fails the suite if it ever does.
    # =======================================================================================
    if args.from_contract:
        from genomic_variant_classifier.agent_layer.harness.correctness_harness import (
            build_reference_slice,
        )
        from genomic_variant_classifier.models.variant_ensemble import (
            EXPECTED_TABULAR_FEATURE_COUNT,
            TABULAR_FEATURES,
            engineer_features,
        )

        print("Deriving the schema from the code (engineer_features over the harness fixture)...")
        raw = build_reference_slice(n=200, seed=7)
        df = engineer_features(raw)

        # The fixture is built to be FULLY POPULATED, so every declared feature must appear.
        # If one does not, the feature builder and the contract have diverged -- which is
        # exactly the condition this baseline exists to detect, and it must not be papered over.
        produced = [c for c in df.columns]
        missing = [c for c in TABULAR_FEATURES if c not in produced]
        extra = [c for c in produced if c not in TABULAR_FEATURES]
        if missing or extra:
            print(
                "ABORT: engineer_features() and TABULAR_FEATURES DISAGREE.\n"
                f"  declared but not produced ({len(missing)}): {missing}\n"
                f"  produced but not declared ({len(extra)}): {extra}\n"
                "\n"
                "The feature builder and the feature contract have drifted apart. Fix that "
                "first -- a schema baseline derived from a broken contract would enshrine the "
                "break."
            )
            return 1

        # Order by the CONTRACT, not by whatever order the builder happens to emit. Column
        # order is part of the hash, and LightGBM maps columns POSITIONALLY (CLAUDE.md sec. 5).
        df = df[list(TABULAR_FEATURES)]
        assert len(df.columns) == EXPECTED_TABULAR_FEATURE_COUNT, (
            f"{len(df.columns)} columns vs EXPECTED_TABULAR_FEATURE_COUNT="
            f"{EXPECTED_TABULAR_FEATURE_COUNT}"
        )

        # ===================================================================================
        # DTYPES: float64 ACROSS THE BOARD -- and this is NOT a convenience cast.
        #
        # The schema gate validates the PERSISTED feature matrix
        # (outputs/<run>/full/splits/X_train.parquet), not the raw output of
        # engineer_features(). Those two things have DIFFERENT DTYPES, and the difference is
        # not cosmetic:
        #
        #   engineer_features() emits int64 for ~40 columns -- the binary indicators
        #   (af_is_absent, is_snv, cadd_high, ...) and the integer counts (ref_len, ...),
        #   several of which it explicitly `.astype(int)`s.
        #
        #   The persisted matrix is STANDARDISED before it is written. Scaling produces
        #   float64 for every numeric column. MEASURED, 2026-07-13, on the real Run-15
        #   artifact: X_train.parquet is float64 for 78 of 78 columns, including af_is_absent,
        #   ref_len, is_snv and cadd_high -- every one of which engineer_features emits as
        #   int64.
        #
        # The FIRST version of --from-contract (written earlier on 2026-07-13) captured the
        # raw builder dtypes and would therefore have baked ~40 int64 columns into a baseline
        # that is compared against an all-float64 matrix. SchemaDriftAgent._dtype_family is
        # IDENTITY for numeric dtypes (it collapses only the pandas 2.x/3.x string spellings),
        # so every one of those ~40 columns would have registered as a DTYPE CHANGE, and the
        # gate would have exited 2 = SCHEMA DRIFT DETECTED on Run 17 -- for drift that does
        # not exist.
        #
        # That would have replaced a stale-columns bug with a wrong-dtypes bug, in a gate whose
        # only job is to be trustworthy. The tell was in the OLD baseline all along:
        # hgmd_is_disease_mutation was recorded as float64 there, even though engineer_features
        # cast it with `.astype(int)`. The old baseline had been captured from a PROCESSED
        # matrix. It was on screen and it was read past.
        #
        # So: cast to the dtype the persisted artifact actually carries, and then PROVE it
        # against a real matrix with --verify-against. Do not take my word for it; do not take
        # this comment's word for it either.
        # ===================================================================================
        df = df.astype("float64")
        source_label = (
            "derived from code: engineer_features(build_reference_slice(n=200, seed=7)), "
            "column set and order asserted equal to TABULAR_FEATURES, dtypes cast to float64 "
            "to match the STANDARDISED persisted matrix the schema gate validates"
        )
    else:
        if not args.matrix.exists():
            print(f"ABORT: matrix not found: {args.matrix}")
            return 1
        df = pd.read_parquet(args.matrix)
        source_label = str(args.matrix).replace("\\", "/")

    expected_dtypes = {str(c): str(df[c].dtype) for c in df.columns}  # column order preserved
    if not expected_dtypes:
        print("ABORT: matrix has no columns")
        return 1

    # -----------------------------------------------------------------------------------
    # PROVE THE DTYPES AGAINST A REAL PERSISTED MATRIX.
    #
    # The whole point of --from-contract is that the COLUMN SET comes from the code and
    # therefore cannot go stale. But the DTYPES cannot come from the code -- the persisted
    # matrix is standardised after engineer_features runs, so its dtypes are a property of
    # the PIPELINE, not of the builder. Deriving them from the builder is precisely the
    # mistake that was made and caught on 2026-07-13 (see the block above).
    #
    # So: assert them against a real matrix. Columns the matrix does not have (features added
    # since it was produced) are reported explicitly as UNVERIFIED -- never silently accepted.
    # -----------------------------------------------------------------------------------
    if args.verify_against is not None:
        if not args.verify_against.exists():
            print(f"ABORT: --verify-against matrix not found: {args.verify_against}")
            return 1

        real = pd.read_parquet(args.verify_against)
        real_dtypes = {str(c): str(real[c].dtype) for c in real.columns}

        shared = [c for c in expected_dtypes if c in real_dtypes]
        mismatched = [
            (c, expected_dtypes[c], real_dtypes[c])
            for c in shared
            if expected_dtypes[c] != real_dtypes[c]
        ]
        unverified = [c for c in expected_dtypes if c not in real_dtypes]

        print(f"\nDTYPE VERIFICATION against {args.verify_against}")
        print(f"  columns in common      : {len(shared)}")
        print(f"  dtype mismatches       : {len(mismatched)}")
        print(f"  UNVERIFIED (not in it) : {len(unverified)}")

        if mismatched:
            print(
                "\nABORT: the baseline's dtypes DISAGREE with the real persisted matrix.\n"
                "SchemaDriftAgent._dtype_family is IDENTITY for numeric dtypes, so every one\n"
                "of these would register as a DTYPE CHANGE and the schema gate would exit 2 =\n"
                "SCHEMA DRIFT DETECTED -- for drift that does not exist. Refusing to write a\n"
                "baseline that would make the gate lie.\n"
            )
            for col, exp, obs in mismatched:
                print(f"    {col:38s} baseline={exp:10s} real={obs}")
            return 1

        if unverified:
            print(
                f"\n  NOTE: {len(unverified)} column(s) are in the feature contract but absent\n"
                f"  from {args.verify_against.name}, so their dtypes could NOT be proven against\n"
                f"  real data. They are recorded as float64 (the dtype every standardised column\n"
                f"  in that matrix carries). Re-run --verify-against the FIRST matrix a run\n"
                f"  produces that contains them, and this list must shrink to zero:\n"
            )
            for c in unverified:
                print(f"    {c}")

        print("\n  VERIFIED: every provable dtype matches the real matrix.")
    expected_hash = SchemaDriftAgent.hash_schema(expected_dtypes)

    # Regression guard: refuse to silently change the committed baseline's column SET.
    # (Original footgun: a stale --matrix would drop columns and shrink the baseline 81->78.)
    if args.out.exists() and not args.allow_schema_change:
        try:
            _prev_cols = set(json.loads(args.out.read_text(encoding="utf-8")).get("expected_dtypes", {}))
        except (json.JSONDecodeError, OSError):
            _prev_cols = set()
        _new_cols = set(expected_dtypes)
        if _prev_cols and _new_cols != _prev_cols:
            print(
                f"ABORT: column set differs from the existing baseline "
                f"({len(_prev_cols)} -> {len(_new_cols)} columns). "
                f"removed={sorted(_prev_cols - _new_cols)} added={sorted(_new_cols - _prev_cols)}. "
                f"This would change the schema; re-run with --allow-schema-change if intentional."
            )
            return 1

    payload = {
        "schema_version": 1,
        "run_label": args.run_label,
        "captured_from": source_label,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "n_columns": len(expected_dtypes),
        "expected_schema_hash": expected_hash,
        "expected_dtypes": expected_dtypes,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # round-trip: the written file must re-hash to the same value
    reloaded = json.loads(args.out.read_text(encoding="utf-8"))
    if reloaded["expected_schema_hash"] != SchemaDriftAgent.hash_schema(reloaded["expected_dtypes"]):
        print("ABORT: hash mismatch after reload")
        return 1

    print(f"OK: wrote {args.out}")
    print(f"  columns={payload['n_columns']}  hash={expected_hash[:16]}...  run={args.run_label}")
    print(f"  source={payload['captured_from']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
