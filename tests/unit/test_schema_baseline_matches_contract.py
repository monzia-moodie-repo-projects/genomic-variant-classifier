"""The committed schema baseline must agree with the feature contract. Always. Mechanically.

Created 2026-07-13 (roadmap 6.22).

WHY THIS EXISTS
---------------
`data/reference/schema/schema_baseline.json` is the artifact the schema-drift gate compares
every incoming feature matrix against. Its job is to catch a column that has been renamed,
retyped, added or dropped -- "before they silently zero a feature", in the words of
run_schema_drift_check.py's own docstring.

On 2026-07-13 it was found to be TEN COLUMNS out of date with TABULAR_FEATURES:

    cosmic_recurrence, cosmic_sig_tier, finngen_r13_af_fin, finngen_r13_af_nfsee,
    finngen_r13_enrichment, genomiclm_delta_norm, genomiclm_llr,
    kegg_disease_pathway_flag, kegg_pathway_count, omim_n_diseases_molecular

Every one of them had been added to the feature contract and never added to the baseline. The
baseline's own `captured_from` field says exactly how that happened:

    "derived: run16b-smoke baseline + hetero_gnn_score(float64) for Run 17
     + 5 rnaseq_* (float64) surgically added for Run-17 RNA-seq branch"

It was not captured from anything. It was HAND-MAINTAINED -- kept up to date by whoever
remembered. Which is to say: not kept up to date.

So the gate that exists to detect a silently-changed column set was ITSELF a silently-changed
column set, and Run 17 would have tripped it -- on the gate's own staleness, not on any real
drift. A gate that cries wolf about itself gets switched off, and then the next real schema
change goes through unnoticed.

THAT IS ROOT PATTERN (a): a number -- or a list -- written down once and never re-derived
becomes a lie on a schedule. It is the same shape as the pytest floor that rotted five times in
two days beneath an all-capitals comment ordering the next person to raise it.

    A COMMENT DOES NOT ENFORCE ITSELF. MAKE FORGETTING FAIL.

THE FIX, IN TWO PARTS
---------------------
1. `scripts/build_schema_baseline.py --from-contract` now DERIVES the baseline by running the
   real feature builder (engineer_features) over the correctness harness's fully-populated
   fixture, and asserts the resulting column set equals TABULAR_FEATURES. The baseline is
   therefore a product of the code, not a transcription of it.

2. This file. It re-derives the agreement ON EVERY TEST RUN. Add a feature to TABULAR_FEATURES
   and forget the baseline, and the suite goes RED immediately, naming the columns. The
   baseline can no longer rot, because rotting now fails.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from genomic_variant_classifier.models.variant_ensemble import (
    EXPECTED_TABULAR_FEATURE_COUNT,
    TABULAR_FEATURES,
)

BASELINE = Path("data/reference/schema/schema_baseline.json")

REBUILD = (
    "\n\nTo regenerate the baseline FROM THE CODE (not by hand):\n"
    "    python scripts/build_schema_baseline.py --from-contract \\\n"
    "        --run-label run17-preflight --allow-schema-change\n"
    "\nDo it IN THE SAME COMMIT as the feature change. That is the whole point."
)


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE.is_file():
        pytest.fail(
            f"The schema baseline is MISSING: {BASELINE}\n"
            f"The schema-drift gate has nothing to compare against, which means it cannot "
            f"fail, which means it is not a gate.{REBUILD}"
        )
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_baseline_column_set_equals_the_feature_contract(baseline):
    """The central assertion. Everything else here is detail."""
    declared = set(TABULAR_FEATURES)
    committed = set(baseline["expected_dtypes"])

    missing = sorted(declared - committed)      # in the contract, absent from the baseline
    extra = sorted(committed - declared)        # in the baseline, absent from the contract

    assert not missing and not extra, (
        f"THE SCHEMA BASELINE HAS DRIFTED FROM THE FEATURE CONTRACT.\n"
        f"\n"
        f"  in TABULAR_FEATURES but NOT in the baseline ({len(missing)}): {missing}\n"
        f"  in the baseline but NOT in TABULAR_FEATURES ({len(extra)}): {extra}\n"
        f"\n"
        f"The schema-drift gate compares incoming matrices against this baseline. While it "
        f"disagrees with the contract, the gate reports drift that is really its own "
        f"staleness -- and a gate that cries wolf about itself is a gate that gets switched "
        f"off.{REBUILD}"
    )


def test_baseline_column_count_matches_the_expected_count(baseline):
    assert baseline["n_columns"] == EXPECTED_TABULAR_FEATURE_COUNT == len(TABULAR_FEATURES), (
        f"n_columns={baseline['n_columns']}, "
        f"EXPECTED_TABULAR_FEATURE_COUNT={EXPECTED_TABULAR_FEATURE_COUNT}, "
        f"len(TABULAR_FEATURES)={len(TABULAR_FEATURES)}. "
        f"These three must agree.{REBUILD}"
    )


def test_baseline_column_ORDER_matches_the_contract(baseline):
    """Order is not cosmetic. It is in the hash, and LightGBM maps columns POSITIONALLY.

    CLAUDE.md section 5: LightGBM is the sole model in the roster that maps columns by
    POSITION and returns silently wrong predictions on mis-ordered input (measured delta
    0.855 -- no error, no warning, not even under `-W error`). A baseline whose column order
    disagreed with the contract would be a standing invitation to reorder the matrix.
    """
    assert list(baseline["expected_dtypes"]) == list(TABULAR_FEATURES), (
        "The baseline's column ORDER differs from TABULAR_FEATURES. Column order is part of "
        "the schema hash, and LightGBM maps columns positionally -- mis-ordered input gives "
        "silently wrong predictions with no error raised." + REBUILD
    )


def test_every_baseline_dtype_is_float64(baseline):
    """The persisted matrix is STANDARDISED, so every column in it is float64.

    This is the guard against a mistake that was made, and caught, on 2026-07-13.

    `engineer_features()` emits int64 for roughly forty columns -- the binary indicators
    (af_is_absent, is_snv, cadd_high, ...) and the integer counts (ref_len, ...), several of
    which it explicitly `.astype(int)`s. But the schema gate does not validate the raw builder
    output. It validates the PERSISTED matrix, `outputs/<run>/full/splits/X_train.parquet`,
    which is standardised before it is written -- and scaling makes every numeric column
    float64. Measured on the real Run-15 artifact: 78 of 78 columns are float64, including
    af_is_absent, ref_len, is_snv and cadd_high.

    The first version of `build_schema_baseline.py --from-contract` captured the RAW builder
    dtypes. Had that baseline been committed, `SchemaDriftAgent._dtype_family` -- which is
    IDENTITY for numeric dtypes, collapsing only the pandas 2.x/3.x string spellings -- would
    have reported roughly forty DTYPE CHANGES against the real matrix, and the gate would have
    exited 2 = SCHEMA DRIFT DETECTED on Run 17, for drift that did not exist.

    A gate that fires on nothing is worse than no gate: it teaches everyone to ignore it, and
    then the next REAL schema change goes through unremarked.

    (The tell was sitting in the OLD baseline the whole time: hgmd_is_disease_mutation was
    recorded there as float64, even though engineer_features cast it with `.astype(int)`. The
    old baseline had been captured from a processed matrix. It was on screen and it was read
    past.)
    """
    non_float = {c: d for c, d in baseline["expected_dtypes"].items() if d != "float64"}
    assert not non_float, (
        f"{len(non_float)} baseline column(s) are not float64: {non_float}\n"
        f"\n"
        f"The schema gate compares this baseline against the STANDARDISED persisted matrix, "
        f"where every column is float64. A non-float64 entry here will be reported as a dtype "
        f"CHANGE and will fail the gate on a run that has nothing wrong with it.\n"
        f"\n"
        f"This almost certainly means the baseline was regenerated from raw engineer_features "
        f"output instead of from the persisted matrix's dtypes. Rebuild with:\n"
        f"    python scripts/build_schema_baseline.py --from-contract \\\n"
        f"        --verify-against outputs/run15_rerun_report/full/splits/X_train.parquet \\\n"
        f"        --run-label run17-preflight --allow-schema-change"
    )


def test_hgmd_has_not_crept_back_into_the_baseline(baseline):
    """HGMD was removed from the contract 2026-07-13. The baseline must not resurrect it.

    Pinned separately because the baseline is a data file, and data files get restored from
    backups, merged badly, and regenerated from stale matrices. hgmd_is_disease_mutation is a
    near-copy of the ClinVar-Pathogenic label; it must not return by the back door.
    """
    hgmd = [c for c in baseline["expected_dtypes"] if "hgmd" in c.lower()]
    assert not hgmd, (
        f"HGMD columns are back in the schema baseline: {hgmd}. They were removed from "
        f"TABULAR_FEATURES on 2026-07-13 -- no license, and variant-level label leakage "
        f"against a ClinVar-Pathogenic target.{REBUILD}"
    )


def test_the_baseline_hash_actually_matches_its_own_contents(baseline):
    """A hash nobody recomputes is a decoration.

    The baseline carries `expected_schema_hash`, and the drift agent compares against it. If
    the file were hand-edited -- which is exactly how it drifted ten columns in the first
    place -- the hash would no longer describe the contents, and the gate would be comparing
    against a fiction.
    """
    pytest.importorskip("pandera")
    from genomic_variant_classifier.agent_layer.agents.schema_drift_agent import (
        SchemaDriftAgent,
    )

    recomputed = SchemaDriftAgent.hash_schema(baseline["expected_dtypes"])
    assert recomputed == baseline["expected_schema_hash"], (
        f"The baseline's stored hash does not match its own contents.\n"
        f"  stored:     {baseline['expected_schema_hash']}\n"
        f"  recomputed: {recomputed}\n"
        f"\n"
        f"The file has been edited by hand. Regenerate it with the script, which recomputes "
        f"and round-trip-verifies the hash.{REBUILD}"
    )
