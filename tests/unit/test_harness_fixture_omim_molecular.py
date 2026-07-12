"""Regression test: build_reference_slice feeds omim_n_diseases_molecular (feature #88),
bounded by total (molecular <= total), so the G1 Section-14 correctness harness does not
flag it as an unexpected silent-zero.

Guards against the fixture-gap that made omim_n_diseases_molecular read as a NON-allowlist
silent-zero after feature #88 was added (Run-17 OMIM genemap2 arc, 2026-06-26).
"""
from __future__ import annotations
import pathlib
import numpy as np
import pytest

import genomic_variant_classifier.agent_layer.harness.correctness_harness as CH
from genomic_variant_classifier.agent_layer.harness import (
    build_reference_slice, KNOWN_ZERO_DEFAULT,
)


def test_fixture_feeds_molecular_column():
    s = build_reference_slice()
    assert "omim_n_diseases_molecular" in s.columns, (
        "build_reference_slice must feed omim_n_diseases_molecular (feature #88); "
        "absence makes the G1 harness flag it as an unexpected silent-zero.")


def test_molecular_not_all_zero():
    s = build_reference_slice()
    assert int((s["omim_n_diseases_molecular"] != 0).sum()) > 0, (
        "molecular column must be populated (non-trivial), else it is a silent-zero.")


def test_molecular_le_total_invariant():
    s = build_reference_slice()
    assert (s["omim_n_diseases_molecular"] <= s["omim_n_diseases"]).all(), (
        "real-world invariant: molecular-basis disease count <= total disease count.")


def test_fixture_deterministic():
    a = build_reference_slice()["omim_n_diseases_molecular"].to_numpy()
    b = build_reference_slice()["omim_n_diseases_molecular"].to_numpy()
    assert np.array_equal(a, b), "fixture must be deterministic for fixed default seed."


def test_molecular_not_in_allowlist():
    # The whole point of Option B: molecular is a LIVE feature, NOT an expected-zero.
    assert "omim_n_diseases_molecular" not in KNOWN_ZERO_DEFAULT, (
        "omim_n_diseases_molecular must NOT be allowlisted (it is live, 71.68% real cohort); "
        "it is fed by the fixture instead.")


def test_allowlist_unchanged_size():
    # Option B feeds finngen in the fixture; the two R12 AF columns
    # (finngen_af_fin, finngen_af_nfsee) were therefore REMOVED from the allowlist
    # (27 -> 25). R13 AF is also fed, never allowlisted. enrichment never zeros.
    #
    # 2026-07-11 (25 -> 24): gene_is_constrained REMOVED. It is a DERIVED binary
    # indicator, (gene_constraint_oe < 0.35).astype(int) -- variant_ensemble.py:439 --
    # not a connector. It takes both 0 and 1 on the reference slice, so stage 5's
    # binary exemption skips it and it can never be flagged; allowlisting it was dead
    # weight that would have SILENTLY swallowed a real constraint-connector regression
    # (which would collapse it to {0} -> non-binary -> flagged).
    #
    # The six KEGG/COSMIC/Nucleotide-Transformer columns from the 91->97 work
    # (80eb9c8) are FED in build_reference_slice, NOT allowlisted -- same rule.
    assert len(KNOWN_ZERO_DEFAULT) == 24, (
        f"KNOWN_ZERO_DEFAULT must be 24 (Option B feeds finngen R12+R13 AF and the six "
        f"kegg/cosmic/genomiclm columns; gene_is_constrained is derived, not a dead "
        f"connector); got {len(KNOWN_ZERO_DEFAULT)}.")


def test_live_connectors_are_fed_not_allowlisted():
    """THE RULE, as an executable guard (2026-07-11).

    Every feature the fixture is supposed to bring alive must be (a) present as an
    input column in build_reference_slice and (b) absent from KNOWN_ZERO_DEFAULT.
    This is the guard whose absence let the 91->97 feature work (80eb9c8, 2026-07-06)
    land six live connectors that the fixture never fed -- red since the day they
    landed (TRIAGE_2026-07-08_test-suite-red, cluster C).
    """
    live_connector_inputs = [
        "genomiclm_delta_norm", "genomiclm_llr",
        "cosmic_recurrence", "cosmic_sig_tier",
        "kegg_pathway_count", "kegg_disease_pathway_flag",
        "finngen_af_fin", "finngen_af_nfsee",
        "finngen_r13_af_fin", "finngen_r13_af_nfsee",
        "esm2_llr", "esm2_delta_norm", "clingen_validity_score",
        "omim_n_diseases_molecular",
    ]
    s = build_reference_slice()
    for col in live_connector_inputs:
        assert col in s.columns, (
            f"build_reference_slice must FEED the live connector input {col!r} "
            f"(Option B). A live feature is never allowlisted.")
        assert col not in KNOWN_ZERO_DEFAULT, (
            f"{col!r} is a LIVE connector and must NOT be in KNOWN_ZERO_DEFAULT -- "
            f"allowlisting it would blind stage 5 to a real regression.")
