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
    assert len(KNOWN_ZERO_DEFAULT) == 25, (
        f"KNOWN_ZERO_DEFAULT must be 25 (Option B feeds finngen R12+R13 AF, does not "
        f"allowlist them); got {len(KNOWN_ZERO_DEFAULT)}.")
