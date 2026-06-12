"""Tests for RNA maxentscan_delta activation (Change 2, option b).

CI-safe (no torch). Exercises the delta plumbing in rna_pipeline.annotate_dataframe
and confirms maxentscan_delta is registered consistently with the feature-count
contract. The real PWM values are irrelevant to the plumbing, so _score_donor is
monkeypatched to a content-sensitive stub where it matters.
Author: Monzia Moodie.
"""
from __future__ import annotations

import pandas as pd
import pytest

from genomic_variant_classifier.pipelines import rna_pipeline as rp
from genomic_variant_classifier.models.variant_ensemble import (
    TABULAR_FEATURES,
    EXPECTED_TABULAR_FEATURE_COUNT,
)

_REF = "ACGT" * 25 + "A"             # 101 bp, center base 'A'
_ALT = _REF[:50] + "C" + _REF[51:]   # SNV at center: A -> C (inside donor 9-mer)


def _content_sensitive_donor(seq9):
    if len(seq9) != 9 or any(b not in "ACGT" for b in seq9):
        return 0.0
    return sum((i + 1) * "ACGT".index(b) for i, b in enumerate(seq9)) / 10.0


def test_maxentscan_delta_nonzero_for_real_variant(monkeypatch):
    monkeypatch.setattr(rp, "_score_donor", _content_sensitive_donor)
    df = pd.DataFrame(
        {"is_splice": [1, 1, 0],
         "fasta_seq_ref": [_REF, _REF, _REF],
         "fasta_seq_alt": [_ALT, _REF, _ALT]}
    )
    out = rp.RNASpliceIsoformPipeline().annotate_dataframe(df)
    assert "maxentscan_delta" in out.columns
    d = out["maxentscan_delta"].tolist()
    assert d[0] != 0.0    # splice, ref != alt -> real delta
    assert d[1] == 0.0    # splice, ref == alt -> zero delta
    assert d[2] == 0.0    # non-splice -> default


def test_legacy_single_fasta_seq_zero_delta(monkeypatch):
    monkeypatch.setattr(rp, "_score_donor", _content_sensitive_donor)
    df = pd.DataFrame({"is_splice": [1], "fasta_seq": [_REF]})
    out = rp.RNASpliceIsoformPipeline().annotate_dataframe(df)
    assert out["maxentscan_delta"].iloc[0] == 0.0    # ref == alt fallback
    assert out["maxentscan_score"].iloc[0] != 0.0    # ref window still scored


def test_absent_sequence_graceful_defaults():
    df = pd.DataFrame({"is_splice": [1]})    # no ref/alt, no fasta_seq
    out = rp.RNASpliceIsoformPipeline().annotate_dataframe(df)
    assert out["maxentscan_delta"].iloc[0] == 0.0
    assert out["maxentscan_score"].iloc[0] == 0.0


def test_non_splice_cohort_still_has_delta_column():
    df = pd.DataFrame(
        {"is_splice": [0, 0],
         "fasta_seq_ref": [_REF, _REF], "fasta_seq_alt": [_ALT, _ALT]}
    )
    out = rp.RNASpliceIsoformPipeline().annotate_dataframe(df)   # early-return path
    assert "maxentscan_delta" in out.columns
    assert (out["maxentscan_delta"] == 0.0).all()


def test_maxentscan_delta_registered_and_contract_holds():
    assert "maxentscan_delta" in TABULAR_FEATURES
    assert TABULAR_FEATURES.count("maxentscan_delta") == 1
    assert len(TABULAR_FEATURES) == EXPECTED_TABULAR_FEATURE_COUNT
