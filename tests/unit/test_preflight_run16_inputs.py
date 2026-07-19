"""Tests for the Run-16 input preflight gate (scripts/preflight_run16_inputs.py).

CI-safe: importorskip pyarrow; the cohort checks use tmp parquets. Mirrors the
exit-code-contract style of test_run_schema_drift_check.py. Author: Monzia Moodie.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import preflight_run16_inputs as pf  # noqa: E402

_REF = "ACGT" * 25 + "A"
_ALT = _REF[:50] + "C" + _REF[51:]
# _POLY = "A" * 101 was removed 2026-07-18 along with its only user. The gate no longer
# judges a window by its content, so a literal placeholder string has nothing to test.


def test_cohort_with_ref_alt_and_provenance_passes(tmp_path):
    """A healthy cohort: windows present AND the builder vouches for them."""
    p = tmp_path / "good.parquet"
    pd.DataFrame({"fasta_seq_ref": [_REF] * 500, "fasta_seq_alt": [_ALT] * 500,
                  "ok": [True] * 500}).to_parquet(p)
    ok, _ = pf.check_cohort_ref_alt(str(p))
    assert ok is True


def test_cohort_without_provenance_fails(tmp_path):
    """Windows present but NO `ok` column -- the gate must refuse, not assume clean.

    This is the case that matters. Before 2026-07-18 the gate counted any window
    differing from "A" * 101 as real, so once the placeholder base became "N" it passed
    unconditionally. A cohort whose placeholder rows cannot be identified is not a
    cohort that has been checked.
    """
    p = tmp_path / "noprov.parquet"
    pd.DataFrame({"fasta_seq_ref": [_REF] * 500,
                  "fasta_seq_alt": [_ALT] * 500}).to_parquet(p)
    ok, msg = pf.check_cohort_ref_alt(str(p))
    assert ok is False
    assert "ok" in msg, msg


def test_cohort_missing_ref_alt_fails(tmp_path):
    p = tmp_path / "noref.parquet"
    pd.DataFrame({"fasta_seq": [None] * 500, "x": range(500)}).to_parquet(p)
    ok, msg = pf.check_cohort_ref_alt(str(p))
    assert ok is False and "INERT" in msg


def test_cohort_mostly_placeholder_fails(tmp_path):
    """Provenance says most windows are placeholders -> refuse.

    Replaces test_cohort_all_dummy_fails, which wrote all-"A"*101 windows with no `ok`
    column. That test still PASSED after the rewrite, but for the wrong reason: the gate
    refused because provenance was missing, not because the windows were placeholders.
    A test that passes for a reason unrelated to its name is not coverage. This one
    supplies provenance so the placeholder path is genuinely exercised.
    """
    p = tmp_path / "placeholder.parquet"
    pd.DataFrame({"fasta_seq_ref": [_REF] * 500, "fasta_seq_alt": [_ALT] * 500,
                  "ok": [False] * 400 + [True] * 100}).to_parquet(p)
    ok, msg = pf.check_cohort_ref_alt(str(p))
    assert ok is False
    assert "placeholder" in msg, msg


def test_missing_cohort_file_fails():
    ok, _ = pf.check_cohort_ref_alt("does_not_exist_12345.parquet")
    assert ok is False


def test_aggregate_exit_codes():
    assert pf.aggregate([(True, "a"), (True, "b")]) == 0
    assert pf.aggregate([(True, "a"), (False, "b")]) == 2
    assert pf.aggregate([(True, "a"), (None, "b")]) == 3
    assert pf.aggregate([(False, "a"), (None, "b")]) == 2   # fail dominates env


def test_check_exists(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("x")
    assert pf.check_exists("f", str(p))[0] is True
    assert pf.check_exists("f", str(tmp_path / "nope"))[0] is False
    assert pf.check_exists("f", None, required=True)[0] is False


def test_gnomad_constraint_present_passes(tmp_path):
    p = tmp_path / "constraint.tsv"
    p.write_bytes(b"x" * 2_000_000)  # 2 MB
    ok, _ = pf.check_gnomad_constraint(str(p))
    assert ok is True


def test_gnomad_constraint_missing_fails():
    ok, _ = pf.check_gnomad_constraint("nope_constraint_12345.tsv")
    assert ok is False


def test_gnomad_constraint_stub_too_small_fails(tmp_path):
    p = tmp_path / "stub.tsv"
    p.write_bytes(b"x" * 1000)  # 1 KB -> below min
    ok, _ = pf.check_gnomad_constraint(str(p))
    assert ok is False


def test_gnomad_constraint_none_fails():
    ok, _ = pf.check_gnomad_constraint(None)
    assert ok is False


def _write_cohort_rev(tmp_path, with_review):
    cols = {"chrom": ["1"], "pos": [100], "ref": ["A"], "alt": ["C"],
            "fasta_seq_ref": [_REF], "fasta_seq_alt": [_ALT]}
    if with_review:
        cols["ReviewStatus"] = ["criteria_provided,_multiple_submitters"]
    name = "rev.parquet" if with_review else "norev.parquet"
    p = tmp_path / name
    pd.DataFrame(cols).to_parquet(p)
    return str(p)


def test_reviewstatus_present_passes(tmp_path):
    ok, _ = pf.check_cohort_reviewstatus(_write_cohort_rev(tmp_path, True))
    assert ok is True


def test_reviewstatus_absent_fails(tmp_path):
    ok, _ = pf.check_cohort_reviewstatus(_write_cohort_rev(tmp_path, False))
    assert ok is False


def test_reviewstatus_missing_file_fails():
    ok, _ = pf.check_cohort_reviewstatus("nope_cohort_98765.parquet")
    assert ok is False
