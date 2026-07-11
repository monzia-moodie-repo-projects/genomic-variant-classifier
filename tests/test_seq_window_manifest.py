"""Tests for the seq_windows coherence gate (verify_seq_windows)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from genomic_variant_classifier.data.seq_window_manifest import (
    cohort_key_hash, reference_signature, verify_seq_windows, CONVENTION, WINDOW,
    SeqWindowsStaleError,
)


def _cohort():
    return pd.DataFrame({
        "chrom": ["1", "2", "3", "X", "1"],
        "pos": [100, 200, 300, 400, 500],
        "ref": ["A", "C", "G", "T", "AC"],
        "alt": ["G", "T", "A", "C", "A"],
    })


def _make_artifact(tmp_path, cohort, ref_path, *, dry=False, window=WINDOW,
                   convention=CONVENTION, n_ok=None, cohort_for_hash=None, ref_for_sig=None):
    wdir = tmp_path / "seq_windows"
    wdir.mkdir(exist_ok=True)
    # artifact parquet: one window row per cohort key
    art = cohort.copy()
    art["fasta_seq_ref"] = ["N" * window] * len(art)
    art["fasta_seq_alt"] = ["N" * window] * len(art)
    art["ok"] = True
    art["reason"] = ""
    art.to_parquet(wdir / "seq_windows.parquet", index=False)
    manifest = {
        "cohort_path": "x", "cohort_row_count": len(cohort),
        "cohort_key_sha256": cohort_key_hash(cohort_for_hash if cohort_for_hash is not None else cohort),
        "reference_path": str(ref_path),
        "reference_signature": reference_signature(ref_for_sig if ref_for_sig is not None else ref_path),
        "window": window, "convention": convention,
        "builder_version": "test", "build_utc": "2026-07-11T00:00:00Z",
        "n_rows_built": len(cohort), "n_ok": n_ok if n_ok is not None else len(cohort),
        "n_poly": 0, "poly_reason_breakdown": {}, "chunk_count": 1, "dry_run": dry,
    }
    (wdir / "seq_windows.manifest.json").write_text(json.dumps(manifest))
    return wdir


def _ref(tmp_path):
    fa = tmp_path / "GRCh38.fa"
    fa.write_text(">1\nACGTACGT\n")
    (tmp_path / "GRCh38.fa.fai").write_text("1\t8\t3\t8\t9\n")
    return fa


def test_gate_passes_on_matching_artifact(tmp_path):
    c = _cohort(); ref = _ref(tmp_path)
    wdir = _make_artifact(tmp_path, c, ref)
    r = verify_seq_windows(c, wdir, ref)
    assert r.ok, r.reasons
    r.raise_if_failed()  # must not raise


def test_gate_aborts_on_missing_manifest(tmp_path):
    c = _cohort(); ref = _ref(tmp_path)
    (tmp_path / "empty").mkdir()
    r = verify_seq_windows(c, tmp_path / "empty", ref)
    assert not r.ok and any("manifest" in x for x in r.reasons)


def test_gate_aborts_on_cohort_change(tmp_path):
    c = _cohort(); ref = _ref(tmp_path)
    wdir = _make_artifact(tmp_path, c, ref)
    c2 = c.copy(); c2.loc[0, "alt"] = "T"  # a variant changed
    r = verify_seq_windows(c2, wdir, ref)
    assert not r.ok and any("cohort" in x for x in r.reasons)


def test_gate_aborts_on_reference_change(tmp_path):
    c = _cohort(); ref = _ref(tmp_path)
    wdir = _make_artifact(tmp_path, c, ref)
    # change the .fai so the reference signature differs
    (tmp_path / "GRCh38.fa.fai").write_text("1\t9\t3\t9\t10\n")
    r = verify_seq_windows(c, wdir, ref)
    assert not r.ok and any("reference" in x for x in r.reasons)


def test_gate_aborts_on_dry_run_artifact(tmp_path):
    c = _cohort(); ref = _ref(tmp_path)
    wdir = _make_artifact(tmp_path, c, ref, dry=True)
    r = verify_seq_windows(c, wdir, ref)
    assert not r.ok and any("dry" in x.lower() for x in r.reasons)


def test_gate_aborts_on_wrong_window(tmp_path):
    c = _cohort(); ref = _ref(tmp_path)
    wdir = _make_artifact(tmp_path, c, ref, window=51)
    r = verify_seq_windows(c, wdir, ref)
    assert not r.ok and any("window" in x for x in r.reasons)


def test_gate_aborts_on_wrong_convention(tmp_path):
    c = _cohort(); ref = _ref(tmp_path)
    wdir = _make_artifact(tmp_path, c, ref, convention="some_other_convention")
    r = verify_seq_windows(c, wdir, ref)
    assert not r.ok and any("convention" in x for x in r.reasons)


def test_gate_aborts_on_low_ok_fraction(tmp_path):
    c = _cohort(); ref = _ref(tmp_path)
    wdir = _make_artifact(tmp_path, c, ref, n_ok=1)  # 1/5 = 20% << 95% floor
    r = verify_seq_windows(c, wdir, ref)
    assert not r.ok and any("built ok" in x for x in r.reasons)


def test_raise_if_failed_raises(tmp_path):
    c = _cohort(); ref = _ref(tmp_path)
    (tmp_path / "empty").mkdir()
    with pytest.raises(SeqWindowsStaleError):
        verify_seq_windows(c, tmp_path / "empty", ref).raise_if_failed()
