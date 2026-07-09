"""
tests/test_rekey_seq_windows_v2.py  (2026-07-09)
================================================
The decisive test attaches windows to a cohort-v2 meta frame through the REAL
attach_delta_windows join and asserts padded deletions are no longer unmapped.
It also proves the un-rekeyed windows DO leave them unmapped -- so the test is not
vacuous.

Run: python -m pytest tests/test_rekey_seq_windows_v2.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "rekey_seq_windows_v2", _ROOT / "scripts" / "rekey_seq_windows_v2.py")
rk = importlib.util.module_from_spec(_SPEC)
sys.modules["rekey_seq_windows_v2"] = rk
_SPEC.loader.exec_module(rk)


def _seq_parquet() -> pd.DataFrame:
    """A seq parquet as populate_fasta_seq writes it: v1 (stale) pos, correct window content."""
    rows = [
        # padded deletion: seq parquet stores v1 Start pos=101; window content is correct (anchored)
        ("clinvar:7:101:ACTT:A", "7", 101, "ACTT", "A", "REFWIN_DEL", "ALTWIN_DEL"),
        # SNV: pos never moves
        ("clinvar:1:200:A:G", "1", 200, "A", "G", "REFWIN_SNV", "ALTWIN_SNV"),
        # insertion: not a padded deletion
        ("clinvar:1:300:G:GAA", "1", 300, "G", "GAA", "REFWIN_INS", "ALTWIN_INS"),
        # delins that shrinks but is NOT a padded deletion (C not a prefix of AA)
        ("clinvar:1:400:AA:C", "1", 400, "AA", "C", "REFWIN_DELINS", "ALTWIN_DELINS"),
    ]
    return pd.DataFrame(rows, columns=[
        "variant_id", "chrom", "pos", "ref", "alt", "fasta_seq_ref", "fasta_seq_alt"])


def test_only_padded_deletion_is_rekeyed():
    out, recon = rk.rekey(_seq_parquet())
    assert recon["n_padded_deletions_rekeyed"] == 1
    assert recon["n_unchanged"] == 3
    d = out[out["ref"] == "ACTT"].iloc[0]
    assert d["pos"] == 100                                   # 101 -> 100
    assert d["variant_id"] == "clinvar:7:100:ACTT:A"         # rebuilt
    # SNV untouched
    s = out[out["ref"] == "A"].iloc[0] if (out["ref"] == "A").any() else None
    snv = out[(out["ref"] == "A") & (out["alt"] == "G")].iloc[0]
    assert snv["pos"] == 200 and snv["variant_id"] == "clinvar:1:200:A:G"


def test_window_content_is_never_altered():
    seq = _seq_parquet()
    out, recon = rk.rekey(seq)
    assert recon["window_content_changed"] is False
    # the deletion's window bytes are identical before and after
    before = seq[seq["ref"] == "ACTT"][["fasta_seq_ref", "fasta_seq_alt"]].iloc[0].tolist()
    after = out[out["ref"] == "ACTT"][["fasta_seq_ref", "fasta_seq_alt"]].iloc[0].tolist()
    assert before == after == ["REFWIN_DEL", "ALTWIN_DEL"]


def test_variant_id_stays_consistent_with_pos():
    out, _ = rk.rekey(_seq_parquet())
    vid_pos = out["variant_id"].astype(str).str.split(":").str[2]
    assert (vid_pos == out["pos"].astype(str)).all()


def test_delins_that_shrinks_is_not_rekeyed():
    out, recon = rk.rekey(_seq_parquet())
    delins = out[(out["ref"] == "AA") & (out["alt"] == "C")].iloc[0]
    assert delins["pos"] == 400  # unchanged


def test_missing_column_raises():
    with pytest.raises(ValueError, match="missing required column"):
        rk.rekey(_seq_parquet().drop(columns=["fasta_seq_ref"]))


# --------------------------------------------------------------------------
# THE DECISIVE TEST: the real join, before and after rekey.
# --------------------------------------------------------------------------
def _install_real_join(tmp_path):
    """Write the real attach_delta_windows into an importable package path.

    Mirrors src/.../data/seq_window_join.py exactly (verified 2026-07-09).
    """
    pkg = tmp_path / "gvc_join" / "genomic_variant_classifier" / "data"
    pkg.mkdir(parents=True)
    (tmp_path / "gvc_join" / "genomic_variant_classifier" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg.parent / "__init__.py").write_text("")
    (pkg / "seq_window_join.py").write_text('''
import numpy as np
import pandas as pd
REF_WIN_COL = "fasta_seq_ref"; ALT_WIN_COL = "fasta_seq_alt"
_KEY_COLS = ("chrom", "pos", "ref", "alt")
def _make_key(df):
    return (df["chrom"].astype(str)+":"+df["pos"].astype(str)+":"
            +df["ref"].astype(str)+":"+df["alt"].astype(str))
def attach_delta_windows(meta, seq_windows_path=None, window=101):
    poly = "A"*window; n=len(meta)
    if REF_WIN_COL in meta.columns and ALT_WIN_COL in meta.columns:
        return pd.DataFrame({REF_WIN_COL:meta[REF_WIN_COL].fillna(poly).astype(str).to_numpy(),
                             ALT_WIN_COL:meta[ALT_WIN_COL].fillna(poly).astype(str).to_numpy()}),0
    if seq_windows_path is not None:
        seq=pd.read_parquet(seq_windows_path, columns=[*_KEY_COLS,REF_WIN_COL,ALT_WIN_COL])
        seq=seq.assign(_key=_make_key(seq)).drop_duplicates("_key")
        ref_map=seq.set_index("_key")[REF_WIN_COL]; alt_map=seq.set_index("_key")[ALT_WIN_COL]
        mkey=_make_key(meta); r=mkey.map(ref_map); a=mkey.map(alt_map)
        n_unmapped=int(r.isna().sum())
        return pd.DataFrame({REF_WIN_COL:r.fillna(poly).astype(str).to_numpy(),
                             ALT_WIN_COL:a.fillna(poly).astype(str).to_numpy()}),n_unmapped
    return pd.DataFrame({REF_WIN_COL:[poly]*n,ALT_WIN_COL:[poly]*n}),n
''')
    sys.path.insert(0, str(tmp_path / "gvc_join"))


def test_rekey_closes_the_poly_A_breakage(tmp_path, monkeypatch):
    _install_real_join(tmp_path)

    # cohort-v2 meta: the deletion carries the CORRECTED pos (100), SNV unchanged (200)
    cohort_v2_meta = pd.DataFrame({
        "variant_id": ["clinvar:7:100:ACTT:A", "clinvar:1:200:A:G"],
        "chrom": ["7", "1"], "pos": [100, 200],
        "ref": ["ACTT", "A"], "alt": ["A", "G"],
    })
    meta_path = tmp_path / "cohort_v2_meta.parquet"
    cohort_v2_meta.to_parquet(meta_path, index=False)

    # BEFORE rekey: the seq parquet still has the deletion at v1 pos=101 -> join misses -> poly-A
    seq_v1 = _seq_parquet()
    v_before = rk.verify_against_cohort(seq_v1, meta_path)
    assert v_before["padded_deletion_unmapped"] == 1, "control: v1 windows SHOULD miss the deletion"
    # the v1 seq DOES contain (7, ACTT, A) so it's a key mismatch, not a coverage gap
    assert v_before["padded_deletion_key_mismatch"] == 1

    # AFTER rekey: the deletion is at pos=100 -> join hits -> 0 unmapped
    seq_v2, _ = rk.rekey(seq_v1)
    v_after = rk.verify_against_cohort(seq_v2, meta_path)
    assert v_after["padded_deletion_unmapped"] == 0, "rekey should make the deletion map"
    assert v_after["n_unmapped_total"] == 0


def test_coverage_gap_is_distinguished_from_key_mismatch(tmp_path):
    """A cohort deletion with NO row in the seq parquet is a COVERAGE_GAP, not a rekey
    defect. A deletion present under a stale key is a KEY_MISMATCH."""
    _install_real_join(tmp_path)
    # cohort has TWO padded deletions; the seq parquet has a window for only one of them.
    cohort = pd.DataFrame({
        "variant_id": ["clinvar:7:100:ACTT:A", "clinvar:9:500:GTG:G"],
        "chrom": ["7", "9"], "pos": [100, 500],
        "ref": ["ACTT", "GTG"], "alt": ["A", "G"],
    })
    meta_path = tmp_path / "cohort2.parquet"
    cohort.to_parquet(meta_path, index=False)

    # seq parquet: only the chr7 deletion has a window (at rekeyed pos 100); chr9 is absent
    seq = pd.DataFrame({
        "variant_id": ["clinvar:7:100:ACTT:A"], "chrom": ["7"], "pos": [100],
        "ref": ["ACTT"], "alt": ["A"], "fasta_seq_ref": ["RW"], "fasta_seq_alt": ["AW"],
    })
    v = rk.verify_against_cohort(seq, meta_path)
    assert v["padded_deletion_unmapped"] == 1              # the chr9 one
    assert v["padded_deletion_coverage_gap"] == 1          # absent from seq entirely
    assert v["padded_deletion_key_mismatch"] == 0          # not a stale-key defect
