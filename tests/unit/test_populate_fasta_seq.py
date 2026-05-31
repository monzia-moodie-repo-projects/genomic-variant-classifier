"""Unit tests for populate_fasta_seq (Phase B window materialization)."""

from __future__ import annotations

import textwrap

import pyarrow.parquet as pq
import pandas as pd
import pytest

from genomic_variant_classifier.data import seq_windows as sw
from genomic_variant_classifier.data.populate_fasta_seq import (
    populate, GuardFailure, REF_COL, ALT_COL,
)

REF_SEQ = ("ACGT" * 75)  # 300 bp on contig "1"


@pytest.fixture()
def fasta(tmp_path):
    fa = tmp_path / "tiny.fa"
    fa.write_text(">1\n" + textwrap.fill(REF_SEQ, width=60) + "\n")
    return str(fa)


def _write_cohort(path, rows):
    pd.DataFrame(rows).to_parquet(path, index=False)


def _clean_rows():
    pos = 100
    ref_base = REF_SEQ[pos - 1]
    alt_base = "A" if ref_base != "A" else "C"
    del_pos = 150
    del_ref = REF_SEQ[del_pos - 1: del_pos - 1 + 3]
    return [
        {"rid": 0, "chrom": "1", "pos": pos, "ref": ref_base, "alt": alt_base, "label": 1},
        {"rid": 1, "chrom": "1", "pos": del_pos, "ref": del_ref, "alt": del_ref[0], "label": 0},
        {"rid": 2, "chrom": "Un", "pos": 50, "ref": "G", "alt": "A", "label": 1},
    ]


def test_passthrough_order_and_windows(tmp_path, fasta):
    cohort = tmp_path / "cohort.parquet"
    out = tmp_path / "cohort_seq.parquet"
    _write_cohort(cohort, _clean_rows())

    stats = populate(str(cohort), fasta, str(out), batch_size=2,
                     abort_mismatch=0.5, abort_degenerate=0.9)

    df = pd.read_parquet(out)
    # every original column preserved + 2 windows, same row count + order
    assert list(df["rid"]) == [0, 1, 2]
    assert {"chrom", "pos", "ref", "alt", "label", REF_COL, ALT_COL} <= set(df.columns)
    assert len(df) == 3
    # windows are 101 chars
    assert all(len(s) == sw.WINDOW for s in df[REF_COL])
    assert all(len(s) == sw.WINDOW for s in df[ALT_COL])

    # SNV row: center swapped, single-base delta
    r0 = df.iloc[0]
    assert r0[REF_COL][50] == REF_SEQ[99]
    assert r0[ALT_COL][50] == r0["alt"]
    assert sum(a != b for a, b in zip(r0[REF_COL], r0[ALT_COL])) == 1

    # 'Un' row: both poly-A (zero delta)
    r2 = df.iloc[2]
    assert r2[REF_COL] == sw.PAD_CHAR * sw.WINDOW
    assert r2[ALT_COL] == sw.PAD_CHAR * sw.WINDOW

    assert stats["n_unmapped"] == 1
    assert stats["n_resolvable"] == 2
    assert stats["n_mismatch"] == 0


def test_mismatch_guard_aborts_and_cleans_temp(tmp_path, fasta):
    cohort = tmp_path / "cohort.parquet"
    out = tmp_path / "cohort_seq.parquet"
    wrong = "A" if REF_SEQ[99] != "A" else "C"  # deliberately wrong ref allele
    _write_cohort(cohort, [
        {"rid": 0, "chrom": "1", "pos": 100, "ref": wrong, "alt": "T", "label": 1},
    ])

    with pytest.raises(GuardFailure):
        populate(str(cohort), fasta, str(out), batch_size=8,
                 abort_mismatch=0.0, abort_degenerate=0.9)

    assert not out.exists()                       # final never created
    assert not (tmp_path / "cohort_seq.parquet.tmp").exists()  # temp removed


def test_refuses_when_window_columns_present(tmp_path, fasta):
    cohort = tmp_path / "cohort.parquet"
    out = tmp_path / "cohort_seq.parquet"
    _write_cohort(cohort, [
        {"chrom": "1", "pos": 100, "ref": "C", "alt": "T", REF_COL: "x", ALT_COL: "y"},
    ])
    with pytest.raises(KeyError):
        populate(str(cohort), fasta, str(out))


def test_missing_required_column_raises(tmp_path, fasta):
    cohort = tmp_path / "cohort.parquet"
    out = tmp_path / "cohort_seq.parquet"
    pd.DataFrame([{"chrom": "1", "pos": 100, "ref": "C"}]).to_parquet(cohort, index=False)
    with pytest.raises(KeyError):
        populate(str(cohort), fasta, str(out))
