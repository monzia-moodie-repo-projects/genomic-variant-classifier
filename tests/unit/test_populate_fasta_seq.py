"""Unit tests for populate_fasta_seq (Phase B window materialization)."""

from __future__ import annotations

import textwrap

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
    snv_pos = 100
    snv_ref = REF_SEQ[snv_pos - 1]
    snv_alt = "A" if snv_ref != "A" else "C"
    # deletion stored at variant_summary convention: pos = first deleted base,
    # anchored ref begins one base earlier (sits at pos-1) -> must shift to anchor
    anchor = 150                       # 1-based VCF anchor
    del_ref = REF_SEQ[anchor - 1: anchor - 1 + 3]
    del_pos = anchor + 1               # cohort stores anchor+1
    return [
        {"rid": 0, "chrom": "1", "pos": snv_pos, "ref": snv_ref, "alt": snv_alt, "label": 1},
        {"rid": 1, "chrom": "1", "pos": del_pos, "ref": del_ref, "alt": del_ref[0], "label": 0},
        {"rid": 2, "chrom": "Un", "pos": 50, "ref": "G", "alt": "A", "label": 1},
    ]


def test_passthrough_order_windows_and_anchoring(tmp_path, fasta):
    cohort = tmp_path / "cohort.parquet"
    out = tmp_path / "cohort_seq.parquet"
    _write_cohort(cohort, _clean_rows())

    stats = populate(str(cohort), fasta, str(out), batch_size=2,
                     abort_unanchored=0.5, abort_degenerate=0.9)

    df = pd.read_parquet(out)
    assert list(df["rid"]) == [0, 1, 2]
    assert {"chrom", "pos", "ref", "alt", "label", REF_COL, ALT_COL} <= set(df.columns)
    assert all(len(s) == sw.WINDOW for s in df[REF_COL])
    assert all(len(s) == sw.WINDOW for s in df[ALT_COL])

    # SNV row: single-base delta at the centre
    r0 = df.iloc[0]
    assert r0[REF_COL][50] == REF_SEQ[99]
    assert r0[ALT_COL][50] == r0["alt"]
    assert sum(a != b for a, b in zip(r0[REF_COL], r0[ALT_COL])) == 1

    # deletion row: anchored to pos-1, so window centre is the anchor base
    r1 = df.iloc[1]
    assert r1[REF_COL][50] == REF_SEQ[(150 - 1)]      # genome at the VCF anchor
    assert r1[REF_COL] != r1[ALT_COL]                 # a real (non-zero) delta

    # 'Un' row: poly-A zero delta
    r2 = df.iloc[2]
    assert r2[REF_COL] == sw.PAD_CHAR * sw.WINDOW and r2[ALT_COL] == sw.PAD_CHAR * sw.WINDOW

    assert stats["n_unmapped"] == 1
    assert stats["n_resolvable"] == 2
    assert stats["n_shifted"] == 1          # the deletion was re-anchored
    assert stats["n_unanchored"] == 0


def test_unanchored_guard_aborts_and_cleans_temp(tmp_path, fasta):
    cohort = tmp_path / "cohort.parquet"
    out = tmp_path / "cohort_seq.parquet"
    wrong = "A" if REF_SEQ[99] != "A" else "C"   # single-base wrong -> cannot anchor
    _write_cohort(cohort, [
        {"rid": 0, "chrom": "1", "pos": 100, "ref": wrong, "alt": "T", "label": 1},
    ])

    with pytest.raises(GuardFailure):
        populate(str(cohort), fasta, str(out), batch_size=8,
                 abort_unanchored=0.0, abort_degenerate=0.9)

    assert not out.exists()
    assert not (tmp_path / "cohort_seq.parquet.tmp").exists()


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


def test_contig_grouped_matches_independent_per_row(tmp_path):
    import random
    rng = random.Random(99)
    seqs = {c: "".join(rng.choice("ACGT") for _ in range(2000)) for c in ("1", "2")}
    fa = tmp_path / "multi.fa"
    fa.write_text("".join(f">{c}\n" + textwrap.fill(s, 60) + "\n" for c, s in seqs.items()))

    rows = []
    for rid in range(400):
        c = rng.choice(["1", "2"])
        s = seqs[c]
        p = rng.randint(60, 1940)
        k = rng.random()
        if k < 0.7:                       # snv
            r = s[p - 1]; a = rng.choice([b for b in "ACGT" if b != r])
            rows.append({"rid": rid, "chrom": c, "pos": p, "ref": r, "alt": a})
        elif k < 0.9:                     # deletion stored at anchor+1
            L = rng.randint(2, 5); r = s[p - 1:p - 1 + L]; a = r[0]
            rows.append({"rid": rid, "chrom": c, "pos": p + 1, "ref": r, "alt": a})
        else:                             # insertion
            r = s[p - 1]; a = r + "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 3)))
            rows.append({"rid": rid, "chrom": c, "pos": p, "ref": r, "alt": a})
    rng.shuffle(rows)
    cohort = tmp_path / "c.parquet"
    out = tmp_path / "c_seq.parquet"
    pd.DataFrame(rows).to_parquet(cohort, index=False)

    populate(str(cohort), str(fa), str(out), batch_size=37,
             abort_unanchored=0.5, abort_degenerate=0.9)
    df = pd.read_parquet(out)

    # independent per-row reference using the same primitives, in original order
    for _, row in df.iterrows():
        apos = sw.find_anchor(seqs, row["chrom"], int(row["pos"]), row["ref"])
        if apos is None:
            erw, eaw = sw.PAD_CHAR * sw.WINDOW, sw.PAD_CHAR * sw.WINDOW
        else:
            erw, eaw = sw.build_delta_windows(seqs, row["chrom"], apos, row["ref"], row["alt"])
        assert row[REF_COL] == erw
        assert row[ALT_COL] == eaw
    assert list(df["rid"]) == [r["rid"] for r in rows]   # original order preserved
