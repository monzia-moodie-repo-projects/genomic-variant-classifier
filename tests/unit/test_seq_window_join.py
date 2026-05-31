"""Alignment + coverage tests for attach_delta_windows."""
import numpy as np
import pandas as pd

from genomic_variant_classifier.data import seq_window_join as J

W = 101


def _cohort(tmp_path, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "chrom": rng.choice(list("123456789") + ["X", "Y"], n),
        "pos": rng.randint(1, 250_000_000, n),
        "ref": rng.choice(list("ACGT"), n),
        "alt": rng.choice(list("ACGT"), n),
    }).drop_duplicates(subset=["chrom", "pos", "ref", "alt"]).reset_index(drop=True)
    key = (df.chrom.astype(str) + ":" + df.pos.astype(str) + ":" + df.ref + ":" + df.alt)
    df[J.REF_WIN_COL] = ("R_" + key).str.slice(0, W).str.ljust(W, "x")
    df[J.ALT_WIN_COL] = ("A_" + key).str.slice(0, W).str.ljust(W, "x")
    p = tmp_path / "cohort_seq.parquet"
    df.to_parquet(p, index=False)
    return df, p


def _expected(meta):
    key = (meta.chrom.astype(str) + ":" + meta.pos.astype(str) + ":" + meta.ref + ":" + meta.alt)
    return (("R_" + key).str.slice(0, W).str.ljust(W, "x").to_numpy(),
            ("A_" + key).str.slice(0, W).str.ljust(W, "x").to_numpy())


def test_keyjoin_alignment_survives_shuffle_and_filter(tmp_path):
    cohort, p = _cohort(tmp_path)
    meta = cohort.sample(frac=0.5, random_state=7).reset_index(drop=True)
    meta_key = meta[["chrom", "pos", "ref", "alt"]].copy()
    out, n_un = J.attach_delta_windows(meta_key, p)
    er, ea = _expected(meta_key)
    assert n_un == 0
    assert (out[J.REF_WIN_COL].to_numpy() == er).all()
    assert (out[J.ALT_WIN_COL].to_numpy() == ea).all()
    assert len(out) == len(meta_key)


def test_windows_already_on_meta_used_directly(tmp_path):
    cohort, _ = _cohort(tmp_path)
    out, n = J.attach_delta_windows(cohort)
    assert n == 0
    assert (out[J.REF_WIN_COL].to_numpy() == cohort[J.REF_WIN_COL].to_numpy()).all()


def test_unmapped_rows_fall_back_and_are_counted(tmp_path):
    cohort, p = _cohort(tmp_path)
    bad = pd.DataFrame({"chrom": ["99", "99"], "pos": [1, 2], "ref": ["A", "C"], "alt": ["G", "T"]})
    mixed = pd.concat([cohort[["chrom", "pos", "ref", "alt"]].head(10), bad], ignore_index=True)
    out, n_un = J.attach_delta_windows(mixed, p)
    assert n_un == 2
    assert (out[J.REF_WIN_COL].iloc[-2:] == "A" * W).all()


def test_duplicate_keys_in_cohort_are_harmless(tmp_path):
    cohort, _ = _cohort(tmp_path)
    dup = pd.concat([cohort, cohort.head(50)], ignore_index=True)
    p = tmp_path / "dup.parquet"
    dup.to_parquet(p, index=False)
    meta = cohort[["chrom", "pos", "ref", "alt"]]
    out, n_un = J.attach_delta_windows(meta, p)
    er, _ = _expected(meta)
    assert n_un == 0 and (out[J.REF_WIN_COL].to_numpy() == er).all()


def test_no_source_polyA(tmp_path):
    cohort, _ = _cohort(tmp_path)
    meta = cohort[["chrom", "pos", "ref", "alt"]]
    out, n_un = J.attach_delta_windows(meta, None)
    assert n_un == len(meta) and (out[J.REF_WIN_COL] == "A" * W).all()
