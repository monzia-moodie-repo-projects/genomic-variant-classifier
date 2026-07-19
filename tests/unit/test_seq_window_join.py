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
    att = J.attach_delta_windows(meta_key, p)
    er, ea = _expected(meta_key)
    assert att.n_unmapped == 0
    assert (att.windows[J.REF_WIN_COL].to_numpy() == er).all()
    assert (att.windows[J.ALT_WIN_COL].to_numpy() == ea).all()
    assert len(att.windows) == len(meta_key)


def test_windows_already_on_meta_used_directly(tmp_path):
    cohort, _ = _cohort(tmp_path)
    att = J.attach_delta_windows(cohort)
    assert att.n_unmapped == 0
    assert (att.windows[J.REF_WIN_COL].to_numpy() == cohort[J.REF_WIN_COL].to_numpy()).all()


def test_unmapped_rows_are_counted_and_marked_unusable(tmp_path):
    """Renamed + rewritten 2026-07-15 (roadmap 6.28).

    It previously asserted `out[REF_WIN_COL].iloc[-2:] == "A" * W` -- i.e. it asserted
    the FABRICATION, pinning the poly-A fill as the contract. The value of the filler is
    now explicitly not part of the contract; `usable` is. Asserting the filler is what
    let four separate consumers grow content-based poly-A detectors, none of which could
    see the builder's poly-N placeholders.
    """
    cohort, p = _cohort(tmp_path)
    bad = pd.DataFrame({"chrom": ["99", "99"], "pos": [1, 2], "ref": ["A", "C"], "alt": ["G", "T"]})
    mixed = pd.concat([cohort[["chrom", "pos", "ref", "alt"]].head(10), bad], ignore_index=True)
    att = J.attach_delta_windows(mixed, p)

    assert att.n_unmapped == 2
    assert att.n_rows == 12
    assert att.n_usable == 10
    # The two bogus keys, and ONLY those two, are unusable.
    assert not att.usable[-2:].any()
    assert att.usable[:10].all()


def test_builder_placeholder_rows_are_unusable_even_though_they_joined(tmp_path):
    """THE REGRESSION THIS WHOLE CHANGE EXISTS FOR -- and there was no test for it.

    A row present in the seq parquet with `ok=False` is a BUILDER PLACEHOLDER: the window
    could not be constructed (missing contig, out-of-range position, non-ACGT allele), so
    delta_window_builder wrote POLY = "N"*101 and flagged it. The key JOINS -- n_unmapped
    is 0 -- so every content-based poly-A check in the repository declared it real.

    On the live 2026-07-10 artifact this is 21,814 rows (n_poly), 0.494% of the cohort.
    They reached cnn_1d as an all-zero tensor with ref == alt (delta channels identically
    zero), and reached Nucleotide Transformer as ||alt_emb - ref_emb|| == exactly 0.0 --
    the same value that module documents as "window unavailable / model unavailable".
    """
    cohort, _ = _cohort(tmp_path, n=20)
    seq = cohort.copy()
    seq[J.OK_COL] = True
    # Two rows the builder could not construct: poly-N content, ok=False, key present.
    seq.loc[0:1, J.REF_WIN_COL] = "N" * W
    seq.loc[0:1, J.ALT_WIN_COL] = "N" * W
    seq.loc[0:1, J.OK_COL] = False
    p = tmp_path / "seq_with_ok.parquet"
    seq.to_parquet(p, index=False)

    meta = cohort[["chrom", "pos", "ref", "alt"]]
    att = J.attach_delta_windows(meta, p)

    assert att.n_unmapped == 0, "every key joins -- that was never the problem"
    assert att.n_placeholder == 2, "the builder's ok=False verdict must be read"
    assert not att.usable[0:2].any()
    assert att.usable[2:].all()
    assert att.provenance == "parquet+ok"


def test_a_real_polyA_window_is_usable(tmp_path):
    """The reason content-matching had to go, stated as a test.

    Poly-A tracts are real biology. A window whose reference sequence genuinely reads
    "A"*101 is REAL DATA and must be usable. Every one of the four content-based
    detectors would have discarded it -- silently, as "unmapped".
    """
    cohort, _ = _cohort(tmp_path, n=10)
    seq = cohort.copy()
    seq[J.OK_COL] = True
    seq.loc[0, J.REF_WIN_COL] = "A" * W   # a genuine poly-A tract, built from the reference
    seq.loc[0, J.ALT_WIN_COL] = "A" * 50 + "G" + "A" * 50
    p = tmp_path / "seq_real_polya.parquet"
    seq.to_parquet(p, index=False)

    att = J.attach_delta_windows(cohort[["chrom", "pos", "ref", "alt"]], p)

    assert att.usable[0], "a real poly-A window was discarded as a fallback"
    assert att.n_usable == att.n_rows


def test_missing_ok_column_is_reported_not_assumed_away(tmp_path, caplog):
    """An artifact predating the `ok` column cannot have its placeholders identified.

    That must be a loud warning, not a silent assumption of health -- CLAUDE.md 1.4: a
    search that returns nothing is not a negative result.
    """
    import logging

    cohort, p = _cohort(tmp_path, n=10)   # _cohort writes NO ok column
    meta = cohort[["chrom", "pos", "ref", "alt"]]
    with caplog.at_level(logging.WARNING):
        att = J.attach_delta_windows(meta, p)

    assert att.provenance == "parquet", "absence of `ok` must be recorded in provenance"
    assert any("no 'ok' column" in r.message for r in caplog.records), (
        "the loss of provenance was not warned about"
    )


def test_duplicate_keys_in_cohort_are_harmless(tmp_path):
    cohort, _ = _cohort(tmp_path)
    dup = pd.concat([cohort, cohort.head(50)], ignore_index=True)
    p = tmp_path / "dup.parquet"
    dup.to_parquet(p, index=False)
    meta = cohort[["chrom", "pos", "ref", "alt"]]
    att = J.attach_delta_windows(meta, p)
    er, _ = _expected(meta)
    assert att.n_unmapped == 0 and (att.windows[J.REF_WIN_COL].to_numpy() == er).all()


def test_no_source_means_no_usable_windows_at_all(tmp_path):
    """Renamed from `test_no_source_polyA` 2026-07-15: the old NAME encoded the old
    design -- that the filler's identity was the thing worth testing. It is not. With no
    window source there is no sequence signal, and the only fact that matters is that
    every row is unusable and something said so."""
    cohort, _ = _cohort(tmp_path)
    meta = cohort[["chrom", "pos", "ref", "alt"]]
    att = J.attach_delta_windows(meta, None)

    assert att.n_unmapped == len(meta)
    assert att.n_usable == 0
    assert not att.usable.any()
    assert att.provenance == "none"


def test_tier1_no_longer_reports_zero_unmapped_when_rows_are_null(tmp_path):
    """A bug the old code shipped, found only because changing the filler broke a test.

    Tier 1 (windows already on `meta`) ended with a hardcoded `return out, 0` -- it
    reported ZERO unmapped unconditionally, while `.fillna(poly)` quietly fabricated a
    window for every null row. `tests/unit/test_train_cnn_activation.py` asserted exactly
    that (`n_unmapped == 0` on a frame containing a None), pinning the defect as the
    contract. A null window is not a window.
    """
    meta = pd.DataFrame({
        J.REF_WIN_COL: ["R" * W, "R" * W, None],
        J.ALT_WIN_COL: ["A" * W, "A" * W, None],
    })
    att = J.attach_delta_windows(meta)

    assert att.n_rows == 3
    assert att.n_usable == 2
    assert att.n_unmapped == 1, "the null row must be counted, not fabricated away"
    assert not att.usable[2]
